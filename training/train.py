import argparse, time
import torch, torch.nn as nn
from collections import defaultdict
from pathlib import Path
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from data.meta_dataset import VariableMetaDataset
from models.LinearPFN import LinearPFN
from config import get_train_config, get_C_Q_ranges
from util.train import validate_model, gaussian_nll_loss, get_opt_lr_schedule
import json

def dump_run_config(path, args, model_id, mparams_dict, data_info, train_config):
    cfg = {
        "args": vars(args),
        "model_id": model_id,
        "model_params": mparams_dict,
        "data": data_info,
        "train_config": train_config,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def _build_model(args):
    seq_len, pred_len = args.seq_len, args.pred_len
    d_model, d_ff = args.d_model, args.d_ff
    L_blk, n_heads = args.L_blk, args.n_heads
    dropout = args.dropout
    mparams_dict = {"L": seq_len, "H": pred_len, "dropout": dropout,
                    "d_model": d_model, "d_ff": d_ff, "L_blk": L_blk, "n_heads": n_heads}
    model_id = f"{args.model}/v{args.data_version}/L{seq_len}_H{pred_len}_d{d_model}_Lblk{L_blk}_n{n_heads}_dff{d_ff}_do{dropout}" 
    if args.model == 'LinearPFN':
        return (
            LinearPFN(L=seq_len,H=pred_len,
                            d=d_model, L_blk=L_blk, n_heads=n_heads,d_ff=d_ff,
                            dropout=dropout), 
            model_id, mparams_dict
            )

def _build_dataset(args, device):
    # shape_config = get_config_shape()
    # hyperprior_params = get_hyperprior_params()
    C_range, Q_range = get_C_Q_ranges(args)
    return (VariableMetaDataset(
        shards_dir=f'./series_bank/v{args.data_version}',
        L=args.seq_len,   # History length
        H=args.pred_len,   # Forecast horizon
        C_range=C_range,
        Q_range=Q_range,
        device=device,
    ), C_range, Q_range)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset, C_range, Q_range = _build_dataset(args, device)
    dataset_meta = f"C{'-'.join(list(map(str, C_range)))}_Q{'-'.join(list(map(str, Q_range)))}"
    model, model_id, mparams_dict = _build_model(args)
    model = model.to(device)
    config = get_train_config()
    optimizer, scheduler = get_opt_lr_schedule(model, config)


    save_dir = f"ckpts/{model_id}_{dataset_meta}"
    Path(save_dir).mkdir(exist_ok=True, parents=True)

    data_info = {
        "shards_dir": f'./series_bank/v{args.data_version}',
        "L": args.seq_len,
        "H": args.pred_len,
        "C_range": list(C_range),
        "Q_range": list(Q_range),
    }

    dump_run_config(
        path=f"{save_dir}/run_config.json",
        args=args,
        model_id=model_id,
        mparams_dict=mparams_dict,
        data_info=data_info,
        train_config=config,
    )
    
    log_dir = f"tb_runs/{model_id}_{dataset_meta}"
    writer = SummaryWriter(log_dir=log_dir)
    

    n_epochs = config['total_tasks'] // config['tasks_per_epoch']

    print(f"\nTraining Plan:")
    print(f"  Total meta-tasks: {config['total_tasks']:,}")
    print(f"  Tasks per epoch: {config['tasks_per_epoch']:,}")
    print(f"  Total epochs: {n_epochs}")
    print(f"  Variable C: {dataset.C_range} (log-uniform)")
    print(f"  Variable Q: {dataset.Q_range}")

    history = defaultdict(list)
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    global_task_count = 0

    criterion = nn.MSELoss()
    
    # Initial validation
    print("Running initial validation...")
    val_metrics = validate_model(model, dataset, device, config['n_val_tasks'], use_time=True)
    print(f"Initial validation - Loss: {val_metrics['loss']:.4f}, Corr: {val_metrics['correlation']:.3f}")
    print(f"Initial C range: {val_metrics['C_range']}, Q range: {val_metrics['Q_range']}")

    writer.add_scalar('val/loss', val_metrics['loss'], 0)
    writer.add_scalar('val/loss_std', val_metrics['loss_std'], 0)
    writer.add_scalar('val/corr', val_metrics['correlation'], 0)
    writer.add_scalar('val/corr_std', val_metrics['correlation_std'], 0)

    model.eval()
    with torch.no_grad():
        eg = dataset.create_meta_task()
        eg_ctx_x = eg['ctx_x'].unsqueeze(0).float().to(device)
        eg_ctx_z = eg['ctx_z'].unsqueeze(0).float().to(device)
        eg_qry_x = eg['qry_x'].unsqueeze(0).float().to(device)
        eg_t_ctx = torch.tensor(eg['endpoints']['ctx'], dtype=torch.long).unsqueeze(0).to(device)
        eg_t_qry = torch.tensor(eg['endpoints']['qry'], dtype=torch.long).unsqueeze(0).to(device)

        try:
            writer.add_graph(model, (eg_ctx_x, eg_ctx_z, eg_qry_x, eg_t_ctx, eg_t_qry))
        except Exception:
            traced = torch.jit.trace(
                model,
                (eg_ctx_x, eg_ctx_z, eg_qry_x, eg_t_ctx, eg_t_qry),
                strict=False
            )
            writer.add_graph(traced, (eg_ctx_x, eg_ctx_z, eg_qry_x, eg_t_ctx, eg_t_qry))

    # training loop
    model.train()
    global_step = 0

    for epoch in tqdm(range(n_epochs)):
        epoch_start = time.time()
        epoch_losses = []
        epoch_C_sizes = []
        epoch_Q_sizes = []
        
        for task_idx in range(config['tasks_per_epoch']):
            optimizer.zero_grad()
            task = dataset.create_meta_task()
            
            C_actual = task['ctx_x'].shape[0]
            Q_actual = task['qry_x'].shape[0]
            epoch_C_sizes.append(C_actual)
            epoch_Q_sizes.append(Q_actual)
            
            ctx_x = task['ctx_x'].unsqueeze(0).float().to(device)
            ctx_z = task['ctx_z'].unsqueeze(0).float().to(device)
            qry_x = task['qry_x'].unsqueeze(0).float().to(device)
            qry_z = task['qry_z'].unsqueeze(0).float().to(device)

            t_ctx = torch.tensor(task['endpoints']['ctx'], dtype=torch.long).unsqueeze(0).to(device)  # [1,C]
            t_qry = torch.tensor(task['endpoints']['qry'], dtype=torch.long).unsqueeze(0).to(device)  # [1,Q]

            pred = model(ctx_x, ctx_z, qry_x, t_ctx, t_qry)
            loss = criterion(pred, qry_z)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_losses.append(loss.item())
            global_task_count += 1
            global_step += 1

            if task_idx % config['log_every'] == 0:
                current_lr = optimizer.param_groups[0]['lr']
                recent_C = epoch_C_sizes[-min(len(epoch_C_sizes), config['log_every']):]
                recent_Q = epoch_Q_sizes[-min(len(epoch_Q_sizes), config['log_every']):]
                avg_C = float(np.mean(recent_C))
                avg_Q = float(np.mean(recent_Q))
                elapsed_hours = (time.time() - start_time) / 3600
                tasks_per_hour = global_task_count / max(elapsed_hours, 1e-9)

                print(f"Epoch {epoch+1:3d}/{n_epochs}, Task {global_task_count:6,}/{config['total_tasks']:,}: "
                    f"Loss={loss.item():.4f}, LR={current_lr:.2e}, "
                    f"C̄={avg_C:.1f}, Q̄={avg_Q:.1f}, t={elapsed_hours:.2f}h")

                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('opt/lr', current_lr, global_step)
                writer.add_scalar('data/C_avg', avg_C, global_step)
                writer.add_scalar('data/Q_avg', avg_Q, global_step)
                writer.add_scalar('speed/tasks_per_hour', tasks_per_hour, global_step)

        avg_train_loss = float(np.mean(epoch_losses))
        avg_C_epoch = float(np.mean(epoch_C_sizes))
        avg_Q_epoch = float(np.mean(epoch_Q_sizes))
        
        history['loss'].append(avg_train_loss)
        history['avg_C'].append(avg_C_epoch)
        history['avg_Q'].append(avg_Q_epoch)
        history['epoch'].append(epoch + 1)
        

        epoch_time = time.time() - epoch_start

        writer.add_histogram('hist/C', np.array(epoch_C_sizes, dtype=np.int64), epoch + 1)
        writer.add_histogram('hist/Q', np.array(epoch_Q_sizes, dtype=np.int64), epoch + 1)
        writer.add_scalar('train/epoch_loss', avg_train_loss, epoch + 1)
        writer.add_scalar('data/C_epoch_avg', avg_C_epoch, epoch + 1)
        writer.add_scalar('data/Q_epoch_avg', avg_Q_epoch, epoch + 1)
        writer.add_scalar('time/epoch_seconds', epoch_time, epoch + 1)
        
        # Validation
        if (epoch + 1) % config['validate_every'] == 0:
            print(f"\nValidation after epoch {epoch + 1}...")
            val_metrics = validate_model(model, dataset, device, config['n_val_tasks'], use_time=True, C_range=C_range)
            
            history['val_loss'].append(val_metrics['loss'])
            history['val_correlation'].append(val_metrics['correlation'])
            history['val_epoch'].append(epoch + 1)
            
            tasks_per_hour = global_task_count / ((time.time() - start_time) / 3600)
            
            print(f"Epoch {epoch+1:3d} Results:")
            print(f"  Train Loss: {avg_train_loss:.4f} (C̄={avg_C_epoch:.1f}, Q̄={avg_Q_epoch:.1f})")
            print(f"  Val Loss: {val_metrics['loss']:.4f} ± {val_metrics['loss_std']:.4f}")
            print(f"  Val Corr: {val_metrics['correlation']:.3f} ± {val_metrics['correlation_std']:.3f}")
            print(f"  Val C range: {val_metrics['C_range']}, Q range: {val_metrics['Q_range']}")
            print(f"  Speed: {tasks_per_hour:,.0f} tasks/hour, Tasks: {global_task_count:,}")

            writer.add_scalar('val/loss', val_metrics['loss'], epoch + 1)
            writer.add_scalar('val/loss_std', val_metrics['loss_std'], epoch + 1)
            writer.add_scalar('val/corr', val_metrics['correlation'], epoch + 1)
            writer.add_scalar('val/corr_std', val_metrics['correlation_std'], epoch + 1)

            for tag, val in val_metrics.get('corr_buckets', {}).items():
                writer.add_scalar(tag, val, epoch + 1)

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_correlation': val_metrics['correlation'],
                    'epoch': epoch + 1,
                    'global_tasks': global_task_count,
                    'config': config,
                    'dataset_config': {
                        'L': dataset.L, 'H': dataset.H,
                        'C_range': dataset.C_range, 'Q_range': dataset.Q_range
                    }
                }, f'{save_dir}/best_model.pt')
                print(f"Best model saved! (tasks: {global_task_count:,})")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= config['early_stopping_patience']:
                print(f"\nEarly stopping after {patience_counter} epochs without improvement")
                writer.flush()
                writer.close()
                break
        
        # Regular checkpoints
        if (epoch + 1) % config['save_every'] == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'global_tasks': global_task_count,
                'history': dict(history)
            }, f'{save_dir}/model_epoch_{epoch+1}.pt')
            writer.flush()

    writer.flush()
    writer.close()

    total_time = time.time() - start_time
    tasks_per_hour = global_task_count / (total_time / 3600)

    print(f"\n TRAINING COMPLETE! ")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Total tasks trained: {global_task_count:,}")
    print(f"Average speed: {tasks_per_hour:,.0f} tasks/hour")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved in {save_dir}")
        
if __name__ == '__main__':
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('--model', type=str, default='LinearPFN', help='model name')
    parser.add_argument('--seq_len', type=int, required=True, default=36, help='history length')
    parser.add_argument('--pred_len', type=int, required=True, default=36, help='prediction horizon')
    parser.add_argument('--d_model', type=int, default=512, help='model dimension')
    parser.add_argument('--d_ff', type=int, default=2048, help='feedforward dimension')
    parser.add_argument('--L_blk', type=int, default=12, help='number of transformer blocks')
    parser.add_argument('--n_heads', type=int, default=16, help='number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--data_version', type=str, default='0', help='synthetic data location')
    parser.add_argument('--c_min', type=int, default=32, help='synthetic data location')
    parser.add_argument('--c_max', type=int, default=1536, help='synthetic data location')

    args = parser.parse_args()
    main(args)