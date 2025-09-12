import argparse
import torch, time
from collections import defaultdict
from pathlib import Path
import numpy as np

# from data.SimpleMetaTaskDataset import SimpleMetaTaskDataset
from data.variable_meta_dataset import VariableMetaDataset
from models.SimpleLinearPFN import SimpleLinearPFN
from config import get_config_shape, get_hyperprior_params, get_train_config, get_C_Q_ranges
from util.train import validate_model, gaussian_nll_loss, get_opt_lr_schedule


def _build_model(args):
    seq_len, pred_len = args.seq_len, args.pred_len
    d_model, d_ff = args.d_model, args.d_ff
    L_blk, n_heads = args.L_blk, args.n_heads
    dropout = args.dropout

    return SimpleLinearPFN(
            L=seq_len,
            H=pred_len,
            d=d_model,
            L_blk=L_blk,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout
        )

def _build_dataset(args, device):
    shape_config = get_config_shape()
    hyperprior_params = get_hyperprior_params()
    C_range, Q_range = get_C_Q_ranges()
    return VariableMetaDataset(
        shape_config=shape_config,
        hyperprior_params=hyperprior_params,
        L=args.seq_len,   # History length
        H=args.pred_len,   # Forecast horizon
        C_range=C_range,
        Q_range=Q_range,
        device=device,
    )


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = _build_dataset(args, device)
    model = _build_model(args).to(device)
    config = get_train_config()
    optimizer, scheduler = get_opt_lr_schedule(model, config)

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
    
    # Initial validation
    print("Running initial validation...")
    val_metrics = validate_model(model, dataset, device, config['n_val_tasks'])
    print(f"Initial validation - Loss: {val_metrics['loss']:.4f}, Corr: {val_metrics['correlation']:.3f}")
    print(f"Initial C range: {val_metrics['C_range']}, Q range: {val_metrics['Q_range']}")

    # training loop
    save_dir = f"ckpts/{args.seq_len}_{args.pred_len}"
    Path(save_dir).mkdir(exist_ok=True)
    model.train()

    for epoch in range(n_epochs):
        epoch_losses = []
        epoch_C_sizes = []
        epoch_Q_sizes = []
        
        for task_idx in range(config['tasks_per_epoch']):
            optimizer.zero_grad()
            
            # Create meta-task with variable C,Q
            task = dataset.create_meta_task()
            
            # Track sizes
            C_actual = task['ctx_x'].shape[0]
            Q_actual = task['qry_x'].shape[0]
            epoch_C_sizes.append(C_actual)
            epoch_Q_sizes.append(Q_actual)
            
            ctx_x = task['ctx_x'].unsqueeze(0).float().to(device)
            ctx_z = task['ctx_z'].unsqueeze(0).float().to(device)
            qry_x = task['qry_x'].unsqueeze(0).float().to(device)
            qry_z = task['qry_z'].unsqueeze(0).float().to(device)

            # Forward pass
            mu, log_sigma2 = model(ctx_x, ctx_z, qry_x)
            loss = gaussian_nll_loss(mu, log_sigma2, qry_z)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_losses.append(loss.item())
            global_task_count += 1
            
            # Logging
            if task_idx % config['log_every'] == 0:
                current_lr = optimizer.param_groups[0]['lr']
                recent_C = epoch_C_sizes[-min(len(epoch_C_sizes), config['log_every']):]
                recent_Q = epoch_Q_sizes[-min(len(epoch_Q_sizes), config['log_every']):]
                avg_C = np.mean(recent_C)
                avg_Q = np.mean(recent_Q)
                elapsed_hours = (time.time() - start_time) / 3600
                
                print(f"Epoch {epoch+1:3d}/{n_epochs}, Task {global_task_count:6,}/{config['total_tasks']:,}: "
                    f"Loss={loss.item():.4f}, LR={current_lr:.2e}, "
                    f"C̄={avg_C:.1f}, Q̄={avg_Q:.1f}, t={elapsed_hours:.2f}h")
        
        # Epoch statistics
        avg_train_loss = np.mean(epoch_losses)
        avg_C_epoch = np.mean(epoch_C_sizes)
        avg_Q_epoch = np.mean(epoch_Q_sizes)
        
        history['loss'].append(avg_train_loss)
        history['avg_C'].append(avg_C_epoch)
        history['avg_Q'].append(avg_Q_epoch)
        history['epoch'].append(epoch + 1)
        
        # Validation
        if (epoch + 1) % config['validate_every'] == 0:
            print(f"\nValidation after epoch {epoch + 1}...")
            val_metrics = validate_model(model, dataset, device, config['n_val_tasks'])
            
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
            
            # Early stopping
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
                
            if patience_counter >= config['early_stopping_patience']:
                print(f"\nEarly stopping after {patience_counter} epochs without improvement")
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
    parser.add_argument('--seq_len', type=int, required=True, default=36, help='history length')
    parser.add_argument('--pred_len', type=int, required=True, default=36, help='prediction horizon')
    parser.add_argument('--d_model', type=int, default=512, help='model dimension')
    parser.add_argument('--d_ff', type=int, default=2048, help='feedforward dimension')
    parser.add_argument('--L_blk', type=int, default=12, help='number of transformer blocks')
    parser.add_argument('--n_heads', type=int, default=16, help='number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')

    args = parser.parse_args()
    main(args)