import torch, math
import numpy as np

import numpy as np
import torch
import torch.nn as nn


def validate_model(model, dataset, device, n_val_tasks=100, use_time=False):
    criterion = nn.MSELoss(reduction='mean')
    model.eval()
    losses, corrs, C_sizes, Q_sizes = [], [], [], []
    with torch.no_grad():
        for _ in range(n_val_tasks):
            t = dataset.create_meta_task()
            C_sizes.append(int(t['ctx_x'].shape[0]))
            Q_sizes.append(int(t['qry_x'].shape[0]))

            ctx_x = t['ctx_x'].unsqueeze(0).float().to(device)
            ctx_z = t['ctx_z'].unsqueeze(0).float().to(device)
            qry_x = t['qry_x'].unsqueeze(0).float().to(device)
            qry_z = t['qry_z'].unsqueeze(0).float().to(device)
            if use_time:
                t_ctx = torch.tensor(t['endpoints']['ctx'], dtype=torch.long).unsqueeze(0).to(device)  # [1,C]
                t_qry = torch.tensor(t['endpoints']['qry'], dtype=torch.long).unsqueeze(0).to(device)  # [1,Q]

                pred = model(ctx_x, ctx_z, qry_x, t_ctx, t_qry)   # must match qry_z shape
            else:
                pred = model(ctx_x, ctx_z, qry_x)
            loss = criterion(pred, qry_z)
            losses.append(loss.item())

            p = pred.flatten().float().cpu()
            q = qry_z.flatten().float().cpu()
            if p.numel() > 1 and torch.var(p) > 0 and torch.var(q) > 0:
                corrs.append(torch.corrcoef(torch.stack([p, q]))[0, 1].item())

    model.train()
    return {
        'loss': float(np.mean(losses)) if losses else 0.0,
        'loss_std': float(np.std(losses)) if losses else 0.0,
        'correlation': float(np.mean(corrs)) if corrs else 0.0,
        'correlation_std': float(np.std(corrs)) if corrs else 0.0,
        'avg_C': float(np.mean(C_sizes)) if C_sizes else 0.0,
        'avg_Q': float(np.mean(Q_sizes)) if Q_sizes else 0.0,
        'C_range': f"{min(C_sizes)}-{max(C_sizes)}" if C_sizes else "0-0",
        'Q_range': f"{min(Q_sizes)}-{max(Q_sizes)}" if Q_sizes else "0-0",
    }


def validate_model_gnll(model, dataset, device, n_val_tasks=100):
    """Validation with variable C,Q sizes"""
    model.eval()
    val_metrics = {'loss': [], 'correlation': [], 'C_sizes': [], 'Q_sizes': []}
    
    with torch.no_grad():
        for _ in range(n_val_tasks):
            task = dataset.create_meta_task()
            
            # Get actual sizes
            C_actual = task['ctx_x'].shape[0] 
            Q_actual = task['qry_x'].shape[0]
            
            ctx_x = task['ctx_x'].unsqueeze(0).float().to(device)
            ctx_z = task['ctx_z'].unsqueeze(0).float().to(device)
            qry_x = task['qry_x'].unsqueeze(0).float().to(device)
            qry_z = task['qry_z'].unsqueeze(0).float().to(device)
            
            # Forward pass
            mu, log_sigma2 = model(ctx_x, ctx_z, qry_x)
            loss = gaussian_nll_loss(mu, log_sigma2, qry_z)
            val_metrics['loss'].append(loss.item())
            
            # Correlation
            mu_flat = mu.flatten().cpu()
            qry_z_flat = qry_z.flatten().cpu()
            corr = torch.corrcoef(torch.stack([mu_flat, qry_z_flat]))[0, 1]
            if not torch.isnan(corr):
                val_metrics['correlation'].append(corr.item())
            
            val_metrics['C_sizes'].append(C_actual)
            val_metrics['Q_sizes'].append(Q_actual)
    
    model.train()
    return {
        'loss': np.mean(val_metrics['loss']),
        'loss_std': np.std(val_metrics['loss']),
        'correlation': np.mean(val_metrics['correlation']) if val_metrics['correlation'] else 0.0,
        'correlation_std': np.std(val_metrics['correlation']) if val_metrics['correlation'] else 0.0,
        'avg_C': np.mean(val_metrics['C_sizes']),
        'avg_Q': np.mean(val_metrics['Q_sizes']),
        'C_range': f"{min(val_metrics['C_sizes'])}-{max(val_metrics['C_sizes'])}",
        'Q_range': f"{min(val_metrics['Q_sizes'])}-{max(val_metrics['Q_sizes'])}"
    }

def get_opt_lr_schedule(model, config):
    warmup_tasks, total_tasks = config['warmup_tasks'], config['total_tasks']

    optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.05,
            betas=(0.9, 0.95),
            eps=1e-8
        ) 

    def lr_lambda(step):
        if step < warmup_tasks:
            return step / warmup_tasks
        else:
            progress = (step - warmup_tasks) / (total_tasks - warmup_tasks)
            return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress))  # Decay to 10%
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def gaussian_nll_loss(mu, log_sigma2, targets):
    """
    Compute the Gaussian negative log-likelihood loss as specified:
    
    ℓ_{j,h} = 1/2 * [ln(2π) + ln σ²_{j,h} + (z̃*_{j,h} - μ_{j,h})² / σ²_{j,h}]
    
    Task loss: L = 1/(QH) * Σ_j Σ_h ℓ_{j,h}
    
    Args:
        mu: Predicted means [batch_size, Q, H]
        log_sigma2: Predicted log variances [batch_size, Q, H]
        targets: Ground truth targets [batch_size, Q, H]
    
    Returns:
        loss: Average loss over all query patches and horizons
    """
    # Compute σ² from log σ²
    sigma2 = torch.exp(log_sigma2)
    
    # Compute loss components
    log_2pi = math.log(2 * math.pi)
    
    # ℓ_{j,h} = 1/2 * [ln(2π) + ln σ²_{j,h} + (z̃*_{j,h} - μ_{j,h})² / σ²_{j,h}]
    loss_per_point = 0.5 * (
        log_2pi +
        log_sigma2 +
        ((targets - mu) ** 2) / sigma2
    )
    
    # Average over Q queries and H horizons: 1/(QH) * Σ_j Σ_h ℓ_{j,h}
    return loss_per_point.mean()