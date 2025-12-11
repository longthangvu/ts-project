import os
import warnings
import numpy as np
import torch
import csv
import heapq, random

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import visual

warnings.filterwarnings('ignore')

class _ReservoirItem:
    __slots__ = ("key", "x", "z")
    def __init__(self, key, x, z):
        self.key = key; self.x = x; self.z = z
    def __lt__(self, other):
        return self.key < other.key  # min-heap by key

@torch.no_grad()
def build_context_sampled(train_loader,
                        pred_len: int,
                        max_ctx: int = 4096,
                        alpha: float = 0.7,     # importance strength [0..1]
                        stride: int = 1,        # thin stream: keep every k-th candidate before reservoir
                        device: str = "cpu"):
    """
    Weighted reservoir over the stream of patches produced by train_loader.
    Importance = volatility of history window. Higher volatility -> higher retention prob.
    stride>1 reduces adjacent-window redundancy before reservoir.
    Returns: x_ctx [N_ctx, L], z_ctx [N_ctx, H] (CPU tensors by default).
    """
    heap = []  # min-heap of _ReservoirItem, size <= max_ctx
    gidx = 0   # global candidate index after stride-thinning

    for bx, by, _, _ in train_loader:
        # bx: [B, L, C], by: [B, label_len+pred_len, C]
        z = by[:, -pred_len:, :]  # [B, H, C]

        # univariate selection/squeeze
        if bx.shape[-1] > 1:
            bx = bx[..., 0]
            z  = z[..., 0]
        else:
            bx = bx.squeeze(-1)  # [B, L]
            z  = z.squeeze(-1)   # [B, H]

        bx = bx.detach().to("cpu")
        z  = z.detach().to("cpu")

        B = bx.shape[0]
        for i in range(B):
            # stride-thinning
            if (gidx % stride) != 0:
                gidx += 1
                continue

            x_i = bx[i]  # [L]
            z_i = z[i]   # [H]

            # importance weight from volatility; bounded and stable
            # w in [1, 1+alpha*~]
            std = torch.std(x_i)
            mean_abs = torch.mean(torch.abs(x_i)) + 1e-6
            vol = (std / mean_abs).clamp_(min=0.0, max=10.0).item()  # scale-invariant, capped
            w = 1.0 + alpha * (vol / (1.0 + vol))                    # maps to (1, 1+alpha)

            # Efraimidis–Spirakis weighted reservoir: key = U^{1/w}, keep top-K keys
            u = random.random()
            key = u ** (1.0 / w)

            if len(heap) < max_ctx:
                heapq.heappush(heap, _ReservoirItem(key, x_i, z_i))
            else:
                if key > heap[0].key:
                    heapq.heapreplace(heap, _ReservoirItem(key, x_i, z_i))

            gidx += 1

    if not heap:
        return torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.float32)

    # convert heap to tensors (unordered; order irrelevant for context)
    xs = torch.stack([it.x for it in heap], dim=0)  # [N_ctx, L]
    zs = torch.stack([it.z for it in heap], dim=0)  # [N_ctx, H]

    if device != "cpu":
        xs = xs.to(device, non_blocking=True)
        zs = zs.to(device, non_blocking=True)

    return xs, zs

def build_model(model_name, L, H, d_model, L_blk, n_heads, d_ff, c_min=32, c_max=1536, version=0):
    if model_name == 'SimplePFN':
        from training.models.SimpleLinearPFN import SimpleLinearPFN
        mdl = SimpleLinearPFN
    elif model_name == 'FITSPFN':
        from training.models.FITSPFN import FitsPFN
        mdl = FitsPFN
    elif model_name == 'LinearPFN_old':
        from training.models.LinearPFN import LinearPFN_old
        mdl = LinearPFN_old
    elif model_name == 'LinearPFN':
        from training.models.LinearPFN import LinearPFN
        mdl = LinearPFN
    model = mdl(L=L, H=H, d=d_model, L_blk=L_blk, n_heads=n_heads, d_ff=d_ff)

    print('Loading model from checkpoint')
    model_path = f'training/ckpts/{model_name}/v{version}/L{L}_H{H}_d{d_model}_Lblk{L_blk}_n{n_heads}_dff{d_ff}_do0.1_C{c_min}-{c_max}_Q1-32'
    
    ckpt = torch.load(os.path.join(model_path, 'best_model.pt'), weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    return model

class Exp_MetaLearningPFN(Exp_Basic):
    def __init__(self, args):
        super(Exp_MetaLearningPFN, self).__init__(args)

    def _build_model(self):
        version = self.args.data_version
        L, H = self.args.seq_len, self.args.pred_len
        d_model, L_blk, n_heads, d_ff = self.args.d_model, self.args.e_layers, self.args.n_heads, self.args.d_ff
        c_min, c_max = self.args.c_min, self.args.c_max
        return build_model(self.args.model, L, H, d_model, L_blk, n_heads, d_ff, c_min, c_max, version)


    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    # @torch.no_grad()
    # def _build_context(self, train_loader):
    #     return build_context_sampled(
    #         train_loader=train_loader,
    #         pred_len=self.args.pred_len,
    #         max_ctx=getattr(self.args, "max_ctx", 8192),
    #         alpha=getattr(self.args, "ctx_importance_alpha", 0.7),
    #         stride=getattr(self.args, "ctx_stride", 2),
    #         device=getattr(self.args, "ctx_device", "cpu"),
    #     )

    @torch.no_grad()
    def _build_context(self, train_loader):
        xs, zs = [], []
        for bx, by, _, _ in train_loader:
            # bx: [B, L, C], by: [B, label_len+pred_len, C]
            # keep only the forecast horizon part for z
            z = by[:, -self.args.pred_len:, :]
            xs.append(bx)
            zs.append(z)
        x = torch.cat(xs, dim=0)  # [N_ctx, L, C]
        z = torch.cat(zs, dim=0)  # [N_ctx, H, C]
        # univariate: squeeze last channel; if multivariate exists, take first channel
        if x.shape[-1] > 1:
            x = x[..., 0]
            z = z[..., 0]
        else:
            x = x.squeeze(-1)
            z = z.squeeze(-1)
        return x, z  # [N_ctx, L], [N_ctx, H]

    def test(self, setting, test=0):
        """
        PFN eval with standard loaders:
          - context = all train windows
          - queries = each test batch (batched)
        """
        print("Loading data...")
        _, train_loader = self._get_data(flag='train')
        test_data, test_loader   = self._get_data(flag='test')

        print("Collecting context from train...")
        ctx_x, ctx_z = self._build_context(train_loader)  # [C, L], [C, H]
        print(f"ctx: x{tuple(ctx_x.shape)}, z{tuple(ctx_z.shape)}")

        ctx_x = ctx_x.to(self.device).float()
        ctx_z = ctx_z.to(self.device).float()

        C = ctx_x.size(0)

        t_ctx = torch.arange(C, dtype=torch.long, device=self.device)          # [C]
        t_ctx_b = t_ctx.unsqueeze(0)  

        preds, trues = [], []
        self.model.eval()
        out_dir = './results/' + setting + f'/C{self.args.c_min}-{self.args.c_max}_Q1-32/{self.args.data}/{self.args.train_budget}/'
        os.makedirs(out_dir, exist_ok=True)

        for i, (bx, by, _, _) in enumerate(test_loader):
            # bx: [B, L, C], by: [B, label_len+pred_len, C]
            bx = bx.to(self.device).float()
            by = by.to(self.device).float()

            # queries are the whole batch
            qx = bx[..., 0] if bx.shape[-1] > 1 else bx.squeeze(-1)                       # [B, L]
            qy = by[:, -self.args.pred_len:, 0] if by.shape[-1] > 1 else \
                 by[:, -self.args.pred_len:, :].squeeze(-1)  # [B, H]
            
            Bq = qx.size(0)

            # pack as one meta-task: [1, C, L], [1, C, H], [1, Q, L]
            ctx_x_b = ctx_x.unsqueeze(0)
            ctx_z_b = ctx_z.unsqueeze(0)
            qx_b    = qx.unsqueeze(0)
            
            # Query indices immediately follow contexts
            t_qry   = torch.arange(C, C + Bq, dtype=torch.long, device=self.device)  # [B]
            t_qry_b = t_qry.unsqueeze(0)   

            # mu, log_sigma2 = self.model(ctx_x_b, ctx_z_b, qx_b)   # [1, B, H]
            if self.args.use_time:
                mu = self.model(ctx_x_b, ctx_z_b, qx_b, t_ctx_b, t_qry_b)
            else:
                mu = self.model(ctx_x_b, ctx_z_b, qx_b)   # [1, B, H]
            mu = mu.squeeze(0).detach().cpu().numpy()             # [B, H]
            y  = qy.detach().cpu().numpy()                        # [B, H]

            if test_data.scale and self.args.inverse:
                B, H = mu.shape
                mu = test_data.inverse_transform(mu.reshape(B * H, 1)).reshape(B, H)
                y  = test_data.inverse_transform(y.reshape(B * H, 1)).reshape(B, H)

            preds.append(mu)
            trues.append(y)

            # inverse the history input 
            inp = bx.detach().cpu().numpy()
            if inp.shape[-1] > 1:
                inp = inp[..., 0:1]  # keep first channel
            if test_data.scale and self.args.inverse:
                B, L, C = inp.shape
                inp = test_data.inverse_transform(inp.reshape(B*L, C)).reshape(B, L, C)
            hist = inp[0, :, -1]  # [L]
            gt = np.concatenate((hist, y[0]), axis=0)  # history + true future
            pd = np.concatenate((hist, mu[0]), axis=0)  # history + predicted future
            visual(gt, pd, os.path.join(out_dir, f'viz_{i}.pdf'))

        preds = np.concatenate(preds, axis=0)  # [N_test, H]
        trues = np.concatenate(trues, axis=0)  # [N_test, H]
        print('preds:', preds.shape, 'trues:', trues.shape)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f'mse:{mse}, mae:{mae}, rmse:{rmse}, mape:{mape}, mspe:{mspe}')
        
        csv_path = 'results.csv'
        header = ['dataset', 'train_budget', 'seq_len', 'pred_len',
                  'd_model', 'e_layers', 'n_heads', 'd_ff',
                  'mae', 'mse', 'rmse', 'mape', 'mspe']
        row = [self.args.data, self.args.train_budget, self.args.seq_len, self.args.pred_len,
               self.args.d_model, self.args.e_layers, self.args.n_heads, self.args.d_ff,
               mae, mse, rmse, mape, mspe]

        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['model'] + header)
            writer.writerow([self.args.model] + row)

        csv_p = './results/' + setting + f'/C{self.args.c_min}-{self.args.c_max}_Q1-32' + '/results.csv'
        write_header = not os.path.exists(csv_p)
        with open(csv_p, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)

        np.save(os.path.join(out_dir, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(out_dir, 'pred.npy'), preds)
        np.save(os.path.join(out_dir, 'true.npy'), trues)
        return
