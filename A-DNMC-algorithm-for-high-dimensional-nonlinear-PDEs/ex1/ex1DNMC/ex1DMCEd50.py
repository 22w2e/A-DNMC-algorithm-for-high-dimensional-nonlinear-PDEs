import os

# ★★★ FIX: 解决 OpenMP Error #15 ★★★
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
import matplotlib.pyplot as plt
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. 网络定义
# ==========================================
class FBSDE_Net_Final(nn.Module):
    def __init__(self, state_dim, noise_dim, hidden_layers=2, hidden_dim_offset=110, output_dim=1):
        super().__init__()
        self.input_dim = 1 + state_dim + noise_dim
        hidden_dim = state_dim + hidden_dim_offset

        layers = []
        layers.append(nn.Linear(self.input_dim, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, t, x, dw_prev):
        if isinstance(t, float) or (isinstance(t, torch.Tensor) and t.ndim == 0):
            t = torch.ones(x.shape[0], 1, device=x.device) * t
        elif t.dim() == 1:
            t = t.unsqueeze(1)
        inputs = torch.cat([t, x, dw_prev], dim=1)
        return self.network(inputs)


# ==========================================
# 2. LRV & 物理方程
# ==========================================
class LearnableBranchingNoise(nn.Module):
    def __init__(self, K, M, d, dt):
        super().__init__()
        self.sqrt_dt = np.sqrt(dt)
        self.theta = nn.Parameter(torch.randn(K, M, d, device=device))

    def forward(self):
        return self.theta * self.sqrt_dt


def analytic_solution(t, X, d):
    sum_x = torch.sum(X, dim=-1, keepdim=True)
    exponent = torch.clamp(t + sum_x / d, -30.0, 30.0)
    return torch.exp(exponent) / (1.0 + torch.exp(exponent))


def get_generator_f(t, X, Y, Z, d, sigma):
    sum_z = torch.sum(Z, dim=-1, keepdim=True)
    val = (Y - (d + 2) / (2.0 * d)) * (sum_z / sigma)
    return val


# ==========================================
# 3. 训练流程 (增加 Step 0 相对误差记录)
# ==========================================
def train_fbsde_dynamic_with_plot(config):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    d = config["d"]
    T = config["T"]
    N = config["N"]
    K_batch = config["K_batch"]
    M_branch = config["M_branch"]
    dt = T / N
    sigma = d / np.sqrt(2)
    true_val = 0.5  # Example 1 真实解

    models_Y = [FBSDE_Net_Final(d, d, config["hidden_layers"], config["hidden_dim_offset"], 1).to(device) for _ in
                range(N)]
    models_Z = [FBSDE_Net_Final(d, d, config["hidden_layers"], config["hidden_dim_offset"], d).to(device) for _ in
                range(N)]

    t_grid = torch.linspace(0, T, N + 1, device=device)

    # 记录 Step 0 的详细数据
    loss_history = {
        "step_0": {"loss_y": [], "rel_err": []}
    }

    # 预定义 Step 0 验证用的输入 (t=0, x=0, dw=0)
    t0_val = torch.zeros(1, 1, device=device)
    x0_val = torch.zeros(1, d, device=device)
    dw0_val = torch.zeros(1, d, device=device)

    # 倒向训练
    for i in reversed(range(N)):
        t_curr = t_grid[i].item()
        t_next = t_grid[i + 1].item()
        t_prev = t_grid[i - 1].item() if i > 0 else 0.0

        lrv_layer = LearnableBranchingNoise(K_batch, M_branch, d, dt).to(device)

        all_params = (list(models_Y[i].parameters()) +
                      list(models_Z[i].parameters()) +
                      list(lrv_layer.parameters()))

        optimizer = Adam(all_params, lr=config["lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config["train_steps"] * 0.6), gamma=0.5)

        print(f"  Step {i} Training...", end="\r")

        for step in range(config["train_steps"]):
            # Dynamic Batching
            if i == 0:
                X_k = torch.zeros(K_batch, d, device=device)
                dW_prev_k = torch.zeros(K_batch, d, device=device)
            else:
                X_prev_sim = sigma * np.sqrt(t_prev) * torch.randn(K_batch, d, device=device)
                dW_prev_k = np.sqrt(dt) * torch.randn(K_batch, d, device=device)
                X_k = X_prev_sim + sigma * dW_prev_k

            # Phase 1: Train Z
            optimizer.zero_grad()
            dW_branch = lrv_layer()
            X_curr_expanded = X_k.unsqueeze(1).expand(-1, M_branch, -1)
            X_next_branch = X_curr_expanded + sigma * dW_branch
            X_next_flat = X_next_branch.reshape(-1, d)
            dW_branch_flat = dW_branch.reshape(-1, d)

            if i == N - 1:
                Y_next_flat = analytic_solution(t_next, X_next_flat, d)
            else:
                with torch.no_grad():
                    Y_next_flat = models_Y[i + 1](t_next, X_next_flat, dW_branch_flat)
            Y_next_branch = Y_next_flat.view(K_batch, M_branch, 1)

            Y_mean = torch.mean(Y_next_branch, dim=1, keepdim=True)
            term_Z = (Y_next_branch - Y_mean) * dW_branch
            target_Z = torch.mean(term_Z, dim=1) / dt

            pred_Z = models_Z[i](t_curr, X_k, dW_prev_k)
            loss_Z = torch.mean((pred_Z - target_Z) ** 2)
            loss_Z.backward()
            optimizer.step()

            # Phase 2: Train Y
            optimizer.zero_grad()
            with torch.no_grad():
                pred_Z_updated = models_Z[i](t_curr, X_k, dW_prev_k)

            dW_branch_2 = lrv_layer()
            X_next_flat_2 = (X_curr_expanded + sigma * dW_branch_2).reshape(-1, d)
            dW_branch_flat_2 = dW_branch_2.reshape(-1, d)

            if i == N - 1:
                Y_next_flat_2 = analytic_solution(t_next, X_next_flat_2, d)
            else:
                with torch.no_grad():
                    Y_next_flat_2 = models_Y[i + 1](t_next, X_next_flat_2, dW_branch_flat_2)
            Y_next_branch_2 = Y_next_flat_2.view(K_batch, M_branch, 1)

            pred_Z_exp = pred_Z_updated.unsqueeze(1).expand(-1, M_branch, -1)
            f_vals = get_generator_f(t_curr, X_curr_expanded, Y_next_branch_2, pred_Z_exp, d, sigma)

            term_Y = Y_next_branch_2 + f_vals * dt
            target_Y = torch.mean(term_Y, dim=1)

            pred_Y = models_Y[i](t_curr, X_k, dW_prev_k)
            loss_Y = torch.mean((pred_Y - target_Y) ** 2)
            loss_Y.backward()
            optimizer.step()
            scheduler.step()

            # --- 仅在 Step 0 记录数据 ---
            if i == 0:
                # 记录 Loss Y
                loss_history["step_0"]["loss_y"].append(loss_Y.item())

                # 计算并记录相对误差
                with torch.no_grad():
                    y0_pred = models_Y[0](t0_val, x0_val, dw0_val).item()
                    rel_error = abs(y0_pred - true_val) / true_val
                    loss_history["step_0"]["rel_err"].append(rel_error)

        print(f"  Step {i} Done. Final Loss Y={loss_Y.item():.2e}    ")

    return models_Y[0], loss_history


# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    config = {
        "d": 50, "T": 1.0, "N": 10,
        "K_batch": 64, "M_branch": 200,
        "hidden_layers": 2, "hidden_dim_offset": 110,
        "train_steps": 6000, "lr": 5e-4,
        "seed": 2, "n_runs": 10
    }

    print(f">>> Running Experiment: d={config['d']}, Runs={config['n_runs']} <<<")

    results = []
    first_run_history = None
    true_val = 0.5

    start_time = time.time()

    for run in range(config["n_runs"]):
        current_config = config.copy()
        current_config["seed"] = config["seed"] + run
        print(f"\n--- Run {run + 1}/{config['n_runs']} (Seed={current_config['seed']}) ---")

        model_Y0, history = train_fbsde_dynamic_with_plot(current_config)

        if run == 0:
            first_run_history = history

        with torch.no_grad():
            x0 = torch.zeros(1, config["d"], device=device)
            t0 = torch.zeros(1, 1, device=device)
            dw0 = torch.zeros(1, config["d"], device=device)
            y0_val = model_Y0(t0, x0, dw0).item()
            print(f"Result: {y0_val:.6f}")
            results.append(y0_val)

