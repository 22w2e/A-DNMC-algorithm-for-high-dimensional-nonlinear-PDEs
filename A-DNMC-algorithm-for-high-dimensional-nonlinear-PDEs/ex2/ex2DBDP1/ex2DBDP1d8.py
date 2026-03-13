import os

# ★★★ FIX: 解决 OpenMP Error #15 ★★★
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
import matplotlib.pyplot as plt
import time
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. 网络定义 (保持不变)
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
# 2. 物理方程: Example 2 (保持不变)
# ==========================================
def get_constants(d):
    return (d + 1) * (2 * d + 1) / 12.0


def func_A(x, d):
    mask = (x < 0).float()
    return torch.sum(torch.sin(x) * mask, dim=-1, keepdim=True) / d


def func_B(x, d):
    mask = (x >= 0).float()
    return torch.sum(x * mask, dim=-1, keepdim=True) / d


def func_sum_lx(x):
    d_dim = x.shape[-1]
    l_vec = torch.arange(1, d_dim + 1, device=x.device, dtype=x.dtype)
    return torch.sum(x * l_vec, dim=-1, keepdim=True)


def analytic_solution(t, x, d):
    T = 1.0
    if isinstance(t, float):
        t_val = t
    elif isinstance(t, torch.Tensor):
        t_val = t.reshape(-1, 1)
        if x.ndim == 3 and t_val.ndim == 2:
            t_val = t_val.unsqueeze(1)
    else:
        t_val = t

    term1 = (T - t_val) * (func_A(x, d) + func_B(x, d))
    term2 = torch.cos(func_sum_lx(x))
    return term1 + term2


def get_generator_f(t, x, y, z, d, sigma):
    T = 1.0
    C = get_constants(d)
    if isinstance(t, float):
        t_val = t
    elif isinstance(t, torch.Tensor):
        t_val = t.reshape(-1, 1)
        if x.ndim == 3 and t_val.ndim == 2:
            t_val = t_val.unsqueeze(1)
    else:
        t_val = t

    term_A = (1 + (T - t_val) / (2 * d)) * func_A(x, d)
    term_B = func_B(x, d)
    term_C = C * torch.cos(func_sum_lx(x))

    return term_A + term_B + term_C


# ==========================================
# 3. 训练流程 (替换为 Branched DBDP1)
# ==========================================
def train_fbsde_dynamic_with_plot(config):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    d = config["d"]
    T = config["T"]
    N = config["N"]
    K_batch = config["K_batch"]
    M_branch = config["M_branch"]  # 读取分支数
    dt = T / N

    sigma = 1.0 / np.sqrt(d)

    models_Y = [FBSDE_Net_Final(d, d, config["hidden_layers"], config["hidden_dim_offset"], 1).to(device) for _ in
                range(N)]
    models_Z = [FBSDE_Net_Final(d, d, config["hidden_layers"], config["hidden_dim_offset"], d).to(device) for _ in
                range(N)]

    t_grid = torch.linspace(0, T, N + 1, device=device)

    loss_history = {
        "step_0": {"loss_y": [], "rel_err": []}
    }

    t0_val = torch.zeros(1, 1, device=device)
    x0_val = torch.full((1, d), 0.5, device=device)
    dw0_val = torch.zeros(1, d, device=device)
    true_val = analytic_solution(0.0, x0_val, d).item()

    for i in reversed(range(N)):
        t_curr = t_grid[i].item()
        t_next = t_grid[i + 1].item()
        t_prev = t_grid[i - 1].item() if i > 0 else 0.0

        all_params = list(models_Y[i].parameters()) + list(models_Z[i].parameters())

        optimizer = Adam(all_params, lr=config["lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config["train_steps"] * 0.6), gamma=0.5)

        print(f"  Step {i} Training...", end="\r")

        for step in range(config["train_steps"]):
            optimizer.zero_grad()

            # 1. 采样当前状态 X_k
            if i == 0:
                X_k = torch.full((K_batch, d), 0.5, device=device)
            else:
                X_prev_sim = torch.full((K_batch, d), 0.5, device=device) + \
                             sigma * np.sqrt(t_prev) * torch.randn(K_batch, d, device=device)
                X_k = X_prev_sim + sigma * np.sqrt(dt) * torch.randn(K_batch, d, device=device)

            # --- 核心改动：引入 M 个分支的随机采样 ---
            X_curr_expanded = X_k.unsqueeze(1).expand(-1, M_branch, -1)  # (K, M, d)
            dW_branch = np.sqrt(dt) * torch.randn(K_batch, M_branch, d, device=device)
            X_next_branch = X_curr_expanded + sigma * dW_branch  # (K, M, d)

            X_next_flat = X_next_branch.reshape(-1, d)
            dW_branch_flat = dW_branch.reshape(-1, d)

            # 2. 获取下一时刻的目标值 Y_{next} (并 reshape 回 M 个分支)
            if i == N - 1:
                Y_next_flat = analytic_solution(t_next, X_next_flat, d)
            else:
                with torch.no_grad():
                    Y_next_flat = models_Y[i + 1](t_next, X_next_flat, dW_branch_flat)
            Y_next_branch = Y_next_flat.view(K_batch, M_branch, 1)  # (K, M, 1)

            # 3. 预测当前时刻的 Y 和 Z (基于单点 X_k)
            dummy_dw = torch.zeros(K_batch, d, device=device)
            Y_pred = models_Y[i](t_curr, X_k, dummy_dw)  # (K, 1)
            Z_pred = models_Z[i](t_curr, X_k, dummy_dw)  # (K, d)

            # 将预测的 Y_curr 和 Z_curr 扩展 M 个分支以便计算
            Y_pred_exp = Y_pred.unsqueeze(1).expand(-1, M_branch, -1)  # (K, M, 1)
            Z_pred_exp = Z_pred.unsqueeze(1).expand(-1, M_branch, -1)  # (K, M, d)

            # 4. 计算生成器 f (使用扩展后的张量)
            f_val = get_generator_f(t_curr, X_curr_expanded, Y_pred_exp, Z_pred_exp, d, sigma)

            # 5. ★ Branched DBDP1 核心损失计算 (Euler-Maruyama 离散化残差) ★
            Z_dW = torch.sum(Z_pred_exp * dW_branch, dim=-1, keepdim=True)  # (K, M, 1)
            Y_forward_step = Y_pred_exp - f_val * dt + Z_dW  # (K, M, 1)

            # 计算每个分支的平方误差
            squared_errors = (Y_next_branch - Y_forward_step) ** 2  # (K, M, 1)

            # 总体损失求均值 (同时惩罚 Y 和 Z)
            loss = torch.mean(squared_errors)

            loss.backward()
            optimizer.step()
            scheduler.step()

            # --- Step 0 记录 ---
            if i == 0:
                loss_history["step_0"]["loss_y"].append(loss.item())
                with torch.no_grad():
                    y0_pred = models_Y[0](t0_val, x0_val, dw0_val).item()
                    rel_error = abs(y0_pred - true_val) / abs(true_val)
                    loss_history["step_0"]["rel_err"].append(rel_error)

        print(f"  Step {i} Done. Final Branched DBDP1 Loss={loss.item():.2e}    ")

    return models_Y[0], loss_history


# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    config = {
        "d": 8,
        "T": 1.0,
        "N": 90,
        "K_batch": 64,
        "M_branch": 400,
        "hidden_layers": 2,
        "hidden_dim_offset": 110,
        "train_steps": 3000,
        "lr": 5e-4,
        "seed": 140,
        "n_runs": 10
    }

    print(f">>> Running Branched DBDP1 (Ex 2): d={config['d']}, N={config['N']}, Runs={config['n_runs']} <<<")

    results = []
    first_run_history = None

    dummy_x0 = torch.full((1, config['d']), 0.5, device=device)
    true_val = analytic_solution(0.0, dummy_x0, config['d']).item()
    print(f"True Value (u(0, x0)): {true_val:.6f}")

    start_time = time.time()

    for run in range(config["n_runs"]):
        current_config = config.copy()
        current_config["seed"] = config["seed"] + run
        print(f"\n--- Run {run + 1}/{config['n_runs']} (Seed={current_config['seed']}) ---")

        model_Y0, history = train_fbsde_dynamic_with_plot(current_config)

        if run == 0:
            first_run_history = history

        with torch.no_grad():
            x0 = torch.full((1, config["d"]), 0.5, device=device)
            t0 = torch.zeros(1, 1, device=device)
            dw0 = torch.zeros(1, config["d"], device=device)
            y0_val = model_Y0(t0, x0, dw0).item()
            print(f"Result: {y0_val:.6f}")
            results.append(y0_val)

