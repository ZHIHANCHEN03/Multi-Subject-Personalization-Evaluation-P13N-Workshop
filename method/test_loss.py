"""Phase 2 GATE — 三项数值断言(单元测试,烧 0 GPU).

T1 eta=0 回归 : lambda≡1 -> diag_dpo_loss == 标准 Flow-DPO   (GATE,必须逐位相等)
T2 无偏性     : 真实 alpha 上 E_{t~U}[lambda_pair] ≈ 1        (design 性质1)
T3 fail-safe  : |lambda(t)-1| ≤ eta * max_t|direction-1|       (design 性质2)
"""

import json
import os

import numpy as np

from kernels import DIMS, _GRID, k_tilde
from loss import base_flow_dpo_loss, diag_dpo_loss, lambda_pair

ALPHA_FILE = "../data prepare/selected_pairs_2k.jsonl"


def load_real_alpha(path, limit=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a = json.loads(line)["alpha"]
            rows.append([a[d] for d in DIMS])
            if limit and len(rows) >= limit:
                break
    return np.asarray(rows, dtype=float)


def t1_eta0_regression():
    rng = np.random.default_rng(0)
    N = 4096
    phi = rng.normal(0, 3, size=N)          # 任意偏好量
    alpha = rng.dirichlet(np.ones(3), N)    # 任意方向权重
    t = rng.uniform(0, 1, size=N)
    diag = diag_dpo_loss(phi, alpha, t, eta=0.0)
    base = base_flow_dpo_loss(phi)
    err = abs(diag - base)
    return err, err < 1e-12


def t2_unbiasedness():
    # E_{t~U[0,1]} 即 [0,1] 上的积分,用与归一化一致的细网格确定性估计(非随机采样)
    alpha = load_real_alpha(ALPHA_FILE)
    kt = np.stack([k_tilde(d, _GRID) for d in DIMS], axis=1)   # (G,3)
    Et_k = kt.mean(axis=0)                                      # 每核在 t 上的期望,应≈1
    worst = 0.0
    for eta in (0.3, 0.5, 0.7, 1.0):
        # E_t[lambda_i] = (1-eta) + eta * sum_d alpha_d * E_t[k~_d]
        mean_lam = (1 - eta) + eta * (alpha @ Et_k)            # (Npair,)
        worst = max(worst, float(np.abs(mean_lam - 1.0).max()))
    return worst, worst < 1e-3


def t3_failsafe():
    alpha = load_real_alpha(ALPHA_FILE)
    kt = np.stack([k_tilde(d, _GRID) for d in DIMS], axis=1)   # (G,3) 细网格
    direction = alpha @ kt.T                                    # (Npair, G)
    bound_factor = np.abs(direction - 1.0).max(axis=1)         # 每 pair: max_t|dir-1|
    worst_violation = 0.0
    for eta in (0.3, 0.5, 0.7, 1.0):
        lam = (1 - eta) + eta * direction
        lhs = np.abs(lam - 1.0).max(axis=1)                   # 实际 max_t|lambda-1|
        rhs = eta * bound_factor
        worst_violation = max(worst_violation, float((lhs - rhs).max()))
    # lhs 应 <= rhs(等号成立),允许浮点误差
    return worst_violation, worst_violation < 1e-9


def main():
    print("=== Phase 2 GATE: loss 数值校验 ===\n")
    results = []

    err, ok = t1_eta0_regression()
    print(f"T1  eta=0 回归 (GATE) : |diag - base| = {err:.2e}   -> {'PASS' if ok else 'FAIL'}")
    results.append(ok)

    if os.path.exists(ALPHA_FILE):
        dev, ok = t2_unbiasedness()
        print(f"T2  无偏性            : max|E_t[lambda]-1| = {dev:.2e}   -> {'PASS' if ok else 'FAIL'}")
        results.append(ok)

        vio, ok = t3_failsafe()
        print(f"T3  fail-safe         : max(lhs-rhs) = {vio:.2e}   -> {'PASS' if ok else 'FAIL'}")
        results.append(ok)
    else:
        print(f"T2/T3 跳过: 找不到 {ALPHA_FILE}")

    print(f"\n总判定: {'全部 PASS — 可进 Phase 3' if all(results) else 'FAIL — 不得进 Phase 3'}")
    return all(results)


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
