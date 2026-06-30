"""Phase 1 — 核实例化(数值,非实验).

三个固定宽核(截断高斯),峰值写死、宽度共用,在 [0,1] 上归一化为均值=1.
进入 DTR 重加权: lambda_pair(t) = (1-eta) + eta * sum_d alpha_d * k_tilde_d(t)

核形 : k_d(t) = exp(-(t-mu_d)^2 / (2 sigma^2)),  t in [0,1]
归一 : k_tilde_d(t) = k_d(t) / Z_d,  Z_d = ∫_0^1 k_d(t) dt
       => E_{t~U[0,1]}[k_tilde_d] = 1  (design 性质1 无偏性的来源)
"""

import numpy as np

DIMS = ["existence", "appearance", "interaction"]

# 峰值写死(归纳偏置,只 claim 顺序 E>I>A,不 claim 坐标)
MU = {"existence": 0.8, "interaction": 0.5, "appearance": 0.2}
SIGMA = 0.2  # 三核共用的宽度旋钮(Phase 5 核宽消融即扫这个)

_GRID = np.linspace(0.0, 1.0, 1001)  # 归一化用的细网格


def _gauss(t, mu, sigma=SIGMA):
    return np.exp(-((np.asarray(t, dtype=float) - mu) ** 2) / (2.0 * sigma ** 2))


def _trapz(f, x):
    """版本无关的梯形数值积分(避免 np.trapz/np.trapezoid 命名差异)."""
    dx = np.diff(x)
    return float(np.sum((f[:-1] + f[1:]) / 2.0 * dx))


# 预先把 3 个归一化常数(曲线下面积)算出来,烘进模块
Z = {d: _trapz(_gauss(_GRID, MU[d]), _GRID) for d in DIMS}


def k_tilde(dim, t):
    """归一化核 k_tilde_d(t),均值=1.  dim in DIMS, t 标量或数组."""
    return _gauss(t, MU[dim]) / Z[dim]


def verify():
    """校验: 每核积分≈1; 报告归一化后的经验峰位(截断会略内移)."""
    rows = []
    for d in DIMS:
        kt = k_tilde(d, _GRID)
        integral = _trapz(kt, _GRID)
        peak_t = float(_GRID[int(np.argmax(kt))])
        rows.append((d, MU[d], peak_t, integral, kt.max()))
    return rows


def plot(out="figures/kernels.png"):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out), exist_ok=True)
    colors = {"existence": "#4C72B0", "appearance": "#DD8452", "interaction": "#55A868"}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for d in DIMS:
        ax.plot(_GRID, k_tilde(d, _GRID), color=colors[d],
                label=f"k_{d[:1].upper()}  (peak mu={MU[d]})")
        ax.axvline(MU[d], color=colors[d], ls=":", alpha=0.5)
    ax.axhline(1.0, color="gray", ls="--", lw=1, label="mean = 1 (normalized)")
    ax.set_xlabel("diffusion time  t")
    ax.set_ylabel("k_tilde(t)")
    ax.set_title(f"DTR fixed-width kernels (truncated Gaussian, sigma={SIGMA})  order E>I>A")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(f"sigma = {SIGMA}")
    print(f"归一化常数 Z = {{ " + ", ".join(f'{d}:{Z[d]:.4f}' for d in DIMS) + " }}")
    print("\n校验表:")
    print(f"{'dim':<12}{'目标峰':>8}{'经验峰':>8}{'∫k~ dt':>10}{'峰值高度':>10}")
    ok = True
    for d, mu, peak, integral, kmax in verify():
        print(f"{d:<12}{mu:>8.2f}{peak:>8.2f}{integral:>10.4f}{kmax:>10.3f}")
        if abs(integral - 1.0) > 1e-3:
            ok = False
    order = [MU[d] for d in ["existence", "interaction", "appearance"]]
    print(f"\n积分=1 校验: {'通过' if ok else '失败'}")
    print(f"峰序 E>I>A 校验: {'通过' if order[0] > order[1] > order[2] else '失败'}")
    out = plot()
    print(f"曲线图 -> {out}")
