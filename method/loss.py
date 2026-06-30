"""Phase 2 — Diag-DPO loss(数学核心,模型无关).

只实现 design 第五部分最终 loss 的"权重折叠"那一层,不含扩散模型本身。
模型耦合(v_theta / e_ref / x_t)留到 Phase 3 接线;这里把 FM 误差 e 当输入。

最终 loss:
  L = -E_pair E_{t~U[0,1]} [ lambda_pair(t) * log sigma(Phi) ]

  lambda_pair(t) = (1-eta) + eta * sum_d alpha_d * k_tilde_d(t)
  Phi = -beta * [ (e_theta(x+) - e_ref(x+)) - (e_theta(x-) - e_ref(x-)) ]

公式逐元素,直接可换成 torch(np.* -> torch.*)。
"""

import numpy as np

from kernels import DIMS, k_tilde


def lambda_pair(alpha, t, eta):
    """逐 pair 时间步重加权系数.

    alpha : (..., 3)  正贡献归一化方向权重,按 DIMS 顺序,sum_d alpha_d = 1
    t     : (...)     扩散时间,[0,1]
    eta   : 标量      唯一旋钮
    返回    (...)     lambda_pair(t) = (1-eta) + eta * sum_d alpha_d k_tilde_d(t)
    """
    alpha = np.asarray(alpha, dtype=float)
    direction = sum(alpha[..., i] * k_tilde(DIMS[i], t) for i in range(len(DIMS)))
    return (1.0 - eta) + eta * direction


def phi_from_errors(e_theta_p, e_ref_p, e_theta_m, e_ref_m, beta):
    """偏好量 Phi,由 FM 速度回归误差算出(x+ 记 p,x- 记 m)."""
    return -beta * ((e_theta_p - e_ref_p) - (e_theta_m - e_ref_m))


def _logsigmoid(x):
    """数值稳定 log sigma(x) = -log(1+e^{-x})."""
    return -np.logaddexp(0.0, -np.asarray(x, dtype=float))


def diag_dpo_loss(phi, alpha, t, eta):
    """完整 Diag-DPO loss(对一个 batch 的逐 pair 量取均值).

    phi   : (N,)    每个 pair 在其采样 t 处的偏好量
    alpha : (N, 3)  每个 pair 的方向权重
    t     : (N,)    每个 pair 采样到的时间步
    """
    lam = lambda_pair(alpha, t, eta)
    return -float(np.mean(lam * _logsigmoid(phi)))


def base_flow_dpo_loss(phi):
    """标准 Flow-DPO(lambda 恒等于 1),即 eta=0 应该退化到的形态."""
    return -float(np.mean(_logsigmoid(phi)))
