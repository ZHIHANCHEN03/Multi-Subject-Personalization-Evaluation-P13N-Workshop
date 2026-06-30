"""Phase 3b — Diag-DPO 训练(短).

只载 transformer + LoRA, 读离线缓存(cache_latents.py 产物), 用 accelerate。
e_ref = 同一 transformer 关 LoRA(disable_adapters); eta=0 精确退化标准 Flow-DPO。
η 扫描 = 改 --eta 重跑(0/0.3/0.5/0.7)。

FLUX.2 调用约定(已对源码):
  hidden = cat([x_tokens, ref_tokens], dim=1)   # refs 拼成干净上下文
  img_ids = cat([target_ids, ref_ids], dim=1)   # 4D ids, refs 靠 T 坐标时间分隔
  timestep ∈ [0,1] 直接传(pipeline 内部是 t/1000)
  out[:, :n_x] 只取目标段 -> FM 误差不沾 ref
"""

import argparse
import glob
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from kernels import DIMS, MU, SIGMA, Z   # 核常数: 单一真值源(Phase 1 已验证 ∫=1)


def lambda_pair(alpha, t, eta):
    """(1-eta) + eta * sum_d alpha_d k~_d(t).  alpha:(B,3) t:(B,)."""
    d = torch.zeros_like(t)
    for i, dim in enumerate(DIMS):
        d = d + alpha[:, i] * (torch.exp(-((t - MU[dim]) ** 2) / (2 * SIGMA ** 2)) / Z[dim])
    return (1 - eta) + eta * d


def fm_error(v, u):
    return ((v - u) ** 2).flatten(1).mean(1)   # (B,)


class CacheDS(torch.utils.data.Dataset):
    def __init__(self, d):
        self.files = sorted(glob.glob(os.path.join(d, "*.pt")))
        assert self.files, f"no .pt in {d}"

    def __len__(self): return len(self.files)
    def __getitem__(self, i): return torch.load(self.files[i])


def collate_b1(batch):
    assert len(batch) == 1, "变长 refs -> 当前 batch_size=1"
    return batch[0]


def velocity(tf, z_t, t, target_ids, prompt_embeds, txt_ids, guidance, ref_tokens, ref_ids):
    """预测 x 段速度. refs 拼为干净上下文; 只返回前 n_x 个 token."""
    hidden, ids = z_t, target_ids
    if ref_tokens is not None:
        hidden = torch.cat([z_t, ref_tokens], dim=1)         # [x | refs]
        ids = torch.cat([target_ids, ref_ids], dim=1)        # 4D ids, refs 用 T 分隔
    out = tf(hidden_states=hidden, timestep=t, guidance=guidance,
             encoder_hidden_states=prompt_embeds, txt_ids=txt_ids, img_ids=ids,
             return_dict=False)[0]
    return out[:, :z_t.shape[1]]                              # ★ FM 误差不沾 ref


def save_lora(model, path):
    from peft.utils import get_peft_model_state_dict
    Path(path).mkdir(parents=True, exist_ok=True)
    torch.save(get_peft_model_state_dict(model), Path(path) / "lora.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--cache_dir", required=True, help="cache_latents.py 的产物目录")
    ap.add_argument("--output_dir", default="/workspace/ckpt")
    ap.add_argument("--eta", type=float, default=0.5, help="0=baseline; >0=Diag-DPO")
    ap.add_argument("--beta", type=float, default=2000.0)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0, help="过几遍数据(max_steps=0 时生效)")
    ap.add_argument("--max_steps", type=int, default=0, help="0=由 --epochs×数据量 推算; >0 手动覆盖(dry run 用)")
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    acc = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=args.grad_accum)

    from diffusers import Flux2Transformer2DModel
    from peft import LoraConfig
    tf = Flux2Transformer2DModel.from_pretrained(
        args.model_path, subfolder="transformer", torch_dtype=torch.bfloat16)
    tf.requires_grad_(False)
    tf.enable_gradient_checkpointing()
    tf.add_adapter(LoraConfig(                                # [CONFIRM] target_modules 名
        r=args.lora_rank, lora_alpha=args.lora_rank, init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0",
                        "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"]))

    ds = CacheDS(args.cache_dir)
    # max_steps: 默认 1 epoch = 过一遍全部 pair; 显式 --max_steps>0 则覆盖(dry run)
    max_steps = args.max_steps if args.max_steps > 0 else round(args.epochs * len(ds))
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_b1)
    opt = torch.optim.AdamW([p for p in tf.parameters() if p.requires_grad], lr=args.lr)
    tf, opt, dl = acc.prepare(tf, opt, dl)
    model = acc.unwrap_model(tf)
    guidance = torch.full((1,), args.guidance, device=acc.device, dtype=torch.float32)
    if acc.is_main_process:
        print(f"[plan] pairs={len(ds)}  epochs={args.epochs}  max_steps={max_steps}  "
              f"grad_accum={args.grad_accum}  -> weight_updates~{max_steps // args.grad_accum}",
              flush=True)

    tf.train()
    step = 0
    while step < max_steps:
        for b in dl:
            if step >= max_steps:
                break
            with acc.accumulate(tf):
                dev = acc.device
                bf = torch.bfloat16
                z_p = b["z_plus"].unsqueeze(0).to(dev, bf)
                z_m = b["z_minus"].unsqueeze(0).to(dev, bf)
                tids = b["target_ids"].unsqueeze(0).to(dev)
                rt = b["ref_tokens"].unsqueeze(0).to(dev, bf) if b["ref_tokens"] is not None else None
                rids = b["ref_ids"].unsqueeze(0).to(dev) if b["ref_ids"] is not None else None
                pe = b["prompt_embeds"].unsqueeze(0).to(dev, bf)
                ti = b["txt_ids"].unsqueeze(0).to(dev)
                alpha = b["alpha"].unsqueeze(0).to(dev)

                # 配对共享: 同一 t、同一 eps
                t = torch.rand(1, device=dev, dtype=bf)
                eps = torch.randn_like(z_p)
                tt = t.view(1, 1, 1)
                zt_p, zt_m = (1 - tt) * z_p + tt * eps, (1 - tt) * z_m + tt * eps
                u_p, u_m = eps - z_p, eps - z_m              # rectified flow 目标

                # theta(LoRA 开)
                v_thp = velocity(tf, zt_p, t, tids, pe, ti, guidance, rt, rids)
                v_thm = velocity(tf, zt_m, t, tids, pe, ti, guidance, rt, rids)
                # e_ref(LoRA 关, 无梯度) —— 同一份权重, 不需第二份模型
                model.disable_adapters()
                with torch.no_grad():
                    v_rfp = velocity(tf, zt_p, t, tids, pe, ti, guidance, rt, rids)
                    v_rfm = velocity(tf, zt_m, t, tids, pe, ti, guidance, rt, rids)
                model.enable_adapters()

                phi = -args.beta * ((fm_error(v_thp, u_p) - fm_error(v_rfp, u_p))
                                    - (fm_error(v_thm, u_m) - fm_error(v_rfm, u_m)))
                lam = lambda_pair(alpha, t.float(), args.eta)
                loss = -(lam * F.logsigmoid(phi.float())).mean()

                acc.backward(loss)
                opt.step()
                opt.zero_grad(set_to_none=True)

            if step % args.log_every == 0 and acc.is_main_process:
                print(f"step {step:>6} | loss {loss.item():.4f} | phi>0 {(phi > 0).float().mean():.3f}",
                      flush=True)
            if step > 0 and step % args.save_every == 0 and acc.is_main_process:
                save_lora(acc.unwrap_model(tf),
                          Path(args.output_dir) / f"lora_eta{args.eta}_step{step}")
            step += 1

    acc.wait_for_everyone()
    if acc.is_main_process:
        out = Path(args.output_dir) / f"lora_eta{args.eta}"
        save_lora(acc.unwrap_model(tf), out)
        print(f"[done] saved LoRA -> {out}")


if __name__ == "__main__":
    main()
