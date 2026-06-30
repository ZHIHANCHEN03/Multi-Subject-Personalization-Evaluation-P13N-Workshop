"""Phase 3a — 离线缓存(只跑一次).

把 FLUX.2 的重 I/O 全部离线做掉,每个 pair 存一个 .pt:
  VAE 编码 x+/x- -> 打包潜 token + 4D ids ; refs 打包成上下文 token + 时间分隔 ids ; 文本编码 prompt
训练循环(train_dpo.py)因此只剩 transformer + DPO 数学, 又快又短。

关键: 缓存阶段 transformer=None 不加载 32B 主干
      (Mistral 文本编码器 ~48G 已占大半, 再加 64G transformer 会 OOM)。

FLUX.2 真实 API(已在 0.38 源码核对):
  _encode_vae_image(image, generator) -> (1,128,32,32) 已含 shift/scale
  _pack_latents(latents)              -> (1, H*W, C)    纯 reshape, 无 patchify
  _prepare_latent_ids(latents)        -> (1, H*W, 4)    目标, T=0
  prepare_image_latents(list,...)     -> (1, N*H*W, C), (1, N*H*W, 4)  refs, 每张 T=10,20,...
  encode_prompt(prompt, device)       -> (prompt_embeds, text_ids)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from kernels import DIMS

IMG_EXTS = [".jpg", ".jpeg", ".png", ".webp"]


def load_img(path, res):
    img = Image.open(path).convert("RGB").resize((res, res))
    t = torch.from_numpy(np.asarray(img)).float() / 127.5 - 1.0
    return t.permute(2, 0, 1)  # CHW in [-1,1]


def resolve(root, key):
    for e in IMG_EXTS:
        p = Path(root) / f"{key}{e}"
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--pairs_file", required=True, help="selected_pairs_{2k,5k}.jsonl")
    ap.add_argument("--prompt_file", required=True, help="train_60k_v13_2.jsonl")
    ap.add_argument("--image_a_root", required=True)
    ap.add_argument("--image_b_root", required=True)
    ap.add_argument("--refs_root", required=True)
    ap.add_argument("--out_dir", default="/workspace/cache")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只缓存前 N 个(冒烟测试)")
    args = ap.parse_args()

    device, dtype = "cuda", torch.bfloat16
    from diffusers import Flux2Pipeline
    # transformer=None: 缓存只需 vae + 文本编码器, 跳过 64G 主干
    pipe = Flux2Pipeline.from_pretrained(
        args.model_path, transformer=None, torch_dtype=dtype).to(device)
    gen = torch.Generator(device=device).manual_seed(0)

    @torch.no_grad()
    def enc_target(path):
        img = load_img(path, args.resolution).unsqueeze(0).to(device, dtype)
        lat = pipe._encode_vae_image(image=img, generator=gen)   # (1,128,32,32)
        z = pipe._pack_latents(lat)[0].cpu()                     # (H*W, C)
        ids = pipe._prepare_latent_ids(lat)[0].cpu()             # (H*W, 4)
        return z, ids

    @torch.no_grad()
    def enc_refs(paths):
        imgs = [load_img(p, args.resolution).unsqueeze(0).to(device, dtype) for p in paths]
        toks, ids = pipe.prepare_image_latents(
            imgs, batch_size=1, generator=gen, device=device, dtype=dtype)
        return toks[0].cpu(), ids[0].cpu()                       # (N*H*W,C),(N*H*W,4)

    @torch.no_grad()
    def enc_prompt(text):
        pe, tids = pipe.encode_prompt(prompt=[text], device=device)
        return pe[0].cpu(), tids[0].cpu()                        # (txt_seq,D),(txt_seq,4)

    # task_id -> prompt / subject 名字
    prompt, names = {}, {}
    for line in open(args.prompt_file, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        tid = str(r["id"])
        prompt[tid] = r.get("prompt_en") or r.get("prompt") or ""
        names[tid] = list(r.get("people_names", [])) + list(r.get("object_names", []))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    for line in open(args.pairs_file, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        tid = str(r["task_id"])
        a, b = resolve(args.image_a_root, tid), resolve(args.image_b_root, tid)
        refp = [resolve(args.refs_root, nm) for nm in names.get(tid, [])]
        if not a or not b or tid not in prompt or not refp or any(x is None for x in refp):
            dropped += 1
            continue
        plus, minus = (a, b) if r["winner"] == "A" else (b, a)

        z_p, tids = enc_target(plus)
        z_m, _ = enc_target(minus)
        rtok, rids = enc_refs(refp)
        pe, txt_ids = enc_prompt(prompt[tid])

        torch.save({
            "z_plus": z_p, "z_minus": z_m, "target_ids": tids,
            "ref_tokens": rtok, "ref_ids": rids,
            "prompt_embeds": pe, "txt_ids": txt_ids,
            "alpha": torch.tensor([r["alpha"][d] for d in DIMS], dtype=torch.float32),
        }, out / f"{tid}.pt")
        kept += 1
        if kept % 100 == 0:
            print(f"cached {kept} ...", flush=True)
        if args.limit and kept >= args.limit:
            break

    print(f"[done] cached {kept} pairs to {out}  (dropped {dropped})")


if __name__ == "__main__":
    main()
