# Diag-DPO 进度小结(2026-06-26)

## 1. 今天的成果

**环境(RunPod H100 NVL 94GB)全部打通:**
- 修复 torch 2.4→**2.5.1**(2.4 会让 FLUX.2 import 直接崩,根因是 flash-attn-3 的 `float|None` 签名)
- 下载 FLUX.2-dev 权重(178GB)到 `/root/flux2-dev`
- 把 FLUX.2 真实 API 摸清并据此**重写** `cache_latents.py` + `train_dpo.py`
  - 4D ids(T,H,W,L);refs 走 `prepare_image_latents`,靠 T 坐标(10,20,…)时间分隔
  - `_pack_latents` 纯 reshape 无 patchify;timestep ∈ [0,1];输出只取目标段
  - 缓存阶段 `transformer=None`(只上 VAE+文本编码器,否则 +64G 会 OAM)

**两个冒烟测试都通过:**
- **缓存 2 个 pair** → 张量形状全对(`z (1024,128)`、`prompt_embeds (512,15360)`、refs=N×1024)
- **训练 2 步** → LoRA `target_modules` 匹配、`disable/enable_adapters` 可用、N=6 不 OOM、loss 出数、LoRA 存盘

**其他:**
- `train_dpo.py` 改为默认 **1 epoch**(`max_steps = epochs × 缓存 pair 数`,可用 `--max_steps` 覆盖)
- 已缓存 10 个 pair 到 `/workspace/cache_dry`(给 dry run 用)


---

## 2. Dry run(下一步,马上做)

**目的**:用 10 个 pair 快速确认 **loss 方向没写反**(不是训模型)。

**现在有**:`/workspace/cache_dry`(10 个 .pt)、更新版 `train_dpo.py`。

**做**:
```bash
cd /workspace/code && python train_dpo.py \
  --model_path /root/flux2-dev --cache_dir /workspace/cache_dry \
  --output_dir /workspace/ckpt_dry \
  --eta 0.5 --max_steps 40 --grad_accum 1 --log_every 5 --save_every 999
```

**要看到**:
- loss 始终**有限**(无 NaN/inf)
- `phi>0` 从 ~0.5 **逐步爬向 1.0** ← 关键,说明梯度方向正确
- 全程不 OOM

通过 → 放心做全量缓存。

---

## 3. Phase 2.5 — η 试点(500,GO/NO-GO)

**目的**:正式烧 4 组前,小规模确认"加 DTR 真有信号",决定 GO/NO-GO。

**做法**:
1. 先全量缓存 ~4824 个 pair → `/workspace/cache`(一次性,几小时)
2. 从中**按 12 格(N×primary_dim)分层等比抽 500**(固定 seed,可复现),软链到 `/workspace/cache_pilot`
3. 在这 500 上各跑 **η=0** 和 **η=0.5**(1 epoch ≈ 500 步),其余超参全同
4. 在 gold 小子集上快速测一轮

**要看到**(满足即 GO):
- 两组都训练**稳定**(loss 降、无崩、FID 不塌)
- η=0.5 在偏好/E·A·I 上 **≥ η=0**(哪怕弱信号),方向为正
- 反之(η=0.5 更差或画质塌)→ NO-GO,回头查 loss/超参

---

## 4. Phase 3 — 正式训练(主结果)

**就是产出论文主表的那步。**

- **缓存**:全部 ~4824 个 pair(Phase 2.5 已做完则复用)
- **训练**:跑 **4 组** η ∈ {0, 0.3, 0.5, 0.7},每组 **1 epoch**,**只改 `--eta`**,其余(seed/lr/beta/grad_accum/步数)完全一致 → 保证可比
  - η=0 = 标准 Flow-DPO baseline
- **产物**:4 个 LoRA → 走 eval_protocol 评测(gold 集 + GPT-4V 裁判 + FID guard)→ 主表

```bash
# 每组(改 --eta 重跑 4 次):
cd /workspace/code && python train_dpo.py \
  --model_path /root/flux2-dev --cache_dir /workspace/cache \
  --output_dir /workspace/ckpt --eta 0.5
```
