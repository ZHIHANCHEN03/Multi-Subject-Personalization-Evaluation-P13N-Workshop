# 数据 & 图片在 GPU 服务器上的位置

> 本地 repo(`submission/` / `companion/` / `meta/`)只保留了一部分数据。
> 大体量生成图、参考图、MIE checkpoint、frozen calibration 都在 GPU 服务器的
> `/workspace` 共享卷上。下面是完整映射,方便日后拉取 / 复现 / 重新跑实验。

## 0. 服务器

- 共享卷根目录:`/workspace`(在 A100 / CPU 服务器上都挂载到同一个网络卷,无需重复下载)
- repo 在服务器上的位置:`/workspace/misc/`(脚本里 `REPO="misc"` 就是这个)
  - 因此本地 `submission/round2/...` 对应服务器 `/workspace/misc/round2/...`
  - 注意:服务器上是**平铺**的 `round1/` `round2/` `prompt/` `refs/` 等,
    不是 `submission/round2/`。本地 `submission/` 是为了打包投稿做的子集。

## 1. 输入参考图(主体身份图,80 张 JPG)

- 服务器:`/workspace/misc/refs/<name>.jpg`(80 个:人 + 物件)
- 本地副本:`meta/refs/`(同样 80 张,gitignored)
- 配套副本:`companion/MIBE_Core/.../Model_Training/data_v2/refs/`
- 脚本里用 `REFS_DIR` 环境变量引用,默认 `$REPO/refs`

## 2. 任务 manifest(prompt 数据)

- 服务器:`/workspace/misc/prompt/train_60k_v13_2.jsonl`(60K 任务,42 MB)
- 本地副本:`companion/prompt/train_60k_v13_2.jsonl`(gitignored,属 MIBE 配套)
- 脚本里用 `DATA_SRC` 引用,默认 `$REPO/prompt/train_60k_v13_2.jsonl`
- 每条记录字段:`id, seed_id, level, class_tag, n_humans, n_objects, total_entities, people_names, object_names, prompt_en, prompt_zh, ...`
  - 主体名 → `refs/<name>.jpg` 由 `round1/select_hard_cases.py --refs` 解析

## 3. MIE verifier checkpoint(LoRA)

- 服务器:`/workspace/Model_Training_runs/v2/unsloth_Qwen3.5-4B/20260503_045230/outputs/unsloth_Qwen3.5-4B-lora_layer-best/`
- base model:`unsloth_Qwen3.5-4B`(Qwen3.5-4B,LoRA fine-tuned,~50 MB)
- 脚本里用 `MIE_CKPT` 引用
- 本地**没有**副本(不进投稿)

## 4. Frozen calibration(MIE per-facet baselines)

- 服务器:`/workspace/misc/round1/results/calibration/mie_baselines.json`(OmniGen2)
- 服务器:`/workspace/misc/round2/results_flux2/mie_baselines_flux2.json`(FLUX.2)
- 脚本里用 `CALIBRATION` 引用
- 本地**没有**副本(round1/results 整个目录 gitignored)

## 5. 生成图 — OmniGen2 主实验(几百张,**本地没有**)

- 服务器分片输出:`/workspace/misc/round2/results_r2/shard_{0,1,2,3}/<method>_s<seed>/images/hard_XXXX.png`
- 合并后:`/workspace/misc/round2/results_r2/merged/<method>_s<seed>/images/hard_XXXX.png`
  - `<method>` ∈ {`ours_v2`, `best_of_n`, `one_shot`, `umo`}
  - `<seed>` ∈ {0, 1, 2}
  - 4 方法 × 3 seed × 500 任务(hard_4 250 + easy_2 250)= 6000 张
- 元数据 records 已提交到本地:`submission/round2/results_r2/merged/<method>_s<seed>/records.jsonl`(12 个文件)
  - 每条记录里 `image_path` 字段是相对路径 `images/hard_XXXX.png`
  - **要复现图片**:在服务器上进入对应 `merged/<method>_s<seed>/` 目录,`images/` 子目录就是 PNG

## 6. 生成图 — FLUX.2 scaling(已进 git,9 张样本)

- 服务器:`/workspace/misc/round2/results_flux2/flux2_{6,8}_{bon,oneshot,ours}_s{0,1,2}/images/hard_XXXX.png`
- 本地已提交:`submission/round2/results_flux2/flux2_8_{bon,oneshot,ours}_s0/images/`
  - 只有 8-entity、seed 0、3 个任务 × 3 方法 = 9 张(论文 figure 用)
  - 6-entity 和 seed 1/2 的图只在服务器上

## 7. 生成图 — Ablation(本地只有 records)

- 服务器:`/workspace/misc/round2/results_ablation/<variant>_s{0,1}/images/hard_XXXX.png`
  - `<variant>` ∈ {`ours_full`, `ours_rawroute`, `ours_strictaccept`, `ours_promptonly`, `ours_noportfolio`, `ours_nodual`}
- 本地已提交 records:`submission/round2/results_ablation/<variant>_s{0,1}/records.jsonl`(12 个文件)
- 图片**本地没有**

## 8. Round-1 结果(本地没有)

- 服务器:`/workspace/misc/round1/results/`(整个目录 gitignored)
  - 包含 calibration、preflight、各方法试跑的 records + images
- 本地只有 round1 的**脚本**(`submission/round1/*.py` / `*.sh`),没有结果

## 9. 人评图包

- 本地:`meta/human_eval_umo_vs_oneshot.tar.gz`(含 key)
- 本地:`meta/human_eval_umo_vs_oneshot_forlabelers.tar.gz`(不含 key,给标注员)
- 解压版:`meta/human_eval_umo_vs_oneshot/`(gitignored)
- 服务器:同路径(`/workspace/misc/round2/human_eval_umo_vs_oneshot/`)

## 10. 论文定性图(已进 git)

- `submission/paper/figures/` — 9 张(3 任务 × 3 方法),论文对比图直接用这些

---

## 如何把服务器上的图拉到本地

```bash
# 例:拉 OmniGen2 主实验 merged 的所有图
ssh root@<server> -p <port> -i ~/.ssh/id_ed25519_2 \
  "rsync -av --include='*/' --include='images/*.png' --exclude='*' \
   /workspace/misc/round2/results_r2/merged/ ./local_r2_images/"
```

或者打包后 scp:
```bash
ssh root@<server> -p <port> -i ~/.ssh/id_ed25519_2 \
  "cd /workspace/misc/round2/results_r2/merged && \
   tar czf /tmp/r2_images.tar.gz */images/"
scp -P <port> -i ~/.ssh/id_ed25519_2 \
  root@<server>:/tmp/r2_images.tar.gz meta/
```

> 注意:OmniGen2 主实验 6000 张图体量较大(估计 ~3-5 GB),拉下来前确认本地有空间。
> 如果只是为了管理 / 归档,建议留在服务器上,本地只保留 records + 论文 figure 用的 9 张。
