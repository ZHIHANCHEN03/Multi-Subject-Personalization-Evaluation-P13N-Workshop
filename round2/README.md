# Round 2 — 把 marginal 变 solid（冲 AAAI）

目标：Round 1 已 GO（免训练 ours 在4主体上 ≤ 重训 UMO）。Round 2 用**规模+显著性+人评+scaling+消融**把它做硬。详见 `../PLAN.md` 第九节。

复用 Round 1 的一切：环境(`.venvs/*`)、模型(`models/*`)、**冻结校准**(`round1/results/calibration/mie_baselines.json`)、pipelines(`round1/p*.py`)。Round 2 只加分析/编排脚本，不重写引擎。

## 文件
```
run_round2_main.sh     主500任务：建manifest→复用校准→分片→每GPU一个tmux
run_shard.sh           单GPU单分片跑4方法(one_shot/best_of_n/ours_v2/umo)
split_manifest.py      manifest 切成 K 片
merge_shards.py        合并各分片 records
analyze.py             均值+95%CI + 配对bootstrap显著性(p值/CI) + 胜率   ← P0
export_human_eval.py   导出盲评A/B成对图 + ballot.csv + key.json          ← P0
aggregate_human_eval.py 汇总人评：胜率+95%CI(多人多数投票)               ← P0
run_ablation.sh        校准路由 vs raw、+参考集 vs 只prompt              ← P2
run_scaling.sh         预算B扫描 ours vs best-of-N                        ← P1
score_precomputed.py   给MOSAIC/MultiCrafter等跨系统图按task_id打SCR      ← P2
```

## 跑法（服务器上）

### P0-1 主500 + 多 seed 显著性（要 GPU，分阶段铺）

`run_shard.sh` 现支持 `SEEDS` 环境变量（逗号分隔），输出目录按 seed 命名空间隔离（`<method>_s<seed>/`），互不覆盖。AAAI P0 要求 ≥3 seeds + 95%CI + 配对显著性，所以分两阶段铺：

```bash
cd round2

# Pass 1：先全 4 片 × seed 0（~1.4 天）→ 出初步信号，决定是否继续
NGPU=4 MIE_CKPT=<ckpt> SEEDS=0 bash run_round2_main.sh
# 全部 *_DONE 后先 merge + analyze seed 0 看初步信号：
../.venvs/omni/bin/python merge_shards.py --shard_glob 'results_r2/shard_*' --out results_r2/merged --seeds 0
../.venvs/omni/bin/python analyze.py --results results_r2/merged --main ours_v2 \
    --others umo best_of_n one_shot --entities 4 2 --metric scr --seeds 0 \
    --out results_r2/analysis_scr_s0.json

# Pass 2：seed 0 信号 OK → 续 seed 1,2（再 ~2.8 天，总 ~4 天）
#   在每台服务器上重跑（同 shard 路径，只换 SEEDS；seed 0 的结果保留不重跑）：
SEEDS=1,2 bash run_shard.sh   # 在每个 tmux 里，SHARD_MANIFEST/RESULTS_DIR 不变
# 全部跑完后做 3-seed 联合分析（per-task 跨 seed 取均值再 bootstrap）：
../.venvs/omni/bin/python merge_shards.py --shard_glob 'results_r2/shard_*' --out results_r2/merged --seeds 0 1 2
../.venvs/omni/bin/python analyze.py --results results_r2/merged --main ours_v2 \
    --others umo best_of_n one_shot --entities 4 2 --metric scr --seeds 0 1 2 \
    --out results_r2/analysis_scr_3seed.json
../.venvs/omni/bin/python analyze.py --results results_r2/merged --main ours_v2 \
    --others umo best_of_n one_shot --metric dino_mean --seeds 0 1 2 \
    --out results_r2/analysis_dino_3seed.json
```
看 `analyze` 输出里 `ours_v2 vs umo` 的 `p` 和 `CI`：**CI 不跨 0 / p<0.05 = 显著**。
`--seeds 0 1 2` 时，每个 task 的 metric 先跨 3 seed 取均值（subject-level），再 bootstrap over tasks——这是多 seed 显著性的标准做法。

### P0-2 人评（GPU 只用于已生成的图，其实不用 GPU）
```bash
../.venvs/omni/bin/python export_human_eval.py --results results_r2/merged \
    --main ours_v2 --vs umo best_of_n --entities 4 --per 80 --out human_eval
# 把 human_eval/pairs/*.png + ballot.csv 发给 ≥3 个标注人，各填一份 ballot
../.venvs/omni/bin/python aggregate_human_eval.py --key human_eval/key.json \
    --ballots ann1.csv ann2.csv ann3.csv
```

### P1 scaling / P2 消融（要 GPU，小subset即可）
```bash
DATA=results_r2/manifests/r2_hard.jsonl CALIBRATION=<frozen> MIE_CKPT=<ckpt> bash run_ablation.sh
DATA=results_r2/manifests/r2_hard.jsonl CALIBRATION=<frozen> MIE_CKPT=<ckpt> bash run_scaling.sh
../.venvs/omni/bin/python analyze.py --results results_ablation --main ours_full --others ours_rawroute ours_promptonly
```

### P2 跨系统 baseline（MOSAIC 等）
1. clone 各自 repo、下 checkpoint，在 `results_r2/manifests/round2_full.jsonl` 上跑推理，
   每 task 存 `<dir>/<task_id>.png`。
2. 打分接入：
```bash
../.venvs/omni/bin/python score_precomputed.py --data results_r2/manifests/round2_full.jsonl \
    --images /path/to/mosaic_images --name mosaic --out_dir results_r2/merged
```

## 判定
- `analyze` 里 ours_v2 vs UMO **显著**（CI 排除0）+ 人评胜率 CI>50% + scaling 优势随主体数增大 → **AAAI 有力竞争**。
- 只有显著、无 scaling 亮点 → borderline。
- 不显著 → workshop(P13N)。

## 注意
- 全程免训练、MIE 只当过程裁判、最终用 SCR/DINO+人评、MIE 权重只读。
- FLUX.2 的 6/8 主体 scaling 需先确认 FLUX.2 原生多参考容量（≥6 refs）；不支持则该实验受限，只能用现有底座讲 2 vs 4 的趋势。
