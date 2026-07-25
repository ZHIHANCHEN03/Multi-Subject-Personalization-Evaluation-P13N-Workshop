# Round 3 — Ablation + FLUX.2 Scaling (冲 AAAI 主轨加分项)

Round 2 已完成 P0（500 任务 × 3 seed + 人评，15/16 显著）。Round 3 做两件加分项：
1. **消融（ablation）**：证明 v2.3 各设计选择必要（AAAI 必须有消融一节）
2. **FLUX.2 scaling（6/8 主体）**：讲"主体越多、崩越狠、ours 增益越大"的 scaling 故事

两者都不影响 P0 已坐实的核心 claim，但决定论文是"AAAI borderline"还是"AAAI 有力竞争"。

## 文件

```
run_ablation.sh            消融：6 变体 × 多 seed，3 GPU 并行
run_flux2_scaling.sh       FLUX.2 6/8 主体 scaling 主跑
probe_flux2.sh             FLUX.2 多参考容量探测（跑 scaling 前必跑）
calibrate_flux2.sh         FLUX.2 专用 MIE 校准（跑 scaling 前必跑）
select_hard_cases_6_8.py   6/8 主体 manifest 构建（disjoint from all prior splits）
score_mie_precomputed.py   给已有 baseline 图补算 MIE E/A/I + 标准化异常
round1/external_generators.py   新增 Flux2Generator adapter
```

## 执行顺序（拿到 4 服务器后）

### Step 0: FLUX.2 容量探测（5 min，单卡，先跑）
```bash
cd round2
MIE_CKPT=<ckpt> bash probe_flux2.sh
# 看 results_flux2/probe.json: 若 max_refs >= 6 -> 6/8 scaling 可行
#                          若 max_refs < 6  -> 只跑 ablation，FLUX.2 scaling 降级或跳过
```

### Step 1: 消融（8 h，3 GPU 并行）
```bash
# 用 round2 hard_4 子集（~150 任务）做消融，复用 round1 冻结校准
DATA=results_r2/manifests/r2_hard.jsonl \
CALIBRATION=<round1 frozen mie_baselines.json> \
MIE_CKPT=<ckpt> SEEDS=0,1 SHARD=0 bash run_ablation.sh   # GPU0: ours_full + ours_rawroute
# GPU1: SHARD=1  -> ours_promptonly + ours_nodual
# GPU2: SHARD=2  -> ours_noportfolio + ours_strictaccept
# GPU3 空闲 -> 可同时跑 Step 2 的 FLUX.2 校准

# 跑完分析：
PY merge_shards.py --shard_glob 'results_ablation' --out results_ablation/merged --seeds 0 1
PY analyze.py --results results_ablation/merged --main ours_full \
  --others ours_rawroute ours_promptonly ours_nodual ours_noportfolio ours_strictaccept \
  --entities 4 --metric scr --seeds 0 1
```

**消融变体**（每个去掉一个设计选择）：
| 变体 | 去掉什么 | 预期 |
|---|---|---|
| `ours_full` | （主，v2.3 weaksel） | best |
| `ours_rawroute` | 校准路由 → raw argmin | 退化为永远选同一维 |
| `ours_promptonly` | 参考集操纵杠杆 | 只改 prompt，增益减半 |
| `ours_nodual` | SCR 双信号诊断 → 轮询选 subject | 修错 subject，增益消失 |
| `ours_noportfolio` | 动作组合 → 单动作 | 搜索空间变小 |
| `ours_strictaccept` | 宽松接受 → 严格 v1 接受 | loop 几乎不 fire |

### Step 2: FLUX.2 校准（~30 min，单卡，与 Step 1 并行）
```bash
MIE_CKPT=<ckpt> bash calibrate_flux2.sh
# 产出 results_flux2/calibration/mie_baselines_flux2.json
```

### Step 3: FLUX.2 scaling 主跑（12 h，2-4 GPU 并行）
```bash
# 前提：probe.json max_refs >= 6 且 mie_baselines_flux2.json 已生成
MIE_CKPT=<ckpt> \
CALIBRATION=results_flux2/calibration/mie_baselines_flux2.json \
SEEDS=0,1,2 SHARD=0 bash run_flux2_scaling.sh   # GPU0: 6-entity
# GPU1: SHARD=1  -> 8-entity
# 若只有 2 GPU：先 SHARD=0 跑完再 SHARD=1

# 跑完分析：
PY merge_shards.py --shard_glob 'results_flux2' --out results_flux2/merged --seeds 0 1 2
PY analyze.py --results results_flux2/merged --main flux2_6_ours \
  --others flux2_6_oneshot flux2_6_bon --entities 6 --metric scr --seeds 0 1 2
PY analyze.py --results results_flux2/merged --main flux2_8_ours \
  --others flux2_8_oneshot flux2_8_bon --entities 8 --metric scr --seeds 0 1 2
```

### Step 4: 给 baseline 补算 MIE（可选，不占 GPU，~1 h）
```bash
# 给 one_shot / umo / best_of_n 的已生成图补 MIE 分，填论文表格
MIE_CKPT=<ckpt> python score_mie_precomputed.py \
  --data results_r2/manifests/round2_full.jsonl \
  --images results_r2/merged/one_shot_s0/images \
  --name one_shot_s0 \
  --calibration <round1 frozen mie_baselines.json> \
  --out_dir results_r2/merged
# 对 umo_s0, best_of_n_s0 同理
```

## 4 服务器并行时间线

| 服务器 | Step 1 (0-8h) | Step 2 (8-12h) | Step 3 (12-24h) |
|---|---|---|---|
| GPU0 | ablation SHARD=0 | FLUX.2 6-entity seed 0,1,2 | 续 |
| GPU1 | ablation SHARD=1 | FLUX.2 8-entity seed 0,1,2 | 续 |
| GPU2 | ablation SHARD=2 | FLUX.2 校准（30min）→ 6-entity seed 0,1 | 续 |
| GPU3 | FLUX.2 probe(5min) + 校准(30min) | FLUX.2 8-entity seed 0,1 | 续 |

**总 wall-clock：~24 h**（一天内全跑完）。

## 判定

| 结果 | 落点 |
|---|---|
| P0 显著 + 消融全对 + FLUX.2 scaling 增益随主体数增大 | **AAAI 有力竞争** |
| P0 显著 + 消融全对 + FLUX.2 scaling 不显著 | AAAI borderline~主轨 |
| P0 显著 + 消融部分对 | AAAI borderline |
| P0 不显著 | workshop(P13N) |

## 注意

- 全程免训练；FLUX.2 只推理不训练。
- FLUX.2 上**没有同底座重训 SOTA**，只比 ours vs one_shot/best_of_n（信号1）。
- MIE 在 FLUX.2 上需**重新校准**（score 分布不同），用 `calibrate_flux2.sh`。
- 若 `probe_flux2.sh` 显示 FLUX.2 不支持 ≥6 refs，降级为 FLUX.1-dev fallback 或只做 4 主体。
- 所有 manifest 互相 disjoint（round1 seed 0/1，round2 seed 200/201，FLUX.2 校准 seed 777/778，scaling seed 300/301）。
