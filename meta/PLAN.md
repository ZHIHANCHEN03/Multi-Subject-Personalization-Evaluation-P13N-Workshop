# Paper 2 — 计划（两轮打法，唯一真源）

> 从零重写。这份文件是唯一的行动依据。复用的唯一旧资产是 `MIBE_Core/`（MIE 评估器 + MIB-Gold + SCR）。

---

## 一、一句话在做什么

做一个**闭环、免训练**的修正器：多主体生成时，用冻结的**分维度评估器 MIE** 诊断"哪一维崩了（存在/外观/交互）"，然后**改 prompt + 操纵参考图集合**再生成，循环几轮，把身份崩塌修好。**不训练任何模型权重。**

- MIE = 闭环里的"诊断医生 + 验收员"（过程用）
- SCR(DINOv2) + 人评 = 外部"体检报告"（最终判定用，**绝不用 MIE 自评**）

## 二、要证明的 claim（已确认没人做过）

> **多主体身份崩塌无需重训**，靠"外部分维度评测器(MIE)闭环诊断 + 迭代修复"就能在语义层修好，在**同底座(OmniGen2)上以零训练成本追平重训 SOTA(UMO)**；并能 scale 到更多主体。

> ⚠️ **底座约束（关键，决定了实验怎么切）**：`ours / one-shot / best-of-N` 需要底座**原生支持多参考图主体条件生成**。**OmniGen2 原生支持但最多 5 张参考图**；纯 FLUX.1 不原生支持（MultiCrafter靠训练、FreeGraftor靠trick才行）。因此：
> - **2/4 主体 → 在 OmniGen2 上做**（有同底座重训 SOTA=UMO 可比）= **核心 claim 的地基**。
> - **6/8 主体 → 换 FLUX.2**（需先确认其多参考容量）；那里**没有同底座重训 SOTA**，只做 `ours vs one-shot/best-of-N` 的 **scaling 故事**，是**加分**不是地基。

**为什么干净（逐个切割相邻工作）：**

| 相邻工作 | 他们做的 | 我们不同 |
|---|---|---|
| MIBE / WIC（我们自己）| 只**测**这个失败 | 我们给**解法** |
| UMO / MultiCrafter | **重训**修身份 | 我们**免训练**、test-time |
| FreeCus / FreeGraftor / MuDI | 免训练多主体，但**开环、单次、内部 trick** | 我们**闭环、外部 verifier、迭代** |
| Ma et al. 2025（test-time scaling）| 闭环修正，但**噪声层 + 标量 verifier + 通用 T2I** | 我们**语义层 + 分维度 verifier + 身份崩塌** |
| SLD / TIR / PromptEnhancer | 通用组合性 / 分项>总分 | 我们专修个性化**身份+交互**崩塌 |

**性质**：这是"组合式新颖"，不是"全新机制"。所以 **claim 成不成立取决于结果**，不是取决于 idea——idea 已过关。

---

## 三、两轮打法

### 第一轮：证"有没有戏"（OmniGen2 上，2/4 主体，零人工）

**跑什么**
- **底座 OmniGen2**，两档任务：**hard=4 实体**（主比较）+ **easy=2 实体**（对照）。强交互+遮挡(`occlusion_interaction`)。
  - 为什么不做 6/8：OmniGen2 参考图上限=5，6/8 跑不了 → 留给 Round 2 的 FLUX.2。
- **5 个方法**：`ours / one-shot / best-of-N / UMO / FreeGraftor`
  - 前四个在 OmniGen2 上 = **受控对比**；**FreeGraftor 在 FLUX.1 = 跨系统参考**（可选，缺了也能出结论）。
- 另取**不重合的校准 split**，用 Qwen-based MIE 算各实体数下 E/A/I 的 median/MAD 冻结 → 标准化异常路由（避免永远选 Interaction）。
- **过程用 MIE，终评用 SCR(DINOv2)**，自动出 `DECISION.md`。

**看什么信号（主看 4 实体切片）**
1. `method_works`：ours 的 SCR < one-shot 且 < best-of-N（方法有效）
2. `noninferior_to_umo`：ours 的 SCR ≤ UMO + 容差（免训练追平重训）← **核心**
3. `beats_freegraftor`：ours < FreeGraftor（可选，跨系统参考）

**决策门（脚本自动判）**
- 信号1&2 都过 → **GO**（进 Round 2）
- 只有信号1过 → **CONDITIONAL**（方法有效但没追平UMO，调整/加样本，至少workshop）
- 信号1不过 → **STOP**（先改方法）
- FreeGraftor 缺失不致命，用信号1&2 下结论。

### 第二轮：坐实核心 + 补 scaling（冲 AAAI）

在 Round1=GO 后做加法：
- **放大 2/4 核心**：500 任务 + 多 seed + 置信区间/显著性；**加人评**（抽100-200对A/B，你 vs UMO / vs best-of-N，3人投票）。
- **scaling 曲线**：算力预算 B=2/4/6/8，横轴算力/纵轴 SCR，证 ours 帕累托压 best-of-N。
- **6/8 主体 on FLUX.2**：换到支持更多参考图的底座，做 `ours vs one-shot/best-of-N` 的"主体越多崩越狠、我增益越大"scaling 故事（该底座无同底座重训SOTA，只证信号1）。
- **消融**（附录）：校准路由 vs 总分/argmin 路由；改prompt vs +参考集操纵；换 VLM 控制器。

### 第一轮如何无缝滚进第二轮（零浪费）
- 数据：Round1 的任务是 500 的子集，task_id 稳定不变。
- 命令/脚本：一样，只把 `N_SUPPORTED/N_STRESS` 从 15-30 调到目标规模。
- 已生成结果：seed/config 不变时 Round1 的 records 直接并入。
- 校准值：Round1 冻结的 E/A/I median/MAD 直接复用。
- 环境/权重：Round1 已装好，Round2 不用重装。

---

## 四、退路阶梯（无论结果好坏都有得发）

| 结果 | 落点 | claim 措辞 |
|---|---|---|
| 最难子集救回来 **且** 整体追平重训 + 人评认 | **AAAI 主轨** | "免训练追平重训" |
| 整体没追平，**但最难子集明显反超** | **AAAI 有戏** | "免训练能救重训救不了的失败 case" |
| 最难子集只小赢，**但明显 > 开环免训练** | **好会议 / workshop** | "闭环诊断 > 开环盲修" |
| 只 > one-shot / best-of-N | **workshop (P13N)** | "test-time 修正有效" |

**第二轮效果好 = AAAI level 吗？** 基本等于，但"好"必须是**最难子集救回来**那种好；只是整体略好过 best-of-N 会掉 workshop。结果硬是必要且几乎充分条件（写作/定位另外把关）。

---

## 五、全程铁律（违反则结果不可信）

1. **不训练任何模型**——别人的(UMO/FreeGraftor)都是下载即用推理。
2. **MIE 只当过程裁判，最终成绩用 SCR(DINOv2) + 人评**——绝不自评。
3. **只在同底座 PK**——跟 UMO 比就用 OmniGen2，别拿 FLUX.2 跨底座硬比。
4. **最难子集事先定义好**——不能跑完看哪批赢了再挑（p-hacking 会被抓）。

## 六、五条撞车红线（措辞层面）

- 别 claim "免训练多主体身份保持"（被 FreeCus/FreeGraftor 占）→ 卖"闭环诊断迭代"
- 别 claim "修比挑省推理算力"（被 Ma et al. 占）→ 卖"零训练成本"
- 别 claim "分项 > 总分"（被 PromptEnhancer 占）→ 只当零件 + 引用，放附录
- 别再建 benchmark / 证"能查病"（MIBE/WIC 已做）
- 别拿"你的 FLUX.2" 跨底座硬比"人家的 FLUX.1/OmniGen2"

## 七、复用的 MIBE_Core 资产

- **MIE**（评估器）→ 闭环控制器
- **SCR / MIB-Gold**（评测）→ 独立判据 + 测试集
- 诊断(Paper 1) → 解法(本文) 的自然续作，评测标准自己掌握

---

## 七bis、当前进度（2026-07-18）— Round 1 GO + Round 1.1 winner 已冻结，Round 2 启动中

- **Round 1 完成（GO）**：4主体(n=30) SCR ours 0.50 / UMO 0.525 / best-of-N 0.55 / one-shot 0.558；2主体 one-shot 0.45 / ours_v2 0.483 / UMO 0.483。详见 `round1/REPORT.md`。
- **Round 1.1 完成（algorithm tuning，找到 winner）**：详见 `round1_1/REPORT.md`。
  - 诊断：Round 1 的 correction loop 基本没干活（`accepted_steps=0.2`），赢靠 init+MIE 挑选不是自纠正。根因：MIE 只给全局维度分不知道哪个 subject 塌，SCR per-subject 信号被浪费，接受条件太严。
  - 结构性改版（`round1/p1_ours_v2.py`）：把 SCR 从"只当裁判"提升进 loop——MIE 定维度 + SCR 定 subject → targeted refset + action portfolio + dual-signal acceptance + **按塌方 subject 的 DINO sim 选候选**（V2_SELECT_MODE=weak_subject）。
  - 消融定位关键杠杆：weak_subject 选择（matched compute 下 SCR 0.513→0.488，-4.9%）；total_tol 反而有害；strict SCR collateral 卡死 loop。
  - **winner（冻结进 Round 2）= `v2.3 weaksel`**：`V2_SELECT_MODE=weak_subject` + `V2_ACCEPT_MODE=relaxed` + `V2_TOTAL_TOL=0.0` + front_dup3 + layout，budget 8（matched to best-of-N）。
  - 20 任务 matched：ours SCR 0.488 / DINO 0.498 vs v1 0.513/0.486 vs UMO 0.563/0.432（**双指标赢 UMO**，head-to-head 8胜4负8平）。
  - 诚实定位：提升真实且 matched-compute 成立，但幅度中等（~5%）、20 任务小样本、无显著性/人评 → Round 2 要坐实。
- **Round 2 启动中**：500 任务 manifest 已建（250 hard_4 + 250 easy_2），分 4 片各 125；`round2/run_shard.sh` 已改用 winner pipeline；shard 0 在 GPU0 跑，shard 1-3 待其余 3 台服务器。
- 环境/模型/结果均在 `/workspace/misc`（持久盘）；MIE 权重只读；服务器可 Stop（勿 Terminate）。

---

## 九、Round 2 可执行清单（把 marginal 变 solid，冲 AAAI）

> 目标：Round 1 证明了方向对；Round 2 要把"免训练追平重训"从 marginal 做成**统计显著 + 人评认可 + 有 scaling 亮点**。按优先级排。

### P0 门槛（不做则只能 workshop）
1. **规模 + 显著性**
   - 500 任务（沿用 select 逻辑，task_id 稳定；Round1 的 60 是子集，可复用其 records）。
   - 每任务 ≥3 seeds；报 **mean ± 95%CI**，ours vs UMO 做**配对 bootstrap / 符号检验**，目标 p<0.05 或 CI 不跨 0。
   - 交付物：主表（SCR/DINO，含 CI）+ 显著性标注。
2. **人评（A/B）**
   - 抽 100–200 对：ours vs UMO、ours vs best-of-N。每对 3 人、强制二选一、随机左右。
   - 报胜率 + CI 下界；目标 CI 下界 > 50%。
   - 交付物：`round2/human_eval/` 导出的成对图 + 打分表模板 + 汇总脚本。

### P1 亮点（决定能否上主会）
3. **6/8 主体 scaling on FLUX.2**
   - 先确认 FLUX.2 原生多参考容量（能否 ≥6 refs）；能则跑 2/4/6/8 的 ours vs one-shot/best-of-N。
   - 讲"**主体越多、崩越狠、ours 增益越大**"的曲线（横轴主体数，纵轴 SCR 降幅）。
   - 该底座无同底座重训 SOTA → 只证信号1（ours>baselines），当 scaling 故事。
4. **算力 scaling 曲线**：预算 B=2/4/6/8，ours vs best-of-N，横轴算力/纵轴 SCR（口径=零训练成本，不吹省推理算力）。

### P2 可信度（补强，不决定生死）
5. **消融**（`round1` 已有开关，Round2 系统跑）：
   - 路由：校准路由 vs 总分/argmin（证明校准必要）
   - 动作：+参考集操纵 vs 只改 prompt
   - 控制器：MIE vs 分维度 VLM（证不只对 MIE 有效）
   - 触发门槛/多提案的消融（OURS_DEFICIT_MIN / OURS_PROPOSALS）
6. **更全 baseline 套件**：
   - 同底座因果：**UMO**（唯一能证"追平重训"）
   - 跨系统参照（下载→按 task_id 出图→用我们 SCR 打分，不训练不改）：**MOSAIC**(FLUX.1，主打4+主体)、**MultiCrafter**(FLUX.1)、**FreeGraftor**(FLUX.1 开环)、有余力加 XVerse/PSR
   - 均明确标注"跨系统、底座不同、仅参照"。

### 判定
- P0 全做且显著 + 人评认 + P1 的 FLUX.2 scaling 优势变大 → **AAAI 有力竞争**。
- 只有 P0 显著、无 scaling 亮点 → borderline。
- P0 做不出显著/人评 → **workshop(P13N)**。

### 复用 Round 1（零浪费）
- 数据/校准/records/环境/模型全在 `/workspace/misc`；Round2 = 调 `N_SUPPORTED/N_STRESS`、加 seed、加检验与人评脚本、接 FLUX.2 与跨系统 baseline。
- 待写脚本（下一步，纯代码不占 GPU）：`round2/` 下 显著性检验、人评导出、scaling 扫描、MOSAIC/MultiCrafter 接入骨架。

> MIE 权重固定在 `/workspace/Model_Training_runs/v2/unsloth_Qwen3.5-4B/20260503_045230/outputs/unsloth_Qwen3.5-4B-lora_layer-best`（只读，绝不删）。
> 服务器：RunPod pod `pwvgfql1co3zv5`，SSH `ssh root@216.81.151.3 -p 19490 -i ~/.ssh/id_ed25519_2`（直连，任务放 tmux）。

---

## 十、Round 2 服务器需求与分工（2026-07-18）

### 已就绪
- 500 任务 manifest：`/workspace/misc/round2/results_r2/manifests/round2_full.jsonl`（250 hard_4 + 250 easy_2）。
- 4 片 × 125 任务：`/workspace/misc/round2/results_r2/shards/shard_{0..3}.jsonl`。
- winner pipeline 已接入 `round2/run_shard.sh`（用 `p1_ours_v2.py` + v2.3 weaksel env）。
- 校准复用 Round 1 冻结值；MIE 权重只读。

### 磁盘
- **不需要更大 disk**。`/workspace` 是 RunPod 网络卷，258T 富余。Round 2 全量结果预估 < 10G。
- **关键**：新服务器若挂在**同一个网络卷**（同 region 同 volume），则模型（98G：OmniGen2/FLUX.1-dev/SAM/Grounding-DINO/UMO）和 venvs 直接共享，新 pod 无需重下、即开即跑。开新 pod 时务必选同一个 `/workspace` 卷。

### GPU 需求：4 张 A100（已有 1，再要 3）
每片 125 任务 × 4 方法（ours/one_shot/best_of_n/UMO），单卡 ~4 min/任务 → 每片 ~33h。4 卡并行 ~1.4 天跑完主跑。

| 服务器 | 跑什么 | 预估耗时 |
|---|---|---|
| **GPU0（当前 pod，已有）** | shard 0（125 任务 × 4 方法） | ~33h |
| **GPU1（待开）** | shard 1 | ~33h |
| **GPU2（待开）** | shard 2 | ~33h |
| **GPU3（待开）** | shard 3 | ~33h |

主跑完成后，同一批卡接着跑：
- **P1 scaling**（FLUX.2 6/8 主体）：1 卡 ~12h
- **P2 跨系 baseline**（MOSAIC/MultiCrafter/FreeGraftor，下载即推理）：1 卡 ~10h
- **P2 消融**（v2.3 各开关）：1 卡 ~8h

### 给我新服务器的方式
每台新 pod 起来后，给我 SSH 地址（host + port + key 路径），我远程执行：
1. 确认挂到同一个 `/workspace`（模型/venv 已在）。
2. `cd /workspace/misc/round2 && CUDA_VISIBLE_DEVICES=0 SHARD_MANIFEST=results_r2/shards/shard_<N>.jsonl RESULTS_DIR=results_r2/shard_<N> CALIBRATION=... MIE_CKPT=... bash run_shard.sh`（在 tmux 里）。
3. 4 片全跑完后我 merge + analyze + 出显著性表 + 导人评。

### 时间线（4 卡到位后）
- Day 1-2：主跑 500 任务（4 卡并行）
- Day 3：merge + analyze + 显著性 + 人评导出
- Day 4-5：人评回收 + scaling(FLUX.2) + 跨系 baseline + 消融
- Day 6-8：写论文
- Day 9-10：buffer / 补实验

> **现在就缺 3 台 A100 的 SSH 地址**。给我地址我立刻在每台启动一个 shard，~1.4 天后主跑完成。
