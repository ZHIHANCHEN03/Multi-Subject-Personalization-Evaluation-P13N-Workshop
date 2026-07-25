# Round 1 — 有没有戏（便宜、快、零人工）

目标：在 ~60 个**最难**的强交互多主体 case 上跑 5 个 pipeline，用 **SCR(DINOv2)** 自动打分，看三个信号决定要不要进 Round 2。详见 `../PLAN.md`。

## 5 个 pipeline

| 文件 | 方法 | 底座 | 训练? | MIE? |
|---|---|---|---|---|
| `p1_ours.py` | 闭环 MIE 引导修正（我们的）| OmniGen2 | 免训练 | ✅ 过程用 |
| `p2_oneshot.py` | 裸跑一次 | OmniGen2 | 免训练 | ❌ |
| `p3_bestofn.py` | best-of-N（MIE 总分挑）| OmniGen2 | 免训练 | ✅ 挑选用 |
| `p4_umo.py` | UMO（重训基线）| OmniGen2 | 他们训好 | ❌ |
| `p5_freegraftor.py` | FreeGraftor（开环免训练）| FLUX.1-dev | 免训练 | ❌ |

- **MIE** = Qwen3.5/Unsloth 多模态骨干 + preference head + E/A/I head。真实运行只需 `MIE_CKPT`；代码自动读取 checkpoint 内配置并加载。MIE 使用独立持久子进程，避免与 OmniGen2 的依赖冲突。
- **SCR(DINOv2)** = Grounding-DINO 定位每个主体后，用 DINOv2 比较 ref↔crop；独立于 MIE。
- 5 个脚本全部是**直接生成 + 自动 SCR**，没有预生成图片占位或 `NotImplementedError`。

## 怎么跑

**0. 环境与权重（GPU 机器，全部放在本 repo 内）**
```bash
export HF_TOKEN=...   # FLUX.1-dev 是 gated model，首次需要
bash setup_round1.sh
```

该脚本会准备：
- `external/UMO` + 官方 OmniGen2 submodule；
- `external/FreeGraftor`；
- `.venvs/omni` + `.venvs/freegraftor` 两套隔离环境；
- `models/OmniGen2`、`models/UMO`、`models/FLUX.1-dev`、Grounding-DINO、SAM。

**1. 一键跑全部 5 个 pipeline**
```bash
# checkpoint 已默认设为服务器上的最新 4B lora_layer-best；
# 如服务器路径变化才需要手工覆盖 MIE_CKPT。
export HF_TOKEN=新生成且未泄露的token
bash run_round1.sh
```
若依赖环境不存在，`run_round1.sh` 会先自动调用 `setup_round1.sh`。

严禁把 HF token 写进文件或提交 Git。即使仓库是 private，token 仍会进入
Git 历史、CI 日志和所有 clone。脚本只读取当前 shell 的 `HF_TOKEN`。

**2. 笔记本只验证 p1/p2/p3 控制逻辑（无 GPU/权重）**
```bash
GEN=mock bash run_round1.sh
```

## 看什么（三信号，`compare_round1.py` 自动打印）
1. ours > best-of-N / one-shot？（方法有效）
2. ours ≥ UMO？（免训练追平重训）
3. ours > FreeGraftor？（闭环打过开环）

脚本最后自动写出：
- `results/evaluation.json`：所有切片、方法统计和预注册信号；
- `results/DECISION.md`：最终 `GO / CONDITIONAL / STOP / INCOMPLETE`；
- `round1_<UTC时间>.log`：完整安装/下载/运行/评估日志；
- `round1_latest.log`：指向最近一次运行；
- `results/round1.log`：同一最新日志的链接。

判定规则：
- `GO`：方法有效、对 UMO 非劣、优于 FreeGraftor，三个信号全过；
- `CONDITIONAL`：核心方法有效，但两个外部信号只过一个；
- `STOP`：核心方法无效，或两个外部信号都不过；
- `INCOMPLETE`：任何方法缺失有效输出。

## 关键参数
- `N_SUPPORTED=30`：4 实体强交互+遮挡，作为五方法较公平的主切片
- `N_STRESS=30`：6 实体（n>4），作为能力边界压力测试
- `N_CAL_SUPPORTED=30 / N_CAL_STRESS=30`：与正式60条完全不重合的校准集
- `B`：对齐算力预算（best-of-N=8；ours=n_init4+k4，共 8）
- `SCR_THRESH`：DINO 相似度低于此判为身份崩塌（默认 0.5，用 dev 数据标定）

为什么分两片：UMO 官方案例只验证少量 refs，FreeGraftor UI 上限为 5。若只测
6/8 实体，baseline 失败可能来自超出支持范围，不能干净支撑方法 claim。因此主表
用 4 实体，n>4 单列为 stress test，并显式报告 generation failure rate。

注意：FreeGraftor 原生底座是 FLUX.1-dev，不是 OmniGen2。因此它是**跨系统 SOTA
参考**，不能单独用于因果证明“闭环优于开环”；受控证据主要来自同 OmniGen2 的
one-shot / best-of-N / ours / UMO。

## 文件
```
select_hard_cases.py  从 60k 筛难例 → hard_cases.jsonl（解析 ref 图路径）
common.py             Task/加载 + MIE critic(可插拔+mock) + SCR(DINOv2) + 驱动循环
mie_server.py         真实 Qwen MIE checkpoint 加载 + 持久 JSONL 评分服务
calibrate_mie.py      按4/6实体计算 E/A/I 各自 median/MAD 基线
generators.py         官方 OmniGen2 wrapper + mock
external_generators.py 官方 UMO + FreeGraftor 直接推理适配器
actions.py            prompt 重写 + reference-set 操纵（闭环修正动作）
p1..p5                5 个 pipeline
compare_round1.py     汇总 + 三信号
setup_round1.sh       repo-local 环境/源码/权重一键准备
run_round1.sh         一键跑
```

## 结果目录（唯一出口）
```
results/
  manifests/                 固定的 30+30 Round-1 任务清单
  calibration/               独立校准图、MIE原始分、冻结的维度基线
  one_shot/images/           各方法生成图
  best_of_n/images/
  ours/images/
  umo/images/
  freegraftor/images/
  */records.jsonl            每任务 SCR、DINO、失败状态与轨迹
  evaluation.json            机器可读汇总和信号
  DECISION.md                是否进入 Round 2
  preflight.json             GPU、磁盘、数据和 MIE checkpoint 完整性
  round1.log                 指向 misc 中最新完整日志
```

默认日志目录为 ``，不存在时自动创建。可通过
`ROUND1_LOG_DIR=/other/path` 覆盖。日志同时实时显示在终端，并在结尾记录
UTC 完成时间、退出码和完整日志路径。

## 注意
- 外部 pipeline 单任务失败会被记录为 `generation_failed=true, SCR=1`，不会静默跳过。
- Round 1 的 `SCR_THRESH` 必须在独立 dev 数据上预先校准并冻结。
- 路由不比较 E/A/I 原始值；使用 `(该维median - 当前分)/MAD-scale`。因此 Interaction 天然偏低不会导致它永远被选择。
- 候选只有在 preference 总分提高、目标维提高且其他维没有明显下降时才接受。
