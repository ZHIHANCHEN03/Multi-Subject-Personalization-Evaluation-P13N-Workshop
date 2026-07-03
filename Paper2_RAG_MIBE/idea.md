# Paper 2：MIE-Guided 训练-免费的多主体生成自纠错范式

> 一句话：**用冻结的结构化验证器 MIE，在推理期充当冻结生成器 FLUX.2 的控制器，
> 迭代修正多主体生成的组合性失败——全程免训练，且设计上保证不劣于 baseline。**
>
> 对标：Self-RAG（LLM）。区别见下方"为什么是范式"。

---

## 0. FLUX.2 提供的免训练杠杆（对照官方 repo + diffusers 0.36 源码）

来源：[black-forest-labs/flux2](https://github.com/black-forest-labs/flux2)（FLUX.2 官方推理仓库）。

**模型档位（决定 test-time scaling 的成本）**：

| 档位 | 参数/特性 | 我们怎么用 |
|---|---|---|
| **FLUX.2 [klein] 4B / 9B** | step-distilled(4步) + guidance-distilled，**亚秒级**，消费级 GPU(4B≈8GB) | 主力：便宜到能大量重生成 ⇒ **让免训练 test-time scaling 可行** |
| **FLUX.2 [dev] 32B** | guidance-distilled(50步)，H100 级，官方称**显著受益于 prompt upsampling** | 上限对照：验证结论在高端模型上也成立 |

三档模型**全部支持** Text-to-Image + 单参考编辑 + **多参考编辑**——多主体正是我们的战场。

**免训练杠杆（动作空间的物理基础）**：

| 杠杆 | 官方/源码事实 | 能用来做什么 |
|---|---|---|
| **多参考条件** | 头牌能力"multi-reference editing"；diffusers 里 `image=[ref1,ref2,...]`，每张 ref 经 `prepare_image_latents`、靠 T 坐标(10,20,…)分隔 | 参考图的**数量/顺序可控** → 可"强化某个主体的参考"(P2) |
| **prompt upsampling（官方一等公民）** | 官方脚本用 **Mistral-Small-3.2-24B**（即其文本编码器本体）本地改写，或走 OpenRouter API；README 明言 dev "significantly benefits"。diffusers 对应 `caption_upsample_temperature`→`upsample_prompt()` | 既是**官方认证的强 baseline**，也证明"改 prompt 能提质"是官方主推机制(P1) |
| **guidance / latents / steps** | guidance-distilled ⇒ `guidance_scale` 是真旋钮(默认4.0)；可传入 `latents`；步数可控 | 生成强度、初始噪声、步数皆可控(P3) |
| **无 mask/inpaint** | `__call__` 无 `mask_image`，永远从纯噪声跑满全程 | ⇒ 像素级局部编辑不可行，**只能走 prompt/参考/旋钮层** |

**关键约束**：FLUX.2 不支持局部重绘。所以范式的"动作"只能作用在**输入侧**（prompt + 参考集 + 旋钮），不能作用在**像素侧**。这不是缺点——反而逼出一个更干净、更通用的抽象。

**为什么这对"免训练范式"是天作之合**：klein 亚秒级 ⇒ 多次重生成便宜 ⇒ test-time scaling 有意义；prompt upsampling 是官方主推 ⇒ 我们的 P1 动作站在官方机制之上，而非另起炉灶；多参考编辑是头牌 ⇒ P2 动作直接调用其核心能力。整套流程**不训练任何权重**，只在 FLUX.2 已有的三个输入接口上，用 MIE 做闭环控制。

---

## 1. 候选范式（brainstorm）

| # | 范式 | 用的 FLUX.2 杠杆 | 优点 | 风险 | 取舍 |
|---|---|---|---|---|---|
| P1 | **Prompt 自纠错**：MIE 诊断→改写 prompt→重生成 | caption 改写 | 有官方机制背书；通用 | prompt 空间较拥挤（self-refine 类） | 作**主动作** |
| P2 | **参考集重配**：MIE 说某主体不像→把该 ref 提前/复制/去干扰 | 多参考条件 | FLUX.2 独有、最贴多主体、更新颖 | 效果需实测 | 作**FLUX.2-native 动作**，进 action 消融 |
| P3 | **旋钮控制**：MIE 弱→调 guidance / seed / 步数 | guidance/latents | 极便宜 | 太"薄"，难成范式主线 | 作**辅助动作**，不单独立题 |
| P4 | **结构化验证器引导的 test-time 搜索**：Best-of-N(标量) 泛化成"分维度引导的搜索" | 全部 | 最"范式"、最 timely（test-time scaling 正热） | 需算力曲线撑 | 作**顶层框架叙事** |

**收敛逻辑**：P4 是**外壳**（把问题定义成"结构化验证器引导的推理期扩展"），P1 是**默认动作**（安全、有背书），P2 是**FLUX.2-native 动作**（新颖性 + 证明 action 接口可插拔），P3 打辅助。四个不是并列，是**一个框架 + 一组 typed 动作**。

---

## 2. 最终范式：MISC（MIE-guided Inference-time Self-Correction）

> 结构化验证器 MIE 作为冻结 FLUX.2 的**控制器**，在推理期决定 ①何时停 ②修哪个维度 ③用哪个动作，迭代修正，单调不退。

```
输入: 参考集 R = {r_1..r_N}, prompt p0
      │
【初始化】Best-of-N：p0 配不同 seed 生成 N 张 → MIE 打分 → 选总分最高 (y0, state0)
      │
┌── 循环 t = 1..K ────────────────────────────────────────────────┐
│ 1. 诊断:  MIE(R, p_t, y_t) → 总分 s_t + typed 三维 (d_E,d_A,d_I) │
│           + weak_subject（哪个主体最可能是问题所在）             │
│ 2. 停止:  s_t ≥ τ  或  t = K  → 退出                             │
│ 3. 路由:  k* = argmin_d d_k  （typed 控制信号；静态先验只做 tie-break）│
│ 4. 动作:  a_t = ACTION[k*](state_t, weak_subject)  （见下方动作表）│
│ 5. 重生成: y_{t+1} = FLUX2.generate(p', R')                      │
│ 6. 复评:  MIE(y_{t+1}) → s_{t+1}                                  │
│ 7. 接受:  s_{t+1} > s_t + ε  且  无任一维度跌破 δ  → 采用；        │
│           否则回滚（保留 state_t），本步仍计入预算                 │
└─────────────────────────────────────────────────────────────────┘
      │
输出: 约束最优 —— 三维都过门槛 θ_k 的步里取总分最高；否则退化取总分最高
```

### Typed 动作表（E/A/I → 动作，这就是"图像版 reflection token"）

| 弱维度 k* | 默认动作（prompt，P1） | FLUX.2-native 变体（参考集，P2） |
|---|---|---|
| **existence**（主体缺失） | 把该主体分句挪到最前 + 加 "clearly include {S}" | 把 {S} 的参考图提到参考序列最前 |
| **appearance**（不像） | 分句末尾加身份强调 "(matching reference identity)" | 复制/前移 {S} 的参考图，强化其条件权重 |
| **interaction**（交互错） | 末尾追加显式空间描述（"facing each other"…） | —（交互是关系，参考集层面无直接动作，回落 prompt） |

- **动作是确定性的**（默认零额外模型调用，可复现、可解释）；
- **通用性实验**：另跑一版用小 LLM 做改写，证明范式不绑定手写规则（回答"只对模板 prompt 有效"）。

### 端到端落地映射（证明"整个流程能跑起来"，全程免训练）

| 步骤 | 具体调用 | 是否训练 |
|---|---|---|
| 生成器 G | FLUX.2 [klein] 4B/9B（主力，亚秒级）或 [dev] 32B（上限），`Flux2Pipeline(image=R, prompt=p, guidance_scale, generator=seed)` | 冻结 |
| 验证器 C | MIE `mie_checkpoint`：`MIE(refs=R, prompt=p, image=y) → {s, d_E, d_A, d_I, weak_subject}` | 冻结 |
| 初始化 | 同 prompt × N 个 seed 调 G，逐张过 MIE，取总分最高 | — |
| 动作·prompt(P1) | 确定性字符串重写；或调用 FLUX.2 官方 upsampler（Mistral-Small-3.2-24B / OpenRouter）做**受控**改写 | 免训练 |
| 动作·参考集(P2) | 重排/复制 `R` 里对应 `weak_subject` 的参考图，再喂回 G 的 `image=` | 免训练 |
| 接受/回滚 | 纯逻辑：比较相邻 MIE 分数，维护"全程最优状态" | — |

> 整条链路只有两个模型（FLUX.2、MIE）且**都不更新权重**；新增部分全是无参数的控制逻辑 + 官方已提供的 upsampler。这就是"train-free 范式"的字面含义：**贡献在协议，不在权重**。

---

## 3. 为什么是"范式"（对标 Self-RAG）

Self-RAG 的范式内核 = **把自我批判变成 typed 离散控制信号（reflection tokens），控制生成轨迹**。MISC 是它的**免训练、外部验证器版本**：

| | Self-RAG | MISC（本文） |
|---|---|---|
| 控制信号 | reflection tokens（typed） | MIE 的 E/A/I 三维（typed） |
| 信号来源 | **训练进模型** | **冻结的外部验证器 MIE（免训练）** |
| 控制什么 | 检索 + 生成 | 停止 / 路由 / 动作 |
| 灾难性遗忘 | 有（改了模型） | 无（不动 FLUX.2/MIE） |

**诚实定位**（防审稿人攻击）：不说"图像版 Self-RAG"。说——
> "Self-RAG 证明 typed 自我批判很强，但需**训进模型**。我们证明：只要有一个足够强的**外部结构化验证器**，同一范式能在**推理期免训练**实现。"

**通用抽象**（跟 FLUX.2/MIE 解耦，这是范式和方法的分界线）：
> 给定任意生成器 G + 任意把质量拆成 k 个可解释维度的验证器 C，如何用 C 的 typed 信号在推理期控制 G 的自我修正。FLUX.2+MIE 只是一个实例。

**三条理论性质（写成 Property，零实验成本）**：
1. **单调不退**：accept 规则要求严格提升，被拒回滚 ⇒ 采纳轨迹的总分单调不降。
2. **优雅退化**：K=0 时严格退化为 Best-of-N + G，是纯结构性质。
3. **有界漂移**：动作只做"增补/重排"不删原内容 ⇒ 相对原始 prompt 的语义漂移有上界（配合语义护栏，见 §5）。

---

## 4. 为什么"保证有用"（这是设计出来的，不是赌出来的）

- **内部指标必不劣**：由性质 1+2，MISC 在 MIE 分数上**理论保证 ≥ Best-of-N ≥ 一次性生成**（随时可拒绝所有修改退回起点）。这是**可证明的**，不依赖实验。
- **独立指标才是真考验**：MIE 分涨了不代表真变好（可能 MIE 有偏、或语义跑偏）。所以"保证有用"= **可证明的内部不劣** + **eval 用独立指标确认**（见 §5）。两条都以 MIBE 为核心资产。

---

## 5. Eval 设计（如何证明"生成得更好"，可预注册）

**评测集**：复用 MIB-Gold，固定子集，task_id 清单写死，不换。dev-mini(~200) 先跑方向，方向对再上全量测试集。

**生成**：每组共用同一批 seed（`resampled` 组除外）；裁判用 GPT-4o（**跟 MIE 换家族**，避免自证循环）。

### 承重实验（全篇核心，只有这一个必须赢）

> **结构化反馈 vs 标量反馈，算力对齐。**
> MISC（用 E/A/I 分维度路由）**vs** Best-of-N（同一个 MIE，但只用**标量总分**，生成次数拉到与 MISC 相同）。
> 赢 ⇒ "decomposition matters" 立住 = 整个范式立住（正好对应 Self-RAG 的 typed vs scalar）。
> 产出一张 **test-time scaling 曲线**（x=算力/生成次数，两条线，结构化在上）。

### 判定线（协议冻结前定死，事后不许改）

| 对比 | 判定线 |
|---|---|
| MISC vs 一次性生成 | 配对胜率 CI 下界 **>50%** 且点估计 **≥60%** |
| **结构化 vs 标量（承重）** | 配对胜率 CI 下界 **>50%**（算力对齐） |
| 诊断路由 vs 随机 / 静态路由 | CI 下界 **>50%**（核心因果 claim） |
| 针对性改写 vs FLUX.2 原生 caption_upsample | CI 下界 **>50%**，点估计 **≥55%** |
| 画质护栏（FID/aesthetic） | 非劣：MISC 不显著劣于 baseline |
| 语义护栏（CLIP(原始p0, 最终图)） | 非劣：不显著下降（防"分涨了但跑偏"） |
| collateral damage rate | 软指标 <30%（工程健康度，非 pass/fail） |

**样本量**：每组 ~150–200 对，50% 附近 95%CI 半宽 ≈±7pt，能分辨 ≥10pt 的真实差异。

### 指标分层
- **内部（仅用于路由/曲线，不作最终判据）**：MIE 三维 + 总分 + collateral damage rate
- **独立（最终判据）**：VQAScore(组合性) + CLIP-I/DINO(主体保真) + GPT-4o A/B + 小规模人审锚点

### 配套消融 / 通用性
- routing：diagnostic / random / static
- action：prompt-only / +参考集重配(P2) / LLM-改写（证明不绑定手写规则）
- seed_mode：fixed / resampled
- 换验证器：`vlm_judge` 替 MIE，证明范式不依赖具体 C
- scaling：K∈{1,2,3,5}，N∈{2,4,6,8}
- 按 MIB 的 N（主体数）分层看增益（多主体是否增益更大 → 边界条件研究，兜底叙事）

---

## 6. 复用 MIBE 资产（不随实验结果变化）

- **MIB-Gold** 全程当评测集，不建新数据。
- **MIE** 全程当唯一结构化验证器/控制器；正式结论只用 `mie_checkpoint`（Paper1 已验证 pairwise acc 0.922 / macro-F1 0.818，反过来证明信任 MIE 是有依据的 → 两篇论文互证）。`vlm_judge`/`mock` 仅作 fallback/通用性/调试。
- 即使主结果偏弱，仍有两条兜底卖点：**"哪些 N/维度上有效"的边界条件研究** + **"达到同质量所需算力更少"的效率主张**——都仍以 MIBE 为核心。

---

## 7. 风险与兜底

| 风险 | 兜底 |
|---|---|
| MIE 自证循环 | 最终判据全用独立指标 + 换家族裁判 |
| 改写带偏整体语义 | 语义护栏（CLIP vs 原始 p0）+ 有界漂移性质 + 动作只增补不删除 |
| 被当成"self-refine 的 T2I 变体" | 承重实验证明"结构化 > 标量"；对标 Self-RAG 的 typed 信号内核 |
| 手写规则只对模板 prompt 有效 | LLM-改写通用性实验 + action 抽象接口 |
| 参考集重配(P2) 效果未知 | 作为 action 变体进消融，不绑进主 claim；主动作是有背书的 prompt 改写 |
| 越改越坏 | accept 要求严格提升 + per-dim 不跌破 δ + 输出取全程最优 |

---

## 8. 相关工作（定位，不硬碰）

- **Self-RAG**：typed 自我批判范式，但需训练 → 我们免训练、外部验证器版。
- **Self-Refine / Reflexion**：迭代自我改进，但用标量/自由文本反馈 → 我们用**结构化 typed** 反馈，且有承重实验证明结构化的增量。
- **MIRA / PhotoAgent / Agentic Retoucher / EditRefiner**（见旧笔记）：多为 SFT+GRPO 训练或像素编辑 agent → 我们训练-免费、输入侧控制，定位为**互补的低成本方案**，引用其数字做参照，不强行复现。
