# MIBE rebuttal — 17 条 review 诉求的分类与行动计划

评分：**ws3J 6/Strong Accept (conf 3) · swx9 3/Borderline reject (conf 5) · yNv5 3/Borderline reject (conf 3)**

## 先说结论

新的 PP1 人评数据打中的是 **yNv5**，而决定这篇能不能翻盘的是 **swx9**。

| 审稿人 | 新数据的帮助 | 判断 |
|---|---|---|
| **yNv5** 3 (conf 3) | W2、W3 被直接正面回答，W1 未答 | **帮助大**，3→4 有实际可能 |
| **ws3J** 6 (conf 3) | Q1/Q2 都答了 | 帮助小，他已经给 6，只是巩固 |
| **swx9** 3 (**conf 5**) | 8 条 weakness 里只碰到 1 条 | **帮助很小** |

swx9 给 Quality=2 并不是认为方法错了 —— 他 Significance 给了 4。他是被**一串「你没做的对比」**压下去的：数据分布表、Hungarian ID 对比、跨 protocol 分析。这些新人评一条都解决不了，但其中一半是能做的，而且目前一条都没动。

**如果只交新人评数据，最可能的结果是 6/4/3 —— 不够。接下来的时间主要应该花在 swx9 身上。**

---

## A 组 — 新 PP1 数据已经答了（4 条，直接写进 rebuttal）

| # | 来源 | 诉求 | 答案 |
|---|---|---|---|
| A1 | **yNv5 W3** | 训练配对单一（60K 全是 nano_banana vs MOSAIC），seen 0.982 → unseen 0.884，泛化证据不足 | pp2 = Flux.2-klein-9B vs GPT-Image-1.5，**两个都不在训练集，且属不同模型家族**（步数蒸馏流模型 vs 闭源 API，对比训练用的 Flux.1+LoRA vs Gemini）。AUC **0.815** [0.755, 0.868]；8 主体时对齐**最高**（92.4%）；margin 单调校准 82.5 → 98.5 → 100% |
| A2 | **yNv5 W2** + **swx9 W3 / Q6** | 参考图是 GPT-Image 生成的，身份可能不自洽，给 Appearance 注入难以界定的噪声 | 同一批 216 prompt 跑三种参考源。**A = 真实授权照片 AUC 0.827** [0.706, 0.921]，B = GPT-Image 0.851，C = Qwen-Image 0.775，**三条 CI 全部离开 0.5，且 A ≈ B** |
| A3 | **ws3J Q2** | MIB 有哪些类别？MIE 能推广到未见类别？ | PP1 的 **22 个 subject 与 MIB 的 80 个零重叠**（slug 比对验证，非假设）。因此 A1/A2 的数字同时是 unseen-generator **和** unseen-subject |
| A4 | **ws3J Q1 + W1** | 各模型得分？在 existence/appearance/interaction 哪个短板？ | 完整 12 格 scorecard（3 参考源 × 4 生成器，2586/2592 cell）。**生成器排序在三种参考源下完全一致**；**interaction 是每个生成器的最弱维度**（0.07–0.64，对比 existence 0.27–0.93），且随主体数衰减最快（total 从 +0.763 掉到 −0.319） |

**A1 额外可用的论证**：补全数据后 pp2 **更加一边倒**（常数基线 89.8% → 92.3%），题目变难，AUC 却从 0.765 升到 0.815。拟合假象在样本增大、失衡加剧时的典型表现是往 0.5 塌，这里是反的。

---

## B 组 — 零实验，改文字就能答（8 条，性价比最高，应最先做完）

| # | 来源 | 做什么 |
|---|---|---|
| B1 | swx9 W1 | benchmark 的 motivation 从附录搬进 introduction |
| B2 | swx9 W7 | 补引 WithAnyone [arXiv 2510.14975] 并写清与本工作的区别 |
| B3 | swx9 Q2 | 贴出输给 MIE 的完整 prompt 模板 |
| B4 | swx9 W5 / Q3 | ranking head 与 diagnosis head 的结构、接在模型哪一层，画出来 |
| B5 | swx9 Paper Formatting | 修 3 条参考文献的作者名（MultiHuman-Testbench、XVerse、T2I-CompBench —— 他逐条列了出来） |
| B6 | swx9 Limitations | 补 societal impact 段落 |
| B7 | ws3J W2 | 从附录挪 3–4 个代表性 prompt 到正文 |
| B8 | ws3J Q3 | 承诺维护 leaderboard（他主动提议的，答应即可） |

> **B5 尤其重要。** conf-5 的审稿人逐条列出了错误的作者名。不修等于告诉他「我们没认真读你的 review」。这 8 条全部是零成本，应当在第一天做完。

---

## C 组 — 要做实验，但做得了（4 条，真正的战场）

### C1 · 数据分布统计 — swx9 W2 / Q5
**最便宜、最该立刻做。** 人物按年龄 / 族裔 / 性别 / 发型分布，物体按类别分布。纯描述统计，CPU 几小时，产出 2 张表 + 2 张图。

> 他在 Weakness 和 Question 里各问了一次，说明在意。不做等于直接送他一条不改分的理由。

### C2 · Hungarian ID similarity 对比 — swx9 W4 / Q4
**swx9 最硬的一条**，他点名了 MultiHuman-Testbench [arXiv 2506.20879] 的具体指标。

执行：人脸检测 + ArcFace 嵌入 → 生成图人脸与参考身份做匈牙利匹配 → 未匹配上的重罚 → 在我们的 split 上，与 MIE 并排比较各自对人类偏好的 AUC。

- 成本：GPU，实现 + 跑 2586 张图，估 1–2 天
- 局限：只对 human subject 有效（动物 / 物体没有 ArcFace），**要主动说明**
- **收益最高**：这条不答，swx9 几乎不可能加分

### C3 · 同族裔身份混淆 — swx9 W6 / Q1
他问的是「prompt 里既然有 "middle eastern man" 这类标识，benchmark 测不测同框两个中东男性被混淆」。

**前置条件：先查数据里到底有没有同族裔同框的 prompt。**
- **有** → 切出「同族裔同框」vs「跨族裔同框」两个子集，比较 appearance / existence，便宜的切片分析
- **没有** → 诚实答「当前设计未隔离这一变量」，列为 future work

> 查询 10 分钟即可完成。**查完再决定，不要提前承诺。**

### C4 · 非 teacher 派生的 baseline — yNv5 W1
他说 MIE 拿自己的 Gemini teacher 当 baseline 不公平。破法不是辩解，是**换锚点**：在 pp2 的**人类标注**上，把 MIE 和 Gemini teacher 各自对人类偏好的 AUC 并排报出来。人评是外部锚，不是 teacher 派生的，这个比较就公平了。

- pp2 人评**已经有了**，只差把 teacher 在同一批 pair 上跑一遍
- 成本：API 调用 648 对，几小时
- 这是 yNv5 三条 weakness 里唯一没答的一条，答掉就是 0/3 → 3/3

---

## D 组 — rebuttal 期做不完，只能收窄或承认（2 条）

| # | 来源 | 为什么做不了 | 怎么答 |
|---|---|---|---|
| D1 | swx9 W8 | 跨 protocol 分析（「XVerseBench 把 flux2 评为 A/B/C，MIBE 评为 D/E/F，D/E/F 告诉了我 A/B/C 没告诉我的什么」）需要完整跑通另一个 benchmark 的流水线 | **不要硬扛，改成 scoped 版本**：拿 A4 的 scorecard，挑一个 XVerseBench 给高分而 MIE 判 interaction 差的具体 case 做**定性对照** + 少量样本的定量。给方向而非全量，并明确承诺 camera-ready 补全 |
| D2 | ws3J W3 | 「更冷门、长尾类别的身份保持」—— 22 个 subject 不构成长尾研究 | **承认**。可以说 flamingo / alpaca / desk_globe 已开始触及更长的尾巴，但**不要称之为长尾研究**。ws3J 已给 6 且 conf 3，这条不扣分，硬吹反而危险 |

---

## 建议顺序

```
第 1 天   B 组全部 8 条            零成本，且是 swx9 的态度分
          C3 前置查询（10 分钟）    决定 C3 做不做
第 2 天   C1 数据分布              CPU，最便宜的硬货
          C4 teacher baseline      补掉 yNv5 最后一条 → yNv5 三条全清
第 3–4 天 C2 Hungarian ID          GPU，最贵，但对 swx9 收益最高
最后      D1 收窄成定性对照 + 承诺
          D2 一段话承认
```

## 覆盖率账

| | 已答 | 计划内可答 | 只能承认 |
|---|---|---|---|
| **ws3J** (6) | Q1 Q2 W1 | W2 Q3 | W3 |
| **swx9** (3, conf 5) | W3/Q6 | W1 W2/Q5 W4/Q4 W5/Q3 W6/Q1 W7 Q2 格式 限制 | W8（收窄） |
| **yNv5** (3) | W2 W3 | W1 | — |

全部执行完的话：**yNv5 三条全清、swx9 除 W8 外全清**。这是 rebuttal 期内可达的最好状态。
