# ws3J — W1 (final rebuttal text)

**Reviewer point.** *The actual generative model results are not presented so far as I
could see, so it isn't immediately obvious how much room for growth there is on this task
(although it seems this is significant).*

---

The reviewer is right, and the omission is structural rather than incidental. MIB attaches
**two label types to every pair** (Sec. 3): a **pairwise preference winner**, which is the
ranking signal, and **three diagnostic labels** — Existence, Appearance, Interaction —
which localise *why* one side won. The submission reports how faithfully **MIE reproduces**
those labels, but never reports **the labels themselves**, i.e. how the generators score on
the benchmark. Both are below, computed directly from the released Gold Set annotations.
No new data collection was needed; this is a reporting gap on our part.

**Denominators.** The Gold Set holds **4,020 pairs** — 2,520 main-generator plus 1,500
NanoBanana/MOSAIC, as in the dataset-statistics table. Removing pairs flagged
`prompt_ilogical` leaves **3,922**; additionally requiring the two annotators to agree on a
winner leaves **3,599**, which is the basis for the win rates.

## 1. The ranking signal — human pairwise win rates

| matchup | split | n | win rate |
|---|---|---|---|
| **nano_banana** vs MOSAIC | seen | 1,365 | 98.8 % : 1.2 % |
| **Seedream 4.5** vs GLM | unseen | 1,186 | 98.3 % : 1.7 % |
| **GPT-Image-1.5** vs Flux.2-klein-9B | unseen | 1,048 | 90.3 % : 9.7 % |

These are decisive contests, and the margin *widens* with scene complexity — GPT-Image-1.5
rises from 83.7 % at two subjects to 91.6 % at eight. Reported on its own, this is precisely
the table that leaves the reviewer's question open: it establishes an ordering and says
nothing about whether the winner is any good.

## 2. What the winners actually achieve — the diagnostic labels

**How to read the columns.** Each annotator marks each dimension **0 or 1 per generated
image**. Two annotators rate every item and disagreement is resolved by averaging (Sec. 3),
so a per-image label is one of **{0, 0.5, 1}** — 0.5 meaning the annotators split, which
occurs on 17.3 % of labels. The three dimension columns are those labels averaged over the
Gold Set: **the fraction of the benchmark on which that dimension is satisfied, where 1.000
would mean no failures of that kind anywhere.**

**The final column is not a metric from the paper — it is the conjunction of the three, and
we add it because the per-dimension means are marginals.** A prompt is only *completed* when
existence, appearance and interaction hold at once, so reading the three columns separately
overstates how often a generator actually succeeds. We define it strictly: **the share of
images where all three labels equal 1**, i.e. both annotators marked all three dimensions
satisfied — six of six marks positive.

| generator | n images | Existence | Appearance | Interaction | all three = 1 |
|---|---|---|---|---|---|
| GPT-Image-1.5 | 1,233 | 0.800 | 0.700 | 0.617 | **28.0 %** |
| Seedream 4.5 | 1,238 | 0.862 | 0.654 | 0.494 | 23.0 % |
| nano_banana | 1,451 | 0.816 | 0.396 | 0.533 | 19.0 % |
| Flux.2-klein-9B | 1,233 | 0.541 | 0.217 | 0.373 | 4.5 % |
| GLM | 1,238 | 0.315 | 0.067 | 0.161 | 0.8 % |
| MOSAIC | 1,451 | 0.262 | 0.070 | 0.216 | 0.6 % |

GPT-Image-1.5 shows the gap the conjunction is there to expose: marginals of 0.800 / 0.700 /
0.617 correspond to **28.0 %** task completion. (Under the looser reading, in which a split
annotation still counts as satisfied, it is 61.2 %; we report the strict figure and state the
loose one so the criterion is unambiguous.)

**This is the answer to the reviewer's question, and it is why both label types are needed.**
nano_banana wins its matchup 98.8 % of the time while preserving appearance on 0.396 of
images. Seedream 4.5 wins 98.3 % of the time with Interaction at 0.494. **Winning a
head-to-head does not mean solving the task — it means being the better of two, both of
which may be failing.** The strongest generator in the set completes 28 % of prompts.

## 3. Where the headroom is

Pooled over all six generators, by subject count:

| subjects | n images | Existence | Appearance | Interaction | all three = 1 |
|---|---|---|---|---|---|
| 2 | 1,996 | 0.876 | 0.580 | 0.626 | 32.1 % |
| 4 | 1,962 | 0.636 | 0.345 | 0.426 | 10.7 % |
| 6 | 1,956 | 0.518 | 0.245 | 0.314 | 5.2 % |
| **8** | 1,930 | **0.345** | **0.200** | **0.217** | **1.5 %** |

At eight subjects even **Existence — the most objective and easiest of the three — sits at
0.345**, meaning roughly two thirds of eight-subject prompts have a subject missing,
duplicated or collapsed. Task completion falls from 32.1 % to 1.5 % overall; for the
strongest generator alone it falls from 56.5 % to **5.9 %**, and three of the six generators
reach **0 %** at eight subjects.

The reviewer's impression that the headroom is significant is correct, and it is concentrated
exactly where MIB was constructed to reach. Which dimension binds is generator-dependent —
Appearance for four of the six, Interaction for the two strongest — which is itself the
argument for reporting the three separately rather than as one score.

## What we will add

All three tables, with the scale description above. Reporting only MIE's agreement, without
MIB's own scores, made the submission read as an evaluator paper when it is equally a
benchmark paper. We thank the reviewer for catching it.
