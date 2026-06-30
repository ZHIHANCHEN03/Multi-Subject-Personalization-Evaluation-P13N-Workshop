# DPO selection report — 2k

## Selection funnel

| stage | pairs | kept % |
|---|---:|---:|
| paired (both sources) | 59,852 | 100.0% |
| winner agree | 56,909 | 95.1% |
| strong signal | 53,746 | 89.8% |
| kept after Layer 0 | 53,746 | 89.8% |
| selected (2k) | 1,992 | 3.3% |

- dropped — winner disagree: **2,943**, weak signal (‖δ‖≈0): **3,163**, missing N: **0**

## Layer 1 — subject-count (N) buckets

| N | selected | share |
|---:|---:|---:|
| 2 | 498 | 25.0% |
| 4 | 498 | 25.0% |
| 6 | 498 | 25.0% |
| 8 | 498 | 25.0% |

## Layer 3 — primary dimension

| dimension | selected | share |
|---|---:|---:|
| existence | 664 | 33.3% |
| appearance | 664 | 33.3% |
| interaction | 664 | 33.3% |

## (N x primary dimension) grid — selected / pool after filter

| N | existence | appearance | interaction | row total |
|---:|---:|---:|---:|---:|
| 2 | 166 / 3230 | 166 / 5531 | 166 / 2973 | 498 |
| 4 | 166 / 9106 | 166 / 2412 | 166 / 1100 | 498 |
| 6 | 166 / 11747 | 166 / 1280 | 166 / 394 | 498 |
| 8 | 166 / 10152 | 166 / 1720 | 166 / 270 | 498 |

- per-cell target = **166**; ⚠ marks cells that fell back to taking the full pool (scarce stratum).

## Winner balance

- A wins: **1957** (98.2%)  
- B wins: **35** (1.8%)

## Signal statistics (selected set)

| metric | mean | median | min | max |
|---|---:|---:|---:|---:|
| significance S | 0.386 | 0.333 | 0.000 | 1.000 |
| ‖δ‖₁ | 1.627 | 1.500 | 0.500 | 3.000 |

## Figures

- ![funnel](figures/funnel_2k.png)
- ![grid](figures/grid_2k.png)
- ![dashboard](figures/dashboard_2k.png)
