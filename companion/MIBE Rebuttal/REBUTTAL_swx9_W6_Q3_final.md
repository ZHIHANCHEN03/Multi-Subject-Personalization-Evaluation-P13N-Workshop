# swx9 — W6 / Q3 (final rebuttal text)

**Reviewer point.** *Some details on architecture are missing in the paper (see
questions); What are the ranking and diagnostic heads comprising of, and where are they
in the model?*

---

The reviewer is right that the architecture is under-specified, and we accept that as a
reproducibility gap rather than a presentational one. Below is the complete
specification; we will add it to the method section together with a diagram, and the
definitions are in the released training code.

## 1. Overall structure

MIE is not a prompted generative judge. It is a **frozen-format scoring model**: a
vision-language backbone followed by two small parallel heads, with no token generation
at any point.

```
k reference images  ─┐
generated candidate ─┼─→  Qwen3.5 VL backbone  ─→  pooled vector h
guiding prompt      ─┘         (LoRA + last 4 layers trainable)        │
                                                                       ├─→ ranking head    → 1 scalar
                                                                       └─→ diagnostic head → 3 logits
```

The backbone is a Qwen3.5 vision-language model; we report **0.8B / 2B / 4B** variants and
the headline model is **4B**. The *k* subject references, the candidate image and the
guiding prompt are encoded as a single multimodal sequence in one forward pass.

**The pooling point, which fixes where the heads live.** We take the hidden state of the
**last non-padding token of the final backbone layer**, located via the attention mask so
that padding never contributes, as one vector summarising the entire (references, prompt,
candidate) context. **Both heads attach to that single vector.** Nothing is attached to
intermediate layers; the heads share no parameters and do not communicate — each reads
the same pooled representation independently.

## 2. What the two heads comprise

Each head is a two-layer MLP with a GELU non-linearity. They are structurally identical
and differ only in output width:

| head | structure | output |
|---|---|---|
| **Ranking (score) head** | `Linear(h → h/2)` → `GELU` → `Linear(h/2 → 1)` | one scalar preference score |
| **Diagnostic (classification) head** | `Linear(h → h/2)` → `GELU` → `Linear(h/2 → 3)` | three logits — existence, appearance, interaction |

where *h* is the backbone hidden size. Both heads are held in **float32** even when the
backbone runs in bfloat16. This is not incidental: in early experiments bfloat16 heads
underflowed and returned numerically identical scores for the two members of a pair,
which destroys the ranking signal while leaving the training curves apparently healthy.

## 3. How the two heads are supervised

A pair (A, B) is passed through the backbone twice — same prompt text, same reference
images, differing only in the final candidate image — producing `(score_A, logits_A)` and
`(score_B, logits_B)`. The heads then take different kinds of supervision from the same
forward passes:

- the **ranking head** is trained with a **margin ranking loss** on `(score_A, score_B)`
  against the binary preference label, so it is supervised **relatively**;
- the **diagnostic head** is trained with **`BCEWithLogitsLoss`** against the three binary
  facet labels, computed per image and averaged over the two sides, so it is supervised
  **absolutely**.

The objective is

**L = α · L_rank + β · L_diag,  α = 1.0, β = 0.5.**

The heads are always trainable regardless of which backbone regime is used.

## 4. Training configuration for the reported model

The reported configuration is `lora_layer`: LoRA adapters (**r = 16, lora_alpha = 32,
dropout 0, no bias**) injected into **both the vision and language towers** across
attention and MLP modules, **plus the last 4 transformer layers unfrozen**. Optimisation
uses lr 2e-5, weight decay 0.01, warmup ratio 0.03, seed 3407, references at 512 px, with
the epoch count auto-scaled (between 2 and 4) to reach roughly 6000 optimizer updates at
an effective batch size of 16.

The choice of `lora_layer` over unfreezing layers alone is empirical rather than
stylistic. At matched backbone size:

| backbone | overall pairwise acc. (Δ lora_layer − layer_only) | unseen split (Δ) | macro-F1 (Δ) |
|---|---|---|---|
| 0.8B | −0.001 | −0.004 | +0.128 |
| 2B | **+0.061** | **+0.096** | +0.116 |
| 4B | **+0.046** | **+0.076** | +0.044 |

Adding LoRA on top of layer unfreezing helps most on the **unseen** split and at the
larger backbones, and it improves diagnostic macro-F1 at every size — i.e. it buys
generalization and facet quality, not just fit.

## 5. Load-time verification

Because a partial weight load can silently masquerade as a result, we emit a load report
on every inference run. For the reported 4B checkpoint it records **688 LoRA parameter
tensors**, **53 of 53 trainable-backbone tensors applied with 0 unexpected keys**, and
both heads (`score_head`, `classification_head`) restored. Every number we report was
produced under a load that passed this check.

## 6. Read-out at inference

No tokens are generated (`add_generation_prompt=False`). The **pairwise decision is the
sign of the score difference** between the two candidates, and the reported facet values
are **sigmoids of the diagnostic logits**. Since the prompt text and the reference images
are shared between A and B, they are encoded once and reused across the pair — which is
what makes scoring the full 4020-pair Gold Set tractable.

## 7. Why the split exists

The ranking head alone would give a single opaque preference score. The diagnostic head is
what makes that score **attributable**: it produces the per-facet probabilities used
throughout our analysis, which is how we can state that interaction is the weakest facet
for every generator we tested, and — importantly for validating the evaluator — how a
reader can check MIE against human facet marks **separately from** its ranking behaviour.
The two heads give two independent handles on the same model, and they do behave
differently: in our reference-extension study MIE's existence probability correlates with
human marks at **r = +0.93**, while appearance is bounded by annotator disagreement
(κ = 0.19–0.34) and cannot be adjudicated by those annotations at all. A single scalar
would have hidden both facts.

## What we will add to the paper

§1–3 as a method subsection with a diagram; §4 as a training-details paragraph and the
LoRA-vs-layer ablation as a table; §5–6 as implementation notes. We are grateful for the
push — these are exactly the details a reader would need to reimplement MIE, and their
absence was an oversight on our part.
