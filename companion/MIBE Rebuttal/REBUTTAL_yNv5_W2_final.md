# yNv5 — W2 (final rebuttal text)

**Reviewer point.** *The "ground truth" is itself VLM-generated imagery. Reference subjects
are GPT-Image-1 generations, and all candidates are generated images. When evaluating
identity preservation, the reference identity is synthetic and may not be self-consistent,
injecting hard-to-bound noise into the Appearance dimension.*

---

This is the most precisely stated of the three weaknesses: it names a mechanism —
synthetic references may not be self-consistent — and the place it should show up, the
Appearance dimension. Because it is that specific it is falsifiable, and we ran the test.

## The test

If synthetic references inject identity noise, then annotators judging appearance
**against a real photograph** should agree with each other more than annotators judging
appearance **against a generated reference**. Our reference-extension study re-runs the
identical 216 structured prompts against **three reference sources — A: licensed real
photographs, B: GPT-Image, C: Qwen-Image** — over 22 subjects disjoint from MIB's 80,
with **every item independently rated by two annotators** (2592 votes, 8 annotators). The
three sources therefore differ in exactly the variable the reviewer identifies and in
nothing else, so inter-annotator agreement is directly comparable across them.

Cohen's κ, reported for **all three facets** so the comparison is not selective
(GPT-Image-1.5 and Flux.2 form the unseen-generator batch; nano_banana and MOSAIC the
lopsided one):

| facet | generator | **A — real photos** | B — GPT-Image | C — Qwen-Image |
|---|---|---|---|---|
| **existence** | GPT-Image-1.5 | +0.945 | +0.879 | +0.852 |
| | Flux.2-klein-9B | +0.818 | +0.829 | +0.760 |
| | nano_banana | +0.728 | +0.859 | +0.864 |
| | MOSAIC | +0.776 | +0.846 | +0.822 |
| **appearance** | GPT-Image-1.5 | **+0.062** | +0.266 | +0.239 |
| | Flux.2-klein-9B | +0.344 | +0.303 | +0.290 |
| | nano_banana | +0.610 | +0.500 | +0.515 |
| | MOSAIC | +0.593 | +0.412 | +0.576 |
| **interaction** | GPT-Image-1.5 | +0.474 | +0.506 | +0.530 |
| | Flux.2-klein-9B | +0.445 | +0.405 | +0.501 |
| | nano_banana | +0.377 | +0.390 | +0.634 |
| | MOSAIC | +0.250 | +0.387 | +0.281 |

## What the test shows

**The direction is weakly consistent with the hypothesis, and we will not overstate the
result.** On appearance, real-photograph references give the highest κ for **three of four
generators**. But no bootstrap interval separates a real-reference cell from its synthetic
counterparts; the differences are small relative to the spread across facets; and the
largest single deviation runs the other way — for **GPT-Image-1.5, the strongest generator,
appearance agreement against real photographs is κ = +0.062, an interval covering zero**,
the lowest value we measured anywhere. We therefore cannot rule out a modest effect of the
kind the reviewer describes.

**What we can rule out is that it accounts for the appearance ceiling.** Appearance κ stays
in the 0.06–0.61 range **even when the reference is a real photograph**, while existence
sits at 0.73–0.95 under every source. On the unseen-generator batch the gap is starkest:
appearance 0.06–0.34 against interaction 0.41–0.53 and existence 0.82–0.95, with the same
ordering whichever source the references came from. The dominant term is the **facet**, not
the reference type — removing synthetic references entirely does not lift appearance out of
the range where these annotations cannot adjudicate it.

Interaction, which the reviewer's account does not predict either way, moves in the
opposite direction (real references give the *lowest* κ for three of four generators) — a
further reason to read these differences as noise rather than as a reference-quality
signal. The GPT-Image-1.5 cell also admits a simpler reading than reference noise: when a
generator preserves identity well, appearance becomes a subtle and genuinely subjective
judgement, and annotators diverge more.

**The overall judgement behaves the same way.** Agreement on the preference vote is
82.4 % (κ = +0.327) with real references, 81.5 % (κ = +0.267) with GPT-Image references
and 83.8 % (κ = +0.371) with Qwen-Image references — indistinguishable.

*A statistical caveat we state ourselves: on the lopsided batch (nano_banana vs MOSAIC)
we report raw preference agreement — 96.8 / 96.8 / 94.9 % — and not κ. With 99.8 % of
votes falling on one side, the chance-agreement term approaches 1 and κ becomes
uninformative, turning slightly negative despite near-perfect observed agreement. This is
the same class-imbalance pathology that leads us to report AUC rather than raw accuracy
elsewhere in this response.*

## Downstream, alignment is likewise unaffected

| reference source | n | AUC [95 % CI] | MIE existence vs human marks |
|---|---|---|---|
| **A — real photographs** | 173 | **0.827** [0.706, 0.921] | **+0.932** |
| B — GPT-Image | 176 | 0.851 [0.760, 0.929] | +0.926 |
| C — Qwen-Image | 181 | 0.775 [0.681, 0.864] | +0.938 |

All intervals clear 0.5, A and B are statistically indistinguishable, the per-facet
existence correlation is essentially invariant, and the **generator ranking is identical
under all three sources**. This used the **identical checkpoint** reported in the paper;
nothing was retrained.

## Where the reviewer is right

**The conclusion holds, and we cannot exclude the proposed cause — only bound how much it
explains.** Appearance *is* the noisiest dimension on the batch that carries our evidence
— annotator κ of 0.19–0.34 pooled — and a modest contribution from synthetic references
remains possible on the evidence above. What the data do settle is that this is not the
binding constraint: the ceiling persists under real photographs. Either way the
consequence for how we report is the same, and we take it seriously enough to change it:
**these annotations cannot adjudicate the appearance head in either direction**, and we do
not claim they validate it. We will present appearance results with that ceiling stated,
rather than alongside existence as though the two carried equal evidential weight.

We also agree there is a real limitation on the reference side, though a different one
from self-consistency. **A single canonical reference per subject cannot separate identity
preservation from identity replication** — the copy-paste failure mode, where a model
reproduces the reference rather than re-rendering that identity under the pose and
lighting the prompt demands. MIB mitigates this only partially: its appearance facet is a
human/VLM judgement rather than a reference-similarity score, so it does not mechanically
reward replication the way ArcFace or DINO similarity does, and because appearance and
interaction are scored separately, a copy-pasted subject in an action scene should
register as high appearance with low interaction. Neither of these *measures* copy-paste,
and we do not claim to have quantified it; that requires multiple real photographs per
identity, which our single-reference design does not provide.

On the remaining part of the framing — that the candidates are themselves generated
images — we would note this is intrinsic to the task rather than a property of our
construction: generated images are the object of evaluation, not a proxy for it.

## What we will do

Release the **real-photograph split (A)** and report both settings; add the κ comparison
above to the paper so readers can see that the appearance ceiling is a property of the
dimension rather than of the reference source; state the appearance ceiling and the
single-reference limitation explicitly; and treat a multi-reference-per-identity
extension as the natural next version of the benchmark. This analysis exists because of
this review point, and we are grateful for it.
