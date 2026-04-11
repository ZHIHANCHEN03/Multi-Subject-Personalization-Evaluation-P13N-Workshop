# Debug Session: ranking-gap-zero
- **Status**: [OPEN]
- **Issue**: In training, `ScoreA == ScoreB` and `Gap = 0.0000` even though A/B share the same prompt and refs but use different generated images.
- **Debug Server**: http://127.0.0.1:7778/event
- **Log File**: `.dbg/trae-debug-log-ranking-gap-zero.ndjson`

## Reproduction Steps
1. Run `bash Model_Training/run_a100_pipeline.sh` on the training server.
2. Observe the first training steps in `scripts/train.py`.
3. Confirm whether `ScoreA`, `ScoreB`, and `Gap` diverge.

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Evidence |
|----|------------|------------|--------|----------|
| A | The processor produces effectively identical multimodal inputs for A and B, so the generated image difference is not reaching the model. | High | Medium | Pending |
| B | The model backbone is receiving image tensors, but the visual branch is ignored or the required multimodal fields are incomplete. | High | Medium | Pending |
| C | The score branch is numerically collapsing even when backbone features differ, so A/B hidden states diverge but scalar scores remain identical. | Medium | Low | Pending |
| D | The current pooling/feature extraction path produces the same representation for both A and B because the selected token does not encode the generated-image difference. | Medium | Low | Pending |

## Log Evidence
- Pending instrumentation.

## Verification Conclusion
- Pending.
