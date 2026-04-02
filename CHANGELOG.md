# Changelog

## 2026-04-02 - Baseline Training And Decoder Pivot

### What We Tried

We trained the current two-stage baseline:

- Stage 1: onset detector over dense audio frames
- Stage 2: pattern generator that predicts a 4-bit arrow activation vector at each detected onset

We also added the infrastructure needed to iterate faster:

- reproducible train/validation song splits
- early stopping
- timestamped runs and checkpoints
- TensorBoard monitoring
- clip-based evaluation helpers
- inference progress bars

### What We Learned

The baseline is directionally correct for timing but weak for playability.

- The onset detector is useful. Subjectively, downbeats and basic note placement are often reasonable.
- The pattern generator is not good enough yet. It can produce legal note rows, but the charts feel unergonomic and musically flat.
- Validation metrics did not capture this gap well. Pattern accuracy improved only slightly, while qualitative playability stayed poor.

This means the current factorized pattern objective is mismatched with the real goal. Predicting each arrow independently is not the same as generating a playable dance pattern.

### Decision

From this point forward, the project shifts from:

- "predict independent arrow bits at each onset"

to:

- "generate playable dance motifs and transitions with explicit ergonomic constraints"

This is a better fit for both product quality and an eventual academic paper because it creates a clean research story:

1. establish a timing baseline
2. show that raw binary pattern decoding underperforms qualitatively
3. introduce constrained / ergonomics-aware decoding
4. move toward tokenized pattern modeling and motif-aware evaluation

### Implemented In This Change

We introduced the first ergonomics-aware decoding layer in inference:

- added a constrained vocabulary of explicitly playable singles and jumps
- added a rule-based transition scorer that penalizes:
  - dense jumps
  - repeated jumps
  - short-interval same-arrow jacks
  - staircase-like continuations
- exposed decoder choice as an ablation:
  - `--decode-strategy ergonomic`
  - `--decode-strategy raw`

This does not solve the whole problem, but it is the first real step away from unconstrained 4-bit sampling.

### What We Will Do Next

Short term:

- expand evaluation beyond loss/accuracy into pattern statistics and clip-based qualitative review
- compare `raw` vs `ergonomic` decoding on fixed clips
- tune the constrained decoder using known StepMania / ITG pattern categories

Medium term:

- replace the pattern head with a token vocabulary over common step patterns and jumps
- train on transitions and motifs rather than independent arrows
- condition generation on style and difficulty

Paper direction:

- baseline: onset-only + raw binary pattern decoding
- method 1: constrained ergonomics-aware decoder
- method 2: tokenized motif model
- evaluation:
  - onset metrics
  - pattern statistics
  - ablations
  - human playability review
