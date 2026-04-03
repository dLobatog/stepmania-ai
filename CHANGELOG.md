# Changelog

## 2026-04-03 - Beat-Aware Token Model Iteration

### What We Tried

We continued from the stronger H100 token-pattern checkpoint and focused on the next two model ideas:

- expand the constrained token vocabulary beyond singles and simple jumps
- give the token model explicit beat-position context during training and inference

This was done incrementally and validated with short H100 smoke runs before attempting any longer continuation.

### Implemented In This Change

- expanded the ergonomic token vocabulary to allow rare triples and quads instead of banning them outright
- increased decode-time penalties for dense 3-arrow and 4-arrow patterns so they stay rare unless the model strongly prefers them
- added beat-position features to the token pattern model:
  - measure-phase sine/cosine
  - beat-phase sine/cosine
- stored beat-position features in cached training sequences so the model can learn phrase placement without reintroducing unstable beat tracking in the Linux feature-extraction path
- added backward-compatible checkpoint loading for the larger token vocabulary and beat-aware architecture
- added a smarter warm-start adapter that:
  - copies old token embeddings into the expanded embedding table
  - copies old output head rows into the larger token heads
  - preserves the old combined feature weights while leaving the new beat branch randomly initialized
- verified that new remote runs now register to Weights & Biases automatically under `dlobatog/stepmania-ai`

### What We Learned

- naive warm-starting into the larger vocab/beat-aware model was too destructive because too many learned weights were dropped
- the adapted warm-start path is much better: only the new beat branch remains randomly initialized
- W&B monitoring is now working reliably on the H100 box for new runs
- the next useful question is no longer "does the run start?" but "does beat-aware conditioning improve generated chart feel over the transition-loss baseline?"

### What We Will Do Next

- run a short multi-epoch H100 continuation from the adapted warm-start checkpoint
- render fresh `migos` and `paseo_estopa` comparison simfiles from that run
- if chart feel improves, continue training this beat-aware token model for a few more epochs
- if chart feel does not improve, keep the safer vocab changes and revisit the beat-conditioning design before investing in a longer run

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
- added a paper-oriented evaluation path that compares decoder strategies on fixed clips
- added machine-readable evaluation outputs so future ablations can become tables and figures instead of one-off terminal logs

This does not solve the whole problem, but it is the first real step away from unconstrained 4-bit sampling.

### What We Will Do Next

Short term:

- expand evaluation beyond loss/accuracy into pattern statistics and clip-based qualitative review
- compare `raw` vs `ergonomic` decoding on fixed clips and save the comparison as structured artifacts
- tune the constrained decoder using known StepMania / ITG pattern categories
- finish the current full-pack pattern run, then regenerate fresh 1-minute `migos` and `paseo_estopa` clips as the fixed qualitative check
- after those clip checks, implement a window-level pattern objective so training scores short motif quality instead of only per-onset token accuracy
- keep the per-token loss, but add short-horizon transition or n-gram supervision so repetitive streams and accidental jacks are penalized during training rather than only at decode time

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
  - reproducible clip-level decoder comparison reports
