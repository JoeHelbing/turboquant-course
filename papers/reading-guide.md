# Paper Reading Guide

## When to read what

Don't try to read the papers front-to-back before starting the course. The
notebooks will point you to specific sections at the right time.

## Main Paper: TurboQuant (arXiv:2504.19874)

### First pass (before Module 01)
- **Abstract** -- just the first paragraph, get the high-level goal
- **Section 1 (Introduction)** -- skim for motivation, skip math

### During Module 06 (Stage 1)
- **Section 3.1** -- MSE-optimal quantizer. This is where the rotation +
  Lloyd-Max approach is described. Read carefully after you've built your
  own version.
- **Lemma 1** -- The Beta distribution result. Short and elegant.
- **Theorem 3.3** -- The MSE distortion bound. Focus on the result, not
  the full proof.
- **Table 1** -- Distortion values. Compare against your experiments.

### During Module 09 (Stage 2)
- **Section 3.2** -- Inner product quantizer. The two-stage composition.
- **Algorithm 1 and 2** -- Compare against your implementations.

### During Module 10 (Shannon bounds)
- **Theorem 3.5** -- The lower bound. Read the statement; the proof is
  information-theoretic and optional.

### During Module 11 (Applications)
- **Section 4** -- Experimental results. KV cache, vector search benchmarks.

## Prerequisite Paper: QJL (arXiv:2406.03482)

### During Module 08 (QJL)
- **Section 1-2** -- Motivation and the QJL definition (Definition 1)
- **Theorem 1** -- The unbiasedness and variance bound
- Skip the KV cache application sections (the TurboQuant paper supersedes them)

## Papers you DON'T need to read

- Shannon's original papers -- the course covers what you need
- Lloyd/Max original papers -- the algorithm is simple enough to learn from code
- Johnson-Lindenstrauss original paper -- the course explains the lemma directly
