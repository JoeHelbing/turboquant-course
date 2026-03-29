# TurboQuant Deep Dive

An interactive course that builds your understanding of
**TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate**
(Zandieh et al., ICLR 2026) from the ground up.

## What you'll learn

- How to compress high-dimensional vectors to 2-4 bits per coordinate
  while preserving distances and inner products
- The Walsh-Hadamard transform and why random rotations make quantization work
- Lloyd-Max optimal scalar quantization
- The Johnson-Lindenstrauss lemma and its 1-bit quantized variant (QJL)
- How these pieces compose into TurboQuant's two-stage system
- Real applications: KV cache compression for LLMs, vector search

## Prerequisites

- Basic Python (loops, functions, numpy arrays)
- Some experience with embeddings / vector similarity
- Curiosity about how compression works

No prior experience with quantization, information theory, or PyTorch required --
the course teaches all of it.

## Setup

```bash
# Clone this repo
cd turboquant-course

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .

# Launch Jupyter
uv run jupyter notebook notebooks/
```

## Course structure

13 modules, ~6 hours total. Work at your own pace.

| # | Module | Time | Topic |
|---|--------|------|-------|
| 00 | Setup & PyTorch Warm-Up | 20m | Environment, tensor basics |
| 01 | Vectors & What Quantization Destroys | 30m | Dot products, naive compression |
| 02 | Scalar Quantization & Lloyd-Max | 35m | Optimal 1D quantizers |
| 03 | The Butterfly -- Walsh-Hadamard Transform | 30m | Fast orthogonal transforms |
| 04 | Random Rotations & Why They're Magic | 35m | Spreading information uniformly |
| 05 | Beta Distribution on the Sphere | 25m | Why we know the distribution in advance |
| 06 | TurboQuant Stage 1 -- MSE Quantizer | 40m | Putting it all together |
| 07 | Random Projections & JL Lemma | 30m | Distance preservation |
| 08 | QJL -- 1-Bit Quantized JL | 35m | Unbiased inner products from sign bits |
| 09 | TurboQuant Stage 2 -- Unbiased IP | 30m | The two-stage composition |
| 10 | Shannon Bounds -- How Good Can It Get? | 20m | Information-theoretic limits |
| 11 | Applications -- KV Cache & Vector Search | 35m | Real-world usage |
| 12 | Full System End-to-End | 25m | Reproduce paper results |

See [syllabus.md](syllabus.md) for detailed module descriptions.

## Papers

- `papers/main-paper.pdf` -- TurboQuant (arXiv:2504.19874)
- `papers/qjl-paper.pdf` -- QJL: 1-Bit Quantized JL Transform (arXiv:2406.03482)
- `papers/reading-guide.md` -- What to read and when

## Reference code

The `reference-code/turboquant-torch/` directory contains an unofficial PyTorch
implementation of TurboQuant. The course builds toy versions of each component
and then shows how they connect to this real implementation.
