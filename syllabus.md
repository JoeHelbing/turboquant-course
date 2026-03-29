# TurboQuant Course Syllabus

## Module 00: Setup & PyTorch Warm-Up (20 min)

Verify the environment works. Quick introduction to PyTorch tensors: creating
them, shapes, broadcasting, indexing, and key operations like `torch.searchsorted`
and `torch.norm` that we'll use throughout the course.

## Module 01: Vectors, Dot Products & What Quantization Destroys (30 min)

Starting from embeddings you already know. What does a dot product measure
geometrically? Why is compressing vectors hard -- it's not the values that
matter, it's the relationships between vectors. Toy experiment: round embeddings
to fewer bits and watch nearest-neighbor search fall apart.

## Module 02: Scalar Quantization & Lloyd-Max (35 min)

The fundamental building block. Map continuous values to a small set of discrete
levels. Start with uniform quantization (equally spaced bins), see why it's
wasteful, then build the Lloyd-Max algorithm that places centroids optimally
for a given distribution. Implement from scratch, compare to reference code.

## Module 03: The Butterfly -- Fast Walsh-Hadamard Transform (30 min)

An FFT-like transform but simpler: only additions and subtractions, no complex
numbers. Build the butterfly algorithm step by step. Verify it's orthogonal
(preserves norms), invertible, and fast (O(d log d) vs O(d^2) for a full
matrix multiply).

## Module 04: Random Rotations & Why They're Magic (35 min)

The key insight. Apply a random rotation to any vector and its coordinates
become nearly independent and identically distributed. This is what allows
coordinate-wise scalar quantization to work. Build a randomized Hadamard
transform, verify the "spreading" effect visually, and understand why this
is the critical enabling trick.

## Module 05: The Beta Distribution on the Sphere (25 min)

After random rotation, each coordinate of a unit vector follows a Beta(d/2, d/2)
distribution that converges to Gaussian for large d. This means we know the
distribution IN ADVANCE -- no training data needed. Visualize the convergence.
Understand the dim >= 64 Gaussian shortcut vs exact Beta for small dimensions.

## Module 06: TurboQuant Stage 1 -- MSE-Optimal Quantization (40 min)

Assemble all the pieces: normalize -> rotate -> Lloyd-Max per coordinate ->
inverse rotate -> rescale. Build the complete MSE quantizer from your toy
components. Measure distortion at 1, 2, 3, 4 bits and compare against the
paper's Table 1. Then read through the reference implementation and understand
every line.

## Module 07: Random Projections & the JL Lemma (30 min)

A different kind of dimensionality magic. Project vectors with a random matrix
and pairwise distances are approximately preserved. Build intuition with 2D/3D
examples, then scale up. This is the theoretical foundation for Stage 2.

## Module 08: QJL -- 1-Bit Quantized Johnson-Lindenstrauss (35 min)

The extreme version: project and keep ONLY the sign bit. Despite throwing away
almost everything, this gives you unbiased inner product estimates. Build QJL,
verify unbiasedness empirically over many trials. Understand the sqrt(pi/2)
scaling factor and why it appears.

## Module 09: TurboQuant Stage 2 -- Unbiased Inner Products (30 min)

The elegant composition: MSE quantizer handles bulk compression with (b-1) bits,
QJL corrects the residual with 1 bit. Total: b bits, unbiased inner products.
Build the full two-stage system. Compare MSE-only vs full TurboQuant on inner
product accuracy.

## Module 10: How Good Can It Get? Shannon Bounds (20 min)

Information theory says there's a floor on how well ANY quantizer can do with
a given bit budget. TurboQuant gets within ~2.7x of this floor. Light treatment:
enough to understand the "near-optimal" claim without a full information theory
course.

## Module 11: Applications -- KV Cache & Vector Search (35 min)

Where this matters. Compress a transformer's KV cache and run attention on it --
see that quality barely changes. Build a vector search index with zero training
time and compare against product quantization on recall and speed.

## Module 12: Full System End-to-End (25 min)

Run the complete TurboQuant pipeline using the reference code. Reproduce key
paper results. Compare your toy implementations against the production code
side by side. Reflect on the journey from dot products to near-optimal
vector quantization.
