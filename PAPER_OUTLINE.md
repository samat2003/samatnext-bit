# Paper Outline

Suggested title:

**Mono-Forward Scheduled Updates for Consumer-GPU Smoke Training of Small Decoder Models**

## Abstract

- Consumer laptop GPU.
- Mono-forward as a training rule, not a new architecture.
- Tiny Shakespeare byte-level throughput gains.
- Chain-rule still better short-run CE in early comparisons.
- Python-token practical smoke runs.
- Dense24 optimized MBPP smoke run under 4GB peak CUDA memory.
- No coding benchmark score, no pass@1, no LLM-scale claim.

## Introduction

- Motivation for small, reproducible CUDA/PyTorch smoke benchmarks.
- Need to separate throughput, quality, active compute, and update count.

## Background / Motivation

- Cost of full chain-rule backprop.
- Scheduled local/mono updates as a throughput-oriented training rule.
- Why consumer-GPU smoke tests are useful but limited.

## Method: Mono-Forward Scheduled Updates

- `fp_chainrule`.
- `fp_mono_update_every_N`.
- Optimizer update accounting.
- Non-update no-grad optimization for dense24 mono.

## Experimental Setup

- Hardware: NVIDIA GeForce RTX 5070 Ti Laptop GPU.
- Software: PyTorch 2.11.0+cu128, CUDA 12.8.
- Datasets:
  - Tiny Shakespeare byte-level.
  - Local Python-code BPE corpus.
  - MBPP sanitized smoke-training corpus.
- Timing rules and FLOP estimate caveat.

## Tiny Shakespeare Byte-Level Experiments

- Chain-rule vs mono 24-layer.
- Dense4 vs sparse4/24.
- Softmax vs `simple_gdn`.

## Regular PyTorch Baseline Audit

- Plain PyTorch 4-layer Transformer-style chain-rule baseline.
- Comparison to SamatNext dense4 chain-rule and mono.

## Practical Python-Token Smoke Experiment

- Hidden 512, seq 1024, vocab 16,000.
- Regular PyTorch vs SamatNext dense4/sparse4 mono.
- Larger setup lowers extreme byte-level tok/s.

## BF16 Softmax vs Simple GDN

- BF16 support.
- `simple_gdn` stable but slower, higher-memory, and not better CE.
- No official Gated DeltaNet claim.

## Dense24 VRAM Optimization

- Dense 24/24 softmax mono.
- Hidden 512, seq 512, batch 8.
- Non-update no-grad optimization.
- Memory reduction from 7.459 GB to 4.641 GB.

## MBPP Dense24 500-Step Smoke Training

- Sanitized MBPP as text-only smoke corpus.
- 81.1M params, 500 FP16 steps, 83.6K tok/s, 3.992GB peak CUDA memory.
- Validation CE 8.7202 to 3.9844.
- No test execution, no pass@1, no coding ability claim.

## Discussion

- Throughput and memory claims.
- Quality caveats.
- Why sparse4/24 did not show depth benefit.
- Why `simple_gdn` work stops for this milestone.

## Limitations

- Small corpora.
- Short runs.
- MBPP repeated exposure.
- No coding benchmark score.
- No long-run convergence.
- No LLM-scale claim.

## Reproducibility

- Commands.
- Config files.
- Generated data paths.
- Hardware/software versions.

## Future Work

- Longer runs.
- Matched chain-rule MBPP comparison.
- More careful local loss accounting.
- Better recurrent mixers only after a faithful implementation exists.
- Kernel optimization as a separate milestone.

## Citation Placeholders

- TODO: cite MBPP / Google Research.
- TODO: cite PyTorch.
- TODO: cite Tiny Shakespeare if needed.
