# Consumer-GPU Smoke Benchmarks for Mono-Forward Scheduled Local Updates in Small Byte-Level Decoders

## Abstract

We report consumer-GPU smoke benchmarks for mono-forward scheduled local updates in small byte-level decoder language models. Experiments run on an NVIDIA GeForce RTX 5070 Ti Laptop GPU using Tiny Shakespeare with byte vocabulary size 256 and hidden size 128. Mono-forward scheduled updates substantially improve tokens/sec in short runs: dense24 mono is 4.58x faster than dense24 chain-rule, and sparse4/24 mono is 17.63x faster than dense24 chain-rule. However, regular chain-rule training reaches lower validation cross entropy in these short runs, sparse active compute is faster but not equivalent to dense depth, and sparse4/24 does not outperform a normal dense4 model at matched active parameters. A simple recurrent mixer labeled `simple_gdn` runs stably and slightly improves mono validation CE, but is slower and higher-memory than softmax here. These results are smoke benchmarks, not claims of LLM-scale quality or official Gated DeltaNet performance.

## 1. Introduction

This document is a scaffold for a short research note. It should be expanded only after the release artifacts are frozen and longer controlled runs are available.

## 2. Motivation

The central question is whether scheduled local updates can improve practical throughput on consumer GPUs while maintaining useful learning behavior in small decoders.

## 3. Methods

The benchmark model is a decoder-only byte-level language model with learned position embeddings, RMSNorm, a causal mixer, MLP blocks, and a final language-model head.

## 4. Training Rules

`fp_chainrule` performs regular full backpropagation from final cross entropy every step. `fp_mono_update_every_N` performs scheduled mono-forward/local updates every `N` steps.

## 5. Dense vs Sparse Active Compute

Dense24 uses 24 active blocks. Sparse4/24 reports a logical 24-layer scaffold but instantiates and trains only 4 active blocks. Dense4 is the matched active-parameter reviewer baseline.

## 6. Softmax vs Simple Recurrent Mixer

`simple_gdn` is a simple causal recurrent/linear mixer and is not official Gated DeltaNet. It is labeled `official_gdn=false`.

## 7. Experimental Setup

The current release uses Tiny Shakespeare byte-level modeling, sequence length 256, hidden size 128, and an NVIDIA GeForce RTX 5070 Ti Laptop GPU.

## 8. Results

See `RESULTS_SUMMARY.md` for the release tables.

## 9. Limitations

These are short smoke benchmarks. They do not establish LLM-scale quality, official Gated DeltaNet behavior, or Transformer-scale competitiveness.

## 10. Future Work

The next technical step is a more faithful and optimized delta-rule recurrent mixer, followed by longer equal-update and equal-token comparisons.

## 11. Reproducibility Checklist

See `REPRODUCIBILITY.md`.
