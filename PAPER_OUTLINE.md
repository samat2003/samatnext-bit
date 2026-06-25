# Consumer-GPU Smoke Benchmarks for Mono-Forward Scheduled Local Updates in Small Byte-Level Decoders

## Abstract

We report consumer-GPU smoke benchmarks for mono-forward scheduled local updates in small byte-level decoder language models. Experiments run on an NVIDIA GeForce RTX 5070 Ti Laptop GPU using Tiny Shakespeare with byte vocabulary size 256 and hidden size 128. Mono-forward scheduled updates substantially improve tokens/sec in short runs: dense24 mono is 4.58x faster than dense24 chain-rule, and sparse4/24 mono is 17.63x faster than dense24 chain-rule. However, regular chain-rule training reaches lower validation cross entropy in these short runs, sparse active compute is faster but not equivalent to dense depth, and sparse4/24 does not outperform a normal dense4 model at matched active parameters. A simple recurrent mixer labeled `simple_gdn` runs stably and slightly improves mono validation CE, but is slower and higher-memory than softmax here. These results are smoke benchmarks, not claims of LLM-scale quality or official Gated DeltaNet performance.

## Introduction

- Small-scale CUDA/PyTorch reproduction environment for mono-forward scheduled update experiments.
- Motivation for measuring practical throughput on consumer GPUs.
- Need to separate throughput claims from quality and active-parameter accounting.

## Motivation

- Full chain-rule backprop is expensive.
- Scheduled local updates can reduce backward/update frequency.
- Sparse active compute can improve throughput, but must be compared to matched active-parameter baselines.

## Methods

- Decoder-only byte-level model.
- Learned token and position embeddings.
- RMSNorm, causal mixer, MLP, LM head.
- Hidden size 128, sequence length 256, vocab size 256.
- Tiny Shakespeare 90/10 train/validation split.

## Training Rules

- `fp_chainrule`: full final-CE backprop every step.
- `fp_mono_update_every_N`: scheduled mono-forward/local update rule.
- Optimizer update counts are reported and not treated as equal unless explicitly matched.

## Dense vs Sparse Active Compute

- dense24: 24 active blocks, 4,851,072 active params.
- sparse4/24: logical 24-layer scaffold, 4 active blocks, 890,752 active params.
- dense4: normal 4-layer model, 890,752 active params.
- Sparse4/24 must not be called dense24 training.

## Softmax vs Simple Recurrent Mixer

- `softmax`: normal causal softmax attention.
- `simple_gdn`: simple causal recurrent/linear mixer.
- `official_gdn=false`.
- Stable first, not optimized, not official Gated DeltaNet.

## Experimental Setup

- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU.
- Dataset: Tiny Shakespeare, byte-level, vocab 256.
- Runs: 500 training steps, validation every 100 steps.
- Training batches preloaded on CUDA.
- Throughput excludes validation and data loading.

## Results

- Chain-rule vs mono 24-layer results.
- Dense4 vs sparse4/24 reviewer baseline.
- Softmax vs simple_gdn 4-active comparison.
- FLOPs/token reported as parameter-count estimates.

## Limitations

- Small byte-level smoke benchmark only.
- Not LLM-scale.
- No claim of beating Transformers at scale.
- No official Gated DeltaNet.
- No real 1.58-bit speedup claim.
- FLOPs are estimates, not hardware counters.
- Short 500-step runs may not reflect long-run quality.

## Future Work

- Longer equal-update and equal-token experiments.
- Better local loss accounting.
- More faithful delta-rule mixer.
- Optimized scan/fused recurrent kernels.
- Larger datasets and model sizes after methodology is stable.

## Reproducibility Checklist

- Code release includes configs.
- Tests pass with `python -m pytest -q`.
- Dataset path documented.
- Hardware documented.
- Seeds and validation checkpoints fixed in configs.
- Negative results reported.
