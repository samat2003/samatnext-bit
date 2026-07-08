# Consumer-GPU Smoke Benchmarks for Mono-Forward Scheduled Local Updates in Small Byte-Level Decoders

## Abstract

We report consumer-GPU smoke benchmarks for mono-forward scheduled local updates in small byte-level decoder language models. Experiments run on an NVIDIA GeForce RTX 5070 Ti Laptop GPU using Tiny Shakespeare, byte vocabulary size 256, sequence length 256, and hidden size 128. Mono-forward scheduled updates substantially improve training throughput in short runs: dense24 mono reaches 4.58x the tokens/sec of dense24 chain-rule, and sparse4/24 mono reaches 17.63x the tokens/sec of dense24 chain-rule. However, regular chain-rule training reaches lower short-run validation cross entropy. The sparse4/24 setting uses only 4 active/instantiated blocks and does not outperform a normal dense4 model at matched active parameter count. A simple recurrent mixer labeled `simple_gdn` runs stably and slightly improves mono validation CE, but is slower and higher-memory than softmax attention in this implementation. These are small byte-level smoke benchmarks on a consumer laptop GPU; we make no LLM-scale claim, no claim of beating Transformers at scale, and no claim that `simple_gdn` is official Gated DeltaNet.

## 1. Introduction

This technical report documents a reproducible milestone for `samatnext-bit`, a CUDA/PyTorch research sandbox for mono-forward scheduled local updates. The goal is narrow: measure short-run throughput and validation behavior on a consumer GPU using a small byte-level decoder.

The report emphasizes accounting. Throughput improvements from scheduled updates are useful only when update counts, active parameter counts, and model structure are reported clearly. In particular, sparse4/24 is a logical 24-layer scaffold with only 4 active/instantiated blocks; it is not dense24 training.

## 2. Motivation

Full chain-rule backpropagation through all active layers is expensive. A scheduled local-update rule can reduce backward/update frequency and improve measured training throughput. The open question for this milestone is whether those throughput gains come with acceptable short-run validation behavior, and whether sparse logical depth or a simple recurrent mixer improves the 4-active-block regime.

## 3. Methods

The model is a decoder-only byte-level language model with:

- byte vocabulary size 256
- learned token embeddings
- learned position embeddings
- RMSNorm
- causal mixer blocks
- MLP blocks
- final language-model head

The release experiments use hidden size 128, sequence length 256, and 4 attention heads. This is a small decoder used for smoke benchmarking, not a production LLM.

## 4. Training Rules

`fp_chainrule` performs regular full backpropagation from final cross entropy every step.

`fp_mono_update_every_N` performs the mono-forward scheduled local update rule. It runs forward every step and performs optimizer updates every `N` steps. Because update counts differ, comparisons must report both tokens/sec and optimizer updates.

## 5. Dense vs Sparse Active Compute

The release uses three model accounting regimes:

| name | description | active params |
|---|---|---:|
| dense24 | 24 active blocks out of 24 | 4,851,072 |
| sparse4/24 | logical 24-layer scaffold with 4 active/instantiated blocks | 890,752 |
| dense4 | normal 4-layer model | 890,752 |

The dense4 baseline is important: it tests whether sparse4/24 provides any quality benefit beyond simply training a normal 4-layer model at the same active parameter count.

## 6. Softmax vs Simple Recurrent Mixer

`softmax` is normal causal softmax attention.

`simple_gdn` is a simple causal recurrent/linear mixer implemented for this smoke benchmark. It is explicitly not official Gated DeltaNet:

- `mixer_type=simple_gdn`
- `official_gdn=false`
- `linear_recurrent_mixer=true`

The implementation is stability-first and not optimized.

## 7. Experimental Setup

Hardware and software:

- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- GPU class: 12GB consumer laptop GPU
- Python: 3.12.3
- PyTorch: 2.11.0+cu128
- CUDA: 12.8
- Triton: 3.6.0

Dataset:

- Tiny Shakespeare
- path: `data/english_validation.txt`
- bytes loaded: 1,115,394
- train/validation split: 90/10
- vocab size: 256

Benchmark settings:

- 500 training steps
- validation every 100 steps
- fixed validation batches
- preloaded CUDA training batches
- measured training speed excludes validation, token loading, and batch sampling

FLOPs/token are simple parameter-count estimates, not hardware counters.

## 8. Results

### 8.1 Chain-Rule vs Mono 24-Layer

| track | rule | active blocks | active params | updates | tok/s | final val CE |
|---|---|---:|---:|---:|---:|---:|
| dense24_chainrule | chainrule | 24/24 | 4,851,072 | 500 | 145,006 | 2.2685 |
| dense24_mono | mono update every 16 | 24/24 | 4,851,072 | 32 | 663,520 | 3.2379 |
| sparse4_24_chainrule | chainrule | 4/24 | 890,752 | 500 | 617,245 | 2.4463 |
| sparse4_24_mono | mono update every 8 | 4/24 | 890,752 | 63 | 2,556,872 | 3.0240 |

Dense24 mono improves throughput by 4.58x over dense24 chain-rule, but dense24 chain-rule reaches lower validation CE in the short run. Sparse4/24 mono improves throughput by 17.63x versus dense24 chain-rule, but it uses only 4 active blocks.

### 8.2 Dense4 vs Sparse4/24

| track | rule | active blocks | active params | updates | tok/s | final val CE |
|---|---|---:|---:|---:|---:|---:|
| dense4_chainrule | chainrule | 4/4 | 890,752 | 500 | 661,312 | 2.4463 |
| dense4_mono | mono update every 8 | 4/4 | 890,752 | 63 | 2,843,794 | 3.0240 |
| sparse4_24_chainrule | chainrule | 4/24 | 890,752 | 500 | 633,070 | 2.4463 |
| sparse4_24_mono | mono update every 8 | 4/24 | 890,752 | 63 | 2,465,589 | 3.0240 |

Sparse4/24 matches the dense4 validation CE in this benchmark but does not beat it. This negative result matters: the logical 24-layer scaffold has not yet shown extra quality beyond 4 active blocks.

### 8.3 Softmax vs `simple_gdn` 4-Active

| track | mixer | rule | tok/s | final val CE | peak GB |
|---|---|---|---:|---:|---:|
| dense4_softmax_chainrule | softmax | chainrule | 705,916 | 2.4463 | 1.003 |
| dense4_gdn_chainrule | simple_gdn | chainrule | 580,433 | 2.4878 | 1.379 |
| dense4_softmax_mono | softmax | mono | 2,492,863 | 3.0240 | 1.022 |
| dense4_gdn_mono | simple_gdn | mono | 2,095,307 | 3.0128 | 1.389 |
| sparse4_24_softmax_chainrule | softmax | chainrule | 642,413 | 2.4463 | 1.012 |
| sparse4_24_gdn_chainrule | simple_gdn | chainrule | 549,344 | 2.4878 | 1.379 |
| sparse4_24_softmax_mono | softmax | mono | 2,596,091 | 3.0240 | 1.022 |
| sparse4_24_gdn_mono | simple_gdn | mono | 2,097,410 | 3.0128 | 1.389 |

`simple_gdn` runs stably with finite gradients. It slightly improves mono validation CE but is slower and higher-memory than softmax. It does not improve chain-rule validation CE.

## 9. Discussion

The strongest result is a throughput result, not a quality result. Scheduled mono-forward updates substantially increase measured tokens/sec in these small byte-level smoke runs. The result is practically interesting because it appears on a consumer laptop GPU without custom kernels.

The quality picture is more conservative. Chain-rule reaches lower validation CE in the dense24 short run, and sparse4/24 does not beat dense4 at matched active parameter count. This suggests that the current sparse logical scaffold behaves like a normal 4-active-block model, not like a 24-layer dense model.

The simple recurrent mixer result is also mixed. Stability is useful, but the current implementation is slower and uses more memory than softmax attention. Any future claim about recurrent mixers should wait for a more faithful and optimized implementation.

## 10. Limitations

- Tiny Shakespeare byte-level validation is a smoke benchmark.
- The decoder is small: hidden size 128, byte vocabulary 256.
- Runs are short: 500 training steps.
- FLOPs are estimated from parameter counts, not measured hardware counters.
- Throughput may vary with GPU, drivers, thermals, power settings, and background load.
- Sparse4/24 is not dense24 training.
- `simple_gdn` is not official Gated DeltaNet.
- No real 1.58-bit speedup is claimed in this report.
- No claim is made that these results beat Transformers at scale.
- No LLM-scale quality claim is made.

## 11. Future Work

- Longer equal-token and equal-update runs.
- More careful local-update accounting.
- A more faithful delta-rule recurrent mixer.
- Optimized CUDA/Triton scan or fused recurrence kernels.
- Larger datasets and model widths after the small-scale methodology is stable.

## 12. Reproducibility Checklist

- Repository: `https://github.com/samat2003/samatnext-bit`
- Release tag: `mono-forward-smoke-v1`
- Base release commit: `ecdd51c`
- Tests: `python -m pytest -q`
- Benchmark configs:
  - `configs/chainrule_vs_mono_24layer.yaml`
  - `configs/dense4_vs_sparse4_24.yaml`
  - `configs/softmax_vs_gdn_4active.yaml`
- Dataset path: `data/english_validation.txt`
- Vocab size: 256
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- Environment: Python 3.12.3, PyTorch 2.11.0+cu128, CUDA 12.8, Triton 3.6.0
- Negative results are included in the tables.
