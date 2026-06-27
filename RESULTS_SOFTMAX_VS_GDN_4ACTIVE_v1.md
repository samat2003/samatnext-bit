# RESULTS_SOFTMAX_VS_GDN_4ACTIVE_v1

Status: completed on CUDA.

## Command

```bash
python -m pytest -q
python -m samatnext_bit.bench_speed --config configs/softmax_vs_gdn_4active.yaml
```

## Experiment

Name: `fp16_softmax_vs_gdn_4active`

Purpose: compare normal softmax attention against a simple causal GDN-style linear mixer when active compute is limited to 4 blocks.

Dataset: Tiny Shakespeare from `data/english_validation.txt`, byte vocab 256, 90/10 train/validation split, fixed validation batches.

Training batches are preloaded on CUDA. Batch sampling/tokenization and validation are excluded from measured training speed.

No 1.58-bit/base3. FP16 autocast only. Mono-forward scheduled update rule is unchanged.

## Mixer Labels

`simple_gdn` is not official Gated DeltaNet.

- `official_gdn=false`
- `linear_recurrent_mixer=true`
- `mixer_type=simple_gdn`

The current `simple_gdn` implementation is a stable causal gated linear recurrence using cumulative sums, not an optimized kernel.

## Tracks

| track | mixer_type | dense_or_sparse | total_layers | active_layers | training_rule | update_every | batch | seq | steps |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| dense4_softmax_chainrule | softmax | dense | 4 | 4 | chainrule | 1 | 64 | 256 | 500 |
| dense4_gdn_chainrule | simple_gdn | dense | 4 | 4 | chainrule | 1 | 64 | 256 | 500 |
| dense4_softmax_mono | softmax | dense | 4 | 4 | mono | 8 | 64 | 256 | 500 |
| dense4_gdn_mono | simple_gdn | dense | 4 | 4 | mono | 8 | 64 | 256 | 500 |
| sparse4_24_softmax_chainrule | softmax | sparse/logical | 24 | 4 | chainrule | 1 | 64 | 256 | 500 |
| sparse4_24_gdn_chainrule | simple_gdn | sparse/logical | 24 | 4 | chainrule | 1 | 64 | 256 | 500 |
| sparse4_24_softmax_mono | softmax | sparse/logical | 24 | 4 | mono | 8 | 64 | 256 | 500 |
| sparse4_24_gdn_mono | simple_gdn | sparse/logical | 24 | 4 | mono | 8 | 64 | 256 | 500 |

## Results

Result JSON: `runs/softmax_vs_gdn_4active_20260625_140814/speed_results.json`

CUDA device: NVIDIA GeForce RTX 5070 Ti Laptop GPU

Tests: `20 passed in 4.68s`

Benchmark command completed:

```bash
python -m samatnext_bit.bench_speed --config configs/softmax_vs_gdn_4active.yaml
```

| track | mixer | type | rule | updates | tok/s | ms/step | peak GB | final val CE | val ppl | grad mean | grad max | NaN/Inf |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense4_softmax_chainrule | softmax | dense | chainrule | 500 | 705,916 | 23.119 | 1.003 | 2.4463 | 11.55 | 0.303 | 1.605 | false |
| dense4_gdn_chainrule | simple_gdn | dense | chainrule | 500 | 580,433 | 28.117 | 1.379 | 2.4878 | 12.04 | 0.267 | 1.515 | false |
| dense4_softmax_mono | softmax | dense | mono | 63 | 2,492,863 | 6.547 | 1.022 | 3.0240 | 20.57 | 0.937 | 1.586 | false |
| dense4_gdn_mono | simple_gdn | dense | mono | 63 | 2,095,307 | 7.789 | 1.389 | 3.0128 | 20.34 | 0.907 | 1.497 | false |
| sparse4_24_softmax_chainrule | softmax | sparse/logical | chainrule | 500 | 642,413 | 25.404 | 1.012 | 2.4463 | 11.55 | 0.303 | 1.605 | false |
| sparse4_24_gdn_chainrule | simple_gdn | sparse/logical | chainrule | 500 | 549,344 | 29.708 | 1.379 | 2.4878 | 12.04 | 0.267 | 1.515 | false |
| sparse4_24_softmax_mono | softmax | sparse/logical | mono | 63 | 2,596,091 | 6.286 | 1.022 | 3.0240 | 20.57 | 0.937 | 1.586 | false |
| sparse4_24_gdn_mono | simple_gdn | sparse/logical | mono | 63 | 2,097,410 | 7.781 | 1.389 | 3.0128 | 20.34 | 0.907 | 1.497 | false |

Generated samples from prompt `"The "` are included in the JSON for every track.

## Answers

1. Did simple_gdn run stably?
   Yes. All `simple_gdn` tracks completed 500 steps with finite gradients and no NaNs/Infs.
2. Did `simple_gdn` beat softmax in dense4 chainrule validation CE?
   No. Dense4 chainrule softmax ended at 2.4463; simple_gdn ended at 2.4878.
3. Did `simple_gdn` beat softmax in dense4 mono validation CE?
   Yes, slightly. Dense4 mono softmax ended at 3.0240; simple_gdn ended at 3.0128.
4. Did `simple_gdn` beat softmax in sparse4/24 chainrule validation CE?
   No. Sparse4/24 chainrule softmax ended at 2.4463; simple_gdn ended at 2.4878.
5. Did `simple_gdn` beat softmax in sparse4/24 mono validation CE?
   Yes, slightly. Sparse4/24 mono softmax ended at 3.0240; simple_gdn ended at 3.0128.
6. Did `simple_gdn` improve tokens/sec?
   No. `simple_gdn` was slower in all matched comparisons.
7. Did `simple_gdn` use less memory?
   No. `simple_gdn` used about 1.379-1.389 GB peak vs about 1.003-1.022 GB for softmax.
8. Did `simple_gdn` have finite gradients and no NaNs/Infs?
   Yes. All rows report finite gradients and `nan_or_inf=false`.
9. Does sparse4/24 still behave the same as dense4?
   Yes for quality in this implementation. Dense4 and sparse4/24 have identical final validation CE for matched mixer/rule pairs; sparse/logical only changes reported logical total layers/params, not instantiated active blocks.
10. Best honest claim.
   This simple non-official causal linear recurrent mixer is stable. It does not beat softmax under chain-rule and does not improve speed or memory here. It gives a small mono validation CE improvement, but with lower throughput and higher memory.
11. Exact next experiment.
   Implement a more faithful delta-rule mixer with an optimized CUDA/Triton scan or fused recurrence, then rerun only the 4 active-block mono and chain-rule pairs with equal optimizer updates and a longer validation horizon.
