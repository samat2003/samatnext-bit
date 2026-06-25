# RESULTS_DENSE4_VS_SPARSE4_24_v1

Status: benchmark config added; full CUDA benchmark not run by Codex.

## Command

```bash
python -m samatnext_bit.bench_speed --config configs/dense4_vs_sparse4_24.yaml
```

## Experiment

Name: `fp16_dense4_vs_sparse4_24`

Purpose: answer whether logical sparse 4/24 is better than simply training a normal dense 4-layer model at similar active parameter count.

Dataset: Tiny Shakespeare from `data/english_validation.txt`, byte vocab 256, 90/10 train/validation split, fixed validation batches.

Training batches are preloaded on CUDA. Batch sampling/tokenization and validation are excluded from measured training speed.

No GDN. No 1.58-bit/base3. FP16 autocast only.

## Active Quick Tracks

| track | dense_or_sparse | total_layers | active_layers | recurrent_passes | training_rule | update_every | batch | seq | steps |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| dense4_chainrule | dense | 4 | 4 | 1 | chainrule | 1 | 64 | 256 | 500 |
| dense4_mono | dense | 4 | 4 | 1 | mono | 8 | 64 | 256 | 500 |
| sparse4_24_chainrule | sparse/logical | 24 | 4 | 1 | chainrule | 1 | 64 | 256 | 500 |
| sparse4_24_mono | sparse/logical | 24 | 4 | 1 | mono | 8 | 64 | 256 | 500 |

Expected active params are the same for all four tracks: about 890,752 for hidden 128, seq 256, vocab 256, 4 instantiated active blocks.

Logical total params differ in reporting:

- dense4 tracks: 890,752 logical/active params
- sparse4_24 tracks: 4,851,072 logical params, 890,752 active params

## FLOP Estimate Formula

These are estimates, not hardware counters.

- forward FLOPs/token = `2 * active_params`
- chain-rule training FLOPs/token = `6 * active_params`
- mono no-update step FLOPs/token = `2 * active_params`
- mono update step FLOPs/token = `6 * active_params`
- average mono FLOPs/token = `no_update * (N - 1) / N + update_step / N`
- estimated effective TFLOP/s = `measured_tokens_sec * estimated_total_training_flops_per_token / 1e12`

The baseline candidate is `dense4_chainrule`, so comparison fields should be read as speed/FLOP/quality vs dense 4-layer chain-rule.

## Results

Pending manual run.
