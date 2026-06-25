# RESULTS_CHAINRULE_VS_MONO_24LAYER_v1

Status: benchmark config and instrumentation added; full CUDA benchmark not run by Codex.

## Command

```bash
python -m samatnext_bit.bench_speed --config configs/chainrule_vs_mono_24layer.yaml
```

## Experiment

Name: `fp16_chainrule_vs_mono_24layer`

Dataset: Tiny Shakespeare from `data/english_validation.txt`, byte vocab 256, 90/10 train/validation split, fixed validation batches.

Training batches are preloaded on CUDA. Batch sampling/tokenization and validation are excluded from measured training speed.

No GDN. No 1.58-bit/base3. FP16 autocast only.

## Active Quick Tracks

| track | dense_or_sparse | total_layers | active_layers | recurrent_passes | training_rule | update_every | batch | seq | steps |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| dense24_chainrule | dense | 24 | 24 | 1 | chainrule | 1 | 96, fallback 64 on OOM | 256 | 500 |
| dense24_mono | dense | 24 | 24 | 1 | mono | 16 | 96 | 256 | 500 |
| sparse4_24_chainrule | sparse/logical | 24 | 4 | 1 | chainrule | 1 | 64 | 256 | 500 |
| sparse4_24_mono | sparse/logical | 24 | 4 | 1 | mono | 8 | 64 | 256 | 500 |

## Optional Manual Fairness Pair

Not enabled in the default config, to keep the default command a quick run.

| track | batch | seq | mode | optimizer updates | steps | tokens/update |
|---|---:|---:|---|---:|---:|---:|
| dense24_mono_equal_tokens_updates | 96 | 256 | fp_mono_update_every_16 | 250 | 4000 | 391,680 |
| sparse4_24_mono_equal_tokens_updates | 64 | 256 | fp_mono_update_every_24 | 250 | 6000 | 391,680 |

## FLOP Estimate Formula

These are estimates, not hardware counters.

- forward FLOPs/token = `2 * active_params`
- chain-rule training FLOPs/token = `6 * active_params`
- mono no-update step FLOPs/token = `2 * active_params`
- mono update step FLOPs/token = `6 * active_params`
- average mono FLOPs/token = `no_update * (N - 1) / N + update_step / N`
- estimated effective TFLOP/s = `measured_tokens_sec * estimated_total_training_flops_per_token / 1e12`

The estimate is intentionally parameter-count based and does not perfectly include attention, softmax, norm, embedding, or optimizer details.

## Results

Pending manual run.

Expected output JSON: `runs/speed_latest.json` and a timestamped `runs/chainrule_vs_mono_24layer_*/speed_results.json`.

The result payload reports:

- total params
- active params
- estimated forward FLOPs/token
- estimated backward/update FLOPs/token
- estimated total training FLOPs/token
- measured tokens/sec
- estimated effective TFLOP/s
- speedup vs `dense24_chainrule`
- FLOP reduction vs `dense24_chainrule`
- quality delta vs `dense24_chainrule`

## Notes

`dense24_chainrule` at batch 96, seq 256 may OOM depending GPU memory. The config requests fallback to batch 64 and the result row records `requested_batch` and `oom_fallback_used`.
