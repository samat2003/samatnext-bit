# RESULTS_CHAINRULE_VS_MONO_24LAYER_v1

Status: completed on CUDA.

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

Result JSON: `runs/chainrule_vs_mono_24layer_20260625_135234/speed_results.json`

CUDA device: NVIDIA GeForce RTX 5070 Ti Laptop GPU

| track | rule | active params | updates | tok/s | ms/step | peak GB | final val CE | val ppl | est FLOPs/token | NaN/Inf |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense24_chainrule | chainrule | 4,851,072 | 500 | 145,006 | 168.820 | 7.088 | 2.2685 | 9.67 | 29,106,432 | false |
| dense24_mono | mono update every 16 | 4,851,072 | 32 | 663,520 | 36.894 | 7.155 | 3.2379 | 25.48 | 10,914,912 | false |
| sparse4_24_chainrule | chainrule | 890,752 | 500 | 617,245 | 26.440 | 1.012 | 2.4463 | 11.55 | 5,344,512 | false |
| sparse4_24_mono | mono update every 8 | 890,752 | 63 | 2,556,872 | 6.383 | 1.022 | 3.0240 | 20.57 | 2,226,880 | false |

## Answers

- Dense24 mono was 4.58x faster than dense24 chain-rule in tokens/sec.
- Sparse4/24 mono was 17.63x faster than dense24 chain-rule in tokens/sec.
- Dense24 chain-rule reached the best validation CE in this short run.
- Sparse4/24 used only 4 active blocks and 890,752 active params; it is not dense24 training.

## Notes

`dense24_chainrule` at batch 96, seq 256 may OOM depending GPU memory. The config requests fallback to batch 64 and the result row records `requested_batch` and `oom_fallback_used`.
