# RESULTS_REGULAR_TORCH_4LAYER_AUDIT_v1

Status: completed on CUDA.

## Purpose

Answer the reviewer question: is the 4-layer baseline speed high because the benchmark uses a custom SamatNext path, or would a plain regular PyTorch CUDA Transformer-style 4-layer chain-rule loop reach similar speed under the same tiny hidden-128 byte-level setup?

## Command

```bash
python -m pytest -q
python -m samatnext_bit.bench_speed --config configs/regular_torch_4layer_audit.yaml
```

## Setup

- Dataset: Tiny Shakespeare at `data/english_validation.txt`
- Vocab size: 256 byte values
- Batch: 64
- Seq: 256
- Hidden: 128
- Heads: 4
- Steps: 500
- Validation: every 100 steps
- AMP: CUDA fp16 autocast
- Token/sec formula: `tokens/sec = batch * (seq - 1) / mean_step_time`
- CUDA batches preloaded: true
- Dataloader/token sampling inside timed region: false

## Tracks

| track | model_family | training_rule | layers | total_layers | active_layers |
|---|---|---|---:|---:|---:|
| regular_torch_4layer_chainrule | regular_torch_transformer | chainrule | 4 | 4 | 4 |
| samatnext_dense4_softmax_chainrule | samatnext | chainrule | 4 | 4 | 4 |
| samatnext_dense4_softmax_mono | samatnext | mono update every 8 | 4 | 4 | 4 |
| samatnext_sparse4_24_softmax_mono | samatnext | mono update every 8 | 4 active / 24 logical | 24 | 4 |

## Results

Result JSON: `runs/regular_torch_4layer_audit_20260625_143314/speed_results.json`

CUDA device: NVIDIA GeForce RTX 5070 Ti Laptop GPU

| track | model_family | rule | total_layers | active_layers | hidden | heads | batch | seq | total params | active params | updates | tok/s | mean ms | p50 | p90 | p99 | peak GB | final train CE | final val CE | val ppl | grad mean | grad max | NaN/Inf | preloaded CUDA batches | token sampling timed |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| regular_torch_4layer_chainrule | regular_torch_transformer | chainrule | 4 | 4 | 128 | 4 | 64 | 256 | 891,904 | 891,904 | 500 | 629,540 | 25.924 | 24.925 | 28.783 | 34.161 | 0.853 | 2.4339 | 2.4464 | 11.55 | 0.300 | 1.611 | false | true | false |
| samatnext_dense4_softmax_chainrule | samatnext | chainrule | 4 | 4 | 128 | 4 | 64 | 256 | 890,752 | 890,752 | 500 | 649,496 | 25.127 | 25.041 | 27.887 | 30.541 | 1.012 | 2.4361 | 2.4463 | 11.55 | 0.303 | 1.605 | false | true | false |
| samatnext_dense4_softmax_mono | samatnext | mono update every 8 | 4 | 4 | 128 | 4 | 64 | 256 | 890,752 | 890,752 | 63 | 2,515,694 | 6.487 | 3.434 | 26.511 | 30.755 | 1.022 | 3.0019 | 3.0240 | 20.57 | 0.937 | 1.586 | false | true | false |
| samatnext_sparse4_24_softmax_mono | samatnext | mono update every 8 | 24 | 4 | 128 | 4 | 64 | 256 | 4,851,072 | 890,752 | 63 | 2,714,659 | 6.012 | 3.211 | 22.627 | 29.500 | 1.022 | 3.0019 | 3.0240 | 20.57 | 0.937 | 1.586 | false | true | false |

Token/sec formula for every row: `tokens/sec = batch * (seq - 1) / mean_step_time`.

All rows use preloaded CUDA batches. Dataloader/token sampling is not inside the timed region.

## Audit Questions

1. Why can a tiny 4-layer chain-rule model reach hundreds of thousands tok/s?
   Because this is a very small hidden-128 byte-level model with only 4 transformer blocks, batch 64, seq 256, CUDA fp16 autocast, SDPA attention, and preloaded CUDA batches. The timed loop excludes token loading, dataloader work, and validation. Under those conditions, hundreds of thousands tok/s is plausible for regular PyTorch as well.
2. Was the previous 90K tok/s expectation from a larger/different model or a less controlled end-to-end loop?
   Most likely yes. This audit shows a plain regular PyTorch 4-layer chain-rule loop reaches 629,540 tok/s in the controlled tiny setup. A 90K tok/s expectation likely reflects a larger model, more layers/hidden size, slower end-to-end data path, different hardware, or less controlled timing.
3. Does regular PyTorch 4-layer chain-rule match current SamatNext dense4 chain-rule speed?
   Yes, within a small margin. Regular PyTorch measured 629,540 tok/s; SamatNext dense4 chain-rule measured 649,496 tok/s, about 1.03x regular PyTorch.
4. How much faster is SamatNext dense4 mono than regular PyTorch 4-layer chain-rule?
   SamatNext dense4 mono measured 2,515,694 tok/s, about 4.00x regular PyTorch 4-layer chain-rule.
5. How much faster is SamatNext sparse4/24 mono than regular PyTorch 4-layer chain-rule?
   SamatNext sparse4/24 mono measured 2,714,659 tok/s, about 4.31x regular PyTorch 4-layer chain-rule.
6. Is the speedup mainly from mono-forward scheduling, sparse active compute, or benchmark setup?
   For the 4-active-block audit, the main speedup is mono-forward scheduling: dense4 mono is about 4.00x regular chain-rule. Sparse4/24 mono is only modestly faster than dense4 mono in this run, and it uses the same 4 active blocks. The benchmark setup explains why both regular and SamatNext chain-rule 4-layer baselines are already fast.
7. Best honest claim.
   The high 4-layer chain-rule speed is not a custom SamatNext artifact; a plain regular PyTorch CUDA 4-layer Transformer-style baseline reaches similar speed under the same tiny controlled setup. Mono-forward scheduling still provides roughly 4x throughput over that regular chain-rule baseline, with fewer optimizer updates and worse final validation CE in this short run.
8. Main limitation.
   This is still a small hidden-128 byte-level Tiny Shakespeare smoke benchmark with preloaded CUDA batches and timing that excludes data loading and validation. It does not establish end-to-end application throughput, LLM-scale behavior, or quality superiority.
