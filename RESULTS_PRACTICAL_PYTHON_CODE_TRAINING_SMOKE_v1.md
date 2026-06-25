# RESULTS_PRACTICAL_PYTHON_CODE_TRAINING_SMOKE_v1

Status: completed on CUDA.

## Purpose

Practical CUDA FP16 Python-code token smoke test comparing a plain regular PyTorch 4-layer Transformer chain-rule baseline against SamatNext 4-active softmax chain-rule and mono-forward tracks.

No GDN. No base3/1.58-bit. No kernel optimization. No coding benchmark or coding-ability claim.

## Commands

```bash
python -m pytest -q
python scripts/build_python_code_corpus.py
python -m samatnext_bit.bench_speed --config configs/practical_python_code_training_smoke.yaml
```

## Timing Rules

- Pre-tokenized: true
- Tokenization in timed region: false
- CUDA batches preloaded: recorded per row
- Dataloader/token sampling in timed region: false
- Token/sec formula: `tokens/sec = batch * (seq - 1) / mean_step_time`

## Results

Result JSON: `runs/practical_python_code_training_smoke_20260625_144354/speed_results.json`

CUDA device: NVIDIA GeForce RTX 5070 Ti Laptop GPU

Calibration selected `seq=1024`, `batch=24`. Calibration tested regular PyTorch chain-rule at seq 1024 with batch 16 and 24; batch 24 had a one-step calibration peak of 7.798 GB. In the full 100-step benchmark, regular PyTorch chain-rule peaked at 10.353 GB, while SamatNext tracks with the same batch/seq peaked at 11.258-11.618 GB. No OOM occurred, but the SamatNext rows exceeded the approximate 10.5 GB target.

| track | model_family | rule | hidden | heads | total/active layers | vocab | batch | seq | params | updates | tok/s | mean ms | p50 | p90 | p99 | peak GB | final train CE | final val CE | val ppl | est train FLOPs/token | eff TFLOP/s | grad mean/max | NaN/Inf |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| regular_torch_4layer_python_chainrule | regular_torch_transformer | chainrule | 512 | 8 | 4/4 | 16,000 | 24 | 1024 | 29,534,848 | 100 | 126,948 | 193.402 | 192.054 | 208.536 | 290.566 | 10.353 | 6.2679 | 6.4704 | 645.72 | 177,209,088 | 22.50 | 1.037/1.808 | false |
| samatnext_dense4_python_chainrule | samatnext | chainrule | 512 | 8 | 4/4 | 16,000 | 24 | 1024 | 29,530,240 | 100 | 142,473 | 172.327 | 171.401 | 178.444 | 186.928 | 11.258 | 6.2587 | 6.4563 | 636.72 | 177,181,440 | 25.24 | 1.024/1.802 | false |
| samatnext_dense4_python_mono | samatnext | mono update every 8 | 512 | 8 | 4/4 | 16,000 | 24 | 1024 | 29,530,240 | 13 | 361,192 | 67.975 | 52.076 | 172.299 | 194.715 | 11.618 | 7.8966 | 7.9963 | 2,970.02 | 73,825,600 | 26.67 | 1.453/1.775 | false |
| samatnext_sparse4_24_python_mono | samatnext | mono update every 8 | 512 | 8 | 24/4 | 16,000 | 24 | 1024 | 92,557,440 logical / 29,530,240 active | 13 | 362,131 | 67.799 | 51.721 | 168.095 | 196.069 | 11.618 | 7.8966 | 7.9963 | 2,970.02 | 73,825,600 | 26.73 | 1.453/1.775 | false |

Generated samples from prompt `def add(a, b):` were poor/noisy after only 100 steps. They are included in the JSON and should not be interpreted as coding ability.

## Audit Questions

1. What batch/seq/hidden/vocab were selected?
   Batch 24, sequence length 1024, hidden size 512, 8 heads, and BPE vocab size 16,000.
2. How much GPU memory was used?
   Regular PyTorch chain-rule peaked at 10.353 GB. SamatNext dense4 chain-rule peaked at 11.258 GB. SamatNext mono tracks peaked at 11.618 GB.
3. Did the benchmark use enough GPU memory to be a practical training smoke test?
   Yes. It used roughly 10.4-11.6 GB peak, which meaningfully exercises a 12GB-class laptop GPU. The SamatNext rows exceeded the approximate 10.5 GB target but did not OOM.
4. Did the regular PyTorch baseline train stably?
   Yes. It completed 100 steps with finite gradients, no NaNs/Infs, and validation CE improved from 9.8383 to 6.4704.
5. Did SamatNext dense4 chain-rule match regular PyTorch chain-rule speed?
   Yes. SamatNext dense4 chain-rule measured 142,473 tok/s vs 126,948 tok/s for regular PyTorch, about 1.12x faster in this run.
6. How much faster/slower was SamatNext dense4 mono than regular PyTorch chain-rule?
   SamatNext dense4 mono measured 361,192 tok/s, about 2.85x faster than regular PyTorch chain-rule.
7. How much faster/slower was SamatNext sparse4/24 mono than regular PyTorch chain-rule?
   SamatNext sparse4/24 mono measured 362,131 tok/s, about 2.85x faster than regular PyTorch chain-rule.
8. Did mono-forward reduce training/validation loss on Python tokens?
   Yes. Dense4 mono training CE improved from 9.8338 to 7.8966 and validation CE improved from 9.8391 to 7.9963 over 100 steps.
9. Did chain-rule reach lower CE after the same number of steps?
   Yes. Regular PyTorch chain-rule ended at validation CE 6.4704 and SamatNext dense4 chain-rule ended at 6.4563, both much lower than mono validation CE 7.9963.
10. Did larger vocab/hidden/seq reduce the extreme byte-level tok/s numbers?
   Yes. Practical Python-token speeds are much lower than the tiny hidden-128 byte-level smoke numbers because hidden size, vocab size, sequence length, and LM-head/attention cost are much larger.
11. What were estimated FLOPs/token and effective TFLOP/s?
   Chain-rule estimated training FLOPs/token were about 177.2M, with effective 22.50 TFLOP/s for regular PyTorch and 25.24 TFLOP/s for SamatNext dense4. Mono estimated training FLOPs/token were about 73.8M, with effective 26.67-26.73 TFLOP/s.
12. What is the strongest honest claim?
   On a practical hidden-512, seq-1024, vocab-16k Python-token smoke setup, mono-forward still improved measured training throughput by about 2.85x over regular PyTorch chain-rule, but with fewer optimizer updates and worse short-run validation CE.
13. What is the main limitation?
   This is a 100-step smoke test on a generated Python source corpus, not a coding benchmark. It uses pretokenized/preloaded batches and does not show coding ability, long-run convergence, or end-to-end data pipeline throughput.
