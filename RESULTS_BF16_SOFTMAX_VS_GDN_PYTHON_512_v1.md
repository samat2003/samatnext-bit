# BF16 Softmax vs Simple GDN Python-Token Smoke Benchmark v1

Status: completed on branch `bf16-softmax-vs-gdn-python-512`.

Result JSON:

```text
runs/bf16_softmax_vs_gdn_python_512_20260625_145225/speed_results.json
```

Command:

```bash
python -m samatnext_bit.bench_speed --config configs/bf16_softmax_vs_gdn_python_512.yaml
```

BF16 support check:

```text
torch.cuda.is_available(): True
torch.cuda.get_device_name(): NVIDIA GeForce RTX 5070 Ti Laptop GPU
torch.cuda.is_bf16_supported(): True
```

## Setup

- Dataset: generated Python source corpus under `data/python_code_smoke`
- Tokenizer: byte-level BPE
- Vocab size: 16,000
- Hidden size: 512
- Heads: 8
- Layers: 4 active layers
- Precision: BF16 mixed precision
- Optimizer: AdamW
- Steps: 100
- Validation checkpoints: 0, 50, 100
- Timed region excludes corpus building, tokenizer training, tokenization, and validation
- CUDA batches preloaded: true
- Dataloader/token sampling in timed region: false
- Token/sec formula: `tokens/sec = batch * (seq - 1) / mean_step_time`

Calibration selected `seq=1024`, `batch=24`, with calibration peak memory 7.798 GB on the regular 4-layer chain-rule probe. The full SamatNext tracks used more memory than the calibration probe, especially `simple_gdn`; no track OOMed.

Note: `oom_fallback_used=true` in the JSON rows reflects auto-selection from the placeholder config batch 16 to calibrated batch 24, not an actual OOM fallback.

## Results

| Track | Mixer | Rule | BF16 | Tok/s | Mean ms/step | Peak CUDA GB | Final train CE | Final val CE | Final val ppl | Updates | Grad finite | NaN/Inf |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| dense4_softmax_bf16_chainrule | softmax | chainrule | yes | 137,322 | 178.79 | 11.260 | 5.1601 | 5.4315 | 228.50 | 100 | true | false |
| dense4_gdn_bf16_chainrule | simple_gdn | chainrule | yes | 61,406 | 399.83 | 13.466 | 5.1868 | 5.4542 | 233.73 | 100 | true | false |
| dense4_softmax_bf16_mono | softmax | mono every 8 | yes | 379,367 | 64.72 | 11.621 | 7.1519 | 7.2706 | 1437.39 | 13 | true | false |
| dense4_gdn_bf16_mono | simple_gdn | mono every 8 | yes | 79,852 | 307.47 | 13.828 | 7.1497 | 7.2734 | 1441.46 | 13 | true | false |

## FLOP Estimates

These are parameter-count estimates, not hardware counters. They do not perfectly include every attention, softmax, norm, or optimizer detail.

| Track | Active params | Forward FLOPs/token | Training FLOPs/token | Effective TFLOP/s |
|---|---:|---:|---:|---:|
| dense4_softmax_bf16_chainrule | 29,530,240 | 59,060,480 | 177,181,440 | 24.33 |
| dense4_gdn_bf16_chainrule | 29,530,240 | 59,060,480 | 177,181,440 | 10.88 |
| dense4_softmax_bf16_mono | 29,530,240 | 59,060,480 | 73,825,600 | 28.01 |
| dense4_gdn_bf16_mono | 29,530,240 | 59,060,480 | 73,825,600 | 5.90 |

Formulas:

- Chain-rule: `forward = 2 * active_params`; `training = 6 * active_params`
- Mono: `no_update = 2 * active_params`; `update_step = 6 * active_params`; `avg = no_update * (N - 1) / N + update_step / N`

## Audit Answers

1. BF16 is supported on this GPU/runtime: CUDA is available, GPU is NVIDIA GeForce RTX 5070 Ti Laptop GPU, and `torch.cuda.is_bf16_supported()` returned true.
2. Selected setup: batch 24, seq 1024, vocab 16,000, hidden 512, heads 8.
3. BF16 softmax trained stably: finite loss/gradients and no NaNs/Infs in both chain-rule and mono tracks.
4. BF16 `simple_gdn` trained stably: finite loss/gradients and no NaNs/Infs in both chain-rule and mono tracks.
5. `simple_gdn` did not beat softmax in chain-rule validation CE: 5.4542 vs softmax 5.4315.
6. `simple_gdn` did not beat softmax in mono validation CE: 7.2734 vs softmax 7.2706.
7. `simple_gdn` did not improve tokens/sec: it was 0.45x the softmax chain-rule speed and 0.21x the softmax mono speed.
8. `simple_gdn` did not use less memory: it used about 13.47-13.83 GB versus 11.26-11.62 GB for softmax.
9. Compared with the FP16 practical Python smoke, BF16 did not materially reduce memory. BF16 softmax chain-rule was slightly slower than FP16 SamatNext chain-rule, while BF16 softmax mono was slightly faster in this run.
10. Mono-forward still improved throughput in BF16: softmax mono was 2.76x faster than softmax chain-rule.
11. Chain-rule still reached lower CE than mono after the same 100 steps: softmax chain-rule final val CE 5.4315 versus softmax mono 7.2706.
12. Best honest claim: in this hidden-512 Python-token BF16 smoke test, `simple_gdn` was stable but softmax remained the stronger practical mixer for speed, memory, and short-run validation CE; mono-forward preserved a throughput advantage but learned less per step than chain-rule.
13. Main limitation: this is a 100-step pretokenized/preloaded smoke benchmark with a simple non-official recurrent mixer. It is not evidence of coding ability, not a long-run convergence result, and not an official Gated DeltaNet comparison.

## Generated Samples

Generated samples from prompt `def add(a, b):` were noisy after 100 steps for all tracks. They are preserved in the result JSON and should not be interpreted as coding ability.

