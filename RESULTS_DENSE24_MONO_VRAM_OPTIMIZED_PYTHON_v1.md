# Dense24 Mono VRAM-Optimized Python-Token Smoke Benchmark v1

Status: completed on branch `dense24-mono-vram-optimized-python`.

Main result JSON:

```text
runs/dense24_mono_vram_optimized_python_20260625_150116/speed_results.json
```

Diagnostic result JSON:

```text
runs/dense24_mono_vram_optimized_python_20260625_150025/speed_results.json
```

The diagnostic run used batch 12 at seq 512. It is not used as the main table because the as-is track peaked at 10.195 GB, above the strict 8GB target. It did show the same optimization reducing peak memory to 6.124 GB at batch 12.

## Setup

- Dataset: generated Python source corpus under `data/python_code_smoke`
- Tokenizer: byte-level BPE
- Vocab size: 16,000
- Hidden size: 512
- Heads: 8
- Layers: 24 active layers
- Dense/sparse: dense 24/24, no sparse skipping
- Mixer: softmax attention only
- Precision: FP16 mixed precision
- Optimizer: AdamW
- Training rule: mono-forward scheduled updates, `fp_mono_update_every_8`
- Steps: 100
- Validation checkpoints: 0, 50, 100
- Tokenization/corpus building/validation excluded from training tok/s timing
- CUDA batches preloaded: true
- Token/sec formula: `tokens/sec = batch * (seq - 1) / mean_step_time`

Calibration selected `seq=512`, `batch=8`, with calibration peak memory 5.187 GB. The full as-is run peaked higher than calibration but stayed under the 8GB target.

Note: `oom_fallback_used=true` in the JSON rows reflects auto-selection from the placeholder config batch 4 to calibrated batch 8, not an actual OOM fallback.

## Main Results

| Track | Optimized | Hidden | Layers | Batch | Seq | Update every | LR | Grad clip | Tok/s | Mean ms/step | p50 | p90 | p99 | Peak CUDA GB | Final train CE | Final val CE | Final val ppl | Updates | NaN/Inf |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense24_softmax_mono_as_is | false | 512 | 24 | 8 | 512 | 8 | 3e-4 | none | 82,909 | 49.31 | 29.22 | 175.98 | 200.53 | 7.459 | 7.0817 | 7.8292 | 2512.85 | 13 | false |
| dense24_softmax_mono_optimized | true | 512 | 24 | 8 | 512 | 8 | 3e-4 | none | 83,587 | 48.91 | 29.89 | 174.55 | 184.57 | 4.641 | 7.0817 | 7.8292 | 2512.85 | 13 | false |

Both tracks had finite gradients and finite losses.

## Parameters and FLOPs

These FLOPs are parameter-count estimates, not hardware counters.

| Track | Total params | Active params | Forward FLOPs/token | Training FLOPs/token | Effective TFLOP/s |
|---|---:|---:|---:|---:|---:|
| dense24_softmax_mono_as_is | 92,295,296 | 92,295,296 | 184,590,592 | 230,738,240 | 19.13 |
| dense24_softmax_mono_optimized | 92,295,296 | 92,295,296 | 184,590,592 | 230,738,240 | 19.29 |

Formula:

- Mono: `no_update = 2 * active_params`; `update_step = 6 * active_params`; `avg = no_update * (N - 1) / N + update_step / N`

## What Changed in the Optimized Track

The optimized track keeps the same dense 24-layer softmax architecture and the same mono-forward update schedule. The implementation change is runtime-only:

- Non-update mono steps run under `torch.no_grad()`.
- Update steps still build a normal autograd graph and run `loss.backward()`.
- Optimizer remains AdamW with `zero_grad(set_to_none=True)`.
- No sparse skipping, no GDN, no quantization, no kernel optimization.

The batch-12 diagnostic also tried a conservative `lr=1e-4` and `grad_clip=1.0`; it was stable but learned less over 100 steps, so the strict main run used the same `lr=3e-4` and no clipping for both tracks to isolate the runtime optimization.

## Audit Answers

1. Hidden 512, seq 512, batch 8 fit under 8GB for both main tracks. The as-is track peaked at 7.459 GB; optimized peaked at 4.641 GB.
2. Dense24 at the selected config has 92,295,296 total parameters and 92,295,296 active parameters.
3. As-is peak memory was 7.459 GB and throughput was 82,909 tok/s.
4. Optimized peak memory was 4.641 GB and throughput was 83,587 tok/s.
5. Yes. Optimized mono reduced peak CUDA memory by 2.819 GB, about 37.8% versus as-is.
6. Slightly. Optimized mono improved throughput by about 1.008x in the strict batch-8 run. The larger practical win is memory headroom, not speed.
7. It did not change CE behavior at the same LR/update schedule. It preserved the same train and validation CE while reducing memory.
8. Yes. Validation CE decreased from 9.8404 to 7.8292 over 100 steps in both tracks.
9. The important optimization was avoiding autograd graph construction on non-update mono steps with `torch.no_grad()`.
10. The remaining bottleneck is the dense 24-layer update step itself; update steps still require a full autograd graph through all 24 active softmax blocks.
11. Yes, for this 100-step Python-token smoke test, dense24 mono is practical under an 8GB VRAM target at hidden 512, seq 512, batch 8. It also has usable headroom after optimization at 4.641 GB.
12. Strongest honest claim: dense 24-layer softmax mono-forward training on Python-code tokens can be run under an 8GB laptop-GPU VRAM target by avoiding autograd on non-update mono steps, without changing the model architecture or mono update rule.
13. Main limitation: this is a 100-step pretokenized/preloaded smoke benchmark. It is not a long-run convergence result, not a coding-ability benchmark, and not evidence that dense24 mono matches chain-rule quality.

## Generated Samples

Generated samples from prompt `def add(a, b):` remained noisy after 100 steps. Samples are preserved in the JSON and should not be interpreted as coding ability.

