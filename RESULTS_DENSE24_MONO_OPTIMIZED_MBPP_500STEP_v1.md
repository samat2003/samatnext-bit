# Dense24 Optimized Mono MBPP 500-Step Smoke Benchmark v1

Status: completed on branch `mbpp-dense24-mono-500step-smoke`.

Result JSON:

```text
runs/dense24_mono_optimized_mbpp_500step_20260625_151132/speed_results.json
```

Command:

```bash
python -m samatnext_bit.bench_speed --config configs/dense24_mono_optimized_mbpp_500step.yaml
```

This is only a training smoke test on MBPP-style Python problem/solution text. It does not execute tests, does not report pass@1, and does not claim MBPP benchmark performance.

## Dataset

- Source: Hugging Face `google-research-datasets/mbpp`
- Config/split style: `sanitized`
- Splits loaded: `train`, `validation`, `test`, `prompt`
- Examples: 427
- Train examples: 384
- Validation examples: 43
- Raw train bytes: 179,801
- Raw validation bytes: 20,193
- Text format:

```text
# Task:
<task description>

# Solution:
<code>

# Tests:
<tests if available>
```

## Tokenizer

- Type: byte-level BPE
- Requested vocab cap: 8,192
- Actual vocab size: 5,037
- Train tokens: 62,595
- Validation tokens: 7,025
- Total tokens loaded: 69,620

The actual vocab is below 8,192 because the sanitized MBPP smoke corpus is small.

## Model and Training

- Track: `dense24_softmax_mono_optimized_mbpp_500step`
- Dense/sparse: dense
- Layers: 24 total, 24 active
- Mixer: softmax
- Hidden size: 512
- Heads: 8
- Sequence length: 512
- Batch: 8
- Precision: FP16 mixed precision
- Training rule: mono-forward scheduled updates
- Update every: 8
- Optimized non-update steps under `torch.no_grad()`: true
- Optimizer: AdamW
- LR: 3e-4
- Grad clipping: none
- Steps: 500
- Optimizer updates: 63
- CUDA batches preloaded: true
- Tokenization/corpus building/validation in timed region: false

Calibration selected `seq=512`, `batch=8`, with calibration peak memory 4.365 GB.

## Results

| Metric | Value |
|---|---:|
| Total params | 81,058,221 |
| Active params | 81,058,221 |
| Peak CUDA memory | 3.992 GB |
| Tok/s | 83,613 |
| Mean ms/step | 48.89 |
| p50 ms/step | 29.27 |
| p90 ms/step | 179.77 |
| p99 ms/step | 203.91 |
| Initial train CE | 8.6989 |
| Final train CE | 3.7430 |
| Initial val CE | 8.7202 |
| Final val CE | 3.9844 |
| Final val perplexity | 53.7520 |
| Val CE delta | 4.7359 |
| Grad norm mean | 1.4391 |
| Grad norm max | 12.8261 |
| Gradients finite | true |
| NaN/Inf | false |

## CE Checkpoints

| Step | Train CE | Val CE | Val PPL |
|---:|---:|---:|---:|
| 0 | 8.6989 | 8.7202 | 6125.62 |
| 100 | 5.1695 | 5.2489 | 190.35 |
| 200 | 4.5082 | 4.6253 | 102.04 |
| 300 | 4.2374 | 4.3025 | 73.88 |
| 400 | 4.0038 | 4.1089 | 60.88 |
| 500 | 3.7430 | 3.9844 | 53.75 |

## FLOP Estimates

These FLOPs are parameter-count estimates, not hardware counters.

| Metric | Value |
|---|---:|
| Estimated forward FLOPs/token | 162,116,442 |
| Estimated training FLOPs/token | 202,645,552.5 |
| Estimated effective TFLOP/s | 16.94 |

Formula:

- Mono: `no_update = 2 * active_params`; `update_step = 6 * active_params`; `avg = no_update * (N - 1) / N + update_step / N`

## Generated Samples

Prompt: `def add(a, b):`

```text
def add(a, b):
  res = 0 for i == (' = 1 -1OF$')==("0])
add     
      == True

# Solution:
def test_sum(0196 to find theWorst_tup of150 n (9,6), (1), (6,4, 3, 4,3, 5, 3
```

Prompt: `# Task:\nWrite a function`

```text
# Task:
Write a function toal of find_lists( matrix = 1) == m$')==("a-1", 100, 600,3, 1, 1, test_sum(0196, - 6, 11.
Write a function to find_resour_items), (6,4, 3, 4,3, 5, 3, 7, 10)
```

Samples are noisy and are not evidence that the model solves MBPP tasks.

## Audit Answers

1. MBPP source used: Hugging Face `google-research-datasets/mbpp`, sanitized config.
2. 427 examples were used: 384 train and 43 validation.
3. Train/val sizes: 179,801 train bytes and 20,193 validation bytes; 62,595 train tokens and 7,025 validation tokens.
4. Yes. The run stayed under 8GB VRAM, peaking at 3.992 GB.
5. Yes. Validation CE decreased from 8.7202 to 3.9844 over 500 steps.
6. Yes. Gradients remained finite and no NaN/Inf was detected.
7. Throughput remained operational across 500 steps: mean 48.89 ms/step, 83,613 tok/s, with p90 179.77 ms and p99 203.91 ms reflecting scheduled update steps.
8. Parameter size trained: 81,058,221 total and active parameters. This is smaller than the Python-code 16k-vocab dense24 run because the MBPP tokenizer vocab is 5,037.
9. Generated samples looked MBPP-like in surface form but were noisy and invalid; examples are shown above and preserved in the JSON.
10. Strongest honest claim: the optimized dense24 softmax mono-forward path remained stable for a 500-step FP16 training smoke run on MBPP-style Python problem/solution text under an 8GB laptop-GPU VRAM target.
11. Main limitation: this is only a tiny-corpus training smoke test. It does not execute MBPP tests, does not report pass@1, does not show coding ability, and does not establish long-run quality.

