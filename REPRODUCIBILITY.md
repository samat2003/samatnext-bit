# Reproducibility

## Environment

Recommended:

- Python 3.10+
- CUDA-capable PyTorch install
- NVIDIA GPU with enough memory for the selected config

Recorded hardware:

- NVIDIA GeForce RTX 5070 Ti Laptop GPU
- 12GB class consumer laptop GPU

Recorded software:

- Python: 3.12.3
- PyTorch: 2.11.0+cu128
- CUDA: 12.8
- Triton: 3.6.0

Release identifiers:

- Repository: `https://github.com/samat2003/samatnext-bit`
- Frozen baseline commit: `ecdd51c`
- Release tag: `mono-forward-smoke-v1`
- Regular PyTorch audit commit: `a69edab`
- Practical Python-token smoke commit: `07b0651`
- BF16 softmax vs simple_gdn commit: `548cadb`
- Dense24 VRAM optimization commit: `4d94d62`
- MBPP dense24 smoke commit: `8ecd6ab`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

## Tests

```bash
python -m pytest -q
```

Current expected result:

```text
21 passed
```

## Tiny Shakespeare Byte-Level Benchmarks

If `data/english_validation.txt` is missing, place Tiny Shakespeare text at that path before running the original validation benchmarks.

```bash
python -m pytest -q
python -m samatnext_bit.bench_speed --config configs/chainrule_vs_mono_24layer.yaml
python -m samatnext_bit.bench_speed --config configs/dense4_vs_sparse4_24.yaml
python -m samatnext_bit.bench_speed --config configs/softmax_vs_gdn_4active.yaml
```

Dataset notes:

- path: `data/english_validation.txt`
- vocab size: 256 byte values
- split: 90% train, 10% validation
- validation batches: fixed by benchmark config and seed

## Local Python-Code Token Benchmarks

Build the generated Python-code corpus and tokenizer:

```bash
python scripts/build_python_code_corpus.py
```

Generated paths:

- `data/python_code_smoke/train.txt`
- `data/python_code_smoke/val.txt`
- `data/python_code_smoke/tokenizer.json`
- `data/python_code_smoke/train_ids.pt`
- `data/python_code_smoke/val_ids.pt`

Run supplemental configs:

```bash
python -m samatnext_bit.bench_speed --config configs/regular_torch_4layer_audit.yaml
python -m samatnext_bit.bench_speed --config configs/practical_python_code_training_smoke.yaml
python -m samatnext_bit.bench_speed --config configs/bf16_softmax_vs_gdn_python_512.yaml
python -m samatnext_bit.bench_speed --config configs/dense24_mono_vram_optimized_python.yaml
```

## MBPP Dense24 Mono 500-Step Smoke

Build the MBPP smoke corpus and tokenizer:

```bash
python scripts/build_mbpp_smoke_corpus.py
```

Generated paths:

- `data/mbpp_smoke/train.txt`
- `data/mbpp_smoke/val.txt`
- `data/mbpp_smoke/metadata.json`
- `data/mbpp_smoke/tokenizer.json`
- `data/mbpp_smoke/train_ids.pt`
- `data/mbpp_smoke/val_ids.pt`

Run the 500-step smoke training config:

```bash
python -m pytest -q
python scripts/build_mbpp_smoke_corpus.py
python -m samatnext_bit.bench_speed --config configs/dense24_mono_optimized_mbpp_500step.yaml
```

Recorded MBPP result:

- branch/commit: `mbpp-dense24-mono-500step-smoke` / `8ecd6ab`
- source: Hugging Face `google-research-datasets/mbpp`
- config: sanitized
- examples: 427 total, 384 train, 43 validation
- train/validation tokens: 62,595 / 7,025
- tokenizer: byte-level BPE, actual vocab size 5,037
- tokenization/corpus building in timed region: false
- CUDA batches preloaded: true

Generated corpora may be ignored by Git unless intentionally committed.

## Expected Approximate Results

| config | track | tok/s | final val CE | peak GB |
|---|---|---:|---:|---:|
| chainrule_vs_mono_24layer | dense24_chainrule | 145,006 | 2.2685 | 7.088 |
| chainrule_vs_mono_24layer | dense24_mono | 663,520 | 3.2379 | 7.155 |
| chainrule_vs_mono_24layer | sparse4_24_mono | 2,556,872 | 3.0240 | 1.022 |
| dense4_vs_sparse4_24 | dense4_chainrule | 661,312 | 2.4463 | ~1.01 |
| dense4_vs_sparse4_24 | dense4_mono | 2,843,794 | 3.0240 | ~1.02 |
| softmax_vs_gdn_4active | dense4_softmax_mono | 2,492,863 | 3.0240 | 1.022 |
| softmax_vs_gdn_4active | dense4_gdn_mono | 2,095,307 | 3.0128 | 1.389 |
| regular_torch_4layer_audit | regular_torch_4layer_python_chainrule | 126,948 | 6.4704 | 10.353 |
| practical_python_code_training_smoke | samatnext_dense4_python_mono | 361,192 | 7.9963 | 11.618 |
| bf16_softmax_vs_gdn_python_512 | dense4_softmax_bf16_mono | 379,367 | 7.2706 | 11.621 |
| bf16_softmax_vs_gdn_python_512 | dense4_gdn_bf16_mono | 79,852 | 7.2734 | 13.828 |
| dense24_mono_vram_optimized_python | dense24_softmax_mono_optimized | 83,587 | 7.8292 | 4.641 |
| dense24_mono_optimized_mbpp_500step | dense24_softmax_mono_optimized_mbpp_500step | 83,613 | 3.9844 | 3.992 |

Exact tok/s can vary by GPU, drivers, thermals, power limits, and background load. Validation CE should be more stable when the same seed, dataset, and config are used.

## Data and Timing Notes

The benchmark uses fixed validation batches and preloaded CUDA training batches. Batch sampling and token loading are not included in measured training speed.

Tokenization and corpus construction are outside timed training loops for Python-code and MBPP token runs.

The FLOP estimates are parameter-count approximations:

- chain-rule: `6 * active_params`
- mono average: `2 * active_params * (N - 1) / N + 6 * active_params / N`

These are not hardware-counter measurements.

## License Notes

Code in this repository is Apache-2.0.

Dataset licenses remain governed by their original sources. MBPP is used only as a small smoke-training corpus and should be attributed to Google Research MBPP.
