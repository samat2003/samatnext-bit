# Reproducibility

## Environment

Recommended:

- Python 3.10+
- CUDA-capable PyTorch install
- NVIDIA GPU with enough memory for the selected config

Milestone hardware:

- NVIDIA GeForce RTX 5070 Ti Laptop GPU
- 12GB class consumer laptop GPU

Recorded milestone software:

- Python: 3.12.3
- PyTorch: 2.11.0+cu128
- CUDA: 12.8
- Triton: 3.6.0

Release identifiers:

- Repository: `https://github.com/samat2003/samatnext-bit`
- Branch: `master`
- Base release commit: `ecdd51c`
- Release tag: `mono-forward-smoke-v1`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

If `data/english_validation.txt` is missing, place Tiny Shakespeare text at that path before running the validation benchmarks. The milestone runs used `data/english_validation.txt` with 1,115,394 bytes.

Dataset details:

- source: Tiny Shakespeare text
- local path: `data/english_validation.txt`
- vocab size: 256 byte values
- split: 90% train, 10% validation
- validation batches: fixed by benchmark config and seed

## Tests

```bash
python -m pytest -q
```

Expected for this release:

```text
20 passed
```

## Benchmarks

```bash
python -m pytest -q
python -m samatnext_bit.bench_speed --config configs/chainrule_vs_mono_24layer.yaml
python -m samatnext_bit.bench_speed --config configs/dense4_vs_sparse4_24.yaml
python -m samatnext_bit.bench_speed --config configs/softmax_vs_gdn_4active.yaml
```

Each run writes a timestamped JSON file under `runs/` and updates `runs/speed_latest.json`. The `runs/` directory is ignored by Git.

## Expected Approximate Results

| config | track | tok/s | final val CE |
|---|---|---:|---:|
| chainrule_vs_mono_24layer | dense24_chainrule | 145,006 | 2.2685 |
| chainrule_vs_mono_24layer | dense24_mono | 663,520 | 3.2379 |
| chainrule_vs_mono_24layer | sparse4_24_chainrule | 617,245 | 2.4463 |
| chainrule_vs_mono_24layer | sparse4_24_mono | 2,556,872 | 3.0240 |
| dense4_vs_sparse4_24 | dense4_chainrule | 661,312 | 2.4463 |
| dense4_vs_sparse4_24 | dense4_mono | 2,843,794 | 3.0240 |
| dense4_vs_sparse4_24 | sparse4_24_chainrule | 633,070 | 2.4463 |
| dense4_vs_sparse4_24 | sparse4_24_mono | 2,465,589 | 3.0240 |
| softmax_vs_gdn_4active | dense4_softmax_chainrule | 705,916 | 2.4463 |
| softmax_vs_gdn_4active | dense4_gdn_chainrule | 580,433 | 2.4878 |
| softmax_vs_gdn_4active | dense4_softmax_mono | 2,492,863 | 3.0240 |
| softmax_vs_gdn_4active | dense4_gdn_mono | 2,095,307 | 3.0128 |
| softmax_vs_gdn_4active | sparse4_24_softmax_chainrule | 642,413 | 2.4463 |
| softmax_vs_gdn_4active | sparse4_24_gdn_chainrule | 549,344 | 2.4878 |
| softmax_vs_gdn_4active | sparse4_24_softmax_mono | 2,596,091 | 3.0240 |
| softmax_vs_gdn_4active | sparse4_24_gdn_mono | 2,097,410 | 3.0128 |

Exact tok/s can vary by GPU, drivers, thermals, power limits, and background load. Validation CE should be much more stable when the same seed, dataset, and config are used.

## Data and Timing Notes

The benchmark uses fixed validation batches and preloaded CUDA training batches. Batch sampling and token loading are not included in measured training speed.

The FLOP estimates are parameter-count approximations:

- chain-rule: `6 * active_params`
- mono average: `2 * active_params * (N - 1) / N + 6 * active_params / N`

These are not hardware-counter measurements.
