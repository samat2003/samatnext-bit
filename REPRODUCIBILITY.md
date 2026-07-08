# Reproducibility

## Environment

Recommended:

- Python 3.10+
- CUDA-capable PyTorch install
- NVIDIA GPU with enough memory for the selected config

Recorded audit environment:

- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- Python: 3.12.3
- PyTorch: 2.11.0+cu128
- CUDA: 12.8
- Triton: 3.6.0

Exact throughput varies with GPU model, drivers, power limits, thermals, and background load.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
python -m pytest -q
```

## Fast Synthetic Smoke

No external dataset is required.

```bash
python -m samatnext_bit.bench_speed --config configs/speed_500k_tiny.yaml
```

Expected historical class: about `3.5M tok/s`.

This uses under 1M active parameters and static/generated CUDA batches. It is not comparable to dense LLM training.

## Dense Python HF Mix Audit

The dense audit expects pretokenized data. Raw corpora and large token files are not included.

Expected layout:

```text
data/generated/python_hf_mix_2p5b/
  train.bin
  val.bin
  metadata.json
  tokenizer.json
```

Run:

```bash
python scripts/bench_dense313m_loss_recovery.py --config configs/dense313m_python_hf_mix_quality_1500.yaml
python scripts/bench_dense313m_loss_recovery.py --config configs/dense313m_python_hf_mix_1500.yaml
```

The harness uses AMP fp16, fused AdamW when available, and PyTorch SDPA with Flash-capable settings.

## 1500-Step Reference Results

| Track | Final val CE | Tok/s | CE/min | Note |
|---|---:|---:|---:|---|
| chain-rule | 5.4794 | 5,074 | 0.9968 | baseline |
| mono UE1 quality | 5.4658 | 5,343 | 1.0528 | conservative schedule |
| mono UE4 speed | 6.0868 | 15,816 | 2.7024 | worse final CE, faster class |

These are single-run audit numbers. Treat them as reproducibility targets, not general claims.

## Custom Dataset

Use synthetic data only for speed-path testing. Use your own tokenized dataset for meaningful loss comparisons.

Supported pretokenized layout:

```text
your_dataset/
  train.bin
  val.bin
  metadata.json
  tokenizer.json
```

The `.bin` files should contain integer token IDs using the dtype declared in `metadata.json`, usually `uint16` for vocabularies up to 65,536. Include tokenizer metadata, token counts, vocabulary size, and source-corpus notes in `metadata.json`.

Point a config at the directory:

```yaml
dataset_path: path/to/your_dataset
train_bin: train.bin
val_bin: val.bin
metadata_file: metadata.json
tokenizer_file: tokenizer.json
```

External datasets, tokenizers, and pretrained models keep their original licenses.

## Output Files

Benchmark runs write JSON under `runs/`. Small `results.json` and `*_latest.json` audit artifacts may be kept. Raw data directories and large binary arrays must not be committed.
