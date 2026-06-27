# Reproducibility Checklist

- Repository: `https://github.com/samat2003/samatnext-bit`
- Branch: `master`
- Base release commit: `ecdd51c`
- Release tag: `mono-forward-smoke-v1`
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- GPU class: 12GB consumer laptop GPU
- Python: 3.12.3
- PyTorch: 2.11.0+cu128
- CUDA: 12.8
- Triton: 3.6.0
- Dataset: Tiny Shakespeare byte-level text
- Dataset path: `data/english_validation.txt`
- Dataset bytes in milestone run: 1,115,394
- Vocab size: 256
- Train/validation split: 90/10
- Tests:

```bash
python -m pytest -q
```

- Benchmarks:

```bash
python -m samatnext_bit.bench_speed --config configs/chainrule_vs_mono_24layer.yaml
python -m samatnext_bit.bench_speed --config configs/dense4_vs_sparse4_24.yaml
python -m samatnext_bit.bench_speed --config configs/softmax_vs_gdn_4active.yaml
```

- Expected approximate results are listed in `REPRODUCIBILITY.md` and `RESULTS_SUMMARY.md`.
- Throughput can vary with GPU model, driver version, thermals, power settings, and background load.
- FLOPs are parameter-count estimates, not hardware counters.
- Negative results are reported, including chain-rule quality advantages, sparse4/24 matching dense4, and `simple_gdn` being slower/higher-memory than softmax.
