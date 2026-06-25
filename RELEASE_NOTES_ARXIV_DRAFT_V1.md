# Release Notes: arXiv Draft v1

This documentation integration prepares `samatnext-bit` for a modest arXiv-style technical report after the frozen `mono-forward-smoke-v1` release.

The frozen tag `mono-forward-smoke-v1` is not modified.

## What Changed Since `mono-forward-smoke-v1`

Supplemental results integrated into the public documentation:

- Regular PyTorch 4-layer baseline audit.
- Practical hidden-512 Python-token smoke benchmark.
- BF16 softmax vs non-official `simple_gdn` negative result.
- Dense24 mono-forward VRAM optimization under an 8GB target.
- MBPP dense24 mono 500-step smoke training.

Documentation updates:

- README updated for GitHub visitors.
- Results summary expanded with supplemental tables.
- Claims and limitations updated with MBPP caveats.
- Reproducibility commands added for generated Python-code and MBPP corpora.
- Paper draft expanded into an arXiv-style technical report.
- Repository code license changed to Apache-2.0.

## Strongest Honest Claim

The optimized dense24 softmax mono-forward path remained stable for a 500-step FP16 smoke-training run on sanitized MBPP problem/solution text with:

- 81.1M parameters
- dense 24/24 active blocks
- 83.6K tok/s
- 3.992GB peak CUDA memory
- validation CE decreasing from 8.7202 to 3.9844

This is a training-rule throughput and memory smoke result on a consumer laptop GPU.

## Main Limitations

- Sanitized MBPP is tiny in this setup: 384 train examples and 62,595 train tokens.
- The 500-step run repeats a very small corpus.
- Generated samples remained noisy and invalid-looking.
- No MBPP tests were executed.
- No pass@1 is reported.
- No coding ability claim is made.
- No LLM-scale claim is made.
- No claim is made that this beats Transformers at scale.
- `simple_gdn` is not official Gated DeltaNet and was not a practical win.

## Reproduction Commands

Original release:

```bash
python -m pytest -q
python -m samatnext_bit.bench_speed --config configs/chainrule_vs_mono_24layer.yaml
python -m samatnext_bit.bench_speed --config configs/dense4_vs_sparse4_24.yaml
python -m samatnext_bit.bench_speed --config configs/softmax_vs_gdn_4active.yaml
```

Supplemental Python-token and MBPP smoke runs:

```bash
python scripts/build_python_code_corpus.py
python -m samatnext_bit.bench_speed --config configs/regular_torch_4layer_audit.yaml
python -m samatnext_bit.bench_speed --config configs/practical_python_code_training_smoke.yaml
python -m samatnext_bit.bench_speed --config configs/bf16_softmax_vs_gdn_python_512.yaml
python -m samatnext_bit.bench_speed --config configs/dense24_mono_vram_optimized_python.yaml

python scripts/build_mbpp_smoke_corpus.py
python -m samatnext_bit.bench_speed --config configs/dense24_mono_optimized_mbpp_500step.yaml
```

## License

Code in this repository is Apache-2.0. Dataset licenses remain governed by their original sources. MBPP is used only as a small smoke-training corpus and should be attributed to Google Research MBPP.
