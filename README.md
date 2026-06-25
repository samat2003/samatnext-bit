# SamatNext-Bit

SamatNext-Bit is a CUDA/PyTorch research artifact for smoke benchmarks of mono-forward scheduled updates in small decoder models on a consumer laptop GPU. Mono-forward is treated here as a training rule, not a new architecture: the model still uses decoder blocks, causal softmax attention unless otherwise stated, and standard next-token cross entropy. The repository is prepared for a modest arXiv-style technical report, with negative results and limitations kept visible.

This is not an LLM-scale claim, not an MBPP pass@1 result, and not a claim that the method beats Transformers at scale.

## Release and Draft State

- Public repository: `https://github.com/samat2003/samatnext-bit`
- Frozen baseline tag: `mono-forward-smoke-v1`
- Frozen baseline commit: `ecdd51c`
- ArXiv draft integration branch: `arxiv-paper-docs-mbpp-integration`
- Paper draft: [paper/main.md](paper/main.md)
- Code license: Apache-2.0

The `mono-forward-smoke-v1` tag remains the frozen public baseline release. Later branches add supplemental audits and smoke-training results for the paper draft without changing the frozen tag.

## Hardware and Software

Recorded runs used:

- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- GPU class: 12GB consumer laptop GPU
- Python: 3.12.3
- PyTorch: 2.11.0+cu128
- CUDA: 12.8
- Triton: 3.6.0

Throughput varies with GPU model, drivers, thermals, laptop power mode, and background load.

## Model and Training Terms

- `fp_chainrule`: regular full final-cross-entropy backprop every step.
- `fp_mono_update_every_N`: mono-forward scheduled update rule. It forwards every step and performs optimizer updates every `N` steps.
- `dense24`: 24 active blocks out of 24.
- `dense4`: normal 4-layer model.
- `sparse4/24`: logical 24-layer scaffold with only 4 active/instantiated blocks. It must not be described as dense24 training.
- `softmax`: normal causal softmax attention mixer.
- `simple_gdn`: experimental simple recurrent/linear mixer, labeled `official_gdn=false`. It is not official Gated DeltaNet and was not a practical win in the current results.

## Datasets

The repository uses three smoke-benchmark corpora:

- Tiny Shakespeare byte-level smoke benchmark at `data/english_validation.txt`, vocab size 256.
- Local Python-code token smoke benchmark built from repo Python files, optional `PYTHON_CORPUS_DIR`, and Python stdlib if discoverable.
- MBPP sanitized smoke-training corpus from `google-research-datasets/mbpp`, used only as Python problem/solution text.

Generated corpora and tokenizers under `data/python_code_smoke/` and `data/mbpp_smoke/` are ignored by Git. Tokenization and corpus building are outside timed training loops.

## Summary Results

### Original Tiny Shakespeare Smoke Release

| Experiment | Track | Tok/s | Final val CE | Peak GB | Notes |
|---|---|---:|---:|---:|---|
| chain-rule vs mono | dense24_chainrule | 145,006 | 2.2685 | 7.088 | best short-run CE |
| chain-rule vs mono | dense24_mono | 663,520 | 3.2379 | 7.155 | 4.58x dense24 chain-rule tok/s |
| chain-rule vs mono | sparse4_24_mono | 2,556,872 | 3.0240 | 1.022 | 4 active blocks only |
| dense4 vs sparse4/24 | dense4_chainrule | 661,312 | 2.4463 | ~1.01 | matched sparse4/24 CE |
| dense4 vs sparse4/24 | sparse4_24_chainrule | 633,070 | 2.4463 | ~1.01 | did not beat dense4 |
| softmax vs simple_gdn | dense4_softmax_mono | 2,492,863 | 3.0240 | 1.022 | softmax faster/lower memory |
| softmax vs simple_gdn | dense4_gdn_mono | 2,095,307 | 3.0128 | 1.389 | stable but slower/higher memory |

### Supplemental Python-Token and MBPP Smoke Runs

| Experiment | Track | Tok/s | Final val CE | Peak GB | Notes |
|---|---|---:|---:|---:|---|
| regular PyTorch audit | regular_torch_4layer_python_chainrule | 126,948 | 6.4704 | 10.353 | hidden 512, seq 1024, vocab 16k |
| practical Python smoke | samatnext_dense4_python_mono | 361,192 | 7.9963 | 11.618 | 2.85x regular PyTorch chain-rule tok/s |
| BF16 mixer smoke | dense4_softmax_bf16_mono | 379,367 | 7.2706 | 11.621 | BF16 softmax stable |
| BF16 mixer smoke | dense4_gdn_bf16_mono | 79,852 | 7.2734 | 13.828 | `simple_gdn` slower and higher memory |
| dense24 VRAM optimization | dense24_softmax_mono_optimized | 83,587 | 7.8292 | 4.641 | hidden 512, seq 512, batch 8 |
| MBPP smoke training | dense24_softmax_mono_optimized_mbpp_500step | 83,613 | 3.9844 | 3.992 | 81.1M params, 500 steps |

The strongest practical supplemental result is the optimized dense24 MBPP smoke run: 81.1M parameters, dense 24/24 softmax, 500 FP16 steps, 83.6K tok/s, 3.992GB peak CUDA memory, and validation CE decreasing from 8.7202 to 3.9844. Because sanitized MBPP is tiny, this is a stability and efficiency smoke test, not a coding benchmark score.

## Reproduction Commands

Setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
python -m pytest -q
```

Original release benchmarks:

```bash
python -m samatnext_bit.bench_speed --config configs/chainrule_vs_mono_24layer.yaml
python -m samatnext_bit.bench_speed --config configs/dense4_vs_sparse4_24.yaml
python -m samatnext_bit.bench_speed --config configs/softmax_vs_gdn_4active.yaml
```

Supplemental Python-token and MBPP runs:

```bash
python scripts/build_python_code_corpus.py
python -m samatnext_bit.bench_speed --config configs/regular_torch_4layer_audit.yaml
python -m samatnext_bit.bench_speed --config configs/practical_python_code_training_smoke.yaml
python -m samatnext_bit.bench_speed --config configs/bf16_softmax_vs_gdn_python_512.yaml
python -m samatnext_bit.bench_speed --config configs/dense24_mono_vram_optimized_python.yaml

python scripts/build_mbpp_smoke_corpus.py
python -m samatnext_bit.bench_speed --config configs/dense24_mono_optimized_mbpp_500step.yaml
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for expected results and generated data paths.

## License

Code in this repository is licensed under Apache-2.0. Dataset licenses remain governed by their original sources. MBPP is used only as a small smoke-training corpus and should be attributed to Google Research MBPP.

## Limitations

- Tiny Shakespeare, local Python-code text, and MBPP sanitized are smoke corpora.
- MBPP sanitized has only 384 training examples and 62,595 train tokens in this run.
- Generated samples from MBPP remained noisy and invalid-looking.
- No MBPP tests were executed; no pass@1 or coding ability claim is made.
- Chain-rule generally reached lower CE than mono at equal step count in earlier comparisons.
- `simple_gdn` was stable but slower and higher-memory than softmax in the BF16 Python-token smoke run.
- FLOPs/token are parameter-count estimates, not hardware counters.
- No real 1.58-bit/base3 speedup is claimed here.
- No result proves LLM-scale behavior or beats Transformers at scale.

## Documents

- [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md)
- [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)
- [CLAIMS_AND_LIMITATIONS.md](CLAIMS_AND_LIMITATIONS.md)
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
- [RELEASE_NOTES_MONO_FORWARD_SMOKE_V1.md](RELEASE_NOTES_MONO_FORWARD_SMOKE_V1.md)
- [RELEASE_NOTES_ARXIV_DRAFT_V1.md](RELEASE_NOTES_ARXIV_DRAFT_V1.md)
- [paper/main.md](paper/main.md)
