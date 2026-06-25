# SamatNext-Bit

SamatNext-Bit is a reproducible CUDA/PyTorch research artifact for consumer-GPU smoke benchmarks of mono-forward scheduled local updates in small byte-level decoder models. The frozen `mono-forward-smoke-v1` release evaluates Tiny Shakespeare byte-level next-token training on an NVIDIA GeForce RTX 5070 Ti Laptop GPU, with honest comparisons against regular chain-rule training, matched dense4 baselines, and a non-official `simple_gdn` recurrent mixer.

This repository is not an LLM-scale claim and does not claim to beat Transformers at scale.

## Frozen Release

- GitHub: `https://github.com/samat2003/samatnext-bit`
- Branch: `master`
- Tag: `mono-forward-smoke-v1`
- Base release commit: `ecdd51c` (`Freeze mono-forward smoke benchmark release`)
- Paper draft: [paper/main.md](paper/main.md)

The release freezes the current benchmark code, configs, result summaries, limitations, and paper scaffold. No new model features are part of this documentation pass.

## Main Results

All tables are 500-step Tiny Shakespeare byte-level smoke runs. Throughput is measured with preloaded CUDA training batches. FLOPs are parameter-count estimates where reported, not hardware counters.

### Chain-Rule vs Mono 24-Layer

| track | active blocks | active params | updates | tok/s | final val CE |
|---|---:|---:|---:|---:|---:|
| dense24_chainrule | 24/24 | 4,851,072 | 500 | 145,006 | 2.2685 |
| dense24_mono | 24/24 | 4,851,072 | 32 | 663,520 | 3.2379 |
| sparse4_24_chainrule | 4/24 | 890,752 | 500 | 617,245 | 2.4463 |
| sparse4_24_mono | 4/24 | 890,752 | 63 | 2,556,872 | 3.0240 |

Dense24 mono reached 4.58x the tok/s of dense24 chain-rule, but dense24 chain-rule reached lower short-run validation CE.

### Dense4 vs Sparse4/24 Reviewer Baseline

| track | active blocks | active params | updates | tok/s | final val CE |
|---|---:|---:|---:|---:|---:|
| dense4_chainrule | 4/4 | 890,752 | 500 | 661,312 | 2.4463 |
| dense4_mono | 4/4 | 890,752 | 63 | 2,843,794 | 3.0240 |
| sparse4_24_chainrule | 4/24 | 890,752 | 500 | 633,070 | 2.4463 |
| sparse4_24_mono | 4/24 | 890,752 | 63 | 2,465,589 | 3.0240 |

Sparse4/24 matched, but did not outperform, the normal dense4 baseline at the same active parameter count.

### Softmax vs `simple_gdn` 4-Active

| track | mixer | rule | tok/s | final val CE | peak GB |
|---|---|---|---:|---:|---:|
| dense4_softmax_chainrule | softmax | chainrule | 705,916 | 2.4463 | 1.003 |
| dense4_gdn_chainrule | simple_gdn | chainrule | 580,433 | 2.4878 | 1.379 |
| dense4_softmax_mono | softmax | mono | 2,492,863 | 3.0240 | 1.022 |
| dense4_gdn_mono | simple_gdn | mono | 2,095,307 | 3.0128 | 1.389 |
| sparse4_24_softmax_chainrule | softmax | chainrule | 642,413 | 2.4463 | 1.012 |
| sparse4_24_gdn_chainrule | simple_gdn | chainrule | 549,344 | 2.4878 | 1.379 |
| sparse4_24_softmax_mono | softmax | mono | 2,596,091 | 3.0240 | 1.022 |
| sparse4_24_gdn_mono | simple_gdn | mono | 2,097,410 | 3.0128 | 1.389 |

`simple_gdn` was stable and slightly improved mono CE, but it was slower and used more memory than softmax. It is not official Gated DeltaNet.

## Hardware and Software

Milestone measurements used:

- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- Class: 12GB consumer laptop GPU
- Python: 3.12.3
- PyTorch: 2.11.0+cu128
- CUDA: 12.8
- Triton: 3.6.0

Exact tok/s varies with GPU model, drivers, thermals, power mode, and background load.

## Model and Data

- Dataset: Tiny Shakespeare at `data/english_validation.txt`
- Task: byte-level next-token prediction
- Vocab size: 256 byte values
- Split: 90% train, 10% validation
- Model: small hidden-128 decoder, not a production LLM
- Sequence length: 256
- Heads: 4

Model scale:

| name | meaning | active params |
|---|---|---:|
| dense24 | 24 active blocks out of 24 | 4,851,072 |
| sparse4/24 | logical 24-layer scaffold with 4 active/instantiated blocks | 890,752 |
| dense4 | normal 4-layer model | 890,752 |

## Training Rules

`fp_chainrule` is regular full final-cross-entropy backprop every step.

`fp_mono_update_every_N` is the scheduled mono-forward/local update rule. It runs forward every step and performs optimizer updates every `N` steps. Throughput improves in these smoke runs, but update counts and learning dynamics differ from chain-rule.

## Mixer Types

`softmax` is normal causal softmax attention.

`simple_gdn` is an experimental simple causal recurrent/linear mixer:

- `mixer_type=simple_gdn`
- `official_gdn=false`
- `linear_recurrent_mixer=true`

It is not official Gated DeltaNet.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
python -m pytest -q
python -m samatnext_bit.bench_speed --config configs/chainrule_vs_mono_24layer.yaml
python -m samatnext_bit.bench_speed --config configs/dense4_vs_sparse4_24.yaml
python -m samatnext_bit.bench_speed --config configs/softmax_vs_gdn_4active.yaml
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for expected approximate results and environment details.

## Limitations

- Tiny Shakespeare byte-level validation is a smoke benchmark, not evidence of LLM-scale quality.
- Models are small hidden-128 decoders.
- Chain-rule reached better short-run validation CE than mono in the main dense24 comparison.
- Sparse4/24 is not dense24 training and did not beat dense4 at matched active params.
- `simple_gdn` is stable but slower and higher-memory than softmax here.
- FLOPs/token are estimated from parameter counts, not hardware counters.
- No claim is made that this beats Transformers at scale.
- No claim is made that this is official Gated DeltaNet.
- No real 1.58-bit speedup is claimed in this release.

## Documents

- [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md)
- [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)
- [CLAIMS_AND_LIMITATIONS.md](CLAIMS_AND_LIMITATIONS.md)
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
- [RELEASE_NOTES_MONO_FORWARD_SMOKE_V1.md](RELEASE_NOTES_MONO_FORWARD_SMOKE_V1.md)
- [paper/main.md](paper/main.md)
- [paper/reproducibility_checklist.md](paper/reproducibility_checklist.md)
