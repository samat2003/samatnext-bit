# SamatNext-Bit

SamatNext-Bit is a CUDA/PyTorch research sandbox for small byte-level decoder smoke benchmarks. This milestone freezes experiments around mono-forward scheduled local updates, dense versus sparse active compute, and a simple recurrent mixer baseline.

This is not a production LLM. The main release benchmark is a small hidden-128 decoder on Tiny Shakespeare with byte vocab size 256.

## Hardware

Primary milestone runs were measured on:

- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- Class: 12GB consumer laptop GPU
- Framework: PyTorch CUDA

Exact throughput varies with GPU, drivers, thermals, power mode, and background load.

## Dataset

The release benchmarks use Tiny Shakespeare as a byte-level next-token task:

- dataset path: `data/english_validation.txt`
- vocab size: 256 byte values
- split: 90% train, 10% validation
- validation batches: fixed
- training batches: preloaded on CUDA for timed benchmark loops

The benchmark excludes tokenization, batch sampling, and validation from measured training speed.

## Model Scale

All current release claims use a small decoder:

- hidden size: 128
- sequence length: 256
- heads: 4
- byte vocabulary: 256
- learned position embeddings
- RMSNorm
- causal mixer block
- MLP
- final LM head

Model sizes used in the milestone:

| model | active blocks | logical blocks | active params |
|---|---:|---:|---:|
| dense24 | 24 | 24 | 4,851,072 |
| sparse4/24 | 4 | 24 | 890,752 |
| dense4 | 4 | 4 | 890,752 |

`sparse4/24` is a logical 24-layer scaffold with only 4 active/instantiated blocks. It must not be described as dense24 training.

## Training Rules

`fp_chainrule` is regular full final-cross-entropy backprop every step.

`fp_mono_update_every_N` is the scheduled mono-forward/local update path. It runs forward every step and performs optimizer updates every `N` steps. This improves throughput in these smoke tests but changes update count and learning dynamics.

## Mixer Types

`softmax` is normal causal softmax attention.

`simple_gdn` is an experimental simple causal recurrent/linear mixer. It is not official Gated DeltaNet:

- `mixer_type=simple_gdn`
- `official_gdn=false`
- `linear_recurrent_mixer=true`

## Key Results

See [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) for tables.

Strongest honest milestone result: mono-forward scheduled updates substantially improve throughput in short byte-level smoke runs. Dense24 mono reached 4.58x the tokens/sec of dense24 chain-rule, while sparse4/24 mono reached 17.63x, but sparse4/24 used only 4 active blocks and 890,752 active params.

Main limitation: chain-rule learned better per update and reached lower validation CE in the short 500-step runs. Sparse4/24 did not outperform a normal dense4 baseline at matched active parameter count.

## Reproduce

Set up:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

Run tests:

```bash
python -m pytest -q
```

Run milestone benchmarks:

```bash
python -m samatnext_bit.bench_speed --config configs/chainrule_vs_mono_24layer.yaml
python -m samatnext_bit.bench_speed --config configs/dense4_vs_sparse4_24.yaml
python -m samatnext_bit.bench_speed --config configs/softmax_vs_gdn_4active.yaml
```

More detailed setup and expected results are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Release Documents

- [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md)
- [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)
- [CLAIMS_AND_LIMITATIONS.md](CLAIMS_AND_LIMITATIONS.md)
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
- [PAPER_OUTLINE.md](PAPER_OUTLINE.md)
- [paper/main.md](paper/main.md)

## Non-Claims

This release does not claim:

- LLM-scale quality
- beating Transformers at scale
- official Gated DeltaNet
- real 1.58-bit speedups
- sparse4/24 equivalence to dense24 training
- hardware-counter exact FLOP accounting
