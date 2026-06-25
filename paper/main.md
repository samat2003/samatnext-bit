# Mono-Forward Scheduled Updates for Consumer-GPU Smoke Training of Small Decoder Models

## Abstract

We report smoke benchmarks for mono-forward scheduled updates in small decoder models on an NVIDIA GeForce RTX 5070 Ti Laptop GPU. Mono-forward is evaluated as a training rule rather than a new architecture: it performs forward passes every step while scheduling optimizer updates every `N` steps. On Tiny Shakespeare byte-level runs, dense24 mono-forward reached 4.58x the tokens/sec of dense24 chain-rule training, while chain-rule reached lower short-run validation cross entropy. Sparse active compute was faster but did not outperform a matched dense4 baseline. On a supplemental sanitized MBPP corpus used only as a training smoke corpus, an optimized dense24 mono-forward model trained for 500 FP16 steps with 81.1M parameters, 83.6K tokens/sec, and 3.992GB peak CUDA memory; validation CE decreased from 8.7202 to 3.9844. This MBPP run does not execute tests and is not a pass@1 or coding-ability result. The experiments are consumer-GPU smoke benchmarks for small decoders, not LLM-scale claims and not claims of beating Transformers at scale.

## 1. Introduction

This technical report documents the `samatnext-bit` mono-forward scheduled-update milestone. The goal is narrow: measure throughput, memory, and validation-loss behavior for small decoder training runs on a consumer laptop GPU.

The report emphasizes accounting. Scheduled updates change optimizer update counts. Sparse active compute changes the number of active parameters. Generated Python-code samples are not coding benchmark results. These distinctions are kept explicit throughout the results.

## 2. Background / Motivation

Full chain-rule backpropagation through all active layers is expensive. A scheduled mono-forward rule can reduce backward/update frequency and avoid unnecessary autograd work on non-update steps. The practical question is whether this improves throughput and memory enough to be useful in small consumer-GPU training smoke tests, while keeping loss finite and validation CE decreasing.

## 3. Method: Mono-Forward Scheduled Updates

The baseline training rule is:

- `fp_chainrule`: regular final-cross-entropy backprop every step.

The mono-forward training rule is:

- `fp_mono_update_every_N`: forward every step, perform optimizer update every `N` steps.

The dense24 VRAM-optimized implementation adds one runtime optimization: non-update mono-forward steps run under `torch.no_grad()`. Update steps still build the normal autograd graph and call `loss.backward()`. This does not change the model architecture, active layer count, mixer, dataset, or update schedule.

## 4. Experimental Setup

Hardware and software:

- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- GPU class: 12GB consumer laptop GPU
- Python: 3.12.3
- PyTorch: 2.11.0+cu128
- CUDA: 12.8
- Triton: 3.6.0

Timing:

- Training batches are preloaded on CUDA for reported benchmark loops.
- Validation is outside measured tok/s timing.
- Tokenizer training and tokenization are outside measured tok/s timing.
- FLOPs/token are parameter-count estimates, not hardware counters.

Datasets:

- Tiny Shakespeare byte-level text, vocab 256.
- Local Python-code BPE smoke corpus, vocab 16,000 in the recorded practical run.
- Sanitized MBPP text corpus from `google-research-datasets/mbpp`, used only as a training smoke corpus.

TODO: cite MBPP / Google Research.

TODO: cite PyTorch.

TODO: cite Tiny Shakespeare if needed.

## 5. Tiny Shakespeare Byte-Level Experiments

The original frozen release uses hidden size 128, sequence length 256, byte vocabulary size 256, and 500-step runs.

### 5.1 Chain-Rule vs Mono 24-Layer

| track | rule | active blocks | active params | updates | tok/s | final val CE |
|---|---|---:|---:|---:|---:|---:|
| dense24_chainrule | chainrule | 24/24 | 4,851,072 | 500 | 145,006 | 2.2685 |
| dense24_mono | mono update every 16 | 24/24 | 4,851,072 | 32 | 663,520 | 3.2379 |
| sparse4_24_chainrule | chainrule | 4/24 | 890,752 | 500 | 617,245 | 2.4463 |
| sparse4_24_mono | mono update every 8 | 4/24 | 890,752 | 63 | 2,556,872 | 3.0240 |

Dense24 mono-forward improved throughput by 4.58x over dense24 chain-rule. Dense24 chain-rule reached lower validation CE.

### 5.2 Dense4 vs Sparse4/24

| track | rule | active blocks | active params | updates | tok/s | final val CE |
|---|---|---:|---:|---:|---:|---:|
| dense4_chainrule | chainrule | 4/4 | 890,752 | 500 | 661,312 | 2.4463 |
| dense4_mono | mono update every 8 | 4/4 | 890,752 | 63 | 2,843,794 | 3.0240 |
| sparse4_24_chainrule | chainrule | 4/24 | 890,752 | 500 | 633,070 | 2.4463 |
| sparse4_24_mono | mono update every 8 | 4/24 | 890,752 | 63 | 2,465,589 | 3.0240 |

Sparse4/24 matched dense4 validation CE but did not beat it. The logical 24-layer scaffold has not shown quality benefit beyond 4 active blocks in this setup.

### 5.3 Softmax vs Simple Recurrent Mixer

`simple_gdn` is not official Gated DeltaNet. It is a simple recurrent/linear mixer labeled `official_gdn=false`.

| track | mixer | rule | tok/s | final val CE | peak GB |
|---|---|---|---:|---:|---:|
| dense4_softmax_chainrule | softmax | chainrule | 705,916 | 2.4463 | 1.003 |
| dense4_gdn_chainrule | simple_gdn | chainrule | 580,433 | 2.4878 | 1.379 |
| dense4_softmax_mono | softmax | mono | 2,492,863 | 3.0240 | 1.022 |
| dense4_gdn_mono | simple_gdn | mono | 2,095,307 | 3.0128 | 1.389 |

`simple_gdn` was stable and slightly improved mono CE in the Tiny Shakespeare run, but it was slower and higher-memory than softmax.

## 6. Regular PyTorch Baseline Audit

A plain PyTorch CUDA 4-layer Transformer-style chain-rule loop was added to test whether the 4-layer speed came from a custom SamatNext path.

| track | family | rule | tok/s | final val CE | peak GB |
|---|---|---|---:|---:|---:|
| regular_torch_4layer_python_chainrule | regular PyTorch | chainrule | 126,948 | 6.4704 | 10.353 |
| samatnext_dense4_python_chainrule | SamatNext | chainrule | 142,473 | 6.4563 | 11.258 |
| samatnext_dense4_python_mono | SamatNext | mono | 361,192 | 7.9963 | 11.618 |

The regular PyTorch baseline was also fast for this small 4-layer hidden-512 setup. Mono-forward improved throughput but reached worse CE after 100 steps.

## 7. Practical Python-Token Smoke Experiment

The practical Python-code smoke run used hidden 512, heads 8, sequence length 1024, batch 24, and a byte-level BPE vocabulary of 16,000.

| track | rule | tok/s | final val CE | peak GB |
|---|---|---:|---:|---:|
| regular_torch_4layer_python_chainrule | chainrule | 126,948 | 6.4704 | 10.353 |
| samatnext_dense4_python_chainrule | chainrule | 142,473 | 6.4563 | 11.258 |
| samatnext_dense4_python_mono | mono | 361,192 | 7.9963 | 11.618 |
| samatnext_sparse4_24_python_mono | mono | 362,131 | 7.9963 | 11.618 |

Larger hidden size, vocabulary, and sequence length reduced the extreme tok/s values observed in byte-level hidden-128 runs.

## 8. BF16 Softmax vs Simple GDN

The BF16 mixer comparison used hidden 512, sequence length 1024, batch 24, and the Python-code BPE setup.

| track | mixer | rule | tok/s | final val CE | peak GB |
|---|---|---|---:|---:|---:|
| dense4_softmax_bf16_chainrule | softmax | chainrule | 137,322 | 5.4315 | 11.260 |
| dense4_gdn_bf16_chainrule | simple_gdn | chainrule | 61,406 | 5.4542 | 13.466 |
| dense4_softmax_bf16_mono | softmax | mono | 379,367 | 7.2706 | 11.621 |
| dense4_gdn_bf16_mono | simple_gdn | mono | 79,852 | 7.2734 | 13.828 |

`simple_gdn` was stable but slower, higher-memory, and not better CE than softmax. GDN work is therefore stopped for this milestone.

## 9. Dense24 VRAM Optimization

The dense24 VRAM optimization used dense 24/24 softmax, hidden 512, heads 8, sequence length 512, batch 8, and FP16 mixed precision.

| track | optimized | params | tok/s | final val CE | peak GB |
|---|---:|---:|---:|---:|---:|
| dense24_softmax_mono_as_is | false | 92,295,296 | 82,909 | 7.8292 | 7.459 |
| dense24_softmax_mono_optimized | true | 92,295,296 | 83,587 | 7.8292 | 4.641 |

The optimization reduced memory by avoiding autograd graph construction on non-update mono steps. The main gain was VRAM headroom rather than speed.

## 10. MBPP Dense24 500-Step Smoke Training

On the sanitized MBPP corpus used only as a training smoke corpus, the optimized dense24 mono-forward model trained for 500 FP16 steps with 81.1M parameters, 83.6K tokens/sec, and 3.992GB peak CUDA memory. Validation CE decreased from 8.7202 to 3.9844. Because the corpus contains only 384 training examples and 62,595 train tokens, this result should be interpreted as a stability and efficiency smoke test rather than evidence of coding ability.

| field | value |
|---|---:|
| Dataset | `google-research-datasets/mbpp`, sanitized |
| Examples | 427 total, 384 train, 43 validation |
| Train/val tokens | 62,595 / 7,025 |
| Tokenizer | byte-level BPE, vocab 5,037 |
| Model | dense 24/24 softmax mono, hidden 512, heads 8 |
| Batch/seq | 8 / 512 |
| Steps/updates | 500 / 63 |
| Params | 81,058,221 |
| Peak CUDA memory | 3.992 GB |
| Tok/s | 83,613 |
| Initial val CE | 8.7202 |
| Final val CE | 3.9844 |
| Final val PPL | 53.7520 |
| Gradients finite | true |
| NaN/Inf | false |

No MBPP tests were executed. No pass@1 is reported. Generated samples remained noisy and invalid-looking.

## 11. Discussion

The strongest result is now a practical memory and throughput result: optimized dense24 mono-forward can run an 81M-parameter dense softmax decoder smoke-training loop under 4GB peak CUDA memory on a consumer laptop GPU. The validation CE decreases on the tiny MBPP text corpus, which supports stability of the training path.

The quality results remain conservative. Earlier controlled comparisons show chain-rule reaching better validation CE at equal step count. Sparse4/24 did not outperform dense4 at matched active parameters. `simple_gdn` did not outperform softmax in the practical BF16 Python-token comparison.

## 12. Limitations

- All experiments are smoke benchmarks.
- MBPP sanitized is tiny: 384 train examples and 62,595 train tokens.
- The 500-step MBPP run repeats a very small corpus.
- No MBPP tests are executed.
- No pass@1 or coding ability claim is made.
- Generated samples are noisy.
- No long-run convergence claim is made.
- No comparison against a trained standard Transformer on the same MBPP corpus for 500 steps is included.
- FLOPs are estimates, not hardware counters.
- No real 1.58-bit/base3 speedup is claimed.
- No result proves LLM-scale behavior or beats Transformers at scale.

## 13. Reproducibility

Core commands:

```bash
python -m pytest -q
python scripts/build_python_code_corpus.py
python scripts/build_mbpp_smoke_corpus.py
python -m samatnext_bit.bench_speed --config configs/dense24_mono_optimized_mbpp_500step.yaml
```

The full command list is in `REPRODUCIBILITY.md`.

## 14. Future Work

- Longer controlled runs.
- A chain-rule MBPP text smoke baseline with matched model size and steps.
- Better local-update accounting.
- Larger corpora after the small methodology is stable.
- Kernel optimization as a separate milestone.
- A faithful recurrent/delta mixer only if implemented and labeled accurately.
