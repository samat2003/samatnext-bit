# Results Summary

This file summarizes the frozen `mono-forward-smoke-v1` release and the supplemental arXiv-draft smoke results. Unless stated otherwise, throughput uses preloaded CUDA batches and excludes validation and tokenization. FLOPs/token are parameter-count estimates, not hardware counters.

## Chain-Rule vs Mono 24-Layer

Tiny Shakespeare byte-level smoke benchmark, hidden 128, vocab 256, seq 256.

| track | rule | total/active blocks | active params | updates | tok/s | final val CE | peak GB | est FLOPs/token | NaN/Inf |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| dense24_chainrule | chainrule | 24/24 | 4,851,072 | 500 | 145,006 | 2.2685 | 7.088 | 29,106,432 | false |
| dense24_mono | mono update every 16 | 24/24 | 4,851,072 | 32 | 663,520 | 3.2379 | 7.155 | 10,914,912 | false |
| sparse4_24_chainrule | chainrule | 24/4 | 890,752 | 500 | 617,245 | 2.4463 | 1.012 | 5,344,512 | false |
| sparse4_24_mono | mono update every 8 | 24/4 | 890,752 | 63 | 2,556,872 | 3.0240 | 1.022 | 2,226,880 | false |

Dense24 mono was 4.58x faster than dense24 chain-rule in tok/s, but dense24 chain-rule reached lower validation CE.

## Dense4 vs Sparse4/24 Reviewer Baseline

Tiny Shakespeare byte-level smoke benchmark, matched active parameter count.

| track | rule | total/active blocks | active params | updates | tok/s | final val CE | peak GB | est FLOPs/token | NaN/Inf |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| dense4_chainrule | chainrule | 4/4 | 890,752 | 500 | 661,312 | 2.4463 | ~1.01 | 5,344,512 | false |
| dense4_mono | mono update every 8 | 4/4 | 890,752 | 63 | 2,843,794 | 3.0240 | ~1.02 | 2,226,880 | false |
| sparse4_24_chainrule | chainrule | 24/4 | 890,752 | 500 | 633,070 | 2.4463 | ~1.01 | 5,344,512 | false |
| sparse4_24_mono | mono update every 8 | 24/4 | 890,752 | 63 | 2,465,589 | 3.0240 | ~1.02 | 2,226,880 | false |

Sparse4/24 matched dense4 but did not outperform it at the same active parameter count.

## Softmax vs Simple GDN 4-Active

Tiny Shakespeare byte-level smoke benchmark. `simple_gdn` is not official Gated DeltaNet.

| track | mixer | rule | active params | updates | tok/s | final val CE | peak GB | est FLOPs/token | NaN/Inf |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| dense4_softmax_chainrule | softmax | chainrule | 890,752 | 500 | 705,916 | 2.4463 | 1.003 | 5,344,512 | false |
| dense4_gdn_chainrule | simple_gdn | chainrule | 890,752 | 500 | 580,433 | 2.4878 | 1.379 | 5,344,512 | false |
| dense4_softmax_mono | softmax | mono update every 8 | 890,752 | 63 | 2,492,863 | 3.0240 | 1.022 | 2,226,880 | false |
| dense4_gdn_mono | simple_gdn | mono update every 8 | 890,752 | 63 | 2,095,307 | 3.0128 | 1.389 | 2,226,880 | false |
| sparse4_24_softmax_chainrule | softmax | chainrule | 890,752 | 500 | 642,413 | 2.4463 | 1.012 | 5,344,512 | false |
| sparse4_24_gdn_chainrule | simple_gdn | chainrule | 890,752 | 500 | 549,344 | 2.4878 | 1.379 | 5,344,512 | false |
| sparse4_24_softmax_mono | softmax | mono update every 8 | 890,752 | 63 | 2,596,091 | 3.0240 | 1.022 | 2,226,880 | false |
| sparse4_24_gdn_mono | simple_gdn | mono update every 8 | 890,752 | 63 | 2,097,410 | 3.0128 | 1.389 | 2,226,880 | false |

`simple_gdn` was stable but slower and higher-memory than softmax.

## Regular PyTorch 4-Layer Baseline Audit

Python-code token smoke benchmark, hidden 512, heads 8, vocab 16,000, seq 1024, batch 24.

| track | family | rule | params | updates | tok/s | final val CE | peak GB | est FLOPs/token | NaN/Inf |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| regular_torch_4layer_python_chainrule | regular PyTorch | chainrule | 29,534,848 | 100 | 126,948 | 6.4704 | 10.353 | 177,209,088 | false |
| samatnext_dense4_python_chainrule | SamatNext | chainrule | 29,530,240 | 100 | 142,473 | 6.4563 | 11.258 | 177,181,440 | false |
| samatnext_dense4_python_mono | SamatNext | mono update every 8 | 29,530,240 | 13 | 361,192 | 7.9963 | 11.618 | 73,825,600 | false |
| samatnext_sparse4_24_python_mono | SamatNext | mono update every 8 | 29,530,240 active | 13 | 362,131 | 7.9963 | 11.618 | 73,825,600 | false |

Regular PyTorch was also fast in this small 4-layer setup; mono improved throughput but reached worse short-run CE.

## Practical Python-Code Token Smoke

Generated local Python-code corpus, byte-level BPE vocab 16,000, hidden 512, heads 8, seq 1024, batch 24.

| track | rule | params | updates | tok/s | final val CE | peak GB | effective TFLOP/s | NaN/Inf |
|---|---|---:|---:|---:|---:|---:|---:|---|
| regular_torch_4layer_python_chainrule | chainrule | 29,534,848 | 100 | 126,948 | 6.4704 | 10.353 | 22.50 | false |
| samatnext_dense4_python_chainrule | chainrule | 29,530,240 | 100 | 142,473 | 6.4563 | 11.258 | 25.24 | false |
| samatnext_dense4_python_mono | mono update every 8 | 29,530,240 | 13 | 361,192 | 7.9963 | 11.618 | 26.67 | false |
| samatnext_sparse4_24_python_mono | mono update every 8 | 29,530,240 active | 13 | 362,131 | 7.9963 | 11.618 | 26.73 | false |

## BF16 Softmax vs Simple GDN Python 512

Python-code token smoke benchmark, BF16 mixed precision, hidden 512, seq 1024, batch 24.

| track | mixer | rule | tok/s | final val CE | peak GB | gradients finite | NaN/Inf |
|---|---|---|---:|---:|---:|---|---|
| dense4_softmax_bf16_chainrule | softmax | chainrule | 137,322 | 5.4315 | 11.260 | true | false |
| dense4_gdn_bf16_chainrule | simple_gdn | chainrule | 61,406 | 5.4542 | 13.466 | true | false |
| dense4_softmax_bf16_mono | softmax | mono update every 8 | 379,367 | 7.2706 | 11.621 | true | false |
| dense4_gdn_bf16_mono | simple_gdn | mono update every 8 | 79,852 | 7.2734 | 13.828 | true | false |

BF16 `simple_gdn` was stable but slower, higher-memory, and not better CE than softmax in this run.

## Dense24 Mono VRAM Optimization

Python-code token smoke benchmark, dense 24/24 softmax, hidden 512, heads 8, seq 512, batch 8.

| track | optimized | params | updates | tok/s | final val CE | peak GB | NaN/Inf |
|---|---:|---:|---:|---:|---:|---:|---|
| dense24_softmax_mono_as_is | false | 92,295,296 | 13 | 82,909 | 7.8292 | 7.459 | false |
| dense24_softmax_mono_optimized | true | 92,295,296 | 13 | 83,587 | 7.8292 | 4.641 | false |

The optimization avoided autograd graph construction on non-update mono steps. It reduced peak memory without changing model architecture or training rule.

## Supplemental MBPP Dense24 Mono 500-Step Smoke Training

Sanitized MBPP was used only as a small training smoke corpus. No tests were executed and no pass@1 is reported.

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
| Mean ms/step | 48.89 |
| Initial val CE | 8.7202 |
| Final val CE | 3.9844 |
| Final val PPL | 53.7520 |
| Gradients finite | true |
| NaN/Inf | false |

This result demonstrates stability and efficiency on a tiny Python problem/solution text corpus, not coding ability.
