# Results Summary

All release results are 500-step Tiny Shakespeare byte-level smoke benchmarks on an NVIDIA GeForce RTX 5070 Ti Laptop GPU. Vocab size is 256. Training batches were preloaded on CUDA. FLOPs/token are parameter-count estimates, not hardware counters.

## Chain-Rule vs Mono 24-Layer

| track | rule | total/active blocks | active params | updates | tok/s | final val CE | peak GB | est FLOPs/token | NaN/Inf |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| dense24_chainrule | chainrule | 24/24 | 4,851,072 | 500 | 145,006 | 2.2685 | 7.088 | 29,106,432 | false |
| dense24_mono | mono update every 16 | 24/24 | 4,851,072 | 32 | 663,520 | 3.2379 | 7.155 | 10,914,912 | false |
| sparse4_24_chainrule | chainrule | 24/4 | 890,752 | 500 | 617,245 | 2.4463 | 1.012 | 5,344,512 | false |
| sparse4_24_mono | mono update every 8 | 24/4 | 890,752 | 63 | 2,556,872 | 3.0240 | 1.022 | 2,226,880 | false |

Key observations:

- Dense24 mono was 4.58x faster than dense24 chain-rule in tokens/sec.
- Sparse4/24 mono was 17.63x faster than dense24 chain-rule in tokens/sec.
- Dense24 chain-rule reached the best validation CE in this short run.
- Sparse4/24 uses only 4 active blocks and 890,752 active params, so it is not dense24 training.

## Dense4 vs Sparse4/24 Reviewer Baseline

| track | rule | total/active blocks | active params | updates | tok/s | final val CE | peak GB | est FLOPs/token | NaN/Inf |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| dense4_chainrule | chainrule | 4/4 | 890,752 | 500 | 661,312 | 2.4463 | ~1.01 | 5,344,512 | false |
| dense4_mono | mono update every 8 | 4/4 | 890,752 | 63 | 2,843,794 | 3.0240 | ~1.02 | 2,226,880 | false |
| sparse4_24_chainrule | chainrule | 24/4 | 890,752 | 500 | 633,070 | 2.4463 | ~1.01 | 5,344,512 | false |
| sparse4_24_mono | mono update every 8 | 24/4 | 890,752 | 63 | 2,465,589 | 3.0240 | ~1.02 | 2,226,880 | false |

Key observations:

- Sparse4/24 currently behaves like a normal dense4 model at the same active parameter count.
- The logical 24-layer scaffold has not yet shown extra validation quality beyond 4 active blocks.

## Softmax vs Simple GDN 4-Active

`simple_gdn` is not official Gated DeltaNet. It is labeled `official_gdn=false`.

| track | mixer | rule | total/active blocks | active params | updates | tok/s | final val CE | peak GB | est FLOPs/token | NaN/Inf |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| dense4_softmax_chainrule | softmax | chainrule | 4/4 | 890,752 | 500 | 705,916 | 2.4463 | 1.003 | 5,344,512 | false |
| dense4_gdn_chainrule | simple_gdn | chainrule | 4/4 | 890,752 | 500 | 580,433 | 2.4878 | 1.379 | 5,344,512 | false |
| dense4_softmax_mono | softmax | mono update every 8 | 4/4 | 890,752 | 63 | 2,492,863 | 3.0240 | 1.022 | 2,226,880 | false |
| dense4_gdn_mono | simple_gdn | mono update every 8 | 4/4 | 890,752 | 63 | 2,095,307 | 3.0128 | 1.389 | 2,226,880 | false |
| sparse4_24_softmax_chainrule | softmax | chainrule | 24/4 | 890,752 | 500 | 642,413 | 2.4463 | 1.012 | 5,344,512 | false |
| sparse4_24_gdn_chainrule | simple_gdn | chainrule | 24/4 | 890,752 | 500 | 549,344 | 2.4878 | 1.379 | 5,344,512 | false |
| sparse4_24_softmax_mono | softmax | mono update every 8 | 24/4 | 890,752 | 63 | 2,596,091 | 3.0240 | 1.022 | 2,226,880 | false |
| sparse4_24_gdn_mono | simple_gdn | mono update every 8 | 24/4 | 890,752 | 63 | 2,097,410 | 3.0128 | 1.389 | 2,226,880 | false |

Key observations:

- `simple_gdn` ran stably with finite gradients.
- `simple_gdn` slightly improved mono final validation CE in this smoke run.
- `simple_gdn` was slower and used more peak memory than softmax in this implementation.
- `simple_gdn` did not beat softmax under chain-rule.
