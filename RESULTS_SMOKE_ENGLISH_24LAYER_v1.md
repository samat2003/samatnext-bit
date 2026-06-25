# RESULTS_SMOKE_ENGLISH_24LAYER_v1
24-layer English-dataset training smoke test on the RTX 5070 Ti Laptop GPU.
## Commands
```bash
python -m pytest -q
python -m samatnext_bit.bench_speed --config configs/smoke_english_24layer.yaml
```
## Test Result
`17 passed in 4.29s`
## Dataset
Dataset: `english_smoke`. Source: local fallback data/english_smoke.txt repeated in memory. No internet download was used. Byte-level vocab size: 256. Total tokens loaded: 1,000,090. Train split tokens: 1,000,090.
Batches were contiguous random slices from the English byte stream and were preloaded onto CUDA before timing. Timed speed excludes text loading, tokenization, and batch sampling. Batch sampling does not happen in the timed region.
## Results
| candidate | dense_or_sparse | total_layers | active_layers | recurrent_passes | effective_active_block_calls | mode | update_every | batch | seq | tokens/step | mean ms/step | p50 | p90 | p99 | tokens/sec | peak CUDA GB | initial CE | final CE | CE delta | initial ppl | final ppl | grad norm mean | grad norm max | finite grads | NaN/Inf | updates | reached 1M | reached 500K |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|---|
| sparse_24_active4_english | sparse/logical | 24 | 4 | 1 | 4 | `fp_mono_update_every_8` | 8 | 64 | 256 | 16320 | 7.678 | 3.671 | 26.284 | 42.362 | 2125497 | 0.901 | 5.7259 | 4.6076 | 1.1183 | 306.71 | 100.25 | 1.423 | 1.703 | true | false | 8 | true | true |
| dense_24_observed_english | dense | 24 | 24 | 1 | 24 | `fp_mono_update_every_16` | 16 | 64 | 256 | 16320 | 30.705 | 20.300 | 23.999 | 173.500 | 531515 | 4.687 | 5.6734 | 4.1362 | 1.5372 | 291.03 | 62.57 | 2.270 | 2.775 | true | false | 5 | false | true |
| dense_24_optimized_english | dense | 24 | 24 | 1 | 24 | `fp_mono_update_every_16` | 16 | 96 | 256 | 24480 | 44.133 | 30.643 | 41.708 | 220.706 | 554691 | 6.973 | 5.6718 | 4.1376 | 1.5343 | 290.57 | 62.65 | 2.281 | 2.777 | true | false | 5 | false | true |

## Answers
1. The English smoke test ran successfully.
2. Fastest on English: `sparse_24_active4_english` at 2125497 tok/s.
3. Sparse 24 active4 reached 1M: yes (2125497 tok/s).
4. Dense 24 observed stayed near 700K: no. It measured 531515 tok/s on English, slower than the synthetic smoke result of 688,248 tok/s.
5. Dense 24 optimized beat observed dense: yes (554691 vs 531515 tok/s).
6. Loss decreased for every candidate.
7. Perplexity decreased for every candidate.
8. Gradients were finite for every candidate.
9. No NaNs/Infs were observed in loss or gradients.
10. Claimable as dense: `dense_24_observed_english` and `dense_24_optimized_english`.
11. Claimable only as sparse/logical: `sparse_24_active4_english`.
12. English data slowed throughput versus synthetic/static tokens: sparse dropped from 3,058,818 to 2,125,497 tok/s, dense observed from 688,248 to 531,515 tok/s, and dense optimized from 763,273 to 554,691 tok/s. The benchmark preloaded CUDA batches, so the slowdown comes from the actual batch contents and run variability/update spikes, not dataloader or tokenizer time.
13. Next exact test: run 200-500 training steps on a larger real corpus with a held-out validation slice, still byte-level, and add CUDA graph capture for fixed-shape FP mono update_every_16 to reduce dense p99 update spikes.
