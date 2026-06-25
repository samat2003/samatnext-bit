# RESULTS_VALIDATE_ENGLISH_24LAYER_FINALFAIR_v1
Clean final fairness check for dense 24/24 vs sparse/logical 4/24 on Tiny Shakespeare byte-level validation.
## Commands
```bash
python -m pytest -q
python -m samatnext_bit.bench_speed --config configs/validate_english_24layer_finalfair.yaml
```
Test result: `17 passed in 4.96s`.
## Dataset
Dataset source: downloaded Tiny Shakespeare data/english_validation.txt. Total tokens: 1,115,394. Train tokens: 1,003,854. Validation tokens: 111,540. Byte vocab: 256. Validation batches were fixed. Training batches were preloaded CUDA byte slices; loading/tokenization/validation are excluded from training speed.
## Summary
| comparison | track | type | steps | updates | tokens processed | wall sec | tok/s | ms mean | p50 | p90 | p99 | peak GB | final train CE | final val CE | final val ppl | final val logged at actual final step | grad finite | NaN/Inf |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| equal_wallclock | dense_24_24_equal_wallclock | dense | 2000 | 125 | 48960000 | 76.97 | 636055 | 38.487 | 27.933 | 38.399 | 203.388 | 7.155 | 2.5895 | 2.5855 | 13.27 | true | true | false |
| equal_wallclock | sparse_24_active4_equal_wallclock | sparse/logical | 11161 | 1396 | 182147520 | 76.43 | 2383046 | 6.848 | 3.745 | 25.424 | 33.020 | 1.022 | 1.9352 | 2.0781 | 7.99 | true | true | false |
| equal_tokens | dense_24_24_equal_tokens | dense | 2000 | 125 | 48960000 | 82.17 | 595843 | 41.085 | 30.383 | 43.162 | 207.666 | 7.155 | 2.5895 | 2.5855 | 13.27 | true | true | false |
| equal_tokens | sparse_24_active4_equal_tokens | sparse/logical | 3000 | 375 | 48960000 | 19.63 | 2494589 | 6.542 | 3.527 | 24.421 | 31.844 | 1.022 | 2.4731 | 2.4914 | 12.08 | true | true | false |
| equal_updates | dense_24_24_equal_updates | dense | 4000 | 250 | 97920000 | 166.04 | 589752 | 41.509 | 31.188 | 40.989 | 215.657 | 7.155 | 2.4433 | 2.4677 | 11.80 | true | true | false |
| equal_updates | sparse_24_active4_equal_updates | sparse/logical | 2000 | 250 | 32640000 | 13.29 | 2455776 | 6.646 | 3.580 | 25.456 | 31.735 | 1.022 | 2.5408 | 2.5352 | 12.62 | true | true | false |

## Comparison Results
- equal_wallclock: sparse 4/24 wins. Dense val CE 2.5855; sparse val CE 2.0781. Dense tokens 48,960,000, updates 125, wall 76.97s. Sparse tokens 182,147,520, updates 1396, wall 76.43s.
- equal_tokens: sparse 4/24 wins. Dense val CE 2.5855; sparse val CE 2.4914. Dense tokens 48,960,000, updates 125, wall 82.17s. Sparse tokens 48,960,000, updates 375, wall 19.63s.
- equal_updates: dense 24/24 wins. Dense val CE 2.4677; sparse val CE 2.5352. Dense tokens 97,920,000, updates 250, wall 166.04s. Sparse tokens 32,640,000, updates 250, wall 13.29s.

## Answers
1. Equal wall-clock final validation CE: dense 2.5855; sparse 2.0781. Sparse wins.
2. Equal tokens final validation CE: dense 2.5855; sparse 2.4914. Sparse wins.
3. Equal updates final validation CE: dense 2.4677; sparse 2.5352. Dense wins.
4. Speed: sparse is faster in every comparison, roughly 2.38M-2.49M tok/s versus dense 0.59M-0.64M tok/s in this run.
5. Tokens processed: equal wall-clock sparse processed 182,147,520 tokens versus dense 48,960,000; equal tokens both processed 48,960,000; equal updates dense processed 97,920,000 versus sparse 32,640,000.
6. Optimizer updates: equal wall-clock dense 125 vs sparse 1396; equal tokens dense 125 vs sparse 375; equal updates both 250.
7. Wall-clock seconds are in the summary table; the wall-clock pair used 76.97s dense and 76.43s sparse.
8. Sparse final validation was logged at the actual final step for every sparse run, including equal wall-clock step 11161.
9. Better quality per update: dense 24/24. At 250 updates it reached lower validation CE.
10. Better quality per token: sparse 4/24. With exactly equal tokens it reached lower validation CE.
11. Better quality per second: sparse 4/24. With equal wall-clock it reached much lower validation CE.
12. Best honest claim: sparse/logical 4/24 is not dense, but it is much more throughput-efficient here; dense 24/24 is more optimizer-update-efficient.
13. Exact next experiment: equalize both tokens and optimizer updates simultaneously by adjusting batch/steps/update cadence, then run longer enough to see whether sparse remains token-efficient after both tracks reach lower validation CE.
