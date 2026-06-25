# RESULTS_VALIDATE_ENGLISH_24LAYER_FAIR_v1
Fairer Tiny Shakespeare byte-level validation for dense 24/24 vs sparse/logical 4/24 on RTX 5070 Ti Laptop GPU.
## Run Status
The full fair run completed once and wrote `runs/validate_english_24layer_fair_20260625_012404/speed_results.json`. A follow-up rerun was started after fixing final checkpoint logging for wall-clock-limited runs, but it was interrupted before producing a new JSON. Therefore this report uses the completed JSON and explicitly labels the sparse equal-wall-clock validation CE as the last checkpoint at step 4000, not the true final step 11057.
## Commands
```bash
python -m pytest -q
python -m samatnext_bit.bench_speed --config configs/validate_english_24layer_fair.yaml
```
Tests passed before the run: `17 passed in 4.21s`. A rerun after the checkpoint fix also passed tests: `17 passed in 7.58s`, but that rerun was stopped before completion.
## Dataset
Dataset: downloaded Tiny Shakespeare data/english_validation.txt. Total tokens: 1,115,394. Train tokens: 1,003,854. Validation tokens: 111,540. Byte vocab: 256. Batches were preloaded CUDA byte slices; data loading/tokenization/validation were excluded from training speed.
## Summary
| comparison | track | type | calls | steps | updates | tokens | wall sec | tok/s | ms mean | p50 | p90 | p99 | peak GB | final/last train CE | final/last val CE | best val CE | grad mean | grad max | finite | NaN/Inf |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| equal_updates | dense_24_24_equal_updates | dense | 24 | 4000 | 250 | 97920000 | 140.78 | 695573 | 35.194 | 25.756 | 32.001 | 184.039 | 7.155 | 2.4433 (final) | 2.4677 (final) | 2.4677 | 0.449 | 2.476 | true | false |
| equal_updates | sparse_24_active4_equal_updates | sparse/logical | 4 | 2000 | 250 | 32640000 | 14.38 | 2269852 | 7.190 | 4.072 | 27.644 | 31.489 | 1.022 | 2.5408 (final) | 2.5352 (final) | 2.5352 | 0.382 | 1.586 | true | false |
| equal_steps | dense_24_24_equal_steps | dense | 24 | 2000 | 125 | 48960000 | 72.63 | 674100 | 36.315 | 27.056 | 33.343 | 183.616 | 7.155 | 2.5895 (final) | 2.5855 (final) | 2.5855 | 0.637 | 2.476 | true | false |
| equal_steps | sparse_24_active4_equal_steps | sparse/logical | 4 | 2000 | 250 | 32640000 | 12.00 | 2719741 | 6.001 | 3.390 | 22.455 | 26.828 | 1.022 | 2.5408 (final) | 2.5352 (final) | 2.5352 | 0.382 | 1.586 | true | false |
| equal_steps | sparse_24_active4_recurrent2_equal_steps | sparse/logical | 8 | 2000 | 250 | 32640000 | 19.86 | 1643646 | 9.929 | 6.271 | 34.294 | 39.101 | 1.764 | 2.5304 (final) | 2.5289 (final) | 2.5289 | 0.438 | 1.832 | true | false |
| equal_wallclock | dense_24_24_equal_wallclock | dense | 24 | 2000 | 125 | 48960000 | 72.56 | 674774 | 36.279 | 27.035 | 33.044 | 184.146 | 7.155 | 2.5895 (final) | 2.5855 (final) | 2.5855 | 0.637 | 2.476 | true | false |
| equal_wallclock | sparse_24_active4_equal_wallclock | sparse/logical | 4 | 11057 | 1383 | 180450240 | 71.99 | 2506545 | 6.511 | 3.616 | 23.947 | 30.464 | 1.022 | 2.4328 (last checkpoint) | 2.4415 (last checkpoint) | 2.4415 | 0.401 | 1.586 | true | false |

## Checkpoint Endpoints
- `dense_24_24_equal_updates`: first val CE 5.6616; last logged step 4000 val CE 2.4677; best logged val CE 2.4677.
- `sparse_24_active4_equal_updates`: first val CE 5.7337; last logged step 2000 val CE 2.5352; best logged val CE 2.5352.
- `dense_24_24_equal_steps`: first val CE 5.6616; last logged step 2000 val CE 2.5855; best logged val CE 2.5855.
- `sparse_24_active4_equal_steps`: first val CE 5.7337; last logged step 2000 val CE 2.5352; best logged val CE 2.5352.
- `sparse_24_active4_recurrent2_equal_steps`: first val CE 5.7449; last logged step 2000 val CE 2.5289; best logged val CE 2.5289.
- `dense_24_24_equal_wallclock`: first val CE 5.6616; last logged step 2000 val CE 2.5855; best logged val CE 2.5855.
- `sparse_24_active4_equal_wallclock`: first val CE 5.7337; last logged step 4000 val CE 2.4415; best logged val CE 2.4415.

## Answers
1. Equal optimizer-update comparison: dense 24/24 had better validation CE. Dense 250 updates ended at val CE 2.4677; sparse 250 updates ended at val CE 2.5352.
2. Equal step-count comparison: sparse 4/24 had better validation CE than dense. Dense 2000 steps val CE 2.5855; sparse 2000 steps val CE 2.5352; recurrent2 sparse val CE 2.5289.
3. Equal wall-clock comparison: sparse 4/24 had better logged validation CE. Dense 72.56s/2000 steps val CE 2.5855; sparse 71.99s/11057 steps last logged val CE 2.4415 at step 4000.
4. Sparse 4/24 kept improving through the logged checkpoints; it did not show a clear plateau in this run.
5. Dense 24/24 caught up and surpassed sparse when optimizer updates were equalized at 250 updates.
6. Fastest track: sparse 4/24 equal_steps at 2719741 tok/s among fixed-step rows; sparse wall-clock measured 2506545 tok/s.
7. Best logged validation loss: sparse 4/24 equal-wall-clock, val CE 2.4415. Among fixed planned runs with true final validation, dense equal-updates was best at 2.4677.
8. Best speed/quality tradeoff: sparse 4/24 for wall-clock-limited training; dense 24/24 for equal optimizer-update quality.
9. Recurrent2 helped sparse quality only slightly at equal steps: val CE 2.5289 vs 2.5352, but throughput dropped from 2719741 to 1643646 tok/s. It is not justified for this speed target.
10. Gradients were finite and stable enough for all completed tracks.
11. No NaNs/Infs were reported.
12. Best honest claim: sparse/logical 4/24 is far faster and wins wall-clock and equal-step validation, but dense 24/24 wins when optimizer updates are equalized. Sparse 4/24 must not be described as dense 24.
13. Exact next experiment: rerun only the equal-wall-clock pair with final checkpoint logging enabled, then run an equal-token-and-equal-update comparison so dense and sparse see comparable token volume and optimizer count.
