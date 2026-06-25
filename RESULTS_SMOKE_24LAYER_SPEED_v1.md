# RESULTS_SMOKE_24LAYER_SPEED_v1
24-layer targeted speed smoke test on the RTX 5070 Ti Laptop GPU.
## Commands
```bash
python -m pytest -q
python -m samatnext_bit.bench_speed --config configs/smoke_24layer_speed.yaml
```
## Test Result
`17 passed in 5.68s`
## Benchmark Rules
Static synthetic CUDA token batches were used. Timed steps include forward every step and backward/optimizer update on the configured mono update interval. Dataloader, tokenizer, and eval time are not included. Tokens/sec is `batch_size * (seq_len - 1) / mean_step_time`.
## Results
| candidate | dense_or_sparse | total_layers | active_layers | recurrent_passes | effective_active_block_calls | mode | batch | seq | tokens/step | mean ms/step | p50 | p90 | p99 | tokens/sec | peak CUDA GB | initial CE | final CE | CE delta | update | reached 1M | OOM |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| A | sparse/logical | 24 | 4 | 1 | 4 | `fp_mono_update_every_8` | 64 | 256 | 16320 | 5.335 | 3.919 | 11.304 | 16.989 | 3058818 | 0.886 | 5.7196 | 5.5885 | 0.1311 | true | true | false |
| B | dense | 24 | 24 | 1 | 24 | `fp_mono_update_every_16` | 64 | 256 | 16320 | 23.712 | 20.147 | 22.350 | 86.199 | 688248 | 4.672 | 5.7133 | 5.5121 | 0.2011 | true | false | false |
| C | dense | 24 | 24 | 1 | 24 | `fp_mono_update_every_16` | 80 | 256 | 20400 | 27.589 | 23.160 | 25.759 | 82.610 | 739423 | 5.911 | 5.7122 | 5.5294 | 0.1828 | true | false | false |
| C | dense | 24 | 24 | 1 | 24 | `fp_mono_update_every_16` | 96 | 256 | 24480 | 32.072 | 26.242 | 30.223 | 104.078 | 763273 | 6.951 | 5.7107 | 5.5375 | 0.1731 | true | false | false |
| C | dense | 24 | 24 | 1 | 24 | `fp_mono_update_every_16` | 64 | 384 | 24512 | 32.165 | 26.440 | 28.250 | 105.430 | 762060 | 6.951 | 5.7124 | 5.5410 | 0.1715 | true | false | false |
| C | dense | 24 | 24 | 1 | 24 | `fp_mono_update_every_16` | 80 | 384 | 30640 | 40.686 | 33.638 | 39.142 | 126.732 | 753079 | 8.756 | 5.7117 | 5.5496 | 0.1621 | true | false | false |

## Answers
1. Fastest candidate: A, sparse/logical 24 with 4 active layers, at 3058818 tok/s.
2. Sparse 24 active4 reached 1M: yes (3058818 tok/s).
3. Dense 24 observed config reproduced the expected range: no; measured 688248 tok/s, below the prior single observation of 797,656 but in the same broad band.
4. Dense 24 optimized reached 1M: no. Best dense row was candidate C batch 96 seq 256 at 763273 tok/s.
5. Best dense tokens/step was 24480; to reach 1M it needs 24.480 ms/step. Measured mean was 32.072 ms/step, short by 7.592 ms.
6. Claimable as dense: B and C rows only, because active_layers equals total_layers.
7. Claimable only as sparse/logical: A, because it computes 4/24 active layers and does not instantiate or compute inactive layers.
8. Next exact optimization: add CUDA graph capture for the fixed FP mono update_every_16 dense loop, split into captured no-update and update steps, then rerun the best dense shapes batch 96 seq 256 and batch 64 seq 384. The p50 step times are already near the 1M requirement, while high p99 update spikes keep the mean around 32 ms.
