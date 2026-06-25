# RESULTS_SPEED_500K_v1

Throughput-first training benchmark on the RTX 5070 Ti Laptop GPU.

## Commands

```bash
python -m pytest -q
python -m samatnext_bit.bench_speed --config configs/speed_500k_tiny.yaml
python -m samatnext_bit.bench_speed --config configs/speed_500k_32layer_tinywidth.yaml
python -m samatnext_bit.bench_speed --config configs/speed_500k_sparse32.yaml
```

## Test Result

`17 passed in 4.44s`

## Device

NVIDIA GeForce RTX 5070 Ti Laptop GPU

## Benchmark Summary

Tokens/sec is computed as:

```text
batch_size * (seq_len - 1) / step_time
```

Static synthetic CUDA batches were used. Timed steps include forward every step and backward/optimizer update on the configured mono update steps. Dataloader, tokenization, and eval time are not included.

| case | best tok/s | mode | active/layers | batch | seq | tokens/step | ms/step | peak GB | final CE | update performed |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| overall | 3,500,332 | `fp_mono_update_every_8` | 4/32 | 64 | 256 | 16,320 | 4.662 | 0.886 | 5.6729 | true |
| tiny dense | 3,444,187 | `fp_mono_update_every_8` | 4/4 | 64 | 256 | 16,320 | 4.738 | 0.886 | 5.6581 | true |
| dense 32-layer | 408,278 | `fp_mono_update_every_8` | 32/32 | 64 | 256 | 16,320 | 39.973 | 6.187 | 5.5425 | true |
| sparse logical 32 | 3,500,332 | `fp_mono_update_every_8` | 4/32 | 64 | 256 | 16,320 | 4.662 | 0.886 | 5.6729 | true |
| best fake ternary | 3,239,774 | `fake_ternary_mono_update_every_8` | 4/32 | 64 | 512 | 32,704 | 10.095 | 1.738 | 5.6266 | true |
| best base3 tile-dot | 1,008,334 | `base3_tile_dot_ternary_mono_update_every_4` | 4/32 | 64 | 256 | 16,320 | 16.185 | 1.227 | 5.6186 | true |

265 of 544 completed rows reached 500K tokens/sec.

## Answers

The fastest configuration was sparse logical 32 with 4 active layers, FP mono update every 8, batch 64, seq 256: 3,500,332 tok/s.

Tiny dense 4/4 also reached 500K, with a best result of 3,444,187 tok/s.

Dense 32/32 did not reach 500K. Its best row processed 16,320 tokens/step in 39.973 ms, for 408,278 tok/s. To hit 500K at the same tokens/step it needed 32.640 ms/step, so it was short by 7.333 ms/step.

Sparse logical 32 with 4/32 active layers reached 500K, but this is not dense 32-layer compute. Inactive layers were not computed.

Packed base3 remained real packed ternary storage at 20 trits per uint32, or 1.6 bits/weight, but it hurt model-level throughput versus FP and fake ternary. The best base3 tile-dot row reached 1,008,334 tok/s only in the 4/32 active sparse case and was slower than FP/fake on comparable sparse runs.

Fake ternary reached high throughput but did not beat FP in the best rows.

FP mono update_every_8 was the fastest path in this benchmark.

## Bottleneck

For dense 32/32, the bottleneck is dense transformer compute and activation memory at 32 layers. The best dense row used 6.187 GB and 39.973 ms/step. Larger batch/sequence combinations did not improve the dense 32-layer best result enough, and high-memory rows became slower rather than more efficient.

## Next Recommendation

Run a 24-layer sweep with FP/fake ternary mono update_every_8 and update_every_16, comparing dense 24/24 against honest sparse 4/24 and sparse recurrent 4/24 with 2 passes. The target should be 1M tok/s, with dense and sparse claims kept separate.
