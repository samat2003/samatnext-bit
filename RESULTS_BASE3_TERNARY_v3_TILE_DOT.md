# Base-3 Packed Ternary v3 Tile-Dot Results

Date: 2026-06-25

Environment:

- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- PyTorch: 2.11.0+cu128
- CUDA: 12.8
- Triton: 3.6.0

## Implementation

Added backend: `base3_tile_dot_ternary`.

- Ternary values: `{-1, 0, +1}`
- Storage remains base-3 packed: 20 trits per 32-bit word
- `packed_ternary=true`
- `base3_packed=true`
- `storage_bits_per_weight=1.6`
- `ideal_entropy_bits_per_weight=1.585`
- `native_kernel=true`
- `persistent_packed_cache=true`
- `decode_strategy=tile_dequant_dot`
- `uses_tensor_cores=true` by intent because the kernel uses `tl.dot`
- `uses_cuda_cores=true` for packed decode work
- `fallback_used=false`

The backend keeps weights stored as persistent base3 packed words. The forward kernel loads packed base3 words, decodes a K tile into temporary ternary values inside the kernel, then computes with `tl.dot`. It does not materialize a full decoded global weight matrix.

Supported shape constraint for v3: input K must be divisible by 128. Unsupported K raises a clear runtime error and does not silently fall back to fake ternary.

## Commands

```bash
python -m pytest -q
python -m samatnext_bit.bench_kernels
python -m samatnext_bit.bench --config configs/bitnet_tiny.yaml --modes fp_mono_update_every_2,bitnet_fake_ternary_mono_update_every_2,bitnet_base3_lut_ternary_mono_update_every_2,bitnet_base3_tile_dot_ternary_mono_update_every_2
python -m samatnext_bit.bench --config configs/bitnet_large.yaml --modes fp_mono_update_every_2,bitnet_fake_ternary_mono_update_every_2,bitnet_base3_lut_ternary_mono_update_every_2,bitnet_base3_tile_dot_ternary_mono_update_every_2
```

Tests: `17 passed`.

## Kernel Microbenchmark

Result file: `runs/kernel_bench_latest.json`

| Shape | Backend | ms | Speedup vs fake | Speedup vs LUT | Max error vs fake | Bits/weight | Tensor cores | Fallback |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| M=1024,K=128,N=128 | fake_ternary | 0.1103 | 1.000 | n/a | 0 | 32.0 | false | false |
| M=1024,K=128,N=128 | base3_packed_v1 | 0.1667 | 0.662 | n/a | 0.000906 | 1.6 | false | false |
| M=1024,K=128,N=128 | base3_lut_ternary | 0.2693 | 0.410 | 1.000 | 0.0000011 | 1.6 | false | false |
| M=1024,K=128,N=128 | base3_tile_dot_ternary | 0.1170 | 0.942 | 2.300 | 0.000906 | 1.6 | true | false |
| M=1024,K=512,N=512 | fake_ternary | 0.1801 | 1.000 | n/a | 0 | 32.0 | false | false |
| M=1024,K=512,N=512 | base3_packed_v1 | 0.9176 | 0.196 | n/a | 0.000858 | 1.6 | false | false |
| M=1024,K=512,N=512 | base3_lut_ternary | 0.8342 | 0.216 | 1.000 | 0.0000027 | 1.6 | false | false |
| M=1024,K=512,N=512 | base3_tile_dot_ternary | 0.7716 | 0.233 | 1.081 | 0.000858 | 1.6 | true | false |
| M=1024,K=768,N=768 | fake_ternary | 0.3340 | 1.000 | n/a | 0 | 32.0 | false | false |
| M=1024,K=768,N=768 | base3_packed_v1 | 1.8568 | 0.180 | n/a | 0.000921 | 1.6 | false | false |
| M=1024,K=768,N=768 | base3_lut_ternary | 1.7817 | 0.187 | 1.000 | 0.0000027 | 1.6 | false | false |
| M=1024,K=768,N=768 | base3_tile_dot_ternary | 1.5151 | 0.220 | 1.176 | 0.000921 | 1.6 | true | false |
| M=256,K=3072,N=768 | fake_ternary | 0.4957 | 1.000 | n/a | 0 | 32.0 | false | false |
| M=256,K=3072,N=768 | base3_packed_v1 | 1.7136 | 0.289 | n/a | 0.000844 | 1.6 | false | false |
| M=256,K=3072,N=768 | base3_lut_ternary | 1.7951 | 0.276 | 1.000 | 0.0000046 | 1.6 | false | false |
| M=256,K=3072,N=768 | base3_tile_dot_ternary | 1.1592 | 0.428 | 1.549 | 0.000844 | 1.6 | true | false |
| M=1024,K=3072,N=768 | fake_ternary | 1.0560 | 1.000 | n/a | 0 | 32.0 | false | false |
| M=1024,K=3072,N=768 | base3_packed_v1 | 5.9501 | 0.177 | n/a | 0.000899 | 1.6 | false | false |
| M=1024,K=3072,N=768 | base3_lut_ternary | 5.5823 | 0.189 | 1.000 | 0.0000068 | 1.6 | false | false |
| M=1024,K=3072,N=768 | base3_tile_dot_ternary | 2.5995 | 0.406 | 2.147 | 0.000899 | 1.6 | true | false |

Kernel-only result: tile-dot beat base3_lut on every requested shape and beat base3 v1 on every requested shape. It did not beat fake ternary.

## Model Benchmarks

Tiny result file: `runs/bitnet_tiny_20260625_002118/results.json`

| Config | Mode | Final CE | Tokens/sec | ms/step | Peak GB | Bits/weight | Decode | Tensor cores | Fallback |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| tiny | fp_mono_update_every_2 | 1.1537 | 234675.6 | 4.36 | 0.117 | n/a | n/a | false | false |
| tiny | bitnet_fake_ternary_mono_update_every_2 | 1.7348 | 196619.9 | 5.21 | 0.124 | 32.0 | n/a | false | false |
| tiny | bitnet_base3_lut_ternary_mono_update_every_2 | 1.7348 | 115344.6 | 8.88 | 0.117 | 1.6 | lut | false | false |
| tiny | bitnet_base3_tile_dot_ternary_mono_update_every_2 | 1.7352 | 120249.7 | 8.52 | 0.117 | 1.6 | tile_dequant_dot | true | false |

Large result file: `runs/bitnet_large_20260625_002143/results.json`

| Config | Mode | Final CE | Tokens/sec | ms/step | Peak GB | Bits/weight | Decode | Tensor cores | Fallback |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| large | fp_mono_update_every_2 | 1.2675 | 8318.0 | 30.78 | 1.743 | n/a | n/a | false | false |
| large | bitnet_fake_ternary_mono_update_every_2 | 1.3549 | 8342.9 | 30.68 | 2.473 | 32.0 | n/a | false | false |
| large | bitnet_base3_lut_ternary_mono_update_every_2 | 1.3548 | 3678.8 | 69.59 | 1.760 | 1.6 | lut | false | false |
| large | bitnet_base3_tile_dot_ternary_mono_update_every_2 | 1.3550 | 3655.4 | 70.03 | 1.760 | 1.6 | tile_dequant_dot | true | false |

## Required Answers

1. Files changed: `src/samatnext_bit/triton_kernels.py`, `src/samatnext_bit/bitlinear.py`, `src/samatnext_bit/train.py`, `src/samatnext_bit/bench_kernels.py`, `src/samatnext_bit/bench.py`, `tests/test_core.py`, `RESULTS_BASE3_TERNARY_v3_TILE_DOT.md`.
2. Commands run: listed above.
3. Test result: `17 passed`.
4. Kernel microbenchmark table: included above.
5. Model benchmark table: included above.
6. Does tile-dot beat base3_lut? Kernel-only: yes on all requested shapes. Model tiny: yes. Model large: no, slightly slower.
7. Does tile-dot beat base3 v1? Kernel-only: yes on all requested shapes. Model comparison against v1 was not requested in this run, so no model-level v1 claim is made here.
8. Does tile-dot beat fake ternary? No.
9. Does tile-dot beat FP? No.
10. Is storage still base3 1.6 bits/weight? Yes.
11. Does the kernel use `tl.dot`/MMA or not? The kernel uses `tl.dot`; metadata reports `uses_tensor_cores=true`. Exact lowering to MMA should be verified with generated kernel inspection before making a hardware-level Tensor Core utilization claim beyond the Triton `tl.dot` path.
12. Was any fallback used? No. `fallback_used=false`; unsupported K raises a clear error.
13. Biggest remaining bottleneck: base3 decode is still done inside the hot matmul loop with per-element integer division/modulo, and decoded tiles are not reused across programs. The tile-dot path improves compute structure but still spends too much time decoding.
14. Exact next kernel recommendation: implement a two-stage persistent tile cache for hot layers: decode base3 packed words into a small block-local FP16/int8 tile once per K/N tile, reuse that tile across multiple M blocks, and tune `BLOCK_M/BLOCK_N/BLOCK_K` with Triton autotuning. Also inspect generated SASS/PTX to confirm whether `tl.dot` lowers to MMA on this GPU.

## Conclusion

`base3_tile_dot_ternary` is correct, keeps real base3 packed ternary storage, and improves kernel-only performance over base3 LUT and base3 v1. It still does not beat fake ternary or FP in measured model benchmarks, so no end-to-end speedup claim is justified.
