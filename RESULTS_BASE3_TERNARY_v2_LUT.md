# Base-3 Packed Ternary v2 LUT Results

Date: 2026-06-25

Environment:

- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- PyTorch: 2.11.0+cu128
- CUDA: 12.8
- Triton: 3.6.0

## Implementation

Added backend: `base3_lut_ternary`.

- Ternary values: `{-1, 0, +1}`
- Storage: base-3 packed, 20 trits per 32-bit word
- `packed_ternary=true`
- `base3_packed=true`
- `storage_bits_per_weight=1.6`
- `ideal_entropy_bits_per_weight=1.585`
- `two_bit_packed=false`
- `int8_ternary=false`
- `native_kernel=true`
- `uses_cuda_cores=true`
- `uses_tensor_cores=false`
- `persistent_packed_cache=true`
- `decode_strategy=lut`

The LUT path keeps the existing GPU-packed persistent cache. It adds a 5-trit lookup table with 243 entries. The forward kernel splits each 20-trit base3 word into four 5-trit chunks and loads decoded `-1/0/+1` values from the LUT.

## Commands

```bash
python -m pytest -q
python -m samatnext_bit.bench_kernels
python -m samatnext_bit.bench --config configs/bitnet_tiny.yaml --modes fp_mono_update_every_2,bitnet_fake_ternary_mono_update_every_2,bitnet_base3_packed_ternary_mono_update_every_2,bitnet_base3_lut_ternary_mono_update_every_2
python -m samatnext_bit.bench --config configs/bitnet_large.yaml --modes fp_mono_update_every_2,bitnet_fake_ternary_mono_update_every_2,bitnet_base3_packed_ternary_mono_update_every_2,bitnet_base3_lut_ternary_mono_update_every_2
```

Tests: `13 passed`.

## Kernel Microbenchmark

Result file: `runs/kernel_bench_latest.json`

| Shape | Backend | ms | Max error vs fake | Bits/weight | Decode |
| --- | --- | ---: | ---: | ---: | --- |
| M=1024,K=512,N=512 | fake_ternary | 0.1139 | 0 | 32.0 | n/a |
| M=1024,K=512,N=512 | int8_ternary_legacy | 0.1744 | 0.000858 | 8.0 | int8 |
| M=1024,K=512,N=512 | two_bit_ternary_legacy | 4.0486 | 0.000858 | 2.0 | bitshift |
| M=1024,K=512,N=512 | base3_packed_v1 | 0.4621 | 0.000858 | 1.6 | divmod |
| M=1024,K=512,N=512 | base3_lut_ternary | 0.5604 | 0.0000027 | 1.6 | lut |
| M=1024,K=768,N=768 | fake_ternary | 0.1643 | 0 | 32.0 | n/a |
| M=1024,K=768,N=768 | int8_ternary_legacy | 0.2790 | 0.000921 | 8.0 | int8 |
| M=1024,K=768,N=768 | two_bit_ternary_legacy | 5.8342 | 0.000921 | 2.0 | bitshift |
| M=1024,K=768,N=768 | base3_packed_v1 | 0.8813 | 0.000921 | 1.6 | divmod |
| M=1024,K=768,N=768 | base3_lut_ternary | 0.9938 | 0.0000027 | 1.6 | lut |
| M=256,K=3072,N=768 | fake_ternary | 0.3038 | 0 | 32.0 | n/a |
| M=256,K=3072,N=768 | int8_ternary_legacy | 0.2079 | 0.000844 | 8.0 | int8 |
| M=256,K=3072,N=768 | two_bit_ternary_legacy | 20.1950 | 0.000844 | 2.0 | bitshift |
| M=256,K=3072,N=768 | base3_packed_v1 | 0.8296 | 0.000844 | 1.6 | divmod |
| M=256,K=3072,N=768 | base3_lut_ternary | 0.9287 | 0.0000046 | 1.6 | lut |
| M=1024,K=3072,N=768 | fake_ternary | 0.6239 | 0 | 32.0 | n/a |
| M=1024,K=3072,N=768 | int8_ternary_legacy | 0.3667 | 0.000899 | 8.0 | int8 |
| M=1024,K=3072,N=768 | two_bit_ternary_legacy | 19.6097 | 0.000899 | 2.0 | bitshift |
| M=1024,K=3072,N=768 | base3_packed_v1 | 2.9757 | 0.000899 | 1.6 | divmod |
| M=1024,K=3072,N=768 | base3_lut_ternary | 3.7039 | 0.0000068 | 1.6 | lut |

Kernel-only result: LUT did not beat base3 v1 on these shapes.

## Model Benchmarks

Tiny result file: `runs/bitnet_tiny_20260625_001601/results.json`

| Config | Mode | Final CE | Tokens/sec | ms/step | Peak GB | Bits/weight | Decode | Correctness error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| tiny | fp_mono_update_every_2 | 1.1537 | 159520.5 | 6.42 | 0.117 | n/a | n/a | n/a |
| tiny | bitnet_fake_ternary_mono_update_every_2 | 1.7348 | 134962.6 | 7.59 | 0.124 | 32.0 | n/a | n/a |
| tiny | bitnet_base3_packed_ternary_mono_update_every_2 | 1.7352 | 80390.6 | 12.74 | 0.117 | 1.6 | divmod | 0.000836 |
| tiny | bitnet_base3_lut_ternary_mono_update_every_2 | 1.7348 | 82049.5 | 12.48 | 0.117 | 1.6 | lut | 0.0000011 |

Large result file: `runs/bitnet_large_20260625_001627/results.json`

| Config | Mode | Final CE | Tokens/sec | ms/step | Peak GB | Bits/weight | Decode | Correctness error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| large | fp_mono_update_every_2 | 1.2675 | 8786.3 | 29.14 | 1.743 | n/a | n/a | n/a |
| large | bitnet_fake_ternary_mono_update_every_2 | 1.3549 | 7842.9 | 32.64 | 2.473 | 32.0 | n/a | n/a |
| large | bitnet_base3_packed_ternary_mono_update_every_2 | 1.3550 | 3628.2 | 70.56 | 1.760 | 1.6 | divmod | 0.000784 |
| large | bitnet_base3_lut_ternary_mono_update_every_2 | 1.3548 | 3496.8 | 73.21 | 1.760 | 1.6 | lut | 0.0000020 |

## Required Answers

1. Files changed: `src/samatnext_bit/triton_kernels.py`, `src/samatnext_bit/bitlinear.py`, `src/samatnext_bit/train.py`, `src/samatnext_bit/bench.py`, `src/samatnext_bit/bench_kernels.py`, `tests/test_core.py`, `RESULTS_BASE3_TERNARY_v2_LUT.md`.
2. Commands run: listed above.
3. Test result: `13 passed`.
4. Kernel microbenchmark table: included above.
5. Model benchmark table: included above.
6. Does LUT decode beat base3 v1? Kernel-only: no. Model tiny: yes, slightly. Model large: no.
7. Does LUT decode beat fake ternary? No.
8. Does LUT decode beat FP? No.
9. Is storage still 1.6 bits/weight? Yes.
10. Is this true base3 packed ternary, not 2-bit? Yes. It stores 20 trits per 32-bit base-3 word. It is not 2-bit storage.
11. Biggest remaining bottleneck: the forward kernel still does scalar per-trit accumulation and many memory/LUT loads instead of a tiled dequantize-to-shared/register block followed by a fast matrix multiply pattern. Cache refresh cost is also nontrivial during training.
12. Exact next kernel recommendation: replace the direct LUT-load accumulation kernel with a tiled two-stage kernel that decodes base3 packed words into a compact FP16/int8 tile in registers or shared memory, then uses `tl.dot` on that tile. Keep base3 packed storage persistent, but make the compute path tile-oriented enough to use Tensor Core-compatible dot operations if the decoded tile is represented in a supported dtype.

## Conclusion

The LUT backend is correct and keeps true base3 packed ternary storage at 1.6 bits/weight. It improves numerical agreement versus the div/mod base3 kernel, but it does not produce a measured speedup over fake ternary or FP. It also does not consistently beat base3 v1, so no speedup claim is justified.
