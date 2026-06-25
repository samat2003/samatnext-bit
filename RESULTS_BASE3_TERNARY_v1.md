# Base-3 Packed Ternary v1 Results

Date: 2026-06-25

Environment:

- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- PyTorch: 2.11.0+cu128
- CUDA: 12.8
- Triton: 3.6.0

## Implementation

This backend is `base3_packed_ternary`.

- Values are ternary `{-1, 0, +1}`.
- Storage is base-3 packed: 20 trits per 32-bit word.
- Effective storage is `32 / 20 = 1.6` bits per weight.
- Ideal entropy is `log2(3) ~= 1.585` bits per weight.
- It is not 2-bit packing.
- It is not int8 ternary storage.
- It is not called true 1.58-bit entropy packing: `packed_1p58bit=false`.
- Triton packs shadow weights on GPU into persistent packed cache buffers.
- Forward Triton kernel reads base-3 packed words and decodes trits.
- Backward is STE fallback: `backward_native=false`, `ste_backward=true`.
- Kernel uses scalar decode/add/sub CUDA-core style logic: `uses_tensor_cores=false`, `uses_cuda_cores=true`.

## Commands

```bash
python -m pytest -q
python -m samatnext_bit.bench --config configs/bitnet_tiny.yaml --modes fp_mono_update_every_2,bitnet_fake_ternary_mono_update_every_2,bitnet_two_bit_ternary_legacy_mono_update_every_2,bitnet_base3_packed_ternary_mono_update_every_2
python -m samatnext_bit.bench --config configs/bitnet_large.yaml --modes fp_mono_update_every_2,bitnet_fake_ternary_mono_update_every_2,bitnet_base3_packed_ternary_mono_update_every_2
```

Tests: `10 passed`.

## Tiny 4x128

Result file: `runs/bitnet_tiny_20260625_000803/results.json`

| Mode | Initial CE | Final CE | Tokens/sec | ms/step | Peak GB | Bits/weight | Cache refreshes | Pack ms | Forward kernel ms | Backend | Base3 | Native |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| fp_mono_update_every_2 | 5.6809 | 1.1537 | 180724.1 | 5.67 | 0.117 | n/a | 0 | 0.0 | 0.0 | fp | false | false |
| bitnet_fake_ternary_mono_update_every_2 | 5.6613 | 1.7348 | 171424.2 | 5.97 | 0.124 | 32.0 | 0 | 0.0 | 0.0 | fake_ternary | false | false |
| bitnet_two_bit_ternary_legacy_mono_update_every_2 | 5.6612 | 1.7352 | 29353.3 | 34.89 | 0.124 | 2.0 | 0 | 0.0 | 0.0 | two_bit_ternary_legacy | false | true |
| bitnet_base3_packed_ternary_mono_update_every_2 | 5.6612 | 1.7353 | 80924.4 | 12.65 | 0.117 | 1.6 | 2567 | 857.8 | 2583.4 | base3_packed_ternary | true | true |

Base3 correctness error vs fake: `0.0008365`.

## Large 12x768

Result file: `runs/bitnet_large_20260625_000823/results.json`

| Mode | Initial CE | Final CE | Tokens/sec | ms/step | Peak GB | Bits/weight | Cache refreshes | Pack ms | Forward kernel ms | Backend | Base3 | Native |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| fp_mono_update_every_2 | 5.5378 | 1.2675 | 7450.4 | 34.36 | 1.743 | n/a | 0 | 0.0 | 0.0 | fp | false | false |
| bitnet_fake_ternary_mono_update_every_2 | 5.7636 | 1.3549 | 7355.2 | 34.81 | 2.473 | 32.0 | 0 | 0.0 | 0.0 | fake_ternary | false | false |
| bitnet_base3_packed_ternary_mono_update_every_2 | 5.7635 | 1.3550 | 3568.8 | 71.73 | 1.759 | 1.6 | 1274 | 1228.2 | 2886.0 | base3_packed_ternary | true | true |

Base3 correctness error vs fake: `0.0007844`.

## Required Answers

1. Is this ternary `{−1,0,+1}`? Yes.
2. Is it base-3 packed? Yes, 20 trits per 32-bit word.
3. Is storage ~1.6 bits/weight? Yes, `32 / 20 = 1.6`.
4. Is it still 2-bit packing? No. The old 2-bit path is only `two_bit_ternary_legacy`.
5. Does the forward kernel read base-3 packed words? Yes.
6. Is packing done on GPU? Yes, via a Triton packing kernel.
7. Is packed cache persistent? Yes. `BitLinear` owns persistent `packed_cache` and `gamma_cache` buffers. The cache is reused between forwards and refreshed after update steps.
8. Does it match fake ternary numerically? Yes within about `8.4e-4` max error in these checks.
9. Does it beat 2-bit packed? Yes on tiny: `80.9k` vs `29.4k` tokens/sec.
10. Does it beat fake ternary? No.
11. Does it beat FP? No.
12. Does it use Tensor Cores or CUDA cores? CUDA cores. `uses_tensor_cores=false`, `uses_cuda_cores=true`.
13. Exact next kernel optimization: eliminate scalar per-trit division/mod decode in the forward path by decoding each base-3 word once into a small register tile, accumulating all 20 trits per loaded word, and fuse cache refresh scheduling so only touched rows/blocks are repacked after optimizer updates.

## Limitations

- Base-3 cache refresh is persistent but still refreshes all BitLinear layers after every optimizer step.
- Forward decode uses integer division/modulo by powers of 3, which is expensive.
- Backward remains STE fallback, not native packed backward.
- No speedup claim is justified because base3 packed did not beat fake ternary or FP throughput.
