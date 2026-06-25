# Packed Ternary v1 Results

Date: 2026-06-24

Environment:

- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- PyTorch: 2.11.0+cu128
- CUDA: 12.8
- Triton: 3.6.0

## Definition

This version implements ternary values `{-1, 0, +1}`.

Packed backend:

- Packs 16 ternary weights per int32/uint32-equivalent word.
- Uses 2-bit codes: `00 = 0`, `01 = +1`, `10 = -1`, `11 = unused`.
- Triton reads packed 2-bit codes and unpacks inside the matmul kernel.
- Applies per-output gamma: `gamma[o] = mean(abs(W_shadow[o, :]))`.
- Uses packed native forward and STE fallback backward.

Metadata:

- `packed_ternary=true`
- `storage_bits_per_weight=2.0`
- `ideal_entropy_bits_per_weight=1.58`
- `packed_1p58bit=false`
- `forward_native=true`
- `backward_native=false`
- `ste_backward=true`

This is real packed ternary below int8, but it is not true 1.58-bit entropy/base-3 packing.

## Commands

```bash
python -m pytest -q
python -m samatnext_bit.bench --config configs/bitnet_tiny.yaml --modes fp_chainrule,fp_mono_update_every_2,bitnet_fake_ternary_mono_update_every_2,bitnet_int8_ternary_legacy_mono_update_every_2,bitnet_packed_ternary_mono_update_every_2
python -m samatnext_bit.bench --config configs/bitnet_large.yaml --modes fp_chainrule,fp_mono_update_every_2,bitnet_fake_ternary_mono_update_every_2,bitnet_int8_ternary_legacy_mono_update_every_2,bitnet_packed_ternary_mono_update_every_2
```

Tests: `8 passed`.

## Tiny 4x128

Result file: `runs/bitnet_tiny_20260624_235850/results.json`

| Mode | Initial CE | Final CE | CE delta | Tokens/sec | ms/step | Peak GB | Ternary -/0/+ | Backend | Native | Packed ternary | Bits/weight | Packed 1.58 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: | --- |
| fp_chainrule | 5.6809 | 0.4138 | 5.2671 | 117880.3 | 8.69 | 0.077 | n/a | fp | false | false | n/a | false |
| fp_mono_update_every_2 | 5.6809 | 1.1537 | 4.5272 | 206804.8 | 4.95 | 0.117 | n/a | fp | false | false | n/a | false |
| bitnet_fake_ternary_mono_update_every_2 | 5.6613 | 1.7348 | 3.9265 | 138382.0 | 7.40 | 0.124 | 36.8/26.4/36.8 | fake_ternary | false | false | 32.0 | false |
| bitnet_int8_ternary_legacy_mono_update_every_2 | 5.6612 | 1.7356 | 3.9256 | 110898.8 | 9.23 | 0.119 | 36.8/26.4/36.8 | int8_ternary_legacy | true | false | 8.0 | false |
| bitnet_packed_ternary_mono_update_every_2 | 5.6612 | 1.7352 | 3.9260 | 32735.4 | 31.28 | 0.124 | 36.8/26.4/36.8 | packed_ternary | true | true | 2.0 | false |

## Large 12x768

Result file: `runs/bitnet_large_20260624_235944/results.json`

| Mode | Initial CE | Final CE | CE delta | Tokens/sec | ms/step | Peak GB | Ternary -/0/+ | Backend | Native | Packed ternary | Bits/weight | Packed 1.58 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: | --- |
| fp_chainrule | 5.5378 | 1.0178 | 4.5200 | 6005.7 | 42.63 | 1.735 | n/a | fp | false | false | n/a | false |
| fp_mono_update_every_2 | 5.5378 | 1.2675 | 4.2703 | 10658.9 | 24.02 | 1.744 | n/a | fp | false | false | n/a | false |
| bitnet_fake_ternary_mono_update_every_2 | 5.7636 | 1.3549 | 4.4087 | 7881.4 | 32.48 | 2.455 | 37.4/25.2/37.4 | fake_ternary | false | false | 32.0 | false |
| bitnet_int8_ternary_legacy_mono_update_every_2 | 5.7635 | 1.3548 | 4.4087 | 7787.3 | 32.87 | 1.934 | 37.4/25.2/37.4 | int8_ternary_legacy | true | false | 8.0 | false |
| bitnet_packed_ternary_mono_update_every_2 | 5.7635 | 1.3548 | 4.4087 | 398.4 | 642.60 | 2.439 | 37.4/25.2/37.4 | packed_ternary | true | true | 2.0 | false |

## Required Answers

1. Is this ternary `{-1,0,+1}`? Yes.
2. Is storage packed below int8? Yes for `packed_ternary`: 2 bits per weight, 16 weights per int32 word.
3. Is it true 1.58-bit entropy packing or 2-bit ternary packing? It is 2-bit ternary packing. `packed_1p58bit=false`.
4. Does Triton read packed codes? Yes, the packed kernel loads packed words and extracts 2-bit codes.
5. Does packed ternary match fake ternary numerically? Yes within test tolerance on small CUDA matrices; tests compare packed vs fake forward.
6. Does packed ternary beat int8 ternary? No. Tiny: 32.7k vs 110.9k tok/s. Large: 398 vs 7787 tok/s.
7. Does packed ternary beat fake ternary? No. Tiny: 32.7k vs 138.4k tok/s. Large: 398 vs 7881 tok/s.
8. Does packed ternary beat FP? No. It is slower than both FP modes in these runs.
9. What is the next exact kernel step? Remove per-forward CPU packing and implement persistent GPU-side packed storage/update plus a tiled unpack+matmul kernel that accumulates multiple packed words per program with less scalar bit extraction overhead. Then add a native or fused STE backward path.

## Limitations

The v1 packed kernel proves the storage and decode path, but it is intentionally naive:

- Packing is done from shadow weights each forward.
- Backward is STE fallback, not packed/native.
- Unpacking happens inside a simple Triton matmul and is not yet optimized for throughput.
- The benchmark does not justify any speedup claim.
