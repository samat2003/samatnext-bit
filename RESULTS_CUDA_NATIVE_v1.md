# CUDA Native v1 Results

Date: 2026-06-24

Environment:

- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- Python: 3.12.3
- PyTorch: 2.11.0+cu128
- CUDA: 12.8
- Triton: 3.6.0

This implementation includes:

- FP decoder-only causal LM.
- BitNet-style ternary shadow-weight LM.
- `mono_update_every_2` trainer.
- Synthetic byte-level `tiny_code` dataset.
- Fake BitLinear backend using PyTorch matmul.
- Native BitLinear backend using a Triton int8 ternary forward kernel.

Important caveat:

- `native_kernel=true` only when the Triton CUDA path runs.
- `packed_1p58bit=false`.
- No real 1-bit or 1.58-bit speedup is claimed because the native kernel did not beat the fake/PyTorch baselines.

## Commands

```bash
python -m pytest -q
python -m samatnext_bit.bench --config configs/bitnet_tiny.yaml
python -m samatnext_bit.bench --config configs/bitnet_large.yaml
python -m samatnext_bit.bench --config configs/bitnet_32x512.yaml
python -m samatnext_bit.bench --config configs/bitnet_32x768_smoke.yaml --modes fp_chainrule,fp_mono_update_every_2,bitnet_fake_mono_update_every_2
```

## Tiny 4x128

Result file: `runs/bitnet_tiny_20260624_230133/results.json`

| Mode | Initial CE | Final CE | CE delta | Tokens/sec | ms/step | Peak GB | Params | Ternary -/0/+ | Gamma mean | Backend | Native | Packed 1.58-bit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| fp_chainrule | 5.6809 | 0.4138 | 5.2671 | 143506.5 | 7.14 | 0.077 | 0.87M | n/a | n/a | fp | false | false |
| fp_mono_update_every_2 | 5.6809 | 1.1537 | 4.5272 | 249167.1 | 4.11 | 0.117 | 0.87M | n/a | n/a | fp | false | false |
| bitnet_fake_chainrule | 5.6332 | 1.1902 | 4.4430 | 106247.2 | 9.64 | 0.080 | 0.87M | 36.2/27.5/36.3 | 0.0456 | fake | false | false |
| bitnet_fake_mono_update_every_2 | 5.6332 | 1.8345 | 3.7987 | 146007.1 | 7.01 | 0.124 | 0.87M | 36.7/26.5/36.8 | 0.0422 | fake | false | false |
| bitnet_native_chainrule | 5.6331 | 1.1905 | 4.4426 | 91810.2 | 11.15 | 0.077 | 0.87M | 36.2/27.5/36.3 | 0.0456 | native | true | false |
| bitnet_native_mono_update_every_2 | 5.6331 | 1.8349 | 3.7983 | 134343.3 | 7.62 | 0.119 | 0.87M | 36.7/26.5/36.8 | 0.0422 | native | true | false |

## Large 12x768

Result file: `runs/bitnet_large_20260624_230156/results.json`

| Mode | Initial CE | Final CE | CE delta | Tokens/sec | ms/step | Peak GB | Params | Ternary -/0/+ | Gamma mean | Backend | Native | Packed 1.58-bit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| fp_chainrule | 5.5378 | 1.0178 | 4.5200 | 5928.2 | 43.18 | 1.735 | 85.53M | n/a | n/a | fp | false | false |
| fp_mono_update_every_2 | 5.5378 | 1.2675 | 4.2703 | 10366.8 | 24.69 | 1.744 | 85.53M | n/a | n/a | fp | false | false |
| bitnet_fake_chainrule | 5.7699 | 1.1997 | 4.5702 | 5650.8 | 45.30 | 1.934 | 85.53M | 37.4/25.3/37.4 | 0.0160 | fake | false | false |
| bitnet_fake_mono_update_every_2 | 5.7699 | 1.3639 | 4.4060 | 8018.7 | 31.93 | 2.455 | 85.53M | 37.4/25.2/37.4 | 0.0159 | fake | false | false |
| bitnet_native_chainrule | 5.7698 | 1.1996 | 4.5702 | 5157.0 | 49.64 | 1.734 | 85.53M | 37.4/25.3/37.4 | 0.0160 | native | true | false |
| bitnet_native_mono_update_every_2 | 5.7698 | 1.3644 | 4.4053 | 7175.6 | 35.68 | 1.934 | 85.53M | 37.4/25.2/37.4 | 0.0159 | native | true | false |

## Depth 32x512

Result file: `runs/bitnet_32x512_20260624_230223/results.json`

| Mode | Initial CE | Final CE | CE delta | Tokens/sec | ms/step | Peak GB | Params | Ternary -/0/+ | Gamma mean | Backend | Native | Packed 1.58-bit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| fp_chainrule | 5.6399 | 1.4384 | 4.2015 | 1665.7 | 76.84 | 2.078 | 101.17M | n/a | n/a | fp | false | false |
| fp_mono_update_every_2 | 5.6399 | 1.9373 | 3.7027 | 3203.5 | 39.96 | 2.080 | 101.17M | n/a | n/a | fp | false | false |
| bitnet_fake_chainrule | 5.5198 | 1.3947 | 4.1251 | 1522.4 | 84.08 | 2.290 | 101.17M | 37.5/25.1/37.4 | 0.0194 | fake | false | false |
| bitnet_fake_mono_update_every_2 | 5.5198 | 1.5446 | 3.9752 | 2314.7 | 55.30 | 2.879 | 101.17M | 37.5/25.1/37.4 | 0.0194 | fake | false | false |
| bitnet_native_chainrule | 5.5198 | 1.3951 | 4.1247 | 1359.8 | 94.13 | 2.080 | 101.17M | 37.5/25.1/37.4 | 0.0194 | native | true | false |
| bitnet_native_mono_update_every_2 | 5.5198 | 1.5449 | 3.9749 | 1860.4 | 68.80 | 2.180 | 101.17M | 37.5/25.1/37.4 | 0.0194 | native | true | false |

## 32x768 Smoke

Result file: `runs/bitnet_32x768_smoke_20260624_230244/results.json`

| Mode | Initial CE | Final CE | CE delta | Tokens/sec | ms/step | Peak GB | Params | Ternary -/0/+ | Gamma mean | Backend | Native | Packed 1.58-bit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| fp_chainrule | 5.5298 | 2.4837 | 3.0461 | 1269.2 | 100.85 | 4.565 | 227.26M | n/a | n/a | fp | false | false |
| fp_mono_update_every_2 | 5.5298 | 5.1475 | 0.3822 | 2382.9 | 53.72 | 4.568 | 227.26M | n/a | n/a | fp | false | false |
| bitnet_fake_mono_update_every_2 | 5.6522 | 1.9448 | 3.7074 | 1758.2 | 72.80 | 6.077 | 227.26M | 37.5/25.0/37.5 | 0.0158 | fake | false | false |

## Interpretation

CUDA matches the MLX learning behavior in the narrow required sense: FP and BitNet losses decrease, and BitNet mono also learns at 32x768 smoke scale.

The native Triton kernel did run in the tiny, large, and 32x512 benchmarks. It currently uses int8 ternary weights generated from shadow weights each forward, with PyTorch code in the custom autograd backward. It is not packed 1.58-bit storage.

Native did not beat fake/PyTorch throughput in these runs, so this is a correctness-oriented first native path, not a speedup result.

## Next Recommendation

Move the native path from per-forward int8 ternary encoding to persistent packed ternary storage plus a fused forward kernel. Then add a native backward or a more efficient STE gradient path before re-benchmarking speed claims.
