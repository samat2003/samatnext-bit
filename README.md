# samatnext-bit

PyTorch CUDA port of the MLX BitNet + `mono_update_every_2` experiment.

This repo intentionally separates three claims:

- FP baselines use normal PyTorch layers.
- BitNet fake uses trainable shadow weights, ternary active weights, and normal PyTorch matmul.
- BitNet native uses a Triton int8 ternary forward kernel for `BitLinear`.

The current native path is not packed 1.58-bit storage. It reports `native_kernel=true` when the Triton CUDA path runs and `packed_1p58bit=false`.

## Run

```bash
python -m pytest -q
python -m samatnext_bit.bench --config configs/bitnet_tiny.yaml
python -m samatnext_bit.bench --config configs/bitnet_large.yaml
python -m samatnext_bit.bench --config configs/bitnet_32x512.yaml
```

The benchmark logs CE, throughput, CUDA memory, parameters, ternary distribution, gamma mean, backend, native-kernel status, and packed-storage status.
