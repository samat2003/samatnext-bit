# Claims and Limitations

## Supported Claims

This repository is a CUDA/PyTorch research prototype for scheduled mono-forward-style training updates in decoder models.

The training rules are Forward-Forward / Mono-Forward inspired, but the repo contribution is the decoder implementation and controlled benchmark harness.

Supported claims:

- algorithm prototype
- scheduled mono-forward-style training updates
- controlled chain-rule vs mono-forward audits
- reproducible synthetic smoke tests
- dense decoder comparison harness
- speed-quality tradeoff measurement using final CE, tok/s, and CE/min

The 1500-step Python HF mix audit found:

- chain-rule final validation CE `5.4794`, `5,074 tok/s`, CE/min `0.9968`
- mono `UE1` quality mode final validation CE `5.4658`, `5,343 tok/s`, CE/min `1.0528`
- mono `UE4` speed mode final validation CE `6.0868`, CE/min `2.7024`

These are single-run audit results, not broad claims.

## Forbidden Claims

Do not describe this repo as:

- SOTA
- replacing backpropagation
- beating backpropagation generally
- a trained coding model
- a HumanEval or MBPP score claim
- 600K tok/s dense training
- a production-ready LLM
- evidence of real 1.58-bit speedup

## Limitations

Full chain-rule backpropagation remains the baseline and quality reference.

Quality mode scheduling, such as `UE1`, is conservative and may match or slightly beat a single chain-rule run under one short audit. That does not prove a general advantage.

Speed mode scheduling, such as `UE4` or higher, can improve CE/min and tok/s, but the final CE can be worse.

The fast synthetic smoke tests use generated/static CUDA token batches under 1M active parameters. They are useful for reproducibility and speed-path checks, but they are not dense LLM training.

Dense Python results are single-seed, short-run audits. More seeds, longer runs, more datasets, and independent reproduction are needed.

Raw datasets, pretrained models, and large token files are not included in the repo.

Historical BitNet/ternary code paths are not claims of real 1.58-bit acceleration. Real low-bit speedups require packed ternary kernels.

Throughput varies with GPU, driver, CUDA/PyTorch build, thermals, power limits, and background load.
