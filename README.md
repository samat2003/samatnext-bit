# samatnext-bit

Scheduled mono-forward-style training updates for small decoder benchmarks.

This repository is a CUDA/PyTorch research prototype for studying whether a decoder must run full chain-rule backpropagation on every training step. It is not a new production LLM and does not claim to replace backpropagation generally.

## What This Is

`samatnext-bit` is a simple, reproducible algorithm repo. It compares standard decoder training against scheduled mono-forward-style update rules in a controlled benchmark harness.

The work is Forward-Forward / Mono-Forward inspired. The contribution here is the decoder-training implementation, the scheduling knobs, and the chain-rule vs scheduled-update audit harness.

## Why It Exists

Full chain-rule backpropagation updates through the whole network every step. That is the normal baseline and it remains the quality reference.

This repo asks a narrower question: can some backward/update work be scheduled less frequently while still making useful progress? The measured tradeoff is:

- final cross entropy
- tokens/sec
- cross-entropy improvement per minute

## Algorithm In Plain Language

Standard chain-rule mode runs a normal forward pass, computes the final token loss, backpropagates through the full decoder, and applies an optimizer update every step.

Scheduled mono-forward-style modes still run the model forward, but they schedule when full backward/update work is applied.

- Quality mode uses conservative scheduling, for example `UE1`.
- Speed mode uses aggressive scheduling, for example `UE4` or higher.
- Higher `UE` settings usually run faster but may finish with worse loss.

The repo reports the tradeoff instead of treating speed alone as success.

## Fast Synthetic Smoke Test

This test uses generated/static CUDA token batches and does not need an external dataset. It is useful for checking the algorithm path and the fastest result class.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
python -m pytest -q
python -m samatnext_bit.bench_speed --config configs/speed_500k_tiny.yaml
```

Historical result class:

| Mode | Data | Scale | Throughput |
|---|---|---:|---:|
| Standard chain-rule baseline | synthetic CUDA tokens | <1M active params | ~198K tok/s |
| Scheduled mono-forward path | synthetic CUDA tokens | <1M active params | ~3.5M tok/s |
| Speedup | same benchmark | same scale | ~17.6x |

Caveats: this is under 1M active parameters, synthetic/static CUDA batches, and not comparable to dense LLM training.

## Dense 313M/369M Comparison

The dense audit uses a true dense decoder class with about 313M/369M parameters depending on the exact config/report, standard AMP, fused AdamW when available, and Flash-capable PyTorch SDPA.

Run the 1500-step Python HF mix comparisons when the tokenized dataset is available:

```bash
python scripts/bench_dense313m_loss_recovery.py --config configs/dense313m_python_hf_mix_quality_1500.yaml
python scripts/bench_dense313m_loss_recovery.py --config configs/dense313m_python_hf_mix_1500.yaml
```

The Python HF mix data is not included in this repo. If it is unavailable, use the MBPP smoke path/configs as a local smoke check only, not as a coding benchmark.

Single-run 1500-step Python HF mix audit:

- chain-rule final val CE: `5.4794`, `5,074 tok/s`, CE/min `0.9968`
- mono `UE1` quality mode final val CE: `5.4658`, `5,343 tok/s`, CE/min `1.0528`
- mono `UE4` speed mode final val CE: `6.0868`, CE/min `2.7024`, around the 3x-4x faster result class

More seeds are needed before making stronger claims.

## Use Your Own Dataset

Synthetic data is fine if you only want to test speed. Use your own tokenized dataset for meaningful loss comparisons.

Supported pretokenized layout:

```text
your_dataset/
  train.bin
  val.bin
  metadata.json
  tokenizer.json
```

`metadata.json` should describe the tokenizer, token dtype, vocabulary size, train token count, validation token count, and source corpus. Raw datasets, pretrained models, and external corpora are not included.

Point a dense config at your dataset directory with `dataset_path`, `train_bin`, `val_bin`, `metadata_file`, and `tokenizer_file`.

## Results

| Experiment | Scale | Data | Best result | Caveat |
|---|---:|---|---|---|
| Fast synthetic smoke | <1M active params | synthetic CUDA tokens | ~3.5M tok/s, ~17.6x vs chain-rule baseline | not dense LLM training |
| Dense Python quality mode | ~369.9M params | Python HF mix | mono UE1 CE 5.4658 vs chain-rule 5.4794 | single seed, 1500 steps |
| Dense Python speed mode | ~369.9M params | Python HF mix | best CE/min 2.7024 | worse final CE |
| Dense scale ceiling | ~313M/369M params | real/smoke corpora | dense 600K tok/s not reached | hardware/compute bound |

## Limitations

- This is an algorithm prototype for scheduled mono-forward-style training updates.
- It is Forward-Forward / Mono-Forward inspired, not a claim that backpropagation is obsolete.
- It does not claim SOTA, HumanEval/MBPP score, a trained coding model, or a production-ready LLM.
- It does not claim 600K tok/s dense training.
- Dense Python results are single-run audits; more seeds and more corpora are needed.
- Synthetic speed tests use static/generated CUDA batches and are not comparable to dense LLM training.
- Raw datasets and large binary token files are intentionally excluded.
- Historical BitNet/ternary files in the repo do not imply real 1.58-bit speedups; packed ternary kernels would be required for that claim.

## License

Code in this repository is released under the MIT License. MIT applies only to this repo's code. Datasets, pretrained models, tokenizers, and external corpora keep their original licenses.