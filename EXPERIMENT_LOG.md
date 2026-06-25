# Experiment Log

This log records the frozen `mono-forward-smoke-v1` experiments and supplemental arXiv-draft smoke runs.

## 1. Chain-Rule vs Mono 24-Layer

Config: `configs/chainrule_vs_mono_24layer.yaml`

Result file: `RESULTS_CHAINRULE_VS_MONO_24LAYER_v1.md`

Purpose: compare regular full chain-rule training with scheduled mono-forward/local updates in dense24 and sparse4/24 settings.

Outcome:

- Dense24 chain-rule reached the best validation CE: 2.2685.
- Dense24 mono improved throughput to 663,520 tok/s, 4.58x dense24 chain-rule.
- Sparse4/24 mono reached 2,556,872 tok/s, 17.63x dense24 chain-rule, but only used 4 active blocks.

## 2. Dense4 vs Sparse4/24 Reviewer Baseline

Config: `configs/dense4_vs_sparse4_24.yaml`

Result file: `RESULTS_DENSE4_VS_SPARSE4_24_v1.md`

Purpose: answer whether sparse4/24 is better than a normal dense4 model at matched active parameter count.

Outcome:

- Dense4 and sparse4/24 produced identical validation CE for matched training rules in this setup.
- Sparse4/24 has not yet shown extra quality beyond the normal 4-block active model.

## 3. Softmax vs Simple GDN 4-Active

Config: `configs/softmax_vs_gdn_4active.yaml`

Result file: `RESULTS_SOFTMAX_VS_GDN_4ACTIVE_v1.md`

Purpose: test whether a simple causal recurrent/linear mixer improves quality or speed over softmax when active compute is limited to 4 blocks.

Outcome:

- `simple_gdn` ran stably with finite gradients and no NaNs/Infs.
- `simple_gdn` did not beat softmax in chain-rule validation CE.
- `simple_gdn` slightly improved mono validation CE in the Tiny Shakespeare smoke run.
- `simple_gdn` was slower and used more memory than softmax.
- This is not official Gated DeltaNet.

## 4. Regular PyTorch 4-Layer Audit

Config: `configs/regular_torch_4layer_audit.yaml`

Result file: `RESULTS_REGULAR_TORCH_4LAYER_AUDIT_v1.md`

Purpose: answer whether a plain PyTorch CUDA 4-layer Transformer-style chain-rule loop reaches similar speed to the SamatNext dense4 path.

Outcome:

- Regular PyTorch 4-layer chain-rule reached 126,948 tok/s and final val CE 6.4704 on the Python-code token smoke setup.
- SamatNext dense4 chain-rule reached 142,473 tok/s and final val CE 6.4563.
- SamatNext dense4 mono reached 361,192 tok/s but worse final val CE 7.9963.

## 5. Practical Python-Code Token Smoke

Config: `configs/practical_python_code_training_smoke.yaml`

Result file: `RESULTS_PRACTICAL_PYTHON_CODE_TRAINING_SMOKE_v1.md`

Purpose: move beyond byte-level Tiny Shakespeare to hidden-512, seq-1024, BPE-tokenized Python-code text.

Outcome:

- Selected batch 24, seq 1024, vocab 16,000.
- Regular PyTorch chain-rule was stable.
- Dense4 mono remained faster than chain-rule but reached worse CE after 100 steps.
- Larger hidden/vocab/seq reduced the extreme tiny byte-level tok/s numbers.

## 6. BF16 Softmax vs Simple GDN Python 512

Config: `configs/bf16_softmax_vs_gdn_python_512.yaml`

Result file: `RESULTS_BF16_SOFTMAX_VS_GDN_PYTHON_512_v1.md`

Purpose: test softmax against the existing non-official `simple_gdn` mixer under BF16 hidden-512 Python-token conditions.

Outcome:

- BF16 was supported on the RTX 5070 Ti Laptop GPU runtime.
- `simple_gdn` was stable but slower, higher-memory, and not better CE than softmax.
- GDN work was stopped for this milestone.

## 7. Dense24 Mono VRAM Optimization

Config: `configs/dense24_mono_vram_optimized_python.yaml`

Result file: `RESULTS_DENSE24_MONO_VRAM_OPTIMIZED_PYTHON_v1.md`

Purpose: make dense 24/24 softmax mono-forward training practical under an 8GB laptop-GPU VRAM target.

Outcome:

- Hidden 512, seq 512, batch 8 fit under 8GB.
- As-is dense24 mono peaked at 7.459 GB and reached 82,909 tok/s.
- Optimized dense24 mono peaked at 4.641 GB and reached 83,587 tok/s.
- The key optimization was avoiding autograd graph construction on non-update mono-forward steps.

## 8. MBPP Dense24 Mono 500-Step Smoke

Config: `configs/dense24_mono_optimized_mbpp_500step.yaml`

Result file: `RESULTS_DENSE24_MONO_OPTIMIZED_MBPP_500STEP_v1.md`

Purpose: run the optimized dense24 mono path for 500 steps on MBPP-style Python problem/solution text.

Outcome:

- Source: Hugging Face `google-research-datasets/mbpp`, sanitized config.
- 427 examples total, 384 train, 43 validation.
- Train/validation tokens: 62,595 / 7,025.
- Dense24 softmax mono optimized trained 81,058,221 parameters.
- Throughput: 83,613 tok/s.
- Peak CUDA memory: 3.992 GB.
- Validation CE decreased from 8.7202 to 3.9844.
- Gradients remained finite and no NaN/Inf was detected.
- Samples remained noisy; no coding ability or MBPP score claim is made.

## Notes

No new architecture features are planned for this milestone. Future work should focus on longer controlled comparisons, stronger baselines for each new corpus, and clean accounting for update count, active parameters, and timed-region boundaries.
