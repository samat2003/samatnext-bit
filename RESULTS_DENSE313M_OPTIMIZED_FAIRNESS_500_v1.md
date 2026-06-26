# RESULTS_DENSE313M_OPTIMIZED_FAIRNESS_500_v1

Strict optimized fairness audit for the true dense313M model.

This is an MBPP smoke loss-mechanics audit, not a coding benchmark. Dataset is `mbpp_smoke` only: 62,595 train tokens and 7,025 validation tokens.

Lower validation CE is better. Higher CE/min means faster validation CE improvement per elapsed minute under the same 500-step condition.

## Model/Data Lock

| Field | Value |
|---|---:|
| hidden | 1024 |
| active layers | 24 |
| heads | 16 |
| seq len | 512 |
| batch size | 2 |
| dtype | float16 |
| precision mode | amp_fp16 |
| configured parameter dtype | fp32 |
| attention impl | sdpa |
| seed | 42 |

## Optimization Status

| Item | Status |
|---|---|
| PyTorch SDPA path | True |
| Manual attention used in main comparison | False |
| Flash SDPA enabled | True |
| Mem-efficient SDPA enabled | True |
| Math SDPA enabled | True |
| cuDNN SDPA enabled | True |
| FlashAttention available | True |
| Forced Flash probe attempted | True |
| Forced Flash probe succeeded for dense313M shape | True |
| Forced Flash probe shape | [2, 16, 512, 64] |
| Forced Flash probe error | None |

Do not treat this as a FlashAttention-optimized baseline unless the forced Flash probe succeeded for the actual dense313M shape above.

## Fairness Checks

| Track | Same init seed | Same data order seed | Same batch/seq | Same dtype | Precision mode | Param dtype | Optimizer state dtype | Autocast dtype | GradScaler | Same attention | Fused AdamW | Pretokenized | Tokenizer outside timing | Validation/gen outside timing | CUDA sync timing | Dataset check |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chainrule_amp_optimized_500 | True | True | True | True | amp_fp16 | float32 | float32 | float16 | True | True | True | True | True | True | True | True |
| mono_ue2_amp_optimized_500 | True | True | True | True | amp_fp16 | float32 | float32 | float16 | True | True | True | True | True | True | True | True |
| mono_ue4_amp_optimized_500 | True | True | True | True | amp_fp16 | float32 | float32 | float16 | True | True | True | True | True | True | True | True |
| mono_ue2_anchor17_amp_optimized_500 | True | True | True | True | amp_fp16 | float32 | float32 | float16 | True | True | True | True | True | True | True | True |
| mono_ue4_anchor17_amp_optimized_500 | True | True | True | True | amp_fp16 | float32 | float32 | float16 | True | True | True | True | True | True | True | True |

## 500-Step Results

| Track | Steps | Final val CE | CE drop | CE/min | Tok/s | Elapsed s | Final PPL | Peak alloc GB | Peak reserved GB | Updates | Anchors | Skipped collisions | Anchors separate | Grad finite | NaN/Inf | Scaler value |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|
| chainrule_amp_optimized_500 | 500 | 2.9660 | 5.7936 | 3.3550 | 4,948.4 | 103.61 | 19.41 | 7.486 | 8.651 | 500 | 0 | 0 | True | True | False | 65536.0 |
| mono_ue2_amp_optimized_500 | 500 | 3.5003 | 5.2593 | 5.2078 | 8,484.9 | 60.59 | 33.12 | 7.487 | 8.647 | 250 | 0 | 0 | True | True | False | 65536.0 |
| mono_ue4_amp_optimized_500 | 500 | 3.6519 | 5.1077 | 9.4664 | 15,905.9 | 32.37 | 38.55 | 7.487 | 8.647 | 125 | 0 | 0 | True | True | False | 65536.0 |
| mono_ue2_anchor17_amp_optimized_500 | 500 | 3.4426 | 5.3169 | 6.0140 | 9,680.1 | 53.05 | 31.27 | 7.487 | 8.647 | 265 | 15 | 14 | True | True | False | 65536.0 |
| mono_ue4_anchor17_amp_optimized_500 | 500 | 3.6318 | 5.1277 | 8.4856 | 14,191.8 | 36.26 | 37.78 | 7.487 | 8.647 | 147 | 22 | 7 | True | True | False | 65536.0 |

## Interpretation

- Best final CE: chainrule_amp_optimized_500 (2.9660).
- Best CE/min: mono_ue4_amp_optimized_500 (9.4664).
- Do not claim mono is better unless it wins final CE or CE/min under this same 500-step condition.
- The only intended track difference is the training rule/update schedule; config, data, tokenizer, dtype, precision mode, parameter dtype, attention path, and fused AdamW settings are locked above.

## Raw JSON

`runs/dense313m_optimized_fairness_500_20260626_113058/results.json`
