# Experiment Log

This log records the milestone experiments frozen in `mono-forward-smoke-v1`.

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
- Dense4 chain-rule and sparse4/24 chain-rule both ended at 2.4463 validation CE.
- Dense4 mono and sparse4/24 mono both ended at 3.0240 validation CE.
- Sparse4/24 has not yet shown extra quality beyond the normal 4-block active model.

## 3. Softmax vs Simple GDN 4-Active

Config: `configs/softmax_vs_gdn_4active.yaml`

Result file: `RESULTS_SOFTMAX_VS_GDN_4ACTIVE_v1.md`

Purpose: test whether a simple causal recurrent/linear mixer improves quality or speed over softmax when active compute is limited to 4 blocks.

Outcome:

- `simple_gdn` ran stably with finite gradients and no NaNs/Infs.
- `simple_gdn` did not beat softmax in chain-rule validation CE.
- `simple_gdn` slightly improved mono validation CE: 3.0128 vs 3.0240.
- `simple_gdn` was slower and used more memory than softmax.
- This is not official Gated DeltaNet.

## Notes

No new architecture features are planned for this milestone. Future work should focus on cleaner local update accounting, longer runs, and better optimized recurrent/linear mixers only after this release is frozen.
