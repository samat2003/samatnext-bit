# RESULTS_DENSE313M_LOSS_RECOVERY_v1

New results are appended by this experiment only; earlier benchmark files are not overwritten.

## Preserved Baselines

| Baseline | Tok/s | Peak GB | Initial val CE | Final val CE | Steps |
|---|---:|---:|---:|---:|---:|
| chain-rule baseline | 7,285.7 | 11.309 | 8.7542 | 3.7325 | 100 |
| mono-forward UE=4 baseline | 14,233.5 | 6.297 | 8.7542 | 4.4964 | 100 |
| fused AdamW mono speed baseline | 21,927.3 | n/a | n/a | n/a | n/a |

## Corrected 500-Step Results

| Track | Dense active | Steps | Updates | Normal mono | Anchors | Skipped anchor collisions | Anchors separate | Tok/s | Peak alloc GB | Peak reserved GB | Init val CE | Final val CE | Delta | CE/min |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| chainrule_500 | yes | 500 | 500 | 0 | 0 | 0 | True | 4,955.0 | 7.486 | 8.651 | 8.7595 | 2.9650 | 5.7945 | 3.3602 |
| mono_ue2_500 | yes | 500 | 250 | 250 | 0 | 0 | True | 8,324.3 | 7.487 | 8.647 | 8.7595 | 3.4821 | 5.2774 | 5.1301 |
| mono_ue4_500 | yes | 500 | 125 | 125 | 0 | 0 | True | 14,096.6 | 7.487 | 8.647 | 8.7595 | 3.6194 | 5.1402 | 8.4330 |
| mono_ue4_anchor17_500 | yes | 500 | 147 | 125 | 22 | 7 | True | 12,534.7 | 7.487 | 8.647 | 8.7595 | 3.6323 | 5.1272 | 7.4850 |
| mono_ue2_anchor17_500 | yes | 500 | 265 | 250 | 15 | 14 | True | 7,196.0 | 7.487 | 8.647 | 8.7595 | 3.3810 | 5.3786 | 4.5212 |

## Equal Wall-Clock Results

| Track | Budget min | Actual steps | Updates | Anchors | Skipped anchor collisions | Anchors separate | Tok/s | Peak alloc GB | Peak reserved GB | Final val CE | CE/min |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| chainrule_equal_time | 3.00 | 500 | 500 | 0 | 0 | True | 4,649.1 | 7.487 | 8.647 | 2.9713 | 3.1492 |
| best_mono_anchor_equal_time | 3.00 | 500 | 147 | 22 | 7 | True | 12,347.2 | 7.487 | 8.647 | 3.6326 | 7.3734 |

Aux-loss tracks were dropped from this corrected audit.

## Answers

1. Did update_every=2 improve loss compared to update_every=4? yes.
2. Did periodic full chain-rule anchor steps improve loss? no for UE4: anchor17 final val CE 3.6323 versus UE4 3.6194; performed anchors were separate=True.
3. Did UE2 anchor17 improve over UE2? yes for UE2: anchor17 3.3810 versus UE2 3.4821.
4. Did auxiliary losses help? not included in this corrected audit.
5. Best final validation CE: chainrule_500 (2.9650).
6. Best CE improvement per minute: mono_ue4_500 (8.4330).
7. Best preserved speedup over chain-rule: mono_ue4_500 (2.84x over chain-rule); raw fastest was mono_ue4_500 (14,096.6 tok/s).
8. Did any 500-step mono method match or beat chain-rule final CE? no.
9. Remaining 500-step CE gap to chain-rule for best non-chain method: 0.4159.
10. Strongest honest claim: this corrected audit measures non-overlapping anchor updates; mono-forward remains faster per step, but loss parity must be judged from the measured rows.
11. Main limitation: MBPP smoke is small and 500 steps is still a short-run training audit; generated text is not evidence of coding ability.

## Raw JSON

`runs/dense313m_loss_recovery_20260626_105823/results.json`
