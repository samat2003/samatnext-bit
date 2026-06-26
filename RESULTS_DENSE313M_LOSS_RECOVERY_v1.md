# RESULTS_DENSE313M_LOSS_RECOVERY_v1

New results are appended by this experiment only; earlier benchmark files are not overwritten.

## Preserved Baselines

| Baseline | Tok/s | Peak GB | Initial val CE | Final val CE | Steps |
|---|---:|---:|---:|---:|---:|
| chain-rule baseline | 7,285.7 | 11.309 | 8.7542 | 3.7325 | 100 |
| mono-forward UE=4 baseline | 14,233.5 | 6.297 | 8.7542 | 4.4964 | 100 |
| fused AdamW mono speed baseline | 21,927.3 | n/a | n/a | n/a | n/a |

## Sweep Results

| Track | Dense active | Updates | Anchors | Aux | Tok/s | Mean ms | p50/p90/p99 ms | Peak GB | Init val CE | Final val CE | Delta | CE/min | PPL |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense313m_chainrule_reference | yes | 100 | 0 | False | 5,166.0 | 197.83 | 196.6/211.2/219.0 | 7.480 | 8.7595 | 3.7164 | 5.0431 | 15.2472 | 41.12 |
| dense313m_mono_ue4_reference | yes | 25 | 0 | False | 13,776.3 | 74.19 | 29.1/216.3/233.4 | 7.480 | 8.7595 | 4.4436 | 4.3159 | 34.6092 | 85.08 |
| dense313m_mono_ue2 | yes | 50 | 0 | False | 8,431.7 | 121.21 | 199.6/222.6/228.9 | 7.480 | 8.7595 | 4.0044 | 4.7552 | 23.4169 | 54.84 |
| dense313m_mono_ue4_anchor8 | yes | 25 | 12 | False | 13,919.3 | 73.42 | 27.4/215.6/230.9 | 7.480 | 8.7595 | 4.3089 | 4.4507 | 36.0940 | 74.36 |
| dense313m_mono_ue4_anchor16 | yes | 25 | 6 | False | 13,788.7 | 74.12 | 28.1/217.8/229.2 | 7.480 | 8.7595 | 4.3089 | 4.4507 | 35.7030 | 74.36 |
| dense313m_chainrule_warmup_then_mono | yes | 40 | 0 | False | 9,607.0 | 106.38 | 31.3/230.4/236.8 | 7.480 | 8.7595 | 4.1035 | 4.6560 | 26.1114 | 60.55 |
| dense313m_mono_ue4_aux_loss | yes | 25 | 0 | True | 13,292.5 | 76.89 | 28.4/223.4/248.4 | 7.480 | 8.7595 | 4.4660 | 4.2936 | 33.2391 | 87.01 |
| dense313m_mono_ue4_anchor8_aux_loss | yes | 25 | 12 | True | 13,166.2 | 77.62 | 27.5/235.4/249.1 | 7.480 | 8.7595 | 4.4660 | 4.2936 | 32.9239 | 87.01 |

## Optional Longer Run

| Track | Steps | Tok/s | Peak GB | Initial val CE | Final val CE | Delta | CE/min |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense313m_mono_ue4_anchor8 | 500 | 13,629.0 | 7.486 | 8.7595 | 3.6193 | 5.1402 | 8.1399 |

The optional longer run is not an equal-step comparison against the 100-step chain-rule reference.

## Answers

1. Did update_every=2 improve loss compared to update_every=4? yes.
2. Did periodic full chain-rule anchor steps improve loss? yes versus UE4 in this run: anchor8/anchor16 final val CE 4.3089/4.3089 versus UE4 4.4436; caveat, anchors overlap the UE4 update cadence here.
3. Did chain-rule warmup improve mono loss? yes: 4.1035 versus UE4 4.4436.
4. Did auxiliary losses help? no: aux final val CE 4.4660 versus UE4 4.4436.
5. Best final validation CE: dense313m_chainrule_reference (3.7164).
6. Best CE improvement per minute: dense313m_mono_ue4_anchor8 (36.0940).
7. Best preserved speedup over chain-rule: dense313m_mono_ue4_anchor8 (2.69x over chain-rule); raw fastest was dense313m_mono_ue4_anchor8 (13,919.3 tok/s).
8. Did any 100-step method match or beat chain-rule final CE? no.
9. Remaining 100-step CE gap to chain-rule for best non-chain method: 0.2879.
10. Strongest honest claim: plain mono-forward is faster and uses less memory, while this sweep measures whether UE2, anchor updates, warmup, or aux CE recover loss without claiming parity unless observed.
11. Main limitation: this is a 100-step MBPP smoke-corpus experiment; it does not prove long-run convergence or coding ability.

## Raw JSON

`runs/dense313m_loss_recovery_20260625_215810/results.json`
