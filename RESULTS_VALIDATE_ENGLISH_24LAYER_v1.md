# RESULTS_VALIDATE_ENGLISH_24LAYER_v1
300-step real-English validation experiment on RTX 5070 Ti Laptop GPU.
## Freeze Status
The previous English 24-layer smoke milestone was committed locally before this validation work. Commit: `87ce7cfc01b7d8ba3a47b8fa6e192e1fb9b0fc3e`. Tag: `english-24layer-speed-smoke-v1`. No remote was configured, so push was not performed.
## Commands
```bash
git status
python -m pytest -q
git init
git add .gitignore AGENTS.md README.md pyproject.toml requirements.txt configs data src tests RESULTS_*.md
git commit -m "freeze english 24layer speed smoke"
git tag english-24layer-speed-smoke-v1
python -m pytest -q
python -m samatnext_bit.bench_speed --config configs/validate_english_24layer.yaml
```
## Dataset
Dataset source: downloaded Tiny Shakespeare data/english_validation.txt. Total tokens: 1,115,394. Train tokens: 1,003,854. Validation tokens: 111,540. Byte vocab size: 256. Split: 90/10. Train and validation batches were preloaded onto CUDA before timing; batch sampling, loading, tokenization, and validation are excluded from training speed.
## Summary
| track | dense_or_sparse | layers | active | passes | calls | mode | batch | seq | steps | updates | tok/s | ms/step | p50 | p90 | p99 | peak GB | grad mean | grad max | finite | NaN/Inf |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| dense_24_24_quality_speed | dense | 24 | 24 | 1 | 24 | `fp_mono_update_every_16` | 96 | 256 | 300 | 19 | 693403 | 35.304 | 24.628 | 30.739 | 186.731 | 7.075 | 1.617 | 2.476 | true | false |
| dense_24_24_quality_speed_b64 | dense | 24 | 24 | 1 | 24 | `fp_mono_update_every_16` | 64 | 256 | 300 | 19 | 660764 | 24.699 | 17.513 | 21.163 | 140.158 | 4.755 | 1.616 | 2.484 | true | false |
| sparse_24_active4_quality_speed | sparse/logical | 24 | 4 | 1 | 4 | `fp_mono_update_every_8` | 64 | 256 | 300 | 38 | 2775294 | 5.880 | 3.239 | 21.430 | 26.955 | 0.969 | 1.221 | 1.586 | true | false |

## Checkpoints
### dense_24_24_quality_speed
| step | train CE | val CE | train ppl | val ppl |
|---:|---:|---:|---:|---:|
| 0 | 5.6576 | 5.6616 | 286.47 | 287.62 |
| 50 | 4.3644 | 4.3841 | 78.60 | 80.17 |
| 100 | 4.1760 | 4.1820 | 65.10 | 65.50 |
| 150 | 4.0020 | 4.0288 | 54.71 | 56.19 |
| 200 | 3.8486 | 3.8746 | 46.93 | 48.16 |
| 250 | 3.7343 | 3.7322 | 41.86 | 41.77 |
| 300 | 3.6107 | 3.6096 | 36.99 | 36.95 |

### dense_24_24_quality_speed_b64
| step | train CE | val CE | train ppl | val ppl |
|---:|---:|---:|---:|---:|
| 0 | 5.6603 | 5.6611 | 287.24 | 287.46 |
| 50 | 4.3737 | 4.3817 | 79.34 | 79.97 |
| 100 | 4.1696 | 4.1813 | 64.69 | 65.45 |
| 150 | 3.9964 | 4.0286 | 54.40 | 56.18 |
| 200 | 3.8549 | 3.8753 | 47.22 | 48.20 |
| 250 | 3.7345 | 3.7329 | 41.87 | 41.80 |
| 300 | 3.5827 | 3.6119 | 35.97 | 37.04 |

### sparse_24_active4_quality_speed
| step | train CE | val CE | train ppl | val ppl |
|---:|---:|---:|---:|---:|
| 0 | 5.7326 | 5.7337 | 308.77 | 309.10 |
| 50 | 4.8605 | 4.8699 | 129.09 | 130.31 |
| 100 | 4.2648 | 4.2737 | 71.15 | 71.78 |
| 150 | 3.8888 | 3.9288 | 48.85 | 50.84 |
| 200 | 3.6836 | 3.7067 | 39.79 | 40.72 |
| 250 | 3.5261 | 3.5176 | 33.99 | 33.70 |
| 300 | 3.3619 | 3.3861 | 28.84 | 29.55 |

## Samples
### dense_24_24_quality_speed
Sample 1:
```text
The �r

R U e� sE�m �ofre 'wd n m�e>���s t f:s re to
o t t�mrit��-�h�
S ���Jt:-uml
d �
Q s at5C1e�$I m�
u� a t�arek�Xskezt
```
Sample 2:
```text
The -AsK w�O te �rtot�mQs�Ye
y o�u
Ithe Xth �e ye l�}-�E s 0 th��en�e t y m��s�mݽ t��
Ak�q�n Tve th, t �le�

o[: �RۦI �e�euo
```
Sample 3:
```text
The Rw 2 |�`b�, letb,�� ȎU vf;skN/ �kAS hegl�izme mus u�e �e bou�Zo%ath�5s t!:�� o.�e �I s '_ ȵ�o�fw��t
y ��e  �d sthe

�� tin
```
### dense_24_24_quality_speed_b64
Sample 1:
```text
The �r

R U e� sE�m �ofre 'wd n m�e>���s t f:s re to
o t t�mrit��-�h�
S ���Jt:-uml
d �
Q s at5C1e�$I m�
u� a t�arek�XsUezt
```
Sample 2:
```text
The -AsK w�O te �rtot�mQs�Ye
y o�u
Ithe Xth �e ye ld}-�t s 0 th��en�e t y m��s�mݽ t��
Ak�q�n Tve th, t �le�

o[: �RۦI �e�euo
```
Sample 3:
```text
The Rw 2 |�`b�, letb,�� ȎU vf;sk y �kAS hegl�izme mus u�e �e bou�Zo%ath�5s t!:�� o.�e �I s '_ ȵ�o�fw��t
y ��e  �d sthe

�� tin
```
### sparse_24_active4_quality_speed
Sample 1:
```text
The ��r
�on an, �s E.m Rofrf>'wd n
;ice cn.s t fhirr�oto,o s Lororoien �e�h

ha� \Jpsuupltdo sQ�sana5o1e t �ff
uu a n�cd knXskeea
```
Sample 2:
```text
The -Aro wtO t e �rtoĸ Qys Ye
y iod 
 theaXt  : i yvold} 1o ei0 tr�nenie t ywe �sruhe ted Ah�qSn indo e, t thee�
ao:: �Rr�I mee uo
```
Sample 3:
```text
The Rw � | `b�, ictb,ho  nUe vfh;su y OkAh hagl��lrst us uiei.e bound eath�5. to:nL o.rn �I syor  �d oorwern
 A�owme ed  t  
f�aitin
```
## Answers
1. The current smoke-test state was committed and tagged locally.
2. Commit hash: `87ce7cfc01b7d8ba3a47b8fa6e192e1fb9b0fc3e`; tag: `english-24layer-speed-smoke-v1`. No remote was configured, so push was skipped.
3. Dense track `dense_24_24_quality_speed` kept improving: train CE 5.6576 -> 3.6107, validation CE 5.6616 -> 3.6096.
3. Dense track `dense_24_24_quality_speed_b64` kept improving: train CE 5.6603 -> 3.5827, validation CE 5.6611 -> 3.6119.
4. Sparse 4/24 kept improving: train CE 5.7326 -> 3.3619, validation CE 5.7337 -> 3.3861.
5. Best final validation loss/perplexity: `sparse_24_active4_quality_speed` with val CE 3.3861, val ppl 29.55.
6. Best speed: `sparse_24_active4_quality_speed` at 2775294 tok/s.
7. Sparse 4/24 did not plateau worse in this short run; it reached the best final validation CE among these tracks. This is a short byte-level validation, not a language-quality claim.
8. No track showed NaNs/Infs.
9. Gradients were finite for every track.
10. Best honest claim: on Tiny Shakespeare byte-level validation for 300 steps, sparse/logical 4/24 was much faster and also reached lower validation CE than the dense 24/24 tracks, while remaining explicitly sparse/logical and not dense.
11. Exact next experiment: run 2K-5K steps on the same Tiny Shakespeare split with held-out validation, add deterministic validation sample seeds, and compare dense 24/24 vs sparse 4/24 at equal optimizer-update counts and equal wall-clock budget.
