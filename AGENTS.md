# AGENTS.md

## Mission
CUDA/PyTorch port of the MLX BitNet + mono-forward experiment.

Goal: reproduce MLX results on RTX before custom packed ternary kernels.

Do not add DFA. Do not redesign architecture. Do not claim real 1.58-bit speedup until packed ternary kernels exist.

## Codex Rules
Before coding, give a short implementation plan:
1. files
2. model
3. trainers
4. benchmarks
5. tests
6. commands

After coding, give a walkthrough:
1. files changed
2. commands run
3. test result
4. CUDA device
5. benchmark table
6. whether CUDA matches MLX
7. limitations
8. next recommendation

## MLX Baseline
MLX validated:
- mono_update_every_2 works
- BitNet-style ternary shadow weights train
- 32x768 BitNet mono ran and learned
- 32x768 result:
  - params: 233,325,312
  - CE: 4.3722 -> 3.1299
  - peak memory: 4.514 GB
  - tokens/sec: 310.1
  - fake_ternary_math=true
  - real_1bit_kernel=false

## Training Modes
Implement:
- fp_chainrule
- fp_mono_update_every_2
- bitnet_chainrule
- bitnet_mono_update_every_2

## BitLinear
Use ternary active weights {-1,0,1} from trainable shadow weights.

Formula:
- gamma = mean(abs(W_shadow))
- W_ternary = round(clip(W_shadow / gamma, -1, 1))
- W_active = gamma * W_ternary
- use STE so shadow weights train

Always log:
- ternary distribution
- gamma mean
- fake_ternary_math=true
- real_1bit_kernel=false

## Model
Decoder-only causal LM:
- vocab 256
- learned position embeddings
- RMSNorm
- causal MHA
- MLP
- final LM head
- configurable layers/hidden/heads/seq/batch

## Dataset
Synthetic byte-level first:
- tiny_code
- pattern
- counting

Default: tiny_code.

## Configs
Create:
- bitnet_tiny: hidden 128, layers 4, heads 4, batch 8, seq 128, steps 300
- bitnet_large: hidden 768, layers 12, heads 12, batch 2, seq 128, steps 50
- bitnet_32x512: hidden 512, layers 32, heads 8, batch 1, seq 128, steps 30
- bitnet_32x768_smoke: hidden 768, layers 32, heads 12, batch 1, seq 128, steps 20

## Metrics
Log:
- initial/final CE
- CE delta
- tokens/sec
- ms/step
- peak CUDA memory GB
- params
- ternary distribution
- gamma mean
- fake_ternary_math
- real_1bit_kernel

Use CUDA timing correctly:
- torch.cuda.synchronize()
- torch.cuda.reset_peak_memory_stats()
- torch.cuda.max_memory_allocated()

## Files
Create:
- requirements.txt
- README.md
- configs/*.yaml
- src/samatnext_bit/data.py
- src/samatnext_bit/model.py
- src/samatnext_bit/bitlinear.py
- src/samatnext_bit/bitnet_model.py
- src/samatnext_bit/train.py
- src/samatnext_bit/bench.py
- src/samatnext_bit/utils.py
- tests/
- RESULTS_CUDA_PARITY_v1.md

## Commands
Run:
python -m pytest -q
python -m samatnext_bit.bench --config configs/bitnet_tiny.yaml
python -m samatnext_bit.bench --config configs/bitnet_large.yaml
python -m samatnext_bit.bench --config configs/bitnet_32x512.yaml

Only after those pass:
python -m samatnext_bit.bench --config configs/bitnet_32x768_smoke.yaml --modes fp_chainrule,fp_mono_update_every_2,bitnet_mono_update_every_2
