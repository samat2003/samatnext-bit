# Release Notes: mono-forward-smoke-v1

This release freezes the mono-forward smoke benchmark milestone for `samatnext-bit`.

This tag remains frozen. Later arXiv-draft documentation integrates supplemental audits and smoke runs, but does not modify `mono-forward-smoke-v1`.

## What Is Included

- CUDA/PyTorch benchmark code for small byte-level decoder smoke tests.
- Three public benchmark configs:
  - `configs/chainrule_vs_mono_24layer.yaml`
  - `configs/dense4_vs_sparse4_24.yaml`
  - `configs/softmax_vs_gdn_4active.yaml`
- Result summaries and limitations.
- ArXiv-style paper draft scaffold in `paper/main.md`.
- Reproducibility checklist.

## Reproduction Commands

```bash
python -m pytest -q
python -m samatnext_bit.bench_speed --config configs/chainrule_vs_mono_24layer.yaml
python -m samatnext_bit.bench_speed --config configs/dense4_vs_sparse4_24.yaml
python -m samatnext_bit.bench_speed --config configs/softmax_vs_gdn_4active.yaml
```

## Strongest Honest Claim

Mono-forward scheduled local updates substantially improved throughput in these short Tiny Shakespeare byte-level smoke benchmarks. Dense24 mono reached 4.58x the tokens/sec of dense24 chain-rule, while sparse4/24 mono reached 17.63x the tokens/sec of dense24 chain-rule.

## Main Limitations

- Chain-rule reached lower short-run validation CE in the main dense24 comparison.
- Sparse4/24 used only 4 active blocks and did not outperform dense4 at matched active params.
- `simple_gdn` is not official Gated DeltaNet; it was stable but slower and higher-memory than softmax.
- FLOPs are estimated from parameter counts, not measured hardware counters.
- Tiny Shakespeare byte-level validation is a smoke benchmark, not LLM-scale evidence.

## Future Work

- Longer equal-token and equal-update comparisons.
- More faithful delta-rule recurrent mixer.
- Optimized CUDA/Triton recurrent kernels.
- Larger datasets and model sizes after the small-scale methodology is stable.
