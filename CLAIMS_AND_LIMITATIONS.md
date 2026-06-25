# Claims and Limitations

## Supported Claims

Mono-forward scheduled updates improve throughput significantly in these small smoke benchmarks.

Dense24 mono achieved 663,520 tok/s versus 145,006 tok/s for dense24 chain-rule, a 4.58x throughput increase in the 500-step Tiny Shakespeare byte-level smoke run.

Sparse4/24 mono achieved 2,556,872 tok/s versus 145,006 tok/s for dense24 chain-rule, a 17.63x throughput increase, but it used only 4 active blocks and 890,752 active params.

Chain-rule still learned better per update and reached lower validation CE in the short Tiny Shakespeare runs.

Sparse4/24 did not outperform a normal dense4 baseline at matched active parameter count. In the reviewer baseline, matched dense4 and sparse4/24 tracks reached the same validation CE.

Regular PyTorch CUDA training was also fast for the tiny 4-layer hidden-512 Python-token baseline. The throughput result is therefore not evidence that the baseline was artificially slow.

`simple_gdn` was stable in both the Tiny Shakespeare and BF16 Python-token smoke benchmarks, with finite gradients and no NaNs/Infs.

`simple_gdn` was slower and higher-memory than softmax in the current implementation. In the BF16 Python-token run it also did not improve validation CE.

The dense24 memory optimization reduced VRAM by avoiding autograd graph construction on non-update mono-forward steps. In the Python-token dense24 run, peak memory dropped from 7.459 GB to 4.641 GB at the same hidden/seq/batch and the same final CE.

The strongest practical supplemental result is the optimized dense24 MBPP smoke run:

- 81.1M parameters
- dense 24/24 softmax
- 500 FP16 steps
- 83.6K tok/s
- 3.992GB peak CUDA memory
- validation CE 8.7202 to 3.9844

The 500-step MBPP run demonstrates stability and efficiency on a small Python problem/solution text corpus.

## Limitations

FLOPs are simple parameter-count estimates, not hardware counters.

Tiny Shakespeare byte-level validation is a smoke benchmark. It is not proof of coding ability, broad language modeling ability, or LLM-scale quality.

The original release models are small hidden-128 byte-level decoders, not production LLMs.

The supplemental Python-token and MBPP models are still small decoder smoke experiments, not LLM-scale systems.

MBPP sanitized is tiny in this setup: 384 train examples and 62,595 train tokens.

The 500-step MBPP run means repeated exposure to a very small corpus.

Generated MBPP samples were noisy and invalid-looking.

No MBPP tests were executed.

No MBPP pass@1 is reported.

No coding ability claim is made.

No long-run convergence claim is made.

There is no trained standard Transformer 500-step MBPP comparison in this repository unless one is added later.

Mono-forward still generally has worse CE than chain-rule at equal step count in earlier comparisons.

No result here proves that mono-forward updates beat Transformers at scale.

No result here proves LLM-scale behavior.

No result here proves that sparse4/24 is equivalent to dense24 training.

Sparse4/24 is a logical scaffold with 4 active/instantiated blocks. It must not be called dense.

`simple_gdn` is not official Gated DeltaNet. It is explicitly labeled `official_gdn=false`.

No real 1.58-bit speedup is claimed in this milestone.

Base3/1.58-bit paths exist historically in the repository, but they are not part of the FP mono-forward smoke benchmark claims.

Throughput can vary with GPU, driver, thermals, power limits, and background load.

The current benchmarks are short smoke runs. Longer runs may change quality rankings.

## License and Data Scope

Code in this repository is Apache-2.0.

Dataset licenses remain governed by their original sources. MBPP is used only as a small smoke-training corpus and should be attributed to Google Research MBPP.
