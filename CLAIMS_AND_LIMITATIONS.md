# Claims and Limitations

## Supported Claims

Mono-forward scheduled updates improve throughput significantly in these small byte-level smoke benchmarks.

Dense24 mono achieved 663,520 tok/s versus 145,006 tok/s for dense24 chain-rule, a 4.58x throughput increase in the 500-step smoke run.

Sparse4/24 mono achieved 2,556,872 tok/s versus 145,006 tok/s for dense24 chain-rule, a 17.63x throughput increase, but it used only 4 active blocks and 890,752 active params.

Chain-rule still learned better per update and reached lower validation CE in the short 500-step runs.

Sparse4/24 did not outperform a normal dense4 baseline at matched active parameter count. In the reviewer baseline, matched dense4 and sparse4/24 tracks reached the same validation CE.

`simple_gdn` was stable in the 4-active-block smoke benchmark and had finite gradients with no NaNs/Infs.

`simple_gdn` slightly improved mono validation CE in the 500-step smoke run, but it was slower and higher-memory than softmax in this implementation.

## Limitations

FLOPs are simple parameter-count estimates, not hardware counters.

Tiny Shakespeare byte-level validation is a smoke benchmark. It is not proof of coding ability, broad language modeling ability, or LLM-scale quality.

The models are small hidden-128 byte-level decoders, not production LLMs.

No result here proves that mono-forward updates beat Transformers at scale.

No result here proves LLM-scale behavior.

No result here proves that sparse4/24 is equivalent to dense24 training.

Sparse4/24 is a logical scaffold with 4 active/instantiated blocks. It must not be called dense.

`simple_gdn` is not official Gated DeltaNet. It is explicitly labeled `official_gdn=false`.

No real 1.58-bit speedup is claimed in this milestone.

Base3/1.58-bit paths exist historically in the repository, but they are not part of the FP mono-forward smoke benchmark claims.

Throughput can vary with GPU, driver, thermals, power limits, and background load.

The current benchmarks are short 500-step runs. Longer runs may change quality rankings.
