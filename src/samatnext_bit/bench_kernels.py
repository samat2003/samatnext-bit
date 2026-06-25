from __future__ import annotations

import json
import time

import torch
import torch.nn.functional as F

from .bitlinear import BitLinear, ternarize_shadow
from .utils import device, versions, write_json


SHAPES = [
    (1024, 128, 128),
    (1024, 512, 512),
    (1024, 768, 768),
    (256, 3072, 768),
    (1024, 3072, 768),
]

BACKENDS = [
    ("fake_ternary", "fake_ternary"),
    ("base3_packed_v1", "base3_packed_ternary"),
    ("base3_lut_ternary", "base3_lut_ternary"),
    ("base3_tile_dot_ternary", "base3_tile_dot_ternary"),
]


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_forward(layer: BitLinear, x: torch.Tensor, iters: int = 30) -> tuple[float, torch.Tensor]:
    for _ in range(5):
        y = layer(x)
    sync()
    start = time.perf_counter()
    for _ in range(iters):
        y = layer(x)
    sync()
    ms = (time.perf_counter() - start) * 1000.0 / iters
    return ms, y


def bench_shape(m: int, k: int, n: int, dev: torch.device) -> list[dict]:
    torch.manual_seed(1234)
    x = torch.randn(m, k, device=dev)
    ref = BitLinear(k, n, backend="fake_ternary").to(dev)
    with torch.no_grad():
        active, _, _ = ternarize_shadow(ref.weight)
        y_ref = F.linear(x, active, ref.bias)
    rows = []
    timings = {}
    for label, backend in BACKENDS:
        layer = BitLinear(k, n, backend=backend).to(dev)
        with torch.no_grad():
            layer.weight.copy_(ref.weight)
            layer.bias.copy_(ref.bias)
        if hasattr(layer, "refresh_packed_cache"):
            layer.refresh_packed_cache()
        try:
            ms, y = time_forward(layer, x)
            err = float((y - y_ref).abs().max().item())
            error = None
        except Exception as exc:
            ms = float("nan")
            err = float("nan")
            error = repr(exc)
        timings[label] = ms
        rows.append(
            {
                "shape": f"M={m},K={k},N={n}",
                "backend": label,
                "ms": ms,
                "speedup_vs_fake_ternary": timings.get("fake_ternary", ms) / ms if ms == ms and ms > 0 else float("nan"),
                "speedup_vs_base3_lut": timings.get("base3_lut_ternary", ms) / ms if ms == ms and ms > 0 else float("nan"),
                "max_error_vs_fake": err,
                "storage_bits_per_weight": layer.storage_bits_per_weight,
                "base3_packed": layer.base3_packed,
                "packed_ternary": layer.packed_ternary,
                "decode_strategy": layer.decode_strategy,
                "native_kernel": bool(layer.native_kernel and dev.type == "cuda"),
                "uses_tensor_cores": layer.uses_tensor_cores,
                "uses_cuda_cores": bool(layer.uses_cuda_cores and dev.type == "cuda"),
                "persistent_packed_cache": layer.persistent_packed_cache,
                "fallback_used": getattr(layer, "fallback_used", False),
                "pack_refresh_count": layer.packed_cache_refresh_count,
                "pack_time_ms": layer.pack_time_ms,
                "forward_kernel_time_ms": layer.forward_kernel_time_ms,
                "error": error,
            }
        )
    return rows


def main() -> None:
    dev = device()
    if dev.type != "cuda":
        raise SystemExit("bench_kernels requires CUDA")
    print(f"device={dev} gpu={versions()['gpu']}", flush=True)
    rows = []
    for shape in SHAPES:
        rows.extend(bench_shape(*shape, dev))
    for row in rows:
        print(row, flush=True)
    write_json("runs/kernel_bench_latest.json", {"versions": versions(), "results": rows})


if __name__ == "__main__":
    main()
