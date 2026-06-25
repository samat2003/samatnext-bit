from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .data import ByteBatcher
from .model import DecoderLM
from .utils import count_params, device as default_device, peak_memory_gb, reset_peak_memory, sync


@dataclass
class TrainResult:
    mode: str
    initial_ce: float
    final_ce: float
    ce_delta: float
    tokens_sec: float
    ms_step: float
    peak_cuda_memory_gb: float
    params: int
    ternary: str
    gamma_mean: float | str
    backend: str
    native_kernel: bool
    packed_ternary: bool
    base3_packed: bool
    two_bit_packed: bool
    int8_ternary: bool
    storage_bits_per_weight: float | str
    ideal_entropy_bits_per_weight: float | str
    forward_native: bool
    backward_native: bool
    ste_backward: bool
    uses_tensor_cores: bool
    uses_cuda_cores: bool
    persistent_packed_cache: bool
    decode_strategy: str
    fallback_used: bool
    packed_cache_refresh_count: int
    pack_time_ms: float
    forward_kernel_time_ms: float
    fake_vs_base3_correctness_error: float | str
    packed_1p58bit: bool
    completed: bool = True


def parse_mode(mode: str) -> tuple[bool, str, bool]:
    bitnet = mode.startswith("bitnet_")
    if "base3_tile_dot_ternary" in mode:
        backend = "base3_tile_dot_ternary"
    elif "base3_lut_ternary" in mode:
        backend = "base3_lut_ternary"
    elif "base3_packed_ternary" in mode:
        backend = "base3_packed_ternary"
    elif "two_bit_ternary_legacy" in mode or "packed_ternary" in mode:
        backend = "two_bit_ternary_legacy"
    elif "int8_ternary_legacy" in mode or "_native_" in mode:
        backend = "int8_ternary_legacy"
    elif "fake_ternary" in mode or "bitnet_fake" in mode:
        backend = "fake_ternary"
    else:
        backend = "fp"
    mono = mode.endswith("mono_update_every_2")
    return bitnet, backend, mono


def build_model(cfg: dict, mode: str, dev: torch.device) -> DecoderLM:
    bitnet, backend, _ = parse_mode(mode)
    model = DecoderLM(
        vocab_size=cfg.get("vocab_size", 256),
        seq_len=cfg["seq_len"],
        hidden=cfg["hidden"],
        layers=cfg["layers"],
        heads=cfg["heads"],
        bitnet=bitnet,
        backend=backend,
    )
    return model.to(dev)


def refresh_packed_caches(model: DecoderLM) -> None:
    for module in model.bitlinear_modules():
        if hasattr(module, "refresh_packed_cache"):
            module.refresh_packed_cache()


def mark_packed_caches_dirty(model: DecoderLM) -> None:
    for module in model.bitlinear_modules():
        if hasattr(module, "mark_packed_cache_dirty"):
            module.mark_packed_cache_dirty()


def fake_vs_base3_error(model: DecoderLM, dev: torch.device) -> float | str:
    if dev.type != "cuda":
        return "n/a"
    base3_mods = [m for m in model.bitlinear_modules() if getattr(m, "base3_packed", False)]
    if not base3_mods:
        return "n/a"
    from .bitlinear import ternarize_shadow
    from .triton_kernels import triton_base3_ternary_linear

    errs = []
    with torch.no_grad():
        for module in base3_mods[:4]:
            module.refresh_packed_cache()
            x = torch.randn(3, module.in_features, device=dev, dtype=module.weight.dtype)
            active, _, _ = ternarize_shadow(module.weight)
            y_fake = torch.nn.functional.linear(x, active, module.bias)
            y_base3 = module(x)
            errs.append(float((y_fake - y_base3).abs().max().item()))
    return max(errs) if errs else "n/a"


def train_mode(cfg: dict, mode: str, dev: torch.device | None = None) -> TrainResult:
    dev = dev or default_device()
    torch.manual_seed(int(cfg.get("seed", 0)))
    model = build_model(cfg, mode, dev)
    batcher = ByteBatcher(cfg.get("dataset", "tiny_code"), cfg["batch_size"], cfg["seq_len"], dev, seed=int(cfg.get("seed", 0)))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 3e-4)), weight_decay=float(cfg.get("weight_decay", 0.01)))
    _, _, mono = parse_mode(mode)
    x, y = batcher.next_batch()
    with torch.no_grad():
        _, loss = model(x, y)
        initial = float(loss.item())
    reset_peak_memory(dev)
    sync(dev)
    start = time.perf_counter()
    final = initial
    for step in range(int(cfg["steps"])):
        x, y = batcher.next_batch()
        _, loss = model(x, y)
        if (not mono) or (step % 2 == 0):
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            mark_packed_caches_dirty(model)
            refresh_packed_caches(model)
        final = float(loss.item())
    sync(dev)
    elapsed = time.perf_counter() - start
    steps = int(cfg["steps"])
    tokens = steps * cfg["batch_size"] * cfg["seq_len"]
    summary = model.ternary_summary()
    bitmods = model.bitlinear_modules()
    native_kernel = any(getattr(m, "native_kernel", False) for m in bitmods)
    packed_ternary = any(getattr(m, "packed_ternary", False) for m in bitmods)
    base3_packed = any(getattr(m, "base3_packed", False) for m in bitmods)
    two_bit_packed = any(getattr(m, "two_bit_packed", False) for m in bitmods)
    int8_ternary = any(getattr(m, "int8_ternary", False) for m in bitmods)
    packed_1p58 = any(getattr(m, "packed_1p58bit", False) for m in bitmods)
    forward_native = any(getattr(m, "forward_native", False) for m in bitmods)
    backward_native = any(getattr(m, "backward_native", False) for m in bitmods)
    ste_backward = any(getattr(m, "ste_backward", False) for m in bitmods)
    uses_tensor_cores = any(getattr(m, "uses_tensor_cores", False) for m in bitmods)
    uses_cuda_cores = any(getattr(m, "uses_cuda_cores", False) for m in bitmods)
    persistent_packed_cache = any(getattr(m, "persistent_packed_cache", False) for m in bitmods)
    decode_strategy = next((getattr(m, "decode_strategy", "n/a") for m in bitmods), "n/a")
    fallback_used = any(getattr(m, "fallback_used", False) for m in bitmods)
    refresh_count = sum(int(getattr(m, "packed_cache_refresh_count", 0)) for m in bitmods)
    pack_time_ms = sum(float(getattr(m, "pack_time_ms", 0.0)) for m in bitmods)
    forward_kernel_time_ms = sum(float(getattr(m, "forward_kernel_time_ms", 0.0)) for m in bitmods)
    storage_bits = next((getattr(m, "storage_bits_per_weight", "n/a") for m in bitmods), "n/a")
    ideal_bits = next((getattr(m, "ideal_entropy_bits_per_weight", "n/a") for m in bitmods), "n/a")
    correctness_error = fake_vs_base3_error(model, dev)
    return TrainResult(
        mode=mode,
        initial_ce=initial,
        final_ce=final,
        ce_delta=initial - final,
        tokens_sec=tokens / elapsed if elapsed > 0 else 0.0,
        ms_step=elapsed * 1000.0 / steps,
        peak_cuda_memory_gb=peak_memory_gb(dev),
        params=count_params(model),
        ternary=str(summary["ternary"]),
        gamma_mean=summary["gamma_mean"] if summary["ternary"] != "n/a" else "n/a",
        backend=parse_mode(mode)[1],
        native_kernel=bool(native_kernel and dev.type == "cuda"),
        packed_ternary=bool(packed_ternary),
        base3_packed=bool(base3_packed),
        two_bit_packed=bool(two_bit_packed),
        int8_ternary=bool(int8_ternary),
        storage_bits_per_weight=storage_bits,
        ideal_entropy_bits_per_weight=ideal_bits,
        forward_native=bool(forward_native and dev.type == "cuda"),
        backward_native=bool(backward_native and dev.type == "cuda"),
        ste_backward=bool(ste_backward),
        uses_tensor_cores=bool(uses_tensor_cores),
        uses_cuda_cores=bool(uses_cuda_cores and dev.type == "cuda"),
        persistent_packed_cache=bool(persistent_packed_cache),
        decode_strategy=str(decode_strategy),
        fallback_used=bool(fallback_used),
        packed_cache_refresh_count=refresh_count,
        pack_time_ms=pack_time_ms,
        forward_kernel_time_ms=forward_kernel_time_ms,
        fake_vs_base3_correctness_error=correctness_error,
        packed_1p58bit=bool(packed_1p58),
    )
