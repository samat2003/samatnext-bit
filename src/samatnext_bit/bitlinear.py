from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .triton_kernels import triton_base3_lut_ternary_linear, triton_base3_pack, triton_base3_ternary_linear, triton_base3_tile_dot_ternary_linear, triton_packed_ternary_linear, triton_ternary_linear


def ternarize_shadow(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gamma = weight.detach().abs().mean(dim=1).clamp_min(1e-6)
    ternary_hard = torch.round(torch.clamp(weight / gamma[:, None], -1, 1))
    active = weight + (gamma[:, None] * ternary_hard - weight).detach()
    return active, ternary_hard.detach(), gamma.detach()


def pack_ternary_2bit(ternary: torch.Tensor) -> torch.Tensor:
    if ternary.dim() != 2:
        raise ValueError("ternary weights must be 2D")
    t = ternary.detach().to(torch.int32).cpu()
    out_features, in_features = t.shape
    words = (in_features + 15) // 16
    packed = torch.zeros((out_features, words), dtype=torch.int32)
    codes = torch.where(t > 0, 1, torch.where(t < 0, 2, 0)).to(torch.int32)
    for i in range(16):
        cols = torch.arange(i, words * 16, 16)
        valid = cols < in_features
        if valid.any():
            packed[:, valid] |= codes[:, cols[valid]] << (2 * i)
    return packed


def unpack_ternary_2bit(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    p = packed.detach().to(torch.int32).cpu()
    out_features, words = p.shape
    vals = torch.zeros((out_features, words * 16), dtype=torch.float32)
    for i in range(16):
        code = (p >> (2 * i)) & 3
        vals[:, i::16] = torch.where(code == 1, 1.0, torch.where(code == 2, -1.0, 0.0))
    return vals[:, :in_features]


POW3_20 = [3**i for i in range(20)]


def pack_ternary_base3(ternary: torch.Tensor) -> torch.Tensor:
    if ternary.dim() != 2:
        raise ValueError("ternary weights must be 2D")
    t = ternary.detach().to(torch.int64).cpu()
    out_features, in_features = t.shape
    words = (in_features + 19) // 20
    packed = torch.zeros((out_features, words), dtype=torch.int64)
    codes = torch.where(t > 0, 2, torch.where(t < 0, 0, 1)).to(torch.int64)
    for i, power in enumerate(POW3_20):
        cols = torch.arange(i, words * 20, 20)
        valid = cols < in_features
        if valid.any():
            packed[:, valid] += codes[:, cols[valid]] * power
    return packed.to(torch.int32)


def unpack_ternary_base3(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    p = packed.detach().cpu().to(torch.int64) & 0xFFFFFFFF
    out_features, words = p.shape
    vals = torch.zeros((out_features, words * 20), dtype=torch.float32)
    for i, power in enumerate(POW3_20):
        code = (p // power) % 3
        vals[:, i::20] = torch.where(code == 2, 1.0, torch.where(code == 0, -1.0, 0.0))
    return vals[:, :in_features]


def base3_lut5_table() -> torch.Tensor:
    table = torch.empty((243, 5), dtype=torch.float32)
    for chunk in range(243):
        x = chunk
        for lane in range(5):
            code = x % 3
            table[chunk, lane] = 1.0 if code == 2 else (-1.0 if code == 0 else 0.0)
            x //= 3
    return table.reshape(-1)


class _NativeTernaryLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shadow_weight, bias):
        gamma = shadow_weight.detach().abs().mean(dim=1).clamp_min(1e-6)
        w_int8 = torch.round(torch.clamp(shadow_weight.detach() / gamma[:, None], -1, 1)).to(torch.int8)
        y = triton_ternary_linear(x, w_int8, bias, gamma)
        ctx.save_for_backward(x, w_int8, gamma)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, w_int8, gamma = ctx.saved_tensors
        gy = grad_y.reshape(-1, grad_y.shape[-1])
        x2 = x.reshape(-1, x.shape[-1])
        w_active = w_int8.to(dtype=grad_y.dtype) * gamma.to(dtype=grad_y.dtype)[:, None]
        grad_x = gy.matmul(w_active).reshape_as(x)
        grad_w = gy.t().matmul(x2).to(dtype=grad_y.dtype)
        grad_b = gy.sum(0)
        return grad_x, grad_w, grad_b


class _PackedTernaryLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shadow_weight, bias):
        gamma = shadow_weight.detach().abs().mean(dim=1).clamp_min(1e-6)
        ternary = torch.round(torch.clamp(shadow_weight.detach() / gamma[:, None], -1, 1))
        packed = pack_ternary_2bit(ternary).to(device=x.device)
        y = triton_packed_ternary_linear(x, packed, bias, gamma, shadow_weight.shape[1])
        ctx.save_for_backward(x, ternary.to(device=x.device, dtype=x.dtype), gamma)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, ternary, gamma = ctx.saved_tensors
        gy = grad_y.reshape(-1, grad_y.shape[-1])
        x2 = x.reshape(-1, x.shape[-1])
        w_active = ternary.to(dtype=grad_y.dtype) * gamma.to(dtype=grad_y.dtype)[:, None]
        grad_x = gy.matmul(w_active).reshape_as(x)
        grad_w = gy.t().matmul(x2).to(dtype=grad_y.dtype)
        grad_b = gy.sum(0)
        return grad_x, grad_w, grad_b


class _Base3PackedTernaryLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shadow_weight, bias, packed, gamma, lut, strategy: str):
        if strategy == "lut":
            y = triton_base3_lut_ternary_linear(x, packed, bias, gamma, lut, shadow_weight.shape[1])
        elif strategy == "tile_dequant_dot":
            y = triton_base3_tile_dot_ternary_linear(x, packed, bias, gamma, shadow_weight.shape[1])
        else:
            y = triton_base3_ternary_linear(x, packed, bias, gamma, shadow_weight.shape[1])
        ctx.save_for_backward(x, shadow_weight, gamma)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, shadow_weight, gamma = ctx.saved_tensors
        gy = grad_y.reshape(-1, grad_y.shape[-1])
        x2 = x.reshape(-1, x.shape[-1])
        ternary = torch.round(torch.clamp(shadow_weight.detach() / gamma[:, None], -1, 1))
        w_active = ternary.to(dtype=grad_y.dtype) * gamma.to(dtype=grad_y.dtype)[:, None]
        grad_x = gy.matmul(w_active).reshape_as(x)
        grad_w = gy.t().matmul(x2).to(dtype=grad_y.dtype)
        grad_b = gy.sum(0)
        return grad_x, grad_w, grad_b, None, None, None, None


class BitLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, backend: str = "fake_ternary"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.backend = backend
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.register_buffer("packed_cache", torch.empty(out_features, self.base3_words, dtype=torch.int32), persistent=False)
        self.register_buffer("gamma_cache", torch.empty(out_features, dtype=torch.float32), persistent=False)
        self.register_buffer("base3_lut5", base3_lut5_table(), persistent=False)
        self.cache_dirty = True
        self.packed_cache_refresh_count = 0
        self.pack_time_ms = 0.0
        self.forward_kernel_time_ms = 0.0
        self.fake_vs_base3_error = 0.0
        self.fallback_used = False
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend in {"native", "int8_ternary_legacy"}:
            if self.bias is None:
                raise RuntimeError("native BitLinear currently requires bias")
            if x.is_cuda:
                return _NativeTernaryLinearFn.apply(x, self.weight, self.bias)
        if self.backend in {"packed_ternary", "two_bit_ternary_legacy"}:
            if self.bias is None:
                raise RuntimeError("packed BitLinear currently requires bias")
            if x.is_cuda:
                return _PackedTernaryLinearFn.apply(x, self.weight, self.bias)
        if self.backend in {"base3_packed_ternary", "base3_lut_ternary", "base3_tile_dot_ternary"}:
            if self.bias is None:
                raise RuntimeError("base3 BitLinear currently requires bias")
            if x.is_cuda:
                if self.cache_dirty:
                    self.refresh_packed_cache()
                start = time.perf_counter()
                y = _Base3PackedTernaryLinearFn.apply(
                    x,
                    self.weight,
                    self.bias,
                    self.packed_cache,
                    self.gamma_cache,
                    self.base3_lut5,
                    self.decode_strategy,
                )
                torch.cuda.synchronize()
                self.forward_kernel_time_ms += (time.perf_counter() - start) * 1000.0
                return y
        active, _, _ = ternarize_shadow(self.weight)
        return F.linear(x, active, self.bias)

    @property
    def base3_words(self) -> int:
        return (self.in_features + 19) // 20

    def refresh_packed_cache(self) -> None:
        if self.backend not in {"base3_packed_ternary", "base3_lut_ternary", "base3_tile_dot_ternary"}:
            return
        start = time.perf_counter()
        with torch.no_grad():
            gamma = self.weight.detach().abs().mean(dim=1).clamp_min(1e-6)
            if self.weight.is_cuda:
                self.gamma_cache.copy_(gamma.to(dtype=torch.float32))
                triton_base3_pack(self.weight.detach(), self.gamma_cache, self.packed_cache)
                torch.cuda.synchronize()
            else:
                _, ternary, gamma_cpu = ternarize_shadow(self.weight)
                self.gamma_cache.copy_(gamma_cpu.to(dtype=torch.float32))
                self.packed_cache.copy_(pack_ternary_base3(ternary))
        self.cache_dirty = False
        self.packed_cache_refresh_count += 1
        self.pack_time_ms += (time.perf_counter() - start) * 1000.0

    def mark_packed_cache_dirty(self) -> None:
        if self.backend in {"base3_packed_ternary", "base3_lut_ternary", "base3_tile_dot_ternary"}:
            self.cache_dirty = True

    def packed_weight(self) -> torch.Tensor:
        _, ternary, _ = ternarize_shadow(self.weight)
        if self.backend in {"base3_packed_ternary", "base3_lut_ternary", "base3_tile_dot_ternary"}:
            return pack_ternary_base3(ternary)
        return pack_ternary_2bit(ternary)

    def ternary_summary(self) -> dict[str, float]:
        _, t, gamma = ternarize_shadow(self.weight)
        total = t.numel()
        return {
            "neg_pct": float((t < 0).sum().item() * 100.0 / total),
            "zero_pct": float((t == 0).sum().item() * 100.0 / total),
            "pos_pct": float((t > 0).sum().item() * 100.0 / total),
            "gamma_mean": float(gamma.mean().item()),
        }

    @property
    def native_kernel(self) -> bool:
        return self.backend in {"native", "int8_ternary_legacy", "packed_ternary", "two_bit_ternary_legacy", "base3_packed_ternary", "base3_lut_ternary", "base3_tile_dot_ternary"}

    @property
    def packed_ternary(self) -> bool:
        return self.backend in {"packed_ternary", "two_bit_ternary_legacy", "base3_packed_ternary", "base3_lut_ternary", "base3_tile_dot_ternary"}

    @property
    def base3_packed(self) -> bool:
        return self.backend in {"base3_packed_ternary", "base3_lut_ternary", "base3_tile_dot_ternary"}

    @property
    def decode_strategy(self) -> str:
        if self.backend == "base3_lut_ternary":
            return "lut"
        if self.backend == "base3_tile_dot_ternary":
            return "tile_dequant_dot"
        if self.backend == "base3_packed_ternary":
            return "divmod"
        if self.two_bit_packed:
            return "bitshift"
        if self.int8_ternary:
            return "int8"
        return "n/a"

    @property
    def persistent_packed_cache(self) -> bool:
        return self.backend in {"base3_packed_ternary", "base3_lut_ternary", "base3_tile_dot_ternary"}

    @property
    def two_bit_packed(self) -> bool:
        return self.backend in {"packed_ternary", "two_bit_ternary_legacy"}

    @property
    def int8_ternary(self) -> bool:
        return self.backend in {"native", "int8_ternary_legacy"}

    @property
    def packed_1p58bit(self) -> bool:
        return False

    @property
    def storage_bits_per_weight(self) -> float:
        if self.base3_packed:
            return 1.6
        if self.two_bit_packed:
            return 2.0
        if self.int8_ternary:
            return 8.0
        return 32.0

    @property
    def ideal_entropy_bits_per_weight(self) -> float:
        return 1.585 if self.base3_packed else (1.58 if self.two_bit_packed else 0.0)

    @property
    def forward_native(self) -> bool:
        return self.native_kernel

    @property
    def backward_native(self) -> bool:
        return False

    @property
    def ste_backward(self) -> bool:
        return True

    @property
    def uses_tensor_cores(self) -> bool:
        return self.backend == "base3_tile_dot_ternary"

    @property
    def uses_cuda_cores(self) -> bool:
        return self.native_kernel
