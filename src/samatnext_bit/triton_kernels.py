from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _base3_pack_kernel(W, G, P, N:tl.constexpr, K:tl.constexpr, WORDS:tl.constexpr, BLOCK_T:tl.constexpr):
        pid_n = tl.program_id(0)
        pid_w = tl.program_id(1)
        offs = tl.arange(0, BLOCK_T)
        k = pid_w * 20 + offs
        gamma = tl.load(G + pid_n)
        w = tl.load(W + pid_n * K + k, mask=k < K, other=0.0)
        r = w / gamma
        code = tl.where(r > 0.5, 2, tl.where(r < -0.5, 0, 1)).to(tl.uint32)
        pow3 = tl.full((BLOCK_T,), 1, tl.uint32)
        pow3 = tl.where(offs == 0, 1, pow3)
        pow3 = tl.where(offs == 1, 3, pow3)
        pow3 = tl.where(offs == 2, 9, pow3)
        pow3 = tl.where(offs == 3, 27, pow3)
        pow3 = tl.where(offs == 4, 81, pow3)
        pow3 = tl.where(offs == 5, 243, pow3)
        pow3 = tl.where(offs == 6, 729, pow3)
        pow3 = tl.where(offs == 7, 2187, pow3)
        pow3 = tl.where(offs == 8, 6561, pow3)
        pow3 = tl.where(offs == 9, 19683, pow3)
        pow3 = tl.where(offs == 10, 59049, pow3)
        pow3 = tl.where(offs == 11, 177147, pow3)
        pow3 = tl.where(offs == 12, 531441, pow3)
        pow3 = tl.where(offs == 13, 1594323, pow3)
        pow3 = tl.where(offs == 14, 4782969, pow3)
        pow3 = tl.where(offs == 15, 14348907, pow3)
        pow3 = tl.where(offs == 16, 43046721, pow3)
        pow3 = tl.where(offs == 17, 129140163, pow3)
        pow3 = tl.where(offs == 18, 387420489, pow3)
        pow3 = tl.where(offs == 19, 1162261467, pow3)
        word = tl.sum(tl.where((offs < 20) & (k < K), code * pow3, 0), axis=0)
        tl.store(P + pid_n * WORDS + pid_w, word)


    @triton.jit
    def _ternary_matmul_kernel(X, W, B, G, Y, M:tl.constexpr, K:tl.constexpr, N:tl.constexpr, BLOCK_M:tl.constexpr, BLOCK_N:tl.constexpr, BLOCK_K:tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k = k0 + offs_k
            x = tl.load(X + offs_m[:, None] * K + k[None, :], mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0)
            w = tl.load(W + offs_n[None, :] * K + k[:, None], mask=(offs_n[None, :] < N) & (k[:, None] < K), other=0).to(tl.float32)
            acc += tl.dot(x, w)
        gamma = tl.load(G + offs_n, mask=offs_n < N, other=0.0)
        b = tl.load(B + offs_n, mask=offs_n < N, other=0.0)
        tl.store(Y + offs_m[:, None] * N + offs_n[None, :], acc * gamma[None, :] + b[None, :], mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


    @triton.jit
    def _packed_ternary_matmul_kernel(X, W, B, G, Y, M:tl.constexpr, K:tl.constexpr, N:tl.constexpr, WORDS:tl.constexpr, BLOCK_M:tl.constexpr, BLOCK_N:tl.constexpr, BLOCK_K:tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k = k0 + offs_k
            x = tl.load(X + offs_m[:, None] * K + k[None, :], mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0)
            word_idx = k // 16
            shift = (k % 16) * 2
            word = tl.load(W + offs_n[None, :] * WORDS + word_idx[:, None], mask=(offs_n[None, :] < N) & (k[:, None] < K), other=0).to(tl.uint32)
            code = (word >> shift[:, None]) & 3
            w = tl.where(code == 1, 1.0, tl.where(code == 2, -1.0, 0.0))
            acc += tl.dot(x, w)
        gamma = tl.load(G + offs_n, mask=offs_n < N, other=0.0)
        b = tl.load(B + offs_n, mask=offs_n < N, other=0.0)
        tl.store(Y + offs_m[:, None] * N + offs_n[None, :], acc * gamma[None, :] + b[None, :], mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


    @triton.jit
    def _base3_ternary_matmul_kernel(X, W, B, G, Y, M:tl.constexpr, K:tl.constexpr, N:tl.constexpr, WORDS:tl.constexpr, BLOCK_M:tl.constexpr, BLOCK_N:tl.constexpr, BLOCK_K:tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k = k0 + offs_k
            x = tl.load(X + offs_m[:, None] * K + k[None, :], mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0)
            word_idx = k // 20
            trit_idx = k % 20
            word = tl.load(W + offs_n[None, :] * WORDS + word_idx[:, None], mask=(offs_n[None, :] < N) & (k[:, None] < K), other=0).to(tl.uint32)
            div = tl.full((BLOCK_K,), 1, tl.uint32)
            div = tl.where(trit_idx == 0, 1, div)
            div = tl.where(trit_idx == 1, 3, div)
            div = tl.where(trit_idx == 2, 9, div)
            div = tl.where(trit_idx == 3, 27, div)
            div = tl.where(trit_idx == 4, 81, div)
            div = tl.where(trit_idx == 5, 243, div)
            div = tl.where(trit_idx == 6, 729, div)
            div = tl.where(trit_idx == 7, 2187, div)
            div = tl.where(trit_idx == 8, 6561, div)
            div = tl.where(trit_idx == 9, 19683, div)
            div = tl.where(trit_idx == 10, 59049, div)
            div = tl.where(trit_idx == 11, 177147, div)
            div = tl.where(trit_idx == 12, 531441, div)
            div = tl.where(trit_idx == 13, 1594323, div)
            div = tl.where(trit_idx == 14, 4782969, div)
            div = tl.where(trit_idx == 15, 14348907, div)
            div = tl.where(trit_idx == 16, 43046721, div)
            div = tl.where(trit_idx == 17, 129140163, div)
            div = tl.where(trit_idx == 18, 387420489, div)
            div = tl.where(trit_idx == 19, 1162261467, div)
            code = (word // div[:, None]) % 3
            w = tl.where(code == 2, 1.0, tl.where(code == 0, -1.0, 0.0))
            acc += tl.dot(x, w)
        gamma = tl.load(G + offs_n, mask=offs_n < N, other=0.0)
        b = tl.load(B + offs_n, mask=offs_n < N, other=0.0)
        tl.store(Y + offs_m[:, None] * N + offs_n[None, :], acc * gamma[None, :] + b[None, :], mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


    @triton.jit
    def _base3_lut_ternary_matmul_kernel(X, W, B, G, LUT, Y, M:tl.constexpr, K:tl.constexpr, N:tl.constexpr, WORDS:tl.constexpr, BLOCK_M:tl.constexpr, BLOCK_N:tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for word_idx in range(0, WORDS):
            word = tl.load(W + offs_n * WORDS + word_idx, mask=offs_n < N, other=0).to(tl.uint32)
            c0 = word % 243
            c1 = (word // 243) % 243
            c2 = (word // 59049) % 243
            c3 = (word // 14348907) % 243
            base_k = word_idx * 20
            x0 = tl.load(X + offs_m * K + (base_k + 0), mask=(offs_m < M) & ((base_k + 0) < K), other=0.0)
            x1 = tl.load(X + offs_m * K + (base_k + 1), mask=(offs_m < M) & ((base_k + 1) < K), other=0.0)
            x2 = tl.load(X + offs_m * K + (base_k + 2), mask=(offs_m < M) & ((base_k + 2) < K), other=0.0)
            x3 = tl.load(X + offs_m * K + (base_k + 3), mask=(offs_m < M) & ((base_k + 3) < K), other=0.0)
            x4 = tl.load(X + offs_m * K + (base_k + 4), mask=(offs_m < M) & ((base_k + 4) < K), other=0.0)
            acc += x0[:, None] * tl.load(LUT + c0 * 5 + 0)[None, :]
            acc += x1[:, None] * tl.load(LUT + c0 * 5 + 1)[None, :]
            acc += x2[:, None] * tl.load(LUT + c0 * 5 + 2)[None, :]
            acc += x3[:, None] * tl.load(LUT + c0 * 5 + 3)[None, :]
            acc += x4[:, None] * tl.load(LUT + c0 * 5 + 4)[None, :]
            x5 = tl.load(X + offs_m * K + (base_k + 5), mask=(offs_m < M) & ((base_k + 5) < K), other=0.0)
            x6 = tl.load(X + offs_m * K + (base_k + 6), mask=(offs_m < M) & ((base_k + 6) < K), other=0.0)
            x7 = tl.load(X + offs_m * K + (base_k + 7), mask=(offs_m < M) & ((base_k + 7) < K), other=0.0)
            x8 = tl.load(X + offs_m * K + (base_k + 8), mask=(offs_m < M) & ((base_k + 8) < K), other=0.0)
            x9 = tl.load(X + offs_m * K + (base_k + 9), mask=(offs_m < M) & ((base_k + 9) < K), other=0.0)
            acc += x5[:, None] * tl.load(LUT + c1 * 5 + 0)[None, :]
            acc += x6[:, None] * tl.load(LUT + c1 * 5 + 1)[None, :]
            acc += x7[:, None] * tl.load(LUT + c1 * 5 + 2)[None, :]
            acc += x8[:, None] * tl.load(LUT + c1 * 5 + 3)[None, :]
            acc += x9[:, None] * tl.load(LUT + c1 * 5 + 4)[None, :]
            x10 = tl.load(X + offs_m * K + (base_k + 10), mask=(offs_m < M) & ((base_k + 10) < K), other=0.0)
            x11 = tl.load(X + offs_m * K + (base_k + 11), mask=(offs_m < M) & ((base_k + 11) < K), other=0.0)
            x12 = tl.load(X + offs_m * K + (base_k + 12), mask=(offs_m < M) & ((base_k + 12) < K), other=0.0)
            x13 = tl.load(X + offs_m * K + (base_k + 13), mask=(offs_m < M) & ((base_k + 13) < K), other=0.0)
            x14 = tl.load(X + offs_m * K + (base_k + 14), mask=(offs_m < M) & ((base_k + 14) < K), other=0.0)
            acc += x10[:, None] * tl.load(LUT + c2 * 5 + 0)[None, :]
            acc += x11[:, None] * tl.load(LUT + c2 * 5 + 1)[None, :]
            acc += x12[:, None] * tl.load(LUT + c2 * 5 + 2)[None, :]
            acc += x13[:, None] * tl.load(LUT + c2 * 5 + 3)[None, :]
            acc += x14[:, None] * tl.load(LUT + c2 * 5 + 4)[None, :]
            x15 = tl.load(X + offs_m * K + (base_k + 15), mask=(offs_m < M) & ((base_k + 15) < K), other=0.0)
            x16 = tl.load(X + offs_m * K + (base_k + 16), mask=(offs_m < M) & ((base_k + 16) < K), other=0.0)
            x17 = tl.load(X + offs_m * K + (base_k + 17), mask=(offs_m < M) & ((base_k + 17) < K), other=0.0)
            x18 = tl.load(X + offs_m * K + (base_k + 18), mask=(offs_m < M) & ((base_k + 18) < K), other=0.0)
            x19 = tl.load(X + offs_m * K + (base_k + 19), mask=(offs_m < M) & ((base_k + 19) < K), other=0.0)
            acc += x15[:, None] * tl.load(LUT + c3 * 5 + 0)[None, :]
            acc += x16[:, None] * tl.load(LUT + c3 * 5 + 1)[None, :]
            acc += x17[:, None] * tl.load(LUT + c3 * 5 + 2)[None, :]
            acc += x18[:, None] * tl.load(LUT + c3 * 5 + 3)[None, :]
            acc += x19[:, None] * tl.load(LUT + c3 * 5 + 4)[None, :]
        gamma = tl.load(G + offs_n, mask=offs_n < N, other=0.0)
        b = tl.load(B + offs_n, mask=offs_n < N, other=0.0)
        tl.store(Y + offs_m[:, None] * N + offs_n[None, :], acc * gamma[None, :] + b[None, :], mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


    @triton.jit
    def _base3_tile_dot_ternary_matmul_kernel(X, W, B, G, Y, M:tl.constexpr, K:tl.constexpr, N:tl.constexpr, WORDS:tl.constexpr, BLOCK_M:tl.constexpr, BLOCK_N:tl.constexpr, BLOCK_K:tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k = k0 + offs_k
            x = tl.load(X + offs_m[:, None] * K + k[None, :], mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0).to(tl.float32)
            word_idx = k // 20
            trit_idx = k % 20
            word = tl.load(W + offs_n[None, :] * WORDS + word_idx[:, None], mask=(offs_n[None, :] < N) & (k[:, None] < K), other=0).to(tl.uint32)
            div = tl.full((BLOCK_K,), 1, tl.uint32)
            div = tl.where(trit_idx == 0, 1, div)
            div = tl.where(trit_idx == 1, 3, div)
            div = tl.where(trit_idx == 2, 9, div)
            div = tl.where(trit_idx == 3, 27, div)
            div = tl.where(trit_idx == 4, 81, div)
            div = tl.where(trit_idx == 5, 243, div)
            div = tl.where(trit_idx == 6, 729, div)
            div = tl.where(trit_idx == 7, 2187, div)
            div = tl.where(trit_idx == 8, 6561, div)
            div = tl.where(trit_idx == 9, 19683, div)
            div = tl.where(trit_idx == 10, 59049, div)
            div = tl.where(trit_idx == 11, 177147, div)
            div = tl.where(trit_idx == 12, 531441, div)
            div = tl.where(trit_idx == 13, 1594323, div)
            div = tl.where(trit_idx == 14, 4782969, div)
            div = tl.where(trit_idx == 15, 14348907, div)
            div = tl.where(trit_idx == 16, 43046721, div)
            div = tl.where(trit_idx == 17, 129140163, div)
            div = tl.where(trit_idx == 18, 387420489, div)
            div = tl.where(trit_idx == 19, 1162261467, div)
            code = (word // div[:, None]) % 3
            w = tl.where(code == 2, 1.0, tl.where(code == 0, -1.0, 0.0)).to(tl.float32)
            acc += tl.dot(x, w, input_precision="tf32")
        gamma = tl.load(G + offs_n, mask=offs_n < N, other=0.0)
        b = tl.load(B + offs_n, mask=offs_n < N, other=0.0)
        tl.store(Y + offs_m[:, None] * N + offs_n[None, :], acc * gamma[None, :] + b[None, :], mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_ternary_linear(x: torch.Tensor, w_int8: torch.Tensor, bias: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("triton is not available")
    if not x.is_cuda:
        raise RuntimeError("triton native backend requires CUDA tensors")
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    w = w_int8.contiguous()
    m, k = x2.shape
    n = w.shape[0]
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(m, 16), triton.cdiv(n, 32))
    gamma_tensor = gamma.detach().to(device=x.device, dtype=torch.float32).contiguous()
    _ternary_matmul_kernel[grid](x2, w, bias, gamma_tensor, y, m, k, n, BLOCK_M=16, BLOCK_N=32, BLOCK_K=32)
    return y.reshape(*x.shape[:-1], n)


def triton_packed_ternary_linear(x: torch.Tensor, w_packed: torch.Tensor, bias: torch.Tensor, gamma: torch.Tensor, in_features: int) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("triton is not available")
    if not x.is_cuda:
        raise RuntimeError("triton packed backend requires CUDA tensors")
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    w = w_packed.contiguous()
    m, k = x2.shape
    if k != in_features:
        raise RuntimeError(f"input K={k} does not match packed K={in_features}")
    n, words = w.shape
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(m, 16), triton.cdiv(n, 32))
    gamma_tensor = gamma.detach().to(device=x.device, dtype=torch.float32).contiguous()
    _packed_ternary_matmul_kernel[grid](x2, w, bias, gamma_tensor, y, m, k, n, words, BLOCK_M=16, BLOCK_N=32, BLOCK_K=32)
    return y.reshape(*x.shape[:-1], n)


def triton_base3_pack(weight: torch.Tensor, gamma: torch.Tensor, packed: torch.Tensor) -> None:
    if triton is None:
        raise RuntimeError("triton is not available")
    if not weight.is_cuda:
        raise RuntimeError("base3 pack requires CUDA tensors")
    n, k = weight.shape
    words = packed.shape[1]
    grid = (n, words)
    _base3_pack_kernel[grid](weight, gamma, packed, n, k, words, BLOCK_T=32)


def triton_base3_ternary_linear(x: torch.Tensor, w_packed: torch.Tensor, bias: torch.Tensor, gamma: torch.Tensor, in_features: int) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("triton is not available")
    if not x.is_cuda:
        raise RuntimeError("triton base3 backend requires CUDA tensors")
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    w = w_packed.contiguous()
    m, k = x2.shape
    if k != in_features:
        raise RuntimeError(f"input K={k} does not match packed K={in_features}")
    n, words = w.shape
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(m, 16), triton.cdiv(n, 32))
    gamma_tensor = gamma.detach().to(device=x.device, dtype=torch.float32).contiguous()
    _base3_ternary_matmul_kernel[grid](x2, w, bias, gamma_tensor, y, m, k, n, words, BLOCK_M=16, BLOCK_N=32, BLOCK_K=64)
    return y.reshape(*x.shape[:-1], n)


def triton_base3_lut_ternary_linear(x: torch.Tensor, w_packed: torch.Tensor, bias: torch.Tensor, gamma: torch.Tensor, lut: torch.Tensor, in_features: int) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("triton is not available")
    if not x.is_cuda:
        raise RuntimeError("triton base3 LUT backend requires CUDA tensors")
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    w = w_packed.contiguous()
    m, k = x2.shape
    if k != in_features:
        raise RuntimeError(f"input K={k} does not match packed K={in_features}")
    n, words = w.shape
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(m, 16), triton.cdiv(n, 16))
    gamma_tensor = gamma.detach().to(device=x.device, dtype=torch.float32).contiguous()
    _base3_lut_ternary_matmul_kernel[grid](x2, w, bias, gamma_tensor, lut.contiguous(), y, m, k, n, words, BLOCK_M=16, BLOCK_N=16)
    return y.reshape(*x.shape[:-1], n)


def triton_base3_tile_dot_ternary_linear(x: torch.Tensor, w_packed: torch.Tensor, bias: torch.Tensor, gamma: torch.Tensor, in_features: int) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("triton is not available")
    if not x.is_cuda:
        raise RuntimeError("triton base3 tile-dot backend requires CUDA tensors")
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    w = w_packed.contiguous()
    m, k = x2.shape
    if k != in_features:
        raise RuntimeError(f"input K={k} does not match packed K={in_features}")
    if k % 128 != 0:
        raise RuntimeError(f"base3_tile_dot_ternary requires K divisible by 128, got K={k}")
    n, words = w.shape
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(m, 16), triton.cdiv(n, 32))
    gamma_tensor = gamma.detach().to(device=x.device, dtype=torch.float32).contiguous()
    _base3_tile_dot_ternary_matmul_kernel[grid](x2, w, bias, gamma_tensor, y, m, k, n, words, BLOCK_M=16, BLOCK_N=32, BLOCK_K=128)
    return y.reshape(*x.shape[:-1], n)
