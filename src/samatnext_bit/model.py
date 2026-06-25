from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bitlinear import BitLinear


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def linear(bitnet: bool, hidden: int, out: int, backend: str) -> nn.Module:
    return BitLinear(hidden, out, backend=backend) if bitnet else nn.Linear(hidden, out)


class Block(nn.Module):
    def __init__(self, hidden: int, heads: int, bitnet: bool = False, backend: str = "fake"):
        super().__init__()
        assert hidden % heads == 0
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.n1 = RMSNorm(hidden)
        self.qkv = linear(bitnet, hidden, hidden * 3, backend)
        self.proj = linear(bitnet, hidden, hidden, backend)
        self.n2 = RMSNorm(hidden)
        self.fc1 = linear(bitnet, hidden, hidden * 4, backend)
        self.fc2 = linear(bitnet, hidden * 4, hidden, backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        q = q.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        x = x + self.proj(y)
        x = x + self.fc2(F.gelu(self.fc1(self.n2(x))))
        return x


class DecoderLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        hidden: int,
        layers: int,
        heads: int,
        bitnet: bool = False,
        backend: str = "fake",
        recurrent_passes: int = 1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.bitnet = bitnet
        self.backend = backend
        self.recurrent_passes = recurrent_passes
        self.tok = nn.Embedding(vocab_size, hidden)
        self.pos = nn.Embedding(seq_len, hidden)
        self.blocks = nn.ModuleList([Block(hidden, heads, bitnet, backend) for _ in range(layers)])
        self.norm = RMSNorm(hidden)
        self.lm_head = linear(bitnet, hidden, vocab_size, backend)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        _, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None, :, :]
        for _ in range(self.recurrent_passes):
            for block in self.blocks:
                x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def bitlinear_modules(self):
        return [m for m in self.modules() if isinstance(m, BitLinear)]

    def ternary_summary(self) -> dict[str, float | str]:
        mods = self.bitlinear_modules()
        if not mods:
            return {"ternary": "n/a", "gamma_mean": float("nan")}
        neg = zero = pos = gamma = 0.0
        for m in mods:
            s = m.ternary_summary()
            neg += s["neg_pct"]
            zero += s["zero_pct"]
            pos += s["pos_pct"]
            gamma += s["gamma_mean"]
        n = len(mods)
        return {"ternary": f"{neg/n:.1f}/{zero/n:.1f}/{pos/n:.1f}", "gamma_mean": gamma / n}
