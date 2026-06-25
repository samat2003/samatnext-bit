import random
from pathlib import Path

import torch


TINY_CODE = (
    "def add(a, b):\n    return a + b\n\n"
    "for i in range(16):\n    print(add(i, 3))\n\n"
    "class Counter:\n    def __init__(self):\n        self.n = 0\n"
    "    def step(self):\n        self.n += 1\n        return self.n\n"
)


def corpus_bytes(name: str) -> bytes:
    if name == "tiny_code":
        return (TINY_CODE * 4096).encode("utf-8")
    if name == "pattern":
        return (("abc123XYZ\n" * 8192)).encode("utf-8")
    if name == "counting":
        return (" ".join(str(i % 1000) for i in range(50000))).encode("utf-8")
    if name == "english_smoke":
        path = Path("data/english_smoke.txt")
        if not path.exists():
            raise FileNotFoundError("data/english_smoke.txt is missing")
        text = path.read_text(encoding="utf-8")
        repeats = max(1, (1_000_000 // max(1, len(text))) + 1)
        return (text * repeats).encode("utf-8")
    if name == "english_validation":
        path = Path("data/english_validation.txt")
        if not path.exists():
            raise FileNotFoundError("data/english_validation.txt is missing")
        return path.read_bytes()
    raise ValueError(f"unknown dataset {name!r}")


def _corpus(name: str) -> bytes:
    return corpus_bytes(name)


def dataset_info(name: str) -> dict[str, int | str]:
    data = corpus_bytes(name)
    source = "builtin synthetic"
    if name == "english_smoke":
        source = "local fallback data/english_smoke.txt repeated in memory"
    if name == "english_validation":
        source = "downloaded Tiny Shakespeare data/english_validation.txt"
    return {
        "dataset": name,
        "source": source,
        "vocab_size": 256,
        "total_tokens_loaded": len(data),
        "train_tokens": len(data),
    }


class ByteBatcher:
    def __init__(self, dataset: str, batch_size: int, seq_len: int, device: torch.device, seed: int = 0):
        self.data = torch.tensor(list(_corpus(dataset)), dtype=torch.long)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.rng = random.Random(seed)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.data) - self.seq_len - 1
        starts = [self.rng.randint(0, max_start) for _ in range(self.batch_size)]
        xs = [self.data[s : s + self.seq_len] for s in starts]
        ys = [self.data[s + 1 : s + self.seq_len + 1] for s in starts]
        return torch.stack(xs).to(self.device), torch.stack(ys).to(self.device)
