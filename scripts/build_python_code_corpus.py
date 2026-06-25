from __future__ import annotations

import hashlib
import os
import random
import sysconfig
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


OUT = Path("data/python_code_smoke")
MAX_FILE_BYTES = 2 * 1024 * 1024
VOCAB_SIZE = int(os.environ.get("PYTHON_CODE_VOCAB_SIZE", "16000"))
SEED = 0


def candidate_roots() -> list[Path]:
    roots = [Path("src"), Path("scripts"), Path("tests")]
    env = os.environ.get("PYTHON_CORPUS_DIR")
    if env:
        roots.append(Path(env))
    stdlib = sysconfig.get_paths().get("stdlib")
    if stdlib:
        roots.append(Path(stdlib))
    return roots


def iter_py_files() -> list[Path]:
    files: list[Path] = []
    for root in candidate_roots():
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            parts = set(path.parts)
            if "__pycache__" in parts or ".venv" in parts:
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            files.append(path)
    return sorted(set(files))


def read_unique_texts(paths: list[Path]) -> list[str]:
    seen: set[str] = set()
    texts: list[str] = []
    for path in paths:
        try:
            data = path.read_bytes()
        except OSError:
            continue
        digest = hashlib.sha256(data).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("utf-8", errors="ignore")
        if text.strip():
            texts.append(f"# file: {path.as_posix()}\n{text}\n")
    return texts


def train_tokenizer(texts: list[str]) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def encode(tokenizer: Tokenizer, text: str) -> torch.Tensor:
    ids = tokenizer.encode(text).ids
    return torch.tensor(ids, dtype=torch.long)


def main() -> None:
    random.seed(SEED)
    paths = iter_py_files()
    texts = read_unique_texts(paths)
    if len(texts) < 2:
        raise SystemExit("not enough Python files found to build corpus")
    random.Random(SEED).shuffle(texts)
    split = max(1, int(len(texts) * 0.9))
    train_text = "\n".join(texts[:split])
    val_text = "\n".join(texts[split:])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train.txt").write_text(train_text, encoding="utf-8")
    (OUT / "val.txt").write_text(val_text, encoding="utf-8")
    tokenizer = train_tokenizer(texts)
    tokenizer.save(str(OUT / "tokenizer.json"))
    train_ids = encode(tokenizer, train_text)
    val_ids = encode(tokenizer, val_text)
    torch.save(train_ids, OUT / "train_ids.pt")
    torch.save(val_ids, OUT / "val_ids.pt")
    print(
        {
            "files": len(paths),
            "unique_texts": len(texts),
            "train_chars": len(train_text),
            "val_chars": len(val_text),
            "vocab_size": tokenizer.get_vocab_size(),
            "train_tokens": int(train_ids.numel()),
            "val_tokens": int(val_ids.numel()),
            "out": str(OUT),
        }
    )


if __name__ == "__main__":
    main()
