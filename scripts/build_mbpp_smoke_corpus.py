from __future__ import annotations

import json
import os
import random
import urllib.request
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


OUT = Path("data/mbpp_smoke")
VOCAB_SIZE = int(os.environ.get("MBPP_SMOKE_VOCAB_SIZE", "5037"))
SEED = 0
DIRECT_URLS = [
    "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json",
    "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl",
]


def load_from_huggingface() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from datasets import load_dataset

    errors = []
    for name in ("google-research-datasets/mbpp", "mbpp"):
        for config in ("sanitized", None):
            try:
                ds = load_dataset(name, config) if config is not None else load_dataset(name)
            except Exception as exc:
                errors.append(f"{name}/{config}: {exc!r}")
                continue
            examples: list[dict[str, Any]] = []
            split_names = list(ds.keys())
            preferred = [s for s in ("sanitized", "train", "validation", "test", "prompt") if s in ds]
            for split in preferred + [s for s in split_names if s not in preferred]:
                for row in ds[split]:
                    item = dict(row)
                    item["_split"] = split
                    examples.append(item)
            if examples:
                return examples, {
                    "dataset_source": f"huggingface:{name}",
                    "huggingface_config": config or "default",
                    "sanitized_or_full": "sanitized" if "sanitized" in split_names or config == "sanitized" else "full/default",
                    "splits": preferred,
                }
    raise RuntimeError("failed to load MBPP from Hugging Face: " + " | ".join(errors))


def load_from_direct_json() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    errors = []
    for url in DIRECT_URLS:
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                raw = response.read().decode("utf-8")
        except Exception as exc:
            errors.append(f"{url}: {exc!r}")
            continue
        examples: list[dict[str, Any]] = []
        try:
            parsed = json.loads(raw)
            values = parsed if isinstance(parsed, list) else parsed.get("examples", parsed.get("data", []))
            examples = [dict(x) for x in values]
        except json.JSONDecodeError:
            examples = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if examples:
            return examples, {
                "dataset_source": url,
                "huggingface_config": None,
                "sanitized_or_full": "sanitized" if "sanitized" in url else "full",
                "splits": ["direct"],
            }
    raise RuntimeError("failed to load direct MBPP JSON: " + " | ".join(errors))


def load_mbpp() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        return load_from_huggingface()
    except Exception as hf_exc:
        examples, meta = load_from_direct_json()
        meta["huggingface_error"] = repr(hf_exc)
        return examples, meta


def first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def tests_text(row: dict[str, Any]) -> str:
    value = row.get("test_list", row.get("tests", row.get("test", "")))
    if isinstance(value, list):
        return "\n".join(str(x) for x in value if str(x).strip())
    return value.strip() if isinstance(value, str) else ""


def task_key(row: dict[str, Any], fallback: int) -> tuple[int, str]:
    for key in ("task_id", "id", "problem_id"):
        value = row.get(key)
        if isinstance(value, int):
            return value, str(value)
        if isinstance(value, str) and value.isdigit():
            return int(value), value
    return fallback, str(fallback)


def format_example(row: dict[str, Any], index: int) -> tuple[tuple[int, str], str]:
    prompt = first_text(row, ("prompt", "text", "task", "description", "question"))
    code = first_text(row, ("code", "solution", "canonical_solution", "source_code"))
    return task_key(row, index), f"# Task:\n{prompt}\n\n# Solution:\n{code}\n\n# Tests:\n{tests_text(row)}\n"


def train_tokenizer(texts: list[str]) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=VOCAB_SIZE, min_frequency=1, special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"])
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def main() -> None:
    random.seed(SEED)
    examples, meta = load_mbpp()
    formatted = [format_example(row, i) for i, row in enumerate(examples)]
    formatted = [(key, text) for key, text in formatted if text.strip()]
    if len(formatted) < 10:
        raise SystemExit("not enough MBPP examples found")
    formatted.sort(key=lambda item: item[0])
    split = max(1, int(len(formatted) * 0.9))
    train_items = formatted[:split]
    val_items = formatted[split:]
    train_text = "\n\n".join(text for _, text in train_items)
    val_text = "\n\n".join(text for _, text in val_items)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train.txt").write_text(train_text, encoding="utf-8")
    (OUT / "val.txt").write_text(val_text, encoding="utf-8")
    tokenizer = train_tokenizer([text for _, text in formatted])
    tokenizer.save(str(OUT / "tokenizer.json"))
    train_ids = torch.tensor(tokenizer.encode(train_text).ids, dtype=torch.long)
    val_ids = torch.tensor(tokenizer.encode(val_text).ids, dtype=torch.long)
    torch.save(train_ids, OUT / "train_ids.pt")
    torch.save(val_ids, OUT / "val_ids.pt")
    metadata = {
        **meta,
        "seed": SEED,
        "examples": len(formatted),
        "train_examples": len(train_items),
        "validation_examples": len(val_items),
        "vocab_size": int(tokenizer.get_vocab_size()),
        "tokenizer_type": "bytelevel_bpe",
        "train_tokens": int(train_ids.numel()),
        "validation_tokens": int(val_ids.numel()),
        "out": str(OUT),
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(metadata)


if __name__ == "__main__":
    main()
