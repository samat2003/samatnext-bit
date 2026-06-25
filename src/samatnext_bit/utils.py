from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()


def reset_peak_memory(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gb(dev: torch.device) -> float:
    if dev.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e9


def versions() -> dict[str, str]:
    try:
        import triton

        triton_version = triton.__version__
    except Exception:
        triton_version = "unavailable"
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "triton": triton_version,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")
