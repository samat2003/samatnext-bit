from __future__ import annotations

import argparse
import itertools
import math
import statistics
import time
from pathlib import Path

import torch
import yaml

from .data import ByteBatcher, dataset_info
from .model import DecoderLM
from .train import mark_packed_caches_dirty, refresh_packed_caches
from .utils import count_params, device, peak_memory_gb, reset_peak_memory, sync, timestamp, versions, write_json


MODES = [
    "fp_mono_update_every_2",
    "fp_mono_update_every_4",
    "fp_mono_update_every_8",
    "fake_ternary_mono_update_every_2",
    "fake_ternary_mono_update_every_4",
    "fake_ternary_mono_update_every_8",
    "base3_tile_dot_ternary_mono_update_every_2",
    "base3_tile_dot_ternary_mono_update_every_4",
]

DEFAULT_EXPERIMENT = {
    "name": "default",
    "layers": None,
    "active_layers": None,
    "recurrent_passes": 1,
}


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def mode_parts(mode: str) -> tuple[str, int]:
    if mode.startswith("fp_"):
        backend = "fp"
    elif mode.startswith("fake_ternary_"):
        backend = "fake_ternary"
    elif mode.startswith("base3_tile_dot_ternary_"):
        backend = "base3_tile_dot_ternary"
    else:
        raise ValueError(f"unsupported speed mode {mode}")
    update_every = int(mode.rsplit("_", 1)[1])
    return backend, update_every


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, int(round((len(xs) - 1) * pct))))
    return xs[idx]


def make_batch(batch: int, seq: int, dev: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, 256, (batch, seq), device=dev, dtype=torch.long)
    y = torch.roll(x, shifts=-1, dims=1)
    return x, y


def make_batches(cfg: dict, batch: int, seq: int, dev: torch.device, count: int) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], dict]:
    dataset = cfg.get("dataset", "synthetic_static")
    if dataset == "synthetic_static":
        x, y = make_batch(batch, seq, dev)
        return [(x, y)], {
            "dataset": dataset,
            "dataset_source": "static torch.randint CUDA batch",
            "vocab_size": 256,
            "total_tokens_loaded": batch * seq,
            "train_tokens": batch * seq,
            "preloaded_cuda_batches": True,
            "batch_sampling_in_timed_region": False,
        }
    info = dataset_info(dataset)
    batcher = ByteBatcher(dataset, batch, seq, dev, seed=int(cfg.get("seed", 0)))
    batches = [batcher.next_batch() for _ in range(count)]
    return batches, {
        "dataset": str(info["dataset"]),
        "dataset_source": str(info["source"]),
        "vocab_size": int(info["vocab_size"]),
        "total_tokens_loaded": int(info["total_tokens_loaded"]),
        "train_tokens": int(info["train_tokens"]),
        "preloaded_cuda_batches": True,
        "batch_sampling_in_timed_region": False,
    }


def grad_norm_and_finite(model: torch.nn.Module) -> tuple[float, bool]:
    total_sq = 0.0
    finite = True
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        finite = finite and bool(torch.isfinite(g).all().item())
        total_sq += float(g.float().pow(2).sum().item())
    return math.sqrt(total_sq), finite


def maybe_compile(model: torch.nn.Module, enabled: bool) -> tuple[torch.nn.Module, bool, str | None]:
    if not enabled:
        return model, False, None
    try:
        return torch.compile(model, mode="reduce-overhead"), True, None
    except Exception as exc:
        return model, False, repr(exc)


def run_one(base_cfg: dict, experiment: dict, mode: str, batch: int, seq: int, dev: torch.device) -> dict:
    backend, update_every = mode_parts(mode)
    bitnet = backend != "fp"
    dtype_name = base_cfg.get("amp_dtype", "bfloat16")
    amp_dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    use_amp = bool(base_cfg.get("amp", True))
    hidden = int(experiment.get("hidden", base_cfg["hidden"]))
    heads = int(experiment.get("heads", base_cfg["heads"]))
    logical_layers = int(experiment.get("layers") or base_cfg["layers"])
    active_layers = int(experiment.get("active_layers") or logical_layers)
    recurrent_passes = int(experiment.get("recurrent_passes", 1))
    experiment_name = str(experiment.get("name", base_cfg["name"]))
    effective_active_block_calls = active_layers * recurrent_passes
    steps = int(base_cfg.get("steps", 30))
    warmup = int(base_cfg.get("warmup_steps", 8))
    lr = float(base_cfg.get("lr", 3e-4))
    compile_requested = bool(base_cfg.get("torch_compile", False))
    cuda_graph_requested = bool(base_cfg.get("cuda_graph", False))
    if backend == "base3_tile_dot_ternary" and hidden % 128 != 0:
        return {
            "completed": False,
            "error": f"base3_tile_dot_ternary requires hidden/K divisible by 128, got hidden={hidden}",
            "config": base_cfg["name"],
            "experiment": experiment_name,
            "mode": mode,
            "batch": batch,
            "seq": seq,
            "layers": logical_layers,
            "active_layers": active_layers,
            "hidden": hidden,
            "heads": heads,
            "backend": backend,
            "update_every": update_every,
        }
    torch.manual_seed(int(base_cfg.get("seed", 0)))
    model = DecoderLM(
        256,
        seq,
        hidden,
        active_layers,
        heads,
        bitnet=bitnet,
        backend=backend,
        recurrent_passes=recurrent_passes,
    ).to(dev)
    model, compile_used, compile_error = maybe_compile(model, compile_requested)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(base_cfg.get("weight_decay", 0.01)))
    batches, data_meta = make_batches(base_cfg, batch, seq, dev, warmup + steps + 2)
    x, y = batches[0]
    tokens_step = batch * (seq - 1)
    training_update_performed = False
    optimizer_updates = 0
    grad_norms = []
    gradients_finite = True
    loss_finite = True
    cuda_graph_used = False
    cuda_graph_error = "not_attempted"
    if cuda_graph_requested:
        cuda_graph_error = "not_supported_for_update_every_dynamic_loop"
    try:
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                _, initial_loss = model(x, y)
        initial_ce = float(initial_loss.item())
        for step in range(warmup):
            x, y = batches[(step + 1) % len(batches)]
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y)
            loss_finite = loss_finite and bool(torch.isfinite(loss).item())
            if step % update_every == 0:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                norm, finite = grad_norm_and_finite(model)
                grad_norms.append(norm)
                gradients_finite = gradients_finite and finite
                opt.step()
                mark_packed_caches_dirty(model)
                refresh_packed_caches(model)
                training_update_performed = True
                optimizer_updates += 1
        sync(dev)
        reset_peak_memory(dev)
        times = []
        final_loss = float(loss.item())
        for step in range(steps):
            x, y = batches[(warmup + step + 1) % len(batches)]
            start = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y)
            loss_finite = loss_finite and bool(torch.isfinite(loss).item())
            if step % update_every == 0:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                norm, finite = grad_norm_and_finite(model)
                grad_norms.append(norm)
                gradients_finite = gradients_finite and finite
                opt.step()
                mark_packed_caches_dirty(model)
                refresh_packed_caches(model)
                training_update_performed = True
                optimizer_updates += 1
            sync(dev)
            times.append((time.perf_counter() - start) * 1000.0)
            final_loss = float(loss.item())
        ms = statistics.mean(times)
        tok_s = tokens_step / (ms / 1000.0)
        active_params = count_params(model)
        return {
            "completed": True,
            "config": base_cfg["name"],
            "experiment": experiment_name,
            "mode": mode,
            "layers": logical_layers,
            "active_layers": active_layers,
            "recurrent_passes": recurrent_passes,
            "effective_active_block_calls": effective_active_block_calls,
            "hidden": hidden,
            "heads": heads,
            "batch": batch,
            "seq": seq,
            "tokens_per_step": tokens_step,
            "ms_step": ms,
            "ms_p50": percentile(times, 0.50),
            "ms_p90": percentile(times, 0.90),
            "ms_p99": percentile(times, 0.99),
            "tokens_sec": tok_s,
            "initial_ce": initial_ce,
            "final_ce": final_loss,
            "ce_delta": initial_ce - final_loss,
            "peak_cuda_memory_gb": peak_memory_gb(dev),
            "initial_perplexity": math.exp(initial_ce),
            "final_perplexity": math.exp(final_loss),
            "grad_norm_mean": statistics.mean(grad_norms) if grad_norms else float("nan"),
            "grad_norm_max": max(grad_norms) if grad_norms else float("nan"),
            "gradients_finite": gradients_finite,
            "loss_finite": loss_finite,
            "nan_or_inf": not (gradients_finite and loss_finite),
            "backend": backend,
            "update_every": update_every,
            "torch_compile": compile_used,
            "torch_compile_error": compile_error,
            "cuda_graph": cuda_graph_used,
            "cuda_graph_error": cuda_graph_error,
            "training_update_performed": training_update_performed,
            "optimizer_updates": optimizer_updates,
            "reached_500k": tok_s >= 500_000,
            "reached_1m": tok_s >= 1_000_000,
            "oom": False,
            "active_params": active_params,
            "logical_params": active_params if active_layers == logical_layers else "not_instantiated",
            **data_meta,
        }
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        return {
            "completed": False,
            "oom": True,
            "error": "CUDA OOM: " + str(exc).splitlines()[0],
            "config": base_cfg["name"],
            "experiment": experiment_name,
            "mode": mode,
            "layers": logical_layers,
            "active_layers": active_layers,
            "recurrent_passes": recurrent_passes,
            "effective_active_block_calls": effective_active_block_calls,
            "hidden": hidden,
            "heads": heads,
            "batch": batch,
            "seq": seq,
            "backend": backend,
            "update_every": update_every,
        }
    except Exception as exc:
        return {
            "completed": False,
            "oom": False,
            "error": repr(exc),
            "config": base_cfg["name"],
            "experiment": experiment_name,
            "mode": mode,
            "layers": logical_layers,
            "active_layers": active_layers,
            "recurrent_passes": recurrent_passes,
            "effective_active_block_calls": effective_active_block_calls,
            "hidden": hidden,
            "heads": heads,
            "batch": batch,
            "seq": seq,
            "backend": backend,
            "update_every": update_every,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default="runs")
    args = parser.parse_args()
    cfg = load_config(args.config)
    dev = device()
    if dev.type != "cuda":
        raise SystemExit("bench_speed requires CUDA")
    print(f"device={dev} gpu={versions()['gpu']}", flush=True)
    explicit_runs = cfg.get("runs")
    if explicit_runs is None:
        modes = cfg.get("modes", MODES)
        pairs = cfg.get("batch_seq_pairs")
        if pairs is None:
            pairs = [{"batch": batch, "seq": seq} for seq, batch in itertools.product(cfg["seq_sweep"], cfg["batch_sweep"])]
        experiments = cfg.get("experiments")
        if experiments is None:
            logical_layers = int(cfg["layers"])
            experiments = [
                {
                    "name": cfg["name"],
                    "layers": logical_layers,
                    "active_layers": active_layers,
                    "recurrent_passes": 1,
                }
                for active_layers in cfg.get("active_layers_sweep", [logical_layers])
            ]
        run_specs = [
            {"experiment": experiment, "batch": int(pair["batch"]), "seq": int(pair["seq"]), "mode": mode}
            for experiment, pair, mode in itertools.product(experiments, pairs, modes)
        ]
    else:
        run_specs = []
        for spec in explicit_runs:
            experiment = {
                "name": spec["candidate"],
                "layers": int(spec["layers"]),
                "active_layers": int(spec["active_layers"]),
                "recurrent_passes": int(spec.get("recurrent_passes", 1)),
            }
            run_specs.append(
                {
                    "experiment": experiment,
                    "batch": int(spec["batch"]),
                    "seq": int(spec["seq"]),
                    "mode": spec["mode"],
                    "stop_if_dense_reaches_1m": bool(spec.get("stop_if_dense_reaches_1m", False)),
                }
            )
    rows = []
    dense_reached_1m = False
    for spec in run_specs:
        experiment = spec["experiment"]
        batch = spec["batch"]
        seq = spec["seq"]
        mode = spec["mode"]
        if dense_reached_1m and spec.get("stop_if_dense_reaches_1m"):
            continue
        print(
            "running "
            f"experiment={experiment.get('name', cfg['name'])} "
            f"active={experiment.get('active_layers', experiment.get('layers', cfg['layers']))}/"
            f"{experiment.get('layers', cfg['layers'])} "
            f"passes={experiment.get('recurrent_passes', 1)} "
            f"seq={seq} batch={batch} mode={mode}",
            flush=True,
        )
        row = run_one(cfg, experiment, mode, batch, seq, dev)
        rows.append(row)
        if (
            row.get("completed")
            and row.get("reached_1m")
            and row.get("active_layers") == row.get("layers")
        ):
            dense_reached_1m = True
        print(row, flush=True)
    payload = {"config": args.config, "versions": versions(), "results": rows}
    out = Path(args.out_dir) / f"{Path(args.config).stem}_{timestamp()}" / "speed_results.json"
    write_json(out, payload)
    write_json("runs/speed_latest.json", payload)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
