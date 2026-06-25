from __future__ import annotations

import argparse
import itertools
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .data import ByteBatcher, corpus_bytes, dataset_info
from .flops import estimate_decoder_params, estimate_training_flops
from .model import DecoderLM
from .train import mark_packed_caches_dirty, refresh_packed_caches
from .utils import count_params, device, peak_memory_gb, reset_peak_memory, sync, timestamp, versions, write_json


MODES = [
    "fp_chainrule",
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


class RegularTorchBlock(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        assert hidden % heads == 0
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.n1 = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.proj = nn.Linear(hidden, hidden)
        self.n2 = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden * 4)
        self.fc2 = nn.Linear(hidden * 4, hidden)

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


class RegularTorchDecoderLM(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, hidden: int, layers: int, heads: int):
        super().__init__()
        self.seq_len = seq_len
        self.mixer_type = "softmax"
        self.tok = nn.Embedding(vocab_size, hidden)
        self.pos = nn.Embedding(seq_len, hidden)
        self.blocks = nn.ModuleList([RegularTorchBlock(hidden, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        _, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def bitlinear_modules(self):
        return []


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def mode_parts(mode: str) -> tuple[str, int, str]:
    if mode == "fp_chainrule":
        return "fp", 1, "chainrule"
    if mode.startswith("fp_"):
        backend = "fp"
    elif mode.startswith("fake_ternary_"):
        backend = "fake_ternary"
    elif mode.startswith("base3_tile_dot_ternary_"):
        backend = "base3_tile_dot_ternary"
    else:
        raise ValueError(f"unsupported speed mode {mode}")
    update_every = int(mode.rsplit("_", 1)[1])
    return backend, update_every, "mono"


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


def random_byte_batches(
    data: torch.Tensor,
    batch: int,
    seq: int,
    count: int,
    dev: torch.device,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    max_start = data.numel() - seq - 1
    batches = []
    for _ in range(count):
        starts = torch.randint(0, max_start, (batch,), generator=gen)
        x = torch.stack([data[s : s + seq] for s in starts.tolist()]).to(dev)
        y = torch.stack([data[s + 1 : s + seq + 1] for s in starts.tolist()]).to(dev)
        batches.append((x, y))
    return batches


@torch.no_grad()
def eval_loss(model: torch.nn.Module, batches: list[tuple[torch.Tensor, torch.Tensor]], amp_dtype: torch.dtype, use_amp: bool) -> float:
    losses = []
    model.eval()
    for x, y in batches:
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            _, loss = model(x, y)
        losses.append(float(loss.item()))
    model.train()
    return statistics.mean(losses)


@torch.no_grad()
def generate_samples(
    model: torch.nn.Module,
    dev: torch.device,
    prompt: str = "The ",
    max_new_chars: int = 128,
    temperature: float = 0.8,
    count: int = 3,
) -> list[str]:
    model.eval()
    samples = []
    for i in range(count):
        torch.manual_seed(1000 + i)
        idx = torch.tensor([[b for b in prompt.encode("utf-8")]], dtype=torch.long, device=dev)
        for _ in range(max_new_chars):
            idx_cond = idx[:, -model.seq_len :]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        text = bytes(idx[0].tolist()).decode("utf-8", errors="replace")
        samples.append(text)
    model.train()
    return samples


def maybe_compile(model: torch.nn.Module, enabled: bool) -> tuple[torch.nn.Module, bool, str | None]:
    if not enabled:
        return model, False, None
    try:
        return torch.compile(model, mode="reduce-overhead"), True, None
    except Exception as exc:
        return model, False, repr(exc)


def run_one(base_cfg: dict, experiment: dict, mode: str, batch: int, seq: int, dev: torch.device) -> dict:
    backend, update_every, training_rule = mode_parts(mode)
    bitnet = backend != "fp"
    dtype_name = base_cfg.get("amp_dtype", "bfloat16")
    amp_dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    use_amp = bool(base_cfg.get("amp", True))
    hidden = int(experiment.get("hidden", base_cfg["hidden"]))
    heads = int(experiment.get("heads", base_cfg["heads"]))
    logical_layers = int(experiment.get("layers") or base_cfg["layers"])
    active_layers = int(experiment.get("active_layers") or logical_layers)
    recurrent_passes = int(experiment.get("recurrent_passes", 1))
    mixer_type = str(experiment.get("mixer_type", base_cfg.get("mixer_type", "softmax")))
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
            "training_rule": training_rule,
            "update_every": update_every,
            "mixer_type": mixer_type,
            "official_gdn": False,
            "linear_recurrent_mixer": mixer_type == "simple_gdn",
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
        mixer_type=mixer_type,
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
            "mixer_type": mixer_type,
            "official_gdn": False,
            "linear_recurrent_mixer": mixer_type == "simple_gdn",
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
            "training_rule": training_rule,
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
            "training_rule": training_rule,
            "update_every": update_every,
            "mixer_type": mixer_type,
            "official_gdn": False,
            "linear_recurrent_mixer": mixer_type == "simple_gdn",
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
            "mixer_type": mixer_type,
            "official_gdn": False,
            "linear_recurrent_mixer": mixer_type == "simple_gdn",
        }


def run_validation_one(base_cfg: dict, spec: dict, dev: torch.device) -> dict:
    mode = spec["mode"]
    backend, update_every, training_rule = mode_parts(mode)
    bitnet = backend != "fp"
    dtype_name = base_cfg.get("amp_dtype", "bfloat16")
    amp_dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    use_amp = bool(base_cfg.get("amp", True))
    dataset = base_cfg.get("dataset", "english_validation")
    raw = torch.tensor(list(corpus_bytes(dataset)), dtype=torch.long)
    split = int(raw.numel() * float(base_cfg.get("train_split", 0.9)))
    train_data = raw[:split]
    val_data = raw[split:]
    batch = int(spec["batch"])
    seq = int(spec["seq"])
    steps = int(spec.get("steps", base_cfg.get("steps", 300)))
    max_seconds = spec.get("max_seconds")
    max_seconds = float(max_seconds) if max_seconds is not None else None
    warmup = int(base_cfg.get("warmup_steps", 5))
    val_batches_n = int(base_cfg.get("validation_batches", 8))
    checkpoints = set(int(x) for x in spec.get("checkpoints", base_cfg.get("checkpoints", [0, 50, 100, 150, 200, 250, 300])))
    update_checkpoints = set(int(x) for x in spec.get("update_checkpoints", base_cfg.get("update_checkpoints", [])))
    hidden = int(spec.get("hidden", base_cfg["hidden"]))
    heads = int(spec.get("heads", base_cfg["heads"]))
    layers = int(spec["layers"])
    active_layers = int(spec["active_layers"])
    recurrent_passes = int(spec.get("recurrent_passes", 1))
    mixer_type = str(spec.get("mixer_type", base_cfg.get("mixer_type", "softmax")))
    model_family = str(spec.get("model_family", "samatnext"))
    if model_family == "regular_torch_transformer" and training_rule != "chainrule":
        raise ValueError("regular_torch_transformer only supports chainrule in this audit")
    torch.manual_seed(int(base_cfg.get("seed", 0)))
    fallback_batch = spec.get("oom_fallback_batch")
    try:
        if model_family == "regular_torch_transformer":
            model = RegularTorchDecoderLM(256, seq, hidden, layers, heads).to(dev)
        else:
            model = DecoderLM(
                256,
                seq,
                hidden,
                active_layers,
                heads,
                bitnet=bitnet,
                backend=backend,
                recurrent_passes=recurrent_passes,
                mixer_type=mixer_type,
            ).to(dev)
    except torch.cuda.OutOfMemoryError as exc:
        if fallback_batch is None:
            torch.cuda.empty_cache()
            return {
                "completed": False,
                "oom": True,
                "error": "CUDA OOM during model init: " + str(exc).splitlines()[0],
                "config": base_cfg["name"],
                "experiment": spec["candidate"],
                "mode": mode,
                "training_rule": training_rule,
                "layers": layers,
                "active_layers": active_layers,
                "recurrent_passes": recurrent_passes,
                "hidden": hidden,
                "heads": heads,
                "batch": batch,
                "seq": seq,
                "backend": backend,
                "update_every": update_every,
                "mixer_type": mixer_type,
                "model_family": model_family,
                "official_gdn": False,
                "linear_recurrent_mixer": mixer_type == "simple_gdn",
            }
        torch.cuda.empty_cache()
        batch = int(fallback_batch)
        if model_family == "regular_torch_transformer":
            model = RegularTorchDecoderLM(256, seq, hidden, layers, heads).to(dev)
        else:
            model = DecoderLM(
                256,
                seq,
                hidden,
                active_layers,
                heads,
                bitnet=bitnet,
                backend=backend,
                recurrent_passes=recurrent_passes,
                mixer_type=mixer_type,
            ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(base_cfg.get("lr", 3e-4)), weight_decay=float(base_cfg.get("weight_decay", 0.01)))
    train_pool = int(spec.get("preloaded_train_batches", base_cfg.get("preloaded_train_batches", min(512, steps + warmup + 2))))
    try:
        train_batches = random_byte_batches(train_data, batch, seq, train_pool, dev, int(base_cfg.get("seed", 0)))
        val_batches = random_byte_batches(val_data, batch, seq, val_batches_n, dev, int(base_cfg.get("seed", 0)) + 99)
    except torch.cuda.OutOfMemoryError as exc:
        if fallback_batch is None or batch == int(fallback_batch):
            torch.cuda.empty_cache()
            return {
                "completed": False,
                "oom": True,
                "error": "CUDA OOM during batch preload: " + str(exc).splitlines()[0],
                "config": base_cfg["name"],
                "experiment": spec["candidate"],
                "mode": mode,
                "training_rule": training_rule,
                "layers": layers,
                "active_layers": active_layers,
                "recurrent_passes": recurrent_passes,
                "hidden": hidden,
                "heads": heads,
                "batch": batch,
                "seq": seq,
                "backend": backend,
                "update_every": update_every,
                "mixer_type": mixer_type,
                "model_family": model_family,
                "official_gdn": False,
                "linear_recurrent_mixer": mixer_type == "simple_gdn",
            }
        torch.cuda.empty_cache()
        batch = int(fallback_batch)
        train_batches = random_byte_batches(train_data, batch, seq, train_pool, dev, int(base_cfg.get("seed", 0)))
        val_batches = random_byte_batches(val_data, batch, seq, val_batches_n, dev, int(base_cfg.get("seed", 0)) + 99)
    data_meta = dataset_info(dataset)
    checkpoint_rows = []
    grad_norms = []
    gradients_finite = True
    loss_finite = True
    optimizer_updates = 0
    training_update_performed = False
    tokens_step = batch * (seq - 1)

    def checkpoint(step: int, train_batch_index: int) -> None:
        x, y = train_batches[train_batch_index % len(train_batches)]
        train_ce = eval_loss(model, [(x, y)], amp_dtype, use_amp)
        val_ce = eval_loss(model, val_batches, amp_dtype, use_amp)
        checkpoint_rows.append(
            {
                "step": step,
                "train_ce": train_ce,
                "val_ce": val_ce,
                "train_ppl": math.exp(train_ce),
                "val_ppl": math.exp(val_ce),
            }
        )

    checkpoint(0, 0)
    for step in range(warmup):
        x, y = train_batches[step]
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            _, loss = model(x, y)
        loss_finite = loss_finite and bool(torch.isfinite(loss).item())
    sync(dev)
    reset_peak_memory(dev)
    times = []
    timed_start = time.perf_counter()
    step = 0
    while step < steps:
        step += 1
        x, y = train_batches[(warmup + step) % len(train_batches)]
        start = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            _, loss = model(x, y)
        loss_finite = loss_finite and bool(torch.isfinite(loss).item())
        if training_rule == "chainrule" or (step - 1) % update_every == 0:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            norm, finite = grad_norm_and_finite(model)
            grad_norms.append(norm)
            gradients_finite = gradients_finite and finite
            opt.step()
            mark_packed_caches_dirty(model)
            refresh_packed_caches(model)
            optimizer_updates += 1
            training_update_performed = True
        sync(dev)
        times.append((time.perf_counter() - start) * 1000.0)
        if step in checkpoints or (optimizer_updates in update_checkpoints and (step - 1) % update_every == 0):
            checkpoint(step, warmup + step)
        if max_seconds is not None and (time.perf_counter() - timed_start) >= max_seconds:
            break
    if checkpoint_rows[-1]["step"] != step:
        checkpoint(step, warmup + step)
    ms = statistics.mean(times)
    wall_clock_seconds = sum(times) / 1000.0
    samples = generate_samples(model, dev)
    active_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = active_params if model_family == "regular_torch_transformer" else estimate_decoder_params(256, seq, hidden, layers)
    flops = estimate_training_flops(
        training_rule=training_rule,
        total_params=total_params,
        active_params=active_params,
        active_layers=active_layers,
        update_every=update_every,
    )
    estimated_tflops = (tokens_step / (ms / 1000.0)) * flops.total_training_flops_per_token / 1e12
    return {
        "completed": True,
        "config": base_cfg["name"],
        "experiment": spec["candidate"],
        "model_family": model_family,
        "mode": mode,
        "training_rule": training_rule,
        "dense_or_sparse": "dense" if active_layers == layers else "sparse/logical",
        "layers": layers,
        "total_layers": layers,
        "active_layers": active_layers,
        "recurrent_passes": recurrent_passes,
        "effective_active_block_calls": active_layers * recurrent_passes,
        "hidden": hidden,
        "heads": heads,
        "mixer_type": mixer_type,
        "official_gdn": False,
        "linear_recurrent_mixer": mixer_type == "simple_gdn",
        "batch": batch,
        "seq": seq,
        "steps": steps,
        "actual_steps": step,
        "comparison_type": spec.get("comparison_type", "validation"),
        "tokens_per_step": tokens_step,
        "ms_step": ms,
        "ms_p50": percentile(times, 0.50),
        "ms_p90": percentile(times, 0.90),
        "ms_p99": percentile(times, 0.99),
        "tokens_processed": tokens_step * step,
        "wall_clock_seconds": wall_clock_seconds,
        "tokens_sec": tokens_step / (ms / 1000.0),
        "peak_cuda_memory_gb": peak_memory_gb(dev),
        "backend": backend,
        "update_every": update_every,
        "torch_compile": False,
        "cuda_graph": False,
        "training_update_performed": training_update_performed,
        "optimizer_updates": optimizer_updates,
        "grad_norm_mean": statistics.mean(grad_norms) if grad_norms else float("nan"),
        "grad_norm_max": max(grad_norms) if grad_norms else float("nan"),
        "gradients_finite": gradients_finite,
        "loss_finite": loss_finite,
        "nan_or_inf": not (gradients_finite and loss_finite),
        "total_params": total_params,
        "active_params": active_params,
        "estimated_forward_flops_per_token": flops.forward_flops_per_token,
        "estimated_backward_update_flops_per_token": flops.backward_update_flops_per_token,
        "estimated_total_training_flops_per_token": flops.total_training_flops_per_token,
        "estimated_effective_tflops": estimated_tflops,
        "flop_estimate_formula": flops.formula,
        "flop_estimate_caveat": "Estimated from active parameter count; attention/softmax/norm/optimizer details are approximate.",
        "estimated_flops_to_final_val_ce": (tokens_step * step * flops.total_training_flops_per_token),
        "checkpoints": checkpoint_rows,
        "initial_ce": checkpoint_rows[0]["train_ce"],
        "final_ce": checkpoint_rows[-1]["train_ce"],
        "ce_delta": checkpoint_rows[0]["train_ce"] - checkpoint_rows[-1]["train_ce"],
        "initial_val_ce": checkpoint_rows[0]["val_ce"],
        "final_val_ce": checkpoint_rows[-1]["val_ce"],
        "final_validation_ce": checkpoint_rows[-1]["val_ce"],
        "final_validation_perplexity": checkpoint_rows[-1]["val_ppl"],
        "val_ce_delta": checkpoint_rows[0]["val_ce"] - checkpoint_rows[-1]["val_ce"],
        "samples": samples,
        "dataset": dataset,
        "dataset_source": data_meta["source"],
        "vocab_size": data_meta["vocab_size"],
        "total_tokens_loaded": int(data_meta["total_tokens_loaded"]),
        "train_tokens": int(split),
        "validation_tokens": int(raw.numel() - split),
        "preloaded_cuda_batches": True,
        "batch_sampling_in_timed_region": False,
        "token_sec_formula": "tokens/sec = batch * (seq - 1) / mean_step_time",
        "oom": False,
        "requested_batch": int(spec["batch"]),
        "oom_fallback_used": batch != int(spec["batch"]),
    }


def run_validation(cfg: dict, dev: torch.device) -> dict:
    rows = []
    budgets = {}
    for spec in cfg["runs"]:
        if "max_seconds_from" in spec:
            source = spec["max_seconds_from"]
            if source not in budgets:
                raise ValueError(f"max_seconds_from={source!r} has not run yet")
            spec = dict(spec)
            spec["max_seconds"] = budgets[source]
        print(f"validating candidate={spec['candidate']} batch={spec['batch']} seq={spec['seq']} mode={spec['mode']}", flush=True)
        try:
            row = run_validation_one(cfg, spec, dev)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            if spec.get("oom_fallback_batch") is None:
                row = {
                    "completed": False,
                    "oom": True,
                    "error": "CUDA OOM: " + str(exc).splitlines()[0],
                    "config": cfg["name"],
                    "experiment": spec["candidate"],
                    "mode": spec["mode"],
                    "requested_batch": int(spec["batch"]),
                    "batch": int(spec["batch"]),
                    "seq": int(spec["seq"]),
                }
            else:
                retry = dict(spec)
                retry["batch"] = int(spec["oom_fallback_batch"])
                retry.pop("oom_fallback_batch", None)
                row = run_validation_one(cfg, retry, dev)
                row["requested_batch"] = int(spec["batch"])
                row["oom_fallback_used"] = True
        rows.append(row)
        if spec.get("budget_name"):
            budgets[str(spec["budget_name"])] = float(row["wall_clock_seconds"])
        print(row, flush=True)
    baseline_name = str(cfg.get("baseline_candidate", "dense24_chainrule"))
    baseline = next((r for r in rows if r.get("completed") and r.get("experiment") == baseline_name), None)
    if baseline is not None:
        baseline_tok_s = float(baseline["tokens_sec"])
        baseline_flops = float(baseline["estimated_total_training_flops_per_token"])
        baseline_val_ce = float(baseline["final_val_ce"])
        for row in rows:
            if not row.get("completed"):
                continue
            speedup = float(row["tokens_sec"]) / baseline_tok_s if baseline_tok_s else float("nan")
            flop_reduction = baseline_flops / float(row["estimated_total_training_flops_per_token"])
            quality_delta = float(row["final_val_ce"]) - baseline_val_ce
            row["baseline_candidate"] = baseline_name
            row["speedup_vs_baseline"] = speedup
            row["flop_reduction_vs_baseline"] = flop_reduction
            row["quality_delta_vs_baseline"] = quality_delta
            row["speedup_vs_dense24_chainrule"] = speedup
            row["flop_reduction_vs_dense24_chainrule"] = flop_reduction
            row["quality_delta_vs_dense24_chainrule"] = quality_delta
    else:
        for row in rows:
            if row.get("completed"):
                row["baseline_candidate"] = baseline_name
                row["speedup_vs_baseline"] = "n/a"
                row["flop_reduction_vs_baseline"] = "n/a"
                row["quality_delta_vs_baseline"] = "n/a"
                row["speedup_vs_dense24_chainrule"] = "n/a"
                row["flop_reduction_vs_dense24_chainrule"] = "n/a"
                row["quality_delta_vs_dense24_chainrule"] = "n/a"
    return {"config": cfg["name"], "versions": versions(), "results": rows}


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
    if cfg.get("validation_experiment", False):
        payload = run_validation(cfg, dev)
        out = Path(args.out_dir) / f"{Path(args.config).stem}_{timestamp()}" / "speed_results.json"
        write_json(out, payload)
        write_json("runs/speed_latest.json", payload)
        print(f"wrote {out}", flush=True)
        return
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
