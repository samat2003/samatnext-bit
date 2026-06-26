#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.samatnext_bit.bench_speed import grad_norm_and_finite, load_mbpp_smoke, percentile, random_token_batches
from src.samatnext_bit.flops import estimate_training_flops
from src.samatnext_bit.model import DecoderLM
from src.samatnext_bit.utils import count_params, device, load_config, peak_memory_gb, reset_peak_memory, sync, timestamp, versions, write_json


BASELINE_NOTES = {
    "chainrule_baseline": {"params_m": 313.1, "tokens_sec": 7285.7, "peak_gb": 11.309, "initial_val_ce": 8.7542, "final_val_ce": 3.7325, "steps": 100},
    "mono_ue4_baseline": {"update_every": 4, "tokens_sec": 14233.5, "peak_gb": 6.297, "initial_val_ce": 8.7542, "final_val_ce": 4.4964, "steps": 100},
    "fused_adamw_mono_speed": {"tokens_sec": 21927.3, "note": "speed-optimized mono baseline if reproducible"},
}


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def make_optimizer(model: torch.nn.Module, lr: float, weight_decay: float, fused_requested: bool) -> tuple[torch.optim.Optimizer, bool, str | None]:
    kwargs: dict[str, Any] = {"lr": lr, "weight_decay": weight_decay}
    if fused_requested:
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(model.parameters(), **kwargs), bool(fused_requested), None
    except TypeError as exc:
        kwargs.pop("fused", None)
        return torch.optim.AdamW(model.parameters(), **kwargs), False, repr(exc)


def forward_with_aux(
    model: DecoderLM,
    idx: torch.Tensor,
    targets: torch.Tensor,
    aux_layers: list[int],
    aux_weights: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    _, t = idx.shape
    pos = torch.arange(t, device=idx.device)
    x = model.tok(idx) + model.pos(pos)[None, :, :]
    total_loss = idx.new_tensor(0.0, dtype=torch.float32)
    main_logits = None
    aux_set = set(aux_layers)
    for layer_idx, block in enumerate(model.blocks, start=1):
        x = block(x)
        if layer_idx in aux_set:
            logits = model.lm_head(model.norm(x))
            weight = float(aux_weights.get(str(layer_idx), aux_weights.get(layer_idx, 0.0)))
            if layer_idx == len(model.blocks):
                main_logits = logits
            if weight:
                total_loss = total_loss + weight * F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    if main_logits is None:
        main_logits = model.lm_head(model.norm(x))
        total_loss = total_loss + F.cross_entropy(main_logits.reshape(-1, main_logits.size(-1)), targets.reshape(-1))
    return main_logits, total_loss


def model_loss(model: DecoderLM, x: torch.Tensor, y: torch.Tensor, spec: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    if spec.get("aux_loss"):
        return forward_with_aux(model, x, y, list(spec.get("aux_layers", [])), dict(spec.get("aux_weights", {})))
    return model(x, y)


@torch.no_grad()
def eval_ce(model: DecoderLM, batches: list[tuple[torch.Tensor, torch.Tensor]], amp_dtype: torch.dtype, use_amp: bool) -> float:
    model.eval()
    losses = []
    for x, y in batches:
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            _, loss = model(x, y)
        losses.append(float(loss.item()))
    model.train()
    return statistics.mean(losses)


@torch.no_grad()
def generate_samples(model: DecoderLM, tokenizer: Any, dev: torch.device, prompts: list[str], max_new_tokens: int) -> dict[str, str]:
    model.eval()
    samples = {}
    for i, prompt in enumerate(prompts):
        torch.manual_seed(3000 + i)
        ids = tokenizer.encode(prompt).ids
        idx = torch.tensor([ids], dtype=torch.long, device=dev)
        for _ in range(max_new_tokens):
            logits, _ = model(idx[:, -model.seq_len :])
            probs = torch.softmax(logits[:, -1, :] / 0.8, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
        samples[prompt] = tokenizer.decode(idx[0].tolist())
    model.train()
    return samples


def should_chainrule_step(step: int, spec: dict[str, Any]) -> tuple[bool, bool]:
    if spec["training_rule"] == "chainrule":
        return True, False
    warmup_steps = int(spec.get("chainrule_warmup_steps", 0) or 0)
    if step <= warmup_steps:
        return True, False
    anchor_interval = spec.get("anchor_interval")
    if anchor_interval and step % int(anchor_interval) == 0:
        return True, True
    return step % int(spec["update_every"]) == 0, False


def run_track(cfg: dict[str, Any], track_name: str, spec: dict[str, Any], args: argparse.Namespace, dev: torch.device) -> dict[str, Any]:
    train_data, val_data, data_meta, tokenizer = load_mbpp_smoke(dev)
    vocab_size = int(data_meta["vocab_size"])
    layers = int(cfg["layers"])
    hidden = int(cfg["hidden"])
    heads = int(cfg["heads"])
    seq = int(cfg["seq_len"])
    batch = int(cfg["batch_size"])
    steps = int(args.steps or spec.get("steps", cfg["steps"]))
    use_amp = bool(cfg.get("amp", True))
    dtype_name = str(args.dtype or cfg.get("dtype", "float16"))
    amp_dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    grad_clip = spec.get("grad_clip", cfg.get("grad_clip"))
    grad_clip = None if grad_clip is None else float(grad_clip)

    torch.cuda.empty_cache()
    torch.manual_seed(int(cfg.get("seed", 0)))
    model = DecoderLM(vocab_size, seq, hidden, layers, heads, bitnet=False, backend="fp").to(dev)
    total_params = count_params(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt, fused_used, fused_error = make_optimizer(model, float(cfg["lr"]), float(cfg["weight_decay"]), bool(spec.get("fused_adamw", True)))
    train_batches = random_token_batches(train_data, batch, seq, steps + int(cfg.get("warmup_timed_steps", 3)) + 4, dev, int(cfg.get("seed", 0)))
    val_batches = random_token_batches(val_data, batch, seq, int(cfg.get("eval_batches", 8)), dev, int(cfg.get("seed", 0)) + 1000)

    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        _, initial_train_loss = model(*train_batches[0])
    initial_train_ce = float(initial_train_loss.item())
    initial_val_ce = eval_ce(model, val_batches, amp_dtype, use_amp)

    for warmup in range(int(cfg.get("warmup_timed_steps", 3))):
        x, y = train_batches[warmup]
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            _, loss = model_loss(model, x, y, spec)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

    sync(dev)
    reset_peak_memory(dev)
    times: list[float] = []
    train_losses: list[float] = []
    grad_norms: list[float] = []
    gradients_finite = True
    loss_finite = True
    optimizer_updates = 0
    anchor_updates = 0
    timed_start = time.perf_counter()

    for step in range(1, steps + 1):
        x, y = train_batches[step]
        do_update, is_anchor = should_chainrule_step(step, spec)
        start = time.perf_counter()
        if do_update:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                _, loss = model_loss(model, x, y, spec)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            norm, finite = grad_norm_and_finite(model)
            grad_norms.append(norm)
            gradients_finite = gradients_finite and finite
            opt.step()
            optimizer_updates += 1
            anchor_updates += int(is_anchor)
        else:
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    _, loss = model(x, y)
        sync(dev)
        times.append((time.perf_counter() - start) * 1000.0)
        train_losses.append(float(loss.item()))
        loss_finite = loss_finite and bool(torch.isfinite(loss).item())

    wall_clock_seconds = time.perf_counter() - timed_start
    final_train_ce = train_losses[-1]
    final_val_ce = eval_ce(model, val_batches, amp_dtype, use_amp)
    tokens_per_step = batch * (seq - 1)
    mean_ms = statistics.mean(times)
    flops = estimate_training_flops(
        training_rule="chainrule" if spec["training_rule"] == "chainrule" else "mono",
        total_params=total_params,
        active_params=total_params,
        active_layers=layers,
        update_every=int(spec["update_every"]),
    )
    samples = generate_samples(model, tokenizer, dev, list(cfg.get("generation_prompts", [])), int(cfg.get("sample_tokens", 96)))
    val_delta = initial_val_ce - final_val_ce
    minutes = wall_clock_seconds / 60.0
    tokens_processed = tokens_per_step * steps
    return {
        "completed": True,
        "track": track_name,
        "true_dense_active": True,
        "total_params": total_params,
        "active_params": total_params,
        "trainable_params": trainable_params,
        "layers": layers,
        "active_layers": layers,
        "hidden": hidden,
        "heads": heads,
        "vocab": vocab_size,
        "batch": batch,
        "seq": seq,
        "dtype": dtype_name,
        "optimizer": "AdamW",
        "fused_optimizer": fused_used,
        "fused_optimizer_error": fused_error,
        "update_every": int(spec["update_every"]),
        "anchor_interval": spec.get("anchor_interval"),
        "warmup_steps": int(spec.get("chainrule_warmup_steps", 0) or 0),
        "aux_loss": bool(spec.get("aux_loss", False)),
        "aux_loss_weights": spec.get("aux_weights", {}),
        "steps": steps,
        "optimizer_updates": optimizer_updates,
        "full_chainrule_anchor_updates": anchor_updates,
        "tokens_sec": tokens_per_step / (mean_ms / 1000.0),
        "mean_ms_step": mean_ms,
        "p50_ms_step": percentile(times, 0.50),
        "p90_ms_step": percentile(times, 0.90),
        "p99_ms_step": percentile(times, 0.99),
        "peak_cuda_memory_gb": peak_memory_gb(dev),
        "initial_train_ce": initial_train_ce,
        "final_train_ce": final_train_ce,
        "initial_val_ce": initial_val_ce,
        "final_val_ce": final_val_ce,
        "val_ce_delta": val_delta,
        "final_val_ppl": math.exp(final_val_ce) if final_val_ce < 50 else float("inf"),
        "ce_improvement_per_minute": val_delta / minutes if minutes > 0 else float("nan"),
        "ce_improvement_per_1b_tokens": val_delta / (tokens_processed / 1e9),
        "gradients_finite": gradients_finite,
        "nan_or_inf": not (gradients_finite and loss_finite),
        "generated_samples": samples,
        "exact_command": " ".join(sys.argv),
        "git_commit_hash": git_commit_hash(),
        "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "dataset": args.dataset,
        "dataset_metadata": data_meta,
        "wall_clock_seconds": wall_clock_seconds,
        "tokens_processed": tokens_processed,
        "estimated_total_training_flops_per_token": flops.total_training_flops_per_token,
    }


def select_best(rows: list[dict[str, Any]]) -> str | None:
    completed = [r for r in rows if r.get("completed")]
    chain = next((r for r in completed if r["track"] == "dense313m_chainrule_reference"), None)
    if chain is None:
        candidates = completed
    else:
        candidates = [r for r in completed if float(r["tokens_sec"]) > float(chain["tokens_sec"]) and r["track"] != chain["track"]]
    if not candidates:
        return None
    return max(candidates, key=lambda r: float(r["ce_improvement_per_minute"]))["track"]


def write_report(path: Path, payload: dict[str, Any]) -> None:
    rows = payload.get("sweep_results", payload["results"])
    long_rows = payload.get("long_run_results", [])
    completed = [r for r in rows if r.get("completed")]
    chain = next((r for r in completed if r["track"] == "dense313m_chainrule_reference"), None)
    ue4 = next((r for r in completed if r["track"] == "dense313m_mono_ue4_reference"), None)
    ue2 = next((r for r in completed if r["track"] == "dense313m_mono_ue2"), None)
    best_ce = min(completed, key=lambda r: float(r["final_val_ce"])) if completed else None
    best_per_min = max(completed, key=lambda r: float(r["ce_improvement_per_minute"])) if completed else None
    fastest = max(completed, key=lambda r: float(r["tokens_sec"])) if completed else None
    lines = [
        "# RESULTS_DENSE313M_LOSS_RECOVERY_v1",
        "",
        "New results are appended by this experiment only; earlier benchmark files are not overwritten.",
        "",
        "## Preserved Baselines",
        "",
        "| Baseline | Tok/s | Peak GB | Initial val CE | Final val CE | Steps |",
        "|---|---:|---:|---:|---:|---:|",
        f"| chain-rule baseline | {BASELINE_NOTES['chainrule_baseline']['tokens_sec']:,.1f} | {BASELINE_NOTES['chainrule_baseline']['peak_gb']:.3f} | {BASELINE_NOTES['chainrule_baseline']['initial_val_ce']:.4f} | {BASELINE_NOTES['chainrule_baseline']['final_val_ce']:.4f} | 100 |",
        f"| mono-forward UE=4 baseline | {BASELINE_NOTES['mono_ue4_baseline']['tokens_sec']:,.1f} | {BASELINE_NOTES['mono_ue4_baseline']['peak_gb']:.3f} | {BASELINE_NOTES['mono_ue4_baseline']['initial_val_ce']:.4f} | {BASELINE_NOTES['mono_ue4_baseline']['final_val_ce']:.4f} | 100 |",
        f"| fused AdamW mono speed baseline | {BASELINE_NOTES['fused_adamw_mono_speed']['tokens_sec']:,.1f} | n/a | n/a | n/a | n/a |",
        "",
        "## Sweep Results",
        "",
        "| Track | Dense active | Updates | Anchors | Aux | Tok/s | Mean ms | p50/p90/p99 ms | Peak GB | Init val CE | Final val CE | Delta | CE/min | PPL |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if not r.get("completed"):
            lines.append(f"| {r.get('track')} | yes | n/a | n/a | n/a | OOM/error | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {r['track']} | {'yes' if r['true_dense_active'] else 'no'} | {r['optimizer_updates']} | {r['full_chainrule_anchor_updates']} | {r['aux_loss']} | "
            f"{r['tokens_sec']:,.1f} | {r['mean_ms_step']:.2f} | {r['p50_ms_step']:.1f}/{r['p90_ms_step']:.1f}/{r['p99_ms_step']:.1f} | "
            f"{r['peak_cuda_memory_gb']:.3f} | {r['initial_val_ce']:.4f} | {r['final_val_ce']:.4f} | {r['val_ce_delta']:.4f} | "
            f"{r['ce_improvement_per_minute']:.4f} | {r['final_val_ppl']:.2f} |"
        )
    best_non_chain = None
    non_chain_completed = [r for r in completed if r["track"] != "dense313m_chainrule_reference"]
    if non_chain_completed:
        best_non_chain = min(non_chain_completed, key=lambda r: float(r["final_val_ce"]))
    gap = None
    if chain and best_non_chain:
        gap = float(best_non_chain["final_val_ce"]) - float(chain["final_val_ce"])
    best_ce_text = "n/a" if best_ce is None else f"{best_ce['track']} ({best_ce['final_val_ce']:.4f})"
    best_per_min_text = "n/a" if best_per_min is None else f"{best_per_min['track']} ({best_per_min['ce_improvement_per_minute']:.4f})"
    fastest_text = "n/a" if fastest is None else f"{fastest['track']} ({fastest['tokens_sec']:,.1f} tok/s)"
    speedup_text = "n/a"
    if chain and fastest:
        speedup_text = f"{fastest['track']} ({float(fastest['tokens_sec']) / float(chain['tokens_sec']):.2f}x over chain-rule)"
    anchor8 = next((r for r in completed if r["track"] == "dense313m_mono_ue4_anchor8"), None)
    anchor16 = next((r for r in completed if r["track"] == "dense313m_mono_ue4_anchor16"), None)
    warmup = next((r for r in completed if r["track"] == "dense313m_chainrule_warmup_then_mono"), None)
    aux = next((r for r in completed if r["track"] == "dense313m_mono_ue4_aux_loss"), None)
    anchor_answer = "not measured"
    if ue4 and anchor8 and anchor16:
        anchor_answer = (
            f"yes versus UE4 in this run: anchor8/anchor16 final val CE {anchor8['final_val_ce']:.4f}/"
            f"{anchor16['final_val_ce']:.4f} versus UE4 {ue4['final_val_ce']:.4f}; caveat, anchors overlap the UE4 update cadence here"
        )
    warmup_answer = "not measured"
    if ue4 and warmup:
        warmup_answer = f"yes: {warmup['final_val_ce']:.4f} versus UE4 {ue4['final_val_ce']:.4f}"
    aux_answer = "not measured"
    if ue4 and aux:
        aux_answer = f"no: aux final val CE {aux['final_val_ce']:.4f} versus UE4 {ue4['final_val_ce']:.4f}"
    lines += [
        "",
        "## Optional Longer Run",
        "",
        "| Track | Steps | Tok/s | Peak GB | Initial val CE | Final val CE | Delta | CE/min |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if long_rows:
        for r in long_rows:
            if not r.get("completed"):
                lines.append(f"| {r.get('track')} | {r.get('steps', 'n/a')} | error | n/a | n/a | n/a | n/a | n/a |")
                continue
            lines.append(
                f"| {r['track']} | {r['steps']} | {r['tokens_sec']:,.1f} | {r['peak_cuda_memory_gb']:.3f} | "
                f"{r['initial_val_ce']:.4f} | {r['final_val_ce']:.4f} | {r['val_ce_delta']:.4f} | {r['ce_improvement_per_minute']:.4f} |"
            )
    else:
        lines.append("| not run | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    lines += [
        "",
        "The optional longer run is not an equal-step comparison against the 100-step chain-rule reference.",
        "",
        "## Answers",
        "",
        f"1. Did update_every=2 improve loss compared to update_every=4? {'not measured' if not (ue2 and ue4) else ('yes' if ue2['final_val_ce'] < ue4['final_val_ce'] else 'no')}.",
        f"2. Did periodic full chain-rule anchor steps improve loss? {anchor_answer}.",
        f"3. Did chain-rule warmup improve mono loss? {warmup_answer}.",
        f"4. Did auxiliary losses help? {aux_answer}.",
        f"5. Best final validation CE: {best_ce_text}.",
        f"6. Best CE improvement per minute: {best_per_min_text}.",
        f"7. Best preserved speedup over chain-rule: {speedup_text}; raw fastest was {fastest_text}.",
        f"8. Did any 100-step method match or beat chain-rule final CE? {'not measured' if not chain else ('yes' if any(r is not chain and r['final_val_ce'] <= chain['final_val_ce'] for r in completed) else 'no')}.",
        f"9. Remaining 100-step CE gap to chain-rule for best non-chain method: {'n/a' if gap is None else f'{gap:.4f}'}.",
        "10. Strongest honest claim: plain mono-forward is faster and uses less memory, while this sweep measures whether UE2, anchor updates, warmup, or aux CE recover loss without claiming parity unless observed.",
        "11. Main limitation: this is a 100-step MBPP smoke-corpus experiment; it does not prove long-run convergence or coding ability.",
        "",
        "## Raw JSON",
        "",
        f"`{payload['json_path']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dense313m_loss_recovery.yaml")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--dataset", default="mbpp_smoke")
    parser.add_argument("--track", default="all")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"])
    parser.add_argument("--out-dir", default="runs")
    args = parser.parse_args()
    if args.dataset != "mbpp_smoke":
        raise SystemExit("this experiment is intentionally restricted to --dataset mbpp_smoke")
    dev = device()
    if dev.type != "cuda":
        raise SystemExit("CUDA is required for dense313m loss recovery")
    cfg = load_config(args.config)
    tracks = dict(cfg["tracks"])
    selected_names = list(tracks)
    prior_sweep = None
    if args.track != "all":
        if args.track == "best":
            latest = Path("runs/dense313m_loss_recovery_latest.json")
            if not latest.exists():
                raise SystemExit("--track best requires a prior sweep result")
            prior_sweep = json.loads(latest.read_text(encoding="utf-8"))
            best = select_best(prior_sweep.get("sweep_results", prior_sweep["results"]))
            if best is None:
                raise SystemExit("no faster-than-chain candidate available for --track best")
            selected_names = [best]
        else:
            selected_names = [name.strip() for name in args.track.split(",") if name.strip()]

    print(f"device={dev} gpu={versions()['gpu']} tracks={selected_names}", flush=True)
    results = []
    for name in selected_names:
        spec = tracks[name]
        print(f"running {name}", flush=True)
        try:
            row = run_track(cfg, name, spec, args, dev)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            row = {"completed": False, "track": name, "oom": True, "error": str(exc).splitlines()[0]}
        except Exception as exc:
            row = {"completed": False, "track": name, "oom": False, "error": repr(exc)}
        results.append(row)
        print(row, flush=True)
    out = Path(args.out_dir) / f"dense313m_loss_recovery_{timestamp()}" / "results.json"
    payload = {
        "config": args.config,
        "versions": versions(),
        "baseline_notes": BASELINE_NOTES,
        "results": results,
        "json_path": str(out),
    }
    if prior_sweep is not None:
        payload["sweep_results"] = prior_sweep.get("sweep_results", prior_sweep["results"])
        payload["long_run_results"] = results
    write_json(out, payload)
    write_json("runs/dense313m_loss_recovery_latest.json", payload)
    write_report(Path("RESULTS_DENSE313M_LOSS_RECOVERY_v1.md"), payload)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
