from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    input_dim: int = 32
    batch_size: int = 16
    steps: int = 20
    learning_rate: float = 3e-4
    seed: int = 0


def make_batches(cfg: Config, count: int, device: torch.device):
    """Random regression data: y = sum(x) + noise."""
    generator = torch.Generator().manual_seed(cfg.seed)
    batches = []

    for _ in range(count):
        x = torch.randn(cfg.batch_size, cfg.input_dim, generator=generator)
        y = x.sum(dim=-1, keepdim=True) + 0.1 * torch.randn(
            cfg.batch_size, 1, generator=generator
        )
        batches.append((x.to(device), y.to(device)))

    return batches


def make_model(cfg: Config) -> nn.Module:
    return nn.Linear(cfg.input_dim, 1)


@torch.no_grad()
def evaluate(model: nn.Module, batch) -> float:
    model.eval()
    x, y = batch
    loss = F.mse_loss(model(x), y)
    return float(loss.item())


def train_mode(cfg: Config, update_every: int, batches, device: torch.device) -> dict:
    """
    update_every=1: normal training, backward + step every batch.
    update_every=N: forward every step, but backward/update only every Nth step
    (gradient accumulation).
    """
    if update_every < 1:
        raise ValueError("update_every must be at least 1")

    torch.manual_seed(cfg.seed)  # same starting weights across modes
    model = make_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    train_batches = batches[: cfg.steps]
    eval_batch = batches[-1]
    initial_loss = evaluate(model, eval_batch)
    optimizer_updates = 0

    model.train()
    started = time.perf_counter()

    for step, (x, y) in enumerate(train_batches):
        loss = F.mse_loss(model(x), y)  # forward pass always runs

        if step % update_every == 0:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()    # full chain-rule backward pass
            optimizer.step()   # update model parameters
            optimizer_updates += 1

    elapsed = time.perf_counter() - started
    final_loss = evaluate(model, eval_batch)

    return {
        "mode": "chain_rule" if update_every == 1 else f"scheduled_UE{update_every}",
        "update_every": update_every,
        "optimizer_updates": optimizer_updates,
        "initial_mse": initial_loss,
        "final_mse": final_loss,
        "improvement": initial_loss - final_loss,
        "seconds": elapsed,
    }


def print_result(result: dict) -> None:
    print(f"\nMode: {result['mode']}")
    print(f"  update every:      {result['update_every']} step(s)")
    print(f"  optimizer updates: {result['optimizer_updates']}")
    print(f"  initial MSE:       {result['initial_mse']:.4f}")
    print(f"  final MSE:         {result['final_mse']:.4f}")
    print(f"  MSE improvement:   {result['improvement']:.4f}")
    print(f"  seconds:           {result['seconds']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-every", type=int, default=4)
    args = parser.parse_args()

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batches = make_batches(cfg, cfg.steps + 1, device)

    print(f"Device: {device}")
    print_result(train_mode(cfg, update_every=1, batches=batches, device=device))
    print_result(
        train_mode(cfg, update_every=args.update_every, batches=batches, device=device)
    )


if __name__ == "__main__":
    main()
