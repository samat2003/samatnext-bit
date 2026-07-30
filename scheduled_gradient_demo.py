import argparse
import time
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    input_dim: int = 32
    batch_size: int = 16
    steps: int = 20
    learning_rate: float = 3e-4
    seed: int = 0


def make_batches(cfg: Config, device: torch.device):
    generator = torch.Generator().manual_seed(cfg.seed)
    batches = []
    for _ in range(cfg.steps + 1):
        x = torch.randn(cfg.batch_size, cfg.input_dim, generator=generator)
        noise = 0.1 * torch.randn(cfg.batch_size, 1, generator=generator)
        batches.append((x.to(device), (x.sum(1, keepdim=True) + noise).to(device)))
    return batches


@torch.no_grad()
def evaluate(model: nn.Module, batch) -> float:
    model.eval()
    x, y = batch
    return nn.functional.mse_loss(model(x), y).item()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train(cfg: Config, update_every: int, batches, device: torch.device) -> dict:
    if update_every < 1:
        raise ValueError("update_every must be at least 1")
    torch.manual_seed(cfg.seed)
    model = nn.Linear(cfg.input_dim, 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    initial_mse = evaluate(model, batches[-1])
    updates = 0
    model.train()
    synchronize(device)
    started = time.perf_counter()
    for step, (x, y) in enumerate(batches[: cfg.steps], start=1):
        loss = nn.functional.mse_loss(model(x), y)
        if (step - 1) % update_every == 0:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            updates += 1
    synchronize(device)
    seconds = time.perf_counter() - started
    final_mse = evaluate(model, batches[-1])
    return {
        "update_every": update_every,
        "optimizer_updates": updates,
        "initial_mse": initial_mse,
        "final_mse": final_mse,
        "improvement": initial_mse - final_mse,
        "seconds": seconds,
    }


def print_result(result: dict) -> None:
    print(
        f"\nupdate_every={result['update_every']}\n"
        f"optimizer updates: {result['optimizer_updates']}\n"
        f"initial MSE: {result['initial_mse']:.4f}\n"
        f"final MSE: {result['final_mse']:.4f}\n"
        f"improvement: {result['improvement']:.4f}\n"
        f"seconds: {result['seconds']:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare scheduled sparse updates.")
    parser.add_argument("--update-every", type=int, default=4)
    args = parser.parse_args()
    if args.update_every < 1:
        parser.error("--update-every must be at least 1")
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batches = make_batches(cfg, device)
    train(cfg, 1, batches, device)
    print(f"device: {device}")
    print_result(train(cfg, 1, batches, device))
    print_result(train(cfg, args.update_every, batches, device))


if __name__ == "__main__":
    main()
