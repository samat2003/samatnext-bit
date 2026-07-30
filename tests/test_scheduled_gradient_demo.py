import io
import tokenize
from pathlib import Path

import pytest
import torch

from scheduled_gradient_demo import Config, make_batches, train


def test_demo_is_under_100_lines_without_comments():
    path = Path(__file__).parents[1] / "scheduled_gradient_demo.py"
    source = path.read_text()
    assert len(source.splitlines()) < 100
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    assert all(token.type != tokenize.COMMENT for token in tokens)


@pytest.mark.parametrize(("update_every", "updates"), [(1, 20), (4, 5), (6, 4), (30, 1)])
def test_update_schedule(update_every, updates):
    cfg = Config()
    device = torch.device("cpu")
    result = train(cfg, update_every, make_batches(cfg, device), device)
    assert result["optimizer_updates"] == updates
    assert result["initial_mse"] == pytest.approx(
        train(cfg, update_every, make_batches(cfg, device), device)["initial_mse"]
    )


def test_invalid_update_every():
    cfg = Config()
    device = torch.device("cpu")
    with pytest.raises(ValueError, match="at least 1"):
        train(cfg, 0, make_batches(cfg, device), device)
