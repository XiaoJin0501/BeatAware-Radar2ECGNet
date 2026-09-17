import json
import random

import numpy as np
import pytest
import torch

from src.utils.atomic_state import (
    atomic_json,
    atomic_npz,
    atomic_torch_save,
    capture_rng_state,
    restore_rng_state,
    transition_state,
)


def _step(model, optimizer, x, y):
    optimizer.zero_grad()
    prediction = model(x)
    loss = torch.nn.functional.mse_loss(prediction, y)
    loss.backward()
    optimizer.step()
    return float(loss)


def test_epoch_checkpoint_resumes_at_epoch_two_without_test_access(tmp_path):
    seed = 47
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    x, y = torch.randn(4, 3), torch.randn(4, 1)
    model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Dropout(0.2),
                                torch.nn.Linear(5, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    state_path = tmp_path / "run_state.json"
    transition_state(state_path, "TRAINING", test_accessed=False)

    history = [{"epoch": 1, "loss": _step(model, optimizer, x, y)}]
    checkpoint = tmp_path / "last_state.pt"
    atomic_torch_save(checkpoint, {
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None,
        "patience_counter": 0,
        "best_val": 0.25,
        "best_epoch": 1,
        "history": history,
        "rng_state": capture_rng_state(generator),
    })

    # Reference epoch 2 without interruption.
    _step(model, optimizer, x, y)
    reference = {key: value.detach().clone() for key, value in model.state_dict().items()}

    # New process: construction may consume RNG, but restoration must reproduce
    # the exact epoch-2 continuation.
    resumed = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Dropout(0.2),
                                  torch.nn.Linear(5, 1))
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    resumed_generator = torch.Generator()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    resumed.load_state_dict(saved["model_state_dict"])
    resumed_optimizer.load_state_dict(saved["optimizer_state_dict"])
    restore_rng_state(saved["rng_state"], resumed_generator)
    assert saved["epoch"] + 1 == 2
    _step(resumed, resumed_optimizer, x, y)

    for key, value in resumed.state_dict().items():
        assert torch.equal(value, reference[key])
    assert json.loads(state_path.read_text())["stage"] == "TRAINING"
    assert not (tmp_path / "test_metrics.json").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_outputs_and_terminal_complete_state(tmp_path):
    atomic_json(tmp_path / "value.json", {"ok": True})
    atomic_npz(tmp_path / "arrays.npz", {"x": np.arange(4)})
    assert json.loads((tmp_path / "value.json").read_text()) == {"ok": True}
    assert np.array_equal(np.load(tmp_path / "arrays.npz")["x"], np.arange(4))

    state = tmp_path / "run_state.json"
    transition_state(state, "TRAINING")
    transition_state(state, "CHECKPOINT_FROZEN")
    transition_state(state, "TEST_EVALUATING", test_accessed=True)
    transition_state(state, "COMPLETE", test_accessed=True)
    with pytest.raises(RuntimeError, match="terminal"):
        transition_state(state, "TEST_EVALUATING")
