import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

import scripts.run_structured_event_gate as runner


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, 4, 1))

    def forward(self, x):
        return self.bias.expand(len(x), 4, 1600)


class _TinyFold(Dataset):
    constructions = []

    def __init__(self, fold, split, allow_test=False, limit=None):
        assert split != "test" or allow_test
        self.constructions.append(split)
        self.points = [{p: np.array([400.0, 800.0]) for p in runner.FIDUCIALS}] * 2

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {"wsst": torch.zeros(50, 2, 2), "event": torch.zeros(4, 1600),
                "index": index}


def _fake_evaluate(model, dataset, device, batch_size=16):
    per = {p: {"gt": 2, "pred": 2, "match": 2, "f1_100ms": .8,
               "coverage": 1.0, "medae_ms": 10.0} for p in runner.FIDUCIALS}
    return {**per, "macro_f1_100ms": .8, "macro_medae_ms": 10.0,
            "order_complete": 2, "order_violations": 0, "order_violation_rate": 0.0}, np.zeros((2, 4, 1600))


def test_interruption_resumes_before_test_and_test_only_after_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "OUT", tmp_path)
    monkeypatch.setattr(runner, "EventFold", _TinyFold)
    monkeypatch.setattr(runner, "StructuredEventREM", _TinyModel)
    monkeypatch.setattr(runner, "evaluate", _fake_evaluate)
    _TinyFold.constructions.clear()

    with pytest.raises(RuntimeError, match="SIMULATED_INTERRUPTION"):
        runner.train_fold(1, torch.device("cpu"), max_epochs=2, patience=15,
                          interrupt_after=1)
    state = json.loads((tmp_path / "fold_01/run_state.json").read_text())
    assert state["stage"] == "TRAINING"
    assert "test" not in _TinyFold.constructions
    saved = torch.load(tmp_path / "fold_01/checkpoints/last_state.pt",
                       map_location="cpu", weights_only=False)
    assert saved["epoch"] == 1

    runner.train_fold(1, torch.device("cpu"), max_epochs=2, patience=15)
    state = json.loads((tmp_path / "fold_01/run_state.json").read_text())
    assert state["stage"] == "COMPLETE"
    assert state["test_accessed"] is True
    assert _TinyFold.constructions[-1] == "test"
    history = json.loads((tmp_path / "fold_01/history.json").read_text())
    assert [row["epoch"] for row in history] == [1, 2]


def test_complete_state_is_terminal(tmp_path):
    path = tmp_path / "state.json"
    runner.transition(path, "TRAINING")
    runner.transition(path, "CHECKPOINT_FROZEN")
    runner.transition(path, "TEST_EVALUATING")
    runner.transition(path, "COMPLETE")
    with pytest.raises(RuntimeError, match="terminal"):
        runner.transition(path, "COMPLETE")


def test_100ms_f1_is_distinct_from_150ms_endpoint_pairing():
    probability = np.zeros((1, 4, 1600), dtype=float)
    probability[:, :, 125] = 1.0  # 25 samples = 125 ms from the reference
    references = [{p: np.array([100.0]) for p in runner.FIDUCIALS}]
    result = runner.score(probability, references)
    for point in runner.FIDUCIALS:
        assert result[point]["match"] == 1       # eligible for 150-ms MedAE pairing
        assert result[point]["match_100ms"] == 0
        assert result[point]["medae_ms"] == 125.0
        assert result[point]["f1_100ms"] == 0.0
