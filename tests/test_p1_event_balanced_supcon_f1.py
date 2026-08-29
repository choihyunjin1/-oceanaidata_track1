from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from p1_qc.event_balanced_supcon_f1 import (
    apply_cell_topk,
    best_topk_rate,
    build_real_events,
    calibrate_cell_topk_rates,
    event_balanced_windows,
    pool_shared_hidden,
    soft_f1_loss,
    supervised_contrastive_loss,
)
from p1_qc.ms_tcn_asrf_data import SegmentLayout, build_asrf_targets


def _encoded() -> SimpleNamespace:
    rows = 48
    time = pd.date_range("2025-01-01", periods=rows, freq="10min", tz="UTC")
    keys = pd.DataFrame(
        {"station": ["A"] * rows, "year": [2025] * rows, "layer": [2] * rows, "time": time}
    )
    layout = SegmentLayout.from_aligned(keys.station, keys.year, keys.layer, keys.time)
    labels = np.zeros(rows, dtype=np.int8)
    labels[8:11] = 1
    labels[31:35] = 1
    kinds = np.asarray([""] * rows, dtype=object)
    kinds[8:11] = "spike"
    kinds[31:35] = "drift"
    targets = build_asrf_targets(labels, kinds, layout)
    surface = SimpleNamespace(rows=rows, keys=keys)
    return SimpleNamespace(surface=surface, layout=layout, targets=targets)


def test_real_event_balancing_uses_one_real_event_and_normal_per_event() -> None:
    encoded = _encoded()
    events, receipt = build_real_events(encoded)
    windows, class_ids, is_event, balance = event_balanced_windows(
        encoded, window_size=8, stride=4, seed=7
    )
    assert len(events) == 2
    assert receipt["synthetic_event_count"] == 0
    assert len(windows) == 4
    assert class_ids.tolist() == [0, 5, 4, 5]
    assert is_event.tolist() == [True, False, True, False]
    assert balance["matched_normal_window_count"] == 2
    for window, positive in zip(windows, is_event, strict=True):
        observed = encoded.targets.row_label[window.row_ids]
        assert bool(observed.any()) is bool(positive)


def test_supcon_soft_f1_and_hidden_pool_have_finite_gradients() -> None:
    embeddings = torch.randn(6, 4, requires_grad=True)
    classes = torch.tensor([0, 0, 1, 1, 5, 5])
    contrastive = supervised_contrastive_loss(embeddings, classes, temperature=0.1)
    logits = torch.randn(2, 5, requires_grad=True)
    labels = torch.tensor([[0, 1, 1, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.float32)
    valid = torch.ones_like(labels, dtype=torch.bool)
    f1 = soft_f1_loss(logits, labels, valid)
    hidden = torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5).requires_grad_()
    kinds = torch.zeros(2, 5, 5)
    kinds[0, 1:3, 0] = 1
    pooled = pool_shared_hidden(hidden, labels, kinds, valid, torch.tensor([0, 5]))
    total = contrastive + f1 + pooled.mean()
    total.backward()
    assert torch.isfinite(total)
    assert embeddings.grad is not None and torch.isfinite(embeddings.grad).all()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


def test_topk_calibration_is_training_only_and_anchor_preserving() -> None:
    score = np.asarray([0.1, 0.9, 0.8, 0.2, 0.7, 0.3])
    truth = np.asarray([0, 1, 1, 0, 0, 0], dtype=np.int8)
    rate, receipt = best_topk_rate(score, truth, maximum_rate=0.5)
    assert receipt["selected_k"] == 2
    assert np.isclose(rate, 2 / 6)
    keys = pd.DataFrame(
        {
            "station": ["A"] * 6,
            "layer": [2] * 6,
            "time": pd.date_range("2025-04-01", periods=6, freq="10min", tz="UTC"),
        }
    )
    rates, fallback, calibration = calibrate_cell_topk_rates(
        keys,
        score,
        truth,
        minimum_rows=4,
        minimum_positives=1,
        maximum_rate=0.5,
    )
    anchor = np.asarray([1, 0, 0, 0, 0, 0], dtype=np.int8)
    proposal, application = apply_cell_topk(
        keys, score, anchor, cell_rates=rates, fallback_rate=fallback
    )
    assert calibration["holdout_truth_rows_used"] == 0
    assert application["holdout_truth_rows_used"] == 0
    assert proposal[0] == 0
    assert int(proposal.sum()) == 2

