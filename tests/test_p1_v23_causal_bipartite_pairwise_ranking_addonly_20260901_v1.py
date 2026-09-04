from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v23_causal_bipartite_pairwise_ranking_addonly_20260901_v1.py"
SPEC = importlib.util.spec_from_file_location("p1_v23_tested", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _frame() -> pd.DataFrame:
    times = pd.date_range("2025-01-01", periods=12, freq="10min", tz="UTC")
    return pd.DataFrame(
        {
            "station": ["A"] * 6 + ["B"] * 6,
            "layer": [1] * 3 + [2] * 3 + [1] * 3 + [2] * 3,
            "_time": times,
            "temp": np.arange(12, dtype=float),
        }
    )


def test_context_basis_is_prefix_category_fitted_and_future_invariant(monkeypatch) -> None:
    monkeypatch.setattr(MODULE.shared, "_set_transport_context", lambda *_: None)
    frame = _frame()
    boundary = int(pd.Timestamp(frame.loc[7, "_time"]).value)
    first = MODULE.causal_context_features(frame, boundary, {})
    perturbed = frame.copy()
    perturbed.loc[8:, "station"] = "FUTURE_ONLY"
    perturbed.loc[8:, "layer"] = 99
    second = MODULE.causal_context_features(perturbed, boundary, {})
    np.testing.assert_array_equal(first[:8], second[:8])


def test_pairwise_logistic_orders_separable_positive_above_negative() -> None:
    positive = np.array([[2.0, 1.0], [3.0, 1.0], [4.0, 1.0]])
    negative = np.array([[-2.0, 1.0], [-3.0, 1.0], [-4.0, 1.0]])
    features = np.vstack([positive, negative])
    labels = np.array([1, 1, 1, 0, 0, 0], dtype=np.int8)
    model = MODULE.PairwiseLogisticRanker(
        loss="log_loss",
        penalty="l2",
        alpha=0.0001,
        max_iter=20,
        tol=None,
        class_weight={0: 1.0, 1: 1.0},
        shuffle=True,
        random_state=20260901,
    ).fit(features, labels)
    score = model.predict_proba(features)[:, 1]
    assert float(score[:3].min()) > float(score[3:].max())
    assert np.isfinite(score).all()


def test_config_freezes_pairwise_objective_and_nine_fit_addonly_contract() -> None:
    config = json.loads(MODULE.CONFIG.read_text(encoding="utf-8"))
    assert config["objective"]["kind"] == "symmetric_pairwise_logistic_bipartite_ranking"
    assert config["objective"]["outer_labels_in_pairs"] == 0
    assert config["representation"]["sensor_value_inputs"] == 0
    assert config["model"]["fits"] == 9
    assert len(config["model"]["seeds"]) * len(config["parts"]) == 9
    assert config["selection"]["outer_tuning"] == 0
    assert config["anchor"]["operation"] == "bitwise_or"
    assert config["anchor"]["removals"] == 0


def test_pairwise_score_is_shift_invariant() -> None:
    features = np.array([[2.0, 0.0], [3.0, 1.0], [-2.0, 0.0], [-3.0, 1.0]])
    labels = np.array([1, 1, 0, 0], dtype=np.int8)
    kwargs = dict(
        loss="log_loss",
        penalty="l2",
        alpha=0.0001,
        max_iter=20,
        tol=None,
        class_weight={0: 1.0, 1: 1.0},
        shuffle=True,
        random_state=20260901,
    )
    first = MODULE.PairwiseLogisticRanker(**kwargs).fit(features, labels)
    shifted = MODULE.PairwiseLogisticRanker(**kwargs).fit(features + 100.0, labels)
    np.testing.assert_allclose(first.coef_, shifted.coef_, atol=1e-12, rtol=0)
