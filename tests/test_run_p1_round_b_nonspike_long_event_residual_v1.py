from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from p1_qc.features import FeatureBundle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_p1_round_b_nonspike_long_event_residual_v1.py"
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "p1_round_b_nonspike_long_event_residual_v1.json"
)


def _runner():
    spec = importlib.util.spec_from_file_location("p1_round_b_residual_runner", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_proves_exact_round_b_reuse_without_source_access(monkeypatch) -> None:
    monkeypatch.delenv("P1_DATA_DIR", raising=False)
    report = _runner().preflight(CONFIG_PATH)

    assert report["status"] == "PASS_READY_TO_SEAL_OR_EXECUTE"
    assert report["rows"] == 421032
    assert report["exact_round_b_default_equivalence"] is True
    assert report["round_b_base_model_fits_required"] == 0
    assert report["residual_model_fits_registered"] == 9
    assert report["protected_source_reads"] == 0
    assert report["outer_scores_computed"] == 0


def test_execute_source_resolver_requires_explicit_environment(monkeypatch) -> None:
    monkeypatch.delenv("P1_DATA_DIR", raising=False)

    try:
        _runner()._training_source()
    except RuntimeError as error:
        assert "P1_DATA_DIR" in str(error)
    else:
        raise AssertionError("missing P1_DATA_DIR must fail closed")


def test_fold_fit_smoke_uses_three_residual_models_only() -> None:
    runner = _runner()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["residual_model"]["parameters"].update(
        {"n_estimators": 3, "num_leaves": 7, "min_child_samples": 2, "n_jobs": 1}
    )
    rows = 60
    time = pd.date_range("2024-01-01T00:00:00+09:00", periods=rows, freq="10min")
    train = pd.DataFrame(
        {
            "station": ["S-ORS"] * rows,
            "year": [2024] * rows,
            "layer": [1] * rows,
            "time": time.astype(str),
            "label": np.r_[np.ones(19, dtype=np.int8), np.zeros(rows - 19, dtype=np.int8)],
            "anomaly_type": np.r_[
                np.full(19, "offset", dtype=object), np.full(rows - 19, "", dtype=object)
            ],
        }
    )
    features = pd.DataFrame(
        {
            "feature_a": np.linspace(-1.0, 1.0, rows),
            "feature_b": np.sin(np.arange(rows) / 5.0),
        }
    )
    bundle = FeatureBundle(features, ("feature_a", "feature_b"), ())
    train_idx = np.arange(50, dtype=np.int64)
    val_idx = np.arange(50, 60, dtype=np.int64)
    part = train.iloc[val_idx][["station", "year", "layer", "time"]].reset_index(drop=True)
    part["row_position"] = val_idx
    part["fold"] = "2025_q2"
    part["spike_candidate"] = False
    part[f"{runner.BASE_PREFIX}__prediction"] = np.r_[1, np.zeros(9, dtype=np.int8)]
    for seed in config["surface"]["seeds"]:
        part[f"{runner.BASE_PREFIX}__seed_{seed}__prediction"] = part[
            f"{runner.BASE_PREFIX}__prediction"
        ]
    fold = {
        "name": "2025_q2",
        "ordinal": 0,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "part": part,
        "train_end_utc": "2024-01-01T08:10:00+00:00",
        "val_start_utc": "2024-01-01T08:20:00+00:00",
    }

    output, audit = runner._fit_fold(train, bundle, config, fold)

    assert len(output) == 10
    assert audit["model_fits"] == 3
    assert audit["residual_positive_rows"] == 19
    assert set(np.unique(output["candidate_prediction"])).issubset({0, 1})
