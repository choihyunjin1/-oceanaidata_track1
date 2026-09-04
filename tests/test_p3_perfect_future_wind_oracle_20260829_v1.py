from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_p3_perfect_future_wind_oracle_20260829_v1.py"
SPEC = importlib.util.spec_from_file_location("p3_future_wind_oracle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _six_lead_frame(cases: int = 6) -> pd.DataFrame:
    folds = ["fold_a", "fold_b", "fold_c"]
    stations = ["G-ORS", "I-ORS", "S-ORS"]
    rows: list[dict[str, object]] = []
    for case in range(cases):
        fold = folds[case % 3]
        station = stations[case % 3]
        anchor = pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=case * 10)
        for lead in [3, 6, 9, 12, 18, 24]:
            rows.append(
                {
                    "fold": fold,
                    "anchor_id": case,
                    "station": station,
                    "anchor_time": anchor,
                    "lead_h": lead,
                    "candidate_final": 2.0,
                    "target_hs": 2.0 + (0.1 if lead in [18, 24] else 0.0),
                    "wave_feature": float(case),
                    "delta_u_3h": 0.0,
                    "delta_v_3h": 0.0,
                    "delta_u_6h": 0.0,
                    "delta_v_6h": 0.0,
                    "delta_u_9h": 0.0,
                    "delta_v_9h": 0.0,
                    "delta_u_12h": 0.0,
                    "delta_v_12h": 0.0,
                    "delta_u_18h": float(case),
                    "delta_v_18h": -float(case),
                    "delta_u_24h": float(case) + 0.5,
                    "delta_v_24h": -float(case) - 0.5,
                }
            )
    return pd.DataFrame(rows)


def test_embargo_removes_less_than_78h_but_keeps_exact_boundary() -> None:
    validation = pd.DataFrame(
        {
            "fold": ["v"],
            "anchor_id": [10],
            "station": ["G-ORS"],
            "anchor_time": [pd.Timestamp("2020-01-10", tz="UTC")],
        }
    )
    train = pd.DataFrame(
        {
            "fold": ["a", "a", "a"],
            "anchor_id": [1, 2, 3],
            "station": ["G-ORS", "G-ORS", "I-ORS"],
            "anchor_time": [
                pd.Timestamp("2020-01-10", tz="UTC") - pd.Timedelta(hours=77),
                pd.Timestamp("2020-01-10", tz="UTC") - pd.Timedelta(hours=78),
                pd.Timestamp("2020-01-10", tz="UTC"),
            ],
        }
    )
    kept = MODULE.purge_embargo(train, validation, 78)
    assert set(kept["anchor_id"]) == {2, 3}


def test_oracle_prediction_is_short_lead_exact_no_op_and_uses_shared_alpha() -> None:
    frame = _six_lead_frame(cases=9)
    config = {
        "surface": {
            "folds": ["fold_a", "fold_b", "fold_c"],
            "active_leads_h": [18, 24],
            "exact_no_op_leads_h": [3, 6, 9, 12],
            "embargo_hours": 78,
            "expected_rows": 54,
        },
        "oracle": {
            "control_wave_features": ["wave_feature"],
            "ridge_alphas": [1.0, 10.0],
        },
    }
    blind, records = MODULE.make_oracle_predictions(config, frame)
    short = blind["lead_h"].isin([3, 6, 9, 12])
    assert np.array_equal(blind.loc[short, "control_prediction"], np.full(short.sum(), 2.0))
    assert np.array_equal(
        blind.loc[short, "treatment_prediction"], np.full(short.sum(), 2.0)
    )
    assert len(blind) == 54
    assert set(records) == {"fold_a", "fold_b", "fold_c"}
    assert all(record["selected_control_only_alpha"] in {1.0, 10.0} for record in records.values())


def test_wave_gate_requires_every_preregistered_check() -> None:
    truth = _six_lead_frame(cases=9)
    blind = truth[["fold", "anchor_id", "station", "lead_h"]].copy()
    blind["control_prediction"] = 2.2
    blind["treatment_prediction"] = 2.1
    short = blind["lead_h"].isin([3, 6, 9, 12])
    blind.loc[short, "control_prediction"] = 2.0
    blind.loc[short, "treatment_prediction"] = 2.0
    gate = {
        "pooled_six_lead_delta_rmse_m_max": -0.001,
        "paired_case_bootstrap_ci90_upper_strictly_below_m": 0.0,
        "minimum_improved_folds": 2,
        "minimum_improved_stations": 2,
        "lead_18_delta_rmse_m_max": 0.0,
        "lead_24_delta_rmse_m_max": 0.0,
        "worst_station_by_lead_delta_rmse_m_max": 0.003,
    }
    metrics = MODULE.evaluate_wave_gate(
        blind,
        truth,
        gate,
        {"replicates": 200, "seed": 7},
    )
    assert metrics["gate_pass"] is True
    assert all(metrics["gate_checks"].values())
    blind.loc[blind["station"].eq("S-ORS"), "treatment_prediction"] = 2.5
    failed = MODULE.evaluate_wave_gate(
        blind,
        truth,
        gate,
        {"replicates": 200, "seed": 7},
    )
    assert failed["gate_pass"] is False
    assert failed["gate_checks"]["worst_station_by_lead_within_limit"] is False


def test_blind_seal_rejects_truth_and_is_exclusive(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"x": 1}), encoding="utf-8")
    state = {
        "config_path": config_path,
        "input_hashes": {"historical": "a" * 64},
    }
    blind = pd.DataFrame(
        {
            "fold": ["a"],
            "anchor_id": [1],
            "station": ["G-ORS"],
            "lead_h": [18],
            "control_prediction": [2.0],
            "treatment_prediction": [1.9],
        }
    )
    prediction_path = tmp_path / "blind.parquet"
    seal_path = tmp_path / "seal.json"
    sealed = MODULE.write_prediction_seal(
        "synthetic_before_metric",
        prediction_path,
        seal_path,
        blind,
        state,
        "b" * 64,
    )
    assert sealed["prediction_sha256"] == MODULE.sha256_file(prediction_path)
    with pytest.raises(FileExistsError):
        MODULE.write_prediction_seal(
            "synthetic_before_metric",
            prediction_path,
            tmp_path / "second_seal.json",
            blind,
            state,
            "b" * 64,
        )
    with pytest.raises(MODULE.ContractError, match="truth"):
        MODULE.write_prediction_seal(
            "synthetic_before_metric",
            tmp_path / "truth.parquet",
            tmp_path / "truth_seal.json",
            blind.assign(target_hs=2.0),
            state,
            "b" * 64,
        )
