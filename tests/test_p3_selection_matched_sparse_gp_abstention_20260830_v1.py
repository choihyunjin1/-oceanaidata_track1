from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p3_wave.p3_selection_matched_sparse_gp_abstention_20260830_v1 import (
    LEADS,
    classify_evidence,
    contiguous_anchor_day_block_bootstrap,
    fit_bayesian_rff_multi_lead,
    predict_with_abstention,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p3_selection_matched_sparse_gp_abstention_20260830_v1"


def _load_script(relative: str, module_name: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _anchors(rows: int, *, start_id: int, fold: str) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    current = 1.7 + 0.08 * np.sin(index / 5.0)
    frame = pd.DataFrame(
        {
            "anchor_id": np.arange(start_id, start_id + rows, dtype=np.int64),
            "station": np.asarray([("G-ORS", "I-ORS", "S-ORS")[i % 3] for i in range(rows)]),
            "anchor_time": pd.date_range("2024-01-01", periods=rows, freq="3D", tz="UTC"),
            "current_hs": current,
            "fold": fold,
        }
    )
    for lead_index, lead in enumerate(LEADS):
        frame[f"target_{lead}"] = current + 0.03 * (lead_index + 1) + 0.02 * np.cos(index / 4.0)
    return frame


def _features(anchors: pd.DataFrame) -> pd.DataFrame:
    index = np.arange(len(anchors), dtype=float)
    return pd.DataFrame(
        {
            "anchor_id": anchors["anchor_id"].to_numpy(),
            "station": anchors["station"].to_numpy(),
            "x0": np.sin(index / 4.0),
            "x1": np.cos(index / 7.0),
            "x2": index / max(len(index), 1),
        }
    )


def _incumbent(anchors: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "anchor_id": np.repeat(anchors["anchor_id"].to_numpy(), len(LEADS)),
            "lead_h": np.tile(np.asarray(LEADS), len(anchors)),
            "incumbent_prediction": np.repeat(
                anchors["current_hs"].to_numpy(), len(LEADS)
            ),
        }
    )


def test_preregistration_uses_governing_metric_and_no_legacy_veto() -> None:
    config = json.loads(
        (ROOT / f"configs/experiments/{EXPERIMENT_ID}.json").read_text(encoding="utf-8")
    )
    assert config["governing_policy"]["path"].endswith(
        "metric_aligned_gate_recalibration_20260830_v1.json"
    )
    assert config["candidate_recipe"]["candidate_total_fit_count"] == 3
    assert config["candidate_recipe"]["hyperparameter_search_count"] == 0
    assert config["candidate_recipe"]["outlier_hard_deletion_count"] == 0
    assert config["decision_policy"]["primary_metric"] == "pooled all-row six-lead Hs RMSE_m"
    assert config["decision_policy"]["legacy_minimum_0_01m_applied"] is False
    assert config["decision_policy"]["legacy_two_of_three_windows_applied"] is False
    assert config["decision_policy"]["legacy_worst_lead_0_02m_cap_applied"] is False
    assert config["data_boundary"]["allowed_source_basenames"] == [
        "README.md",
        "train_wave.csv",
        "train_atmos.csv",
    ]


def test_fixed_bayesian_rff_is_deterministic_one_fit_and_interval_abstains() -> None:
    train = _anchors(72, start_id=0, fold="train")
    valid = _anchors(12, start_id=1000, fold="synthetic_fold")
    train_features = _features(train)
    valid_features = _features(valid)
    kwargs = {
        "feature_columns": ("x0", "x1", "x2"),
        "seed": 20260830,
        "random_feature_count": 16,
        "ridge_precision": 25.0,
        "clip_abs": 8.0,
        "ood_quantile": 0.995,
        "minimum_radius": 1.0,
    }
    first = fit_bayesian_rff_multi_lead(train_features, train, **kwargs)
    second = fit_bayesian_rff_multi_lead(train_features, train, **kwargs)
    assert first.fit_receipt["fit_count"] == 1
    assert first.fit_receipt["hyperparameter_search_count"] == 0
    assert first.fit_receipt["rows_deleted"] == 0
    assert first.fit_receipt["coefficient_sha256"] == second.fit_receipt["coefficient_sha256"]
    incumbent = _incumbent(valid)
    prediction = predict_with_abstention(
        first,
        valid_features,
        valid,
        incumbent,
        interval_z=1.0e9,
        correction_cap_m=0.1,
    )
    paired = prediction.frame.merge(incumbent, on=["anchor_id", "lead_h"])
    assert prediction.receipt["active_correction_rows"] == 0
    assert prediction.receipt["exact_incumbent_rows"] == len(valid) * len(LEADS)
    assert np.array_equal(
        paired["candidate_prediction"].to_numpy(),
        paired["incumbent_prediction"].to_numpy(),
    )


def test_ood_cases_use_exact_incumbent_and_active_corrections_are_capped() -> None:
    train = _anchors(60, start_id=0, fold="train")
    valid = _anchors(6, start_id=2000, fold="synthetic_fold")
    train_features = _features(train)
    valid_features = _features(valid)
    valid_features.loc[0, ["x0", "x1", "x2"]] = [1.0e6, -1.0e6, 1.0e6]
    model = fit_bayesian_rff_multi_lead(
        train_features,
        train,
        feature_columns=("x0", "x1", "x2"),
        seed=7,
        random_feature_count=12,
        ridge_precision=25.0,
        clip_abs=8.0,
        ood_quantile=0.995,
        minimum_radius=1.0,
    )
    incumbent = _incumbent(valid)
    prediction = predict_with_abstention(
        model,
        valid_features,
        valid,
        incumbent,
        interval_z=0.0,
        correction_cap_m=0.1,
    )
    first_case = prediction.frame.loc[prediction.frame["anchor_id"].eq(2000)]
    assert first_case["ood_case"].all()
    assert np.array_equal(
        first_case["candidate_prediction"].to_numpy(),
        incumbent.loc[incumbent["anchor_id"].eq(2000), "incumbent_prediction"].to_numpy(),
    )
    assert prediction.frame["correction_applied_m"].abs().max() <= 0.1
    assert prediction.receipt["rows_deleted"] == 0


def test_contiguous_day_block_keeps_six_leads_and_is_deterministic() -> None:
    rows: list[dict[str, object]] = []
    anchor_id = 0
    for fold, start in (("fold_a", "2024-01-01"), ("fold_b", "2024-02-01")):
        for day in range(5):
            for station in ("G-ORS", "I-ORS"):
                for lead in LEADS:
                    rows.append(
                        {
                            "fold": fold,
                            "anchor_id": anchor_id,
                            "anchor_time": pd.Timestamp(start, tz="UTC")
                            + pd.Timedelta(days=day),
                            "station": station,
                            "lead_h": lead,
                            "target_hs": 2.0,
                            "candidate_prediction": 2.1,
                            "incumbent_prediction": 2.5,
                        }
                    )
                anchor_id += 1
    frame = pd.DataFrame(rows)
    first = contiguous_anchor_day_block_bootstrap(
        frame, replicates=200, seed=20260830, block_length_days=3
    )
    second = contiguous_anchor_day_block_bootstrap(
        frame, replicates=200, seed=20260830, block_length_days=3
    )
    assert first == second
    assert first["unit"].endswith("six_leads_intact")
    assert first["cases"] == 20
    assert first["rows"] == 120
    assert first["benefit_ci90_m"][0] > 0.0
    assert (
        classify_evidence(
            benefit_point=first["benefit_incumbent_minus_candidate_point_m"],
            benefit_ci90=first["benefit_ci90_m"],
            fatal_integrity_checks={"schema": True, "leakage": True},
        )
        == "HIGH_VALUE_CHALLENGER_RESEARCH_ONLY"
    )


def test_fatal_integrity_is_separate_from_performance_and_warning() -> None:
    assert (
        classify_evidence(
            benefit_point=0.2,
            benefit_ci90=(0.1, 0.3),
            fatal_integrity_checks={"schema": False},
        )
        == "QA_BLOCKED"
    )
    assert (
        classify_evidence(
            benefit_point=-0.1,
            benefit_ci90=(-0.2, -0.05),
            fatal_integrity_checks={"schema": True},
        )
        == "PRIMARY_HARM_RESEARCH_ONLY"
    )


def test_runner_contract_only_never_accepts_source_path(tmp_path: Path) -> None:
    runner = _load_script(
        f"scripts/run_{EXPERIMENT_ID}.py", "p3_sparse_gp_runner_test"
    )
    source = (ROOT / f"scripts/run_{EXPERIMENT_ID}.py").read_text(encoding="utf-8")
    assert "load_p3_data(" not in source
    assert "glob(" not in source
    assert "rglob(" not in source
    assert "minimum_0_01m_applied\"] = True" not in source
    payload = {"sealed": True}
    output = tmp_path / "receipt.json"
    digest = runner._write_exclusive_json(output, payload)
    assert digest == runner.sha256_file(output)
    with pytest.raises(FileExistsError):
        runner._write_exclusive_json(output, payload)
