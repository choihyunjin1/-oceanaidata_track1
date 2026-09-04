from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_p1_iors_external_profile_conformal_v2 import (
    apply_v2_gate,
    conformal_interval_metrics,
    cqr_widening,
)

V1_CONFIG = Path("configs/experiments/p1_iors_external_loo_precheck_v1.json")
V2_CONFIG = Path("configs/experiments/p1_iors_external_profile_conformal_v2.json")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v2_preregisters_disjoint_fit_calibration_test_and_same_model() -> None:
    v1 = _read(V1_CONFIG)
    v2 = _read(V2_CONFIG)
    split = v2["split"]

    assert split["fit_years"] == list(range(2014, 2022))
    assert split["calibration_year"] == 2022
    assert split["test_year"] == 2023
    assert not set(split["fit_years"]) & {split["calibration_year"], split["test_year"]}
    assert v2["model"] == {
        **v1["model"],
        "early_stopping": False,
        "hyperparameter_search": False,
    }
    assert v2["conformal"]["grid_search"] is False
    assert v2["decision_scope"]["competition_labels_used"] is False
    assert v2["decision_scope"]["competition_oof_used"] is False
    assert v2["decision_scope"]["competition_outer_validation_used"] is False


def test_v2_pins_current_external_catalog() -> None:
    contract = _read(V2_CONFIG)
    catalog = Path(contract["external_catalog"])

    assert _sha256(catalog) == contract["external_catalog_sha256"]
    text = catalog.read_text(encoding="utf-8")
    source = text.split('source_id = "i_ors_ctd_2014_2023"', maxsplit=1)[1].split(
        "[[source]]", maxsplit=1
    )[0]
    assert "value_accessed = true" in source
    assert 'priority = "P1_high_P2_fallback"' in source


def test_cqr_uses_exact_finite_sample_order_statistic_without_interpolation() -> None:
    y = np.arange(5, dtype=float)
    q10 = y + np.asarray([0.0, 0.1, 0.2, 0.3, 0.4])
    q90 = y + 1.0

    value = cqr_widening(y, q10, q90, alpha=0.2)

    assert value["one_based_rank"] == 5
    assert value["correction"] == pytest.approx(0.4)
    assert value["scope"] == "single global non-negative widening scalar"
    assert value["corrected_coverage"] >= value["raw_coverage"]


def test_cqr_never_narrows_an_already_wide_interval() -> None:
    y = np.asarray([1.0, 2.0, 3.0, 4.0])
    q10 = y - 1.0
    q90 = y + 1.0

    value = cqr_widening(y, q10, q90, alpha=0.25)

    assert value["correction"] == 0.0
    assert value["corrected_mean_width"] == value["raw_mean_width"]


def test_conformal_interval_metrics_apply_one_scalar_to_every_layer() -> None:
    y = np.asarray([0.0, 1.0, 2.0, 3.0])
    q10 = np.asarray([0.2, 1.2, 2.2, 3.2])
    q90 = np.asarray([0.4, 1.4, 2.4, 3.4])
    layer = np.asarray([1, 1, 2, 2])

    raw = conformal_interval_metrics(y, q10, q90, correction=0.0, layer=layer)
    widened = conformal_interval_metrics(y, q10, q90, correction=0.2, layer=layer)

    assert raw["coverage"] == 0.0
    assert widened["coverage"] == 1.0
    assert widened["mean_width"] == pytest.approx(raw["mean_width"] + 0.4)
    assert set(widened["per_layer"]) == {"1", "2"}


def test_v2_gate_requires_target_coverage_and_five_layers() -> None:
    per_layer = {
        str(layer): {
            "candidate": {"rmse": 0.9},
            "baseline": {"rmse": 1.0},
        }
        for layer in range(1, 7)
    }
    metrics = {
        "rows": 60000,
        "rmse_relative_improvement": 0.1,
        "mae_relative_improvement": 0.1,
        "per_layer": per_layer,
    }
    calibration = {"calibration_rows": 60000}
    conformal_test = {"coverage": 0.8}
    gate = _read(V2_CONFIG)["stop_gate"]

    passed = apply_v2_gate(
        metrics,
        calibration,
        conformal_test,
        gate,
        source_integrity_verified=True,
    )
    conformal_test["coverage"] = 0.74
    failed = apply_v2_gate(
        metrics,
        calibration,
        conformal_test,
        gate,
        source_integrity_verified=True,
    )

    assert passed["decision"] == "GO_TO_ISOLATED_P1_OOF"
    assert failed["decision"] == "NO_GO_EXTERNAL_PROFILE"
    assert failed["checks"]["conformal_coverage_min"] is False


def test_v1_result_and_receipt_sha_are_unchanged() -> None:
    provenance = _read(V2_CONFIG)["cross_experiment_provenance"]

    assert _sha256(Path(provenance["v1_result"])) == provenance["v1_result_sha256"]
    assert _sha256(Path(provenance["v1_receipt"])) == provenance["v1_receipt_sha256"]
    assert provenance["global_2023_first_look"] is False


def test_v2_runner_has_no_p1_label_oof_or_outer_api() -> None:
    source = Path("scripts/run_p1_iors_external_profile_conformal_v2.py").read_text(
        encoding="utf-8"
    )

    assert "from p1_qc" not in source
    assert "P1_DATA_DIR" not in source
    assert "train.csv" not in source
    assert "load_train_test" not in source
    assert "run_cross_validation" not in source
    assert 'hyperparameter_search": true' not in source.lower()
