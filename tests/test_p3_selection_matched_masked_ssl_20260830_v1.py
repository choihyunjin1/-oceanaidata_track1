from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p3_wave.features import summarize_context
from p3_wave.loss_router import OBSERVED_FEATURES
from p3_wave.selection_matched_masked_ssl_20260830_v1 import (
    CONTEXT_ROWS,
    LEADS,
    RAW_COLUMNS,
    STATIONS,
    apply_paired_prequential_reference,
    comparison_metrics,
    evaluate_promotion_gate,
    extract_history_sequences,
    fit_candidate_fold,
    fit_sequence_transform,
    paired_case_bootstrap,
    transform_sequences,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiments/p3_selection_matched_masked_ssl_20260830_v1.json"
RUNNER_PATH = ROOT / "scripts/run_p3_selection_matched_masked_ssl_20260830_v1.py"
SPEC = importlib.util.spec_from_file_location("p3_selection_matched_masked_ssl_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _raw_sequences(cases: int, *, offset: float = 0.0) -> np.ndarray:
    step = np.arange(CONTEXT_ROWS, dtype=np.float32)
    result = np.empty((cases, CONTEXT_ROWS, len(RAW_COLUMNS)), dtype=np.float32)
    for case in range(cases):
        phase = step / 20.0 + case * 0.1 + offset
        result[case, :, 0] = 1.2 + 0.002 * step + 0.03 * np.sin(phase)
        result[case, :, 1] = 7.0 + 0.2 * np.cos(phase)
        result[case, :, 2] = 1.8 + 0.003 * step
        result[case, :, 3] = np.mod(90.0 + step, 360.0)
        result[case, :, 4] = 5.0 + 0.1 * np.sin(phase)
        result[case, :, 5] = 7.0 + 0.1 * np.cos(phase)
        result[case, :, 6] = np.mod(120.0 + step, 360.0)
        result[case, :, 7] = 18.0 + 0.1 * np.sin(phase)
        result[case, :, 8] = 70.0 + np.cos(phase)
        result[case, :, 9] = 1010.0 + np.sin(phase)
    result[:, 1::2, :4] = np.nan
    return result


def _anchors(cases: int, *, first_id: int, fold: str = "fold") -> pd.DataFrame:
    current = 1.65 + 0.01 * np.arange(cases)
    frame = pd.DataFrame(
        {
            "anchor_id": np.arange(first_id, first_id + cases, dtype=np.int64),
            "station": [STATIONS[index % len(STATIONS)] for index in range(cases)],
            "anchor_time": pd.date_range("2024-01-01", periods=cases, freq="96h", tz="UTC"),
            "grid_position": CONTEXT_ROWS - 1,
            "current_hs": current,
            "hs_minus_12h": current - 0.3,
            "rise_12h": 0.3,
            "fold": fold,
        }
    )
    for lead_index, lead in enumerate(LEADS):
        frame[f"target_{lead}"] = current + 0.04 * (lead_index + 1)
    return frame


def test_preregistration_freezes_distinct_recipe_and_closed_lanes() -> None:
    config = RUNNER.load_config(CONFIG_PATH)
    assert tuple(config["cohort_contract"]["official_leads_hours"]) == LEADS
    assert config["cohort_contract"]["history_rows_including_anchor"] == CONTEXT_ROWS
    assert config["representation"]["reused_module"] == "src/p1_qc/models_ssl.py"
    assert config["fit_and_runtime_budget"]["total_fit_calls"] == 8
    assert config["fit_and_runtime_budget"]["catboost_fits"] == 0
    assert config["closed_family_boundary"]["hierarchical_residual_basis_dense72"][
        "status"
    ] == "CLOSED_EXACT_45_FIT_FAMILY"
    assert config["closed_family_boundary"]["generic_nhits_reopened"] is False
    assert config["data_boundary"]["allowed_source_basenames"] == [
        "README.md",
        "train_wave.csv",
        "train_atmos.csv",
    ]
    changed = copy.deepcopy(config)
    changed["cohort_contract"]["official_leads_hours"] = [12, 24, 36, 48, 60, 72]
    with pytest.raises(RUNNER.ContractError, match="official leads"):
        RUNNER.validate_config(changed)


def test_sealed_dependencies_and_prior_aggregate_evidence_validate() -> None:
    config = RUNNER.load_config(CONFIG_PATH)
    snapshot = RUNNER._verify_dependencies(config)  # noqa: SLF001
    evidence = RUNNER._validate_prior_evidence(config)  # noqa: SLF001
    assert snapshot["stage0_receipt"] == config["stage0_authorization"]["receipt_sha256"]
    assert evidence["stage0_validation_cases"] == 157
    assert evidence["closed_dense72_fit_count"] == 45
    assert evidence["closed_dense72_full_delta_m"] == pytest.approx(0.067295)
    assert evidence["champion_original_surface_rmse_m"] == pytest.approx(0.7791048399763751)


def test_history_transform_preserves_rows_masks_and_extreme_values() -> None:
    raw = _raw_sequences(4)
    raw[0, 100, 0] = 8.0
    raw[1, 120, 0] = -2.0
    original = raw.copy()
    transform = fit_sequence_transform(raw[:3], clip_abs=50.0)
    encoded = transform_sequences(raw, transform)
    assert encoded.shape == (4, CONTEXT_ROWS, 24)
    assert np.isfinite(encoded).all()
    assert np.array_equal(raw, original, equal_nan=True)
    assert encoded[0, 100, 12] == 1.0
    assert encoded[0, 101, 12] == 0.0
    assert len(encoded) == len(raw)


def test_extract_history_sequences_keeps_anchor_order_and_does_not_mutate() -> None:
    rows = CONTEXT_ROWS + 12
    pieces: list[pd.DataFrame] = []
    anchors: list[dict[str, object]] = []
    for number, station in enumerate(STATIONS):
        time = pd.date_range("2024-01-01", periods=rows, freq="10min", tz="UTC")
        raw = _raw_sequences(1, offset=float(number))[0]
        extension = np.repeat(raw[-1:, :], rows - CONTEXT_ROWS, axis=0)
        values = np.concatenate([raw, extension], axis=0)
        part = pd.DataFrame(values, columns=RAW_COLUMNS)
        part.insert(0, "time", time)
        part.insert(0, "station", station)
        pieces.append(part)
        anchors.append(
            {"anchor_id": 50 + number, "station": station, "grid_position": CONTEXT_ROWS - 1}
        )
    grid = pd.concat(pieces, ignore_index=True)
    anchor_frame = pd.DataFrame(anchors).iloc[[2, 0, 1]].reset_index(drop=True)
    grid_before = grid.copy(deep=True)
    anchor_before = anchor_frame.copy(deep=True)
    result = extract_history_sequences(grid, anchor_frame)
    assert result.shape == (3, CONTEXT_ROWS, len(RAW_COLUMNS))
    assert result[0, 0, 7] != result[1, 0, 7]
    pd.testing.assert_frame_equal(grid, grid_before)
    pd.testing.assert_frame_equal(anchor_frame, anchor_before)


def test_tiny_cpu_masked_ssl_and_frozen_huber_integration() -> None:
    config = RUNNER.load_config(CONFIG_PATH)
    representation = copy.deepcopy(config["representation"])
    representation["model"]["channels"] = [4]
    representation["model"]["aggregated_embedding_dimension"] = 12
    representation["masked_training"].update(
        {
            "batch_size": 4,
            "maximum_epochs": 1,
            "patience": 1,
            "use_bfloat16": False,
        }
    )
    train_anchor = _anchors(12, first_id=100)
    valid_anchor = _anchors(4, first_id=500)
    fitted = fit_candidate_fold(
        _raw_sequences(12),
        _raw_sequences(4, offset=0.4),
        train_anchor,
        valid_anchor,
        fold="synthetic_fold",
        representation_config=representation,
        head_config=config["robust_residual_head"],
        seed=17,
        device="cpu",
    )
    assert len(fitted.frame) == 4 * len(LEADS)
    assert fitted.receipt["ssl"]["fit_count"] == 1
    assert fitted.receipt["huber"]["fit_count"] == 1
    assert fitted.receipt["ssl"]["outer_validation_cases_exposed_to_ssl_fit_or_early_stop"] == 0
    assert fitted.receipt["rows_deleted"] == 0
    assert np.isfinite(fitted.frame["candidate_prediction"]).all()


def _synthetic_reference_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_order = ("2024_h2_storm", "winter_transition", "2025_h1")
    anchors = pd.concat(
        [
            _anchors(3, first_id=100 * (number + 1), fold=fold)
            for number, fold in enumerate(fold_order)
        ],
        ignore_index=True,
    )
    feature_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    for anchor in anchors.itertuples(index=False):
        context = pd.DataFrame(_raw_sequences(1)[0], columns=RAW_COLUMNS)
        summary = summarize_context(context)
        feature_rows.append(
            {"anchor_id": int(anchor.anchor_id), "station": str(anchor.station), **summary}
        )
        for lead in LEADS:
            target = float(getattr(anchor, f"target_{lead}"))
            component_rows.append(
                {
                    "fold": str(anchor.fold),
                    "anchor_id": int(anchor.anchor_id),
                    "station": str(anchor.station),
                    "lead_h": lead,
                    "current_hs": float(anchor.current_hs),
                    "target_hs": target,
                    "single_prediction": target + 0.05,
                    "multi_prediction": target - 0.03,
                    "persistence": float(anchor.current_hs),
                }
            )
    features = pd.DataFrame(feature_rows)
    assert set(OBSERVED_FEATURES).issubset(features.columns)
    return anchors, features, pd.DataFrame(component_rows)


def test_fixed_prequential_reference_has_zero_catboost_fits_and_two_router_fits() -> None:
    config = RUNNER.load_config(CONFIG_PATH)
    anchors, features, components = _synthetic_reference_inputs()
    reference, receipt = apply_paired_prequential_reference(
        components,
        features,
        anchors,
        fold_order=("2024_h2_storm", "winter_transition", "2025_h1"),
        reference_config=config["paired_incumbent_reference"],
    )
    assert len(reference) == len(anchors) * len(LEADS)
    assert receipt["catboost_fit_count"] == 0
    assert receipt["fixed_router_fit_count"] == 2
    assert receipt["router_receipts"][0]["past_fit_rows"] == 0
    assert all(item["current_fold_target_used_for_router"] is False for item in receipt["router_receipts"])


def test_metrics_bootstrap_and_gate_are_case_paired() -> None:
    config = RUNNER.load_config(CONFIG_PATH)
    rows: list[dict[str, object]] = []
    for fold_number, fold in enumerate(("a", "b", "c")):
        for case in range(3):
            for lead in LEADS:
                truth = 2.0 + 0.01 * lead
                rows.append(
                    {
                        "fold": fold,
                        "anchor_id": fold_number * 10 + case,
                        "station": STATIONS[case],
                        "lead_h": lead,
                        "target_hs": truth,
                        "candidate_prediction": truth + 0.02,
                        "incumbent_prediction": truth + 0.08,
                        "persistence": truth + 0.12,
                    }
                )
    frame = pd.DataFrame(rows)
    metrics = comparison_metrics(frame)
    bootstrap = paired_case_bootstrap(frame, replicates=50, seed=3)
    gate = evaluate_promotion_gate(
        metrics,
        bootstrap,
        gate_config=config["evaluation"]["promotion_gate"],
        integrity_checks={"synthetic": True},
    )
    assert metrics["overall"]["delta_candidate_minus_incumbent_m"] < -0.01
    assert bootstrap["cases"] == 9
    assert gate["passed"] is True


def test_runner_has_no_broad_loader_csv_writer_or_git_call_and_exclusive_json(tmp_path: Path) -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "load_p3_data(" not in source
    assert "pd.read_csv(" not in source
    assert "to_csv(" not in source
    assert "subprocess" not in source
    target = tmp_path / "receipt.json"
    digest = RUNNER._write_exclusive_json(target, {"ok": True})  # noqa: SLF001
    assert digest == RUNNER.sha256_file(target)
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}
    with pytest.raises(FileExistsError):
        RUNNER._write_exclusive_json(target, {"ok": False})  # noqa: SLF001
