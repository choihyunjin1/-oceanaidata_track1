from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p3_wave.kma_calibrated_longlead_blend import (
    ACTIVE_LEADS,
    ALPHA_GRID,
    INNER_COLUMNS,
    LEADS,
    NO_OP_LEADS,
    OUTER_COLUMNS,
    KMALongLeadError,
    RidgeAffineCalibrator,
    add_calibrated_source,
    apply_fixed_control_shrink,
    apply_ridge_affine,
    blend_long_leads,
    evaluate_inner_gate,
    evaluate_outer_promotion,
    fit_ridge_affine,
    fit_ridge_pair,
    load_preregistration,
    select_fold_alpha,
    validate_inner_predictions,
    validate_outer_blind,
    validate_preregistration,
)
from scripts.run_p3_kma_calibrated_longlead_blend_v2 import (
    AUTHORIZATION_TOKEN,
    CANONICAL_CONFIG,
    TargetVault,
    _actual,
    _assemble_inner_rows,
    _canonical_config,
    _coefficient_payload,
    _expand_source_rows,
    _implementation_hashes,
    _preouter_no_go_result,
    _read_incumbent_label_free,
    _synthetic_contract,
    _targets_to_long,
    _verify_v1_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiments/p3_kma_calibrated_longlead_blend_v2.json"


def _ridge_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    anchor = 0
    intercept = {"G-ORS": -0.1, "I-ORS": 0.0, "S-ORS": 0.2}
    for lead in ACTIVE_LEADS:
        for station in ("G-ORS", "I-ORS", "S-ORS"):
            for source_residual in (-0.5, 0.2, 0.8):
                current = 2.0
                rows.append(
                    {
                        "anchor_id": anchor,
                        "station": station,
                        "lead_h": lead,
                        "current_hs": current,
                        "source_prediction": current + source_residual,
                        "target_hs": current + 1.5 * source_residual + intercept[station],
                    }
                )
                anchor += 1
    return pd.DataFrame(rows)


def _inner_fold(
    fold: str,
    *,
    control_error: float,
    source_error: float,
    selected_alpha: float | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for anchor_id in range(3):
        for lead in LEADS:
            target = 2.0 + anchor_id / 10
            control = target + control_error
            source = target + source_error
            rows.append(
                {
                    "fold": fold,
                    "anchor_id": anchor_id,
                    "station": ("G-ORS", "I-ORS", "S-ORS")[anchor_id],
                    "lead_h": lead,
                    "current_hs": 2.0,
                    "target_hs": target,
                    "source_prediction": source,
                    "calibrated_source": source,
                    "control_single_prediction": control,
                    "control_final": control,
                    "selected_alpha": 0.0 if selected_alpha is None else selected_alpha,
                    "candidate_final": control,
                }
            )
    return pd.DataFrame(rows)


def test_preregistration_freezes_single_calibrated_longlead_hypothesis() -> None:
    config = load_preregistration(CONFIG_PATH)
    assert config["calibrator"]["ridge_alpha"] == 10.0
    assert config["calibrator"]["hyperparameter_grid_size"] == 0
    assert config["calibrator"]["active_leads"] == [18, 24]
    assert config["blend"]["alpha_grid"] == list(ALPHA_GRID)
    assert config["blend"]["fold_alpha_zero_is_an_honest_no_op_and_is_allowed"] is True
    assert config["blend"]["deployment_alpha"] == "median_of_three_inner_selected_fold_alphas"
    assert config["inner_control_proxy"]["model_or_iteration_search"] == 0
    assert config["execution"]["actual_authorized"] is False


def test_preregistration_rejects_alpha_zero_prohibition_and_parameter_drift() -> None:
    config = load_preregistration(CONFIG_PATH)
    changed = copy.deepcopy(config)
    changed["blend"]["fold_alpha_zero_is_an_honest_no_op_and_is_allowed"] = False
    with pytest.raises(KMALongLeadError, match="alpha zero"):
        validate_preregistration(changed)
    changed = copy.deepcopy(config)
    changed["calibrator"]["ridge_alpha"] = 1.0
    with pytest.raises(KMALongLeadError, match="ridge_alpha"):
        validate_preregistration(changed)


def test_ridge_pair_is_fixed_deterministic_and_serializable() -> None:
    frame = _ridge_rows()
    first = fit_ridge_pair(frame)
    second = fit_ridge_pair(frame)
    assert set(first) == set(ACTIVE_LEADS)
    assert first == second
    for lead, calibrator in first.items():
        assert calibrator.lead_h == lead
        assert calibrator.ridge_alpha == 10.0
        assert calibrator.fit_intercept is False
        assert calibrator.solver == "cholesky"
        assert len(calibrator.coefficients) == 4
        assert RidgeAffineCalibrator.from_dict(calibrator.to_dict()) == calibrator
        json.dumps(calibrator.to_dict())


def test_ridge_rejects_wrong_lead_unknown_station_and_duplicate_cases() -> None:
    frame = _ridge_rows()
    with pytest.raises(KMALongLeadError, match="18h or 24h"):
        fit_ridge_affine(frame, lead_h=12)
    bad_station = frame.copy()
    bad_station.loc[bad_station.index[0], "station"] = "X-ORS"
    with pytest.raises(KMALongLeadError, match="unknown station"):
        fit_ridge_affine(bad_station, lead_h=18)
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(KMALongLeadError, match="duplicate"):
        fit_ridge_affine(duplicate, lead_h=18)


def test_apply_calibrator_and_pair_cover_only_long_leads() -> None:
    frame = _ridge_rows()
    calibrators = fit_ridge_pair(frame)
    applied = add_calibrated_source(frame, calibrators)
    assert np.isfinite(applied["calibrated_source"]).all()
    subset = frame.loc[frame["lead_h"].eq(18)]
    direct = apply_ridge_affine(subset, calibrators[18])
    assert np.allclose(
        direct,
        applied.loc[applied["lead_h"].eq(18), "calibrated_source"].to_numpy(),
    )
    with pytest.raises(KMALongLeadError, match="both fixed"):
        add_calibrated_source(frame, {18: calibrators[18]})


def test_fixed_control_shrink_and_blend_preserve_short_leads_exactly() -> None:
    leads = np.asarray(LEADS)
    single = np.asarray([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    current = np.asarray([2.0] * 6)
    shrunk = apply_fixed_control_shrink(single, current, leads)
    assert np.array_equal(shrunk[:3], single[:3])
    assert shrunk[3] == pytest.approx(0.8 * 1.3 + 0.2 * 2.0)
    source = np.asarray([9.0, 9.0, 9.0, 9.0, 1.0, 1.0])
    candidate = blend_long_leads(shrunk, source, leads, alpha=0.2)
    assert np.array_equal(candidate[:4], shrunk[:4])
    no_op = blend_long_leads(shrunk, source, leads, alpha=0.0)
    assert np.array_equal(no_op, shrunk)
    with pytest.raises(KMALongLeadError, match="frozen grid"):
        blend_long_leads(shrunk, source, leads, alpha=0.25)


def test_fold_alpha_selection_uses_full_six_leads_and_smallest_tie() -> None:
    inner = _inner_fold("a", control_error=0.4, source_error=0.0)
    alpha, selected, scores = select_fold_alpha(inner)
    assert alpha == 0.4
    assert len(scores) == len(ALPHA_GRID)
    assert np.array_equal(
        selected.loc[selected["lead_h"].isin(NO_OP_LEADS), "candidate_final"].to_numpy(),
        selected.loc[selected["lead_h"].isin(NO_OP_LEADS), "control_final"].to_numpy(),
    )
    tie = inner.copy()
    tie["calibrated_source"] = tie["control_final"]
    alpha, selected, _ = select_fold_alpha(tie)
    assert alpha == 0.0
    assert np.array_equal(selected["candidate_final"], selected["control_final"])


def test_inner_gate_allows_one_alpha_zero_fold_and_uses_median_for_deployment() -> None:
    blocks: list[pd.DataFrame] = []
    for fold, alpha in (("a", 0.1), ("b", 0.2), ("c", 0.0)):
        block = _inner_fold(fold, control_error=0.4, source_error=0.0, selected_alpha=alpha)
        block["candidate_final"] = blend_long_leads(
            block["control_final"], block["calibrated_source"], block["lead_h"], alpha=alpha
        )
        blocks.append(block)
    result = evaluate_inner_gate(pd.concat(blocks, ignore_index=True))
    assert result["pass"] is True
    assert result["strictly_improved_folds"] == 2
    assert result["alpha_zero_folds"] == ["c"]
    assert result["deployment_alpha_median"] == pytest.approx(0.1)


def test_inner_gate_fails_when_only_one_fold_strictly_improves() -> None:
    blocks: list[pd.DataFrame] = []
    for fold, alpha in (("a", 0.2), ("b", 0.0), ("c", 0.0)):
        block = _inner_fold(fold, control_error=0.4, source_error=0.0, selected_alpha=alpha)
        block["candidate_final"] = blend_long_leads(
            block["control_final"], block["calibrated_source"], block["lead_h"], alpha=alpha
        )
        blocks.append(block)
    result = evaluate_inner_gate(pd.concat(blocks, ignore_index=True))
    assert result["pass"] is False
    assert result["strictly_improved_folds"] == 1


def test_inner_prediction_validator_requires_schema_truth_and_noop() -> None:
    frame = _inner_fold("a", control_error=0.4, source_error=0.0, selected_alpha=0.2)
    frame["candidate_final"] = blend_long_leads(
        frame["control_final"], frame["calibrated_source"], frame["lead_h"], alpha=0.2
    )
    frame = frame.loc[:, list(INNER_COLUMNS)]
    assert validate_inner_predictions(frame)["rows"] == len(frame)
    bad = frame.copy()
    bad.loc[bad["lead_h"].eq(3), "candidate_final"] += 0.1
    with pytest.raises(KMALongLeadError, match="does not reconstruct"):
        validate_inner_predictions(bad)


def test_outer_blind_validator_forbids_truth_and_requires_exact_short_noop() -> None:
    rows: list[dict[str, object]] = []
    for lead in LEADS:
        rows.append(
            {
                "fold": "a",
                "anchor_id": 1,
                "station": "G-ORS",
                "lead_h": lead,
                "current_hs": 2.0,
                "incumbent_final": 2.2,
                "source_prediction": 2.1,
                "calibrated_source": 2.0,
                "selected_alpha": 0.2,
                "candidate_final": 2.2 if lead in NO_OP_LEADS else 2.16,
            }
        )
    frame = pd.DataFrame(rows).loc[:, list(OUTER_COLUMNS)]
    assert validate_outer_blind(frame)["cases"] == 1
    with pytest.raises(KMALongLeadError, match="truth leaked"):
        validate_outer_blind(frame.assign(target_hs=2.0))


def test_source_row_expansion_and_inner_assembly_replay_actual_schema() -> None:
    anchors = pd.DataFrame(
        {
            "anchor_id": [1, 2],
            "station": ["G-ORS", "S-ORS"],
            "current_hs": [2.0, 2.5],
        }
    )
    source = pd.DataFrame({"anchor_id": [1, 2]})
    targets = pd.DataFrame({"anchor_id": [1, 2]})
    for lead in LEADS:
        source[f"kma_source_hs_pred_{lead}h"] = [2.0 + lead / 100, 2.5 + lead / 100]
        targets[f"target_{lead}"] = [2.1, 2.6]
    source_rows = _expand_source_rows(source, anchors, [1, 2], targets=targets)
    assert len(source_rows) == 12
    source_rows["calibrated_source"] = source_rows["source_prediction"]
    control = source_rows[["anchor_id", "station", "lead_h", "current_hs"]].copy()
    control["control_single_prediction"] = 2.2
    control["control_final"] = 2.2
    assembled = _assemble_inner_rows(fold_name="a", source_rows=source_rows, control_rows=control)
    alpha, selected, scores = select_fold_alpha(assembled)
    selected = selected.loc[:, list(INNER_COLUMNS)]
    assert alpha in ALPHA_GRID
    assert len(scores) == 5
    assert validate_inner_predictions(selected)["cases"] == 2


def test_coefficient_payload_is_complete_and_json_replayable() -> None:
    calibrators = fit_ridge_pair(_ridge_rows())
    payload = _coefficient_payload(calibrators)
    assert set(payload) == {"18", "24"}
    replayed = {
        int(lead): RidgeAffineCalibrator.from_dict(values) for lead, values in payload.items()
    }
    assert replayed == calibrators
    json.dumps(payload)


def test_incumbent_reader_requests_no_target_column(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_preregistration(CONFIG_PATH)
    original = pd.read_parquet
    requested: list[str] | None = None

    def recording_read(path: Path, *, columns: list[str]) -> pd.DataFrame:
        nonlocal requested
        requested = list(columns)
        return original(path, columns=columns)

    monkeypatch.setattr(pd, "read_parquet", recording_read)
    frame = _read_incumbent_label_free(config)
    assert requested == ["fold", "anchor_id", "station", "lead_h", "current_hs", "prediction"]
    assert "target_hs" not in frame
    assert len(frame) == 1092


def test_v1_evidence_and_new_implementation_surface_are_exactly_pinned() -> None:
    config = load_preregistration(CONFIG_PATH)
    evidence = _verify_v1_evidence(config)
    assert evidence["v1_outer_designated_scoring_open_count"] == 0
    assert evidence["raw_source_minus_persistence_rmse"] < 0.0
    hashes = _implementation_hashes()
    assert set(hashes) == {
        "configs/experiments/p3_kma_calibrated_longlead_blend_v2.json",
        "src/p3_wave/kma_calibrated_longlead_blend.py",
        "scripts/run_p3_kma_calibrated_longlead_blend_v2.py",
        "tests/test_p3_kma_calibrated_longlead_blend_v2.py",
        "src/p3_wave/kma_source_meta.py",
        "src/p3_wave/revin_patch.py",
        "src/p3_wave/models.py",
    }
    assert all(len(value) == 64 for value in hashes.values())


def test_target_long_expansion_aligns_only_blind_keys() -> None:
    targets = pd.DataFrame({"anchor_id": [1, 2]})
    for lead in LEADS:
        targets[f"target_{lead}"] = [1.0 + lead / 100, 2.0 + lead / 100]
    blind = pd.DataFrame(
        {
            "fold": np.repeat(["a", "b"], 6),
            "anchor_id": np.repeat([1, 2], 6),
            "station": np.repeat(["G-ORS", "S-ORS"], 6),
            "lead_h": list(LEADS) * 2,
        }
    )
    long = _targets_to_long(targets, blind)
    assert len(long) == 12
    assert set(long.columns) == {"fold", "anchor_id", "station", "lead_h", "target_hs"}


def test_outer_promotion_applies_only_exact_incumbent_comparison() -> None:
    config = load_preregistration(CONFIG_PATH)
    rows: list[dict[str, object]] = []
    for fold_index, fold in enumerate(("a", "b", "c")):
        for anchor_id in range(10):
            for lead in LEADS:
                target = 2.0
                incumbent = 2.2
                candidate = incumbent if lead in NO_OP_LEADS else 2.0
                rows.append(
                    {
                        "fold": fold,
                        "anchor_id": 100 * fold_index + anchor_id,
                        "station": ("G-ORS", "I-ORS", "S-ORS")[fold_index],
                        "lead_h": lead,
                        "current_hs": 2.0,
                        "incumbent_final": incumbent,
                        "source_prediction": 2.0,
                        "calibrated_source": 2.0,
                        "selected_alpha": 0.4,
                        "candidate_final": candidate,
                        "target_hs": target,
                    }
                )
    result = evaluate_outer_promotion(pd.DataFrame(rows), config)
    assert result["decision"] == "GO_TO_INTEGRATION"
    assert result["paired_case_bootstrap"]["ci90_upper"] < 0.0
    assert result["delta_by_lead"]["3"] == 0.0
    assert result["delta_by_lead"]["18"] < 0.0


def test_target_vault_enforces_fold_local_current_future_and_prior_scope(tmp_path: Path) -> None:
    targets = pd.DataFrame({"anchor_id": [1, 2, 3]})
    for lead in LEADS:
        targets[f"target_{lead}"] = [1.0, 2.0, 3.0]
    path = tmp_path / "targets.parquet"
    targets.to_parquet(path, index=False)
    vault = TargetVault(path)
    with pytest.raises(PermissionError, match="current-fold"):
        vault.read_training(
            [2],
            forbidden_current_validation_ids=[2],
            all_outer_validation_ids=[1, 2, 3],
            allowed_prior_validation_ids=[1],
            fold="b",
            purpose="test",
        )
    with pytest.raises(PermissionError, match="future-fold"):
        vault.read_training(
            [3],
            forbidden_current_validation_ids=[2],
            all_outer_validation_ids=[1, 2, 3],
            allowed_prior_validation_ids=[1],
            fold="b",
            purpose="test",
        )
    released = vault.read_training(
        [1],
        forbidden_current_validation_ids=[2],
        all_outer_validation_ids=[1, 2, 3],
        allowed_prior_validation_ids=[1],
        fold="b",
        purpose="prior_history",
    )
    assert released["anchor_id"].tolist() == [1]
    assert vault.access_log[-1]["permitted_prior_validation_history_rows"] == 1


def test_target_vault_requires_blind_seal_before_designated_scoring(tmp_path: Path) -> None:
    targets = pd.DataFrame({"anchor_id": [1]})
    for lead in LEADS:
        targets[f"target_{lead}"] = [1.0]
    path = tmp_path / "targets.parquet"
    targets.to_parquet(path, index=False)
    manifest = tmp_path / "manifest.json"
    receipt = tmp_path / "receipt.json"
    manifest.write_text(json.dumps({"sealed": False}), encoding="utf-8")
    receipt.write_text(json.dumps({}), encoding="utf-8")
    vault = TargetVault(path)
    with pytest.raises(PermissionError, match="sealed"):
        vault.open_designated_scoring_once([1], blind_manifest=manifest, exposure_receipt=receipt)
    assert vault.designated_scoring_open_count == 0
    manifest.write_text(
        json.dumps({"sealed": True, "designated_outer_scoring_read_performed": False}),
        encoding="utf-8",
    )
    receipt.write_text(
        json.dumps(
            {
                "blind_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "fsync_completed_before_designated_scoring": True,
            }
        ),
        encoding="utf-8",
    )
    assert (
        len(
            vault.open_designated_scoring_once(
                [1], blind_manifest=manifest, exposure_receipt=receipt
            )
        )
        == 1
    )
    with pytest.raises(PermissionError, match="only once"):
        vault.open_designated_scoring_once([1], blind_manifest=manifest, exposure_receipt=receipt)


def test_canonical_config_and_in_memory_actual_override_are_fail_closed(tmp_path: Path) -> None:
    copied = tmp_path / "copied.json"
    copied.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(PermissionError, match="override"):
        _canonical_config(copied)
    assert _canonical_config(CANONICAL_CONFIG) == CANONICAL_CONFIG
    config = load_preregistration(CONFIG_PATH)
    mutated = copy.deepcopy(config)
    mutated["calibrator"]["ridge_alpha"] = 9.0
    with pytest.raises(PermissionError, match="in-memory config"):
        _actual(
            mutated,
            p3_data_dir=tmp_path,
            authorization_token=AUTHORIZATION_TOKEN,
            started=0.0,
        )


def test_preouter_stop_receipt_reports_consumed_attempt_lock(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt.lock"
    attempt.write_text('{"experiment_id":"v2"}', encoding="utf-8")
    manifest = tmp_path / "inner_manifest.json"
    manifest.write_text('{"sealed":true}', encoding="utf-8")
    result = _preouter_no_go_result(
        inner_gate={"pass": False}, inner_manifest=manifest, attempt_lock=attempt
    )
    assert result["one_shot_locks_created"] == 1
    assert result["global_attempt_lock_created"] is True
    assert result["outer_truth_locks_created"] == 0
    assert result["rerun_prohibited"] is True
    assert result["designated_outer_scoring_open_count"] == 0
    with pytest.raises(PermissionError, match="attempt lock"):
        _preouter_no_go_result(
            inner_gate={"pass": False},
            inner_manifest=manifest,
            attempt_lock=tmp_path / "missing.lock",
        )


def test_dry_synthetic_contract_fits_no_model_and_reads_no_target() -> None:
    result = _synthetic_contract()
    assert result["ridge_fit_count"] == 0
    assert result["catboost_fit_count"] == 0
    assert result["target_value_read_count"] == 0
    assert result["short_lead_no_op_max_abs_difference"] == 0.0
