from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p3_wave.kma_source_meta import (
    LEADS,
    META_COLUMNS,
    PAIR_KEYS,
    ROUTER_COLUMNS,
    KMASourceMetaError,
    append_meta_features,
    apply_source_median_imputer,
    canonicalize_kma_observations,
    compact_source_feature_columns,
    evaluate_inner_incremental_signal,
    evaluate_promotion,
    extract_target_common_history,
    fit_source_median_imputer,
    integrate_frozen_router,
    load_preregistration,
    read_frozen_outer_key_membership,
    read_frozen_router_components,
    resolve_domain_route,
    source_predictions_to_meta,
    summarize_common_history,
    validate_blind_prediction_frame,
    validate_preregistration,
)
from scripts.run_p3_kma_source_prediction_meta_v1 import (
    AUTHORIZATION_TOKEN,
    CANONICAL_CONFIG,
    TargetVault,
    _actual,
    _assemble_inner_gate_frame,
    _canonical_config,
    _inner_gate_no_go_result,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiments/p3_kma_source_prediction_meta_v1.json"


def _history() -> pd.DataFrame:
    step = np.arange(97, dtype=float)
    return pd.DataFrame(
        {
            "hs": 1.5 + step / 100.0,
            "hmax": 2.0 + step / 80.0,
            "wspd": 5.0 + step / 50.0,
            "gust": 7.0 + step / 50.0,
            "airt": 20.0 - step / 100.0,
            "relh": 65.0 + step / 100.0,
            "caph": 1015.0 - step / 50.0,
            "wvdir": np.mod(200.0 + step, 360.0),
            "wdir": np.mod(170.0 + step / 2.0, 360.0),
        }
    )


def test_preregistration_freezes_single_hypothesis_and_high_auc_inner_gate() -> None:
    config = load_preregistration(CONFIG_PATH)
    assert config["source_model"]["hyperparameter_grid_size"] == 0
    assert config["target_model"]["hyperparameter_grid_size"] == 0
    assert config["domain_shift"]["direct_source_target_row_concatenation_allowed"] is False
    assert config["domain_shift"]["observed_oof_auc"] == pytest.approx(0.9967791897555192)
    high = config["domain_shift"]["high_auc_inner_gate"]
    assert high["minimum_improved_inner_blocks"] == 2
    assert high["station_lead_or_ci_veto"] is False
    assert config["execution"]["actual_authorized"] is False
    integration = config["frozen_final_integration"]
    assert integration["allowed_pretruth_columns"] == [*PAIR_KEYS, *ROUTER_COLUMNS]
    assert integration["forbidden_pretruth_column"] == "target_hs"
    assert integration["weight_router_or_shrink_reselection"] is False
    assert config["promotion_gate"]["applies_to"] == (
        "challenger_final_vs_exact_incumbent_final_only"
    )
    scope = config["validation"]["rolling_origin_label_scope"]
    assert scope[
        "current_fold_validation_targets_excluded_from_that_fold_training_and_inner_selection"
    ]
    assert scope["earlier_fold_validation_targets_allowed_only_as_later_fold_training_history"]
    assert scope["future_fold_validation_targets_forbidden_from_earlier_fold_training"]
    assert (
        scope["global_process_level_zero_outer_target_exposure_before_blind_seal_claimed"] is False
    )


def test_preregistration_rejects_direct_concat_or_any_parameter_grid() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    bad = copy.deepcopy(config)
    bad["domain_shift"]["direct_source_target_row_concatenation_allowed"] = True
    with pytest.raises(KMASourceMetaError, match="concatenation"):
        validate_preregistration(bad)
    bad = copy.deepcopy(config)
    bad["target_model"]["hyperparameter_grid_size"] = 1
    with pytest.raises(KMASourceMetaError, match="search"):
        validate_preregistration(bad)


def test_source_surface_is_447_mask_free_and_period_free() -> None:
    config = load_preregistration(CONFIG_PATH)
    columns = list(compact_source_feature_columns())
    digest = hashlib.sha256(json.dumps(columns, separators=(",", ":")).encode()).hexdigest()
    assert len(columns) == 447
    assert digest == config["representation"]["expected_source_feature_columns_sha256"]
    assert not any("_valid_" in column for column in columns)
    assert not any(column.startswith("tp_") for column in columns)
    assert not any(column.startswith("steepness_proxy_") for column in columns)
    assert not any(
        column.startswith(("station_", "proxy_", "time_", "calendar_")) for column in columns
    )


def test_common_history_harmonizes_short_gaps_without_exposing_masks() -> None:
    history = _history()
    history.loc[90:91, "relh"] = np.nan
    row = summarize_common_history(history)
    assert tuple(row) == compact_source_feature_columns()
    assert row["relh_current"] == pytest.approx(history.loc[96, "relh"])
    assert not any("valid" in column or "missing" in column for column in row)
    frame = pd.DataFrame([row, row], columns=compact_source_feature_columns())
    frame.iloc[0, 0] = np.nan
    medians = fit_source_median_imputer(frame)
    imputed = apply_source_median_imputer(frame, medians)
    assert np.isfinite(imputed.to_numpy()).all()


def test_kma_wp_and_station_are_never_mapped_into_model_vocabulary() -> None:
    frame = pd.DataFrame(
        {
            "TM": pd.to_datetime(["2023-12-01T00:00:00+09:00"]),
            "STN": [22107],
            "WD1": [180.0],
            "WS1": [5.0],
            "WS1_GST": [7.0],
            "WD2": [181.0],
            "WS2": [5.1],
            "WS2_GST": [7.1],
            "PA": [1012.0],
            "HM": [70.0],
            "TA": [18.0],
            "WH_MAX": [2.5],
            "WH_SIG": [1.7],
            "WP": [8.0],
            "WO": [220.0],
        }
    )
    mapped = canonicalize_kma_observations(frame)
    assert "tp" not in mapped
    assert "WP" not in mapped
    assert "_source_group" in mapped
    assert not any(
        column.startswith(("station", "proxy")) for column in compact_source_feature_columns()
    )


def test_target_common_history_is_causal_and_period_free() -> None:
    anchor = pd.Timestamp("2025-05-01T00:00:00+00:00")
    wave_time = pd.date_range(
        anchor - pd.Timedelta(hours=48), anchor + pd.Timedelta(hours=1), freq="20min"
    )
    atmos_time = pd.date_range(
        anchor - pd.Timedelta(hours=48), anchor + pd.Timedelta(hours=1), freq="10min"
    )
    wave = pd.DataFrame(
        {
            "station": "G-ORS",
            "time": wave_time,
            "hs": np.linspace(1.0, 2.0, len(wave_time)),
            "tp": 9999.0,
            "hmax": np.linspace(1.5, 3.0, len(wave_time)),
            "wvdir": 210.0,
        }
    )
    atmos = pd.DataFrame(
        {
            "station": "G-ORS",
            "time": atmos_time,
            "wspd": 6.0,
            "gust": 8.0,
            "wdir": 180.0,
            "airt": 20.0,
            "relh": 70.0,
            "caph": 1010.0,
        }
    )
    first = extract_target_common_history(wave, atmos, station="G-ORS", anchor_time=anchor)
    wave.loc[wave["time"].gt(anchor), ["hs", "hmax"]] = 9999.0
    atmos.loc[atmos["time"].gt(anchor), ["wspd", "gust"]] = 9999.0
    second = extract_target_common_history(wave, atmos, station="G-ORS", anchor_time=anchor)
    pd.testing.assert_frame_equal(first, second)
    assert "tp" not in first


def test_domain_auc_routes_never_enable_direct_concat() -> None:
    assert resolve_domain_route(0.5).route == "full_six_prediction_meta"
    assert resolve_domain_route(0.7).route == "prediction_meta_only"
    high = resolve_domain_route(0.9967791897555192)
    assert high.route == "prediction_meta_only_with_inner_incremental_signal_gate"
    assert high.requires_inner_incremental_signal is True
    assert high.direct_concat_allowed is False


def test_exactly_six_source_predictions_are_the_only_appended_columns() -> None:
    target_columns = json.loads(
        (ROOT / "submissions/p3_frozen_catboost/feature_columns.json").read_text(encoding="utf-8")
    )
    base = pd.DataFrame(np.zeros((2, len(target_columns))), columns=target_columns)
    base.insert(0, "station", ["G-ORS", "I-ORS"])
    base.insert(0, "anchor_id", [1, 2])
    meta = source_predictions_to_meta(np.zeros((2, 6)), anchor_ids=[1, 2], current_hs=[1.5, 2.0])
    result = append_meta_features(base, meta, expected_base_columns=target_columns)
    assert list(result.columns[-6:]) == list(META_COLUMNS)
    assert len(result.columns) == 2 + 591 + 6
    with pytest.raises(KMASourceMetaError, match="exactly six"):
        append_meta_features(base, meta.assign(extra=1.0), expected_base_columns=target_columns)


def test_frozen_oof_reader_requests_key_columns_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = pd.DataFrame(
        {
            "fold": np.repeat(["a", "b", "c"], 6),
            "anchor_id": np.repeat([1, 2, 3], 6),
            "station": np.repeat(["G-ORS", "I-ORS", "S-ORS"], 6),
            "lead_h": list(LEADS) * 3,
            "target_hs": 9999.0,
            "prediction": 9999.0,
        }
    )
    path = tmp_path / "oof.parquet"
    frame.to_parquet(path, index=False)
    original = pd.read_parquet
    requested: list[str] | None = None

    def recording_read(path_value: Path, *, columns: list[str]) -> pd.DataFrame:
        nonlocal requested
        requested = list(columns)
        return original(path_value, columns=columns)

    monkeypatch.setattr(pd, "read_parquet", recording_read)
    keys, membership = read_frozen_outer_key_membership(path, expected_folds=["a", "b", "c"])
    assert requested == list(PAIR_KEYS)
    assert "target_hs" not in keys and "prediction" not in keys
    assert {key: value.tolist() for key, value in membership.items()} == {
        "a": [1],
        "b": [2],
        "c": [3],
    }


def test_frozen_router_reader_allows_only_label_free_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lead = np.asarray(list(LEADS) * 3)
    frame = pd.DataFrame(
        {
            "fold": np.repeat(["a", "b", "c"], 6),
            "anchor_id": np.repeat([1, 2, 3], 6),
            "station": np.repeat(["G-ORS", "I-ORS", "S-ORS"], 6),
            "lead_h": lead,
            "multi_prediction": 1.8,
            "persistence": 1.5,
            "weight_single": 0.5,
            "weight_multi": 0.3,
            "weight_persistence": 0.2,
            "second_stage_persistence_weight": np.where(np.isin(lead, [12, 18, 24]), 0.2, 0.0),
            "prediction": 1.7,
            "target_hs": 9999.0,
        }
    )
    path = tmp_path / "router.parquet"
    frame.to_parquet(path, index=False)
    original = pd.read_parquet
    requested: list[str] | None = None

    def recording_read(path_value: Path, *, columns: list[str]) -> pd.DataFrame:
        nonlocal requested
        requested = list(columns)
        return original(path_value, columns=columns)

    monkeypatch.setattr(pd, "read_parquet", recording_read)
    router = read_frozen_router_components(path)
    assert requested == [*PAIR_KEYS, *ROUTER_COLUMNS]
    assert "target_hs" not in router
    assert list(router.columns) == [*PAIR_KEYS, *ROUTER_COLUMNS]


def test_inner_high_auc_gate_uses_only_pooled_delta_and_two_of_three_blocks() -> None:
    inner = pd.DataFrame(
        {
            "fold": np.repeat(["a", "b", "c"], 2),
            "target_hs": [2.0] * 6,
            "control_prediction": [2.3] * 6,
            "challenger_prediction": [2.1, 2.1, 2.1, 2.1, 2.4, 2.4],
        }
    )
    result = evaluate_inner_incremental_signal(inner)
    assert result["improved_blocks"] == 2
    assert result["pooled_delta_rmse"] < 0.0
    assert result["pass"] is True
    assert result["station_lead_or_ci_veto_applied"] is False


def test_actual_inner_result_schema_connects_to_inner_gate() -> None:
    blocks: list[pd.DataFrame] = []
    for fold_number, fold in enumerate(("a", "b", "c")):
        metadata = pd.DataFrame(
            {
                "anchor_id": [fold_number],
                "station": ["S-ORS"],
                "lead_h": [3],
                "target_hs": [2.0],
            }
        )
        blocks.append(
            _assemble_inner_gate_frame(
                metadata,
                fold_name=fold,
                control_prediction=np.asarray([2.3]),
                challenger_prediction=np.asarray([2.1]),
            )
        )
    result = evaluate_inner_incremental_signal(pd.concat(blocks, ignore_index=True))
    assert result["pass"] is True
    assert result["improved_blocks"] == 3


def test_failed_inner_gate_receipt_reports_consumed_attempt_lock(tmp_path: Path) -> None:
    attempt_lock = tmp_path / "attempt.lock"
    attempt_lock.write_text('{"experiment_id":"test"}', encoding="utf-8")
    result = _inner_gate_no_go_result(
        auc=0.99,
        domain_route="prediction_meta_only_with_inner_incremental_signal_gate",
        inner_gate={"pass": False},
        attempt_lock=attempt_lock,
    )
    assert attempt_lock.is_file()
    assert result["one_shot_locks_created"] == 1
    assert result["global_attempt_lock_created"] is True
    assert result["outer_truth_locks_created"] == 0
    assert result["rerun_prohibited"] is True
    assert (
        result["global_attempt_lock_sha256"]
        == hashlib.sha256(attempt_lock.read_bytes()).hexdigest()
    )
    with pytest.raises(PermissionError, match="attempt lock"):
        _inner_gate_no_go_result(
            auc=0.99,
            domain_route="prediction_meta_only_with_inner_incremental_signal_gate",
            inner_gate={"pass": False},
            attempt_lock=tmp_path / "missing.lock",
        )


def test_blind_prediction_rejects_truth_and_nonfinite_values() -> None:
    single = pd.DataFrame(
        {
            "fold": ["a"],
            "anchor_id": [1],
            "station": ["G-ORS"],
            "lead_h": [3],
            "current_hs": [1.5],
            "control_single_prediction": [1.6],
            "challenger_single_prediction": [1.55],
        }
    )
    router = pd.DataFrame(
        {
            "fold": ["a"],
            "anchor_id": [1],
            "station": ["G-ORS"],
            "lead_h": [3],
            "multi_prediction": [1.7],
            "persistence": [1.5],
            "weight_single": [0.5],
            "weight_multi": [0.3],
            "weight_persistence": [0.2],
            "second_stage_persistence_weight": [0.0],
            "prediction": [1.62],
        }
    )
    blind = integrate_frozen_router(single, router)
    assert validate_blind_prediction_frame(blind)["rows"] == 1
    with pytest.raises(KMASourceMetaError, match="schema"):
        validate_blind_prediction_frame(blind.assign(target_hs=1.7))
    bad = blind.copy()
    bad["challenger_final"] = np.nan
    with pytest.raises(KMASourceMetaError, match="non-finite"):
        validate_blind_prediction_frame(bad)


def test_target_vault_blocks_outer_train_overlap_and_unsealed_outer(tmp_path: Path) -> None:
    targets = pd.DataFrame({"anchor_id": [1, 2]})
    for lead in LEADS:
        targets[f"target_{lead}"] = [1.0, 2.0]
    path = tmp_path / "targets.parquet"
    targets.to_parquet(path, index=False)
    vault = TargetVault(path)
    with pytest.raises(PermissionError, match="validation"):
        vault.read_outer_train(
            np.asarray([1]),
            forbidden_outer_validation_ids=np.asarray([1]),
            all_outer_validation_ids=np.asarray([1, 2]),
            allowed_prior_validation_ids=np.asarray([], dtype=np.int64),
            fold="a",
        )
    released = vault.read_outer_train(
        np.asarray([1]),
        forbidden_outer_validation_ids=np.asarray([2]),
        all_outer_validation_ids=np.asarray([1, 2]),
        allowed_prior_validation_ids=np.asarray([1]),
        fold="later_fold",
    )
    assert released["anchor_id"].tolist() == [1]
    assert vault.access_log[-1]["permitted_prior_validation_history_rows"] == 1
    with pytest.raises(PermissionError, match="future or current"):
        vault.read_outer_train(
            np.asarray([2]),
            forbidden_outer_validation_ids=np.asarray([3]),
            all_outer_validation_ids=np.asarray([1, 2, 3]),
            allowed_prior_validation_ids=np.asarray([1]),
            fold="earlier_fold",
        )
    manifest = tmp_path / "manifest.json"
    receipt = tmp_path / "receipt.json"
    manifest.write_text(json.dumps({"sealed": False}), encoding="utf-8")
    receipt.write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(PermissionError, match="sealed"):
        vault.open_outer_once(np.asarray([2]), sealed_manifest=manifest, exposure_receipt=receipt)
    assert vault.outer_open_count == 0


def test_actual_is_blocked_and_config_path_cannot_be_relocated(tmp_path: Path) -> None:
    config = load_preregistration(CONFIG_PATH)
    mutated = copy.deepcopy(config)
    mutated["target_model"]["depth"] += 1
    with pytest.raises(PermissionError, match="in-memory config"):
        _actual(
            mutated,
            p3_data_dir=tmp_path,
            authorization_token=AUTHORIZATION_TOKEN,
            started=0.0,
        )
    with pytest.raises(PermissionError, match="authorization amendment"):
        _actual(
            config,
            p3_data_dir=tmp_path,
            authorization_token=AUTHORIZATION_TOKEN,
            started=0.0,
        )
    copied = tmp_path / "copied.json"
    copied.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(PermissionError, match="override"):
        _canonical_config(copied)
    assert _canonical_config(CANONICAL_CONFIG) == CANONICAL_CONFIG


def test_outer_promotion_gate_is_case_paired_and_strict() -> None:
    config = load_preregistration(CONFIG_PATH)
    rows: list[dict[str, object]] = []
    for fold_number, fold in enumerate(("a", "b", "c")):
        for lead in LEADS:
            rows.append(
                {
                    "fold": fold,
                    "anchor_id": fold_number,
                    "station": ("G-ORS", "I-ORS", "S-ORS")[fold_number],
                    "lead_h": lead,
                    "target_hs": 2.0,
                    "incumbent_final": 2.2,
                    "challenger_final": 2.02,
                }
            )
    result = evaluate_promotion(pd.DataFrame(rows), config)
    assert result["decision"] == "GO_TO_INTEGRATION"
    assert result["paired_case_bootstrap"]["ci90_upper"] < 0.0
    assert all(result["checks"].values())
    with pytest.raises(KMASourceMetaError, match="only compare"):
        evaluate_promotion(
            pd.DataFrame(rows),
            config,
            control_column="control_single_prediction",
            challenger_column="challenger_single_prediction",
        )
