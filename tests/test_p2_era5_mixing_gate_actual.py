from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import p2_restore.era5_mixing_gate_actual as actual
from p2_restore.era5_mixing_gate import (
    ERA5_MIXING_FEATURES,
    ERA5_VALUE_COLUMNS,
    build_hourly_ocean_mixing_features,
)
from p2_restore.era5_mixing_gate_actual import (
    AppendOnlyLedger,
    FoldLocalTruthVault,
    GrantBoundEra5Reader,
    assert_arm_symmetry,
    mask_target_layers_for_validation_block,
    outer_promotion_decision,
    paired_kst_day_bootstrap,
    purge_training_rows,
    run_fold_local_gate,
    sha256_file,
    validate_native_flux_sign_contract,
    validate_truth_shard_key_contract,
    verify_append_only_ledger,
    write_and_seal_blind_predictions,
    write_json_exclusive_fsync,
)
from p2_restore.regime_gate import STATE_FEATURES

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_runner_module():
    path = REPO_ROOT / "scripts/run_p2_era5_mixing_gate_actual_v1.py"
    spec = importlib.util.spec_from_file_location("p2_era5_actual_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _observation_fixture() -> pd.DataFrame:
    times = pd.to_datetime(
        [
            "2024-08-31T23:50:00+09:00",
            "2024-09-01T00:00:00+09:00",
            "2024-10-31T23:50:00+09:00",
            "2024-11-01T00:00:00+09:00",
        ]
    )
    rows = []
    for time in times:
        for layer in (1, 2, 3, 4, 5, 6, 7, 8):
            rows.append(
                {
                    "station": "S-ORS",
                    "year": 2024,
                    "layer": layer,
                    "time": time.isoformat(),
                    "temp": np.nan if layer == 3 and time == times[1] else 10.0 + layer,
                    "psal": np.nan if layer == 2 and time == times[1] else 30.0 + layer,
                    "depth": float(layer),
                    "nominal_depth": float(layer),
                }
            )
    return pd.DataFrame(rows)


def _blind_fixture() -> pd.DataFrame:
    deep = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    physical = deep + np.array([0.2, -0.1, 0.5, -0.2, 0.3, -0.4])
    control_weight = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    selected_weight = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.4])
    return pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=6, freq="10min", tz="UTC"),
            "layer": [2, 3, 4, 2, 3, 4],
            "block": pd.Series(["a"] * 6, dtype="string"),
            "deep_prediction": deep,
            "physical_prediction": physical,
            "control_physical_weight": control_weight,
            "selected_physical_weight": selected_weight,
            "control_prediction": (1 - control_weight) * deep + control_weight * physical,
            "selected_prediction": (1 - selected_weight) * deep + selected_weight * physical,
            "selected_arm": pd.Series(["challenger"] * 6, dtype="string"),
        }
    )


def _truth_shards(root: Path) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    rows = []
    for block_number, block in enumerate(actual.OUTER_BLOCKS):
        for layer in (2, 3, 4):
            for number, time in enumerate(
                pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC")
            ):
                rows.append(
                    {
                        "time": time + pd.Timedelta(days=100 * block_number),
                        "layer": layer,
                        "block": block,
                        "truth": 10.0 + layer + number,
                    }
                )
    frame = pd.DataFrame(rows)
    specs: dict[str, dict[str, object]] = {}
    for block in actual.OUTER_BLOCKS:
        path = root / f"{block}.parquet"
        current = frame.loc[frame["block"].eq(block)].copy()
        current.to_parquet(path, index=False)
        specs[block] = {
            "path": path.name,
            "sha256": sha256_file(path),
            "rows": len(current),
            "row_groups": 1,
            "row_group_block_min": block,
            "row_group_block_max": block,
        }
    return frame, specs


def _fold_design() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    truth_rows = []
    for block_number, block in enumerate(actual.OUTER_BLOCKS):
        start = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=120 * block_number)
        for day in range(66):
            state = -1.0 + 2.0 * day / 65
            for layer in (2, 3, 4):
                deep = 10.0 + layer + state + 0.1
                physical = 10.0 + layer + state - 0.1
                row = {
                    "time": start + pd.Timedelta(days=day),
                    "layer": layer,
                    "block": block,
                    "deep_prediction": deep,
                    "physical_prediction": physical,
                }
                for index, name in enumerate(STATE_FEATURES):
                    row[name] = state if index == 0 else 0.01 * index
                for name in ERA5_MIXING_FEATURES:
                    row[name] = 0.0
                rows.append(row)
                truth_rows.append(
                    {
                        "time": row["time"],
                        "layer": layer,
                        "block": block,
                        "truth": deep if state < 0 else physical,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(truth_rows)


def test_canonical_config_and_hardcoded_dry_contract_are_equal() -> None:
    runner = _load_runner_module()
    contract = runner._load_canonical_config()
    dry = runner._assert_dry_generation(contract)
    assert dry["in_memory_equality"] is True
    assert runner.CANONICAL_CONFIG_PATH == (
        REPO_ROOT / "configs/experiments/p2_era5_mixing_gate_actual_v1.json"
    )
    assert contract["outputs"]["actual_directory"] == (
        "artifacts/p2_era5_mixing_gate_actual_v1_run1"
    )


def test_superseding_v2_grant_exactly_pins_nine_variable_value_contract() -> None:
    runner = _load_runner_module()
    contract = runner._load_canonical_config()
    grant = runner._scope_grant(contract)
    audit = runner._assert_grant_value_contract(contract, grant)
    assert audit["allowed_variables"] == list(ERA5_VALUE_COLUMNS)
    assert audit["surface_energy_flux_positive_direction"] == "downward"
    assert audit["read_path_source"] == "grant.parquet_path_only"
    assert audit["component_sign_flips_applied"] == 0


def test_expert_preflight_reads_keys_but_no_prediction_or_truth_values() -> None:
    runner = _load_runner_module()
    contract = runner._load_canonical_config()
    audit = runner._preflight_expert_key_contract(contract)
    assert audit["deep_physical_keys_exact_equal"] is True
    assert audit["union_key_sha256"] == contract["oof_contract"]["key_sha256"]
    assert audit["prediction_columns_requested"] == 0
    assert audit["truth_value_columns_requested"] == 0


def test_o_excl_json_lock_rejects_second_create_without_changing_bytes(tmp_path: Path) -> None:
    path = tmp_path / "attempt.lock"
    write_json_exclusive_fsync(path, {"attempt": 1})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_json_exclusive_fsync(path, {"attempt": 2})
    assert path.read_bytes() == before


def test_append_only_ledger_detects_hash_tamper_and_truncation(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = AppendOnlyLedger(path, experiment_id="x")
    ledger.append("attempt_reserved", {"value": 1})
    with pytest.raises(ValueError, match="duplicate"):
        ledger.append("attempt_reserved", {"value": 1})
    audit = verify_append_only_ledger(path, experiment_id="x")
    assert audit["record_count"] == 2

    tampered = tmp_path / "tampered.jsonl"
    tampered.write_bytes(path.read_bytes().replace(b'"value":1', b'"value":2'))
    with pytest.raises(ValueError, match="SHA"):
        verify_append_only_ledger(tampered, experiment_id="x")
    truncated = tmp_path / "truncated.jsonl"
    truncated.write_bytes(path.read_bytes()[:-3])
    with pytest.raises((ValueError, json.JSONDecodeError)):
        verify_append_only_ledger(truncated, experiment_id="x")


def test_era5_value_read_is_rejected_before_grant(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def forbidden(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("Parquet read must not be reached")

    monkeypatch.setattr(actual.pq, "read_table", forbidden)
    reader = GrantBoundEra5Reader()
    with pytest.raises(PermissionError, match="before"):
        reader.read_values(columns=["time_utc"])
    assert calls == 0
    assert reader.events == []


def test_joint_mask_uses_half_open_kst_boundaries_and_is_idempotent() -> None:
    observations = _observation_fixture()
    original = observations.copy(deep=True)
    keys = pd.DataFrame(
        {
            "time": ["2024-09-01T00:00:00+09:00"],
            "layer": [2],
            "block": ["2024_sep_oct"],
        }
    )
    masked, audit = mask_target_layers_for_validation_block(observations, keys)
    times = pd.to_datetime(masked["time"])
    inside = times.ge(pd.Timestamp("2024-09-01T00:00:00+09:00")) & times.lt(
        pd.Timestamp("2024-11-01T00:00:00+09:00")
    )
    target = masked["layer"].isin([2, 3, 4])
    assert masked.loc[inside & target, ["temp", "psal"]].isna().all().all()
    assert masked.loc[~inside | ~target].equals(original.loc[~inside | ~target])
    assert observations.equals(original)
    assert audit["post_mask_nonmissing_target_values"] == 0
    twice, _ = mask_target_layers_for_validation_block(masked, keys)
    assert twice.equals(masked)


def test_public_feature_builder_receives_jointly_masked_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations = _observation_fixture()
    keys = pd.DataFrame(
        {
            "time": ["2024-09-01T00:00:00+09:00"] * 3,
            "layer": [2, 3, 4],
            "block": ["2024_sep_oct"] * 3,
        }
    )
    seen_masked = False

    def spy(frame: pd.DataFrame, current_keys: pd.DataFrame) -> pd.DataFrame:
        nonlocal seen_masked
        time = pd.to_datetime(frame["time"])
        selected = (
            frame["layer"].isin([2, 3, 4])
            & time.ge(pd.Timestamp("2024-09-01T00:00:00+09:00"))
            & time.lt(pd.Timestamp("2024-11-01T00:00:00+09:00"))
        )
        seen_masked = bool(frame.loc[selected, ["temp", "psal"]].isna().all().all())
        result = current_keys.copy()
        for name in STATE_FEATURES:
            result[name] = 0.0
        return result

    monkeypatch.setattr(actual, "build_public_state_features", spy)
    panel, audits = actual.build_block_masked_public_state_panel(
        observations,
        keys,
        outer_blocks=("2024_sep_oct",),
    )
    assert seen_masked is True
    assert len(panel) == 3 and len(audits) == 1


def test_seven_day_purge_has_half_open_right_boundary() -> None:
    validation = pd.DataFrame(
        {
            "time": pd.to_datetime(["2025-01-10T00:00:00Z", "2025-01-11T00:00:00Z"]),
            "layer": [2, 2],
            "block": ["a", "a"],
        }
    )
    start = pd.Timestamp("2025-01-10T00:00:00Z") - pd.Timedelta(hours=168)
    end = pd.Timestamp("2025-01-11T00:00:00Z") + pd.Timedelta(hours=168)
    train = pd.DataFrame(
        {
            "time": [
                start - pd.Timedelta(minutes=10),
                start,
                end - pd.Timedelta(minutes=10),
                end,
            ],
            "layer": [2] * 4,
            "block": ["a"] * 4,
        }
    )
    keep, _ = purge_training_rows(train, validation)
    assert keep.tolist() == [True, False, False, True]


def test_era5_rolling_windows_reset_at_cross_block_boundary() -> None:
    rows = []
    for block_number, block in enumerate(("a", "b")):
        for hour, time in enumerate(
            pd.date_range("2025-01-01", periods=170, freq="h", tz="UTC")
            + pd.Timedelta(days=400 * block_number)
        ):
            row = {
                "block": block,
                "time_utc": time,
                "latitude": 37.5,
                "longitude": 124.75,
                "land_sea_mask": 0.0,
            }
            for name in ERA5_VALUE_COLUMNS[:-1]:
                row[name] = float(hour + 1)
            rows.append(row)
    features = build_hourly_ocean_mixing_features(
        pd.DataFrame(rows), expected_ocean_cells_per_hour=1
    )
    for block in ("a", "b"):
        current = features.loc[features["block"].eq(block)].reset_index(drop=True)
        assert pd.isna(current.loc[166, "era5_qnet_energy_168h_jm2"])
        assert np.isfinite(current.loc[167, "era5_qnet_energy_168h_jm2"])


def test_arm_contract_is_symmetric_except_exact_era5_suffix() -> None:
    assert_arm_symmetry()
    control = actual.arm_specification(STATE_FEATURES)
    challenger = actual.arm_specification((*STATE_FEATURES, *ERA5_MIXING_FEATURES))
    assert challenger["prediction_columns"] == control["prediction_columns"]
    assert challenger["regularization"] == control["regularization"]
    assert challenger["max_iterations"] == control["max_iterations"]
    assert challenger["feature_names"] == [*control["feature_names"], *ERA5_MIXING_FEATURES]


def test_native_flux_semantics_are_positive_downward_without_numeric_sign_flip() -> None:
    runner = _load_runner_module()
    contract = runner._load_canonical_config()
    audit = validate_native_flux_sign_contract(contract)
    assert set(audit["component_sign_semantics"].values()) == {
        "positive downward accumulated J m-2"
    }
    assert audit["component_sign_flips_applied"] == 0

    row = {
        "block": "a",
        "time_utc": pd.Timestamp("2025-01-01T00:00:00Z"),
        "latitude": 37.5,
        "longitude": 124.75,
        "land_sea_mask": 0.0,
        "10m_u_component_of_wind": 0.0,
        "10m_v_component_of_wind": 0.0,
        "eastward_turbulent_surface_stress": 0.0,
        "northward_turbulent_surface_stress": 0.0,
        "surface_net_solar_radiation": 100.0,
        "surface_net_thermal_radiation": -30.0,
        "surface_latent_heat_flux": -20.0,
        "surface_sensible_heat_flux": -10.0,
    }
    features = build_hourly_ocean_mixing_features(
        pd.DataFrame([row]), expected_ocean_cells_per_hour=1
    )
    assert features.loc[0, "era5_qnet_native_wm2"] == pytest.approx(40.0 / 3600.0)


def test_fold_local_gate_rejects_current_outer_truth_and_returns_convex_predictions() -> None:
    design, truth = _fold_design()
    outer = "2025_nov_dec"
    forbidden = truth.copy()
    with pytest.raises(PermissionError, match="current outer"):
        run_fold_local_gate(design, forbidden, outer_block=outer)

    allowed = truth.loc[~truth["block"].eq(outer)]
    outcome = run_fold_local_gate(design, allowed, outer_block=outer)
    assert outcome.summary["outer_truth_rows_seen_during_fit"] == 0
    assert len(outcome.summary["inner_blocks"]) == 3
    assert len(outcome.predictions) == len(design.loc[design["block"].eq(outer)])
    lower = np.minimum(
        outcome.predictions["deep_prediction"], outcome.predictions["physical_prediction"]
    )
    upper = np.maximum(
        outcome.predictions["deep_prediction"], outcome.predictions["physical_prediction"]
    )
    assert outcome.predictions["selected_prediction"].between(lower, upper).all()


def test_blind_parquet_is_fsynced_reloaded_exact_and_sealed(tmp_path: Path) -> None:
    frame = _blind_fixture()
    parquet = tmp_path / "blind.parquet"
    seal = tmp_path / "blind.seal.json"
    result = write_and_seal_blind_predictions(
        frame,
        parquet_path=parquet,
        seal_path=seal,
        seal_metadata={"experiment_id": "x"},
    )
    assert result["rows"] == len(frame)
    payload = json.loads(seal.read_text(encoding="utf-8"))
    assert payload["truth_columns_present"] is False
    assert payload["reload_values_exact_equal"] is True
    assert payload["parquet_sha256_unchanged_after_reload"] is True
    assert payload["parquet_sha256"] == sha256_file(parquet)
    with pytest.raises(FileExistsError):
        write_and_seal_blind_predictions(
            frame,
            parquet_path=parquet,
            seal_path=tmp_path / "second.seal.json",
            seal_metadata={},
        )

    invalid = frame.copy()
    invalid.loc[0, "selected_physical_weight"] = 1.1
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        write_and_seal_blind_predictions(
            invalid,
            parquet_path=tmp_path / "invalid.parquet",
            seal_path=tmp_path / "invalid.seal.json",
            seal_metadata={},
        )


def test_truth_shard_preflight_reads_keys_only_and_pins_union(tmp_path: Path) -> None:
    truth, specs = _truth_shards(tmp_path)
    normalized = truth.loc[:, actual.KEY_COLUMNS].copy()
    normalized["time"] = pd.to_datetime(normalized["time"], utc=True).map(
        lambda value: value.isoformat()
    )
    digest = hashlib.sha256(
        normalized.to_csv(index=False, header=False, lineterminator="\n").encode()
    ).hexdigest()
    audit = validate_truth_shard_key_contract(
        tmp_path,
        specs,
        expected_union_rows=len(truth),
        expected_union_key_sha256=digest,
    )
    assert audit["union_keys_unique"] is True
    assert audit["truth_value_columns_requested"] == 0
    assert all(row["truth_value_column_requested"] is False for row in audit["shards"].values())


def test_truth_vault_never_opens_current_shard_and_designated_open_is_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truth, specs = _truth_shards(tmp_path)
    original_read_table = actual.pq.read_table
    for outer in actual.OUTER_BLOCKS:
        opened: list[str] = []

        def logged_read_table(path, *args, _opened=opened, **kwargs):
            _opened.append(Path(path).name)
            return original_read_table(path, *args, **kwargs)

        monkeypatch.setattr(actual.pq, "read_table", logged_read_table)
        vault = FoldLocalTruthVault(tmp_path, specs)
        train = vault.open_fold_train(outer)
        assert outer not in set(train["block"].astype(str))
        assert Path(str(specs[outer]["path"])).name not in opened
        assert all(row["block"] != outer for row in vault.audit()["truth_value_open_log"])
        with pytest.raises(PermissionError, match="exactly one"):
            vault.open_fold_train(outer)
        monkeypatch.setattr(actual.pq, "read_table", original_read_table)

    seal = tmp_path / "seal.json"
    write_json_exclusive_fsync(seal, {"sealed": True})
    scoring_vault = FoldLocalTruthVault(tmp_path, specs)
    designated = scoring_vault.open_designated_outer_once(
        blind_seal_path=seal,
        expected_blind_seal_sha256=sha256_file(seal),
        completed_outer_blocks=actual.OUTER_BLOCKS,
    )
    assert len(designated) == len(truth)
    with pytest.raises(PermissionError, match="exactly once"):
        scoring_vault.open_designated_outer_once(
            blind_seal_path=seal,
            expected_blind_seal_sha256=sha256_file(seal),
            completed_outer_blocks=actual.OUTER_BLOCKS,
        )


def test_promotion_threshold_boundaries_are_exact() -> None:
    metrics = {
        "pooled": {"improvement_rmse": 0.005},
        "by_block": {
            "a": {"delta_rmse": -0.1},
            "b": {"delta_rmse": -0.01},
            "c": {"delta_rmse": 0.02},
        },
        "by_layer": {
            "2": {"delta_rmse": -0.01},
            "3": {"delta_rmse": 0.0},
            "4": {"delta_rmse": 0.01},
        },
    }
    passed = outer_promotion_decision(metrics, {"delta_rmse_ci_upper": -1e-12})
    assert passed["promoted"] is True
    failed = outer_promotion_decision(metrics, {"delta_rmse_ci_upper": 0.0})
    assert failed["promoted"] is False


def test_bootstrap_uses_kst_day_and_is_seed_reproducible() -> None:
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2025-01-01T14:50:00Z", "2025-01-01T15:00:00Z", "2025-01-02T15:00:00Z"]
            ),
            "truth": [1.0, 1.0, 1.0],
            "control_prediction": [1.2, 1.3, 1.4],
            "selected_prediction": [1.1, 1.2, 1.3],
        }
    )
    first = paired_kst_day_bootstrap(frame, replicates=100, seed=9)
    second = paired_kst_day_bootstrap(frame.sample(frac=1, random_state=3), replicates=100, seed=9)
    assert first == second
    assert first["day_count"] == 3
