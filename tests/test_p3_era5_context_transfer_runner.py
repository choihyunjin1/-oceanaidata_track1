from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_p3_era5_context_transfer_v1.py"
SPEC = importlib.util.spec_from_file_location("p3_era5_transfer_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _metric_rows(*, baseline_column: str, baseline_value: float) -> pd.DataFrame:
    rows = []
    stations = ("G-ORS", "I-ORS", "S-ORS")
    folds = ("2024_h2_storm", "winter_transition", "2025_h1")
    for episode, (station, fold) in enumerate(zip(stations, folds, strict=True), start=1):
        for lead in runner.LEADS:
            rows.append(
                {
                    "episode_id": episode,
                    "station": station,
                    "fold": fold,
                    "lead_h": lead,
                    "target_hs": 1.0,
                    "transfer_prediction": 1.0,
                    baseline_column: baseline_value,
                }
            )
    return pd.DataFrame(rows)


def _full_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    blind_rows = []
    truth_rows = []
    fold_names = ("2024_h2_storm", "winter_transition", "2025_h1")
    stations = ("G-ORS", "I-ORS", "S-ORS")
    for anchor_id in range(1, runner.EXPECTED_CASES + 1):
        fold = fold_names[(anchor_id - 1) % 3]
        station = stations[(anchor_id - 1) % 3]
        for lead in runner.LEADS:
            key = {
                "fold": fold,
                "anchor_id": anchor_id,
                "station": station,
                "lead_h": lead,
            }
            blind_rows.append(
                {
                    **key,
                    "current_hs": 1.0,
                    "incumbent_prediction": 1.1,
                    "transfer_prediction": 1.0,
                    "local_control_prediction": 1.05,
                    "episode_id": anchor_id,
                }
            )
            truth_rows.append({"prefix_fraction": 1.0, **key, "target_hs": 1.0})
    return pd.DataFrame(blind_rows), pd.DataFrame(truth_rows)


def test_external_hourly_mapping_is_exact_and_ordered() -> None:
    assert runner.EXTERNAL_COLUMN_MAP == {
        "swh_m": "hs",
        "mwp_s": "tp",
        "hmax_m": "hmax",
        "mwd_deg": "wvdir",
        "wspd10_m_s": "wspd",
        "wdir10_from_deg": "wdir",
        "t2m_c": "airt",
        "relh2m_pct": "relh",
        "msl_hpa": "caph",
    }


def test_source_value_reader_runs_generic_preflight_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    quarantine = tmp_path / "quarantine"
    candidate = quarantine / "derived" / runner.COMBINED_NAME
    manifest = quarantine / "manifests" / "manifest.json"
    candidate.parent.mkdir(parents=True)
    manifest.parent.mkdir(parents=True)
    candidate.write_bytes(b"sealed external bytes")
    manifest.write_text("{}", encoding="utf-8")
    paths = runner.RunPaths(
        root=tmp_path,
        experiment_config=tmp_path / "config.json",
        external_scope=tmp_path / "scope.json",
        catalog=tmp_path / "catalog.toml",
        permission_receipt=tmp_path / "receipt.json",
        train_features=tmp_path / "features.parquet",
        train_anchors=tmp_path / "anchors.parquet",
        validation_keys=tmp_path / "keys.parquet",
        incumbent_oof=tmp_path / "oof.parquet",
        quarantine=quarantine,
        manifest=manifest,
        output=tmp_path / "output",
        attempt_lock=tmp_path / "attempt.lock",
    )
    events: list[str] = []

    selected = {
        name: types.SimpleNamespace(public_dict=lambda station=name: {"station": station})
        for name in runner.era5_data.STATIONS
    }
    selected_payload = [selected[name].public_dict() for name in runner.era5_data.STATIONS]

    def preflight(_: runner.RunPaths) -> tuple[dict[str, bool], Path, dict[str, object]]:
        events.append("preflight")
        return {
            "accepted": True
        }, candidate, {
            "row_count": 1,
            "observed_start": None,
            "observed_end": None,
            "selected_cells": selected_payload,
        }

    frame = pd.DataFrame({column: [1.0] for column in runner.era5_data.DERIVED_COLUMNS})
    frame["station"] = "G-ORS"
    frame["time_utc"] = pd.Timestamp("2023-01-01", tz="UTC")

    summary = {
        "row_count": 1,
        "observed_start": None,
        "observed_end": None,
        "station_count": 1,
        "rows_per_station": 1,
    }

    def validated_loader(
        layout: runner.era5_data.QuarantineLayout, selections: dict[str, object]
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        events.append("read")
        assert layout.root == quarantine
        assert selections == selected
        return frame.copy(), summary

    monkeypatch.setattr(runner, "_external_preflight", preflight)
    monkeypatch.setattr(
        runner.era5_data, "_validate_completed_manifest_payload", lambda *_: None
    )
    monkeypatch.setattr(runner.era5_data, "read_selected_cells", lambda *_: selected)
    monkeypatch.setattr(runner.era5_data, "load_validated_combined_file", validated_loader)
    observed, provenance = runner._load_source_hourly(paths)
    assert events == ["preflight", "read"]
    assert list(observed.columns) == [
        "station",
        "time",
        *runner.EXTERNAL_COLUMN_MAP.values(),
    ]
    assert provenance["generic_preflight"]["accepted"] is True


def test_matched_local_control_is_frozen_before_execution() -> None:
    assert runner.LOCAL_CONTROL_STAGE1_PARAMETERS == {
        "loss_function": "RMSE",
        "iterations": 600,
        "depth": 8,
        "learning_rate": 0.04,
        "l2_leaf_reg": 8.0,
        "random_seed": 20260824,
        "thread_count": -1,
        "allow_writing_files": False,
        "verbose": False,
    }
    assert runner.LOCAL_CONTROL_STAGE2_PARAMETERS["iterations"] == 250
    assert runner.LOCAL_CONTROL_STAGE2_PARAMETERS["learning_rate"] == 0.03
    assert runner.LOCAL_CONTROL_STAGE2_PARAMETERS["l2_leaf_reg"] == 12.0
    assert runner.MATCHED_CONTROL_CONFIG["stage1"]["sample_weight"] == "uniform"
    assert (
        runner.MATCHED_CONTROL_CONFIG["stage2"]["init_model"]
        == "independent_stage1_local_model"
    )


def test_matched_control_uses_unweighted_stage1_then_weighted_stage2_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed = []

    class FakeCatBoost:
        def __init__(self, **parameters: object) -> None:
            self.parameters = parameters
            self.fit_calls = []
            constructed.append(self)

        def fit(self, x: pd.DataFrame, y: np.ndarray, **kwargs: object) -> None:
            self.fit_calls.append((x.copy(), np.asarray(y).copy(), kwargs))

    monkeypatch.setitem(
        sys.modules, "catboost", types.SimpleNamespace(CatBoostRegressor=FakeCatBoost)
    )
    features = pd.DataFrame(
        np.zeros((1, runner.EXPECTED_FEATURES)),
        columns=runner.common_feature_columns(),
    )
    control = runner._LocalOnlyControl().fit(
        features,
        np.zeros((1, len(runner.LEADS))),
        current_hs=np.array([2.0]),
    )
    assert len(constructed) == 2
    stage1, stage2 = constructed
    assert stage1.parameters == runner.LOCAL_CONTROL_STAGE1_PARAMETERS
    assert stage2.parameters == runner.LOCAL_CONTROL_STAGE2_PARAMETERS
    assert "sample_weight" not in stage1.fit_calls[0][2]
    stage2_kwargs = stage2.fit_calls[0][2]
    assert stage2_kwargs["init_model"] is not stage1
    np.testing.assert_allclose(
        stage2_kwargs["sample_weight"],
        np.repeat(np.exp(-0.45 * 0.5), len(runner.LEADS)),
    )
    assert control._model is stage2


def test_fixed_shrink_only_changes_registered_long_leads() -> None:
    raw = np.arange(12, dtype=float).reshape(2, 6)
    current = np.array([10.0, 20.0])
    observed = runner._apply_fixed_shrink(raw, current)
    np.testing.assert_allclose(observed[:, :3], raw[:, :3])
    np.testing.assert_allclose(observed[:, 3:], 0.8 * raw[:, 3:] + 0.2 * current[:, None])


def test_source_held_year_predictions_receive_the_same_fixed_postprocess() -> None:
    raw = np.arange(6, dtype=float).reshape(1, 6)

    class SourceModel:
        def predict_hs(self, features: pd.DataFrame, *, current_hs: np.ndarray) -> np.ndarray:
            return raw.copy()

    observed = runner._predict_source_with_fixed_postprocess(
        SourceModel(), pd.DataFrame(index=[0]), np.array([10.0])
    )
    np.testing.assert_allclose(observed[:, :3], raw[:, :3])
    np.testing.assert_allclose(observed[:, 3:], 0.8 * raw[:, 3:] + 2.0)


def test_fold_target_reader_uses_only_requested_ids(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "anchor_id": [1, 2, 3],
            "current_hs": [1.0, 2.0, 3.0],
            **{f"target_{lead}": [1.1, 2.1, 3.1] for lead in runner.LEADS},
        }
    )
    path = tmp_path / "anchors.parquet"
    frame.to_parquet(path, index=False)
    observed = runner._read_training_targets(path, [3, 1])
    assert observed["anchor_id"].tolist() == [3, 1]
    assert 2 not in set(observed["anchor_id"])
    with pytest.raises(runner.ContractError):
        runner._read_training_targets(path, [1, 1])


def test_restricted_folds_independently_reject_noncausal_training_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchors = pd.read_parquet(
        ROOT / "artifacts/p3/features_all20_v1/train_anchors.parquet",
        columns=["anchor_id", "station", "anchor_time", "current_hs"],
    )
    anchors["anchor_time"] = pd.to_datetime(anchors["anchor_time"], utc=True)
    keys = pd.read_parquet(
        ROOT / "artifacts/p3_meaningful_learning_curve_20260823_v1/validation_keys.parquet"
    )
    genuine = runner.build_forecast_folds(
        anchors, windows=runner.DEFAULT_WINDOWS, embargo_hours=78
    )
    bad_first = runner.replace(
        genuine[0],
        train_ids=np.append(genuine[0].train_ids, genuine[0].validation_ids[0]),
    )
    monkeypatch.setattr(
        runner,
        "build_forecast_folds",
        lambda *_args, **_kwargs: (bad_first, *genuine[1:]),
    )
    with pytest.raises(runner.ContractError, match="strictly earlier"):
        runner._restricted_folds(anchors, keys)


def test_comparator_loader_requests_no_outcome_column(monkeypatch: pytest.MonkeyPatch) -> None:
    original = pd.read_parquet
    calls: list[list[str] | None] = []

    def spy(path: Path, *, columns: list[str] | None = None, **kwargs: object) -> pd.DataFrame:
        calls.append(columns)
        return original(path, columns=columns, **kwargs)

    monkeypatch.setattr(runner.pd, "read_parquet", spy)
    keys = original(ROOT / "artifacts/p3_meaningful_learning_curve_20260823_v1/validation_keys.parquet")
    observed = runner._load_blind_comparator(
        ROOT / "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_v1r2/oof.parquet",
        keys,
    )
    assert len(observed) == runner.EXPECTED_ROWS
    assert calls == [list(runner.BLIND_COMPARATOR_COLUMNS)]
    assert "target_hs" not in calls[0]


def test_truth_reader_is_unreachable_before_matching_durable_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blind, truth = _full_rows()
    seal = tmp_path / "blind.parquet"
    with pytest.raises(runner.ContractError):
        runner._attach_truth_after_seal(
            blind, seal_path=seal, seal_sha256="0" * 64, truth_path=tmp_path / "oof.parquet"
        )

    seal_sha = runner._atomic_parquet(blind, seal)
    called = False

    def truth_reader(path: Path, *, columns: list[str]) -> pd.DataFrame:
        nonlocal called
        called = True
        assert seal.is_file()
        assert runner._sha256(seal) == seal_sha
        assert columns == list(runner.TRUTH_COLUMNS)
        return truth.copy()

    monkeypatch.setattr(runner.pd, "read_parquet", truth_reader)
    evaluated = runner._attach_truth_after_seal(
        blind,
        seal_path=seal,
        seal_sha256=seal_sha,
        truth_path=tmp_path / "oof.parquet",
    )
    assert called
    assert len(evaluated) == runner.EXPECTED_ROWS


def test_paired_episode_bootstrap_is_deterministic_and_whole_case() -> None:
    rows = _metric_rows(baseline_column="incumbent_prediction", baseline_value=1.5)
    first = runner._paired_episode_bootstrap(
        rows, baseline="incumbent_prediction", replicates=5000, seed=20260824
    )
    second = runner._paired_episode_bootstrap(
        rows, baseline="incumbent_prediction", replicates=5000, seed=20260824
    )
    assert first == second
    assert first["episodes"] == 3
    assert first["replicates"] == 5000
    assert first["ci90_upper_m"] < 0.0


def test_source_gate_requires_every_held_year_long_leads_ci_and_coverage() -> None:
    rows = []
    for year in runner.HELD_YEARS:
        for lead in runner.LEADS:
            rows.append(
                {
                    "year": year,
                    "episode_id": year,
                    "lead_h": lead,
                    "source_future_hs": 1.0,
                    "transfer_prediction": 1.0,
                    "persistence": 1.5,
                }
            )
    passed = runner._source_gate(pd.DataFrame(rows), {name: 1.0 for name in runner.EXTERNAL_COLUMN_MAP.values()})
    assert passed["passed"]
    failed = runner._source_gate(pd.DataFrame(rows), {"hs": 0.994})
    assert not failed["passed"]
    assert not failed["checks"]["finite_coverage_at_least_0_995"]


def test_solution_and_viewpoint_gates_are_separate() -> None:
    incumbent_rows = _metric_rows(
        baseline_column="incumbent_prediction", baseline_value=1.5
    )
    incumbent_rows["local_control_prediction"] = 1.2
    solution = runner._local_gate(incumbent_rows)
    viewpoint = runner._viewpoint_gate(incumbent_rows)
    assert solution["passed"]
    assert viewpoint["passed"]
    assert viewpoint["comparison"] == "transfer_minus_matched_local_only_control"


def test_all_three_fold_predictions_use_independent_transfer_clones_and_controls() -> None:
    feature_columns = list(runner.common_feature_columns())
    validation_ids = np.arange(1, runner.EXPECTED_CASES + 1, dtype=np.int64)
    train_ids = np.array([1001, 1002], dtype=np.int64)
    all_ids = np.concatenate([validation_ids, train_ids])
    features = pd.DataFrame(0.0, index=np.arange(len(all_ids)), columns=feature_columns)
    features.insert(0, "anchor_id", all_ids)
    anchors = pd.DataFrame(
        {
            "anchor_id": all_ids,
            "station": ["G-ORS"] * len(all_ids),
            "anchor_time": pd.date_range("2024-01-01", periods=len(all_ids), freq="78h", tz="UTC"),
            "current_hs": np.ones(len(all_ids)),
        }
    )
    sizes = (49, 79, 53)
    names = ("2024_h2_storm", "winter_transition", "2025_h1")
    folds = []
    keys = []
    offset = 0
    for name, size in zip(names, sizes, strict=True):
        held = validation_ids[offset : offset + size]
        offset += size
        folds.append(
            runner.ForecastFold(
                name=name,
                train_ids=train_ids,
                validation_ids=held,
                validation_start=pd.Timestamp("2024-01-01", tz="UTC"),
                validation_end=pd.Timestamp("2030-01-01", tz="UTC"),
            )
        )
        keys.extend(
            {"fold": name, "anchor_id": int(value), "station": "G-ORS", "episode_id": int(value)}
            for value in held
        )

    clone_objects = []

    class TransferModel:
        def continue_local(self, *_: object, **__: object) -> TransferModel:
            return self

        def predict_hs(self, frame: pd.DataFrame, *, current_hs: np.ndarray) -> np.ndarray:
            return np.repeat((current_hs + 0.1)[:, None], len(runner.LEADS), axis=1)

    class Pretrained:
        def clone_pretrained(self) -> TransferModel:
            value = TransferModel()
            clone_objects.append(value)
            return value

    class Control:
        def fit(self, *_: object, **__: object) -> Control:
            return self

        def predict_hs(self, frame: pd.DataFrame, *, current_hs: np.ndarray) -> np.ndarray:
            return np.repeat((current_hs + 0.2)[:, None], len(runner.LEADS), axis=1)

    def target_reader(ids: np.ndarray) -> pd.DataFrame:
        assert set(ids) == set(train_ids)
        return pd.DataFrame(
            {
                "anchor_id": train_ids,
                "current_hs": [1.0, 1.0],
                **{column: [1.1, 1.1] for column in runner.TARGET_COLUMNS},
            }
        )

    observed = runner._produce_local_blind_predictions(
        pretrained=Pretrained(),
        folds=folds,
        features=features,
        anchors=anchors,
        keys=pd.DataFrame(keys),
        target_reader=target_reader,
        control_factory=Control,
    )
    assert len(observed) == runner.EXPECTED_ROWS
    assert len(clone_objects) == 3
    assert len({id(value) for value in clone_objects}) == 3
    assert {"transfer_prediction", "local_control_prediction"} <= set(observed)


def test_one_shot_attempt_lock_uses_exclusive_creation(tmp_path: Path) -> None:
    path = tmp_path / "attempt.lock"
    digest = runner._create_attempt_lock(path, "a" * 64)
    assert digest == runner._sha256(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["research_only"] is True
    assert payload["operational_prediction_allowed"] is False
    with pytest.raises(FileExistsError):
        runner._create_attempt_lock(path, "a" * 64)


def test_check_only_validates_canonical_inputs_without_writes_or_model_fit() -> None:
    observed = runner.check_only(ROOT)
    assert observed["passed"] is True
    assert observed["writes"] == 0
    assert observed["model_fits"] == 0
    assert observed["outcome_values_read"] == 0
    assert observed["common_feature_count"] == 286
    assert observed["validation_cases"] == 181
    assert observed["full_prefix_rows"] == 1086
    assert sum(item["validation_cases"] for item in observed["folds"].values()) == 181


def test_runner_source_contains_no_forbidden_operational_path_literal() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8").casefold()
    forbidden = ("test_context", "test_index", "sample_submission", "submissions/")
    assert not any(token in source for token in forbidden)
