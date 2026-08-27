from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import p2_restore.authoritative_nested_surrogate_execution as execution
from p2_restore.architecture_matched_stage_a_execution_v2 import RouterContext
from p2_restore.authoritative_nested_surrogate_conformance import build_prefix_plan
from p2_restore.features import FeatureTable
from p2_restore.max_rounds import MaxRoundRouterModel
from p2_restore.regime_gate import STATE_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    PROJECT_ROOT
    / "configs/experiments/p2_authoritative_nested_surrogate_execution_20260825_v1.json"
)
V2_CONFIG = (
    PROJECT_ROOT
    / "configs/experiments/p2_authoritative_nested_surrogate_execution_20260825_v2.json"
)


def _synthetic_plan():
    times = pd.date_range("2023-01-01", periods=500, freq="12h", tz="UTC")
    metadata = pd.DataFrame(
        [("SYNTHETIC", layer, time.isoformat()) for time in times for layer in range(1, 9)],
        columns=["station", "layer", "time"],
    )
    return build_prefix_plan(
        metadata,
        outer_fold="synthetic_outer",
        validation_start_kst="2023-08-01T00:00:00+09:00",
        validation_stop_kst="2023-09-01T00:00:00+09:00",
        fraction=1.0,
    )


def _table(rows: int, feature_count: int, *, lean: bool = False) -> FeatureTable:
    feature_names = [f"feature_{index}" for index in range(feature_count)]
    if lean:
        feature_names[0] = "temp_1_minus_5"
    frame = pd.DataFrame(
        {
            "station": "SYNTHETIC",
            "layer": np.resize(np.array([2, 3, 4]), rows),
            "time": pd.date_range("2024-01-01", periods=rows, freq="10min", tz="UTC").astype(str),
            "target": np.linspace(10.0, 11.0, rows),
            "baseline": np.linspace(9.8, 10.8, rows),
            "residual": 0.2,
        }
    )
    for index, name in enumerate(feature_names):
        frame[name] = np.linspace(0.0, 1.0 + index / 10.0, rows)
    return FeatureTable(frame, tuple(feature_names))


def test_config_declares_actual_dag_not_900_fits() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    graph = config["execution_graph"]
    assert graph["top_level_component_jobs"] == 900
    assert graph["underlying_deep_fits"] == 720
    assert graph["underlying_lightgbm_fits"] == 720
    assert graph["underlying_base_estimator_fits"] == 1440
    assert graph["meta_optimizations"] == 405
    assert not config["permissions"]["actual_45_cell_fit_authorized"]


def test_v2_overlay_pins_atomic_publish_and_exact_v2_command_namespace() -> None:
    config = json.loads(V2_CONFIG.read_text(encoding="utf-8"))
    assert config["base_config"] == {
        "path": "configs/experiments/p2_authoritative_nested_surrogate_execution_20260825_v1.json",
        "sha256": "9de00eb2dab48d80f0f3342be98ec5a6b0395bfb4ae2d98665acd09553367acd",
    }
    assert config["supersedes"]["status"] == (
        "SUPERSEDED_ROBUSTNESS_CAVEAT_DO_NOT_AUTHORIZE"
    )
    assert config["robustness_revision"]["crash_resume_test_required"] is True
    command = config["exact_command"]
    assert (
        '--config "configs\\experiments\\'
        'p2_authoritative_nested_surrogate_execution_20260825_v2.json"'
    ) in command
    assert (
        '--preexecution-seal "artifacts\\'
        'p2_authoritative_nested_surrogate_execution_ready_20260825_v2\\'
        'preexecution_seal.json"'
    ) in command
    assert (
        '--authorization "artifacts\\'
        'p2_authoritative_nested_surrogate_execution_ready_20260825_v2\\'
        'EXECUTION_AUTHORIZATION.json"'
    ) in command
    assert "execution_ready_20260825_v1\\preexecution_seal.json" not in command


def test_superseded_v1_preexecution_seal_fails_closed_on_current_sources() -> None:
    base = json.loads(CONFIG.read_text(encoding="utf-8"))
    overlay = json.loads(V2_CONFIG.read_text(encoding="utf-8"))
    seal = PROJECT_ROOT / overlay["supersedes"]["readiness_directory"] / "preexecution_seal.json"
    runner = PROJECT_ROOT / "scripts/run_p2_authoritative_nested_surrogate_45cell_v1.py"
    with pytest.raises(ValueError, match="preexecution (module|runner) pin changed"):
        execution.verify_preexecution_seal(
            seal,
            config_sha256=overlay["base_config"]["sha256"],
            module_sha256=execution.sha256_file(Path(execution.__file__).resolve()),
            runner_sha256=execution.sha256_file(runner),
            exact_command=base["exact_command"],
        )


def test_exact_checkpoint_key_accepts_sub_micro_improvement_and_earliest_tie() -> None:
    rows = [
        execution.exact_checkpoint_key(0.5, 2),
        execution.exact_checkpoint_key(0.4999995, 4),
        execution.exact_checkpoint_key(0.4999995, 6),
    ]
    assert min(rows) == (0.4999995, 4)
    with pytest.raises(ValueError, match="checkpoint key"):
        execution.exact_checkpoint_key(float("nan"), 1)


def test_job_store_reuses_only_hash_verified_complete_job(tmp_path: Path) -> None:
    calls = 0

    def factory() -> execution.JobProduct:
        nonlocal calls
        calls += 1
        return execution.JobProduct(
            pd.DataFrame(
                {
                    "station": ["SYNTHETIC"],
                    "layer": [2],
                    "time": ["2024-01-01T00:00:00+00:00"],
                    "truth": [1.0],
                    "prediction": [1.1],
                }
            ),
            {"status": "complete"},
            {"payload.bin": b"checkpoint"},
        )

    first = execution.JobStore(tmp_path / "jobs", contract_sha256="contract")
    product = first.materialize("job_1", factory)
    assert calls == 1 and first.new_jobs == 1
    resumed = execution.JobStore(tmp_path / "jobs", contract_sha256="contract")
    assert resumed.materialize("job_1", factory).frame.equals(product.frame)
    assert calls == 1 and resumed.reused_jobs == 1
    (tmp_path / "jobs/job_1/receipt.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash changed"):
        execution.JobStore(tmp_path / "jobs", contract_sha256="contract").materialize(
            "job_1", factory
        )
    assert calls == 1


def test_process_lock_rejects_second_execution(tmp_path: Path) -> None:
    lock = tmp_path / "execution.lock"
    with execution.process_lock(lock):
        with pytest.raises(RuntimeError, match="holds the lock"):
            with execution.process_lock(lock):
                raise AssertionError("second lock must never be acquired")


def test_atomic_evaluated_oof_publish_survives_crash_and_resume(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "evaluated_oof_100.parquet"
    payload = b"synthetic-parquet-bytes"
    original_rename = execution.os.rename

    def crash_before_commit(source, destination):
        raise OSError(f"simulated crash: {source} -> {destination}")

    monkeypatch.setattr(execution.os, "rename", crash_before_commit)
    with pytest.raises(OSError, match="simulated crash"):
        execution.atomic_write_or_verify(target, payload)
    assert not target.exists()
    failed_partials = sorted(tmp_path.glob(".evaluated_oof_100.parquet.partial.*"))
    assert len(failed_partials) == 1
    assert failed_partials[0].read_bytes() == payload

    monkeypatch.setattr(execution.os, "rename", original_rename)
    committed = execution.atomic_write_or_verify(target, payload)
    assert committed["status"] == "COMMITTED_BY_FSYNC_AND_ATOMIC_RENAME"
    assert target.read_bytes() == payload
    assert sorted(tmp_path.glob(".evaluated_oof_100.parquet.partial.*")) == failed_partials
    reused = execution.atomic_write_or_verify(target, payload)
    assert reused["status"] == "REUSED_VERIFIED_FINAL"
    assert reused["partial_created"] is False

    target.write_bytes(b"corrupt-final")
    with pytest.raises(ValueError, match="atomic final (size|hash) changed"):
        execution.atomic_write_or_verify(target, payload)
    assert sorted(tmp_path.glob(".evaluated_oof_100.parquet.partial.*")) == failed_partials


def test_tiny_full_cell_proves_twenty_job_resume_without_fit() -> None:
    receipt = execution.temporary_tiny_fixture(_synthetic_plan())
    assert receipt["status"] == "PASS_TINY_FULL_CELL_AND_RESUME"
    assert receipt["synthetic_component_callbacks"] == 20
    assert receipt["second_pass_reused_jobs"] == 20
    assert receipt["second_pass_callbacks"] == 0
    assert receipt["actual_model_fits"] == 0


def test_router_is_four_estimator_400_round_graph_with_four_threads(monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    class FakeEstimator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.n_estimators = int(kwargs["n_estimators"])
            captured.append(kwargs)

        def fit(self, inputs, target):
            self.fit_shape = (inputs.shape, target.shape)
            return self

        def predict(self, inputs, *, num_iteration=None):
            assert num_iteration in (None, 400)
            return np.zeros(len(inputs), dtype=float)

    monkeypatch.setattr(execution, "LGBMRegressor", FakeEstimator)
    rows = 360
    base = _table(rows, 41)
    lean = _table(rows, 61, lean=True)
    phase = _table(rows, 81)
    state = pd.DataFrame({name: np.linspace(0.0, 1.0, rows) for name in STATE_FEATURES})
    state.insert(0, "time", base.frame["time"])
    state.insert(1, "layer", base.frame["layer"])
    context = RouterContext(
        base,
        lean,
        phase,
        state,
        np.ones(rows, dtype=bool),
        pd.DatetimeIndex(pd.to_datetime(base.frame["time"], utc=True)),
    )
    model = execution._fit_router_model_4threads(
        context,
        seed=101,
        layer_arms={"2": "phase", "3": "phase", "4": "state"},
    )
    assert isinstance(model, MaxRoundRouterModel)
    assert len(captured) == 4
    assert [item["random_state"] for item in captured] == [101, 101, 202, 303]
    assert all(item["n_estimators"] == 400 for item in captured)
    assert all(item["n_jobs"] == 4 for item in captured)
    predicted = execution._predict_router(model, context, fold="synthetic")
    assert len(predicted) == rows
    assert np.isfinite(predicted["prediction"]).all()


def test_causal_contract_maps_to_implementation_defaults_and_fixed_guards() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    spec = execution._causal_spec(config)
    assert spec.public_layers == (1, 5, 6, 7, 8)
    assert (spec.rolling_hours, spec.minimum_samples) == (24, 72)
    assert (spec.minimum_anchors, spec.ridge_slope_lambda) == (4, 10.0)
    assert (spec.correction_scale, spec.correction_clip_c) == (0.25, 0.125)


def test_authorization_is_fail_closed_and_does_not_expand_scope(tmp_path: Path) -> None:
    command = "sealed command"
    path = tmp_path / "authorization.json"
    value = {
        "status": "APPROVED_EXACT_P2_45_CELL_COMMAND",
        "training_authorized": True,
        "preexecution_seal_sha256": "seal",
        "exact_command_sha256": execution.hashlib.sha256(command.encode()).hexdigest(),
        "official_test_access_authorized": False,
        "sample_submission_access_authorized": False,
        "submission_generation_authorized": False,
        "public_score_selection_authorized": False,
        "upload_authorized": False,
        "p3_process_mutation_authorized": False,
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    assert execution.verify_authorization(
        path,
        preexecution_seal_sha256="seal",
        exact_command=command,
    )["training_authorized"]
    value["official_test_access_authorized"] = True
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="expands forbidden scope"):
        execution.verify_authorization(
            path,
            preexecution_seal_sha256="seal",
            exact_command=command,
        )
