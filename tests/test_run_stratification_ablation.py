from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from p1_qc.config import FeatureConfig, P1QCConfig, PathsConfig
from p1_qc.experiment import sha256_file
from p1_qc.features import FeatureBundle
from scripts import run_stratification_ablation as runner


def test_fixed_runner_calls_cv_once_and_records_hashed_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    project = tmp_path / "project"
    config_path = project / "configs" / "p1.toml"
    data_dir = tmp_path / "immutable_input"
    run_root = tmp_path / "runs"
    status_path = tmp_path / "status" / "progress.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text('[project]\nmode = "offline"\n', encoding="utf-8")
    data_dir.mkdir()
    for name in ("train.csv", "test.csv"):
        (data_dir / name).write_text("aggregate-only-test\n", encoding="utf-8")

    config = P1QCConfig(
        paths=PathsConfig(data_dir=data_dir, artifacts_dir=run_root),
        features=FeatureConfig(mode="offline"),
    )
    train = pd.DataFrame({"label": [0, 1]})
    test = pd.DataFrame({"station": ["SYNTH"], "layer": [1]})
    base = FeatureBundle(
        pd.DataFrame({"base": [0.0, 1.0]}),
        ("base",),
        (),
    )
    candidate_columns = (*base.feature_columns, *runner.PEER_GATE_FEATURES)
    candidate = FeatureBundle(
        pd.DataFrame({column: [0.0, 1.0] for column in candidate_columns}),
        candidate_columns,
        (),
    )
    oof = pd.DataFrame({"label": [0, 1], "prediction": [0, 1]})
    metrics = {
        "aggregate": {
            "micro": {"f1": 0.8},
            "weighted": {"f1": 0.75},
        }
    }
    selection = {"feature_hash": "base-only"}
    cv_calls: list[dict[str, object]] = []
    gate_calls: list[object] = []
    phases: list[tuple[str, int]] = []

    monkeypatch.setattr(runner, "PROJECT_ROOT", project)
    monkeypatch.setenv("P1_DATA_DIR", str(data_dir))
    monkeypatch.setattr(runner, "load_config", lambda path, env: config)
    monkeypatch.setattr(runner, "resolve_data_dir", lambda value: data_dir)
    monkeypatch.setattr(
        runner,
        "load_train_test",
        lambda path, audit, strict: (train, test),
    )

    def fake_load_features(frame, value, *, kind, use_cache):
        assert frame is train
        assert value is config
        assert kind == "train"
        assert use_cache is True
        return base

    monkeypatch.setattr(runner, "load_or_build_features", fake_load_features)

    def fake_append(bundle, source, *, config, cadence_minutes, group_columns):
        assert bundle is base
        assert source is train
        gate_calls.append(config)
        assert config == runner.PeerGateConfig(
            mode="offline", window_hours=24, min_period_fraction=0.5
        )
        assert cadence_minutes == 10
        assert tuple(group_columns) == ("station", "layer")
        return candidate

    monkeypatch.setattr(runner, "append_stratification_peer_gate", fake_append)

    def fake_cv(train_arg, test_arg, bundle_arg, config_arg, **kwargs):
        cv_calls.append(kwargs)
        assert train_arg is train
        assert test_arg is test
        assert bundle_arg is candidate
        assert config_arg is config
        return oof, metrics, selection

    monkeypatch.setattr(runner, "run_cross_validation", fake_cv)
    original_status = runner._write_status

    def capture_status(path, **kwargs):
        phases.append((kwargs["phase"], kwargs["progress"]))
        original_status(path, **kwargs)

    monkeypatch.setattr(runner, "_write_status", capture_status)
    import p1_qc.experiment as experiment

    monkeypatch.setattr(experiment, "environment_summary", lambda: {"test": True})
    monkeypatch.setattr(experiment, "seed_everything", lambda seed: None)

    assert runner.main(["--status-file", str(status_path)]) == 0
    assert len(gate_calls) == 1
    assert cv_calls == [
        {
            "backend": "xgboost",
            "bootstrap_replicates": 2000,
            "augmentation": False,
        }
    ]
    assert phases == [
        ("start", 25),
        ("feature", 35),
        ("cv", 70),
        ("save", 90),
        ("done", 100),
    ]
    final_status = json.loads(status_path.read_text(encoding="utf-8"))
    assert final_status["status"] == "complete"
    assert final_status["progress"] == 100
    assert not status_path.with_suffix(".json.tmp").exists()

    run_directories = [path for path in run_root.iterdir() if path.is_dir()]
    assert len(run_directories) == 1
    run_dir = run_directories[0]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["command"] == "strat_gate_fixed24h"
    assert manifest["status"] == "complete"
    for name in ("oof.parquet", "metrics.json", "selection.json", "feature_gate.json"):
        path = run_dir / name
        assert path.is_file()
        assert manifest["artifacts"][name]["sha256"] == sha256_file(path)
    gate_metadata = json.loads((run_dir / "feature_gate.json").read_text(encoding="utf-8"))
    assert gate_metadata["feature_count"] == 4
    assert gate_metadata["outer_labels_used_for_gate_configuration"] is False
    saved_selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    assert saved_selection["feature_hash"] == gate_metadata["feature_contract_sha256"]
    assert (
        saved_selection["fixed_gate_ablation"]["promotion_decision_performed_by_this_run"] is False
    )
