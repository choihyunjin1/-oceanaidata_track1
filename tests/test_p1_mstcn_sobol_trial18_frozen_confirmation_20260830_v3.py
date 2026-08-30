from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_mstcn_sobol_trial18_frozen_confirmation_20260830_v3"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _load(
        f"{EXPERIMENT_ID}_runner_test",
        ROOT / "scripts" / f"run_{EXPERIMENT_ID}.py",
    )


@pytest.fixture(scope="module")
def qa_module():
    return _load(
        f"{EXPERIMENT_ID}_qa_test",
        ROOT / "scripts" / f"qa_{EXPERIMENT_ID}.py",
    )


def test_preregistered_cell_is_exact_and_metric_aligned(runner) -> None:
    config = runner._config(root=ROOT)
    recipe = config["frozen_recipe"]
    assert recipe["trial"]["trial_id"] == "trial_18"
    assert recipe["trial"]["trial_index"] == 18
    assert recipe["trial"]["width"] == 512
    assert recipe["trial"]["batch_size"] == 64
    assert recipe["threshold"] == 0.8
    assert recipe["epoch"] == 150
    assert recipe["seeds"] == [20260827, 20260839, 20260863]
    primary = config["evaluation_contract"]["level_1_primary"]
    assert primary["metric"] == "pooled Q3+Q4 row-level binary micro-F1"
    assert primary["directional_margin"] == 0.0
    assert primary["arbitrary_positive_delta_margin"] is None
    assert config["confirmation_contract"]["maximum_lifetime_fit_count"] == 6
    assert config["selection_lineage"]["q2_search_or_threshold_replay_authorized"] is False


def test_selection_lineage_is_fixed_margin_only_and_untouched(runner) -> None:
    config = runner._config(root=ROOT)
    lineage = runner._verify_selection_lineage(config, root=ROOT)
    assert lineage["selected_trial_id"] == "trial_18"
    assert lineage["selected_threshold"] == 0.8
    assert lineage["selected_epoch"] == 150
    assert lineage["legacy_q2_pooled_delta_f1"] == pytest.approx(0.0005656370384116149)
    assert lineage["legacy_all_months_positive"] is True
    assert lineage["legacy_fixed_delta_gate_passed"] is False
    assert lineage["legacy_q3_q4_training_started"] is False
    assert lineage["legacy_confirmatory_artifacts_absent"] is True
    assert lineage["q2_search_replayed"] is False
    assert lineage["threshold_search_replayed"] is False


def test_evidence_state_uses_point_and_uncertainty_only(runner) -> None:
    assert (
        runner.classify_evidence_state(
            delta_f1=0.001, ci90_lower=0.0001, ci90_upper=0.002, level_0_pass=True
        )
        == "HIGH_VALUE_CHALLENGER_RESEARCH_ONLY"
    )
    assert (
        runner.classify_evidence_state(
            delta_f1=0.001, ci90_lower=-0.001, ci90_upper=0.002, level_0_pass=True
        )
        == "EXPLORATORY_CHALLENGER_RESEARCH_ONLY"
    )
    assert (
        runner.classify_evidence_state(
            delta_f1=-0.001, ci90_lower=-0.002, ci90_upper=-0.0001, level_0_pass=True
        )
        == "PRIMARY_HARM_RESEARCH_ONLY"
    )
    assert (
        runner.classify_evidence_state(
            delta_f1=-0.001, ci90_lower=-0.002, ci90_upper=0.0001, level_0_pass=True
        )
        == "INCONCLUSIVE_RESEARCH_ONLY"
    )
    assert (
        runner.classify_evidence_state(
            delta_f1=0.1, ci90_lower=0.01, ci90_upper=0.2, level_0_pass=False
        )
        == "QA_BLOCKED"
    )


def test_terminal_writer_is_exclusive_and_never_overwrites(runner, tmp_path: Path) -> None:
    path = tmp_path / "terminal_result.json"
    runner._exclusive_json(path, {"status": "FIRST"})
    first = path.read_bytes()
    with pytest.raises(FileExistsError):
        runner._exclusive_json(path, {"status": "SECOND"})
    assert path.read_bytes() == first
    assert json.loads(first)["status"] == "FIRST"


def test_manifest_excludes_itself_and_terminal(runner, tmp_path: Path) -> None:
    (tmp_path / "payload.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "terminal_result.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "artifact_manifest.json").write_text("{}\n", encoding="utf-8")
    manifest = runner._manifest(tmp_path)
    assert [row["path"] for row in manifest["files"]] == ["payload.json"]


def test_independent_binary_metric_formula(qa_module) -> None:
    metrics = qa_module._binary_metrics(
        np.asarray([1, 1, 1, 0, 0, 0], dtype=np.int8),
        np.asarray([1, 1, 0, 1, 0, 0], dtype=np.int8),
    )
    assert metrics["tp"] == 2
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["f1"] == pytest.approx(2.0 / 3.0)


def test_smoke_is_synthetic_and_creates_no_attempt_lock(runner) -> None:
    before = (ROOT / "artifacts" / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json").exists()
    result = runner.run_smoke()
    after = (ROOT / "artifacts" / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json").exists()
    assert result["result"] == "PASS"
    assert result["real_input_rows_read"] == 0
    assert result["attempt_lock_created"] is False
    assert after == before


def test_source_has_no_official_path_resolution_or_csv_write() -> None:
    runner_source = (ROOT / "scripts" / f"run_{EXPERIMENT_ID}.py").read_text(encoding="utf-8")
    qa_source = (ROOT / "scripts" / f"qa_{EXPERIMENT_ID}.py").read_text(encoding="utf-8")
    combined = runner_source + qa_source
    assert "P1_DATA_DIR" not in combined
    assert "test.csv" not in combined
    assert "sample_submission.csv" not in combined
    assert ".to_csv(" not in combined
    assert "read_csv(" not in combined
    assert "requests." not in combined
    assert "subprocess" not in combined


def test_config_prohibits_retry_outlier_deletion_and_outputs() -> None:
    config = json.loads(
        (ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json").read_text(encoding="utf-8")
    )
    assert all(config["prohibitions"].values())
    assert config["output_contract"]["prediction_csv"] is None
    assert config["output_contract"]["checkpoint_storage"] is None
    assert config["prohibitions"]["outlier_hard_deletion"] is True
    assert config["prohibitions"]["label_1_or_anomaly_event_deletion"] is True
