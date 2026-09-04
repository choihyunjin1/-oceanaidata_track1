from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_p1_long_event_segment_proposal_rescore_reactivation_20260901_v1.py"
CONFIG = ROOT / "configs/experiments/p1_long_event_segment_proposal_rescore_reactivation_20260901_v1.json"


def _load():
    specification = importlib.util.spec_from_file_location("p1_segment_reactivation", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_contract_is_fresh_same_process_and_science_frozen() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["experiment_id"] == "p1_long_event_segment_proposal_rescore_reactivation_20260901_v1"
    assert payload["scientific_experiment_id"] == "p1_long_event_segment_proposal_rescore_20260826_v1"
    assert payload["execution_contract"] == {
        "same_process": True,
        "hidden_worker": False,
        "parent_process_capability": False,
        "subprocess_launches": 0,
        "single_attempt": True,
        "automatic_resume_or_rerun_count": 0,
        "failure_retains_lock": True,
        "success_removes_lock_as_final_commit": True,
    }
    assert payload["frozen_operation_graph"]["maximum_lifetime_physical_fits"] == 72
    assert payload["frozen_operation_graph"]["scientific_materializations"] == 21
    assert payload["unchanged_science"]["decision_gates"] is True
    assert payload["unchanged_science"]["result_driven_tuning_or_search_expansion"] is False


def test_later_long_event_family_is_not_semantic_duplicate() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    overlap = payload["family_overlap_adjudication"]
    assert overlap["dependence_calibration_meta_audit"]["semantic_duplicate"] is False
    rescue = overlap["p1_e150_f1aware_long_event_cp_rescue_20260828_v1"]
    assert rescue["terminal_status"] == "NO_GO_LOCAL_BLIND_GATE"
    assert rescue["exact_duplicate"] is False
    assert rescue["semantic_duplicate"] is False
    assert rescue["gate_reuse_or_mutation"] is False


def test_launcher_has_no_hidden_worker_parent_auth_or_subprocess() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "--worker" not in source
    assert "import subprocess" not in source
    assert "from subprocess" not in source
    assert "ctypes" not in source
    assert "msvcrt" not in source
    assert "run_authorized_screen(state, numerical, closure, journal, deadline)" in source


def test_source_guard_opens_only_readme_and_train(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load()
    readme = tmp_path / "README.md"
    train = tmp_path / "train.csv"
    forbidden = tmp_path / "test.csv"
    readme.write_bytes(b"approved readme")
    train.write_bytes(b"timestamp,value\n2024-01-01,1\n")
    forbidden.write_bytes(b"must not be read")

    module.EXPECTED_SOURCE_ROOT = tmp_path
    module.EXPECTED_README_BYTES = readme.stat().st_size
    module.EXPECTED_TRAIN_BYTES = train.stat().st_size
    module.EXPECTED_README_SHA256 = hashlib.sha256(readme.read_bytes()).hexdigest()
    module.EXPECTED_TRAIN_SHA256 = hashlib.sha256(train.read_bytes()).hexdigest()
    monkeypatch.setenv("P1_DATA_DIR", str(tmp_path))

    opened: list[str] = []
    original_open = Path.open

    def tracked_open(path: Path, *args, **kwargs):
        opened.append(path.name)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    receipt = module._validate_source(include_readme=True)
    assert sorted(opened) == ["README.md", "train.csv"]
    assert receipt["official_test_reads"] == 0
    assert forbidden.read_bytes() == b"must not be read"


def test_exact_fit_and_materialization_ledgers(tmp_path: Path) -> None:
    module = _load()
    module.ARTIFACT_DIR = tmp_path / "fresh_namespace"
    journal = module.AttemptJournal(time.time() + 60)
    selected_outer_cell = module.STRUCTURE_CELLS[0]

    for ordinal in range(1, 73):
        phase, window, cell, seed = journal._expected_fit(ordinal)
        if cell is None:
            cell = selected_outer_cell
        reserved = journal.reserve_fit(phase, window, cell, seed)
        journal.complete_fit(reserved)

    labels = [f"inner_anchor_surface:{window}" for window in module.INNER_WINDOWS]
    labels.extend(
        f"inner_context_surface:{window}:{bank}"
        for window in module.INNER_WINDOWS
        for bank in module.CONTEXT_BANKS
    )
    labels.extend(
        f"outer_context_surface:{fold}:{bank}"
        for fold in module.OUTER_FOLDS
        for bank in module.CONTEXT_BANKS
    )
    for label in labels:
        journal.reserve_materialization(label)

    assert (journal.fit_reservations, journal.fits_completed, journal.materializations) == (72, 72, 21)
    assert journal.lock_path.exists()
    assert not list(module.ARTIFACT_DIR.glob("*.csv"))


def test_two_zero_operation_preflights_are_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load()
    module.ARTIFACT_DIR = Path("Z:/definitely_absent_reactivation_namespace")
    frozen = {
        "readiness": {"status": "PASS", "binding": "frozen"},
        "snapshot_static_inventory": {"input": {"bytes": 1, "sha256": "0" * 64}},
        "operation_counters": {
            "claims": 0,
            "physical_fits": 0,
            "scientific_materializations": 0,
            "outer_scores": 0,
            "candidate_files": 0,
            "official_test_reads": 0,
            "sample_format_reads": 0,
            "submission_candidate_reads": 0,
            "uploads": 0,
        },
    }
    source = {
        "source_root": "approved",
        "directory_enumerations": 0,
        "train": {"filename": "train.csv", "bytes": 1, "sha256": "1" * 64},
        "readme": {"filename": "README.md", "bytes": 1, "sha256": "2" * 64},
        "official_test_reads": 0,
        "sample_submission_reads": 0,
        "submission_candidate_reads": 0,
    }
    helper = SimpleNamespace(_complete_readiness=lambda retain_snapshot: (frozen, None, None))
    monkeypatch.setattr(module, "_validate_config", lambda: {})
    monkeypatch.setattr(module, "_validate_source", lambda include_readme: source)
    monkeypatch.setattr(module, "_load_v6", lambda: helper)

    first, *_ = module.prepare(retain_snapshot=False)
    second, *_ = module.prepare(retain_snapshot=False)
    assert first == second
    assert all(value == 0 for value in first["operation_counters"].values())
