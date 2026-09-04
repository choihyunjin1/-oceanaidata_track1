from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from p1_qc.experiment import sha256_file
from scripts.run_p1_iors_external_point_residual_oof import (
    CANONICAL_EXPERIMENT_CONFIG,
    GLOBAL_EXPOSURE_LEDGER,
    GLOBAL_EXPOSURE_LOCK,
    _exclusive_json_fsync,
    _implementation_hashes,
    _noop_fold_byte_audit,
    _require_hardcoded_canonical_paths,
    _verify_incumbent_oof_sha,
    _write_parquet_fsync,
    parse_args,
)


def test_exposure_lock_uses_atomic_exclusive_creation(tmp_path: Path) -> None:
    lock = tmp_path / "outer_exposure.lock"

    _exclusive_json_fsync(lock, {"first": True})

    with pytest.raises(FileExistsError):
        _exclusive_json_fsync(lock, {"second": True})
    assert json.loads(lock.read_text(encoding="utf-8")) == {"first": True}


def test_blind_parquet_is_fsynced_hashable_and_reloadable(tmp_path: Path) -> None:
    path = tmp_path / "blind.parquet"
    frame = pd.DataFrame({"value": [1, 2, 3]})

    _write_parquet_fsync(path, frame)
    before = sha256_file(path)
    reloaded = pd.read_parquet(path)
    after = sha256_file(path)

    assert before == after
    pd.testing.assert_frame_equal(frame, reloaded)


def test_noop_fold_audit_requires_exact_incumbent_bytes() -> None:
    reference = pd.DataFrame(
        {
            "station": ["I-ORS"],
            "year": [2025],
            "layer": [1],
            "time": ["2025-04-01T00:00:00+09:00"],
            "fold": ["2025_q2"],
            "prediction": pd.Series([0], dtype="int8"),
            "probability": pd.Series([0.25], dtype="float32"),
        }
    )
    blind = reference.loc[:, ["station", "year", "layer", "time", "fold"]].copy()
    blind["candidate_prediction"] = reference["prediction"].copy()
    blind["candidate_probability"] = reference["probability"].copy()

    audit = _noop_fold_byte_audit(reference, blind, ["2025_q2"])

    assert audit["2025_q2"]["byte_identical"] is True
    changed = blind.copy()
    changed["candidate_probability"] = pd.Series([0.5], dtype="float32")
    with pytest.raises(AssertionError, match="probability bytes changed"):
        _noop_fold_byte_audit(reference, changed, ["2025_q2"])


def test_implementation_seal_contains_all_required_paths() -> None:
    hashes = _implementation_hashes(
        Path("configs/experiments/p1_iors_external_point_residual_oof_v1.json").resolve()
    )

    assert {
        "experiment_config",
        "runner",
        "point_residual_helper",
        "iors_ctd",
        "pipeline",
        "models_tabular",
    }.issubset(hashes)
    assert all(len(value) == 64 for value in hashes.values())


def test_copied_experiment_config_cannot_relocate_one_shot_lock(tmp_path: Path) -> None:
    copied_config = tmp_path / "copied.json"
    copied_config.write_bytes(CANONICAL_EXPERIMENT_CONFIG.read_bytes())
    copied_args = parse_args(["--experiment-config", str(copied_config)])

    with pytest.raises(ValueError, match="hardcoded one-shot canonical path mismatch"):
        _require_hardcoded_canonical_paths(copied_args)

    moved_output_args = parse_args(["--output-dir", str(tmp_path / "output")])
    with pytest.raises(ValueError, match="hardcoded one-shot canonical path mismatch"):
        _require_hardcoded_canonical_paths(moved_output_args)


def test_global_exposure_paths_are_independent_of_experiment_config() -> None:
    assert GLOBAL_EXPOSURE_LOCK.name == "p1_iors_external_point_residual_oof_v1.lock"
    assert GLOBAL_EXPOSURE_LEDGER.name == "one_shot_exposure_ledger.jsonl"
    assert "configs" not in GLOBAL_EXPOSURE_LOCK.parts


def test_incumbent_oof_sha_is_rechecked_fail_closed(tmp_path: Path) -> None:
    incumbent = tmp_path / "incumbent.parquet"
    incumbent.write_bytes(b"frozen")
    contract = {
        "p1_reference": {
            "incumbent_oof": str(incumbent),
            "incumbent_oof_sha256": sha256_file(incumbent),
        }
    }

    assert _verify_incumbent_oof_sha(contract) == sha256_file(incumbent)
    incumbent.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="incumbent OOF changed"):
        _verify_incumbent_oof_sha(contract)
