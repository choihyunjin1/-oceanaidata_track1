"""Fresh same-process one-shot execution of the frozen P1 segment-rescore science.

The runner deliberately has no hidden-worker or parent-capability entry point.
Read-only preflight writes only canonical JSON to stdout. Execution uses a new
create-only namespace and never reads official test, sample, or submission data.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import sys
import time
import uuid
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiments/p1_long_event_segment_proposal_rescore_reactivation_20260901_v1.json"
V6_RUNNER_PATH = ROOT / "scripts/run_p1_long_event_segment_proposal_rescore_v6.py"
ARTIFACT_DIR = ROOT / "artifacts/p1_long_event_segment_proposal_rescore_reactivation_20260901_v1"

EXPERIMENT_ID = "p1_long_event_segment_proposal_rescore_reactivation_20260901_v1"
SCIENTIFIC_EXPERIMENT_ID = "p1_long_event_segment_proposal_rescore_20260826_v1"
HARD_WALL_SECONDS = 21600
MAXIMUM_FITS = 72
MAXIMUM_MATERIALIZATIONS = 21
ROUND_B_SEEDS = (20260813, 20260829, 20260847)
SEGMENT_SEEDS = (20260826, 20260843, 20260871)
INNER_WINDOWS = ("inner_2024_jul_aug", "inner_2024_oct_nov", "inner_2025_jan_feb")
OUTER_FOLDS = ("2025_q2", "2025_q3", "2025_q4")
CONTEXT_BANKS = ("24_72", "48_168", "24_72_168")
DECODERS = ("connected_only", "dual_boundary_disconnected_allowed")
STRUCTURE_CELLS = tuple(f"bank_{bank}__{decoder}" for bank in CONTEXT_BANKS for decoder in DECODERS)
EXPECTED_HELPER_SHA256 = "001109137fa4daa5408977d14473d95639ba51b8c70301e802626c60ea59509e"
EXPECTED_SOURCE_ROOT = Path("C:/Users/cedis/Downloads/데이터셋_P1/P1_qc_anomaly")
EXPECTED_README_SHA256 = "cb658f09cd3a19824bc9113d6a49568a24edebbd4ecd024554b4d3b37c87eafd"
EXPECTED_TRAIN_SHA256 = "20b656b0cbd524ad9da0bae8ecb6e0bacfc006e05810b37e83f29a5fa8e65cd2"
EXPECTED_README_BYTES = 1586
EXPECTED_TRAIN_BYTES = 50584654
PLANNING_POINTS_PER_F1 = 0.6778 / 0.0255


def _now_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path.name}")
    return value


def _atomic_create_bytes(path: Path, raw: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> Path:
    return _atomic_create_bytes(path, _canonical_bytes(value))


def _is_reparse(path: Path) -> bool:
    attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
    return bool(attributes & 0x400)


def _validate_config() -> dict[str, Any]:
    config = _read_json(CONFIG_PATH)
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("reactivation experiment identity changed")
    if config.get("scientific_experiment_id") != SCIENTIFIC_EXPERIMENT_ID:
        raise RuntimeError("frozen scientific identity changed")
    execution = config.get("execution_contract", {})
    if execution != {
        "same_process": True,
        "hidden_worker": False,
        "parent_process_capability": False,
        "subprocess_launches": 0,
        "single_attempt": True,
        "automatic_resume_or_rerun_count": 0,
        "failure_retains_lock": True,
        "success_removes_lock_as_final_commit": True,
    }:
        raise RuntimeError("same-process one-shot contract changed")
    graph = config.get("frozen_operation_graph", {})
    expected_graph = {
        "inner_anchor_physical_fits": 9,
        "inner_segment_physical_fits": 54,
        "outer_segment_physical_fits": 9,
        "maximum_lifetime_physical_fits": 72,
        "scientific_materializations": 21,
        "outer_scores": 1,
        "hard_wall_seconds": 21600,
        "candidate_files": 0,
        "row_level_prediction_files": 0,
    }
    if graph != expected_graph:
        raise RuntimeError("frozen operation graph changed")
    for item in config["immutable_scientific_authorities"].values():
        path = (ROOT / str(item["path"])).resolve(strict=True)
        if not path.is_relative_to(ROOT) or _sha256(path) != item["sha256"]:
            raise RuntimeError(f"frozen authority changed: {item['path']}")
    return config


def _validate_source(*, include_readme: bool) -> dict[str, Any]:
    raw = os.environ.get("P1_DATA_DIR")
    if not raw:
        raise RuntimeError("P1_DATA_DIR is required")
    source_root = Path(raw)
    if not source_root.is_absolute():
        raise RuntimeError("P1_DATA_DIR must be absolute")
    source_root = source_root.resolve(strict=True)
    if source_root != EXPECTED_SOURCE_ROOT.resolve(strict=True):
        raise RuntimeError("P1_DATA_DIR differs from the approved local source root")
    if not stat.S_ISDIR(source_root.lstat().st_mode) or _is_reparse(source_root):
        raise RuntimeError("P1_DATA_DIR must be a non-reparse directory")

    train = (source_root / "train.csv").resolve(strict=True)
    if train.parent != source_root or train.name != "train.csv":
        raise RuntimeError("only the direct historical train.csv is allowed")
    if not stat.S_ISREG(train.lstat().st_mode) or _is_reparse(train):
        raise RuntimeError("train.csv must be a non-reparse regular file")
    train_size_before = train.stat().st_size
    train_sha = _sha256(train)
    train_size_after = train.stat().st_size
    if (
        train_size_before != EXPECTED_TRAIN_BYTES
        or train_size_after != EXPECTED_TRAIN_BYTES
        or train_sha != EXPECTED_TRAIN_SHA256
    ):
        raise RuntimeError("historical train.csv binding changed")

    receipt: dict[str, Any] = {
        "source_root": str(source_root),
        "directory_enumerations": 0,
        "train": {"filename": "train.csv", "bytes": train_size_after, "sha256": train_sha},
        "official_test_reads": 0,
        "sample_submission_reads": 0,
        "submission_candidate_reads": 0,
    }
    if include_readme:
        readme = (source_root / "README.md").resolve(strict=True)
        if readme.parent != source_root or readme.name != "README.md":
            raise RuntimeError("source README path changed")
        readme_size = readme.stat().st_size
        readme_sha = _sha256(readme)
        if readme_size != EXPECTED_README_BYTES or readme_sha != EXPECTED_README_SHA256:
            raise RuntimeError("source README binding changed")
        receipt["readme"] = {"filename": "README.md", "bytes": readme_size, "sha256": readme_sha}
    return receipt


def _load_v6() -> ModuleType:
    if _sha256(V6_RUNNER_PATH) != EXPECTED_HELPER_SHA256:
        raise RuntimeError("frozen v6 readiness helper changed")
    specification = importlib.util.spec_from_file_location("p1_segment_reactivation_v6_helper", V6_RUNNER_PATH)
    if specification is None or specification.loader is None:
        raise RuntimeError("unable to load frozen v6 helper")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _public_preflight(v6_preflight: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    output = {
        "schema_version": "p1_long_event_segment_proposal_rescore.reactivation_preflight.v1",
        "experiment_id": EXPERIMENT_ID,
        "scientific_experiment_id": SCIENTIFIC_EXPERIMENT_ID,
        "status": "PASS_STRICT_SAME_PROCESS_ZERO_OPERATION_READINESS",
        "source": dict(source),
        "frozen_readiness": v6_preflight["readiness"],
        "snapshot_static_inventory": v6_preflight["snapshot_static_inventory"],
        "namespace": {
            "fresh_artifact_namespace": not ARTIFACT_DIR.exists(),
            "claim_created": False,
            "lock_created": False,
        },
        "execution_boundary": {
            "same_process": True,
            "hidden_worker": False,
            "parent_capability": False,
            "subprocess_launches": 0,
        },
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
    output["verification_sha256"] = _canonical_sha(output)
    return output


def prepare(*, retain_snapshot: bool) -> tuple[dict[str, Any], ModuleType, Path | None, dict[str, Any]]:
    if ARTIFACT_DIR.exists():
        raise FileExistsError("fresh reactivation namespace is already consumed")
    _validate_config()
    source = _validate_source(include_readme=True)
    v6 = _load_v6()
    base, snapshot, records = v6._complete_readiness(retain_snapshot=retain_snapshot)
    if base["operation_counters"] != {
        "claims": 0,
        "physical_fits": 0,
        "scientific_materializations": 0,
        "outer_scores": 0,
        "candidate_files": 0,
        "official_test_reads": 0,
        "sample_format_reads": 0,
        "submission_candidate_reads": 0,
        "uploads": 0,
    }:
        raise RuntimeError("frozen readiness is not zero-operation")
    if retain_snapshot and (snapshot is None or records is None):
        raise RuntimeError("retained readiness snapshot is absent")
    return _public_preflight(base, source), v6, snapshot, base


class AttemptJournal:
    """Minimal create-only journal implementing the frozen numerical interface."""

    def __init__(self, deadline_epoch: float) -> None:
        self.deadline_epoch = deadline_epoch
        self.attempt_id = uuid.uuid4().hex
        self.lock_path = ARTIFACT_DIR / "execution.lock"
        self.journal_dir = ARTIFACT_DIR / "attempt_journal"
        self.fit_reservations = 0
        self.fits_completed = 0
        self.materializations = 0
        self._sequence = 0
        self._selected_outer_cell: str | None = None
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=False)
        descriptor = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.write(
                descriptor,
                _canonical_bytes(
                    {
                        "experiment_id": EXPERIMENT_ID,
                        "attempt_id": self.attempt_id,
                        "pid": os.getpid(),
                        "deadline_epoch": deadline_epoch,
                        "created_at_kst": _now_kst(),
                    }
                ),
            )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self.journal_dir.mkdir()
        self._write("started", {"maximum_fits": 72, "maximum_materializations": 21})

    def _write(self, kind: str, payload: Mapping[str, Any]) -> Path:
        self._sequence += 1
        value = {
            "experiment_id": EXPERIMENT_ID,
            "attempt_id": self.attempt_id,
            "sequence": self._sequence,
            "kind": kind,
            "created_at_kst": _now_kst(),
            **dict(payload),
        }
        return _atomic_create_json(self.journal_dir / f"{self._sequence:04d}_{kind}.json", value)

    def record_readiness(self, readiness: Mapping[str, Any]) -> None:
        self._write(
            "readiness",
            {
                "status": readiness["status"],
                "verification_sha256": _canonical_sha(readiness),
                "official_test_reads": 0,
                "sample_format_reads": 0,
                "submission_candidate_reads": 0,
            },
        )

    def _expected_fit(self, ordinal: int) -> tuple[str, str, str | None, int]:
        if ordinal <= 9:
            zero = ordinal - 1
            return "INNER_ANCHOR", INNER_WINDOWS[zero // 3], "ROUND_B_SHARED", ROUND_B_SEEDS[zero % 3]
        if ordinal <= 63:
            zero = ordinal - 10
            per_window = len(STRUCTURE_CELLS) * len(SEGMENT_SEEDS)
            window = INNER_WINDOWS[zero // per_window]
            within = zero % per_window
            return "INNER_SEGMENT", window, STRUCTURE_CELLS[within // 3], SEGMENT_SEEDS[within % 3]
        zero = ordinal - 64
        return "OUTER_SEGMENT", OUTER_FOLDS[zero // 3], None, SEGMENT_SEEDS[zero % 3]

    def reserve_fit(self, phase: str, window: str, cell: str, seed: int) -> int:
        if time.time() >= self.deadline_epoch:
            raise TimeoutError("hard wall expired before fit reservation")
        ordinal = self.fit_reservations + 1
        if ordinal > MAXIMUM_FITS:
            raise RuntimeError("72-fit ceiling exceeded")
        expected = list(self._expected_fit(ordinal))
        if ordinal >= 64:
            if cell not in STRUCTURE_CELLS:
                raise RuntimeError("outer selected cell is not frozen")
            if self._selected_outer_cell is None:
                self._selected_outer_cell = cell
            expected[2] = self._selected_outer_cell
        if (phase, window, cell, int(seed)) != tuple(expected):
            raise RuntimeError(f"fit plan differs at ordinal {ordinal}")
        self._write(
            "fit_reserved",
            {"ordinal": ordinal, "phase": phase, "window_or_fold": window, "cell": cell, "seed": int(seed)},
        )
        self.fit_reservations = ordinal
        return ordinal

    def complete_fit(self, ordinal: int) -> None:
        if ordinal != self.fits_completed + 1 or ordinal > self.fit_reservations:
            raise RuntimeError("fit completion order changed")
        self.fits_completed = ordinal
        self._write("fit_completed", {"ordinal": ordinal})
        print(
            _canonical_bytes(
                {
                    "progress": "fit_completed",
                    "fits": ordinal,
                    "fit_ceiling": 72,
                    "materializations": self.materializations,
                }
            ).decode("utf-8"),
            file=sys.stderr,
            flush=True,
        )

    def reserve_materialization(self, label: str) -> int:
        ordinal = self.materializations + 1
        if ordinal > MAXIMUM_MATERIALIZATIONS:
            raise RuntimeError("21-materialization ceiling exceeded")
        if ordinal <= 3:
            expected = f"inner_anchor_surface:{INNER_WINDOWS[ordinal - 1]}"
        elif ordinal <= 12:
            zero = ordinal - 4
            expected = f"inner_context_surface:{INNER_WINDOWS[zero // 3]}:{CONTEXT_BANKS[zero % 3]}"
        else:
            zero = ordinal - 13
            expected = f"outer_context_surface:{OUTER_FOLDS[zero // 3]}:{CONTEXT_BANKS[zero % 3]}"
        if label != expected:
            raise RuntimeError(f"materialization plan differs at ordinal {ordinal}")
        self.materializations = ordinal
        self._write("materialization_reserved", {"ordinal": ordinal, "label": label})
        return ordinal

    def record_outer_freeze(self, freeze: Mapping[str, Any]) -> None:
        if (self.fit_reservations, self.fits_completed, self.materializations) != (72, 72, 21):
            raise RuntimeError("outer freeze requires the complete frozen operation graph")
        self._write("outer_predictions_frozen", {"freeze": dict(freeze), "outer_scores_before": 0})

    def record_aggregate(self, result: Mapping[str, Any]) -> None:
        self._write("aggregate_scored", {"result_payload_sha256": _canonical_sha(result), "outer_scores": 1})

    def fail(self, phase: str, error: BaseException) -> None:
        if not (self.journal_dir / "9999_failed.json").exists():
            _atomic_create_json(
                self.journal_dir / "9999_failed.json",
                {
                    "experiment_id": EXPERIMENT_ID,
                    "attempt_id": self.attempt_id,
                    "status": "FAILED_FAIL_CLOSED_LOCK_RETAINED",
                    "phase": phase,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "fit_reservations": self.fit_reservations,
                    "fits_completed": self.fits_completed,
                    "scientific_materializations": self.materializations,
                    "created_at_kst": _now_kst(),
                },
            )


def _report(result: Mapping[str, Any]) -> bytes:
    pooled = result["metrics"]["pooled"]
    ci = result["metrics"]["paired_bootstrap"]["difference_ci90"]
    selected = result["selected_inner_cell"]
    lines = [
        "# P1 long-event segment-rescore reactivation",
        "",
        f"결론: **{result['decision']}**",
        "",
        f"- candidate F1: {pooled['candidate']['f1']:.9f}",
        f"- anchor F1: {pooled['anchor']['f1']:.9f}",
        f"- F1 delta: {pooled['f1_delta']:+.9f}",
        f"- paired bootstrap 90% CI: {ci}",
        f"- planning score conversion: {result['planning_score_conversion']['estimated_point_delta']:+.6f}",
        f"- selected cell: {selected['cell_id']}, threshold={selected['threshold']}",
        "- fits/materializations: 72/21",
        f"- runtime seconds: {result['runtime']['wall_seconds']:.3f}",
        "- official test/sample/submission reads: 0/0/0",
        "- row-level predictions, CSV candidates, and uploads: 0",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _publish(screen: Mapping[str, Any], journal: AttemptJournal, started: float) -> Path:
    result = dict(screen)
    result["experiment_id"] = EXPERIMENT_ID
    pooled_delta = float(result["metrics"]["pooled"]["f1_delta"])
    result.update(
        {
            "attempt_id": journal.attempt_id,
            "completed_at_kst": _now_kst(),
            "runtime": {
                "wall_seconds": time.time() - started,
                "physical_fit_reservations": journal.fit_reservations,
                "physical_fits_completed": journal.fits_completed,
                "scientific_materializations": journal.materializations,
            },
            "planning_score_conversion": {
                "basis": "v1 preregistration planning map; not an official-score guarantee",
                "points_per_local_f1": PLANNING_POINTS_PER_F1,
                "estimated_point_delta": pooled_delta * PLANNING_POINTS_PER_F1,
            },
            "stability": {
                "fold_f1_deltas": {
                    key: value["f1_delta"] for key, value in result["metrics"]["folds"].items()
                },
                "station_f1_deltas": {
                    key: value["f1_delta"] for key, value in result["metrics"]["stations"].items()
                },
                "seed_f1_deltas": result["metrics"]["gate_metrics"]["seed_f1_deltas"],
                "paired_ci90": result["metrics"]["paired_bootstrap"]["difference_ci90"],
                "research_checks": result["metrics"]["gates"]["research_checks"],
            },
            "forbidden_access": {
                "official_test_reads": 0,
                "sample_submission_reads": 0,
                "submission_candidate_reads": 0,
                "candidate_csv_files": 0,
                "uploads": 0,
            },
        }
    )
    journal.record_aggregate(result)
    metrics_path = _atomic_create_json(ARTIFACT_DIR / "metrics.json", result["metrics"])
    report_path = _atomic_create_bytes(ARTIFACT_DIR / "report_ko.md", _report(result))
    result_path = _atomic_create_json(ARTIFACT_DIR / "result.json", result)
    manifest = {
        "schema_version": "p1_long_event_segment_proposal_rescore.reactivation_manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "scientific_experiment_id": SCIENTIFIC_EXPERIMENT_ID,
        "status": "AGGREGATE_OUTPUTS_COMPLETE_BEFORE_FINAL_COMMIT",
        "attempt_id": journal.attempt_id,
        "inputs": {
            "config": {"path": str(CONFIG_PATH.relative_to(ROOT)).replace("\\", "/"), "sha256": _sha256(CONFIG_PATH)},
            "runner": {"path": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"), "sha256": _sha256(Path(__file__).resolve())},
            "source_train_sha256": EXPECTED_TRAIN_SHA256,
        },
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (metrics_path, report_path, result_path)
        },
        "operation_counters": result["operation_counters"],
        "forbidden_access": result["forbidden_access"],
        "created_at_kst": _now_kst(),
    }
    manifest_path = _atomic_create_json(ARTIFACT_DIR / "manifest.json", manifest)
    journal._write(
        "completed",
        {
            "status": "SUCCESS_ALL_OUTPUTS_VERIFIED_READY_FOR_LOCK_RELEASE",
            "result_sha256": _sha256(result_path),
            "manifest_sha256": _sha256(manifest_path),
            "fits": 72,
            "materializations": 21,
            "outer_scores": 1,
        },
    )
    if (journal.fit_reservations, journal.fits_completed, journal.materializations) != (72, 72, 21):
        raise RuntimeError("final operation accounting changed")
    journal.lock_path.unlink()
    return result_path


def execute() -> tuple[Path, dict[str, Any]]:
    started = time.time()
    public, v6, snapshot, base = prepare(retain_snapshot=True)
    assert snapshot is not None
    journal: AttemptJournal | None = None
    phase = "STATE_LOAD"
    try:
        execution_relative = v6._relative_literal(v6.EXECUTION_MODULE_PATH)
        execution_raw = (snapshot / execution_relative).read_bytes()
        numerical, execution, readiness, state = v6._load_worker_state(
            snapshot,
            {"strict_readiness": base["readiness"]},
            execution_raw,
        )
        if readiness != base["readiness"]:
            raise RuntimeError("same-process readiness changed before claim")
        deadline = started + HARD_WALL_SECONDS
        journal = AttemptJournal(deadline)
        journal.record_readiness(public)
        phase = "FIXED_72_FIT_NUMERICAL_SCREEN"
        closure = _read_json(snapshot / v6._relative_literal(v6.CLOSURE_V3_PATH))
        previous_id = execution.EXPERIMENT_ID
        execution.EXPERIMENT_ID = EXPERIMENT_ID
        try:
            screen = execution.run_authorized_screen(state, numerical, closure, journal, deadline)
        finally:
            execution.EXPERIMENT_ID = previous_id
        phase = "AGGREGATE_ONLY_PUBLICATION"
        result_path = _publish(screen, journal, started)
        result = _read_json(result_path)
        return result_path, result
    except BaseException as error:
        if journal is not None and journal.lock_path.exists():
            journal.fail(phase, error)
        raise
    finally:
        v6._remove_snapshot_modules(snapshot)
        v6._cleanup_snapshot(snapshot)


def qa() -> dict[str, Any]:
    preflight, _v6, _snapshot, _base = prepare(retain_snapshot=False)
    checks = {
        "zero_operation": all(value == 0 for value in preflight["operation_counters"].values()),
        "fresh_namespace": preflight["namespace"]["fresh_artifact_namespace"] is True,
        "same_process": preflight["execution_boundary"]["same_process"] is True,
        "hidden_worker_absent": preflight["execution_boundary"]["hidden_worker"] is False,
        "parent_capability_absent": preflight["execution_boundary"]["parent_capability"] is False,
        "forbidden_source_reads_zero": all(
            preflight["source"][key] == 0
            for key in ("official_test_reads", "sample_submission_reads", "submission_candidate_reads")
        ),
        "source_train_hash": preflight["source"]["train"]["sha256"] == EXPECTED_TRAIN_SHA256,
        "source_readme_hash": preflight["source"]["readme"]["sha256"] == EXPECTED_README_SHA256,
    }
    return {
        "schema_version": "p1_long_event_segment_proposal_rescore.reactivation_independent_qa.v1",
        "experiment_id": EXPERIMENT_ID,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "preflight_verification_sha256": preflight["verification_sha256"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--qa", action="store_true")
    action.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.preflight:
        output, _v6, _snapshot, _base = prepare(retain_snapshot=False)
        print(_canonical_bytes({"output": output}).decode("utf-8"), end="")
        return
    if args.qa:
        print(_canonical_bytes(qa()).decode("utf-8"), end="")
        return
    path, result = execute()
    print(
        _canonical_bytes(
            {
                "status": "complete",
                "result_path": str(path),
                "decision": result["decision"],
                "f1_delta": result["metrics"]["pooled"]["f1_delta"],
            }
        ).decode("utf-8"),
        end="",
    )


if __name__ == "__main__":
    main()
