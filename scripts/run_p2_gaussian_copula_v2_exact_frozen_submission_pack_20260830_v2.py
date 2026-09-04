"""Complete the frozen P2 Gaussian-copula v2 pack after one loader-only repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # completion-only repair: expose existing scripts package
sys.path.insert(0, str(REPO / "src"))

import numpy  # noqa: E402
import pandas  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from p2_restore.p2_gaussian_copula_v2_exact_frozen_submission_pack_20260830_v1 import (  # noqa: E402
    SubmissionPackError,
    duplicate_receipt,
    ensure_external_output_dir,
    load_json,
    materialize_candidate,
    sha256_file,
)

EXPERIMENT_ID = "p2_gaussian_copula_v2_exact_frozen_submission_pack_20260830_v2"
EXPECTED_OVERLAY_CONFIG_SHA256 = (
    "6e17046601dbb9f433a472e2eaac39f38ca68155707f6533071ce33f1ab79c76"
)
EXPECTED_BASE_CONFIG_SHA256 = (
    "061e8667085278282d4a23d792ac04c90bef91dcbad3b32395ac00bbcfa6773f"
)
DEFAULT_CONFIG = REPO / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
DEFAULT_BASE_CONFIG = (
    REPO
    / "configs"
    / "experiments"
    / "p2_gaussian_copula_v2_exact_frozen_submission_pack_20260830_v1.json"
)
DEFAULT_REPORT_DIR = REPO / "reports" / EXPERIMENT_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--p2-dir", type=Path, required=True)
    parser.add_argument("--base-u", type=Path, required=True)
    parser.add_argument("--alpha50-reference", type=Path, required=True)
    parser.add_argument("--official-incumbent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def _within_repo(repo_root: Path, path: Path) -> Path:
    root = repo_root.resolve()
    candidate = path if path.is_absolute() else root / path
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise SubmissionPackError(f"Repository output escapes root: {path}") from exc
    return candidate


def _write_json_exclusive(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise SubmissionPackError(f"exclusive output exists: {path}")
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".partial", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    if path.exists():
        raise SubmissionPackError(f"exclusive output exists: {path}")
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, suffix=".partial", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_overlay(
    repo_root: Path,
    overlay: dict[str, Any],
    base_config: dict[str, Any],
) -> dict[str, object]:
    if overlay.get("experiment_id") != EXPERIMENT_ID:
        raise SubmissionPackError("completion overlay experiment_id drifted")
    if overlay.get("overlay_scope") != "PREATTEMPT_LOADER_CONTRACT_REPAIR_ONLY":
        raise SubmissionPackError("completion overlay scope drifted")
    repair = overlay["repair"]
    if repair != {
        "cause": (
            "The v1 runner inserted REPO/src but not REPO before importing an "
            "unchanged module that imports the existing scripts package."
        ),
        "exact_change": (
            "ADD_REPO_TO_SYS_PATH_BEFORE_UNCHANGED_V1_SCIENTIFIC_MODULE_IMPORT"
        ),
        "loader_contract_lines_added": 1,
        "performance_result_existed_before_repair": False,
        "model_or_prediction_logic_changed": False,
        "data_or_query_mapping_changed": False,
        "gates_or_qa_changed": False,
    }:
        raise SubmissionPackError("repair scope drifted")

    receipts: dict[str, object] = {}
    for role, record in overlay["base_attempt"].items():
        actual = sha256_file(repo_root / record["path"])
        if actual != record["sha256"]:
            raise SubmissionPackError(f"completion overlay pin mismatch: {role}")
        receipts[role] = {"path": record["path"], "sha256": actual}

    frozen = overlay["frozen_scientific_invariants"]
    base_frozen = base_config["frozen_recipe"]
    checks = {
        "candidate": base_config["candidate"],
        "deployment_training_outer": base_frozen["deployment_training_outer"],
        "training_block_count": len(base_frozen["training_blocks"]),
        "selected_shrinkage": base_frozen["selected_shrinkage"],
        "expected_refit_model_receipt_sha256": base_frozen[
            "expected_refit_model_receipt_sha256"
        ],
        "maximum_copula_fits": base_config["execution_policy"][
            "maximum_copula_fits"
        ],
        "inner_search_or_hpo": base_frozen["inner_search_or_hpo"],
        "correction_rms_cap_c": base_frozen["correction_rms_cap_c"],
        "correction_p99_cap_c": base_frozen["correction_p99_cap_c"],
        "incomplete_profile_behavior": base_frozen["incomplete_profile_behavior"],
        "profile_projection": base_frozen["profile_projection"],
    }
    if frozen != checks:
        raise SubmissionPackError("frozen scientific invariants drifted")
    if overlay["execution_policy"] != base_config["execution_policy"]:
        raise SubmissionPackError("execution policy differs from sealed v1 pack")
    return receipts


def _reserve_attempt(
    report_dir: Path, overlay_sha256: str, base_config_sha256: str
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=False)
    lock = report_dir / "attempt.lock.json"
    _write_json_exclusive(
        lock,
        {
            "experiment_id": EXPERIMENT_ID,
            "maximum_executions": 1,
            "attempt_number": 1,
            "overlay_config_sha256": overlay_sha256,
            "base_scientific_config_sha256": base_config_sha256,
            "repair_scope": "PREATTEMPT_LOADER_CONTRACT_REPAIR_ONLY",
            "result_based_retry": False,
            "reserved_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
            "process_id": os.getpid(),
        },
    )
    return lock


def main() -> int:
    args = parse_args()
    if not args.execute:
        raise SystemExit("--execute is required; there is no same-ID retry")
    repo_root = args.repo_root.resolve()
    if repo_root != REPO.resolve():
        raise SubmissionPackError("repo-root does not match runner repository")
    overlay_path = _within_repo(repo_root, args.config)
    base_config_path = _within_repo(repo_root, args.base_config)
    report_dir = _within_repo(repo_root, args.report_dir)
    output_dir = ensure_external_output_dir(repo_root, args.output_dir)
    if output_dir.exists() or report_dir.exists():
        raise SubmissionPackError("one-shot report or staging output already exists")
    overlay_sha = sha256_file(overlay_path)
    base_config_sha = sha256_file(base_config_path)
    if overlay_sha != EXPECTED_OVERLAY_CONFIG_SHA256:
        raise SubmissionPackError("canonical completion overlay hash mismatch")
    if base_config_sha != EXPECTED_BASE_CONFIG_SHA256:
        raise SubmissionPackError("sealed v1 scientific config hash mismatch")
    overlay = load_json(overlay_path)
    base_config = load_json(base_config_path)
    overlay_receipts = validate_overlay(repo_root, overlay, base_config)
    lock_path = _reserve_attempt(report_dir, overlay_sha, base_config_sha)
    started_at = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
    started = time.perf_counter()

    with threadpool_limits(limits=1):
        candidate, materialization = materialize_candidate(
            repo_root,
            base_config,
            args.p2_dir.expanduser().resolve(),
            args.base_u.expanduser().resolve(),
            args.alpha50_reference.expanduser().resolve(),
            args.official_incumbent.expanduser().resolve(),
        )

    csv_payload = candidate.to_csv(
        index=False, encoding="utf-8", lineterminator="\n"
    ).encode("utf-8")
    candidate_sha = hashlib.sha256(csv_payload).hexdigest()
    duplicates = duplicate_receipt(candidate_sha, base_config)
    if duplicates["exact_hash_duplicate"]:
        raise SubmissionPackError("candidate exactly duplicates a recorded prior submission")

    output_dir.mkdir(parents=True, exist_ok=False)
    candidate_path = output_dir / base_config["submission_contract"]["output_filename"]
    _write_bytes_exclusive(candidate_path, csv_payload)
    if sha256_file(candidate_path) != candidate_sha:
        raise SubmissionPackError("staged candidate hash changed after write")
    note_path = output_dir / "제출정보.txt"
    note = (
        f"제출물 제목: {base_config['title']}\n"
        f"한줄요약(접근방식): {base_config['one_line_summary']}\n"
        f"파일 SHA-256: {candidate_sha}\n"
        "상태: READY_NOT_UPLOADED_RESEARCH_ONLY\n"
    ).encode("utf-8-sig")
    _write_bytes_exclusive(note_path, note)

    result = {
        "schema_version": (
            "p2.gaussian_copula_v2.exact_frozen_submission_pack.result.20260830.v2"
        ),
        "experiment_id": EXPERIMENT_ID,
        "status": overlay["decision_policy"][
            "all_materialization_and_qa_checks_pass"
        ],
        "candidate": base_config["candidate"],
        "title": base_config["title"],
        "one_line_summary": base_config["one_line_summary"],
        "research_only": True,
        "overlay_config_sha256": overlay_sha,
        "base_scientific_config_sha256": base_config_sha,
        "repair": overlay["repair"],
        "overlay_receipts": overlay_receipts,
        "output": {
            "path": str(candidate_path),
            "bytes": candidate_path.stat().st_size,
            "sha256": candidate_sha,
        },
        "note_path": str(note_path),
        "materialization": materialization,
        "duplicate_qa": duplicates,
        "expected_information_value": base_config["decision_policy"][
            "official_probe_value"
        ],
        "runtime": {
            "started_at_kst": started_at,
            "completed_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "thread_budget": 1,
        },
        "execution_receipt": {
            "attempts": 1,
            "copula_fits": 1,
            "inner_search_or_hpo": 0,
            "result_based_retries": 0,
            "hidden_truth_rows_read": 0,
            "score_py_reads": 0,
            "baseline_reads": 0,
            "csv_files_created": 1,
            "uploads": 0,
            "commits": 0,
            "pushes": 0,
        },
    }
    result_path = report_dir / "result.json"
    _write_json_exclusive(result_path, result)
    terminal = {
        "experiment_id": EXPERIMENT_ID,
        "status": result["status"],
        "candidate_path": str(candidate_path),
        "candidate_sha256": candidate_sha,
        "candidate_bytes": candidate_path.stat().st_size,
        "result_sha256": sha256_file(result_path),
        "attempt_lock_sha256": sha256_file(lock_path),
        "copula_fits": 1,
        "thread_budget": 1,
        "uploads": 0,
    }
    print(json.dumps(terminal, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
