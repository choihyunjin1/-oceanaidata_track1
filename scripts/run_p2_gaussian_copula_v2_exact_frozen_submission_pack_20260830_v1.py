"""Materialize the exact frozen P2 Gaussian-copula v2 submission pack once."""

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
sys.path.insert(0, str(REPO / "src"))

import numpy  # noqa: E402
import pandas  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from p2_restore.p2_gaussian_copula_v2_exact_frozen_submission_pack_20260830_v1 import (  # noqa: E402
    EXPERIMENT_ID,
    SubmissionPackError,
    duplicate_receipt,
    ensure_external_output_dir,
    load_json,
    materialize_candidate,
    sha256_file,
)

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

EXPECTED_CONFIG_SHA256 = "061e8667085278282d4a23d792ac04c90bef91dcbad3b32395ac00bbcfa6773f"
DEFAULT_CONFIG = REPO / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
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


def _reserve_attempt(report_dir: Path, config_sha256: str) -> Path:
    report_dir.mkdir(parents=True, exist_ok=False)
    lock = report_dir / "attempt.lock.json"
    _write_json_exclusive(
        lock,
        {
            "experiment_id": EXPERIMENT_ID,
            "maximum_executions": 1,
            "attempt_number": 1,
            "config_sha256": config_sha256,
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
    config_path = _within_repo(repo_root, args.config)
    report_dir = _within_repo(repo_root, args.report_dir)
    output_dir = ensure_external_output_dir(repo_root, args.output_dir)
    if output_dir.exists() or report_dir.exists():
        raise SubmissionPackError("one-shot report or staging output already exists")
    config_sha = sha256_file(config_path)
    if config_sha != EXPECTED_CONFIG_SHA256:
        raise SubmissionPackError("canonical config hash mismatch")
    config = load_json(config_path)
    lock_path = _reserve_attempt(report_dir, config_sha)
    started_at = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
    started = time.perf_counter()

    with threadpool_limits(limits=1):
        candidate, materialization = materialize_candidate(
            repo_root,
            config,
            args.p2_dir.expanduser().resolve(),
            args.base_u.expanduser().resolve(),
            args.alpha50_reference.expanduser().resolve(),
            args.official_incumbent.expanduser().resolve(),
        )

    csv_payload = candidate.to_csv(
        index=False, encoding="utf-8", lineterminator="\n"
    ).encode("utf-8")
    candidate_sha = hashlib.sha256(csv_payload).hexdigest()
    duplicates = duplicate_receipt(candidate_sha, config)
    if duplicates["exact_hash_duplicate"]:
        raise SubmissionPackError("candidate exactly duplicates a recorded prior submission")

    output_dir.mkdir(parents=True, exist_ok=False)
    candidate_path = output_dir / config["submission_contract"]["output_filename"]
    _write_bytes_exclusive(candidate_path, csv_payload)
    if sha256_file(candidate_path) != candidate_sha:
        raise SubmissionPackError("staged candidate hash changed after write")
    note_path = output_dir / "제출정보.txt"
    note = (
        f"제출물 제목: {config['title']}\n"
        f"한줄요약(접근방식): {config['one_line_summary']}\n"
        f"파일 SHA-256: {candidate_sha}\n"
        "상태: READY_NOT_UPLOADED_RESEARCH_ONLY\n"
    ).encode("utf-8-sig")
    _write_bytes_exclusive(note_path, note)

    result = {
        "schema_version": "p2.gaussian_copula_v2.exact_frozen_submission_pack.result.20260830.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": config["decision_policy"]["all_materialization_and_qa_checks_pass"],
        "candidate": config["candidate"],
        "title": config["title"],
        "one_line_summary": config["one_line_summary"],
        "research_only": True,
        "config_sha256": config_sha,
        "output": {
            "path": str(candidate_path),
            "bytes": candidate_path.stat().st_size,
            "sha256": candidate_sha,
        },
        "note_path": str(note_path),
        "materialization": materialization,
        "duplicate_qa": duplicates,
        "expected_information_value": config["decision_policy"]["official_probe_value"],
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
