"""Read-only candidate hash and cold-archive inventory checks; no model imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "P1": {
        "csv": "artifacts/p1_champion_reconstruction_20260906_v1/candidate/05_answer/P1_submission.csv",
        "sha256": "b2f17f5cda8030cb3d97fbb504e6babb6aef8ba7fe555092901479677af0625e",
        "rows": 169011,
        "receipt": "artifacts/p1_champion_reconstruction_20260906_v1/candidate/05_answer/terminal.json",
        "qa": "artifacts/p1_champion_reconstruction_20260906_v1/candidate_replay/05_answer/terminal.json",
    },
    "P2": {
        "csv": "artifacts/p2_c3_training_comparison_full_20260906_v1/05_answer/submission_p2_L120_3seed.csv",
        "sha256": "fee6118bb4a4d1d421094aa0174d634cf73e919804ea1b62ab0bab4f1384ce4d",
        "rows": 26061,
        "receipt": "artifacts/p2_c3_training_comparison_full_20260906_v1/04_logs/terminal_result.json",
        "qa": "reports/parallel_core_training_20260906_v1/p2-answer-root-independent-qa.json",
    },
    "P3": {
        "csv": "artifacts/p3_numeric_candidate_20260906_v1/05_answer/submission.csv",
        "sha256": "ff42a6a08c76f0d58ed2f3a9ea31a08819ada5fa6e9007af942fe0b891937960",
        "rows": 1200,
        "receipt": "reports/p3_numeric_candidate_20260906_v1/result.json",
        "qa": "reports/parallel_core_training_20260906_v1/p3-root-independent-qa.json",
    },
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def owned(root, relative):
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("path outside chosen root")
    return result


def candidate_check(root, spec):
    csv = owned(root, spec["csv"])
    if sha(csv) != spec["sha256"]:
        raise ValueError("candidate SHA mismatch")
    evidence = {}
    for key in ("receipt", "qa"):
        path = owned(root, spec[key])
        value = json.loads(path.read_text(encoding="utf-8"))
        evidence[key] = {"path": spec[key], "sha256": sha(path), "recorded_status": value.get("status")}
        if not isinstance(value.get("status"), str):
            raise ValueError("receipt lacks explicit status")
    return {
        "status": "PINNED_CSV_HASH_MATCH",
        "csv": spec["csv"], "sha256": spec["sha256"],
        "previously_validated_rows": spec["rows"], "bytes": csv.stat().st_size,
        "evidence": evidence,
        "scope": "Bytes and receipt metadata only; no new schema, numerical or cold-training certification.",
    }


def cold_archive_check(path):
    """Inspect names/CRC without extracting; reject raw data, weights and attempts."""
    forbidden_suffixes = {".csv", ".parquet", ".npz", ".npy", ".pt", ".pth", ".cbm", ".joblib", ".pkl", ".ckpt"}
    names = []
    with zipfile.ZipFile(path) as archive:
        seen = set()
        for info in archive.infolist():
            name = info.orig_filename
            p = PurePosixPath(name)
            if "\x00" in name or "\\" in name or p.is_absolute() or ".." in p.parts or re.match(r"^[A-Za-z]:", name):
                raise ValueError("unsafe archive path")
            if name.casefold() in seen:
                raise ValueError("duplicate archive path")
            seen.add(name.casefold())
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("archive symlink forbidden")
            lower = name.lower()
            if p.suffix.lower() in forbidden_suffixes or any(
                part in {".git", ".env", "__pycache__"} for part in p.parts
            ) or any(term in lower for term in ("attempt_lock", "credential", "private_key")):
                raise ValueError("non-source entry in cold archive")
            names.append({"path": name, "bytes": info.file_size})
        if archive.testzip() is not None:
            raise ValueError("archive CRC mismatch")
    return {
        "status": "SOURCE_ARCHIVE_INVENTORY_PASS_NOT_EXECUTION",
        "sha256": sha(path), "entries": names,
        "limits": "No content-level secret audit, dependency completeness or offline execution claim.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cold-zip", action="append", type=Path, default=[])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("new output receipt required")
    result = {
        "checked_utc": datetime.now(UTC).isoformat(),
        "status": "HASH_INVENTORY_COMPLETE_NOT_FINAL_SUBMISSION_APPROVAL",
        "candidates": {key: candidate_check(args.root, spec) for key, spec in SPECS.items()},
        "cold_archives": [cold_archive_check(path) for path in args.cold_zip],
        "model_fits": 0, "official_source_reads": 0, "uploads": 0,
        "new_full_cold_execution": "NOT_PERFORMED_BY_THIS_CHECKER",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"status": result["status"], "candidate_count": len(SPECS), "output": str(args.output)}))


if __name__ == "__main__":
    main()
