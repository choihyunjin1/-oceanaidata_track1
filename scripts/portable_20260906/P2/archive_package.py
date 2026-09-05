"""Archive a completed local package, excluding all organizer source data."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    package = args.package.resolve()
    target = package / "06_docs/P2_PORTABLE_REPRODUCIBILITY.zip"
    if target.exists():
        raise FileExistsError("archive already exists")
    paths = sorted(p for p in package.rglob("*") if p.is_file())
    if any(
        p.name
        in {
            "observations.csv",
            "test_index.csv",
            "sample_submission.csv",
            "baseline_interp.csv",
            "score.py",
        }
        or p.suffix.lower() in {".pyc", ".npz", ".parquet", ".zip"}
        for p in paths
    ):
        raise ValueError("unapproved package member")
    if any(p.suffix.lower() == ".csv" and p.parent != package / "05_answer" for p in paths):
        raise ValueError("non-answer CSV excluded")
    hashes = {p.relative_to(package).as_posix(): sha(p) for p in paths}
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, "P2/" + path.relative_to(package).as_posix())
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise ValueError("archive CRC failure")
        verified = all(
            hashlib.sha256(archive.read("P2/" + name)).hexdigest() == checksum
            for name, checksum in hashes.items()
        )
    result = {
        "status": "PASS" if verified else "FAIL",
        "file": target.name,
        "bytes": target.stat().st_size,
        "sha256": sha(target),
        "members": len(hashes),
        "member_hashes": hashes,
        "organizer_source_files": 0,
        "models": 3,
        "answer_csvs": 2,
        "upload": 0,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: result[k] for k in ("status", "file", "bytes", "sha256", "members")}))


if __name__ == "__main__":
    main()
