"""Create a local P1 reproducibility ZIP from an explicit completed package."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    root = args.package.resolve()
    target = root / "06_docs/P1_PORTABLE_REPRODUCIBILITY.zip"
    allowed_folders = {"02_code", "03_model", "05_answer", "06_docs"}
    paths = [p for p in sorted(root.rglob("*")) if p.is_file()
             and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".zip"}
             and (p == root / "README.md" or p.relative_to(root).parts[0] in allowed_folders)]
    for path in paths:
        if path.suffix in {".npz", ".parquet"} or path.name in {
            "train.csv", "test.csv", "sample_submission.csv"}:
            raise ValueError("source data or unapproved payload")
        if path.suffix == ".csv" and path.parent != root / "05_answer":
            raise ValueError("non-answer CSV")
    hashes = {p.relative_to(root).as_posix(): sha(p) for p in paths}
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, "P1/" + path.relative_to(root).as_posix())
        for folder in ("01_data", "04_logs"):
            archive.writestr(f"P1/{folder}/", "")
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert all(hashlib.sha256(archive.read("P1/" + name)).hexdigest() == digest
                   for name, digest in hashes.items())
    result = {"status": "CRC_AND_MEMBER_HASH_PASS", "zip": str(target),
              "bytes": target.stat().st_size, "sha256": sha(target), "members": hashes,
              "source_data_files": 0, "logs_and_cache_files": 0, "upload": 0,
              "cold_start_instruction": "Use a new copy with empty 03_model and 05_answer; preserve this verified package."}
    with args.receipt.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "bytes", "sha256")}))


if __name__ == "__main__":
    main()
