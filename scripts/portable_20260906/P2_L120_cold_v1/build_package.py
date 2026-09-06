"""Source-only cold package and exact saved-result archive; no model fit or official I/O."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "artifacts/p2_c3_training_comparison_full_20260906_v1"
EXPECTED_MANIFEST = "ef251e369253b99bc270929b48582bca6a99ea0024eb2a06648607359cc60773"
EXPECTED_ANSWER = "fee6118bb4a4d1d421094aa0174d634cf73e919804ea1b62ab0bab4f1384ce4d"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def package_source(destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("new output directory required")
    if sha(SOURCE / "PACKAGE_MANIFEST.json") != EXPECTED_MANIFEST:
        raise ValueError("approved full source package manifest drift")
    parent = json.loads((SOURCE / "PACKAGE_MANIFEST.json").read_text())
    for relative, expected in parent["files"].items():
        if sha(SOURCE / relative) != expected:
            raise ValueError("source lineage drift")
    for folder in ("01_data", "02_code", "03_model", "04_logs", "05_answer", "06_docs"):
        (destination / folder).mkdir(parents=True)
    for relative in parent["files"]:
        shutil.copy2(SOURCE / relative, destination / relative)
    for filename in ("boot.py", "stages.py"):
        shutil.copy2(HERE / filename, destination / "02_code" / filename)
    shutil.copy2(ROOT / "scripts/execute_candidate_notebooks_20260906_v1.py", destination / "02_code/execute_notebooks.py")
    for filename in ("TRAIN.ipynb", "PREDICT.ipynb"):
        shutil.copy2(HERE / filename, destination / filename)
    shutil.copy2(HERE / "README.md", destination / "README.md")
    (destination / "01_data/README.md").write_text("Only a source reference: set P2_DATA_DIR to the distributed P2_profile_restore directory. No data is bundled.\n", encoding="utf-8")
    (destination / "requirements.txt").write_text("\n".join(f"{name}=={importlib.metadata.version(name)}" for name in
        ("numpy", "pandas", "torch", "threadpoolctl", "nbformat", "nbclient", "jupyter_client", "ipykernel")) + "\n", encoding="utf-8")
    write_json(destination / "06_docs/expected-answer.json", {"sha256": EXPECTED_ANSWER,
         "role": "hash-only comparison after inference; not a source prediction or training coefficient",
         "historical_full_package_manifest_sha256": EXPECTED_MANIFEST})
    paths = [p for p in destination.rglob("*") if p.is_file()]
    write_json(destination / "PACKAGE_MANIFEST.json", {"status": "SEALED_SOURCE_ONLY_ZERO_FITS",
         "files": {p.relative_to(destination).as_posix(): sha(p) for p in sorted(paths)},
         "old_numerical_code_unchanged": {p: parent["files"][p] for p in ("02_code/core.py", "02_code/base.py", "02_code/run.py")},
         "new_adapters": ["fresh process TRAIN/PREDICT orchestration", "network/original-repository-denial bootstrap"],
         "fits": 0, "source_data_included": False, "old_models_answers_included": False})
    return destination


def archive(directory, output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("new archive path required")
    directory = Path(directory).resolve()
    forbidden = {"observations.csv", "test_index.csv", "sample_submission.csv"}
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    if any(p.name in forbidden or p.suffix.lower() in {".npz", ".parquet", ".npy", ".pyc"} for p in files):
        raise ValueError("raw data/cache excluded from archive")
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(p for p in directory.rglob("*") if p.is_dir()):
            zipped.writestr(path.relative_to(directory).as_posix() + "/", b"")
        for path in files:
            zipped.write(path, path.relative_to(directory).as_posix())
    return {"path": str(output), "sha256": sha(output), "bytes": output.stat().st_size,
            "files": len(files), "source_data_included": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--zip")
    parser.add_argument("--archive-existing", action="store_true")
    args = parser.parse_args()
    package = Path(args.output).resolve() if args.archive_existing else package_source(args.output)
    receipt = {"status": "SOURCE_PACKAGE_READY_NOT_TRAINED", "path": str(package), "fits": 0,
               "manifest_sha256": sha(package / "PACKAGE_MANIFEST.json")}
    if args.zip:
        receipt["archive"] = archive(package, args.zip)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
