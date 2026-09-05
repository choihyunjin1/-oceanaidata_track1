"""Build two explicit portable package roles; never include source/OOF truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

FOLDERS = ("01_data", "02_code", "03_model", "04_logs", "05_answer", "06_docs")


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def read_sealed(root):
    manifest = json.loads((root / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    files = {"PACKAGE_MANIFEST.json": (root / "PACKAGE_MANIFEST.json").read_bytes()}
    for name, expected in manifest["sha256"].items():
        path = (root / name).resolve()
        if root not in path.parents:
            raise ValueError("manifest path escape")
        payload = path.read_bytes()
        if sha(payload) != expected:
            raise ValueError("sealed code or recipe changed")
        files[name] = payload
    return files


def write_zip(path, files):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for folder in FOLDERS:
            entry = zipfile.ZipInfo(folder + "/", date_time=(2026, 9, 6, 0, 0, 0))
            entry.external_attr = 0o40755 << 16
            archive.writestr(entry, b"")
        for name, payload in sorted(files.items()):
            entry = zipfile.ZipInfo(name, date_time=(2026, 9, 6, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, payload)
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC failed")
        for name, payload in files.items():
            if sha(archive.read(name)) != sha(payload):
                raise ValueError("ZIP entry bytes changed")
        if not all(folder + "/" in archive.namelist() for folder in FOLDERS):
            raise ValueError("required package directories missing")
    return {
        "path": path.name,
        "sha256": sha(path.read_bytes()),
        "bytes": path.stat().st_size,
        "files": len(files),
        "required_directories": list(FOLDERS),
        "all_entry_hashes_verified": True,
    }


def build(cold_root, completed_root, out):
    cold_files = read_sealed(cold_root)
    cold_files["ARCHIVE_USAGE.md"] = (
        b"# Cold-start code package\n\n"
        b"This ZIP has no model or answer. Extract into a new empty directory. "
        b"Set P3_DATA_DIR to the organizer dataset and provision the pinned Python environment.\n\n"
        b"Follow README.md: RUN_TRAINING -> --replay -> training-only audit -> "
        b"RUN_INFERENCE -> --verify-answer -> full audit. All 01_data through 06_docs folders are present.\n\n"
        b"Raw data and offline dependency wheels are not included. This is not a portal submission or lock receipt.\n"
    )
    cold = write_zip(out / "P3_cold_start_code_v2.zip", cold_files)
    trained_files = read_sealed(completed_root)
    receipt_root = completed_root / "04_logs/receipts"
    train = json.loads((receipt_root / "training-result.json").read_text(encoding="utf-8"))
    selected = ["seal.json", "PREPARE_LOCK.json"]
    selected += list(train["model_sha256"]) + list(train["files_sha256"])
    selected += [
        "04_logs/receipts/" + name
        for name in (
            "training-result.json",
            "training-independent-qa.json",
            "fresh-process-replay.json",
            "answer-qa.json",
            "answer-replay-qa.json",
            "independent-qa.json",
            "source-to-answer-wall.json",
        )
    ]
    for name in selected:
        path = (completed_root / name).resolve()
        if completed_root not in path.parents:
            raise ValueError("saved artifact path escape")
        payload = path.read_bytes()
        expected = {**train["model_sha256"], **train["files_sha256"]}.get(name)
        if expected and sha(payload) != expected:
            raise ValueError("saved model or replay input hash differs")
        trained_files[name] = payload
    trained_files["02_code/archive_infer.py"] = (
        Path(__file__).with_name("archive_infer.py").read_bytes()
    )
    trained_files["ARCHIVE_USAGE.md"] = (
        "# Saved-model inference archive — NOT cold-start training\n\n"
        "This archive contains the already trained clean run_b models. It does not prove a second cold start. "
        "For actual source-to-empty-model training use P3_cold_start_code_v2.zip instead.\n\n"
        "Extract into a new directory, set P3_DATA_DIR, provision the pinned environment, then run:\n\n"
        "    python -I <package>/02_code/archive_infer.py --official-approved\n\n"
        "Output: 05_answer/replayed_submission.csv (P3 / OCN-03; 1200 keys). "
        "This command performs zero fits and calls the unchanged frozen official_frame/predict implementation. "
        "It validates the original training/model/recipe hashes and exact recorded answer SHA. "
        "Each extraction is one-shot; do not overwrite an existing replay output.\n\n"
        "The original run.py and PACKAGE_MANIFEST are byte-preserved. "
        "Do not RUN_TRAINING in this nonempty directory. Original --verify-answer uses the original training wall clock; "
        "use this inference-only adapter for later replay. Full audit.py is intentionally not promised here because "
        "training anchors/features/OOF truth were excluded. Original --replay also is not an archive entrypoint: "
        "its existing receipt is preserved and is exclusive-write. Only archive_infer.py is supported for this archive.\n\n"
        "Included replay_cases are train-derived past features, replay_expected is model predictions, and feature_columns is schema. "
        "No target/hidden truth, raw source CSV, training anchors/full feature table, or OOF target table is bundled. "
        "All nine model files are included because the unchanged load_models checks their hashes, although inference uses only full single/multi/router. "
        "These models/derived probes/ZIP are local-only and excluded from Git.\n\n"
        "This is current-environment saved-model replay, not a new venv/OS-offline certification, official upload, or final-model lock.\n"
    ).encode()
    companion = {
        "role": "saved-model inference only; fixed package manifest unchanged",
        "sha256": {name: sha(payload) for name, payload in trained_files.items()},
        "original_package_manifest_unchanged": True,
        "new_fits": 0,
    }
    trained_files["INFERENCE_ADAPTER_MANIFEST.json"] = (
        json.dumps(companion, indent=2) + "\n"
    ).encode()
    forbidden = (
        "oof.parquet",
        "train_features.parquet",
        "train_anchors.parquet",
        "train_wave.csv",
        "train_atmos.csv",
    )
    if any(name.endswith(forbidden) for name in trained_files):
        raise ValueError("unnecessary training target or raw source bundled")
    trained = write_zip(out / "P3_saved_model_replay_v2.zip", trained_files)
    return {
        "status": "ARCHIVES_BUILT_ENTRY_HASH_QA_PASS",
        "cold_start": cold,
        "saved_model_replay": trained,
        "original_manifest_bytes_preserved": True,
        "raw_source_included": False,
        "training_target_tables_included": False,
        "offline_wheel_bundle_included": False,
        "uploads": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cold", required=True, type=Path)
    parser.add_argument("--completed", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    result = build(args.cold.resolve(), args.completed.resolve(), args.out.resolve())
    with args.receipt.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
