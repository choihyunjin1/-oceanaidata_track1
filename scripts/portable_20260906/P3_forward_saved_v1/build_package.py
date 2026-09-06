"""Package only three QA-approved full models and pinned inference source; no fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = ("full_single.cbm", "full_multi.cbm", "full_router.joblib")
RECEIPTS = (
    "prepare.json",
    "training-result.json",
    "training-qa.json",
    "answer-qa.json",
    "answer-replay-qa.json",
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("manifest path escapes package")
    return path


def validate_cold(cold, pins):
    cold = cold.resolve()
    code = cold / "02_code"
    manifest = read(code / "manifest.json")
    for name, expected in manifest.items():
        if sha(inside(code, name)) != expected:
            raise ValueError("cold source manifest mismatch")
    for name, expected in pins.items():
        if manifest.get(name) != expected or sha(inside(code, name)) != expected:
            raise ValueError("reviewed inference source differs")
    frozen = read(code / "frozen.json")
    receipt = {name: read(cold / "06_docs" / name) for name in RECEIPTS}
    p, r, q, a, replay = (receipt[name] for name in RECEIPTS)
    result_sha = sha(cold / "06_docs/training-result.json")
    if (
        p["status"] != "PREPARED_SOURCE_ONLY"
        or p["fits"] != 0
        or p["old_cache_reads"] != 0
        or not p["empty_03_model"]
    ):
        raise ValueError("cold source-only prepare evidence missing")
    if (
        r["status"] != "TWELVE_BACKBONE_FIVE_ROUTER_COMPLETE"
        or (r["new_backbone_fits"], r["new_router_fits"]) != (12, 5)
        or r["old_oof_model_answer_reads"] != 0
        or r["official_rows"] != 0
        or sum(name.endswith(".cbm") for name in r["models"]) != 12
        or sum(name.endswith(".joblib") for name in r["models"]) != 5
    ):
        raise ValueError("complete clean cold training required")
    if r["variant"] not in ("numeric", "hmax") or r["source_sha256"] != frozen["source_files"]:
        raise ValueError("cold fixed variant/source differs")
    if (
        q["status"] != "PASS"
        or q["training_result_sha256"] != result_sha
        or q["pid"] == r["pid"]
        or q["rows"] != 103602
        or q["new_fits"] != 0
        or q["official_rows"] != 0
        or not q["oof_keys_probabilities_exact"]
        or not q["full_probe_exact"]
    ):
        raise ValueError("cold independent fresh-process QA missing")
    if (
        a["status"] != "LOCAL_ANSWER_NOT_UPLOADED"
        or replay["status"] != "EXACT_ANSWER_REPLAY_PASS"
        or a["pid"] == replay["pid"]
    ):
        raise ValueError("cold answer and separate process replay required")
    for row in (a, replay):
        if (
            row["training_result_sha256"] != result_sha
            or row["training_qa_sha256"] != sha(cold / "06_docs/training-qa.json")
            or row["rows"] != 1200
            or row["sample_rows"] != 0
            or row["hidden_rows"] != 0
            or row["uploads"] != 0
            or row["new_fits"] != 0
            or row["elapsed_seconds"] >= 21600
        ):
            raise ValueError("cold answer QA/source/six-hour boundary differs")
    if (
        a["sha256"] != replay["sha256"]
        or len(a["sha256"]) != 64
        or a["public_input_sha256"] != replay["public_input_sha256"]
        or set(a["public_input_sha256"]) != {"test_context.parquet", "test_index.csv"}
    ):
        raise ValueError("cold answer/input hashes differ")
    if (
        r["prepare_sha256"] != sha(cold / "06_docs/prepare.json")
        or r["manifest_sha256"] != sha(code / "manifest.json")
        or p["manifest_sha256"] != r["manifest_sha256"]
    ):
        raise ValueError("cold receipt chain mismatch")
    columns_path = cold / "04_logs/columns.json"
    if (
        sha(columns_path) != r["work"]["columns.json"]
        or sha(columns_path) != p["files"]["columns.json"]
    ):
        raise ValueError("column metadata changed")
    columns = read(columns_path)["columns"]
    if len(columns) != len(set(columns)) or len(columns) != 591:
        raise ValueError("original compact feature metadata differs")
    chosen = (
        [name for name in columns if not name.startswith("hmax_")]
        if r["variant"] == "hmax"
        else columns
    )
    if len(chosen) != (527 if r["variant"] == "hmax" else 591):
        raise ValueError("selected feature metadata differs")
    model_hashes = {}
    for name in MODELS:
        expected = r["models"]["03_model/" + name]
        if sha(cold / "03_model" / name) != expected:
            raise ValueError("accepted full model bytes changed")
        model_hashes[name] = expected
    provenance_path = cold / "06_docs/shrink-provenance.json"
    if sha(provenance_path) != frozen["shrink_provenance_sha256"]:
        raise ValueError("train-only fixed-shrink ancestry differs")
    return {
        "purpose": "LONG_TERM_SAVED_INFERENCE_NOT_COLD_TRAINING",
        "variant": r["variant"],
        "selected_columns": chosen,
        "recipe": frozen["recipe"],
        "resource_overrides": {"inference_cpu_threads": 2, "gpu": 0},
        "worker_timeout_seconds": 600,
        "full_models_sha256": model_hashes,
        "expected_answer_sha256": a["sha256"],
        "public_input_sha256": a["public_input_sha256"],
        "cold_training_qa_status": "PASS",
        "cold_source_sha256": frozen["source_files"],
        "cold_manifest_sha256": sha(code / "manifest.json"),
        "cold_receipts_sha256": {name: sha(cold / "06_docs" / name) for name in RECEIPTS},
        "cold_training_pid": r["pid"],
        "cold_answer_pid": a["pid"],
        "cold_answer_replay_pid": replay["pid"],
        "cold_whole_elapsed_seconds": replay["elapsed_seconds"],
        "reviewed_source_sha256": pins,
        "historical_cache_oof_probe_required": False,
        "model_fits": 0,
        "uploads": 0,
        "final_model_locked": False,
    }


def build(cold, destination):
    cold, destination = cold.resolve(), destination.resolve()
    if destination.exists() or destination.is_relative_to(cold):
        raise FileExistsError("new package outside cold attempt required")
    pins = read(HERE / "source-pins.json")
    accepted = validate_cold(cold, pins)
    archive = destination.parent / (destination.name + ".zip")
    if archive.exists():
        raise FileExistsError("archive already exists; preserve it")
    for name in ("01_data", "02_code", "03_model", "04_logs", "05_answer", "06_docs"):
        (destination / name).mkdir(parents=True)
    allowed = set()
    for name in pins:
        target = destination / "02_code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cold / "02_code" / name, target)
        allowed.add(target.relative_to(destination).as_posix())
    for name in MODELS:
        target = destination / "03_model" / name
        shutil.copy2(cold / "03_model" / name, target)
        allowed.add(target.relative_to(destination).as_posix())
    for source, target in (
        (HERE / "infer.py", destination / "02_code/infer.py"),
        (HERE / "README.md", destination / "README.md"),
        (cold / "02_code/requirements.txt", destination / "02_code/requirements.txt"),
        (cold / "06_docs/shrink-provenance.json", destination / "06_docs/shrink-provenance.json"),
    ):
        shutil.copy2(source, target)
        allowed.add(target.relative_to(destination).as_posix())
    save(destination / "02_code/frozen.json", accepted)
    save(
        destination / "06_docs/cold-provenance.json",
        {k: v for k, v in accepted.items() if k not in ("selected_columns", "recipe")},
    )
    allowed |= {"02_code/frozen.json", "06_docs/cold-provenance.json"}
    files = {
        p.relative_to(destination).as_posix(): sha(p) for p in destination.rglob("*") if p.is_file()
    }
    if set(files) != allowed or any(
        Path(name).suffix.lower() in {".csv", ".parquet", ".npz", ".npy", ".log"} or "LOCK" in name
        for name in files
    ):
        raise ValueError("unexpected archive payload")
    save(destination / "02_code/manifest.json", files)
    allowed.add("02_code/manifest.json")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(allowed):
            z.write(destination / name, name)
    with zipfile.ZipFile(archive) as z:
        if set(z.namelist()) != allowed or z.testzip() is not None:
            raise ValueError("archive file membership/CRC differs")
        for name, expected in files.items():
            if hashlib.sha256(z.read(name)).hexdigest() != expected:
                raise ValueError("archive bytes differ")
    receipt = {
        "status": "BUILT_PENDING_ACTUAL_EXTRACTED_INFERENCE",
        "variant": accepted["variant"],
        "archive_sha256": sha(archive),
        "archive_path": str(archive),
        "package_path": str(destination),
        "manifest_sha256": sha(destination / "02_code/manifest.json"),
        "files": len(allowed),
        "expected_answer_sha256": accepted["expected_answer_sha256"],
        "models": 3,
        "historical_cache_oof_probe_payload": 0,
        "raw_observations_payload": 0,
        "csv_payload": 0,
        "new_fits": 0,
        "official_data_reads": 0,
        "builder_sha256": sha(Path(__file__)),
        "adapter_sha256": sha(HERE / "infer.py"),
    }
    save(destination.parent / (destination.name + "-build-qa.json"), receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cold-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--build-approved", action="store_true")
    args = parser.parse_args()
    if not args.build_approved:
        parser.error("completed cold result and explicit packaging authorization required")
    print(json.dumps(build(args.cold_root, args.destination)), flush=True)


if __name__ == "__main__":
    main()
