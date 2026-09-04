"""Build the validated P1 lower-bound-veto probe package without uploading it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_mstcn_official_shadow_lower_bound_veto_20260829_v1 as shadow  # noqa: E402

from p1_qc.submission import build_submission, validate_submission  # noqa: E402

EXPERIMENT_ID = "p1_mstcn_lower_bound_veto_probe_package_20260829_v1"
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
SHADOW_CONFIG_PATH = (
    ROOT
    / "configs"
    / "experiments"
    / "p1_mstcn_official_shadow_lower_bound_veto_20260829_v1.json"
)
MODEL_CONFIG_PATH = shadow.MODEL_CONFIG_PATH
SHADOW_RESULT_PATH = shadow.OUTPUT_DIR / "result.json"
E150_PATH = shadow.E150_PATH
ENVIRONMENT_PATHS = {
    "data_dir": "P1_DATA_DIR",
    "current_router": "P1_CURRENT_ROUTER",
    "champion": "P1_CHAMPION_SUBMISSION",
    "submission_root": "P1_SUBMISSION_ROOT",
}
KEYS = ["station", "year", "layer", "time"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_paths(environment: dict[str, str] | None = None) -> dict[str, Path]:
    values = os.environ if environment is None else environment
    paths: dict[str, Path] = {}
    for name, variable in ENVIRONMENT_PATHS.items():
        configured = values.get(variable)
        require(bool(configured), f"set {variable}")
        paths[name] = Path(str(configured)).expanduser().resolve()
    require(paths["data_dir"].is_dir(), "missing P1_DATA_DIR")
    require((paths["data_dir"] / "test.csv").is_file(), "missing P1 test.csv")
    require(paths["current_router"].is_file(), "missing current router")
    require(paths["champion"].is_file(), "missing champion")
    require(paths["submission_root"].is_dir(), "missing submission root")
    return paths


def _git_state() -> dict[str, str | bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty_worktree": dirty}


def _write_csv_atomic(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=False)
    temporary = destination.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(temporary, destination)


def _write_json_atomic(value: dict, destination: Path) -> None:
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, destination)


def execute() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    shadow_config = json.loads(SHADOW_CONFIG_PATH.read_text(encoding="utf-8"))
    model_config = json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    shadow_result = json.loads(SHADOW_RESULT_PATH.read_text(encoding="utf-8"))
    expected = config["expected"]
    paths = resolve_paths()
    require(config["experiment_id"] == EXPERIMENT_ID, "config id")
    require(config["candidate_creation_authorized"] is True, "candidate creation contract")
    require(config["upload_authorized"] is False, "upload contract")
    require(shadow_result["status"] == "PASS_LABEL_FREE_SHADOW_RELEVANCE", "shadow status")
    require(shadow_result["official_shadow"]["shadow_label_sha256"] == expected["shadow_label_sha256"], "shadow hash contract")
    test = pd.read_csv(paths["data_dir"] / "test.csv")
    anchor = pd.read_csv(paths["current_router"])
    e150 = pd.read_csv(E150_PATH)
    champion = pd.read_csv(paths["champion"])
    rows = int(expected["rows"])
    require(len(test) == len(anchor) == len(e150) == len(champion) == rows, "row count")
    require(test[KEYS].equals(anchor[KEYS]), "test/anchor keys")
    require(test[KEYS].equals(e150[KEYS]), "test/e150 keys")
    require(test[KEYS].equals(champion[KEYS]), "test/champion keys")
    anchor_label = anchor["label"].to_numpy(np.int8)
    e150_label = e150["label"].to_numpy(np.int8)
    champion_label = champion["label"].to_numpy(np.int8)
    row_probability, boundary_probability, type_probability = shadow._load_prediction_ensemble(rows)
    segments, segment_indices = shadow._build_shadow_segments(
        e150[KEYS].copy(),
        anchor_label,
        e150_label,
        row_probability,
        boundary_probability,
        type_probability,
    )
    training, utility, groups = shadow._load_historical_training(shadow_config)
    frequency, attempts = shadow._bootstrap_frequency(
        training,
        utility,
        groups,
        segments,
        shadow_config,
        model_config,
    )
    acceptance = frequency >= float(shadow_config["acceptance_frequency"])
    candidate_label = champion_label.copy()
    for keep, positions in zip(acceptance, segment_indices, strict=True):
        if not keep:
            candidate_label[positions] = 0
    label_sha256 = hashlib.sha256(candidate_label.tobytes()).hexdigest()
    require(label_sha256 == expected["shadow_label_sha256"], "candidate label hash")
    candidate = build_submission(test, candidate_label)
    in_memory_validation = validate_submission(candidate, test)
    require(in_memory_validation["positive"] == int(expected["positive_rows"]), "positive rows")
    delivery_root = paths["submission_root"] / config["package_directory"]
    require(not delivery_root.exists(), f"delivery already exists: {delivery_root}")
    candidate_path = delivery_root / config["candidate_directory"] / "P1_submission.csv"
    backup_path = delivery_root / config["backup_directory"] / "P1_submission.csv"
    _write_csv_atomic(candidate, candidate_path)
    backup_path.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(paths["champion"], backup_path)
    note_path = candidate_path.parent / "제출정보.txt"
    note_path.write_text(
        f"제출물 제목: {config['submission_title']}\n"
        f"한줄요약(접근방식): {config['one_line_summary']}\n"
        "상태: 독립 QA 전, 업로드 미승인\n",
        encoding="utf-8",
    )
    disk_validation = validate_submission(candidate_path, test)
    require(disk_validation["sha256"] == sha256(candidate_path), "candidate disk hash")
    manifest = {
        "schema_version": "p1.mstcn_lower_bound_veto_probe_package.manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "created_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "status": "BUILT_AWAITING_INDEPENDENT_QA_NOT_UPLOADED",
        "submission_title": config["submission_title"],
        "one_line_summary": config["one_line_summary"],
        "git": _git_state(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "validation_contract": config["validation_contract"],
        "candidate": {
            **disk_validation,
            "label_sha256": label_sha256,
            "accepted_segments": int(acceptance.sum()),
            "accepted_e150_rows": int(sum(len(segment_indices[index]) for index in np.flatnonzero(acceptance))),
            "acceptance_frequency_minimum": float(np.min(frequency[acceptance])),
        },
        "backup_champion": {
            "path": str(backup_path.resolve()),
            "bytes": backup_path.stat().st_size,
            "sha256": sha256(backup_path),
        },
        "input_hashes": {
            "config": sha256(CONFIG_PATH),
            "shadow_config": sha256(SHADOW_CONFIG_PATH),
            "model_config": sha256(MODEL_CONFIG_PATH),
            "shadow_result": sha256(SHADOW_RESULT_PATH),
            "test": sha256(paths["data_dir"] / "test.csv"),
            "current_router": sha256(paths["current_router"]),
            "e150": sha256(E150_PATH),
            "official_champion": sha256(paths["champion"]),
        },
        "bootstrap": {
            "replicates": int(shadow_config["bootstrap_replicates"]),
            "model_fits": int(2 * int(shadow_config["bootstrap_replicates"])),
            "resampling_attempts": attempts,
        },
        "operation_counters": {
            "candidate_files_created": 1,
            "backup_files_created": 1,
            "uploads": 0,
            "official_truth_values_read": 0,
        },
    }
    _write_json_atomic(manifest, delivery_root / "SET_MANIFEST.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"experiment_id": EXPERIMENT_ID, "status": "READY_TO_BUILD_NOT_UPLOADED"}, indent=2))
        return
    result = execute()
    print(json.dumps({"status": result["status"], "candidate": result["candidate"], "uploads": result["operation_counters"]["uploads"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
