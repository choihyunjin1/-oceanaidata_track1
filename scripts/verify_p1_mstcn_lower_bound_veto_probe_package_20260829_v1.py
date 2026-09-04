"""Independently verify the ready P1 lower-bound-veto probe package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from p1_qc.submission import validate_submission

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_mstcn_lower_bound_veto_probe_package_20260829_v1"
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
E150_PATH = ROOT / "artifacts" / "p1_mstcn_e150_full_deployment_20260827_v1" / "P1_MSTCN_E150_ROUTER_UNION_ALL.csv"
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


def resolve_paths() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, variable in ENVIRONMENT_PATHS.items():
        configured = os.environ.get(variable)
        require(bool(configured), f"set {variable}")
        paths[name] = Path(str(configured)).expanduser().resolve()
    return paths


def execute() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected = config["expected"]
    paths = resolve_paths()
    delivery_root = paths["submission_root"] / config["package_directory"]
    candidate_path = delivery_root / config["candidate_directory"] / "P1_submission.csv"
    backup_path = delivery_root / config["backup_directory"] / "P1_submission.csv"
    note_path = candidate_path.parent / "제출정보.txt"
    manifest_path = delivery_root / "SET_MANIFEST.json"
    require(all(path.is_file() for path in (candidate_path, backup_path, note_path, manifest_path)), "missing package file")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    test = pd.read_csv(paths["data_dir"] / "test.csv")
    anchor = pd.read_csv(paths["current_router"])
    e150 = pd.read_csv(E150_PATH)
    champion = pd.read_csv(paths["champion"])
    candidate = pd.read_csv(candidate_path)
    require(test[KEYS].equals(anchor[KEYS]) and test[KEYS].equals(e150[KEYS]), "input keys")
    require(test[KEYS].equals(champion[KEYS]) and test[KEYS].equals(candidate[KEYS]), "candidate keys")
    validation = validate_submission(candidate_path, test)
    anchor_label = anchor["label"].astype(int)
    e150_label = e150["label"].astype(int)
    champion_label = champion["label"].astype(int)
    candidate_label = candidate["label"].astype(int)
    e150_addition = (e150_label == 1) & (anchor_label == 0)
    gi_only = (champion_label == 1) & (e150_label == 0)
    removed = (champion_label == 1) & (candidate_label == 0)
    accepted_e150 = e150_addition & (candidate_label == 1)
    require(validation["rows"] == int(expected["rows"]), "rows")
    require(validation["positive"] == int(expected["positive_rows"]), "positives")
    require(int(e150_addition.sum()) == int(expected["e150_added_rows"]), "e150 additions")
    require(int(accepted_e150.sum()) == int(expected["accepted_e150_rows"]), "accepted e150")
    require(int(gi_only.sum()) == int(expected["gi_only_rows"]), "GI-only count")
    require(int((gi_only & (candidate_label == 0)).sum()) == 0, "GI-only removal")
    require(int(((anchor_label == 1) & (candidate_label == 0)).sum()) == 0, "anchor removal")
    require(int((removed & ~e150_addition).sum()) == 0, "non-e150 removal")
    require(int(((candidate_label == 1) & (champion_label == 0)).sum()) == 0, "candidate addition outside champion")
    label_sha256 = hashlib.sha256(candidate_label.to_numpy("int8").tobytes()).hexdigest()
    require(label_sha256 == expected["shadow_label_sha256"], "label hash")
    require(validation["sha256"] == manifest["candidate"]["sha256"], "candidate hash")
    require(sha256(backup_path) == sha256(paths["champion"]), "champion backup")
    note = note_path.read_text(encoding="utf-8")
    require(config["submission_title"] in note and config["one_line_summary"] in note, "submission note")
    require(manifest["operation_counters"]["uploads"] == 0, "upload counter")
    result = {
        "schema_version": "p1.mstcn_lower_bound_veto_probe_package.independent_qa.v1",
        "experiment_id": EXPERIMENT_ID,
        "checked_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "status": "PASS_READY_NOT_UPLOADED",
        "candidate": validation,
        "candidate_label_sha256": label_sha256,
        "accepted_e150_rows": int(accepted_e150.sum()),
        "removed_e150_rows": int((removed & e150_addition).sum()),
        "anchor_rows_removed": int(((anchor_label == 1) & (candidate_label == 0)).sum()),
        "gi_only_rows_removed": int((gi_only & (candidate_label == 0)).sum()),
        "backup_champion_sha256": sha256(backup_path),
        "uploads": 0,
    }
    destination = delivery_root / "INDEPENDENT_QA.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, destination)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "READY_TO_VERIFY"}, indent=2))
        return
    print(json.dumps(execute(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
