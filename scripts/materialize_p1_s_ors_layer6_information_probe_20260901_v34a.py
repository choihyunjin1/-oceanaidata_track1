"""Materialize the exact S-ORS layer-6 information probe after internal terminal."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_s_ors_layer6_information_probe_20260901_v34a"
CONFIG_PATH = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
RESULT_PATH = ROOT / "reports" / EXPERIMENT_ID / "result.json"
RECEIPT_PATH = ROOT / "reports" / EXPERIMENT_ID / "materialization-result.json"
KEYS = ["station", "year", "layer", "time"]
COLUMNS = [*KEYS, "label"]


class ContractError(RuntimeError):
    """Raised when the exact materialization contract drifts."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate(frame: pd.DataFrame, rows: int) -> dict[str, bool]:
    checks = {
        "schema_exact": list(frame.columns) == COLUMNS,
        "rows_exact": len(frame) == rows,
        "keys_unique": not frame.duplicated(KEYS).any(),
        "rows_unique": not frame.duplicated().any(),
        "labels_finite": bool(np.isfinite(frame["label"].to_numpy(float)).all()),
        "labels_binary": set(frame["label"].unique()).issubset({0, 1}),
        "labels_integer": pd.api.types.is_integer_dtype(frame["label"].dtype),
    }
    if not all(checks.values()):
        raise ContractError(f"frame QA failed: {checks}")
    return checks


def preflight() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    output = resolve(config["official_materializer"]["output"])
    checks = {
        "internal_terminal": result["status"] in {"PERFORMANCE_GATE_PASS", "INFORMATION_PROBE_ONLY_PERFORMANCE_GATE_FAIL"},
        "information_eligible": result["candidate"]["information_value"]["eligible"] is True,
        "fit0": result["fit_count"] == 0,
        "official_zero": result["operations"]["official_candidate_reads"] == 0,
        "hidden_zero": result["operations"]["hidden_truth_reads"] == 0,
        "csv_zero": result["operations"]["submission_csv_created"] == 0,
        "output_absent": not output.exists(),
    }
    if not all(checks.values()):
        raise ContractError(f"preflight failed: {checks}")
    return {"status": "READY_TO_MATERIALIZE_INFORMATION_PROBE", "checks": checks}


def execute() -> dict[str, Any]:
    readiness = preflight()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    spec = config["official_materializer"]
    frames: dict[str, pd.DataFrame] = {}
    for name in ("champion", "e150", "anchor"):
        path = resolve(spec[name]["path"])
        if not path.is_file() or sha256_file(path) != spec[name]["sha256"]:
            raise ContractError(f"official source mismatch: {name}")
        frame = pd.read_csv(path, dtype={"station": "string", "time": "string", "label": "int8"})[COLUMNS]
        validate(frame, int(spec["expected_rows"]))
        frames[name] = frame
    if not frames["champion"][KEYS].equals(frames["e150"][KEYS]) or not frames["champion"][KEYS].equals(frames["anchor"][KEYS]):
        raise ContractError("official key order differs")
    champion_label = frames["champion"]["label"].to_numpy(np.int8)
    e150_label = frames["e150"]["label"].to_numpy(np.int8)
    anchor_label = frames["anchor"]["label"].to_numpy(np.int8)
    removal = (
        frames["champion"]["station"].astype(str).eq("S-ORS").to_numpy()
        & frames["champion"]["layer"].eq(6).to_numpy()
        & (anchor_label == 0)
        & (e150_label == 1)
    )
    candidate = frames["champion"].copy()
    candidate.loc[removal, "label"] = 0
    output_label = candidate["label"].to_numpy(np.int8)
    gi2 = (champion_label == 1) & (e150_label == 0)
    if np.any((anchor_label == 1) & (output_label == 0)):
        raise ContractError("anchor positive removed")
    if int(gi2.sum()) != 2 or np.any(gi2 & (output_label == 0)):
        raise ContractError("GI2 not preserved")
    if np.any((champion_label == 0) & (output_label == 1)):
        raise ContractError("unexpected addition")
    if int(removal.sum()) != int(spec["expected_removed_rows"]):
        raise ContractError("official removal count differs")
    if int(output_label.sum()) != int(spec["expected_positive_rows"]):
        raise ContractError("official positive count differs")
    checks = validate(candidate, int(spec["expected_rows"]))
    output = resolve(spec["output"])
    output.parent.mkdir(parents=True, exist_ok=False)
    candidate.to_csv(output, index=False, lineterminator="\n")
    receipt = {
        "schema_version": "p1.s_ors_layer6_information_probe.materialization.v34a",
        "experiment_id": EXPERIMENT_ID,
        "status": "MATERIALIZED_NOT_UPLOADED",
        "path": str(output),
        "rows": len(candidate),
        "positive_rows": int(output_label.sum()),
        "removed_s_ors_layer6_e150_additions": int(removal.sum()),
        "checks": checks,
        "sha256": sha256_file(output),
        "title": spec["title"],
        "summary": spec["summary"],
        "source_hashes": {name: spec[name]["sha256"] for name in ("champion", "e150", "anchor")},
        "preflight": readiness,
        "hidden_truth_reads": 0,
        "uploads": 0,
    }
    with RECEIPT_PATH.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(execute() if args.execute else preflight(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
