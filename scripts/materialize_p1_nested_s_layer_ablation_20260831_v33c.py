"""Prepared v33c official materializer; never executed by the historical runner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_nested_s_layer_ablation_20260831_v33c"
CONFIG_PATH = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
RESULT_PATH = ROOT / "artifacts" / EXPERIMENT_ID / "result.json"
RECEIPT_PATH = ROOT / "reports" / EXPERIMENT_ID / "materialization-result.json"
KEYS = ["station", "year", "layer", "time"]
COLUMNS = [*KEYS, "label"]


class ContractError(RuntimeError):
    """Raised if a later authorized materialization drifts from v33c."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def validate_frame(frame: pd.DataFrame, expected_rows: int) -> dict[str, bool]:
    checks = {
        "schema_exact": list(frame.columns) == COLUMNS,
        "rows_exact": len(frame) == expected_rows,
        "keys_unique": not frame.duplicated(KEYS).any(),
        "rows_unique": not frame.duplicated().any(),
        "labels_finite": bool(np.isfinite(frame["label"].to_numpy(float)).all()),
        "labels_binary": set(frame["label"].unique()).issubset({0, 1}),
        "labels_integer": pd.api.types.is_integer_dtype(frame["label"].dtype),
    }
    if not all(checks.values()):
        raise ContractError(f"frame QA failed: {checks}")
    return checks


def build_candidate(
    champion: pd.DataFrame,
    e150: pd.DataFrame,
    anchor: pd.DataFrame,
    selected_layers: list[int],
) -> tuple[pd.DataFrame, np.ndarray]:
    champion_label = champion["label"].to_numpy(np.int8)
    e150_label = e150["label"].to_numpy(np.int8)
    anchor_label = anchor["label"].to_numpy(np.int8)
    removal = (
        champion["station"].astype(str).eq("S-ORS").to_numpy()
        & champion["layer"].isin(selected_layers).to_numpy()
        & (anchor_label == 0)
        & (e150_label == 1)
    )
    candidate = champion.copy()
    candidate.loc[removal, "label"] = 0
    output = candidate["label"].to_numpy(np.int8)
    gi2 = (champion_label == 1) & (e150_label == 0)
    if np.any((anchor_label == 1) & (output == 0)):
        raise ContractError("anchor positive removed")
    if np.any(gi2 & (output == 0)):
        raise ContractError("GI2 positive removed")
    if np.any((champion_label == 0) & (output == 1)):
        raise ContractError("unexpected addition")
    return candidate, removal


def preflight() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    selected = result["candidate"]["full_deployment_selection"]["selected_layers"]
    checks = {
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "internal_pass": result["status"] == "PASS_MATERIALIZER_READY",
        "strict_gate": result["candidate"]["strict_pass"] is True,
        "selected_layers_nonempty": len(selected) > 0,
        "fit0": result["fit_count"] == 0,
        "official0": result["operations"]["official_reads"] == 0,
        "csv0": result["operations"]["submission_csv_created"] == 0,
    }
    if not all(checks.values()):
        raise ContractError(f"materializer preflight failed: {checks}")
    return {
        "status": "READY_NO_OFFICIAL_READS",
        "checks": checks,
        "selected_layers": selected,
        "official_reads": 0,
        "submission_csv_created": 0,
        "uploads": 0,
    }


def execute(champion_path: Path, e150_path: Path, anchor_path: Path, output: Path) -> dict[str, Any]:
    readiness = preflight()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    spec = config["official_materializer"]
    source_paths = {"champion": champion_path, "e150": e150_path, "anchor": anchor_path}
    frames: dict[str, pd.DataFrame] = {}
    for name, path in source_paths.items():
        if not path.is_file() or sha256_file(path) != spec["source_sha256"][name]:
            raise ContractError(f"official source mismatch: {name}")
        frame = pd.read_csv(path, dtype={"station": "string", "time": "string", "label": "int8"})[COLUMNS]
        validate_frame(frame, int(spec["expected_rows"]))
        frames[name] = frame
    if not frames["champion"][KEYS].equals(frames["e150"][KEYS]) or not frames["champion"][KEYS].equals(frames["anchor"][KEYS]):
        raise ContractError("official key order differs")
    candidate, removal = build_candidate(frames["champion"], frames["e150"], frames["anchor"], readiness["selected_layers"])
    checks = validate_frame(candidate, int(spec["expected_rows"]))
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(output, index=False, lineterminator="\n")
    receipt = {
        "schema_version": "p1.nested_s_layer_ablation.materialization.v33c",
        "experiment_id": EXPERIMENT_ID,
        "status": "MATERIALIZED_NOT_UPLOADED",
        "path": str(output),
        "rows": len(candidate),
        "positive_rows": int(candidate["label"].sum()),
        "selected_layers": readiness["selected_layers"],
        "removed_s_e150_additions": int(removal.sum()),
        "checks": checks,
        "sha256": sha256_file(output),
        "title": spec["title"],
        "summary": spec["summary"],
        "hidden_truth_reads": 0,
        "uploads": 0,
    }
    with RECEIPT_PATH.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--champion", type=Path)
    parser.add_argument("--e150", type=Path)
    parser.add_argument("--anchor", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.preflight == args.execute:
        raise SystemExit("choose exactly one of --preflight or --execute")
    if args.preflight:
        print(json.dumps(preflight(), ensure_ascii=False, indent=2, sort_keys=True))
        return
    if not all([args.champion, args.e150, args.anchor, args.output]):
        raise SystemExit("--execute requires --champion --e150 --anchor --output")
    print(json.dumps(execute(args.champion, args.e150, args.anchor, args.output), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
