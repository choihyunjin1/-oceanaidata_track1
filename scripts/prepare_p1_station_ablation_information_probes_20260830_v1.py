"""Prepare three frozen, zero-fit P1 station-ablation information probes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_station_ablation_information_probes_20260830_v1"
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
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


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate_frame(frame: pd.DataFrame, rows: int, columns: list[str]) -> None:
    require(list(frame.columns) == columns, "column order mismatch")
    require(len(frame) == rows, "row count mismatch")
    require(not frame.isna().any().any(), "missing values")
    label = frame["label"].to_numpy()
    require(np.isfinite(label).all(), "nonfinite label")
    require(np.isin(label, [0, 1]).all(), "nonbinary label")
    require(frame[KEYS].duplicated().sum() == 0, "duplicate keys")


def build_candidate(
    champion: pd.DataFrame,
    e150: pd.DataFrame,
    anchor: pd.DataFrame,
    stations: set[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    champion_label = champion["label"].to_numpy(np.int8)
    e150_label = e150["label"].to_numpy(np.int8)
    anchor_label = anchor["label"].to_numpy(np.int8)
    e150_addition = (e150_label == 1) & (anchor_label == 0)
    removal = e150_addition & champion["station"].astype(str).isin(stations).to_numpy()
    output = champion.copy()
    output.loc[removal, "label"] = 0
    result = output["label"].to_numpy(np.int8)
    require(int(np.sum((anchor_label == 1) & (result == 0))) == 0, "anchor positive removed")
    gi2 = (champion_label == 1) & (e150_label == 0)
    require(int(gi2.sum()) == 2 and int(np.sum(gi2 & (result == 0))) == 0, "GI2 not preserved")
    require(int(np.sum((champion_label == 0) & (result == 1))) == 0, "unexpected addition")
    return output, removal


def prior_semantic_hashes(root: Path, excluded_root: Path, rows: int) -> dict[str, list[str]]:
    labels: dict[str, list[str]] = {}
    for path in root.rglob("*.csv"):
        if excluded_root in path.parents:
            continue
        try:
            frame = pd.read_csv(path, usecols=["label"])
        except (ValueError, OSError, UnicodeDecodeError):
            continue
        if len(frame) != rows or not np.isin(frame["label"].to_numpy(), [0, 1]).all():
            continue
        digest = hashlib.sha256(frame["label"].to_numpy(np.int8).tobytes()).hexdigest()
        labels.setdefault(digest, []).append(str(path))
    return labels


def execute() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    require(config["experiment_id"] == EXPERIMENT_ID, "experiment id")
    contracts = config["contracts"]
    require(contracts["hidden_truth_reads"] == 0 and contracts["score_py_reads"] == 0, "truth contract")
    require(contracts["model_fits"] == 0 and contracts["uploads"] == 0, "operation contract")
    frames: dict[str, pd.DataFrame] = {}
    for name in ("champion", "e150", "anchor"):
        source = config["sources"][name]
        path = resolve_path(source["path"])
        require(path.is_file() and sha256(path) == source["sha256"], f"source mismatch: {name}")
        frame = pd.read_csv(path)
        require(set(contracts["columns"]).issubset(frame.columns), f"required columns: {name}")
        frame = frame[list(contracts["columns"])].copy()
        validate_frame(frame, int(contracts["rows"]), list(contracts["columns"]))
        frames[name] = frame
    require(frames["champion"][KEYS].equals(frames["e150"][KEYS]), "champion/e150 key order")
    require(frames["champion"][KEYS].equals(frames["anchor"][KEYS]), "champion/anchor key order")

    output_root = resolve_path(config["output_root"])
    if output_root.exists():
        require(output_root.is_dir() and not any(output_root.iterdir()), "output root not empty")
    else:
        output_root.mkdir(parents=True)
    prior_root = Path("C:/Users/cedis/Downloads/해양 해커톤 제출용")
    prior_labels = prior_semantic_hashes(prior_root, output_root, int(contracts["rows"]))
    known_hashes = set(config["known_platform_semantic_duplicate_hashes"])
    records: list[dict] = []
    seen_labels: set[str] = set()
    for candidate_id in config["candidate_order"]:
        spec = config["candidates"][candidate_id]
        frame, removal = build_candidate(
            frames["champion"], frames["e150"], frames["anchor"],
            set(spec["remove_e150_additions_at_stations"]),
        )
        validate_frame(frame, int(contracts["rows"]), list(contracts["columns"]))
        require(int(removal.sum()) == int(spec["expected_removed_rows"]), f"removed rows: {candidate_id}")
        require(int(frame["label"].sum()) == int(spec["expected_positive_rows"]), f"positive rows: {candidate_id}")
        label_hash = hashlib.sha256(frame["label"].to_numpy(np.int8).tobytes()).hexdigest()
        require(label_hash not in prior_labels, f"prior local semantic duplicate: {candidate_id}")
        require(label_hash not in seen_labels, f"within-set semantic duplicate: {candidate_id}")
        seen_labels.add(label_hash)
        directory = output_root / spec["directory"]
        directory.mkdir()
        destination = directory / "P1_submission.csv"
        frame.to_csv(destination, index=False)
        file_hash = sha256(destination)
        require(file_hash not in known_hashes, f"known platform file duplicate: {candidate_id}")
        note = {"title": spec["title"], "summary": spec["summary"]}
        (directory / "SUBMISSION_NOTE.json").write_text(
            json.dumps(note, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        records.append({
            "candidate_id": candidate_id,
            "path": str(destination),
            "title": spec["title"],
            "summary": spec["summary"],
            "rows": len(frame),
            "positive_rows": int(frame["label"].sum()),
            "removed_rows_vs_champion": int(removal.sum()),
            "sha256": file_hash,
            "label_sha256": label_hash,
            "prior_local_semantic_matches": 0,
            "known_platform_hash_match": False,
        })
    manifest = {
        "schema_version": "p1.station_ablation_information_probes.manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "READY_NOT_UPLOADED",
        "config_sha256": sha256(CONFIG_PATH),
        "source_hashes": {name: config["sources"][name]["sha256"] for name in frames},
        "candidates": records,
        "operation_counters": {
            "model_fits": 0,
            "hidden_truth_reads": 0,
            "score_py_reads": 0,
            "csv_files_created": len(records),
            "uploads": 0,
        },
        "claim_limit": "Platform information probes only; no expected-improvement claim.",
    }
    manifest_path = output_root / "SET_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["set_manifest_sha256"] = sha256(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "READY_ZERO_FIT"}))
        return
    print(json.dumps(execute(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
