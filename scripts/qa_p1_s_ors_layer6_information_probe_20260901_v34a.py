"""Independent structural QA for the materialized P1 S-layer6 probe."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_s_ors_layer6_information_probe_20260901_v34a"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
RESULT = ROOT / "reports" / EXPERIMENT_ID / "result.json"
RECEIPT = ROOT / "reports" / EXPERIMENT_ID / "materialization-result.json"
OUTPUT = ROOT / "reports" / EXPERIMENT_ID / "independent-qa.json"
KEYS = ["station", "year", "layer", "time"]
COLUMNS = [*KEYS, "label"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    spec = config["official_materializer"]
    frames = {}
    for name in ("champion", "e150", "anchor"):
        path = resolve(spec[name]["path"])
        frames[name] = pd.read_csv(path, dtype={"station": "string", "time": "string", "label": "int8"})[COLUMNS]
    candidate_path = Path(receipt["path"])
    candidate = pd.read_csv(candidate_path, dtype={"station": "string", "time": "string", "label": "int8"})[COLUMNS]
    champion = frames["champion"]
    e150 = frames["e150"]
    anchor = frames["anchor"]
    labels = {name: frame["label"].to_numpy(np.int8) for name, frame in {**frames, "candidate": candidate}.items()}
    expected_removal = (
        champion["station"].astype(str).eq("S-ORS").to_numpy()
        & champion["layer"].eq(6).to_numpy()
        & (labels["anchor"] == 0)
        & (labels["e150"] == 1)
    )
    actual_removal = (labels["champion"] == 1) & (labels["candidate"] == 0)
    gi2 = (labels["champion"] == 1) & (labels["e150"] == 0)
    checks = {
        "config_hash_matches_result": sha256_file(CONFIG) == result["hashes"]["config_sha256"],
        "candidate_hash_matches_receipt": sha256_file(candidate_path) == receipt["sha256"],
        "candidate_rows_169011": len(candidate) == 169011,
        "schema_exact": list(candidate.columns) == COLUMNS,
        "keys_unique": not candidate.duplicated(KEYS).any(),
        "rows_unique": not candidate.duplicated().any(),
        "source_key_order_exact": candidate[KEYS].equals(champion[KEYS]) and champion[KEYS].equals(e150[KEYS]) and champion[KEYS].equals(anchor[KEYS]),
        "labels_binary_finite": bool(np.isfinite(labels["candidate"].astype(float)).all() and np.isin(labels["candidate"], [0, 1]).all()),
        "positive_rows_6289": int(labels["candidate"].sum()) == 6289,
        "exact_107_row_action": int(actual_removal.sum()) == 107 and np.array_equal(actual_removal, expected_removal),
        "no_additions": not np.any((labels["champion"] == 0) & (labels["candidate"] == 1)),
        "anchor_preserved": not np.any((labels["anchor"] == 1) & (labels["candidate"] == 0)),
        "gi2_two_rows_preserved": int(gi2.sum()) == 2 and not np.any(gi2 & (labels["candidate"] == 0)),
        "internal_fit0": result["fit_count"] == 0,
        "hidden_truth0": receipt["hidden_truth_reads"] == 0 and result["operations"]["hidden_truth_reads"] == 0,
        "uploads0": receipt["uploads"] == 0 and result["operations"]["uploads"] == 0,
        "strict_performance_failure_disclosed": result["status"] == "INFORMATION_PROBE_ONLY_PERFORMANCE_GATE_FAIL",
    }
    qa = {
        "schema_version": "p1.s_ors_layer6_information_probe.independent_qa.v34a",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS_INFORMATION_PROBE_QA" if all(checks.values()) else "FAIL_QA",
        "checks": checks,
        "candidate_sha256": sha256_file(candidate_path),
        "label_sha256": hashlib.sha256(labels["candidate"].tobytes()).hexdigest(),
        "rows": len(candidate),
        "positive_rows": int(labels["candidate"].sum()),
        "removed_rows": int(actual_removal.sum()),
        "claim_limit": "Structurally valid information probe; internal strict performance gate failed and improvement is not claimed.",
    }
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(qa, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True))
    if qa["status"] != "PASS_INFORMATION_PROBE_QA":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
