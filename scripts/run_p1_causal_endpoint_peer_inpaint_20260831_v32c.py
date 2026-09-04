"""Exactly-once zero-fit causal endpoint/peer inpaint falsification."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v13 as base  # noqa: E402

EXPERIMENT_ID = "p1_causal_endpoint_peer_inpaint_20260831_v32c"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID


class ContractError(RuntimeError):
    """Frozen v32c contract violation."""


def causal_endpoint_peer_additions(
    frame: pd.DataFrame,
    *,
    cadence_minutes: int = 10,
    peer_lookback_minutes: int = 10,
) -> np.ndarray:
    required = {"station", "year", "layer", "time", "current_router_prediction"}
    if not required.issubset(frame.columns):
        raise ContractError("anchor frame lacks endpoint/peer keys")
    work = pd.DataFrame(
        {
            "station": frame["station"].astype(str),
            "year": frame["year"].astype(int),
            "layer": frame["layer"].astype(int),
            "time": pd.to_datetime(frame["time"], utc=True),
            "anchor": frame["current_router_prediction"].to_numpy(np.int8),
            "position": np.arange(len(frame), dtype=np.int64),
        }
    ).sort_values(["station", "year", "time", "layer", "position"], kind="stable")
    additions = np.zeros(len(frame), dtype=bool)
    cadence = pd.Timedelta(minutes=cadence_minutes)
    peer_age = pd.Timedelta(minutes=peer_lookback_minutes)
    for _, group in work.groupby(["station", "year"], sort=False, observed=True):
        last_by_layer: dict[int, tuple[pd.Timestamp, int]] = {}
        for current_time, batch in group.groupby("time", sort=True, observed=True):
            current = {int(row.layer): int(row.anchor) for row in batch.itertuples(index=False)}
            for row in batch.itertuples(index=False):
                layer = int(row.layer)
                prior = last_by_layer.get(layer)
                trailing_endpoint = prior is not None and prior[1] == 1 and current_time - prior[0] == cadence
                peer_support = any(
                    other_layer != layer
                    and other_anchor == 1
                    and pd.Timedelta(0) <= current_time - seen_time <= peer_age
                    for other_layer, (seen_time, other_anchor) in last_by_layer.items()
                ) or any(other_layer != layer and other_anchor == 1 for other_layer, other_anchor in current.items())
                if int(row.anchor) == 0 and trailing_endpoint and peer_support:
                    additions[int(row.position)] = True
            for layer, anchor in current.items():
                last_by_layer[layer] = (current_time, anchor)
    return additions


def load_contract() -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    candidate = config["candidate"]
    checks = {
        "id": config["experiment_id"] == EXPERIMENT_ID,
        "cadence": candidate["exact_cadence_minutes"] == 10,
        "peer": candidate["peer_lookback_minutes_inclusive"] == 10,
        "quorum": candidate["distinct_other_layer_quorum"] == 1,
        "zero_fit": config["fit_budget"]["maximum"] == 0,
        "calibration": config["transport_family"]["calibration_sha256"] == base.sha256_file(base.CALIBRATION_PATH),
    }
    if not all(checks.values()):
        raise ContractError(f"v32c contract mismatch: {checks}")
    return config


def execute() -> dict:
    started = time.perf_counter()
    config = load_contract()
    ARTIFACT.mkdir(parents=True, exist_ok=False)
    REPORT.mkdir(parents=True, exist_ok=True)
    lock = {
        "experiment_id": EXPERIMENT_ID,
        "pid": os.getpid(),
        "config_sha256": base.sha256_file(CONFIG),
        "runner_sha256": base.sha256_file(Path(__file__)),
        "fit_budget": 0,
    }
    base.write_json(ARTIFACT / "attempt_lock.json", lock)
    anchor_frame = pd.read_parquet(base.ANCHOR_PATH)
    if len(anchor_frame) != 421_032 or anchor_frame.duplicated(["station", "year", "layer", "time"]).any():
        raise ContractError("anchor key contract changed")
    spec = config["candidate"]
    additions = causal_endpoint_peer_additions(
        anchor_frame,
        cadence_minutes=int(spec["exact_cadence_minutes"]),
        peer_lookback_minutes=int(spec["peer_lookback_minutes_inclusive"]),
    )
    anchor = anchor_frame["current_router_prediction"].to_numpy(np.int8)
    candidate = anchor.copy()
    candidate[additions & (anchor == 0)] = 1
    seal_path = ARTIFACT / "proposal_blind.npz"
    np.savez_compressed(seal_path, additions=additions, candidate=candidate)
    seal = {
        "rows": len(anchor_frame),
        "additions": int(additions.sum()),
        "additions_sha256": base.sha256_bool(additions),
        "candidate_sha256": base.sha256_bool(candidate),
        "npz_sha256": base.sha256_file(seal_path),
        "target_columns_read_before_seal": 0,
        "raw_feature_columns_read": 0,
        "official_reads": 0,
    }
    base.write_json(ARTIFACT / "proposal_seal.json", seal)
    frame, candidate = base.attach_truth(anchor_frame, candidate)
    evaluation = base.evaluate(frame, candidate, config)
    result = {
        "status": "COMPLETE_INTERNAL_ONLY",
        "fit_count": 0,
        "runtime_seconds": time.perf_counter() - started,
        "candidate": evaluation,
        "seal": seal,
        "operations": {"official_reads": 0, "hidden_truth_reads": 0, "csv": 0, "uploads": 0},
        "hashes": {
            "config": base.sha256_file(CONFIG),
            "runner": base.sha256_file(Path(__file__)),
            "lock": base.sha256_file(ARTIFACT / "attempt_lock.json"),
            "proposal": base.sha256_file(seal_path),
        },
    }
    base.write_json(ARTIFACT / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "PREFLIGHT_ONLY", "contract": load_contract()}, indent=2))
        return
    print(json.dumps(execute(), indent=2))


if __name__ == "__main__":
    main()
