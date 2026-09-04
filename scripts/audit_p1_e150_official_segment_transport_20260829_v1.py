"""Aggregate-only audit of e150 official-test segment transport and veto severity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_mstcn_official_shadow_lower_bound_veto_20260829_v1 as shadow  # noqa: E402

EXPERIMENT_ID = "p1_e150_official_segment_transport_audit_20260829_v1"
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
OUTPUT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
CURRENT_ROUTER_ENV = "P1_CURRENT_ROUTER"
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


def _current_router_path() -> Path:
    configured = os.environ.get(CURRENT_ROUTER_ENV)
    require(bool(configured), f"set {CURRENT_ROUTER_ENV}")
    path = Path(str(configured)).expanduser().resolve()
    require(path.is_file(), f"missing current router: {path}")
    return path


def _candidate_summary(
    *,
    anchor_positive_rows: int,
    gi_only_rows: int,
    segment_indices: list[np.ndarray],
    keep: np.ndarray,
) -> dict[str, int]:
    kept_rows = int(sum(len(segment_indices[index]) for index in np.flatnonzero(keep)))
    return {
        "kept_e150_segments": int(keep.sum()),
        "kept_e150_rows": kept_rows,
        "removed_e150_segments": int(len(segment_indices) - keep.sum()),
        "removed_e150_rows": int(sum(len(value) for value in segment_indices) - kept_rows),
        "candidate_positive_rows": int(anchor_positive_rows + gi_only_rows + kept_rows),
    }


def execute() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    shadow_config = json.loads(shadow.CONFIG_PATH.read_text(encoding="utf-8"))
    model_config = json.loads(shadow.MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    require(config["experiment_id"] == EXPERIMENT_ID, "config id")
    require(config["official_truth_values_read"] == 0, "truth contract")
    require(config["candidate_csv_creation"] is False and config["upload"] is False, "mutation contract")
    anchor_path = _current_router_path()
    champion_path = shadow.resolve_champion_path()
    anchor = pd.read_csv(anchor_path)
    e150 = pd.read_csv(shadow.E150_PATH)
    champion = pd.read_csv(champion_path)
    require(anchor[KEYS].equals(e150[KEYS]) and anchor[KEYS].equals(champion[KEYS]), "key mismatch")
    anchor_label = anchor["label"].to_numpy(np.int8)
    e150_label = e150["label"].to_numpy(np.int8)
    champion_label = champion["label"].to_numpy(np.int8)
    rows = len(anchor)
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
    lengths = np.asarray([len(value) for value in segment_indices], dtype=int)
    station = segments["station"].astype(str).to_numpy()
    dominant_type = segments[
        [f"type_{name}_mean" for name in shadow.prior.base.TYPE_NAMES]
    ].to_numpy(float).argmax(axis=1)
    type_names = np.asarray(shadow.prior.base.TYPE_NAMES, dtype=object)
    station_summary: dict[str, dict] = {}
    for name in sorted(np.unique(station)):
        mask = station == name
        station_summary[name] = {
            "segments": int(mask.sum()),
            "rows": int(lengths[mask].sum()),
            "frequency_quantiles": [
                float(value)
                for value in np.quantile(frequency[mask], [0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
            ],
            "segments_at_frequency_090": int(np.sum(mask & (frequency >= 0.9))),
            "rows_at_frequency_090": int(lengths[mask & (frequency >= 0.9)].sum()),
            "dominant_type_segment_counts": {
                str(type_name): int(np.sum(mask & (dominant_type == index)))
                for index, type_name in enumerate(type_names)
            },
        }
    threshold_summary = {}
    for threshold in config["acceptance_frequency_thresholds"]:
        keep = frequency >= float(threshold)
        threshold_summary[str(threshold)] = {
            "segments": int(keep.sum()),
            "rows": int(lengths[keep].sum()),
            "segments_by_station": {
                str(name): int(np.sum(keep & (station == name)))
                for name in sorted(np.unique(station))
            },
            "rows_by_station": {
                str(name): int(lengths[keep & (station == name)].sum())
                for name in sorted(np.unique(station))
            },
        }
    all_keep = np.ones(len(segments), dtype=bool)
    sparse = frequency >= 0.9
    station_i = station == "I-ORS"
    station_g = station == "G-ORS"
    station_s = station == "S-ORS"
    families = {
        "full_e150_plus_gi2": all_keep,
        "sparse_lower_bound": sparse,
        "all_I_plus_sparse_GS": station_i | sparse,
        "all_I_G_plus_sparse_S": station_i | station_g | (station_s & sparse),
        "all_I_S_plus_sparse_G": station_i | station_s | (station_g & sparse),
        "all_I_plus_GS_frequency_025": station_i | (frequency >= 0.25),
        "all_I_plus_GS_frequency_050": station_i | (frequency >= 0.50),
        "all_I_plus_GS_frequency_075": station_i | (frequency >= 0.75),
    }
    require(set(families) == set(config["fixed_candidate_families"]), "candidate family contract")
    gi_only_rows = int(np.sum((champion_label == 1) & (e150_label == 0)))
    anchor_positive_rows = int(anchor_label.sum())
    return {
        "schema_version": "p1.e150_official_segment_transport_audit.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "completed_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "status": "PASS_AGGREGATE_AUDIT",
        "official_e150": {
            "segments": int(len(segments)),
            "rows": int(lengths.sum()),
            "anchor_positive_rows": anchor_positive_rows,
            "gi_only_rows": gi_only_rows,
            "station_summary": station_summary,
            "threshold_summary": threshold_summary,
            "candidate_families": {
                name: _candidate_summary(
                    anchor_positive_rows=anchor_positive_rows,
                    gi_only_rows=gi_only_rows,
                    segment_indices=segment_indices,
                    keep=keep,
                )
                for name, keep in families.items()
            },
        },
        "bootstrap": {
            "replicates": int(shadow_config["bootstrap_replicates"]),
            "model_fits": int(2 * int(shadow_config["bootstrap_replicates"])),
            "resampling_attempts": attempts,
        },
        "input_hashes": {
            "config": sha256(CONFIG_PATH),
            "shadow_config": sha256(shadow.CONFIG_PATH),
            "model_config": sha256(shadow.MODEL_CONFIG_PATH),
            "current_router": sha256(anchor_path),
            "e150": sha256(shadow.E150_PATH),
            "official_champion": sha256(champion_path),
        },
        "operation_counters": {
            "official_truth_values_read": 0,
            "candidate_csv_files_created": 0,
            "uploads": 0,
            "model_fits": int(2 * int(shadow_config["bootstrap_replicates"])),
        },
        "claim_limit": "Aggregate deployment audit only; candidate families are not promoted or written.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "READY_AGGREGATE_ONLY"}, indent=2))
        return
    result = execute()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIR / "result.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, destination)
    print(json.dumps({"status": result["status"], "official_e150": result["official_e150"], "operation_counters": result["operation_counters"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
