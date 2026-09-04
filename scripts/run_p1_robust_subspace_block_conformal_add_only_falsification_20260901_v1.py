"""Exactly-once champion-preserving robust-subspace conformal falsification."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_robust_subspace_block_conformal_add_only_falsification_20260901_v1"
CONFIG_PATH = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT_DIR = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK_PATH = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
AUDIT_RUNNER = ROOT / "scripts/run_p1_robust_subspace_block_conformal_feasibility_20260901_v1.py"
SCORER_RUNNER = ROOT / "scripts/run_p1_clean_state_capa_falsification_20260831_v1.py"
INPUT_COLUMNS = ("station", "year", "layer", "time", "temp", "psal", "depth")
KEY_COLUMNS = ("station", "year", "layer", "time")
PART_COLUMNS = (*KEY_COLUMNS, "row_position", "baseline_prediction")
POINTS_PER_F1 = 0.6778 / 0.0255
TRANSPORT_FACTOR = 0.30


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("module load failed")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


audit_module = _module(AUDIT_RUNNER, "subspace_audit_helpers")
scorer = _module(SCORER_RUNNER, "historical_score_helpers")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        os.write(descriptor, json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2).encode() + b"\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def preflight(train_csv: Path):
    if ARTIFACT_DIR.exists() or LOCK_PATH.exists():
        raise FileExistsError("namespace consumed")
    config = _read(CONFIG_PATH)
    audit_path = ROOT / config["audit"]["path"]
    audit = _read(audit_path)
    if _sha(audit_path) != config["audit"]["sha256"] or audit["decision"] != config["audit"]["required_decision"] or not audit["performance_stage_authorized"]:
        raise RuntimeError("audit authority invalid")
    train = train_csv.resolve(strict=True)
    if train.name != "train.csv" or _sha(train) != config["train_sha256"]:
        raise RuntimeError("train binding invalid")
    parts = {}
    for fold, item in config["parts"].items():
        part, fold_audit = ROOT / item["path"], _read(ROOT / item["audit"])
        if _sha(part) != item["sha256"] or fold_audit.get("target_fold_validation_labels_read_before_prediction") != 0:
            raise RuntimeError("part binding invalid")
        parts[fold] = {"cutoff": fold_audit["adjusted_cutoff_utc"], "sha256": item["sha256"]}
    return {"experiment_id": EXPERIMENT_ID, "status": "PASS_ZERO_OPERATION", "train": str(train), "config_sha256": _sha(CONFIG_PATH), "runner_sha256": _sha(Path(__file__)), "audit_sha256": _sha(audit_path), "parts": parts, "counters": {"claims": 0, "fits": 0, "targets": 0, "predictions": 0, "official": 0, "csv": 0, "uploads": 0}}


def _proposal(calibration: pd.DataFrame, cal_stat: np.ndarray, validation: pd.DataFrame, val_stat: np.ndarray, rule: dict[str, Any]):
    cal_max, _x, _y = audit_module._blocks(calibration, cal_stat, rule["block_length_rows"])
    val_max, members, _cells = audit_module._blocks(validation, val_stat, rule["block_length_rows"])
    p = (1 + (cal_max[:, None] >= val_max[None, :]).sum(axis=0)) / (len(cal_max) + 1)
    e = 0.5 / np.sqrt(p)
    order = np.argsort(-e, kind="stable")
    eligible = np.flatnonzero(e[order] >= len(e) / (rule["e_bh_q"] * np.arange(1, len(e) + 1)))
    rejected = np.zeros(len(e), dtype=bool)
    if len(eligible):
        rejected[order[: eligible[-1] + 1]] = True
    raw = np.zeros(len(validation), dtype=bool)
    for use, positions in zip(rejected, members, strict=True):
        if use:
            raw[positions] = True
    work = validation.loc[:, ["station", "layer", "time"]].reset_index(drop=True).copy()
    work["raw"] = raw
    coherent = raw & work.groupby(["station", "time"], observed=True)["raw"].transform("sum").ge(rule["minimum_concurrent_layers"]).to_numpy()
    additions = np.zeros(len(validation), dtype=bool)
    for _key, positions in work.groupby(["station", "layer"], sort=True, observed=True).indices.items():
        positions = np.asarray(positions)
        flags = coherent[positions]
        start = 0
        while start < len(flags):
            if not flags[start]:
                start += 1
                continue
            end = start + 1
            while end < len(flags) and flags[end]:
                end += 1
            if end - start >= rule["minimum_run_rows"]:
                additions[positions[start:end]] = True
            start = end
    return additions, {"calibration_blocks": len(cal_max), "validation_blocks": len(val_max), "rejected_blocks": int(rejected.sum()), "raw_rows": int(raw.sum()), "coherent_rows": int(coherent.sum()), "addition_rows": int(additions.sum())}


def execute(train_csv: Path):
    started = time.monotonic()
    ready, config = preflight(train_csv), _read(CONFIG_PATH)
    _write(LOCK_PATH, {"experiment_id": EXPERIMENT_ID, "status": "CONSUMED_EXACTLY_ONCE", "config_sha256": ready["config_sha256"], "runner_sha256": ready["runner_sha256"]})
    ARTIFACT_DIR.mkdir(exist_ok=False)
    _write(ARTIFACT_DIR / "preflight.json", ready)
    frame = pd.read_csv(Path(ready["train"]), usecols=list(INPUT_COLUMNS)).loc[:, list(INPUT_COLUMNS)]
    parsed = pd.to_datetime(frame["time"], utc=True, errors="raise", format="mixed")
    seals = []
    for fold, item in config["parts"].items():
        part = pd.read_parquet(ROOT / item["path"], columns=list(PART_COLUMNS))
        positions = part["row_position"].to_numpy(dtype=np.int64)
        prefix = frame.loc[parsed <= pd.Timestamp(ready["parts"][fold]["cutoff"])].reset_index(drop=True)
        times = pd.to_datetime(prefix["time"], utc=True, errors="raise", format="mixed")
        unique = np.sort(times.unique())
        boundary = unique[int(len(unique) * config["frozen_rule"]["prefix_fit_fraction"]) - 1]
        fit, calibration = prefix.loc[times <= boundary].reset_index(drop=True), prefix.loc[times > boundary].reset_index(drop=True)
        params = audit_module._fit(fit)
        validation = frame.iloc[positions].reset_index(drop=True)
        additions, proposal_audit = _proposal(calibration, audit_module._innovation(calibration, params), validation, audit_module._innovation(validation, params), config["frozen_rule"])
        if additions.mean() > config["frozen_rule"]["proposal_share_ceiling"]:
            raise RuntimeError("proposal budget exceeded before targets")
        incumbent = part["baseline_prediction"].to_numpy(dtype=np.int8)
        candidate = np.bitwise_or(incumbent, additions.astype(np.int8))
        path = ARTIFACT_DIR / f"{fold}_sealed.npz"
        np.savez_compressed(path, positions=positions, incumbent=incumbent, additions=additions, candidate=candidate)
        seal = {"fold": fold, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": _sha(path), "proposal": proposal_audit, "target_reads": 0}
        _write(ARTIFACT_DIR / f"{fold}_seal.json", seal)
        seals.append(seal)
    _write(ARTIFACT_DIR / "predictions_complete.json", {"experiment_id": EXPERIMENT_ID, "seals": seals, "target_reads": 0, "fits": 3, "supervised_fits": 0})
    targets = pd.read_csv(Path(ready["train"]), usecols=["label", "anomaly_type"])
    truth_all = targets["label"].to_numpy(dtype=np.int8)
    fold_scores, pool = [], {key: [] for key in ("truth", "incumbent", "candidate", "additions", "types", "metadata")}
    for seal in seals:
        with np.load(ROOT / seal["path"], allow_pickle=False) as value:
            positions, incumbent, additions, candidate = value["positions"], value["incumbent"], value["additions"], value["candidate"]
        truth, metadata, types = truth_all[positions], frame.iloc[positions].loc[:, list(KEY_COLUMNS)].reset_index(drop=True), targets.iloc[positions]["anomaly_type"].reset_index(drop=True)
        score = scorer._score_surface(truth, incumbent, candidate, additions, types, metadata)
        fold_scores.append({"fold": seal["fold"], **score})
        for key, value in (("truth", truth), ("incumbent", incumbent), ("candidate", candidate), ("additions", additions), ("types", types), ("metadata", metadata)):
            pool[key].append(value)
    truth, incumbent, candidate, additions = (np.concatenate(pool[key]) for key in ("truth", "incumbent", "candidate", "additions"))
    types, metadata = pd.concat(pool["types"], ignore_index=True), pd.concat(pool["metadata"], ignore_index=True)
    pooled = scorer._score_surface(truth, incumbent, candidate, additions, types, metadata)
    bootstrap = scorer._paired_cluster_bootstrap(truth, incumbent, candidate, metadata, replicates=config["decision"]["bootstrap_replicates"], seed=config["decision"]["seed"])
    passed = pooled["delta_f1"] > 0 and bootstrap["ci90"][0] >= 0 and all(item["delta_f1"] >= 0 for item in fold_scores) and pooled["additions_precision"] > pooled["incumbent"]["f1"] / 2
    result = {"experiment_id": EXPERIMENT_ID, "decision": config["decision"]["pass"] if passed else config["decision"]["fail"], "pooled": pooled, "fold_scores": fold_scores, "worst_slices": sorted(pooled["station_layer_diagnostics"], key=lambda item: item["delta_f1"])[:5], "block_bootstrap": bootstrap, "points": {"nominal": pooled["delta_f1"] * POINTS_PER_F1, "transport_adjusted": pooled["delta_f1"] * POINTS_PER_F1 * TRANSPORT_FACTOR}, "counters": {"fits": 3, "supervised_fits": 0, "anchor_removals": pooled["incumbent_positive_removals"], "target_rows_after_seal": len(targets), "official": 0, "csv": 0, "uploads": 0}, "runtime_seconds": time.monotonic() - started, "hashes": {"config": ready["config_sha256"], "runner": ready["runner_sha256"], "audit": ready["audit_sha256"], "completion": _sha(ARTIFACT_DIR / "predictions_complete.json"), "lock": _sha(LOCK_PATH)}}
    _write(ARTIFACT_DIR / "result.json", result)
    return result


def qa(train_csv: Path):
    ready, config = preflight(train_csv), _read(CONFIG_PATH)
    checks = {"zero": all(value == 0 for value in ready["counters"].values()), "audit_bound": ready["audit_sha256"] == config["audit"]["sha256"], "max9": config["operations"]["maximum_fits"] <= 9, "add_only": config["frozen_rule"]["anchor_operation"] == "bitwise_or_no_removal", "no_tuning": config["frozen_rule"]["label_tuning"] == 0}
    return {"experiment_id": EXPERIMENT_ID, "verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--qa", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    value = preflight(args.train_csv) if args.preflight else qa(args.train_csv) if args.qa else execute(args.train_csv)
    print(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False), end="")


if __name__ == "__main__":
    main()
