"""Label-free robust cross-layer subspace and block-conformal feasibility audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_robust_subspace_block_conformal_feasibility_20260901_v1"
CONFIG_PATH = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT_DIR = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK_PATH = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
INPUT_COLUMNS = ("station", "year", "layer", "time", "temp", "psal", "depth")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("JSON object required")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        os.write(descriptor, json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2).encode() + b"\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _train(path: Path, expected: str) -> Path:
    resolved = path.resolve(strict=True)
    lowered = str(resolved).casefold()
    if resolved.name != "train.csv" or any(value in lowered for value in ("test.csv", "sample_submission", "submission")) or _sha(resolved) != expected:
        raise RuntimeError("only exact historical train.csv is allowed")
    return resolved


def preflight(train_csv: Path) -> dict[str, Any]:
    if ARTIFACT_DIR.exists() or LOCK_PATH.exists():
        raise FileExistsError("namespace consumed")
    config = _read(CONFIG_PATH)
    authority = ROOT / config["design_authority"]["path"]
    if _sha(authority) != config["design_authority"]["sha256"]:
        raise RuntimeError("design authority changed")
    train = _train(train_csv, config["train_sha256"])
    audits = {}
    for fold, relative in config["fold_audits"].items():
        audit = _read(ROOT / relative)
        if audit.get("fold") != fold or audit.get("target_fold_validation_labels_read_before_prediction") != 0:
            raise RuntimeError("fold audit changed")
        audits[fold] = {"sha256": _sha(ROOT / relative), "cutoff": audit["adjusted_cutoff_utc"]}
    return {"schema_version": "p1.robust_subspace_conformal.preflight.v1", "experiment_id": EXPERIMENT_ID, "status": "PASS_ZERO_OPERATION_LABEL_FREE", "train_path": str(train), "config_sha256": _sha(CONFIG_PATH), "runner_sha256": _sha(Path(__file__)), "design_sha256": config["design_authority"]["sha256"], "fold_audits": audits, "frozen_method": config["frozen_method"], "sealed_ceilings": config["sealed_ceilings"], "counters": {"claims": 0, "fits": 0, "supervised_fits": 0, "targets": 0, "official": 0, "csv": 0, "uploads": 0}}


def _split(frame: pd.DataFrame, fractions: list[float]):
    parsed = pd.to_datetime(frame["time"], utc=True, errors="raise", format="mixed")
    unique = np.sort(parsed.unique())
    one = unique[int(len(unique) * fractions[0]) - 1]
    two = unique[int(len(unique) * (fractions[0] + fractions[1])) - 1]
    return (frame.loc[parsed <= one].reset_index(drop=True), frame.loc[(parsed > one) & (parsed <= two)].reset_index(drop=True), frame.loc[parsed > two].reset_index(drop=True))


def _fit(frame: pd.DataFrame) -> dict[tuple[str, int], tuple[float, float]]:
    params = {}
    for key, group in frame.groupby(["station", "layer"], sort=True, observed=True):
        values = group["temp"].to_numpy(dtype=float)
        center = float(np.median(values))
        scale = max(float(1.4826 * np.median(np.abs(values - center))), 1e-6)
        params[(str(key[0]), int(key[1]))] = (center, scale)
    return params


def _innovation(frame: pd.DataFrame, params: dict[tuple[str, int], tuple[float, float]]) -> np.ndarray:
    working = frame.loc[:, ["station", "layer", "time", "temp"]].reset_index(drop=True).copy()
    z = np.zeros(len(working), dtype=float)
    for key, positions in working.groupby(["station", "layer"], sort=True, observed=True).indices.items():
        center, scale = params.get((str(key[0]), int(key[1])), (0.0, float("inf")))
        z[np.asarray(positions)] = (working.iloc[positions]["temp"].to_numpy(dtype=float) - center) / scale
    working["z"] = z
    common = working.groupby(["station", "time"], sort=False, observed=True)["z"].transform("median").to_numpy(dtype=float)
    return np.abs(z - common)


def _blocks(frame: pd.DataFrame, statistic: np.ndarray, length: int):
    working = frame.loc[:, ["station", "layer", "time"]].reset_index(drop=True).copy()
    working["position"] = np.arange(len(frame))
    working["stat"] = statistic
    maxima, members, cells = [], [], []
    for key, group in working.groupby(["station", "layer"], sort=True, observed=True):
        group = group.sort_values("time", kind="stable")
        values = group["stat"].to_numpy(dtype=float)
        positions = group["position"].to_numpy(dtype=np.int64)
        for start in range(0, len(values) - length + 1, length):
            maxima.append(float(np.max(values[start : start + length])))
            members.append(positions[start : start + length])
            cells.append((str(key[0]), int(key[1])))
    if len(maxima) < 20:
        raise RuntimeError("insufficient blocks")
    return np.asarray(maxima), members, cells


def _evaluate(calibration: pd.DataFrame, cal_stat: np.ndarray, heldout: pd.DataFrame, held_stat: np.ndarray, length: int, q: float, min_layers: int, min_run: int):
    calibration_max, _cm, _cc = _blocks(calibration, cal_stat, length)
    held_max, held_members, _cells = _blocks(heldout, held_stat, length)
    p = (1 + (calibration_max[:, None] >= held_max[None, :]).sum(axis=0)) / (len(calibration_max) + 1)
    e = 0.5 / np.sqrt(p)
    order = np.argsort(-e, kind="stable")
    eligible = np.flatnonzero(e[order] >= len(e) / (q * np.arange(1, len(e) + 1)))
    rejected = np.zeros(len(e), dtype=bool)
    if len(eligible):
        rejected[order[: eligible[-1] + 1]] = True
    raw = np.zeros(len(heldout), dtype=bool)
    for use, positions in zip(rejected, held_members, strict=True):
        if use:
            raw[positions] = True
    working = heldout.loc[:, ["station", "layer", "time"]].reset_index(drop=True).copy()
    working["raw"] = raw
    coherent = raw & working.groupby(["station", "time"], observed=True)["raw"].transform("sum").ge(min_layers).to_numpy()
    proposal = np.zeros(len(heldout), dtype=bool)
    for _key, positions in working.groupby(["station", "layer"], sort=True, observed=True).indices.items():
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
            if end - start >= min_run:
                proposal[positions[start:end]] = True
            start = end
    counts = working.loc[proposal].groupby(["station", "layer"], observed=True).size() if proposal.any() else pd.Series(dtype=int)
    return {"block_length_rows": length, "calibration_blocks": len(calibration_max), "heldout_blocks": len(held_max), "rejected_blocks": int(rejected.sum()), "false_alarm_rate": float(rejected.mean()), "proposal_rows": int(proposal.sum()), "proposal_row_share": float(proposal.mean()), "cell_concentration": float(counts.max() / proposal.sum()) if proposal.any() else 0.0}


def execute(train_csv: Path) -> dict[str, Any]:
    started = time.monotonic()
    ready = preflight(train_csv)
    config = _read(CONFIG_PATH)
    _write(LOCK_PATH, {"experiment_id": EXPERIMENT_ID, "status": "CONSUMED_EXACTLY_ONCE", "config_sha256": ready["config_sha256"], "runner_sha256": ready["runner_sha256"]})
    ARTIFACT_DIR.mkdir(exist_ok=False)
    _write(ARTIFACT_DIR / "preflight.json", ready)
    frame = pd.read_csv(Path(ready["train_path"]), usecols=list(INPUT_COLUMNS)).loc[:, list(INPUT_COLUMNS)]
    parsed = pd.to_datetime(frame["time"], utc=True, errors="raise", format="mixed")
    method, ceilings = config["frozen_method"], config["sealed_ceilings"]
    folds, decisions = [], []
    for fold in config["fold_audits"]:
        prefix = frame.loc[parsed <= pd.Timestamp(ready["fold_audits"][fold]["cutoff"])].reset_index(drop=True)
        fit, calibration, heldout = _split(prefix, method["prefix_split"])
        params = _fit(fit)
        cal_stat, held_stat = _innovation(calibration, params), _innovation(heldout, params)
        sensitivity = []
        for length in method["block_lengths_rows"]:
            item = _evaluate(calibration, cal_stat, heldout, held_stat, length, method["e_bh_q"], method["minimum_concurrent_layers"], method["minimum_run_rows"])
            item["decision_pass"] = item["false_alarm_rate"] <= ceilings["false_alarm_rate_lte"] and item["proposal_row_share"] <= ceilings["proposal_row_share_lte"] and item["cell_concentration"] <= ceilings["cell_concentration_lte"]
            sensitivity.append(item)
            decisions.append(item["decision_pass"])
        folds.append({"fold": fold, "fit_rows": len(fit), "calibration_rows": len(calibration), "heldout_rows": len(heldout), "sensitivity": sensitivity, "stable": len({item["decision_pass"] for item in sensitivity}) == 1})
    passed = all(decisions) and all(item["stable"] for item in folds)
    result = {"schema_version": "p1.robust_subspace_conformal.result.v1", "experiment_id": EXPERIMENT_ID, "decision": config["decision"]["pass"] if passed else config["decision"]["fail"], "folds": folds, "performance_stage_authorized": passed, "counters": {"executions": 1, "input_only_fits": 3, "supervised_fits": 0, "targets": 0, "official": 0, "csv": 0, "uploads": 0}, "runtime_seconds": time.monotonic() - started, "hashes": {"config": ready["config_sha256"], "runner": ready["runner_sha256"], "lock": _sha(LOCK_PATH)}}
    _write(ARTIFACT_DIR / "result.json", result)
    return result


def qa(train_csv: Path):
    ready = preflight(train_csv)
    config = _read(CONFIG_PATH)
    checks = {"zero_operation": all(value == 0 for value in ready["counters"].values()), "single_q": config["frozen_method"]["e_bh_q"] == 0.01, "single_e_transform": config["frozen_method"]["e_value"] == "0.5/sqrt(p)", "targets_zero": config["operations"]["target_reads"] == 0, "supervised_zero": config["operations"]["supervised_fits"] == 0}
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
