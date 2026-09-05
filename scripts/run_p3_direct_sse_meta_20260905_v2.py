"""Frozen two-policy P3 meta comparison using only earlier distributed-train OOF.

No backbone fitting, official inputs, submission CSV, network or uploads. Historical
predictions/targets remain in ignored artifacts; reports contain aggregates only.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = "p3_direct_sse_meta_20260905_v2"
CONFIG = ROOT / "configs/experiments" / f"{NAME}.json"
OUT = ROOT / "artifacts" / NAME
REPORT = ROOT / "reports" / NAME
LOCK = ROOT / "artifacts" / f"{NAME}.ATTEMPT_LOCK.json"
COMPONENTS = ["single_prediction", "multi_prediction", "persistence"]
LEADS = (3, 6, 9, 12, 18, 24)
LONG = (12, 18, 24)
KEYS = ["anchor_id", "station", "lead_h"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, allow_nan=False)


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(y) - np.asarray(p)))))


def simplex_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Exact active-set enumeration on a three-component probability simplex."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or x.shape[1] != 3 or y.shape != (len(x),) or not len(y):
        raise ValueError("simplex shape or empty-data error")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("simplex inputs must be finite")
    candidates = []
    for size in (1, 2, 3):
        for active in itertools.combinations(range(3), size):
            part = x[:, active]
            gram = part.T @ part
            kkt = np.block([[gram, np.ones((size, 1))], [np.ones((1, size)), np.zeros((1, 1))]])
            rhs = np.r_[part.T @ y, 1.0]
            solution = np.linalg.lstsq(kkt, rhs, rcond=None)[0][:size]
            if (solution < -1e-9).any() or abs(solution.sum() - 1.0) > 1e-7:
                continue
            solution = np.maximum(solution, 0.0)
            solution /= solution.sum()
            weights = np.zeros(3)
            weights[list(active)] = solution
            candidates.append((float(np.sum((x @ weights - y) ** 2)), active, weights))
    if not candidates:
        raise ValueError("no feasible simplex solution")
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def fit_policy(frame: pd.DataFrame, policy: str) -> dict:
    if policy == "global_bias":
        return {"policy": policy, "bias_m": float((frame.target_hs - frame.final_prediction).mean())}
    if policy == "long_simplex":
        active = frame.lead_h.isin(LONG)
        weight = simplex_fit(frame.loc[active, COMPONENTS].to_numpy(), frame.loc[active, "target_hs"].to_numpy())
        return {"policy": policy, "components": COMPONENTS, "weights": weight.tolist()}
    raise ValueError("unknown policy")


def apply_policy(frame: pd.DataFrame, parameters: dict) -> np.ndarray:
    prediction = frame.final_prediction.to_numpy(dtype=float).copy()
    if parameters["policy"] == "no_op":
        return prediction
    if parameters["policy"] == "global_bias":
        prediction += parameters["bias_m"]
    elif parameters["policy"] == "long_simplex":
        active = frame.lead_h.isin(LONG).to_numpy()
        prediction[active] = frame.loc[active, COMPONENTS].to_numpy() @ np.asarray(parameters["weights"])
    else:
        raise ValueError("unknown policy")
    return np.clip(prediction, 0.0, 30.0)


def prequential(frame: pd.DataFrame, order: list[str]) -> tuple[dict, list[dict]]:
    predictions = {name: frame.final_prediction.to_numpy(float).copy() for name in ("no_op", "long_simplex", "global_bias")}
    receipts = []
    for i, fold in enumerate(order):
        current = frame.fold.eq(fold).to_numpy()
        past = frame.fold.isin(order[:i]).to_numpy()
        if not current.any():
            raise ValueError("missing current fold")
        if not past.any():
            continue
        for policy in ("long_simplex", "global_bias"):
            parameters = fit_policy(frame.loc[past], policy)
            predictions[policy][current] = apply_policy(frame.loc[current], parameters)
            receipts.append({"fold": fold, "policy": policy, "fit_cases": int(frame.loc[past, "anchor_id"].nunique()), "fit_rows": int(past.sum()), "applied_cases": int(frame.loc[current, "anchor_id"].nunique()), "future_or_current_target_fit": False, "parameters": parameters, "fit_count": 1})
    return predictions, receipts


def slices(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    truth, baseline = frame.target_hs.to_numpy(float), frame.final_prediction.to_numpy(float)
    result = {"rows": len(frame), "cases": int(frame.anchor_id.nunique()), "rmse_m": rmse(truth, prediction), "delta_rmse_m": rmse(truth, prediction) - rmse(truth, baseline), "sse_m2": float(np.sum((truth - prediction) ** 2)), "mean_signed_error_m": float(np.mean(prediction - truth)), "changed_rows": int(np.count_nonzero(prediction != baseline))}
    for column in ("fold", "station", "lead_h"):
        result[f"by_{column}"] = {}
        for value, indices in frame.groupby(column, sort=True).indices.items():
            result[f"by_{column}"][str(value)] = {"rows": len(indices), "rmse_m": rmse(truth[indices], prediction[indices]), "baseline_rmse_m": rmse(truth[indices], baseline[indices]), "delta_rmse_m": rmse(truth[indices], prediction[indices]) - rmse(truth[indices], baseline[indices])}
    later = ~frame.fold.eq("2024_h2_storm").to_numpy()
    result["intervened_folds_2_3"] = {"rows": int(later.sum()), "rmse_m": rmse(truth[later], prediction[later]), "delta_rmse_m": rmse(truth[later], prediction[later]) - rmse(truth[later], baseline[later])}
    case_peak = frame.groupby("anchor_id").target_hs.max()
    cutoff = float(case_peak.quantile(0.9))
    peak_rows = frame.anchor_id.isin(case_peak[case_peak >= cutoff].index).to_numpy()
    result["observed_case_peak_top_decile_diagnostic_only"] = {"case_count": int(frame.loc[peak_rows, "anchor_id"].nunique()), "cutoff_m": cutoff, "rmse_m": rmse(truth[peak_rows], prediction[peak_rows]), "delta_rmse_m": rmse(truth[peak_rows], prediction[peak_rows]) - rmse(truth[peak_rows], baseline[peak_rows]), "used_for_selection": False}
    return result


def bootstrap(frame: pd.DataFrame, prediction: np.ndarray, config: dict) -> dict:
    work = frame[["anchor_id", "target_hs", "final_prediction"]].copy()
    work["candidate_sse"] = (work.target_hs.to_numpy() - prediction) ** 2
    work["baseline_sse"] = (work.target_hs - work.final_prediction) ** 2
    errors = work.groupby("anchor_id")[["candidate_sse", "baseline_sse"]].sum().to_numpy()
    rng = np.random.default_rng(config["seed"])
    indices = rng.integers(0, len(errors), size=(config["replicates"], len(errors)))
    sampled = errors[indices].mean(axis=1) / 6.0
    delta = np.sqrt(sampled[:, 0]) - np.sqrt(sampled[:, 1])
    return {"unit": "complete_six_lead_case", "cases": len(errors), "replicates": config["replicates"], "delta_rmse_ci95_m": np.quantile(delta, [0.025, 0.975]).tolist(), "descriptive_probability_improves": float(np.mean(delta < 0)), "selection_bias_removed": False, "not_fresh_confirmation": True}


def validate_frame(oof: pd.DataFrame, keys: pd.DataFrame, anchors: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    if oof.duplicated(KEYS).any() or keys.anchor_id.duplicated().any() or anchors.anchor_id.duplicated().any():
        raise ValueError("duplicate input keys")
    if len(oof) != 1086 or len(keys) != 181 or set(oof.anchor_id) != set(keys.anchor_id):
        raise ValueError("historical surface mismatch")
    if not oof.groupby("anchor_id").lead_h.agg(lambda x: tuple(sorted(x))).map(lambda values: values == LEADS).all():
        raise ValueError("six-lead completeness mismatch")
    frame = oof.merge(keys, on=["anchor_id", "station", "fold"], validate="many_to_one")
    if len(frame) != len(oof):
        raise ValueError("metadata join lost rows")
    frame.anchor_time = pd.to_datetime(frame.anchor_time, utc=True)
    frame = frame.sort_values(["anchor_time", "station", "anchor_id", "lead_h"]).reset_index(drop=True)
    numeric = frame[["target_hs", "final_prediction", *COMPONENTS]].to_numpy()
    if not np.isfinite(numeric).all() or not frame.current_hs.ge(1.5).all():
        raise ValueError("invalid train-derived values")
    lookup = anchors.set_index("anchor_id")
    max_difference = 0.0
    for lead in LEADS:
        selected = frame.lead_h.eq(lead)
        expected = lookup.loc[frame.loc[selected, "anchor_id"], f"target_{lead}"].to_numpy()
        max_difference = max(max_difference, float(np.max(np.abs(frame.loc[selected, "target_hs"].to_numpy() - expected))))
    if max_difference != 0.0:
        raise ValueError("OOF truth differs from distributed-train anchor targets")
    if not np.isclose(rmse(frame.target_hs.to_numpy(), frame.final_prediction.to_numpy()), 0.7791048399763751, rtol=0, atol=1e-12):
        raise ValueError("reference metric mismatch")
    chronology = []
    order = config["fold_order"]
    for i, fold in enumerate(order):
        current, past = keys[keys.fold.eq(fold)].copy(), keys[keys.fold.isin(order[:i])].copy()
        current.anchor_time = pd.to_datetime(current.anchor_time, utc=True)
        past.anchor_time = pd.to_datetime(past.anchor_time, utc=True)
        if len(current) != config["expected_cases_by_fold"][i] or len(past) != config["expected_meta_past_cases"][i]:
            raise ValueError("meta case budget mismatch")
        receipt = {"fold": fold, "cases": len(current), "past_cases": len(past)}
        if i:
            ready_margin = (pd.Timestamp(config["fold_starts_utc"][i], tz="UTC") - (past.anchor_time.max() + pd.Timedelta(hours=24))).total_seconds() / 3600
            minimum_gap = min((current.loc[current.station.eq(s), "anchor_time"].min() - past.loc[past.station.eq(s), "anchor_time"].max()).total_seconds() / 3600 for s in current.station.unique() if past.station.eq(s).any())
            overlap = set(zip(current.station, current.episode_id, strict=True)) & set(zip(past.station, past.episode_id, strict=True))
            if ready_margin < 0 or minimum_gap < 78 or overlap:
                raise ValueError("past target availability, 78h separation or episode contract failed")
            receipt.update({"global_past_target_ready_before_fold_start_h": ready_margin, "minimum_same_station_anchor_gap_h": minimum_gap, "minimum_same_station_footprint_gap_h": minimum_gap - 72, "same_station_episode_overlap": 0})
        chronology.append(receipt)
    key_hash = hashlib.sha256(pd.util.hash_pandas_object(frame[KEYS], index=False).to_numpy().tobytes()).hexdigest()
    return frame, {"rows": len(frame), "cases": len(keys), "duplicate_keys": 0, "truth_alignment_max_abs_m": max_difference, "key_sha256": key_hash, "chronology": chronology, "source_values_printed": False}


def install_guard(source: Path, config: dict) -> None:
    permitted = {(ROOT / name).resolve() for name in config["inputs"]}
    source_allowed = {source / name for name in config["source_files"]}

    def guard(event, args):
        if event == "socket.connect":
            raise PermissionError("network is disabled")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if "external_data" in path.parts or "hidden" in path.name.lower():
            raise PermissionError("external or hidden file is forbidden")
        if source in path.parents:
            if path not in source_allowed or (isinstance(args[1], str) and any(c in args[1] for c in "wax+")):
                raise PermissionError("source path not authorized")
        if path.suffix.lower() == ".csv" and path not in source_allowed:
            raise PermissionError("CSV read/write forbidden in meta research")
        if path.suffix.lower() in {".cbm", ".ckpt", ".pt", ".joblib", ".tabpfn_fit"}:
            raise PermissionError("backbone/checkpoint access forbidden in meta research")
        if path.suffix.lower() == ".parquet" and path not in permitted and OUT not in path.parents:
            raise PermissionError("unapproved parquet")

    sys.addaudithook(guard)


def execute() -> dict:
    started = time.perf_counter()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = Path(os.environ["P3_DATA_DIR"]).resolve()
    if config["experiment_id"] != NAME or config["policies"] != ["no_op", "long_simplex", "global_bias"]:
        raise ValueError("frozen contract mismatch")
    if LOCK.exists() or OUT.exists() or (REPORT / "result.json").exists():
        raise RuntimeError("exactly-once output/lock exists; no restart")
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_status = subprocess.check_output(["git", "status", "--short", "--branch"], cwd=ROOT, text=True)
    install_guard(source, config)
    verified = {}
    for name, expected in config["inputs"].items():
        verified[name] = sha(ROOT / name)
        if verified[name] != expected:
            raise ValueError("pinned train-derived input hash mismatch")
    for name, expected in config["source_files"].items():
        verified[f"source/{name}"] = sha(source / name)
        if verified[f"source/{name}"] != expected:
            raise ValueError("distributed source hash mismatch")
    receipt = {"experiment_id": NAME, "created_utc": datetime.now(UTC).isoformat(), "pid": os.getpid(), "config_sha256": sha(CONFIG), "runner_sha256": sha(Path(__file__)), "inputs": verified, "git_head": git_head, "dirty_worktree": bool(git_status.count("\n") > 1), "fit_budget": config["fit_budget"]}
    save(LOCK, receipt)
    OUT.mkdir(parents=True)
    try:
        base = ROOT / "artifacts/p3_score_repair_deploy_20260905_v1"
        frame, integrity = validate_frame(pd.read_parquet(base / "oof.parquet"), pd.read_parquet(base / "validation_keys.parquet"), pd.read_parquet(base / "train_anchors.parquet"), config)
        predictions, fits = prequential(frame, config["fold_order"])
        results = {}
        for name, values in predictions.items():
            results[name] = slices(frame, values)
            results[name]["paired_bootstrap"] = bootstrap(frame, values, config["bootstrap"])
        precedence = {"no_op": 0, "global_bias": 1, "long_simplex": 2}
        winner = min(results, key=lambda name: (results[name]["rmse_m"], precedence[name]))
        deployment = {"policy": "no_op"} if winner == "no_op" else fit_policy(frame, winner)
        save(OUT / "selected_meta.json", {"parameters": deployment, "train_cases": 181, "in_sample_performance_not_reported_as_validation": True, "fit_count": int(winner != "no_op")})
        saved = json.loads((OUT / "selected_meta.json").read_text(encoding="utf-8"))
        replay_difference = float(np.max(np.abs(apply_policy(frame, deployment) - apply_policy(frame, saved["parameters"]))))
        arrays = frame[[*KEYS, "fold", "anchor_time", "episode_id", "target_hs", "final_prediction", *COMPONENTS]].copy()
        for name, values in predictions.items():
            arrays[name] = values
        arrays.to_parquet(OUT / "oof.parquet", index=False)
        frame[[*KEYS, "fold", "anchor_time", "episode_id"]].to_parquet(OUT / "validation_keys.parquet", index=False)
        final = {"experiment_id": NAME, "status": "COMPLETE", "decision": "INTERNAL_CANDIDATE_AVAILABLE" if winner != "no_op" else "NO_INTERNAL_GAIN_PROCEED_TO_EPISODE_WEIGHT_BRANCH", "winner": winner, "surface": "reused_historical_development_not_fresh_confirmation", "integrity": integrity, "candidates": results, "meta_fit_receipts": fits, "fit_count": {"historical_meta": len(fits), "selected_full_meta": int(winner != "no_op"), "backbone": 0}, "selected_parameters": deployment, "selected_parameters_reload_max_abs_m": replay_difference, "elapsed_seconds": time.perf_counter() - started, "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "official_access": {"test": 0, "sample": 0, "hidden": 0, "submission_csv": 0, "uploads": 0}, "public_score_fitting": False, "expected_official_score": None, "manifest": receipt, "artifact_sha256": {"oof.parquet": sha(OUT / "oof.parquet"), "selected_meta.json": sha(OUT / "selected_meta.json"), "validation_keys.parquet": sha(OUT / "validation_keys.parquet")}}
        save(REPORT / "result.json", final)
        save(OUT / "terminal_result.json", {"status": "COMPLETE", "decision": final["decision"], "winner": winner, "result_sha256": sha(REPORT / "result.json"), "fit_count": final["fit_count"]})
        return final
    except Exception as error:
        save(OUT / "terminal_result.json", {"status": "TERMINAL_TECHNICAL_FAILURE", "exception_type": type(error).__name__, "message": str(error), "automatic_retry": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("--execute is required; this is an exactly-once research run")
    result = execute()
    print(json.dumps({"status": result["status"], "decision": result["decision"], "winner": result["winner"], "rmse_m": {k: v["rmse_m"] for k, v in result["candidates"].items()}, "fits": result["fit_count"], "official_access": result["official_access"]}, ensure_ascii=False))
