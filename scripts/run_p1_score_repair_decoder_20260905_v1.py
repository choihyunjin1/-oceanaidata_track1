"""One binary sequence decoder; reuses sealed O/B models, no backbone fits."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import joblib
import numpy as np
import pandas as pd
import run_p1_score_repair_20260905_v1 as screen

RUN = "p1_score_repair_decoder_20260905_v1"


def source_runner_path():
    path = Path(screen.__file__).resolve()
    if path != ROOT / "scripts" / "run_p1_score_repair_20260905_v1.py" or not path.is_file():
        raise ValueError("unexpected screen module path")
    return path


def transition_fit(training, laplace=1.0):
    y = training.label.to_numpy(dtype=np.int8)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("binary training targets required")
    seg = screen.segments(training).to_numpy()
    allowed = seg[1:] == seg[:-1]
    counts = np.bincount(2 * y[:-1][allowed] + y[1:][allowed], minlength=4).reshape(2, 2)
    smoothed = counts.astype(float) + laplace
    transitions = smoothed / smoothed.sum(axis=1, keepdims=True)
    initial = np.bincount(y, minlength=2).astype(float) + laplace
    initial /= initial.sum()
    return {
        "transition": transitions.tolist(),
        "initial": initial.tolist(),
        "counts": counts.tolist(),
        "training_rows": len(y),
        "valid_adjacent_pairs": int(allowed.sum()),
        "segment_count": int(len(np.unique(seg))),
        "laplace": laplace,
    }


def decode_viterbi(frame, unary, transition, hard, strength=1.0):
    unary = np.asarray(unary, dtype=float)
    if unary.shape != (len(frame),) or not np.isfinite(unary).all():
        raise ValueError("finite aligned unary required")
    matrix = np.asarray(transition["transition"], dtype=float)
    initial = np.asarray(transition["initial"], dtype=float)
    if (
        not np.isfinite(matrix).all()
        or (matrix <= 0).any()
        or not np.allclose(matrix.sum(axis=1), 1)
    ):
        raise ValueError("invalid transition matrix")
    if not np.isfinite(initial).all() or (initial <= 0).any() or not np.isclose(initial.sum(), 1):
        raise ValueError("invalid initial distribution")
    log_t, log_i = strength * np.log(matrix), strength * np.log(initial)
    result = np.zeros(len(frame), dtype=np.int8)
    for positions in frame.groupby(screen.segments(frame), sort=False).indices.values():
        scores = unary[positions]
        back = np.zeros((len(positions), 2), dtype=np.int8)
        d0, d1 = log_i[0], log_i[1] + scores[0]
        for row in range(1, len(scores)):
            a00, a10 = d0 + log_t[0, 0], d1 + log_t[1, 0]
            a01, a11 = d0 + log_t[0, 1], d1 + log_t[1, 1]
            back[row, 0], back[row, 1] = int(a10 > a00), int(a11 > a01)
            d0, d1 = max(a00, a10), max(a01, a11) + scores[row]
        state = int(d1 > d0)
        for row in range(len(scores) - 1, -1, -1):
            result[positions[row]] = state
            state = int(back[row, state])
    result[np.asarray(hard, dtype=bool)] = 1
    return result


def control_components(frame, probabilities, rules, cfg, selection, calibrations, clip=1e-6):
    relevant = {
        "original": [("original", "original")],
        "balanced": [("balanced", "balanced")],
        "original_balanced_union": [("original", "original"), ("balanced", "balanced")],
        "balanced_union": [("original", "original"), ("balanced", "balanced_union")],
    }[selection]
    bits, scores, confirmed_spikes = [], [], []
    for model, calibration in relevant:
        p = np.clip(probabilities[model], clip, 1 - clip)
        threshold = calibrations[calibration]["threshold"]
        bits.append(screen.decode(frame, p, rules, cfg, threshold))
        scores.append(np.log(p) - np.log1p(-p) - np.log(threshold) + np.log1p(-threshold))
        confirmed_spikes.append(rules[1] & (p >= threshold))
    reference = np.maximum.reduce(bits)
    unary = np.maximum.reduce(scores)
    hard = (reference == 1) & (rules[0] | np.logical_or.reduce(confirmed_spikes))
    return reference, unary, hard


def run(config_path):
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg["experiment_id"] != RUN or cfg["official_access_authorized"]:
        raise ValueError("contract violation")
    artifact, report = ROOT / "artifacts" / RUN, ROOT / "reports" / RUN
    artifact.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    source = ROOT / "artifacts" / cfg["source_screen"]
    source_result = source / "terminal_result.json"
    if screen.sha(source_result) != cfg["source_result_sha256"]:
        raise ValueError("source result hash mismatch")
    result = json.loads(source_result.read_text(encoding="utf-8"))
    frozen = json.loads((source / "contract.json").read_text(encoding="utf-8"))
    if screen.sha(source_runner_path()) != result["runner_sha256"]:
        raise ValueError("source runner changed")
    start = time.monotonic()
    receipt = {
        "experiment_id": RUN,
        "status": "RUNNING",
        "pid": os.getpid(),
        "backbone_fits": 0,
        "transition_estimates": 0,
        "inner_on_off_selections": 0,
        "official_rows": 0,
        "csv_written": 0,
        "upload": 0,
        "config_sha256": screen.sha(config_path),
        "code_sha256": screen.sha(__file__),
        "source_result_sha256": screen.sha(source_result),
        "folds": [],
    }
    with (artifact / "ATTEMPT_LOCK.json").open("x", encoding="utf-8") as f:
        json.dump(receipt, f)
    screen.write_json(artifact / "contract.json", cfg)
    try:
        train_path = Path(os.environ["P1_DATA_DIR"]) / "train.csv"
        if screen.sha(train_path) != result["train_sha256"]:
            raise ValueError("distributed training hash mismatch")
        frame = pd.read_csv(train_path, usecols=screen.RAW + ["label", "anomaly_type"])
        frame["row_id"] = np.arange(len(frame))
        frame.sort_values(["station", "layer", "time"], kind="stable", inplace=True)
        frame.reset_index(drop=True, inplace=True)
        times = pd.to_datetime(frame.time, utc=True)
        pooled = []
        for fold, selection in zip(frozen["folds"], result["folds"], strict=True):
            name, begin, end = fold["name"], pd.Timestamp(fold["start"]), pd.Timestamp(fold["end"])
            cutoff = begin - pd.Timedelta(days=frozen["purge_days"])
            inner_begin = cutoff - pd.Timedelta(days=frozen["inner_days"])
            fold_result = {"fold": name, "control": selection["selected_control"]}
            enabled = False
            for stage, eval_start, eval_end, train_end in [
                (
                    "inner",
                    inner_begin,
                    cutoff,
                    inner_begin - pd.Timedelta(days=frozen["purge_days"]),
                ),
                ("outer", begin, end, cutoff),
            ]:
                ev = frame.loc[(times >= eval_start) & (times < eval_end)].reset_index(drop=True)
                training = screen.train_slice(frame, train_end)
                transition = transition_fit(training, cfg["laplace"])
                receipt["transition_estimates"] += 1
                package = joblib.load(source / f"{name}_{stage}_balanced.joblib")
                bundle = screen.feature_pair(ev, package["train_stats"], frozen)[0]
                rules = screen.rule_masks(ev, package["train_stats"])
                probabilities = {}
                for model in ["original", "balanced"]:
                    path = source / f"{name}_{stage}_{model}.joblib"
                    expected = next(
                        r
                        for r in result["fit_receipts"]
                        if r["fold"] == name and r["stage"] == stage and r["model"] == model
                    )
                    if screen.sha(path) != expected["model_sha256"]:
                        raise ValueError("backbone hash mismatch")
                    loaded = joblib.load(path)
                    probabilities[model] = loaded["model"].predict_proba(
                        loaded["encoder"].transform(bundle)
                    )[:, 1]
                reference, unary, hard = control_components(
                    ev,
                    probabilities,
                    rules,
                    frozen,
                    selection["selected_control"],
                    selection["calibrations"],
                    cfg["probability_clip"],
                )
                decoded = decode_viterbi(ev, unary, transition, hard, cfg["lambda"])
                ref_metric, decoder_metric = (
                    screen.metric(ev.label, reference),
                    screen.metric(ev.label, decoded),
                )
                if stage == "inner":
                    enabled = decoder_metric["f1"] > ref_metric["f1"]
                    receipt["inner_on_off_selections"] += 1
                else:
                    prior = pd.read_parquet(source / f"{name}_intact_oof.parquet")
                    if ev.row_id.to_list() != prior.row_id.to_list() or not np.array_equal(
                        reference, prior.selected_control.to_numpy()
                    ):
                        raise ValueError("control OOF mismatch")
                chosen = decoded if enabled else reference
                part = ev[screen.KEYS + ["row_id", "label", "anomaly_type"]].copy()
                part["fold"], part["stage"] = name, stage
                part["reference"], part["raw_decoder"], part["prediction"] = (
                    reference,
                    decoded,
                    chosen,
                )
                part["unary"], part["hard"] = unary, hard
                part.to_parquet(artifact / f"{name}_{stage}_oof.parquet", index=False)
                fold_result[stage] = {
                    "reference": ref_metric,
                    "raw_decoder": decoder_metric,
                    "chosen": screen.metric(ev.label, chosen),
                    "enabled_from_inner": enabled,
                    "transition": transition,
                    "hard_rows": int(hard.sum()),
                    "hard_removed": int((hard & (decoded == 0)).sum()),
                    "added_tp": int(
                        ((chosen == 1) & (reference == 0) & (ev.label.to_numpy() == 1)).sum()
                    ),
                    "added_fp": int(
                        ((chosen == 1) & (reference == 0) & (ev.label.to_numpy() == 0)).sum()
                    ),
                    "removed_tp": int(
                        ((chosen == 0) & (reference == 1) & (ev.label.to_numpy() == 1)).sum()
                    ),
                    "removed_fp": int(
                        ((chosen == 0) & (reference == 1) & (ev.label.to_numpy() == 0)).sum()
                    ),
                }
                if stage == "outer":
                    pooled.append(part)
            receipt["folds"].append(fold_result)
            receipt["runtime_seconds"] = time.monotonic() - start
            screen.write_json(artifact / "progress.json", receipt)
            print(
                json.dumps(
                    {
                        "fold": name,
                        "completed_folds": len(receipt["folds"]),
                        "runtime_seconds": receipt["runtime_seconds"],
                    }
                ),
                flush=True,
            )
            if receipt["runtime_seconds"] > cfg["wall_cap_seconds"]:
                raise RuntimeError("decoder wall cap exceeded")
        combined = pd.concat(pooled, ignore_index=True)
        receipt["pooled"] = {
            k: screen.metric(combined.label, combined[k])
            for k in ["reference", "raw_decoder", "prediction"]
        }
        receipt["delta_f1"] = (
            receipt["pooled"]["prediction"]["f1"] - receipt["pooled"]["reference"]["f1"]
        )
        receipt["conditional_public_points_if_transferred"] = receipt["delta_f1"] * 26.6
        receipt["status"] = (
            "DECODER_POSITIVE" if receipt["delta_f1"] > 0 else "DECODER_NO_GO_RETAIN_CLEAN_CONTROL"
        )
        np.savez_compressed(
            artifact / "qa_oof.npz",
            key=combined[screen.KEYS].astype(str).agg("|".join, axis=1).to_numpy(dtype=str),
            fold=combined.fold.to_numpy(dtype=str),
            truth=combined.label.to_numpy(dtype=np.int8),
            reference=combined.reference.to_numpy(dtype=np.int8),
            prediction=combined.prediction.to_numpy(dtype=np.int8),
        )
        if receipt["transition_estimates"] != 6 or receipt["inner_on_off_selections"] != 3:
            raise ValueError("transition/selection count mismatch")
    except Exception as exc:
        receipt.update(status="TERMINAL_TECHNICAL_FAILURE", error=str(exc))
        raise
    finally:
        receipt["runtime_seconds"] = time.monotonic() - start
        screen.write_json(report / "result.json", receipt)
        screen.write_json(artifact / "terminal_result.json", receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/experiments" / (RUN + ".json")
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("--execute required")
    run(args.config)
