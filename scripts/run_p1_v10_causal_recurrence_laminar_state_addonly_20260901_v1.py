"""Exactly-once causal recurrence/laminar-state P1 falsification."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_v10_causal_recurrence_laminar_state_addonly_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
V9_RUNNER = ROOT / "scripts/run_p1_v9_conditional_dependence_change_addonly_20260901_v1.py"
KEY_COLUMNS = ("station", "year", "layer", "time")
PART_COLUMNS = (*KEY_COLUMNS, "row_position", "baseline_prediction")
POINTS_PER_F1 = 0.6778 / 0.0255
TRANSPORT_FACTOR = 0.30
CADENCE_NS = 600_000_000_000


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("module load failed")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


base = _module(V9_RUNNER, "p1_v10_shared_helpers")
scorer = base.scorer


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        ).encode() + b"\n"
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _time_ns(values: pd.Series | pd.Index) -> np.ndarray:
    return base._time_ns(values)


def recurrence_laminar_features(
    frame: pd.DataFrame,
    train_boundary_ns: int,
    representation: dict[str, Any],
) -> np.ndarray:
    """Causal lag recurrence without subsequence normalization."""

    lags = tuple(int(value) for value in representation["lags_rows"])
    radius = float(representation["recurrence_radius_prefix_scale"])
    laminar_radius = float(representation["laminar_radius_prefix_scale"])
    output = np.zeros((len(frame), 10), dtype=np.float32)
    for _key, group in frame.groupby(
        ["station", "layer"],
        sort=True,
        observed=True,
    ):
        ordered = group.sort_values("_time", kind="stable")
        positions = ordered.index.to_numpy(np.int64)
        times = _time_ns(ordered["_time"])
        values = ordered["temp"].to_numpy(np.float64)
        prefix_values = values[(times <= train_boundary_ns) & np.isfinite(values)]
        if len(prefix_values) < 96:
            continue
        center = float(np.median(prefix_values))
        scale = float(1.4826 * np.median(np.abs(prefix_values - center)))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = float(np.std(prefix_values))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        normalized = np.nan_to_num(
            (values - center) / scale,
            nan=0.0,
            posinf=12.0,
            neginf=-12.0,
        )
        normalized = np.clip(normalized, -12.0, 12.0)
        segments = np.cumsum(np.r_[True, np.diff(times) != CADENCE_NS])
        increment = np.r_[0.0, np.diff(normalized)]
        recurrence = np.zeros((len(ordered), len(lags)), dtype=np.float64)
        diagonal = np.zeros_like(recurrence)
        for column, lag in enumerate(lags):
            if lag >= len(ordered):
                continue
            valid = segments[lag:] == segments[:-lag]
            distance = np.abs(normalized[lag:] - normalized[:-lag])
            increment_distance = np.abs(increment[lag:] - increment[:-lag])
            recurrence[lag:, column] = np.where(
                valid,
                np.exp(-distance / radius),
                0.0,
            )
            diagonal[lag:, column] = np.where(
                valid,
                np.exp(-increment_distance / radius),
                0.0,
            )
        recurrence_sum = recurrence.sum(axis=1)
        probability = np.divide(
            recurrence,
            recurrence_sum[:, None],
            out=np.zeros_like(recurrence),
            where=recurrence_sum[:, None] > 0,
        )
        entropy_terms = np.zeros_like(probability)
        positive_probability = probability > 0
        entropy_terms[positive_probability] = probability[positive_probability] * np.log(
            probability[positive_probability]
        )
        entropy = -np.sum(entropy_terms, axis=1) / math.log(len(lags))
        laminar = np.zeros(len(ordered), dtype=np.float64)
        for row in range(1, len(ordered)):
            if (
                segments[row] == segments[row - 1]
                and abs(normalized[row] - normalized[row - 1]) <= laminar_radius
            ):
                laminar[row] = min(96.0, laminar[row - 1] + 1.0)
        quantiles = np.quantile(
            (prefix_values - center) / scale,
            representation["prefix_reference_quantiles"],
        )
        reference_distance = np.abs(normalized[:, None] - quantiles[None, :])
        recurrence_mean = recurrence.mean(axis=1)
        persistence = (
            pd.Series(recurrence_mean)
            .rolling(representation["persistence_rows"], min_periods=1)
            .mean()
            .to_numpy()
        )
        reference_mean = (
            pd.Series(recurrence_mean)
            .rolling(representation["reference_rows"], min_periods=1)
            .mean()
            .to_numpy()
        )
        feature = np.column_stack(
            [
                recurrence_mean,
                recurrence.max(axis=1),
                entropy,
                diagonal.mean(axis=1),
                diagonal.max(axis=1),
                laminar / 96.0,
                np.mean(reference_distance <= radius, axis=1),
                np.minimum(reference_distance.min(axis=1), 12.0) / 12.0,
                persistence,
                np.abs(recurrence_mean - reference_mean),
            ]
        )
        if not np.isfinite(feature).all():
            raise RuntimeError("recurrence features are nonfinite")
        output[positions] = feature.astype(np.float32)
    return output


def preflight(data_dir: Path) -> dict[str, Any]:
    if ARTIFACT.exists() or LOCK.exists():
        raise FileExistsError("namespace consumed")
    config = _read(CONFIG)
    resolved = data_dir.resolve(strict=True)
    readme = (resolved / "README.md").resolve(strict=True)
    train = (resolved / "train.csv").resolve(strict=True)
    if readme.parent != resolved or train.parent != resolved:
        raise RuntimeError("source path escaped P1_DATA_DIR")
    source = config["source"]
    if source["allowed_files"] != ["README.md", "train.csv"]:
        raise RuntimeError("source allowlist drifted")
    if _sha(readme) != source["readme_sha256"] or _sha(train) != source["train_sha256"]:
        raise RuntimeError("source binding invalid")
    audit = config["semantic_audit"]
    if audit["decision"] != "NOVEL_P1_REPRESENTATION_PROCEED_ONCE":
        raise RuntimeError("semantic gate closed")
    if audit["exact_duplicate"] or audit["semantic_duplicate"]:
        raise RuntimeError("duplicate representation is forbidden")
    for relative, expected in audit["evidence"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"semantic evidence drifted: {relative}")
    frame = pd.read_csv(train, usecols=["station", "layer", "time", "temp"])
    frame["_time"] = pd.to_datetime(
        frame["time"],
        utc=True,
        errors="raise",
        format="mixed",
    )
    all_ns = _time_ns(frame["_time"])
    parts = {}
    boundaries = []
    for fold, item in config["parts"].items():
        audit_part = _read(ROOT / item["audit"])
        if _sha(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError("champion part binding invalid")
        if audit_part["target_fold_validation_labels_read_before_prediction"] != 0:
            raise RuntimeError("champion target isolation failed")
        cutoff = pd.Timestamp(audit_part["adjusted_cutoff_utc"]).value
        prefix = np.sort(np.unique(all_ns[all_ns <= cutoff]))
        boundary = int(
            prefix[
                max(
                    0,
                    int(len(prefix) * config["selection"]["inner_train_fraction"]) - 1,
                )
            ]
        )
        if not boundary < cutoff:
            raise RuntimeError("inner boundary must precede cutoff")
        boundaries.append(boundary)
        parts[fold] = {
            "cutoff": pd.Timestamp(cutoff, tz="UTC").isoformat(),
            "boundary": pd.Timestamp(boundary, tz="UTC").isoformat(),
        }
    if len(set(boundaries)) != 3:
        raise RuntimeError("cutoff boundaries are not distinct")
    support = recurrence_laminar_features(frame, boundaries[0], config["representation"])
    variances = np.var(support, axis=0)
    nonzero_share = float(np.mean(np.any(np.abs(support) > 1e-12, axis=1)))
    gate = config["representation_support_gate"]
    if (
        nonzero_share < gate["minimum_nonzero_feature_share"]
        or float(variances.max()) < gate["minimum_feature_variance"]
    ):
        raise RuntimeError(gate["failure"])
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS_ZERO_OPERATION",
        "surface": config["surface"],
        "source": {"readme": str(readme), "train": str(train)},
        "config_sha256": _sha(CONFIG),
        "runner_sha256": _sha(Path(__file__)),
        "parts": parts,
        "semantic_audit": audit,
        "representation_support": {
            "nonzero_feature_share": nonzero_share,
            "feature_variances": variances.tolist(),
            "gate": "PASS",
        },
        "counters": {"fits": 0, "targets": 0, "official": 0, "csv": 0, "uploads": 0},
    }


def execute(data_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    ready = preflight(data_dir)
    config = _read(CONFIG)
    _write(
        LOCK,
        {
            "experiment_id": EXPERIMENT_ID,
            "status": "CONSUMED_EXACTLY_ONCE",
            "config_sha256": ready["config_sha256"],
            "runner_sha256": ready["runner_sha256"],
        },
    )
    ARTIFACT.mkdir(exist_ok=False)
    _write(ARTIFACT / "preflight.json", ready)
    frame = pd.read_csv(
        ready["source"]["train"],
        usecols=[
            "station",
            "year",
            "layer",
            "time",
            "temp",
            "label",
            "anomaly_type",
        ],
    )
    frame["_time"] = pd.to_datetime(
        frame["time"],
        utc=True,
        errors="raise",
        format="mixed",
    )
    times_ns = _time_ns(frame["_time"])
    labels = frame["label"].to_numpy(np.int8)
    seals = []
    fit_count = 0
    for fold, part_config in config["parts"].items():
        cutoff = pd.Timestamp(ready["parts"][fold]["cutoff"]).value
        boundary = pd.Timestamp(ready["parts"][fold]["boundary"]).value
        features = recurrence_laminar_features(frame, boundary, config["representation"])
        train_mask = times_ns <= boundary
        inner_mask = (times_ns > boundary) & (times_ns <= cutoff)
        probabilities = []
        model_hashes = []
        for seed in config["model"]["seeds"]:
            scaler = StandardScaler().fit(features[train_mask])
            model = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=config["model"]["alpha"],
                max_iter=config["model"]["epochs"],
                tol=None,
                class_weight={0: 1.0, 1: config["model"]["positive_class_weight"]},
                shuffle=True,
                random_state=seed,
            )
            model.fit(scaler.transform(features[train_mask]), labels[train_mask])
            probabilities.append(
                model.predict_proba(scaler.transform(features))[:, 1].astype(np.float32)
            )
            model_hashes.append(base._model_hash(model, scaler))
            fit_count += 1
        scores = np.mean(np.stack(probabilities), axis=0)
        selection = base._select(scores[inner_mask], labels[inner_mask], config["selection"])
        part = pd.read_parquet(ROOT / part_config["path"], columns=list(PART_COLUMNS))
        positions = part["row_position"].to_numpy(np.int64)
        incumbent = part["baseline_prediction"].to_numpy(np.int8)
        outer_scores = scores[positions]
        additions = base._additions(
            outer_scores,
            incumbent,
            selection["chosen"],
            config["selection"]["maximum_addition_share"],
        )
        candidate = np.bitwise_or(incumbent, additions.astype(np.int8))
        path = ARTIFACT / f"{fold}_sealed.npz"
        np.savez_compressed(
            path,
            positions=positions,
            incumbent=incumbent,
            additions=additions,
            candidate=candidate,
            scores=outer_scores,
        )
        seal = {
            "fold": fold,
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha(path),
            "selection": selection,
            "model_hashes": model_hashes,
            "fits": 3,
            "outer_target_reads_before_seal": 0,
        }
        _write(ARTIFACT / f"{fold}_seal.json", seal)
        seals.append(seal)
    _write(
        ARTIFACT / "predictions_complete.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "fits": fit_count,
            "seals": seals,
            "outer_target_reads_before_all_seals": 0,
        },
    )
    fold_scores = []
    pool = {
        key: []
        for key in ("truth", "incumbent", "candidate", "additions", "types", "metadata")
    }
    for seal in seals:
        with np.load(ROOT / seal["path"], allow_pickle=False) as values:
            positions = values["positions"]
            incumbent = values["incumbent"]
            additions = values["additions"]
            candidate = values["candidate"]
        truth = labels[positions]
        metadata = frame.iloc[positions].loc[:, list(KEY_COLUMNS)].reset_index(drop=True)
        types = frame.iloc[positions]["anomaly_type"].reset_index(drop=True)
        fold_scores.append(
            {
                "fold": seal["fold"],
                **scorer._score_surface(
                    truth,
                    incumbent,
                    candidate,
                    additions,
                    types,
                    metadata,
                ),
            }
        )
        for key, value in (
            ("truth", truth),
            ("incumbent", incumbent),
            ("candidate", candidate),
            ("additions", additions),
            ("types", types),
            ("metadata", metadata),
        ):
            pool[key].append(value)
    truth, incumbent, candidate, additions = (
        np.concatenate(pool[key])
        for key in ("truth", "incumbent", "candidate", "additions")
    )
    types = pd.concat(pool["types"], ignore_index=True)
    metadata = pd.concat(pool["metadata"], ignore_index=True)
    pooled = scorer._score_surface(
        truth,
        incumbent,
        candidate,
        additions,
        types,
        metadata,
    )
    bootstrap = scorer._paired_cluster_bootstrap(
        truth,
        incumbent,
        candidate,
        metadata,
        replicates=config["decision"]["bootstrap_replicates"],
        seed=config["decision"]["seed"],
    )
    passed = (
        pooled["delta_f1"] > 0
        and bootstrap["ci90"][0] >= 0
        and all(item["delta_f1"] >= 0 for item in fold_scores)
    )
    result = {
        "schema_version": "p1.v10_recurrence_laminar.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "surface": config["surface"],
        "decision": config["decision"]["pass"] if passed else config["decision"]["fail"],
        "pooled": pooled,
        "fold_scores": fold_scores,
        "block_bootstrap": bootstrap,
        "long_event_interior": base._long_event_interior(
            truth,
            incumbent,
            candidate,
            metadata,
        ),
        "worst_slices": sorted(
            pooled["station_layer_diagnostics"],
            key=lambda item: item["delta_f1"],
        )[:10],
        "action_slices": base._action_slices(additions, metadata),
        "points": {
            "nominal": pooled["delta_f1"] * POINTS_PER_F1,
            "transport_adjusted": pooled["delta_f1"] * POINTS_PER_F1 * TRANSPORT_FACTOR,
        },
        "counters": {
            "fits": fit_count,
            "anchor_removals": pooled["incumbent_positive_removals"],
            "outer_target_reads_before_all_seals": 0,
            "official": 0,
            "csv": 0,
            "uploads": 0,
        },
        "runtime_seconds": time.monotonic() - started,
        "hashes": {
            "config": ready["config_sha256"],
            "runner": ready["runner_sha256"],
            "completion": _sha(ARTIFACT / "predictions_complete.json"),
            "lock": _sha(LOCK),
        },
    }
    _write(ARTIFACT / "result.json", result)
    return result


def qa(data_dir: Path) -> dict[str, Any]:
    ready = preflight(data_dir)
    config = _read(CONFIG)
    checks = {
        "zero": all(value == 0 for value in ready["counters"].values()),
        "novel": config["semantic_audit"]["decision"]
        == "NOVEL_P1_REPRESENTATION_PROCEED_ONCE",
        "support": ready["representation_support"]["gate"] == "PASS",
        "ns_boundaries": len({item["boundary"] for item in ready["parts"].values()}) == 3,
        "past_only": config["representation"]["past_only"],
        "no_future": config["representation"]["future_interpolation"] == 0,
        "no_subsequence_normalization": config["representation"]["subsequence_normalization"] == 0,
        "fits9": config["model"]["fits"] == 9,
        "sweep0": config["model"]["sweep"] == 0,
        "outer_tuning0": config["selection"]["outer_tuning"] == 0,
        "add_only": config["anchor"]["removals"] == 0,
        "access0": config["source"]["official_test_sample_submission_hidden_reads"] == 0,
    }
    return {
        "experiment_id": EXPERIMENT_ID,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--qa", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        value = preflight(args.data_dir)
    elif args.qa:
        value = qa(args.data_dir)
    else:
        value = execute(args.data_dir)
    print(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False), end="")


if __name__ == "__main__":
    main()
