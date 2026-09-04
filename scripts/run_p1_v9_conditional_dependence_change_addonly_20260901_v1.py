"""Exactly-once prefix conditional-dependence-change P1 falsification."""

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
EXPERIMENT_ID = "p1_v9_conditional_dependence_change_addonly_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
SCORER_PATH = ROOT / "scripts/run_p1_clean_state_capa_falsification_20260831_v1.py"
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


scorer = _module(SCORER_PATH, "p1_v9_score_helpers")


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
    index = pd.DatetimeIndex(values)
    if index.tz is None or index.hasnans:
        raise RuntimeError("time must be complete and timezone-aware")
    result = index.as_unit("ns").asi8
    lower = pd.Timestamp("2020-01-01T00:00:00Z").value
    upper = pd.Timestamp("2030-01-01T00:00:00Z").value
    if len(result) and (result.min() < lower or result.max() >= upper):
        raise RuntimeError("time integer is not authorized epoch nanoseconds")
    return result


def _source_paths(data_dir: Path) -> tuple[Path, Path]:
    resolved = data_dir.resolve(strict=True)
    readme = (resolved / "README.md").resolve(strict=True)
    train = (resolved / "train.csv").resolve(strict=True)
    if readme.parent != resolved or train.parent != resolved:
        raise RuntimeError("source path escaped P1_DATA_DIR")
    return readme, train


def _partial_correlation(precision: np.ndarray) -> np.ndarray:
    diagonal = np.sqrt(np.clip(np.diag(precision), 1e-12, None))
    result = -precision / np.outer(diagonal, diagonal)
    np.fill_diagonal(result, 0.0)
    return result


def conditional_dependence_features(
    frame: pd.DataFrame,
    train_boundary_ns: int,
    half_life_rows: int,
    ridge: float,
) -> np.ndarray:
    """Return causal covariance, precision, and partial-correlation changes."""

    if half_life_rows <= 0 or ridge <= 0:
        raise ValueError("half-life and ridge must be positive")
    output = np.zeros((len(frame), 5), dtype=np.float32)
    for _station, group in frame.groupby("station", sort=True, observed=True):
        if group.duplicated(["_time", "layer"]).any():
            raise RuntimeError("duplicate station-layer timestamp")
        wide = (
            group.pivot(index="_time", columns="layer", values="temp")
            .sort_index(kind="stable")
            .sort_index(axis=1)
        )
        layers = wide.columns.to_numpy()
        layer_count = len(layers)
        if layer_count < 2:
            continue
        times_ns = _time_ns(wide.index)
        prefix = times_ns <= train_boundary_ns
        if prefix.sum() < 20:
            raise RuntimeError("dependence prefix has too few synchronized rows")
        prefix_values = wide.loc[prefix].to_numpy(np.float64)
        means = np.zeros(layer_count, dtype=np.float64)
        scales = np.ones(layer_count, dtype=np.float64)
        prefix_supported = np.zeros(layer_count, dtype=bool)
        for layer_index in range(layer_count):
            observed = prefix_values[:, layer_index]
            observed = observed[np.isfinite(observed)]
            if len(observed):
                prefix_supported[layer_index] = True
                means[layer_index] = observed.mean()
                scale = observed.std()
                scales[layer_index] = scale if scale > 1e-6 else 1.0
        standardized = (wide - means) / scales
        standardized = standardized.ffill().fillna(0.0)
        standardized.loc[:, ~prefix_supported] = 0.0
        values = standardized.to_numpy(np.float64)
        baseline_covariance = np.cov(values[prefix], rowvar=False, bias=True)
        baseline_covariance = np.atleast_2d(baseline_covariance)
        identity = np.eye(layer_count, dtype=np.float64)
        baseline_precision = np.linalg.inv(baseline_covariance + ridge * identity)
        baseline_partial = _partial_correlation(baseline_precision)
        recent_mean = standardized.ewm(
            halflife=half_life_rows,
            adjust=False,
        ).mean().to_numpy(np.float64)
        recent_covariance = np.empty(
            (len(wide), layer_count, layer_count),
            dtype=np.float32,
        )
        for left in range(layer_count):
            for right in range(left, layer_count):
                product = standardized.iloc[:, left] * standardized.iloc[:, right]
                product_mean = product.ewm(
                    halflife=half_life_rows,
                    adjust=False,
                ).mean().to_numpy(np.float64)
                covariance = product_mean - recent_mean[:, left] * recent_mean[:, right]
                recent_covariance[:, left, right] = covariance
                recent_covariance[:, right, left] = covariance
        station_features = np.zeros((len(wide), layer_count, 5), dtype=np.float32)
        for start in range(0, len(wide), 4096):
            end = min(len(wide), start + 4096)
            covariance = recent_covariance[start:end].astype(np.float64)
            precision = np.linalg.inv(covariance + ridge * identity[None, :, :])
            diagonal = np.sqrt(
                np.clip(np.diagonal(precision, axis1=1, axis2=2), 1e-12, None)
            )
            partial = -precision / (diagonal[:, :, None] * diagonal[:, None, :])
            index = np.arange(layer_count)
            partial[:, index, index] = 0.0
            covariance_change = covariance - baseline_covariance
            precision_change = precision - baseline_precision
            partial_change = partial - baseline_partial
            normalizer = math.sqrt(layer_count * layer_count)
            station_features[start:end, :, 0] = (
                np.linalg.norm(covariance_change, axis=(1, 2)) / normalizer
            )[:, None]
            station_features[start:end, :, 1] = (
                np.linalg.norm(precision_change, axis=(1, 2)) / normalizer
            )[:, None]
            station_features[start:end, :, 2] = (
                np.linalg.norm(partial_change, axis=(1, 2)) / normalizer
            )[:, None]
            station_features[start:end, :, 3] = np.mean(
                np.abs(precision_change),
                axis=2,
            )
            station_features[start:end, :, 4] = np.mean(
                np.abs(partial_change),
                axis=2,
            )
        time_positions = wide.index.get_indexer(group["_time"])
        layer_positions = pd.Index(layers).get_indexer(group["layer"])
        if (time_positions < 0).any() or (layer_positions < 0).any():
            raise RuntimeError("station panel alignment failed")
        output[group.index.to_numpy(np.int64)] = station_features[
            time_positions,
            layer_positions,
        ]
    if not np.isfinite(output).all():
        raise RuntimeError("conditional-dependence features are nonfinite")
    return output


def preflight(data_dir: Path) -> dict[str, Any]:
    if ARTIFACT.exists() or LOCK.exists():
        raise FileExistsError("namespace consumed")
    config = _read(CONFIG)
    readme, train = _source_paths(data_dir)
    source = config["source"]
    if source["allowed_files"] != ["README.md", "train.csv"]:
        raise RuntimeError("source allowlist drifted")
    if _sha(readme) != source["readme_sha256"] or _sha(train) != source["train_sha256"]:
        raise RuntimeError("source binding invalid")
    audit = config["semantic_audit"]
    if audit["decision"] != "NOVEL_REPRESENTATION_PROCEED_ONCE":
        raise RuntimeError("semantic gate closed")
    if audit["exact_duplicate"] or audit["semantic_duplicate"]:
        raise RuntimeError("duplicate architecture is forbidden")
    for relative, expected in audit["evidence"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"semantic evidence drifted: {relative}")
    frame = pd.read_csv(
        train,
        usecols=["station", "layer", "time", "temp"],
    )
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
        part_audit = _read(ROOT / item["audit"])
        if _sha(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError("champion part binding invalid")
        if part_audit["target_fold_validation_labels_read_before_prediction"] != 0:
            raise RuntimeError("champion target isolation failed")
        cutoff_ns = pd.Timestamp(part_audit["adjusted_cutoff_utc"]).value
        prefix = np.sort(np.unique(all_ns[all_ns <= cutoff_ns]))
        boundary = int(
            prefix[
                max(
                    0,
                    int(len(prefix) * config["selection"]["inner_train_fraction"]) - 1,
                )
            ]
        )
        if not boundary < cutoff_ns:
            raise RuntimeError("inner boundary must precede cutoff")
        boundaries.append(boundary)
        parts[fold] = {
            "cutoff": pd.Timestamp(cutoff_ns, tz="UTC").isoformat(),
            "boundary": pd.Timestamp(boundary, tz="UTC").isoformat(),
        }
    if len(set(boundaries)) != 3:
        raise RuntimeError("cutoff-specific boundaries are not distinct")
    representation = config["representation"]
    support_features = conditional_dependence_features(
        frame,
        boundaries[0],
        representation["ewm_half_life_rows"],
        representation["ridge"],
    )
    variances = np.var(support_features, axis=0)
    nonzero_share = float(np.mean(np.any(np.abs(support_features) > 1e-12, axis=1)))
    multilayer = int(
        (frame.groupby("station", observed=True)["layer"].nunique() >= 2).sum()
    )
    gate = config["representation_support_gate"]
    if (
        multilayer < gate["minimum_multilayer_stations"]
        or nonzero_share < gate["minimum_nonzero_feature_share"]
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
            "multilayer_stations": multilayer,
            "nonzero_feature_share": nonzero_share,
            "feature_variances": variances.tolist(),
            "gate": "PASS",
        },
        "counters": {"fits": 0, "targets": 0, "official": 0, "csv": 0, "uploads": 0},
    }


def _wilson_lower(successes: int, count: int, z: float = 1.6448536269514722) -> float:
    if count == 0:
        return 0.0
    rate = successes / count
    denominator = 1 + z * z / count
    center = rate + z * z / (2 * count)
    margin = z * math.sqrt(
        rate * (1 - rate) / count + z * z / (4 * count * count)
    )
    return (center - margin) / denominator


def _select(scores: np.ndarray, labels: np.ndarray, selection: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for quantile in selection["threshold_quantiles"]:
        threshold = float(np.quantile(scores, quantile))
        use = scores >= threshold
        count = int(use.sum())
        true = int(labels[use].sum())
        candidates.append(
            {
                "quantile": quantile,
                "threshold": threshold,
                "count": count,
                "share": count / len(scores),
                "precision": true / count if count else 0.0,
                "precision_lcb": _wilson_lower(true, count),
            }
        )
    eligible = [
        item
        for item in candidates
        if item["count"] >= selection["minimum_additions"]
        and item["share"] <= selection["maximum_addition_share"]
        and item["precision_lcb"] >= selection["wilson90_lcb_minimum"]
    ]
    return {"chosen": eligible[0] if eligible else None, "candidates": candidates}


def _additions(
    scores: np.ndarray,
    incumbent: np.ndarray,
    chosen: dict[str, Any] | None,
    share: float,
) -> np.ndarray:
    result = np.zeros(len(scores), dtype=bool)
    if chosen is None:
        return result
    eligible = np.flatnonzero((incumbent == 0) & (scores >= chosen["threshold"]))
    maximum = int(math.floor(len(scores) * share))
    if len(eligible) > maximum:
        eligible = eligible[np.lexsort((eligible, -scores[eligible]))[:maximum]]
    result[eligible] = True
    return result


def _model_hash(model: SGDClassifier, scaler: StandardScaler) -> str:
    digest = hashlib.sha256()
    for value in (model.coef_, model.intercept_, scaler.mean_, scaler.scale_):
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def _long_event_interior(
    truth: np.ndarray,
    incumbent: np.ndarray,
    candidate: np.ndarray,
    metadata: pd.DataFrame,
) -> dict[str, Any]:
    work = metadata.loc[:, ["station", "layer", "time"]].copy()
    work["_position"] = np.arange(len(work))
    work["_ns"] = _time_ns(
        pd.to_datetime(work["time"], utc=True, errors="raise", format="mixed")
    )
    interior: list[int] = []
    runs = 0
    for _key, group in work.groupby(["station", "layer"], sort=True, observed=True):
        ordered = group.sort_values("_ns", kind="stable")
        positions = ordered["_position"].to_numpy(np.int64)
        times = ordered["_ns"].to_numpy(np.int64)
        positive = truth[positions].astype(bool)
        start = 0
        while start < len(positions):
            if not positive[start]:
                start += 1
                continue
            end = start + 1
            while end < len(positions) and positive[end] and times[end] - times[end - 1] == CADENCE_NS:
                end += 1
            if end - start >= 18:
                runs += 1
                interior.extend(positions[start + 6 : end - 6].tolist())
            start = end
    selected = np.asarray(interior, dtype=np.int64)
    anchor = float(incumbent[selected].mean()) if len(selected) else 0.0
    proposed = float(candidate[selected].mean()) if len(selected) else 0.0
    return {
        "definition": "positive 10-minute runs >=18 rows; exclude 6 boundary rows per side",
        "runs": runs,
        "interior_rows": len(selected),
        "anchor_recall": anchor,
        "candidate_recall": proposed,
        "delta_recall": proposed - anchor,
    }


def _action_slices(additions: np.ndarray, metadata: pd.DataFrame) -> list[dict[str, Any]]:
    work = metadata.loc[:, ["station", "layer"]].copy()
    work["addition"] = additions
    rows = []
    for (station, layer), group in work.groupby(
        ["station", "layer"],
        sort=True,
        observed=True,
    ):
        count = int(group["addition"].sum())
        if count:
            rows.append({"station": str(station), "layer": int(layer), "additions": count})
    return sorted(rows, key=lambda item: (-item["additions"], item["station"], item["layer"]))


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
        cutoff_ns = pd.Timestamp(ready["parts"][fold]["cutoff"]).value
        boundary_ns = pd.Timestamp(ready["parts"][fold]["boundary"]).value
        features = conditional_dependence_features(
            frame,
            boundary_ns,
            config["representation"]["ewm_half_life_rows"],
            config["representation"]["ridge"],
        )
        train_mask = times_ns <= boundary_ns
        inner_mask = (times_ns > boundary_ns) & (times_ns <= cutoff_ns)
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
            model_hashes.append(_model_hash(model, scaler))
            fit_count += 1
        scores = np.mean(np.stack(probabilities), axis=0)
        selection = _select(scores[inner_mask], labels[inner_mask], config["selection"])
        part = pd.read_parquet(ROOT / part_config["path"], columns=list(PART_COLUMNS))
        positions = part["row_position"].to_numpy(np.int64)
        incumbent = part["baseline_prediction"].to_numpy(np.int8)
        outer_scores = scores[positions]
        additions = _additions(
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
        "schema_version": "p1.v9_conditional_dependence_change.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "surface": config["surface"],
        "decision": config["decision"]["pass"] if passed else config["decision"]["fail"],
        "pooled": pooled,
        "fold_scores": fold_scores,
        "block_bootstrap": bootstrap,
        "long_event_interior": _long_event_interior(
            truth,
            incumbent,
            candidate,
            metadata,
        ),
        "worst_slices": sorted(
            pooled["station_layer_diagnostics"],
            key=lambda item: item["delta_f1"],
        )[:10],
        "action_slices": _action_slices(additions, metadata),
        "points": {
            "nominal": pooled["delta_f1"] * POINTS_PER_F1,
            "transport_adjusted": (
                pooled["delta_f1"] * POINTS_PER_F1 * TRANSPORT_FACTOR
            ),
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
        "zero_operation": all(value == 0 for value in ready["counters"].values()),
        "semantic_novel": (
            config["semantic_audit"]["decision"] == "NOVEL_REPRESENTATION_PROCEED_ONCE"
        ),
        "support": ready["representation_support"]["gate"] == "PASS",
        "ns_boundaries_distinct": len(
            {item["boundary"] for item in ready["parts"].values()}
        )
        == 3,
        "past_only": config["representation"]["past_only"],
        "no_future_interpolation": config["representation"]["future_interpolation"] == 0,
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
