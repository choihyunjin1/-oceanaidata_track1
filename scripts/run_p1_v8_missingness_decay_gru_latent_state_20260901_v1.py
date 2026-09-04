"""Exactly-once missingness-decay recurrent latent-state P1 falsification."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_v8_missingness_decay_gru_latent_state_20260901_v1"
CONFIG_PATH = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT_DIR = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK_PATH = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
SCORER_PATH = ROOT / "scripts/run_p1_clean_state_capa_falsification_20260831_v1.py"
INPUT_COLUMNS = ("station", "year", "layer", "time", "temp", "psal", "depth")
KEY_COLUMNS = ("station", "year", "layer", "time")
PART_COLUMNS = (*KEY_COLUMNS, "row_position", "baseline_prediction")
POINTS_PER_F1 = 0.6778 / 0.0255
TRANSPORT_FACTOR = 0.30
HOUR_NS = 3_600_000_000_000


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("module load failed")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


scorer = _module(SCORER_PATH, "p1_v8_score_helpers")


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


def preflight(data_dir: Path) -> dict[str, Any]:
    if ARTIFACT_DIR.exists() or LOCK_PATH.exists():
        raise FileExistsError("namespace consumed")
    config = _read(CONFIG_PATH)
    readme, train = _source_paths(data_dir)
    source = config["source"]
    if source["allowed_files"] != ["README.md", "train.csv"]:
        raise RuntimeError("source allowlist drifted")
    if _sha(readme) != source["readme_sha256"]:
        raise RuntimeError("README binding invalid")
    if _sha(train) != source["train_sha256"]:
        raise RuntimeError("train binding invalid")
    audit = config["semantic_audit"]
    if audit["decision"] != "NOVEL_REPRESENTATION_PROCEED_ONCE":
        raise RuntimeError("semantic gate closed")
    if audit["exact_duplicate"] or audit["semantic_duplicate"]:
        raise RuntimeError("duplicate architecture is forbidden")
    for relative, expected in audit["evidence"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"semantic evidence drifted: {relative}")
    architecture = config["architecture"]
    if architecture["maximum_fits"] != 9 or architecture["sweeps"] != 0:
        raise RuntimeError("fit budget drifted")
    if not architecture["past_only"] or architecture["future_interpolation"] != 0:
        raise RuntimeError("causal representation contract drifted")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("one CUDA device is required")
    properties = torch.cuda.get_device_properties(0)
    input_only = pd.read_csv(
        train,
        usecols=["station", "layer", "time", "temp", "psal", "depth"],
    )
    parsed = pd.to_datetime(
        input_only["time"],
        utc=True,
        errors="raise",
        format="mixed",
    )
    input_only["_time"] = parsed
    all_ns = _time_ns(parsed)
    parts: dict[str, dict[str, str]] = {}
    boundaries: list[int] = []
    for fold, item in config["parts"].items():
        part_path = ROOT / item["path"]
        part_audit = _read(ROOT / item["audit"])
        if _sha(part_path) != item["sha256"]:
            raise RuntimeError(f"champion part binding invalid: {fold}")
        if part_audit["target_fold_validation_labels_read_before_prediction"] != 0:
            raise RuntimeError("champion part target isolation failed")
        cutoff_ns = pd.Timestamp(part_audit["adjusted_cutoff_utc"]).value
        prefix = np.sort(np.unique(all_ns[all_ns <= cutoff_ns]))
        index = max(
            0,
            int(len(prefix) * config["selection"]["inner_train_fraction"]) - 1,
        )
        boundary_ns = int(prefix[index])
        if not boundary_ns < cutoff_ns:
            raise RuntimeError("inner boundary must precede outer cutoff")
        boundaries.append(boundary_ns)
        parts[fold] = {
            "cutoff": pd.Timestamp(cutoff_ns, tz="UTC").isoformat(),
            "boundary": pd.Timestamp(boundary_ns, tz="UTC").isoformat(),
        }
    if len(set(boundaries)) != 3:
        raise RuntimeError("cutoff-specific boundaries are not distinct")
    raw_values = input_only.loc[:, ["temp", "psal", "depth"]].to_numpy(np.float64)
    support_means = np.nanmean(raw_values, axis=0).astype(np.float32)
    support_scales = np.nanstd(raw_values, axis=0).astype(np.float32)
    support_scales = np.where(support_scales > 1e-6, support_scales, 1.0)
    support_features = missingness_decay_features(
        input_only,
        support_means,
        support_scales,
        architecture["decay_half_life_hours"],
    )
    missing_counts = {
        name: int(input_only[name].isna().sum())
        for name in ("temp", "psal", "depth")
    }
    elapsed_variance = {
        name: float(np.var(support_features[:, 6 + index]))
        for index, name in enumerate(("temp", "psal", "depth"))
    }
    elapsed_nonzero_rows = {
        name: int((support_features[:, 6 + index] > 0).sum())
        for index, name in enumerate(("temp", "psal", "depth"))
    }
    support_gate = config["representation_support_gate"]
    supported = [
        name
        for name in support_gate["required_supported_channels"]
        if missing_counts[name]
        >= support_gate["minimum_missing_rows_per_supported_channel"]
        and elapsed_variance[name] >= support_gate["minimum_elapsed_variance"]
    ]
    if len(supported) < support_gate["minimum_supported_channels"]:
        raise RuntimeError(support_gate["failure"])
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS_ZERO_OPERATION",
        "surface": config["surface"],
        "source": {"readme": str(readme), "train": str(train)},
        "config_sha256": _sha(CONFIG_PATH),
        "runner_sha256": _sha(Path(__file__)),
        "parts": parts,
        "semantic_audit": audit,
        "resource": {
            "torch": torch.__version__,
            "cuda_devices": torch.cuda.device_count(),
            "cuda_name": properties.name,
            "cuda_memory_bytes": properties.total_memory,
        },
        "representation_support": {
            "rows": len(input_only),
            "station_layers": int(
                input_only.loc[:, ["station", "layer"]].drop_duplicates().shape[0]
            ),
            "missing_counts": missing_counts,
            "elapsed_nonzero_rows": elapsed_nonzero_rows,
            "elapsed_variance": elapsed_variance,
            "supported_channels": supported,
            "feature_finite": bool(np.isfinite(support_features).all()),
            "gate": "PASS",
        },
        "counters": {
            "fits": 0,
            "targets": 0,
            "official": 0,
            "csv": 0,
            "uploads": 0,
        },
    }


def _normalizer(
    frame: pd.DataFrame,
    times_ns: np.ndarray,
    train_boundary_ns: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = frame.loc[
        times_ns <= train_boundary_ns,
        ["temp", "psal", "depth"],
    ].to_numpy(np.float64)
    means = np.nanmean(values, axis=0)
    scales = np.nanstd(values, axis=0)
    if not np.isfinite(means).all():
        raise RuntimeError("fold normalizer mean is nonfinite")
    scales = np.where(np.isfinite(scales) & (scales > 1e-6), scales, 1.0)
    return means.astype(np.float32), scales.astype(np.float32)


def missingness_decay_features(
    frame: pd.DataFrame,
    means: np.ndarray,
    scales: np.ndarray,
    half_life_hours: float,
) -> np.ndarray:
    """Build strictly causal decay-imputed, mask, elapsed, and gap channels."""

    if half_life_hours <= 0:
        raise ValueError("half_life_hours must be positive")
    output = np.zeros((len(frame), 10), dtype=np.float32)
    for _key, group in frame.groupby(
        ["station", "layer"],
        sort=True,
        observed=True,
    ):
        ordered = group.sort_values("_time", kind="stable")
        positions = ordered.index.to_numpy(np.int64)
        times = _time_ns(ordered["_time"])
        if len(times) > 1 and np.any(np.diff(times) <= 0):
            raise RuntimeError("station-layer time must be strictly increasing")
        raw = ordered.loc[:, ["temp", "psal", "depth"]].to_numpy(np.float64)
        observed = np.isfinite(raw)
        normalized = (raw - means) / scales
        last = np.zeros(3, dtype=np.float64)
        elapsed = np.zeros(3, dtype=np.float64)
        features = np.zeros((len(ordered), 10), dtype=np.float64)
        previous = times[0] if len(times) else 0
        for row in range(len(ordered)):
            gap_hours = max(0.0, float(times[row] - previous) / HOUR_NS)
            elapsed += gap_hours
            current = np.zeros(3, dtype=np.float64)
            for channel in range(3):
                if observed[row, channel]:
                    current[channel] = normalized[row, channel]
                    last[channel] = current[channel]
                    elapsed[channel] = 0.0
                else:
                    gamma = math.exp(
                        -math.log(2.0) * elapsed[channel] / half_life_hours
                    )
                    current[channel] = gamma * last[channel]
            features[row, :3] = current
            features[row, 3:6] = observed[row].astype(float)
            features[row, 6:9] = np.log1p(elapsed) / math.log(25.0)
            features[row, 9] = math.log1p(gap_hours) / math.log(25.0)
            previous = times[row]
        if not np.isfinite(features).all():
            raise RuntimeError("missingness features are nonfinite")
        output[positions] = features.astype(np.float32)
    return output


@dataclass(frozen=True)
class SensorSequence:
    positions: np.ndarray
    times_ns: np.ndarray
    features: np.ndarray
    labels: np.ndarray


def _sequences(
    frame: pd.DataFrame,
    features: np.ndarray,
    labels: np.ndarray,
) -> list[SensorSequence]:
    result = []
    for _key, group in frame.groupby(
        ["station", "layer"],
        sort=True,
        observed=True,
    ):
        ordered = group.sort_values("_time", kind="stable")
        positions = ordered.index.to_numpy(np.int64)
        result.append(
            SensorSequence(
                positions=positions,
                times_ns=_time_ns(ordered["_time"]),
                features=features[positions],
                labels=labels[positions],
            )
        )
    return result


class CausalMissingnessGRU(nn.Module):
    """One-layer recurrent latent state with a rowwise anomaly head."""

    def __init__(self, input_width: int, hidden_width: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_width, hidden_width, batch_first=True)
        self.head = nn.Linear(hidden_width, 1)

    def forward(
        self,
        values: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, next_state = self.gru(values.unsqueeze(0), state)
        return self.head(hidden).squeeze(0).squeeze(-1), next_state


def _state_sha(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _fit_seed(
    sequences: list[SensorSequence],
    row_count: int,
    train_boundary_ns: int,
    architecture: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, str]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    model = CausalMissingnessGRU(10, architecture["hidden_width"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=architecture["learning_rate"],
        weight_decay=architecture["weight_decay"],
    )
    positive_weight = torch.tensor(
        architecture["positive_class_weight"],
        device=device,
    )
    chunk_rows = architecture["chunk_rows"]
    for _epoch in range(architecture["epochs"]):
        model.train()
        for sequence in sequences:
            stop = int(np.searchsorted(sequence.times_ns, train_boundary_ns, side="right"))
            state = None
            for start in range(0, stop, chunk_rows):
                end = min(stop, start + chunk_rows)
                values = torch.as_tensor(
                    sequence.features[start:end],
                    dtype=torch.float32,
                    device=device,
                )
                target = torch.as_tensor(
                    sequence.labels[start:end],
                    dtype=torch.float32,
                    device=device,
                )
                optimizer.zero_grad(set_to_none=True)
                logits, state = model(values, state)
                loss = F.binary_cross_entropy_with_logits(
                    logits,
                    target,
                    pos_weight=positive_weight,
                )
                loss.backward()
                optimizer.step()
                state = state.detach()
    predictions = np.zeros(row_count, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for sequence in sequences:
            state = None
            for start in range(0, len(sequence.positions), chunk_rows):
                end = min(len(sequence.positions), start + chunk_rows)
                values = torch.as_tensor(
                    sequence.features[start:end],
                    dtype=torch.float32,
                    device=device,
                )
                logits, state = model(values, state)
                probability = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
                predictions[sequence.positions[start:end]] = probability
    return predictions, _state_sha(model)


def _wilson_lower(successes: int, count: int, z: float) -> float:
    if count == 0:
        return 0.0
    rate = successes / count
    denominator = 1 + z * z / count
    center = rate + z * z / (2 * count)
    margin = z * math.sqrt(
        rate * (1 - rate) / count + z * z / (4 * count * count)
    )
    return (center - margin) / denominator


def _select_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    selection: dict[str, Any],
) -> dict[str, Any]:
    candidates = []
    for quantile in selection["threshold_quantiles"]:
        threshold = float(np.quantile(scores, quantile))
        proposed = scores >= threshold
        count = int(proposed.sum())
        true = int(labels[proposed].sum())
        candidates.append(
            {
                "quantile": quantile,
                "threshold": threshold,
                "count": count,
                "share": count / len(scores),
                "true": true,
                "precision": true / count if count else 0.0,
                "precision_lcb": _wilson_lower(
                    true,
                    count,
                    selection["wilson_z"],
                ),
            }
        )
    eligible = [
        item
        for item in candidates
        if item["count"] >= selection["minimum_additions_for_precision_gate"]
        and item["share"] <= selection["maximum_addition_share"]
        and item["precision_lcb"] >= selection["precision_lcb_minimum"]
    ]
    chosen = eligible[0] if eligible else None
    return {"chosen": chosen, "candidates": candidates}


def _capped_additions(
    scores: np.ndarray,
    incumbent: np.ndarray,
    chosen: dict[str, Any] | None,
    maximum_share: float,
) -> np.ndarray:
    additions = np.zeros(len(scores), dtype=bool)
    if chosen is None:
        return additions
    eligible = np.flatnonzero(
        (incumbent == 0) & (scores >= float(chosen["threshold"]))
    )
    maximum = int(math.floor(len(scores) * maximum_share))
    if len(eligible) > maximum:
        eligible = eligible[np.lexsort((eligible, -scores[eligible]))[:maximum]]
    additions[eligible] = True
    return additions


def _action_geometry(additions: np.ndarray, metadata: pd.DataFrame) -> dict[str, Any]:
    selected = metadata.loc[additions, ["station", "layer", "time"]].copy()
    if selected.empty:
        return {
            "rows": 0,
            "runs": 0,
            "stations": 0,
            "station_layers": 0,
            "run_length_min": 0,
            "run_length_median": 0.0,
            "run_length_max": 0,
            "singleton_share": 0.0,
        }
    selected["_ns"] = _time_ns(
        pd.to_datetime(selected["time"], utc=True, errors="raise", format="mixed")
    )
    lengths: list[int] = []
    for _key, group in selected.groupby(
        ["station", "layer"],
        sort=True,
        observed=True,
    ):
        times = np.sort(group["_ns"].to_numpy(np.int64))
        starts = np.r_[True, np.diff(times) != 600_000_000_000]
        run_ids = np.cumsum(starts)
        lengths.extend(np.bincount(run_ids)[1:].tolist())
    array = np.asarray(lengths, dtype=np.int64)
    return {
        "rows": int(additions.sum()),
        "runs": len(lengths),
        "stations": int(selected["station"].nunique()),
        "station_layers": int(
            selected.loc[:, ["station", "layer"]].drop_duplicates().shape[0]
        ),
        "run_length_min": int(array.min()),
        "run_length_median": float(np.median(array)),
        "run_length_max": int(array.max()),
        "singleton_share": float(np.mean(array == 1)),
    }


def execute(data_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    ready = preflight(data_dir)
    config = _read(CONFIG_PATH)
    _write(
        LOCK_PATH,
        {
            "experiment_id": EXPERIMENT_ID,
            "status": "CONSUMED_EXACTLY_ONCE",
            "config_sha256": ready["config_sha256"],
            "runner_sha256": ready["runner_sha256"],
        },
    )
    ARTIFACT_DIR.mkdir(exist_ok=False)
    _write(ARTIFACT_DIR / "preflight.json", ready)
    frame = pd.read_csv(
        ready["source"]["train"],
        usecols=[*INPUT_COLUMNS, "label", "anomaly_type"],
    )
    frame["_time"] = pd.to_datetime(
        frame["time"],
        utc=True,
        errors="raise",
        format="mixed",
    )
    times_ns = _time_ns(frame["_time"])
    labels = frame["label"].to_numpy(np.int8)
    device = torch.device("cuda:0")
    seals = []
    fit_count = 0
    for fold, part_config in config["parts"].items():
        cutoff_ns = pd.Timestamp(ready["parts"][fold]["cutoff"]).value
        boundary_ns = pd.Timestamp(ready["parts"][fold]["boundary"]).value
        means, scales = _normalizer(frame, times_ns, boundary_ns)
        features = missingness_decay_features(
            frame,
            means,
            scales,
            config["architecture"]["decay_half_life_hours"],
        )
        sequences = _sequences(frame, features, labels)
        probabilities = []
        model_hashes = []
        for seed in config["architecture"]["seeds"]:
            prediction, model_hash = _fit_seed(
                sequences,
                len(frame),
                boundary_ns,
                config["architecture"],
                seed,
                device,
            )
            probabilities.append(prediction)
            model_hashes.append(model_hash)
            fit_count += 1
            print(
                f"{EXPERIMENT_ID} progress fits={fit_count}/9 fold={fold} seed={seed}",
                file=sys.stderr,
                flush=True,
            )
        scores = np.mean(np.stack(probabilities), axis=0)
        inner = (times_ns > boundary_ns) & (times_ns <= cutoff_ns)
        selection = _select_threshold(
            scores[inner],
            labels[inner],
            config["selection"],
        )
        part = pd.read_parquet(
            ROOT / part_config["path"],
            columns=list(PART_COLUMNS),
        )
        positions = part["row_position"].to_numpy(np.int64)
        incumbent = part["baseline_prediction"].to_numpy(np.int8)
        outer_scores = scores[positions]
        additions = _capped_additions(
            outer_scores,
            incumbent,
            selection["chosen"],
            config["selection"]["maximum_addition_share"],
        )
        candidate = np.bitwise_or(incumbent, additions.astype(np.int8))
        path = ARTIFACT_DIR / f"{fold}_sealed.npz"
        np.savez_compressed(
            path,
            positions=positions,
            incumbent=incumbent,
            scores=outer_scores,
            additions=additions,
            candidate=candidate,
        )
        seal = {
            "fold": fold,
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha(path),
            "model_hashes": model_hashes,
            "selection": selection,
            "train_boundary_utc": ready["parts"][fold]["boundary"],
            "outer_target_reads_before_seal": 0,
            "fits": 3,
        }
        _write(ARTIFACT_DIR / f"{fold}_seal.json", seal)
        seals.append(seal)
    completion = {
        "experiment_id": EXPERIMENT_ID,
        "fits": fit_count,
        "seals": seals,
        "outer_target_reads_before_all_seals": 0,
    }
    _write(ARTIFACT_DIR / "predictions_complete.json", completion)
    fold_scores = []
    pool: dict[str, list[Any]] = {
        key: []
        for key in (
            "truth",
            "incumbent",
            "candidate",
            "additions",
            "types",
            "metadata",
        )
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
        seed=config["decision"]["bootstrap_seed"],
    )
    passed = (
        pooled["delta_f1"] > 0
        and bootstrap["ci90"][0] >= 0
        and all(item["delta_f1"] >= 0 for item in fold_scores)
    )
    result = {
        "schema_version": "p1.v8_missingness_decay_gru.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "surface": config["surface"],
        "decision": config["decision"]["pass"] if passed else config["decision"]["fail"],
        "semantic_audit": config["semantic_audit"],
        "resource": ready["resource"],
        "pooled": pooled,
        "fold_scores": fold_scores,
        "block_bootstrap": bootstrap,
        "worst_slices": sorted(
            pooled["station_layer_diagnostics"],
            key=lambda item: item["delta_f1"],
        )[:10],
        "action_geometry": _action_geometry(additions, metadata),
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
            "completion": _sha(ARTIFACT_DIR / "predictions_complete.json"),
            "lock": _sha(LOCK_PATH),
        },
    }
    _write(ARTIFACT_DIR / "result.json", result)
    return result


def qa(data_dir: Path) -> dict[str, Any]:
    ready = preflight(data_dir)
    config = _read(CONFIG_PATH)
    checks = {
        "zero_operation": all(value == 0 for value in ready["counters"].values()),
        "semantic_novel": (
            config["semantic_audit"]["decision"]
            == "NOVEL_REPRESENTATION_PROCEED_ONCE"
        ),
        "ns_boundaries_distinct": len(
            {item["boundary"] for item in ready["parts"].values()}
        )
        == 3,
        "past_only": config["architecture"]["past_only"],
        "future_interpolation_zero": (
            config["architecture"]["future_interpolation"] == 0
        ),
        "max9": config["architecture"]["maximum_fits"] == 9,
        "sweep_zero": config["architecture"]["sweeps"] == 0,
        "outer_tuning_zero": config["selection"]["outer_tuning"] == 0,
        "anchor_removals_zero": config["anchor"]["removals"] == 0,
        "source_access_zero": (
            config["source"]["official_test_sample_submission_hidden_reads"] == 0
        ),
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
    print(
        json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False),
        end="",
    )


if __name__ == "__main__":
    main()
