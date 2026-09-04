"""Exactly-once within-station vertical causal graph falsification."""

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
EXPERIMENT_ID = "p1_v5_within_station_vertical_causal_graph_20260901_v1"
CONFIG_PATH = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT_DIR = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK_PATH = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
SCORER_PATH = ROOT / "scripts/run_p1_clean_state_capa_falsification_20260831_v1.py"
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


scorer = _module(SCORER_PATH, "p1_v5_score_helpers")


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
        payload = json.dumps(
            value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2
        ).encode() + b"\n"
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _source_paths(data_dir: Path) -> tuple[Path, Path]:
    source = data_dir.resolve(strict=True)
    return source / "README.md", source / "train.csv"


def _vertical_contract(train_csv: Path) -> dict[str, Any]:
    frame = pd.read_csv(train_csv, usecols=["station", "layer", "depth"])
    stations: dict[str, Any] = {}
    for station, group in frame.groupby("station", sort=True, observed=True):
        medians = group.groupby("layer", sort=True, observed=True)["depth"].median()
        layers = [int(value) for value in medians.index]
        depths = [float(value) for value in medians.to_numpy()]
        contiguous = layers == list(range(layers[0], layers[-1] + 1))
        increasing = len(depths) == 1 or bool(np.all(np.diff(depths) > 0))
        stations[str(station)] = {
            "layers": layers,
            "median_depths": depths,
            "contiguous": contiguous,
            "strictly_increasing_depth": increasing,
            "vertical_edge_count": max(0, 2 * (len(layers) - 1)),
            "horizontal_edge_count": 0,
        }
    return {
        "status": "PASS"
        if all(item["contiguous"] and item["strictly_increasing_depth"] for item in stations.values())
        else "FAIL",
        "stations": stations,
        "horizontal_edges": 0,
        "authority": "train-only observed layer order and median depth",
    }


def preflight(data_dir: Path) -> dict[str, Any]:
    if ARTIFACT_DIR.exists() or LOCK_PATH.exists():
        raise FileExistsError("namespace consumed")
    config = _read(CONFIG_PATH)
    readme, train = _source_paths(data_dir)
    if _sha(readme) != config["source"]["readme_sha256"]:
        raise RuntimeError("README binding invalid")
    if _sha(train) != config["source"]["train_sha256"]:
        raise RuntimeError("train binding invalid")
    for relative, expected in config["semantic_audit"]["evidence"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"semantic evidence drifted: {relative}")
    if config["semantic_audit"]["decision"] != "NOVEL_PROCEED_ONCE":
        raise RuntimeError("semantic novelty gate closed")
    contract = _vertical_contract(train)
    if contract["status"] != "PASS" or contract["horizontal_edges"] != 0:
        raise RuntimeError("vertical-only data contract failed")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA resource contract failed")
    properties = torch.cuda.get_device_properties(0)
    parts = {}
    for fold, item in config["parts"].items():
        part_path, audit_path = ROOT / item["path"], ROOT / item["audit"]
        audit = _read(audit_path)
        if _sha(part_path) != item["sha256"]:
            raise RuntimeError("champion part binding invalid")
        if audit.get("target_fold_validation_labels_read_before_prediction") != 0:
            raise RuntimeError("champion part target firewall invalid")
        parts[fold] = {"cutoff": audit["adjusted_cutoff_utc"], "sha256": item["sha256"]}
    return {
        "schema_version": "p1.v5.vertical_graph.preflight.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS_ZERO_OPERATION",
        "config_sha256": _sha(CONFIG_PATH),
        "runner_sha256": _sha(Path(__file__)),
        "readme_sha256": _sha(readme),
        "train_sha256": _sha(train),
        "train_path": str(train),
        "semantic_audit": config["semantic_audit"],
        "vertical_contract": contract,
        "resource": {
            "torch": torch.__version__,
            "cuda_devices": torch.cuda.device_count(),
            "gpu": properties.name,
            "memory_mib": properties.total_memory // (1024 * 1024),
        },
        "parts": parts,
        "counters": {
            "claims": 0,
            "fits": 0,
            "targets": 0,
            "predictions": 0,
            "official": 0,
            "csv": 0,
            "uploads": 0,
        },
    }


@dataclass
class Panel:
    station: str
    times_ns: np.ndarray
    layers: np.ndarray
    values: np.ndarray
    observed: np.ndarray
    labels: np.ndarray
    positions: np.ndarray
    adjacency: np.ndarray


class CausalVerticalGraph(nn.Module):
    """One causal temporal convolution and one fixed vertical message pass."""

    def __init__(self, input_width: int, hidden_width: int, kernel: int) -> None:
        super().__init__()
        self.kernel = kernel
        self.temporal = nn.Conv1d(input_width, hidden_width, kernel_size=kernel)
        self.self_map = nn.Linear(hidden_width, hidden_width)
        self.neighbor_map = nn.Linear(hidden_width, hidden_width, bias=False)
        self.head = nn.Linear(hidden_width, 1)

    def forward(self, values: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # values: time x layer x channel. Only the left side is padded.
        temporal_input = values.permute(1, 2, 0)
        hidden = self.temporal(F.pad(temporal_input, (self.kernel - 1, 0)))
        hidden = torch.relu(hidden.permute(2, 0, 1))
        neighbors = torch.einsum("lm,tmh->tlh", adjacency, hidden)
        graph = torch.relu(self.self_map(hidden) + self.neighbor_map(neighbors))
        return self.head(graph).squeeze(-1)


def _build_panels(frame: pd.DataFrame, labels: np.ndarray) -> list[Panel]:
    panels = []
    for station, group in frame.groupby("station", sort=True, observed=True):
        if group.duplicated(["_time", "layer"]).any():
            raise RuntimeError("duplicate station-layer timestamp")
        times_ns = np.sort(group["_time"].astype("int64").unique())
        layers = np.sort(group["layer"].astype(int).unique())
        time_index = np.searchsorted(times_ns, group["_time"].astype("int64").to_numpy())
        layer_lookup = {int(layer): index for index, layer in enumerate(layers)}
        layer_index = np.asarray([layer_lookup[int(value)] for value in group["layer"]], dtype=np.int64)
        shape = (len(times_ns), len(layers))
        values = np.full((*shape, 3), np.nan, dtype=np.float32)
        observed = np.zeros(shape, dtype=bool)
        positions = np.full(shape, -1, dtype=np.int64)
        group_positions = group.index.to_numpy(dtype=np.int64)
        values[time_index, layer_index, :] = group.loc[:, ["temp", "psal", "depth"]].to_numpy(np.float32)
        observed[time_index, layer_index] = True
        positions[time_index, layer_index] = group_positions
        panel_labels = np.full(shape, -1, dtype=np.int8)
        panel_labels[time_index, layer_index] = labels[group_positions]
        adjacency = np.zeros((len(layers), len(layers)), dtype=np.float32)
        for index in range(len(layers) - 1):
            adjacency[index, index + 1] = 1.0
            adjacency[index + 1, index] = 1.0
        degree = adjacency.sum(axis=1, keepdims=True)
        adjacency = np.divide(adjacency, degree, out=np.zeros_like(adjacency), where=degree > 0)
        panels.append(
            Panel(str(station), times_ns, layers, values, observed, panel_labels, positions, adjacency)
        )
    return panels


def _normalizer(frame: pd.DataFrame, train_boundary_ns: int) -> tuple[np.ndarray, np.ndarray]:
    use = frame["_time"].astype("int64").to_numpy() <= train_boundary_ns
    values = frame.loc[use, ["temp", "psal", "depth"]].to_numpy(np.float64)
    means = np.nanmean(values, axis=0)
    scales = np.nanstd(values, axis=0)
    scales = np.where(scales > 1e-6, scales, 1.0)
    return means.astype(np.float32), scales.astype(np.float32)


def _tensor_values(panel: Panel, means: np.ndarray, scales: np.ndarray, device: torch.device) -> torch.Tensor:
    numeric = (panel.values - means) / scales
    missing = np.isnan(panel.values[:, :, 1:]).astype(np.float32)
    numeric = np.nan_to_num(numeric, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.linspace(-1.0, 1.0, len(panel.layers), dtype=np.float32)
    order = np.broadcast_to(order[None, :, None], (*panel.observed.shape, 1))
    values = np.concatenate([numeric, missing, order], axis=2)
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _state_sha(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _fit_seed(
    panels: list[Panel],
    means: np.ndarray,
    scales: np.ndarray,
    train_boundary_ns: int,
    architecture: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, str]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    model = CausalVerticalGraph(6, architecture["hidden_width"], architecture["temporal_kernel"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=architecture["learning_rate"],
        weight_decay=architecture["weight_decay"],
    )
    total = sum(
        int((panel.observed & (panel.times_ns[:, None] <= train_boundary_ns)).sum())
        for panel in panels
    )
    positive_weight = torch.tensor(architecture["positive_class_weight"], device=device)
    for _epoch in range(architecture["epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for panel in panels:
            values = _tensor_values(panel, means, scales, device)
            adjacency = torch.as_tensor(panel.adjacency, device=device)
            logits = model(values, adjacency)
            mask_np = panel.observed & (panel.times_ns[:, None] <= train_boundary_ns)
            mask = torch.as_tensor(mask_np, device=device)
            target = torch.as_tensor(panel.labels, dtype=torch.float32, device=device)
            loss = F.binary_cross_entropy_with_logits(
                logits[mask], target[mask], pos_weight=positive_weight, reduction="sum"
            ) / total
            loss.backward()
        optimizer.step()
    predictions = np.zeros(max(int(panel.positions.max()) for panel in panels) + 1, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for panel in panels:
            values = _tensor_values(panel, means, scales, device)
            adjacency = torch.as_tensor(panel.adjacency, device=device)
            probability = torch.sigmoid(model(values, adjacency)).cpu().numpy()
            observed = panel.observed
            predictions[panel.positions[observed]] = probability[observed]
    return predictions, _state_sha(model)


def _wilson_lower(successes: int, count: int, z: float) -> float:
    if count == 0:
        return 0.0
    rate = successes / count
    denominator = 1 + z * z / count
    center = rate + z * z / (2 * count)
    margin = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count))
    return (center - margin) / denominator


def _select_threshold(
    scores: np.ndarray, labels: np.ndarray, selection: dict[str, Any]
) -> dict[str, Any]:
    candidates = []
    for quantile in selection["threshold_quantiles"]:
        threshold = float(np.quantile(scores, quantile))
        proposed = scores >= threshold
        count = int(proposed.sum())
        true = int(labels[proposed].sum())
        lcb = _wilson_lower(true, count, selection["wilson_z"])
        candidates.append(
            {
                "quantile": quantile,
                "threshold": threshold,
                "count": count,
                "share": count / len(scores),
                "true": true,
                "precision": true / count if count else 0.0,
                "precision_lcb": lcb,
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
    return {
        "chosen": chosen,
        "threshold": chosen["threshold"] if chosen else None,
        "candidates": candidates,
    }


def _capped_additions(
    scores: np.ndarray, incumbent: np.ndarray, threshold: float | None, share: float
) -> np.ndarray:
    additions = np.zeros(len(scores), dtype=bool)
    if threshold is None:
        return additions
    eligible = np.flatnonzero((incumbent == 0) & (scores >= threshold))
    maximum = int(math.floor(len(scores) * share))
    if maximum <= 0:
        return additions
    if len(eligible) > maximum:
        order = np.lexsort((eligible, -scores[eligible]))
        eligible = eligible[order[:maximum]]
    additions[eligible] = True
    return additions


def _counts(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    truth_bool, prediction_bool = truth.astype(bool), prediction.astype(bool)
    tp = int((truth_bool & prediction_bool).sum())
    fp = int((~truth_bool & prediction_bool).sum())
    fn = int((truth_bool & ~prediction_bool).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _worst_station_layer_month(
    truth: np.ndarray,
    incumbent: np.ndarray,
    candidate: np.ndarray,
    additions: np.ndarray,
    metadata: pd.DataFrame,
) -> list[dict[str, Any]]:
    work = metadata.loc[:, ["station", "layer", "time"]].copy()
    work["month"] = pd.to_datetime(work["time"], utc=True, format="mixed").dt.strftime("%Y-%m")
    rows = []
    for (station, layer, month), indices in work.groupby(
        ["station", "layer", "month"], sort=True, observed=True
    ).indices.items():
        positions = np.asarray(indices)
        anchor_score = _counts(truth[positions], incumbent[positions])
        candidate_score = _counts(truth[positions], candidate[positions])
        rows.append(
            {
                "station": str(station),
                "layer": int(layer),
                "month": str(month),
                "rows": len(positions),
                "additions": int(additions[positions].sum()),
                "anchor_f1": anchor_score["f1"],
                "candidate_f1": candidate_score["f1"],
                "delta_f1": candidate_score["f1"] - anchor_score["f1"],
            }
        )
    return sorted(rows, key=lambda item: (item["delta_f1"], item["station"], item["layer"]))[:10]


def execute(data_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    ready, config = preflight(data_dir), _read(CONFIG_PATH)
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
    frame = pd.read_csv(ready["train_path"], usecols=[*INPUT_COLUMNS, "label", "anomaly_type"])
    frame["_time"] = pd.to_datetime(frame["time"], utc=True, errors="raise", format="mixed")
    labels = frame["label"].to_numpy(dtype=np.int8)
    panels = _build_panels(frame, labels)
    device = torch.device("cuda:0")
    seals = []
    fit_count = 0
    for fold, part_config in config["parts"].items():
        cutoff_ns = pd.Timestamp(ready["parts"][fold]["cutoff"]).value
        prefix_times = np.sort(frame.loc[frame["_time"].astype("int64") <= cutoff_ns, "_time"].astype("int64").unique())
        boundary_index = max(0, int(len(prefix_times) * config["selection"]["inner_train_fraction"]) - 1)
        train_boundary_ns = int(prefix_times[boundary_index])
        means, scales = _normalizer(frame, train_boundary_ns)
        seed_predictions, model_hashes = [], []
        for seed in config["architecture"]["seeds"]:
            predictions, model_hash = _fit_seed(
                panels,
                means,
                scales,
                train_boundary_ns,
                config["architecture"],
                seed,
                device,
            )
            seed_predictions.append(predictions)
            model_hashes.append(model_hash)
            fit_count += 1
            print(f"{EXPERIMENT_ID} progress fits={fit_count}/9 fold={fold} seed={seed}", file=sys.stderr, flush=True)
        scores = np.mean(np.stack(seed_predictions), axis=0)
        times_ns = frame["_time"].astype("int64").to_numpy()
        inner = (times_ns > train_boundary_ns) & (times_ns <= cutoff_ns)
        threshold = _select_threshold(scores[inner], labels[inner], config["selection"])
        part = pd.read_parquet(ROOT / part_config["path"], columns=list(PART_COLUMNS))
        positions = part["row_position"].to_numpy(dtype=np.int64)
        incumbent = part["baseline_prediction"].to_numpy(dtype=np.int8)
        outer_scores = scores[positions]
        additions = _capped_additions(
            outer_scores,
            incumbent,
            threshold["threshold"],
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
            "threshold_selection": threshold,
            "train_boundary_utc": pd.Timestamp(train_boundary_ns, tz="UTC").isoformat(),
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
    fold_scores, pool = [], {key: [] for key in ("truth", "incumbent", "candidate", "additions", "types", "metadata")}
    for seal in seals:
        with np.load(ROOT / seal["path"], allow_pickle=False) as values:
            positions = values["positions"]
            incumbent = values["incumbent"]
            additions = values["additions"]
            candidate = values["candidate"]
        truth = labels[positions]
        metadata = frame.iloc[positions].loc[:, list(KEY_COLUMNS)].reset_index(drop=True)
        types = frame.iloc[positions]["anomaly_type"].reset_index(drop=True)
        score = scorer._score_surface(truth, incumbent, candidate, additions, types, metadata)
        fold_scores.append({"fold": seal["fold"], **score})
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
        np.concatenate(pool[key]) for key in ("truth", "incumbent", "candidate", "additions")
    )
    types = pd.concat(pool["types"], ignore_index=True)
    metadata = pd.concat(pool["metadata"], ignore_index=True)
    pooled = scorer._score_surface(truth, incumbent, candidate, additions, types, metadata)
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
        "schema_version": "p1.v5.vertical_graph.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "decision": config["decision"]["pass"] if passed else config["decision"]["fail"],
        "semantic_audit": config["semantic_audit"],
        "vertical_contract": ready["vertical_contract"],
        "resource": ready["resource"],
        "pooled": pooled,
        "fold_scores": fold_scores,
        "block_bootstrap": bootstrap,
        "worst_station_layer_month": _worst_station_layer_month(
            truth, incumbent, candidate, additions, metadata
        ),
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
            "completion": _sha(ARTIFACT_DIR / "predictions_complete.json"),
            "lock": _sha(LOCK_PATH),
        },
    }
    _write(ARTIFACT_DIR / "result.json", result)
    return result


def qa(data_dir: Path) -> dict[str, Any]:
    ready, config = preflight(data_dir), _read(CONFIG_PATH)
    checks = {
        "zero_operation": all(value == 0 for value in ready["counters"].values()),
        "semantic_novelty": config["semantic_audit"]["decision"] == "NOVEL_PROCEED_ONCE",
        "vertical_only": ready["vertical_contract"]["horizontal_edges"] == 0,
        "past_only": config["architecture"]["temporal_encoder"].startswith("one past-only"),
        "one_message_pass": config["architecture"]["graph_encoder"].startswith("one fixed"),
        "max9": config["architecture"]["maximum_fits"] == 9,
        "frozen_capacity": config["architecture"]["sweeps"] == 0,
        "add_only": config["anchor"] == {"operation": "bitwise_or", "removals": 0},
        "outer_tuning_zero": config["selection"]["outer_tuning"] == 0,
        "official_zero": config["source"]["official_test_sample_submission_hidden_reads"] == 0,
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
    print(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False), end="")


if __name__ == "__main__":
    main()
