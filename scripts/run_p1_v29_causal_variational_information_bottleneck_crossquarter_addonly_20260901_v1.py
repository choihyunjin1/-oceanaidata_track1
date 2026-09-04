"""Exactly-once cross-quarter causal variational-information-bottleneck P1 audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_v29_causal_variational_information_bottleneck_crossquarter_addonly_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
V27 = ROOT / "scripts/run_p1_v27_causal_dirichlet_evidential_addonly_20260901_v1.py"
KEY_COLUMNS = ("station", "year", "layer", "time")
PART_COLUMNS = (*KEY_COLUMNS, "row_position", "baseline_prediction")


def _module(path: Path):
    spec = importlib.util.spec_from_file_location("p1_v29_shared", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("shared module load failed")
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


shared = _module(V27)
core, base, scorer = shared.core, shared.base, shared.core.scorer


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2).encode() + b"\n"
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _time_ns(values: pd.Series | pd.Index) -> np.ndarray:
    return core._time_ns(values)


class _VIBNetwork(nn.Module):
    def __init__(self, inputs: int, hidden: int, latent: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh())
        self.mean = nn.Linear(hidden, latent)
        self.log_variance = nn.Linear(hidden, latent)
        self.head = nn.Linear(latent, 1)

    def encode(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(values)
        return self.mean(encoded), torch.clamp(self.log_variance(encoded), -8.0, 8.0)

    def forward(self, values: torch.Tensor, *, sample: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_variance = self.encode(values)
        latent = mean
        if sample:
            latent = mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
        return self.head(latent).squeeze(1), mean, log_variance


def vib_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    *,
    positive_weight: float,
    kl_coefficient: float,
) -> torch.Tensor:
    prediction = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=torch.tensor(positive_weight, dtype=logits.dtype, device=logits.device),
        reduction="none",
    )
    divergence = 0.5 * torch.sum(mean.square() + torch.exp(log_variance) - log_variance - 1.0, dim=1)
    return prediction + kl_coefficient * divergence


class VIBClassifier:
    def __init__(self, inputs: int, config: dict[str, Any], seed: int) -> None:
        self.config = config
        self.seed = int(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.network = _VIBNetwork(inputs, int(config["hidden_units"]), int(config["latent_units"])).to(self.device)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> VIBClassifier:
        values = np.asarray(features, dtype=np.float32)
        targets = np.asarray(labels, dtype=np.int8)
        rng = np.random.default_rng(self.seed)
        maximum = int(self.config["maximum_sample_rows_per_fit"])
        ratio = int(self.config["negative_to_positive_sample_ratio"])
        positive = np.flatnonzero(targets == 1)
        negative = np.flatnonzero(targets == 0)
        positive = rng.choice(positive, size=min(len(positive), maximum // (ratio + 1)), replace=False)
        negative = rng.choice(negative, size=min(len(negative), ratio * len(positive)), replace=False)
        selected = np.concatenate([positive, negative])
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=float(self.config["learning_rate"]),
            weight_decay=float(self.config["weight_decay"]),
        )
        batch = int(self.config["batch_rows"])
        self.network.train()
        for _epoch in range(int(self.config["epochs"])):
            order = selected[rng.permutation(len(selected))]
            for start in range(0, len(order), batch):
                rows = order[start : start + batch]
                x = torch.from_numpy(values[rows]).to(self.device)
                y = torch.from_numpy(targets[rows].astype(np.float32)).to(self.device)
                logits, mean, log_variance = self.network(x, sample=True)
                loss = vib_loss(
                    logits,
                    y,
                    mean,
                    log_variance,
                    positive_weight=float(self.config["positive_class_weight"]),
                    kl_coefficient=float(self.config["kl_coefficient"]),
                ).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        return self

    def predict_score(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        output = np.empty(len(values), dtype=np.float32)
        self.network.eval()
        with torch.no_grad():
            for start in range(0, len(values), 32768):
                x = torch.from_numpy(values[start : start + 32768]).to(self.device)
                logits, _mean, _log_variance = self.network(x, sample=False)
                output[start : start + len(x)] = torch.sigmoid(logits).cpu().numpy()
        return output


def _model_hash(model: VIBClassifier, scaler: StandardScaler) -> str:
    digest = hashlib.sha256()
    for parameter in model.network.parameters():
        digest.update(parameter.detach().cpu().numpy().tobytes())
    for value in (scaler.mean_, scaler.scale_):
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def _environment_gate(
    use: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    contract: dict[str, Any],
) -> dict[str, Any]:
    times = _time_ns(pd.to_datetime(metadata["time"], utc=True, errors="raise", format="mixed"))
    ordered = np.sort(np.unique(times))
    half_cutoff = ordered[max(0, len(ordered) // 2 - 1)]
    work = pd.DataFrame(
        {
            "station": metadata["station"].astype(str).to_numpy(),
            "layer": metadata["layer"].astype(str).to_numpy(),
            "half": (times > half_cutoff).astype(np.int8),
            "use": np.asarray(use, dtype=bool),
            "truth": np.asarray(labels, dtype=np.int8),
        }
    )
    environments = []
    for (station, layer, half), group in work.loc[work["use"]].groupby(["station", "layer", "half"], sort=True):
        count = int(len(group))
        if count < int(contract["minimum_proposals_per_supported_environment"]):
            continue
        true = int(group["truth"].sum())
        environments.append(
            {"station": station, "layer": layer, "half": int(half), "count": count, "true": true, "precision": true / count}
        )
    identities = {(item["station"], item["layer"]) for item in environments}
    stations = {item["station"] for item in environments}
    halves = {item["half"] for item in environments}
    passed = bool(
        len(environments) >= int(contract["minimum_supported_environments"])
        and (not contract["require_both_chronological_halves"] or halves == {0, 1})
        and len(identities) >= int(contract["minimum_distinct_station_layer_identities"])
        and len(stations) >= int(contract["minimum_distinct_stations"])
        and all(item["precision"] > float(contract["minimum_environment_precision_strictly_gt"]) for item in environments)
        and all(item["true"] > 0 for item in environments)
    )
    return {
        "passed": passed,
        "supported_environments": environments,
        "distinct_station_layer_identities": len(identities),
        "distinct_stations": len(stations),
        "chronological_halves_present": sorted(halves),
    }


def _candidate_receipt(
    scores: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    use: np.ndarray,
    selection: dict[str, Any],
    *,
    quantile: float,
    threshold: float,
) -> dict[str, Any]:
    count = int(use.sum())
    true = int(labels[use].sum())
    precision = true / count if count else 0.0
    environment = _environment_gate(use, labels, metadata, selection["environment"])
    passed = bool(
        count >= int(selection["minimum_additions"])
        and count / len(scores) <= float(selection["maximum_addition_share"])
        and precision > float(selection["precision_strictly_gt"])
        and true > 0
        and base._wilson_lower(true, count) >= float(selection["wilson90_lcb_minimum"])
        and environment["passed"]
    )
    return {
        "quantile": quantile,
        "threshold": threshold,
        "count": count,
        "share": count / len(scores),
        "true": true,
        "precision": precision,
        "precision_lcb": base._wilson_lower(true, count),
        "environment": environment,
        "passed": passed,
    }


def _select_pre_q2_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    selection: dict[str, Any],
) -> dict[str, Any]:
    candidates = []
    for quantile in selection["threshold_quantiles"]:
        threshold = float(np.quantile(scores, quantile))
        use = scores >= threshold
        candidates.append(
            _candidate_receipt(scores, labels, metadata, use, selection, quantile=float(quantile), threshold=threshold)
        )
    eligible = [item for item in candidates if item["passed"]]
    eligible.sort(key=lambda item: (item["quantile"], item["precision_lcb"]), reverse=True)
    return {"candidates": candidates, "chosen": eligible[0] if eligible else None}


def _fixed_threshold_additions(
    scores: np.ndarray,
    incumbent: np.ndarray,
    threshold: float,
    maximum_share: float,
) -> np.ndarray:
    eligible = np.flatnonzero((incumbent == 0) & (scores >= threshold))
    maximum = int(math.floor(len(scores) * maximum_share))
    if len(eligible) > maximum:
        eligible = eligible[np.lexsort((eligible, -scores[eligible]))[:maximum]]
    result = np.zeros(len(scores), dtype=bool)
    result[eligible] = True
    return result


def _window_gate(
    scores: np.ndarray,
    incumbent: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    threshold: float,
    selection: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    additions = _fixed_threshold_additions(scores, incumbent, threshold, float(selection["maximum_addition_share"]))
    receipt = _candidate_receipt(
        scores,
        labels,
        metadata,
        additions,
        selection,
        quantile=-1.0,
        threshold=threshold,
    )
    return additions, receipt


def preflight(data_dir: Path) -> dict[str, Any]:
    if ARTIFACT.exists() or LOCK.exists():
        raise FileExistsError("namespace consumed")
    config = _read(CONFIG)
    resolved = data_dir.resolve(strict=True)
    readme = (resolved / "README.md").resolve(strict=True)
    train = (resolved / "train.csv").resolve(strict=True)
    if readme.parent != resolved or train.parent != resolved:
        raise RuntimeError("source path escaped P1_DATA_DIR")
    if config["source"]["allowed_files"] != ["README.md", "train.csv"]:
        raise RuntimeError("source allowlist drifted")
    if _sha(readme) != config["source"]["readme_sha256"] or _sha(train) != config["source"]["train_sha256"]:
        raise RuntimeError("source binding invalid")
    audit = config["semantic_audit"]
    if audit["decision"] != "NOVEL_P1_OBJECTIVE_PROCEED_ONCE" or audit["exact_duplicate"] or audit["semantic_duplicate"]:
        raise RuntimeError("semantic gate closed")
    for relative, expected in audit["evidence"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"semantic evidence drifted: {relative}")
    guard = config["cross_quarter_guard"]
    if _sha(ROOT / guard["path"]) != guard["sha256"]:
        raise RuntimeError("cross-quarter amendment drifted")
    frame = pd.read_csv(train, usecols=["station", "layer", "time", "temp"])
    frame["_time"] = pd.to_datetime(frame["time"], utc=True, errors="raise", format="mixed")
    times_ns = _time_ns(frame["_time"])
    q2_audit = _read(ROOT / config["parts"]["2025_q2"]["audit"])
    cutoff = pd.Timestamp(q2_audit["adjusted_cutoff_utc"]).value
    prefix = np.sort(np.unique(times_ns[times_ns <= cutoff]))
    boundary = int(prefix[max(0, int(len(prefix) * config["selection"]["inner_train_fraction"]) - 1)])
    if not boundary < cutoff:
        raise RuntimeError("pre-Q2 calibration boundary invalid")
    part_receipts = {}
    position_sets = {}
    for fold, item in config["parts"].items():
        if _sha(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"champion part drifted: {fold}")
        part_audit = _read(ROOT / item["audit"])
        if part_audit["target_fold_validation_labels_read_before_prediction"] != 0:
            raise RuntimeError("champion target isolation failed")
        positions = pd.read_parquet(ROOT / item["path"], columns=["row_position"])["row_position"].to_numpy(np.int64)
        position_sets[fold] = set(positions.tolist())
        selected_times = times_ns[positions]
        part_receipts[fold] = {
            "role": item["role"],
            "rows": int(len(positions)),
            "minimum_time": pd.Timestamp(int(selected_times.min()), tz="UTC").isoformat(),
            "maximum_time": pd.Timestamp(int(selected_times.max()), tz="UTC").isoformat(),
        }
    if any(position_sets[left] & position_sets[right] for left, right in (("2025_q2", "2025_q3"), ("2025_q2", "2025_q4"), ("2025_q3", "2025_q4"))):
        raise RuntimeError("cross-quarter row sets overlap")
    q2_minimum = min(times_ns[list(position_sets["2025_q2"])])
    if not cutoff < q2_minimum:
        raise RuntimeError("pre-Q2 cutoff reaches transport rows")
    features = shared.causal_evidential_features(frame, boundary, config["representation"])
    variances = np.var(features, axis=0)
    support = float(np.mean(np.any(np.abs(features) > 1e-12, axis=1)))
    gate = config["representation_support_gate"]
    if support < gate["minimum_nonzero_feature_share"] or float(variances.max()) < gate["minimum_feature_variance"]:
        raise RuntimeError(gate["failure"])
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS_ZERO_OPERATION",
        "source": {"readme": str(readme), "train": str(train)},
        "config_sha256": _sha(CONFIG),
        "runner_sha256": _sha(Path(__file__)),
        "cross_quarter_guard_sha256": _sha(ROOT / guard["path"]),
        "pre_q2": {"fit_boundary": pd.Timestamp(boundary, tz="UTC").isoformat(), "calibration_cutoff": pd.Timestamp(cutoff, tz="UTC").isoformat()},
        "parts": part_receipts,
        "semantic_audit": audit,
        "representation_support": {"nonzero_feature_share": support, "feature_variances": variances.tolist(), "gate": "PASS"},
        "counters": {"fits": 0, "targets": 0, "official": 0, "csv": 0, "uploads": 0},
    }


def _terminal_result(
    ready: dict[str, Any],
    config: dict[str, Any],
    started: float,
    fit_count: int,
    threshold_selection: dict[str, Any],
    transport_receipts: list[dict[str, Any]],
    *,
    failed_stage: str,
) -> dict[str, Any]:
    completion = {
        "experiment_id": EXPERIMENT_ID,
        "performance_window_opened": False,
        "failed_stage": failed_stage,
        "transport_windows_read": len(transport_receipts),
        "q4_target_reads": 0,
        "q4_actions": 0,
    }
    _write(ARTIFACT / "predictions_complete.json", completion)
    result = {
        "schema_version": config["result_schema_version"],
        "experiment_id": EXPERIMENT_ID,
        "surface": config["surface"],
        "decision": config["decision"]["transport_fail"],
        "failed_stage": failed_stage,
        "threshold_selection": threshold_selection,
        "transport_receipts": transport_receipts,
        "performance": None,
        "performance_window_opened": False,
        "points": {"nominal": 0.0, "transport_adjusted": 0.0},
        "counters": {
            "fits": fit_count,
            "transport_target_windows_read": len(transport_receipts),
            "q4_target_reads": 0,
            "q4_actions": 0,
            "anchor_removals": 0,
            "official": 0,
            "csv": 0,
            "uploads": 0,
        },
        "runtime_seconds": time.monotonic() - started,
        "hashes": {
            "config": ready["config_sha256"],
            "runner": ready["runner_sha256"],
            "guard": ready["cross_quarter_guard_sha256"],
            "completion": _sha(ARTIFACT / "predictions_complete.json"),
            "lock": _sha(LOCK),
        },
    }
    _write(ARTIFACT / "result.json", result)
    return result


def execute(data_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    ready = preflight(data_dir)
    config = _read(CONFIG)
    _write(LOCK, {"experiment_id": EXPERIMENT_ID, "status": "CONSUMED_EXACTLY_ONCE", "config_sha256": ready["config_sha256"], "runner_sha256": ready["runner_sha256"]})
    ARTIFACT.mkdir(exist_ok=False)
    _write(ARTIFACT / "preflight.json", ready)
    frame = pd.read_csv(
        ready["source"]["train"],
        usecols=["station", "year", "layer", "time", "temp", "label", "anomaly_type"],
    )
    frame["_time"] = pd.to_datetime(frame["time"], utc=True, errors="raise", format="mixed")
    times_ns = _time_ns(frame["_time"])
    labels = frame["label"].to_numpy(np.int8)
    boundary = pd.Timestamp(ready["pre_q2"]["fit_boundary"]).value
    cutoff = pd.Timestamp(ready["pre_q2"]["calibration_cutoff"]).value
    features = shared.causal_evidential_features(frame, boundary, config["representation"])
    train_mask = times_ns <= boundary
    calibration_mask = (times_ns > boundary) & (times_ns <= cutoff)
    scaler = StandardScaler().fit(features[train_mask])
    scaled = scaler.transform(features).astype(np.float32)
    scores = []
    model_hashes = []
    for seed in config["model"]["seeds"]:
        model = VIBClassifier(scaled.shape[1], config["model"], int(seed)).fit(scaled[train_mask], labels[train_mask])
        scores.append(model.predict_score(scaled))
        model_hashes.append(_model_hash(model, scaler))
    ensemble_scores = np.mean(np.stack(scores), axis=0)
    threshold_selection = _select_pre_q2_threshold(
        ensemble_scores[calibration_mask],
        labels[calibration_mask],
        frame.loc[calibration_mask, list(KEY_COLUMNS)].reset_index(drop=True),
        config["selection"],
    )
    threshold_seal = {
        "experiment_id": EXPERIMENT_ID,
        "fit_boundary": ready["pre_q2"]["fit_boundary"],
        "calibration_cutoff": ready["pre_q2"]["calibration_cutoff"],
        "selection": threshold_selection,
        "model_hashes": model_hashes,
        "fits": len(model_hashes),
        "q2_q3_q4_target_reads_before_seal": 0,
    }
    _write(ARTIFACT / "pre_q2_candidate_threshold_seal.json", threshold_seal)
    chosen = threshold_selection["chosen"]
    if chosen is None:
        return _terminal_result(ready, config, started, len(model_hashes), threshold_selection, [], failed_stage="PRE_Q2_CALIBRATION_GATE")
    transport_receipts = []
    for fold in ("2025_q2", "2025_q3"):
        part = pd.read_parquet(ROOT / config["parts"][fold]["path"], columns=list(PART_COLUMNS))
        positions = part["row_position"].to_numpy(np.int64)
        incumbent = part["baseline_prediction"].to_numpy(np.int8)
        window_scores = ensemble_scores[positions]
        additions = _fixed_threshold_additions(window_scores, incumbent, float(chosen["threshold"]), float(config["selection"]["maximum_addition_share"]))
        action_path = ARTIFACT / f"{fold}_transport_actions_sealed.npz"
        np.savez_compressed(action_path, positions=positions, incumbent=incumbent, additions=additions, scores=window_scores)
        _write(
            ARTIFACT / f"{fold}_transport_action_seal.json",
            {"fold": fold, "path": str(action_path.relative_to(ROOT)).replace("\\", "/"), "sha256": _sha(action_path), "threshold": chosen["threshold"], "target_reads_before_seal": 0},
        )
        window_labels = labels[positions]
        metadata = frame.iloc[positions].loc[:, list(KEY_COLUMNS)].reset_index(drop=True)
        _same_additions, receipt = _window_gate(window_scores, incumbent, window_labels, metadata, float(chosen["threshold"]), config["selection"])
        if not np.array_equal(additions, _same_additions):
            raise RuntimeError("sealed transport action drifted")
        receipt = {"fold": fold, **receipt, "same_threshold": float(chosen["threshold"]), "refits_after_pre_q2": 0}
        _write(ARTIFACT / f"{fold}_transport_gate.json", receipt)
        transport_receipts.append(receipt)
        if not receipt["passed"]:
            return _terminal_result(ready, config, started, len(model_hashes), threshold_selection, transport_receipts, failed_stage=f"{fold}_TRANSPORT_GATE")
    _write(
        ARTIFACT / "cross_quarter_gate_pass.json",
        {"experiment_id": EXPERIMENT_ID, "threshold": chosen["threshold"], "transport_windows": ["2025_q2", "2025_q3"], "both_passed": True, "q4_target_reads_before_gate_pass_receipt": 0},
    )
    q4_part = pd.read_parquet(ROOT / config["parts"]["2025_q4"]["path"], columns=list(PART_COLUMNS))
    positions = q4_part["row_position"].to_numpy(np.int64)
    incumbent = q4_part["baseline_prediction"].to_numpy(np.int8)
    q4_scores = ensemble_scores[positions]
    additions = _fixed_threshold_additions(q4_scores, incumbent, float(chosen["threshold"]), float(config["selection"]["maximum_addition_share"]))
    candidate = np.bitwise_or(incumbent, additions.astype(np.int8))
    q4_path = ARTIFACT / "2025_q4_sealed.npz"
    np.savez_compressed(q4_path, positions=positions, incumbent=incumbent, additions=additions, candidate=candidate, scores=q4_scores)
    q4_seal = {"fold": "2025_q4", "path": str(q4_path.relative_to(ROOT)).replace("\\", "/"), "sha256": _sha(q4_path), "threshold": chosen["threshold"], "model_hashes": model_hashes, "fits": len(model_hashes), "q4_target_reads_before_seal": 0}
    _write(ARTIFACT / "2025_q4_seal.json", q4_seal)
    completion = {"experiment_id": EXPERIMENT_ID, "performance_window_opened": True, "transport_windows_read": 2, "q4_target_reads_before_action_seal": 0, "q4_actions": int(additions.sum()), "seals": [q4_seal]}
    _write(ARTIFACT / "predictions_complete.json", completion)
    truth = labels[positions]
    metadata = frame.iloc[positions].loc[:, list(KEY_COLUMNS)].reset_index(drop=True)
    types = frame.iloc[positions]["anomaly_type"].reset_index(drop=True)
    performance = scorer._score_surface(truth, incumbent, candidate, additions, types, metadata)
    bootstrap = scorer._paired_cluster_bootstrap(truth, incumbent, candidate, metadata, replicates=config["decision"]["bootstrap_replicates"], seed=config["decision"]["seed"])
    passed = performance["delta_f1"] > 0 and bootstrap["ci90"][0] >= 0
    result = {
        "schema_version": config["result_schema_version"],
        "experiment_id": EXPERIMENT_ID,
        "surface": config["surface"],
        "decision": config["decision"]["pass"] if passed else config["decision"]["performance_fail"],
        "threshold_selection": threshold_selection,
        "transport_receipts": transport_receipts,
        "performance": performance,
        "performance_window": "2025_q4",
        "performance_window_opened": True,
        "block_bootstrap": bootstrap,
        "long_event_interior": base._long_event_interior(truth, incumbent, candidate, metadata),
        "long_event_boundary": shared._boundary_recall(truth, incumbent, candidate, metadata),
        "worst_slices": sorted(performance["station_layer_diagnostics"], key=lambda item: item["delta_f1"])[:10],
        "action_slices": base._action_slices(additions, metadata),
        "points": {"nominal": performance["delta_f1"] * core.POINTS_PER_F1, "transport_adjusted": performance["delta_f1"] * core.POINTS_PER_F1 * core.TRANSPORT_FACTOR},
        "counters": {"fits": len(model_hashes), "transport_target_windows_read": 2, "q4_target_reads": 1, "q4_actions": int(additions.sum()), "anchor_removals": performance["incumbent_positive_removals"], "official": 0, "csv": 0, "uploads": 0},
        "runtime_seconds": time.monotonic() - started,
        "hashes": {"config": ready["config_sha256"], "runner": ready["runner_sha256"], "guard": ready["cross_quarter_guard_sha256"], "completion": _sha(ARTIFACT / "predictions_complete.json"), "lock": _sha(LOCK)},
    }
    _write(ARTIFACT / "result.json", result)
    return result


def qa(data_dir: Path) -> dict[str, Any]:
    result_path = ARTIFACT / "result.json"
    config = _read(CONFIG)
    if not result_path.exists():
        ready = preflight(data_dir)
        checks = {
            "zero": all(value == 0 for value in ready["counters"].values()),
            "novel": config["semantic_audit"]["decision"] == "NOVEL_P1_OBJECTIVE_PROCEED_ONCE",
            "support": ready["representation_support"]["gate"] == "PASS",
            "pre_q2_fixed": config["selection"]["candidate_threshold_fixed_before_q2"],
            "q2_q3_no_selection": config["selection"]["q2_q3_threshold_selection"] == 0,
            "q2_q3_no_refits": config["selection"]["q2_q3_refits"] == 0,
            "q4_gated": config["selection"]["q4_open_only_after_q2_q3_pass"],
            "fits_within_budget": config["model"]["fits"] <= config["model"]["maximum_fits"] <= 9,
            "add_only": config["anchor"]["removals"] == 0,
            "access0": config["source"]["official_test_sample_submission_hidden_reads"] == 0,
        }
        return {"experiment_id": EXPERIMENT_ID, "phase": "PRE_EXECUTION", "verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
    result = _read(result_path)
    completion = _read(ARTIFACT / "predictions_complete.json")
    checks = {
        "terminal_result": result["experiment_id"] == EXPERIMENT_ID,
        "fits_within_budget": result["counters"]["fits"] <= 9,
        "add_only": result["counters"]["anchor_removals"] == 0,
        "access0": result["counters"]["official"] == result["counters"]["csv"] == result["counters"]["uploads"] == 0,
        "config_hash": result["hashes"]["config"] == _sha(CONFIG),
        "runner_hash": result["hashes"]["runner"] == _sha(Path(__file__)),
        "guard_hash": result["hashes"]["guard"] == config["cross_quarter_guard"]["sha256"],
        "completion_hash": result["hashes"]["completion"] == _sha(ARTIFACT / "predictions_complete.json"),
        "lock_hash": result["hashes"]["lock"] == _sha(LOCK),
        "q4_lifecycle": bool(
            (result["performance_window_opened"] and completion["performance_window_opened"] and result["counters"]["q4_target_reads"] == 1)
            or (not result["performance_window_opened"] and not completion["performance_window_opened"] and result["counters"]["q4_target_reads"] == result["counters"]["q4_actions"] == 0)
        ),
        "same_threshold": len({item["same_threshold"] for item in result["transport_receipts"]}) <= 1,
    }
    return {"experiment_id": EXPERIMENT_ID, "phase": "POST_TERMINAL_IMMUTABLE_REVALIDATION", "verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "result_sha256": _sha(result_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--qa", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    value = preflight(args.data_dir) if args.preflight else qa(args.data_dir) if args.qa else execute(args.data_dir)
    print(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False), end="")


if __name__ == "__main__":
    main()
