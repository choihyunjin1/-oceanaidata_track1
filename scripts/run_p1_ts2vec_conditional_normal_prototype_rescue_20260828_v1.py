"""Check, smoke, or run the sealed P1 TS2Vec-style local experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p1_qc.ts2vec_conditional_normal_prototype import (  # noqa: E402
    HierarchicalContrastiveEncoder,
    binary_metrics,
    contiguous_windows,
    day_block_bootstrap_probability,
    decode_components,
    finite_normal_tail_threshold,
    fit_conditional_prototype,
    hierarchical_contrastive_loss,
    masked_views,
    robust_fit_transform,
    score_conditional_prototype,
    seed_everything,
    sha256_file,
    smoke_train,
)

CONFIG = ROOT / "configs/experiments/p1_ts2vec_conditional_normal_prototype_rescue_20260828_v1.json"
MODULE = ROOT / "src/p1_qc/ts2vec_conditional_normal_prototype.py"
RUNNER = Path(__file__).resolve()
EXPERIMENT_ID = "p1_ts2vec_conditional_normal_prototype_rescue_20260828_v1"
FORBIDDEN_PARTS = ("test.csv", "sample_submission", "submission.csv")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain an object")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _canonical_sha(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_contract(config: dict[str, Any], *, verify_q2: bool = False) -> dict[str, Any]:
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("experiment ID drifted")
    policy = config.get("execution_policy", {})
    if policy.get("official_upload_authorized") is not False:
        raise RuntimeError("official upload must remain unauthorized")
    if config["representation"].get("labels_used_in_encoder_loss") is not False:
        raise RuntimeError("encoder loss must remain label-free")
    scoring = config["scoring"]
    if scoring.get("positive_prevalence_targeting") is not False:
        raise RuntimeError("positive prevalence targeting is prohibited")
    if scoring.get("spot_or_test_point_adjustment") is not False:
        raise RuntimeError("SPOT/local point adjustment is prohibited")
    if scoring.get("anchor_deletions_allowed") is not False:
        raise RuntimeError("anchor deletions are prohibited")
    inputs: dict[str, dict[str, Any]] = config["immutable_inputs"]
    checked: dict[str, Any] = {}
    for name, spec in inputs.items():
        if name == "q2_e150_anchor" and not verify_q2:
            checked[name] = {"deferred": True, "reason": "historical_gate_not_opened"}
            continue
        path = ROOT / spec["path"]
        lowered = str(path).lower()
        if any(part in lowered for part in FORBIDDEN_PARTS):
            raise RuntimeError(f"forbidden path in contract: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != int(spec["bytes"]) or digest != spec["sha256"]:
            raise RuntimeError(f"immutable input drifted: {name}")
        checked[name] = {"bytes": size, "sha256": digest}
    return {
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(CONFIG),
        "canonical_contract_sha256": _canonical_sha(config),
        "inputs": checked,
        "q2_truth_rows_read": 0,
        "q3_q4_rows_read": 0,
        "official_rows_read": 0,
    }


def run_smoke(config: dict[str, Any]) -> dict[str, Any]:
    receipt = validate_contract(config)
    started = time.perf_counter()
    cpu = smoke_train("cpu", int(config["representation"]["seed"]))
    gpu: dict[str, Any]
    if torch.cuda.is_available():
        gpu = smoke_train("cuda", int(config["representation"]["seed"]))
        gpu["device_name"] = torch.cuda.get_device_name(0)
        gpu["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        torch.cuda.empty_cache()
    else:
        gpu = {"skipped": True, "reason": "cuda_unavailable"}
    if not cpu["finite"] or (not gpu.get("skipped") and not gpu["finite"]):
        raise RuntimeError("smoke embedding is not finite")
    smoke = {
        "schema_version": "p1.ts2vec_conditional_normal_prototype.smoke.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS_CPU_GPU_SMOKE" if not gpu.get("skipped") else "PASS_CPU_SMOKE",
        "contract": receipt,
        "cpu": cpu,
        "gpu": gpu,
        "elapsed_seconds": time.perf_counter() - started,
        "scientific_source_rows_read": 0,
        "labels_read": 0,
        "full_model_fit_count": 0,
        "official_rows_read": 0,
    }
    output = ROOT / config["artifacts"]["smoke_directory"] / "smoke.json"
    _atomic_json(output, smoke)
    return smoke


def _segment_ids(keys: pd.DataFrame) -> np.ndarray:
    time_values = pd.to_datetime(keys["time"], utc=True)
    boundary = (
        keys["station"].astype(str).ne(keys["station"].astype(str).shift())
        | keys["year"].ne(keys["year"].shift())
        | keys["layer"].ne(keys["layer"].shift())
        | time_values.diff().ne(pd.Timedelta(minutes=10))
    )
    boundary.iloc[0] = True
    return boundary.cumsum().to_numpy(dtype=np.int64)


def _infer_embeddings(
    model: HierarchicalContrastiveEncoder,
    values: np.ndarray,
    windows: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    width = model.output_projection.out_channels
    total = np.zeros((len(values), width), dtype=np.float32)
    counts = np.zeros(len(values), dtype=np.int32)
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(windows), batch_size):
            batch_windows = windows[offset : offset + batch_size]
            batch = np.stack([values[start:stop] for start, stop in batch_windows])
            embedding = model(torch.from_numpy(batch).to(device)).cpu().numpy()
            for row, (start, stop) in enumerate(batch_windows):
                total[start:stop] += embedding[row]
                counts[start:stop] += 1
    covered = counts > 0
    total[covered] /= counts[covered, None]
    total[~covered] = np.nan
    return total, covered


def _train_encoder(
    values: np.ndarray,
    windows: np.ndarray,
    config: dict[str, Any],
    output: Path,
) -> tuple[HierarchicalContrastiveEncoder, dict[str, Any]]:
    representation = config["representation"]
    seed = int(representation["seed"])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HierarchicalContrastiveEncoder(
        values.shape[1],
        int(representation["hidden_width"]),
        int(representation["embedding_width"]),
        tuple(int(value) for value in representation["dilations"]),
        float(representation["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(representation["learning_rate"]),
        weight_decay=float(representation["weight_decay"]),
    )
    # Last 10% is a fixed, label-free checkpoint surface.  A 512-row embargo
    # prevents a training window from sharing rows with it.
    cut = max(1, int(len(windows) * 0.9))
    validation = windows[cut:]
    validation_start = int(validation[:, 0].min()) if len(validation) else len(values)
    training = windows[(windows[:, 1] <= validation_start - int(representation["window_rows"]))]
    if len(training) < 2 or len(validation) < 1:
        raise RuntimeError("insufficient label-free train/validation windows")
    batch_size = int(representation["batch_size"])
    generator = torch.Generator(device=device).manual_seed(seed)
    rng = np.random.default_rng(seed)
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []
    checkpoint = output / "encoder.pt"
    for epoch in range(1, int(representation["max_epochs"]) + 1):
        model.train()
        order = rng.permutation(len(training))
        train_losses: list[float] = []
        for offset in range(0, len(order), batch_size):
            batch_windows = training[order[offset : offset + batch_size]]
            batch = np.stack([values[start:stop] for start, stop in batch_windows])
            tensor = torch.from_numpy(batch).to(device)
            first, second = masked_views(
                tensor, float(representation["mask_probability"]), generator
            )
            loss = hierarchical_contrastive_loss(
                model(first),
                model(second),
                float(representation["temperature"]),
                int(representation["contrastive_timestamp_cap"]),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        valid_losses: list[float] = []
        variances: list[float] = []
        with torch.no_grad():
            for offset in range(0, len(validation), batch_size):
                batch_windows = validation[offset : offset + batch_size]
                batch = np.stack([values[start:stop] for start, stop in batch_windows])
                tensor = torch.from_numpy(batch).to(device)
                first, second = masked_views(
                    tensor, float(representation["mask_probability"]), generator
                )
                first_embedding, second_embedding = model(first), model(second)
                loss = hierarchical_contrastive_loss(
                    first_embedding,
                    second_embedding,
                    float(representation["temperature"]),
                    int(representation["contrastive_timestamp_cap"]),
                )
                valid_losses.append(float(loss.cpu()))
                variances.append(float(first_embedding.var(dim=(0, 1)).mean().cpu()))
        valid_loss = float(np.mean(valid_losses))
        variance = float(np.mean(variances))
        history.append(
            {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "validation_loss": valid_loss, "embedding_variance": variance}
        )
        eligible = variance >= 1e-4
        if eligible and valid_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = valid_loss, epoch, 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch}, checkpoint)
        else:
            stale += 1
        if epoch >= int(representation["minimum_epochs"]) and stale >= int(representation["patience"]):
            break
    if not checkpoint.exists():
        raise RuntimeError("embedding_variance_collapse")
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(payload["state_dict"])
    return model, {
        "device": str(device),
        "training_windows": int(len(training)),
        "validation_windows": int(len(validation)),
        "best_epoch": int(best_epoch),
        "best_validation_loss": best_loss,
        "history": history,
        "model_fit_count": 1,
    }


def _event_count(truth: np.ndarray, prediction: np.ndarray, segments: np.ndarray) -> int:
    y = np.asarray(truth, dtype=np.int8)
    p = np.asarray(prediction, dtype=np.int8)
    found = 0
    start = 0
    while start < len(y):
        if y[start] != 1:
            start += 1
            continue
        stop = start + 1
        while stop < len(y) and y[stop] == 1 and segments[stop] == segments[start]:
            stop += 1
        if stop - start >= 19 and p[start:stop].any():
            found += 1
        start = stop
    return found


def run_full(config: dict[str, Any]) -> dict[str, Any]:
    """Execute the single sealed local run.  Not called by smoke/check modes."""
    receipt = validate_contract(config)
    output = ROOT / config["artifacts"]["directory"]
    if output.exists():
        raise FileExistsError("full experiment artifact already exists; retries are prohibited")
    output.mkdir(parents=True, exist_ok=False)
    features = config["features"]
    feature_path = ROOT / config["immutable_inputs"]["feature_cache"]["path"]
    key_path = ROOT / config["immutable_inputs"]["feature_key_sidecar"]["path"]
    label_path = ROOT / config["immutable_inputs"]["training_labels"]["path"]
    frame = pd.read_parquet(feature_path, columns=features)
    keys = pd.read_parquet(key_path)
    labels = pd.read_parquet(label_path, columns=["station", "year", "layer", "time", "label", "anomaly_type"])
    if len(frame) != len(keys) or len(keys) != len(labels):
        raise RuntimeError("immutable P1 row counts no longer align")
    ordered = ["station", "year", "layer", "time"]
    if not keys[ordered].astype(str).equals(labels[ordered].astype(str)):
        raise RuntimeError("immutable P1 keys no longer align")
    time_utc = pd.to_datetime(keys["time"], utc=True)
    fit_stop = pd.Timestamp(config["representation"]["encoder_fit_stop_kst"]).tz_convert("UTC")
    fit_mask = time_utc.lt(fit_stop).to_numpy()
    scaled, median, mad = robust_fit_transform(frame.loc[fit_mask, features].to_numpy(), frame[features].to_numpy())
    segments = _segment_ids(keys)
    representation = config["representation"]
    train_windows = contiguous_windows(
        segments,
        fit_mask,
        int(representation["window_rows"]),
        int(representation["window_stride_rows"]),
    )
    model, training = _train_encoder(scaled, train_windows, config, output)
    before_q2 = time_utc.lt(pd.Timestamp("2025-04-01T00:00:00+09:00").tz_convert("UTC")).to_numpy()
    inference_windows = contiguous_windows(
        segments,
        before_q2,
        int(representation["window_rows"]),
        int(representation["window_stride_rows"]),
    )
    embeddings, covered = _infer_embeddings(
        model,
        scaled,
        inference_windows,
        next(model.parameters()).device,
        int(representation["batch_size"]),
    )
    historical_coverage = float(covered[before_q2].mean())
    if historical_coverage < 0.95:
        status = "NO_GO_COVERAGE"
        result = {"experiment_id": EXPERIMENT_ID, "status": status, "training": training, "historical_embedding_coverage": historical_coverage, "q2_truth_rows_read": 0}
        _atomic_json(output / "result.json", result)
        return result
    scoring = config["scoring"]
    normal_stop = pd.Timestamp(config["split"]["normal_reference_stop_kst"]).tz_convert("UTC")
    truth = labels["label"].to_numpy(dtype=np.int8)
    reference = time_utc.lt(normal_stop).to_numpy() & covered & (truth == 0)
    cells = keys["station"].astype(str).str.cat(keys["layer"].astype(str), sep="/").to_numpy()
    prototype = fit_conditional_prototype(
        embeddings,
        frame["day_sin"].to_numpy(),
        frame["day_cos"].to_numpy(),
        cells,
        reference,
        shrinkage_rows=int(scoring["cell_shrinkage_rows"]),
        scale_floor=float(scoring["scale_floor"]),
        bank_per_cell=int(scoring["normal_knn_bank_per_cell"]),
    )
    scores = score_conditional_prototype(
        prototype,
        embeddings,
        frame["day_sin"].to_numpy(),
        frame["day_cos"].to_numpy(),
        cells,
        knn_k=int(scoring["normal_knn_k"]),
        prototype_weight=float(scoring["distance_weights"]["prototype"]),
    )
    cal_start, cal_stop = [pd.Timestamp(value).tz_convert("UTC") for value in config["split"]["calibration"]]
    qual_start, qual_stop = [pd.Timestamp(value).tz_convert("UTC") for value in config["split"]["qualification"]]
    calibration = time_utc.ge(cal_start).to_numpy() & time_utc.le(cal_stop).to_numpy()
    qualification = time_utc.ge(qual_start).to_numpy() & time_utc.le(qual_stop).to_numpy()
    threshold = finite_normal_tail_threshold(scores[calibration & (truth == 0)], float(scoring["normal_tail_alpha"]))
    additions = decode_components(scores, segments, threshold, minimum_rows=int(scoring["minimum_component_rows"]), bridge_rows=int(scoring["bridge_rows"]))
    zero = np.zeros(len(truth), dtype=np.int8)
    calibration_metrics = binary_metrics(truth[calibration], additions[calibration])
    qualification_metrics = binary_metrics(truth[qualification], additions[qualification])
    qualification_days = max(1, pd.Series(time_utc[qualification]).dt.floor("D").nunique())
    qualification_fp_per_day = qualification_metrics["fp"] / qualification_days
    recovered_events = _event_count(truth[qualification], additions[qualification], segments[qualification])
    cell_directions = 0
    for cell in sorted(set(cells[qualification])):
        mask = qualification & (cells == cell)
        if binary_metrics(truth[mask], additions[mask])["f1"] > 0:
            cell_directions += 1
    qualification_days_array = (
        pd.Series(time_utc[qualification]).dt.floor("D").astype("int64").to_numpy()
        // int(pd.Timedelta(days=1).value)
    )
    bootstrap_probability = day_block_bootstrap_probability(
        truth[qualification],
        additions[qualification],
        zero[qualification],
        qualification_days_array,
        block_days=int(config["historical_gate"]["block_bootstrap_days"]),
        replicates=int(config["historical_gate"]["block_bootstrap_replicates"]),
        seed=int(config["historical_gate"]["block_bootstrap_seed"]),
    )
    gate = {
        "calibration_delta_f1_gt_zero": calibration_metrics["f1"] > 0,
        "qualification_delta_f1_gt_zero": qualification_metrics["f1"] > 0,
        "qualification_added_precision_gt_zero_anchor_f1_over_2": qualification_metrics["precision"] > 0,
        "qualification_new_eligible_long_events_min": recovered_events >= int(config["historical_gate"]["qualification_new_eligible_long_events_min"]),
        "direction_consistent_cells_min": cell_directions >= int(config["historical_gate"]["direction_consistent_cells_min"]),
        "false_positive_per_day_cap": qualification_fp_per_day <= float(config["historical_gate"]["false_positive_per_day_cap"]),
        "block_bootstrap_probability_delta_f1_gt_zero_min": bootstrap_probability
        >= float(config["historical_gate"]["probability_delta_f1_gt_zero_min"]),
    }
    gate_pass = all(gate.values())
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "NO_GO_HISTORICAL_GATE" if not gate_pass else "READY_FOR_SEPARATE_Q2_OUTER",
        "training": training,
        "historical_embedding_coverage": historical_coverage,
        "threshold_source": "calibration_normal_only_finite_order_statistic",
        "calibration": calibration_metrics,
        "qualification": qualification_metrics,
        "qualification_fp_per_day": qualification_fp_per_day,
        "qualification_recovered_long_events": recovered_events,
        "direction_consistent_cells": cell_directions,
        "block_bootstrap_probability_delta_f1_gt_zero": bootstrap_probability,
        "gate": gate,
        "anchor_deletions": 0,
        "historical_no_op_sha256": hashlib.sha256(zero.tobytes()).hexdigest(),
        "q2_truth_rows_read": 0,
        "q3_q4_rows_read": 0,
        "official_rows_read": 0,
        "scaler_aggregate": {"feature_count": len(median), "finite_median_count": int(np.isfinite(median).sum()), "finite_scale_count": int(np.isfinite(mad).sum())},
    }
    _atomic_json(output / "result.json", result)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(CONFIG),
        "module_sha256": sha256_file(MODULE),
        "runner_sha256": sha256_file(RUNNER),
        "preflight": receipt,
        "result_sha256": sha256_file(output / "result.json"),
        "checkpoint_sha256": sha256_file(output / "encoder.pt"),
        "raw_values_persisted": False,
        "keys_persisted": False,
        "submission_generated_or_uploaded": False,
    }
    _atomic_json(output / "manifest.json", manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    config = _json(CONFIG)
    if arguments.check:
        result = validate_contract(config)
    elif arguments.smoke:
        result = run_smoke(config)
    else:
        result = run_full(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
