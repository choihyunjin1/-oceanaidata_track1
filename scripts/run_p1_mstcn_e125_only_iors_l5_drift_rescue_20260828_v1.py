"""Replay the frozen full-data MS-TCN path and audit one checkpoint disagreement cell.

The only allowed candidate is the current champion OR the fixed cell
I-ORS/layer-5/e125-positive/e150-negative.  The runner does not read test
labels and never uploads.  It trains each frozen seed once through epoch 150,
captures epoch 125, and requires the epoch-150 test arrays to replay exactly.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_mstcn_e125_only_iors_l5_drift_rescue_20260828_v1"
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
KEYS = ["station", "year", "layer", "time"]


class ContractError(RuntimeError):
    """Raised when the frozen experiment contract changes."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def equal_arrays(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> bool:
    return left.keys() == right.keys() and all(
        np.array_equal(left[name], right[name]) for name in left
    )


def prediction_arrays(bundle: Any) -> dict[str, np.ndarray]:
    return {
        "row_probability": bundle.row_probability.astype(np.float32, copy=False),
        "boundary_probability": bundle.boundary_probability.astype(np.float32, copy=False),
        "type_probability": bundle.type_probability.astype(np.float32, copy=False),
    }


def fit_seed(
    source: Any,
    source_config: dict[str, Any],
    training: Any,
    holdout: Any,
    *,
    seed: int,
    artifact_dir: Path,
    device: Any,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    _, _, torch, _, _ = source._load_scientific()
    e125_path = artifact_dir / f"seed_{seed}_epoch_125_test_prediction.npz"
    e150_path = artifact_dir / f"seed_{seed}_epoch_150_replay_test_prediction.npz"
    history_path = artifact_dir / f"seed_{seed}_history.json"
    receipt_path = artifact_dir / f"seed_{seed}_receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        for path, expected in (
            (e125_path, receipt["epoch125_sha256"]),
            (e150_path, receipt["epoch150_sha256"]),
            (history_path, receipt["history_sha256"]),
        ):
            require(path.is_file() and sha256(path) == expected, f"seed artifact changed: {path}")
        return load_arrays(e125_path), load_arrays(e150_path), receipt

    capacity = source._config_for_capacity(source_config, width=512, seed=seed)
    require(int(capacity["training"]["maximum_epochs"]) == 300, "schedule horizon changed")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    source._reset_cuda_peak_memory(torch, device)
    model = source._new_model(training.features.shape[1], capacity, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(capacity["training"]["learning_rate"]),
        weight_decay=float(capacity["training"]["weight_decay"]),
    )
    windows = source._selected_windows(training, capacity)
    holdout_windows = source._all_windows(holdout, capacity)
    positive_weight = source._positive_weight(training.surface.labels)
    _, total_steps, _ = source._schedule_geometry(capacity, window_count=len(windows))
    global_step = 0
    history: list[dict[str, Any]] = []
    captured125: dict[str, np.ndarray] | None = None
    started = time.perf_counter()
    for epoch in range(1, 151):
        epoch_started = time.perf_counter()
        telemetry, global_step, learning_rate = source._train_epoch(
            model,
            optimizer,
            training,
            windows,
            config=capacity,
            positive_weight=positive_weight,
            device=device,
            epoch=epoch,
            global_step=global_step,
            total_steps=total_steps,
        )
        record = source._history_record(
            epoch=epoch,
            telemetry=telemetry,
            global_step=global_step,
            learning_rate=learning_rate,
            elapsed_seconds=time.perf_counter() - epoch_started,
        )
        history.append(record)
        if epoch == 125:
            captured125 = prediction_arrays(
                source.predict_encoded(
                    model,
                    holdout,
                    holdout_windows,
                    batch_size=int(capacity["training"]["batch_size"]),
                    device=device,
                )
            )
            write_npz(e125_path, captured125)
        if epoch == 1 or epoch % 5 == 0:
            write_json(history_path, history)
            print(
                json.dumps(
                    {
                        "event": "training_progress",
                        "seed": seed,
                        "epoch": epoch,
                        "loss": record["total_loss"],
                        "seconds": record["epoch_wall_seconds"],
                    },
                    allow_nan=False,
                ),
                flush=True,
            )
    require(captured125 is not None, "epoch 125 was not captured")
    captured150 = prediction_arrays(
        source.predict_encoded(
            model,
            holdout,
            holdout_windows,
            batch_size=int(capacity["training"]["batch_size"]),
            device=device,
        )
    )
    write_npz(e150_path, captured150)
    write_json(history_path, history)
    receipt = {
        "schema_version": "p1.mstcn_e125_only.seed.20260828.v1",
        "seed": seed,
        "epochs": 150,
        "capture_epoch": 125,
        "optimizer_steps": int(global_step),
        "wall_seconds": float(time.perf_counter() - started),
        "parameter_count": int(model.trainable_parameter_count),
        "epoch125_sha256": sha256(e125_path),
        "epoch150_sha256": sha256(e150_path),
        "history_sha256": sha256(history_path),
        **source._cuda_peak_memory_receipt(torch, device),
    }
    write_json(receipt_path, receipt)
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return captured125, captured150, receipt


def ensemble(arrays: list[dict[str, np.ndarray]], source: Any) -> Any:
    count = float(len(arrays))
    return source.PredictionBundle(
        sum(item["row_probability"] for item in arrays) / count,
        sum(item["boundary_probability"] for item in arrays) / count,
        sum(item["type_probability"] for item in arrays) / count,
    )


def decode(bundle: Any, source: Any, source_config: dict[str, Any], layout: Any) -> np.ndarray:
    return source.decode_long_event_segments(
        source._decoder_row_probability(bundle, source_config),
        bundle.boundary_probability,
        layout,
        high_threshold=0.8,
        snap_radius=int(source_config["decoder"]["boundary_peak_snap_radius_rows"]),
        minimum_rows=int(source_config["decoder"]["minimum_added_segment_rows"]),
        maximum_rows=source._maximum_segment_rows(source_config),
    ).astype(np.int8)


def component_lengths(keys: pd.DataFrame, mask: np.ndarray) -> list[int]:
    selected = keys.loc[mask, ["station", "layer", "time"]].copy()
    if selected.empty:
        return []
    selected["parsed"] = pd.to_datetime(selected["time"], utc=True)
    selected = selected.sort_values(["station", "layer", "parsed"])
    station = selected["station"].astype(str).to_numpy()
    layer = selected["layer"].to_numpy()
    time_gap = selected["parsed"].diff()
    new_component = np.ones(len(selected), dtype=bool)
    if len(selected) > 1:
        new_component[1:] = (
            (station[1:] != station[:-1])
            | (layer[1:] != layer[:-1])
            | (time_gap.iloc[1:].to_numpy() != pd.Timedelta(minutes=10))
        )
    group = np.cumsum(new_component) - 1
    return [int(np.sum(group == value)) for value in np.unique(group)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--old-router", type=Path, required=True)
    parser.add_argument("--current-champion", type=Path, required=True)
    parser.add_argument("--third-candidate", type=Path, required=True)
    parser.add_argument("--ready-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    require(config["experiment_id"] == EXPERIMENT_ID, "config identity changed")
    require(not any(config["leakage_contract"].values()), "leakage/upload contract changed")
    prior_path = ROOT / config["pins"]["source_deployment_runner"]
    prior = load_module(prior_path, f"{EXPERIMENT_ID}_prior")
    _, context = prior._preflight(
        data_dir=args.data_dir.expanduser().resolve(),
        current_router=args.old_router.expanduser().resolve(),
        third_candidate=args.third_candidate.expanduser().resolve(),
        delivery_root=args.ready_root.expanduser().resolve(),
    )
    champion_path = args.current_champion.expanduser().resolve()
    require(sha256(champion_path) == config["pins"]["champion_sha256"], "champion hash changed")

    artifact_dir = ROOT / config["outputs"]["artifact_dir"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    source = context["source"]
    _, _, torch, _, _ = source._load_scientific()
    require(torch.cuda.is_available(), "CUDA is required")
    device = torch.device("cuda")
    _, training, holdout, sample, _old_router = prior._load_surfaces(
        context, args.old_router.expanduser().resolve()
    )
    champion = pd.read_csv(champion_path)
    require(prior._keys_equal(champion, sample), "champion keys differ")
    champion_bits = champion["label"].to_numpy(dtype=np.int8)
    require(np.isin(champion_bits, [0, 1]).all(), "champion label is not binary")

    epoch125: list[dict[str, np.ndarray]] = []
    epoch150: list[dict[str, np.ndarray]] = []
    receipts = []
    exact_by_seed = {}
    source_dir = ROOT / config["pins"]["source_artifact_dir"]
    for seed in config["recipe"]["seeds"]:
        arrays125, arrays150, receipt = fit_seed(
            source,
            context["source_config"],
            training,
            holdout,
            seed=int(seed),
            artifact_dir=artifact_dir,
            device=device,
        )
        pinned_path = source_dir / f"full_width_512_seed_{seed}_epoch_150_test_prediction.npz"
        require(
            sha256(pinned_path) == config["pins"]["source_e150_prediction_sha256"][str(seed)],
            f"pinned e150 prediction changed: {seed}",
        )
        pinned = load_arrays(pinned_path)
        exact = equal_arrays(arrays150, pinned)
        exact_by_seed[str(seed)] = exact
        require(exact, f"epoch150 prediction replay differs: {seed}")
        epoch125.append(arrays125)
        epoch150.append(arrays150)
        receipts.append(receipt)

    bundle125 = ensemble(epoch125, source)
    bundle150 = ensemble(epoch150, source)
    proposal125 = decode(bundle125, source, context["source_config"], holdout.layout)
    proposal150 = decode(bundle150, source, context["source_config"], holdout.layout)
    pinned150 = ensemble(
        [
            load_arrays(source_dir / f"full_width_512_seed_{seed}_epoch_150_test_prediction.npz")
            for seed in config["recipe"]["seeds"]
        ],
        source,
    )
    pinned_proposal150 = decode(pinned150, source, context["source_config"], holdout.layout)
    decoded_exact = bool(np.array_equal(proposal150, pinned_proposal150))
    require(decoded_exact, "epoch150 decoded proposal differs")

    stations = sample["station"].astype(str).to_numpy()
    layers = pd.to_numeric(sample["layer"], errors="raise").to_numpy()
    cell = (
        (stations == config["cell"]["station"])
        & (layers == int(config["cell"]["layer"]))
        & (proposal125 == 1)
        & (proposal150 == 0)
        & (champion_bits == 0)
    )
    lengths = component_lengths(sample, cell)
    gates = config["gates"]
    gate_checks = {
        "added_rows": int(gates["minimum_added_rows"]) <= int(cell.sum()) <= int(gates["maximum_added_rows"]),
        "component_count": int(gates["minimum_components"]) <= len(lengths) <= int(gates["maximum_components"]),
        "component_lengths": bool(lengths) and min(lengths) >= int(gates["minimum_component_rows"]),
        "anchor_positive_removals": True,
        "e150_prediction_array_exact": all(exact_by_seed.values()),
        "e150_decoded_exact": decoded_exact,
    }
    passed = all(gate_checks.values())
    candidate_bits = np.maximum(champion_bits, cell.astype(np.int8))
    removed = int(np.sum((champion_bits == 1) & (candidate_bits == 0)))
    require(removed == 0, "candidate removed champion positives")

    result = {
        "schema_version": "p1.mstcn_e125_only_iors_l5_drift_rescue.result.20260828.v1",
        "status": "PASS_READY_NOT_UPLOADED" if passed else "NO_GO_EXACT_NO_OUTPUT",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "experiment_id": EXPERIMENT_ID,
        "device": torch.cuda.get_device_name(device),
        "recipe": config["recipe"],
        "cell": config["cell"],
        "added_rows": int(cell.sum()),
        "component_lengths": lengths,
        "gate_checks": gate_checks,
        "gate_pass": passed,
        "epoch150_exact_by_seed": exact_by_seed,
        "epoch150_decoded_exact": decoded_exact,
        "champion_positive_rows": int(champion_bits.sum()),
        "candidate_positive_rows": int(candidate_bits.sum()),
        "anchor_positive_removed_rows": removed,
        "seed_receipts": receipts,
        "test_labels_read": False,
        "upload_performed": False,
    }
    if passed:
        candidate = sample.loc[:, KEYS].copy().assign(label=candidate_bits)
        canonical = artifact_dir / config["outputs"]["candidate_filename"]
        ready_dir = args.ready_root.expanduser().resolve() / config["outputs"]["ready_directory_name"]
        ready_dir.mkdir(parents=True, exist_ok=True)
        ready = ready_dir / config["outputs"]["candidate_filename"]
        candidate.to_csv(canonical, index=False, lineterminator="\n")
        candidate.to_csv(ready, index=False, lineterminator="\n")
        require(canonical.read_bytes() == ready.read_bytes(), "ready copy differs")
        result["candidate"] = {
            "canonical": str(canonical.resolve()),
            "ready": str(ready.resolve()),
            "bytes": ready.stat().st_size,
            "sha256": sha256(ready),
            "title": "P1 MS-TCN e125-only I-ORS L5 drift rescue v1",
            "one_line_summary": "현 공식 champion을 보존하고 두 과거 구간에서 반복된 e125-only I-ORS 5층 drift 셀만 추가합니다.",
        }
        write_json(ready_dir / "제출승인정보.json", result["candidate"] | {"upload_performed": False})
    write_json(artifact_dir / "result.json", result)
    print(json.dumps({key: result[key] for key in ("status", "added_rows", "component_lengths", "gate_checks")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
