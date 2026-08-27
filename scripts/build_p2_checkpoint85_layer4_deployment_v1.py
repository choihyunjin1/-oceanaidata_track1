"""Legacy builder for the frozen P2 checkpoint-0.85 research surface.

The source experiment explicitly forbids candidate/test prediction.  The guard in
``main`` therefore fails closed; the pure ``assemble_candidate`` helper remains for
auditing the already-quarantined artifact and its unit tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from p2_restore import (
    deep_data,
)
from p2_restore import (
    joint_hydrographic_multitask as model_module,
)
from p2_restore import (
    joint_hydrographic_multitask_layer4_execution_r3 as r3,
)
from p2_restore.data import KEYS, load_p2_data, resolve_data_dir
from p2_restore.submission import build_submission, validate_submission

SEEDS = (20260823, 20260824, 20260825)
MODEL_RECIPE = {
    "input_channels": 54,
    "hidden_width": 160,
    "dilations": [1, 2, 4, 8, 16, 32],
    "dropout": 0.05,
    "chunk_length": 512,
    "chunk_stride": 384,
    "batch_size": 12,
}
SOURCE_EXPERIMENT_CONFIG = Path(
    "configs/experiments/p2_joint_hydrographic_multitask_layer4_checkpoint_v1.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_submission(path: Path, test_index: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"station": "string", "time": "string"})
    validate_submission(frame, test_index)
    return frame


def assemble_candidate(
    anchor: pd.DataFrame,
    test_index: pd.DataFrame,
    layer4_prediction: np.ndarray,
    *,
    alpha: float,
) -> pd.DataFrame:
    """Preserve Layers 2/3 exactly and blend only registered Layer-4 rows."""

    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be within [0, 1]")
    if not anchor[KEYS].equals(test_index[KEYS]):
        raise ValueError("anchor keys differ from test_index")
    layer4 = test_index["layer"].to_numpy(int) == 4
    direct = np.asarray(layer4_prediction, dtype=np.float64)
    if direct.shape != (int(layer4.sum()),) or not np.isfinite(direct).all():
        raise ValueError("Layer-4 prediction shape or finiteness is invalid")
    values = anchor["temp"].to_numpy(np.float64).copy()
    before = values.copy()
    values[layer4] = before[layer4] + float(alpha) * (direct - before[layer4])
    if not np.array_equal(values[~layer4], before[~layer4]):
        raise AssertionError("Layer 2/3 anchor values changed")
    return build_submission(test_index, values)


def _predict_seed(
    checkpoint_path: Path,
    panel: model_module.JointHydrographicPanel,
    required_positions: np.ndarray,
) -> np.ndarray:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected = {
        "schema_version",
        "seed",
        "selected_epoch",
        "model_state",
        "input_center",
        "input_scale",
        "target_center",
        "target_scale",
    }
    if set(payload) != expected:
        raise ValueError(f"checkpoint payload changed: {checkpoint_path}")
    network = model_module.JointHydrographicTCN(
        MODEL_RECIPE["input_channels"],
        hidden=MODEL_RECIPE["hidden_width"],
        dilations=tuple(MODEL_RECIPE["dilations"]),
        dropout=MODEL_RECIPE["dropout"],
    )
    network.load_state_dict(payload["model_state"], strict=True)
    network = network.to(torch.device("cuda"))
    normalizer = model_module.JointHydrographicNormalizer(
        input_center=payload["input_center"].numpy().astype(np.float64),
        input_scale=payload["input_scale"].numpy().astype(np.float64),
        target_center=payload["target_center"].numpy().astype(np.float64),
        target_scale=payload["target_scale"].numpy().astype(np.float64),
    )
    numerical = SimpleNamespace(np=np, torch=torch, deep_data=deep_data)
    physical, _audit = r3._predict_panel_temperature(
        network,
        panel,
        normalizer,
        config={
            "model_and_training": MODEL_RECIPE,
            "curve_protocol": {"batch_size": MODEL_RECIPE["batch_size"]},
        },
        numerical=numerical,
        device=torch.device("cuda"),
        required_layer4_positions=required_positions,
    )
    network.cpu()
    torch.cuda.empty_cache()
    return physical[required_positions, 2]


def _write_candidate(
    output_dir: Path,
    name: str,
    frame: pd.DataFrame,
    test_index: pd.DataFrame,
) -> dict[str, object]:
    candidate_dir = output_dir / name
    candidate_dir.mkdir(parents=True, exist_ok=False)
    path = candidate_dir / "P2_submission.csv"
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    validation = validate_submission(path, test_index)
    return {
        "name": name,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "validation": validation,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--axis-anchor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("dry run only: pass --execute to build non-uploaded candidates")
    source_config = json.loads(SOURCE_EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
    if (
        source_config.get("candidate_or_test_prediction_allowed") is not True
        or source_config.get("upload_allowed") is not True
    ):
        raise PermissionError(
            "source experiment is research-only and forbids candidate/test prediction; "
            "do not build or upload from this checkpoint surface"
        )
    if args.output_dir.exists():
        raise FileExistsError(f"append-only output exists: {args.output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen checkpoint inference")

    data = load_p2_data(resolve_data_dir(args.data_dir))
    incumbent = _load_submission(args.incumbent.resolve(strict=True), data.test_index)
    axis_anchor = _load_submission(args.axis_anchor.resolve(strict=True), data.test_index)
    panel = model_module.build_joint_hydrographic_panel(data.observations)
    index_times = pd.to_datetime(data.test_index["time"], utc=True)
    positions = panel.times.get_indexer(index_times)
    if (positions < 0).any():
        raise ValueError("test_index contains timestamps absent from observations")
    layer4_mask = data.test_index["layer"].to_numpy(int) == 4
    required_positions = positions[layer4_mask].astype(np.int64, copy=False)
    if len(np.unique(required_positions)) != int(layer4_mask.sum()):
        raise ValueError("Layer-4 test positions are not unique")

    predictions = []
    checkpoint_pins = []
    for seed in SEEDS:
        path = (
            args.checkpoint_root
            / "cells"
            / "outer_2025_jul_aug"
            / "fraction_085"
            / f"seed_{seed}"
            / "full_refit.pt"
        ).resolve(strict=True)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if int(payload["seed"]) != seed or int(payload["selected_epoch"]) != 97:
            raise ValueError("frozen seed or common checkpoint epoch changed")
        del payload
        predictions.append(_predict_seed(path, panel, required_positions))
        checkpoint_pins.append(
            {"seed": seed, "path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    direct = np.mean(np.stack(predictions, axis=0), axis=0)
    if not np.isfinite(direct).all() or not ((direct >= -5) & (direct <= 45)).all():
        raise ValueError("ensemble Layer-4 prediction is invalid")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    candidates = [
        _write_candidate(
            args.output_dir,
            "P2_1_CHECKPOINT85_L4_BLEND50",
            assemble_candidate(incumbent, data.test_index, direct, alpha=0.5),
            data.test_index,
        ),
        _write_candidate(
            args.output_dir,
            "P2_2_CHECKPOINT85_L4_FULL",
            assemble_candidate(incumbent, data.test_index, direct, alpha=1.0),
            data.test_index,
        ),
        _write_candidate(
            args.output_dir,
            "P2_3_AXIS_U_PLUS_CHECKPOINT85_L4",
            assemble_candidate(axis_anchor, data.test_index, direct, alpha=1.0),
            data.test_index,
        ),
    ]
    incumbent_values = incumbent["temp"].to_numpy(float)
    axis_values = axis_anchor["temp"].to_numpy(float)
    for item, anchor_values in zip(
        candidates,
        (incumbent_values, incumbent_values, axis_values),
        strict=True,
    ):
        frame = pd.read_csv(item["path"])
        delta = frame["temp"].to_numpy(float) - anchor_values
        item["changed_rows_vs_anchor"] = int(np.count_nonzero(delta))
        item["rms_change_vs_anchor_c"] = float(np.sqrt(np.mean(delta**2)))
        item["max_abs_change_vs_anchor_c"] = float(np.max(np.abs(delta)))

    manifest = {
        "schema_version": "p2.checkpoint85_layer4_deployment.v1",
        "status": "READY_NOT_UPLOADED",
        "scientific_role": "EXPOSED_SURFACE_OFFICIAL_PROBE_ELIGIBLE",
        "checkpoint_fraction": 0.85,
        "selected_common_epoch": 97,
        "seeds": list(SEEDS),
        "checkpoint_pins": checkpoint_pins,
        "incumbent": {
            "path": str(args.incumbent.resolve()),
            "sha256": _sha256(args.incumbent),
        },
        "axis_anchor": {
            "path": str(args.axis_anchor.resolve()),
            "sha256": _sha256(args.axis_anchor),
        },
        "candidates": candidates,
        "official_upload_performed": False,
        "hidden_target_values_read": False,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
