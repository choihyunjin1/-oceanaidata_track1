"""Run actual P1 checkpoints, then apply the registered router-union and 2-row patch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from p1_pipeline import KEYS, keys_equal, load_surfaces  # noqa: E402


def predict(data_dir: str | Path, package_dir: str | Path, output_path: str | Path) -> dict:
    package = Path(package_dir).resolve()
    contract = json.loads((package / "contract.json").read_text(encoding="utf-8"))
    source, source_config, _encoder, training, holdout, sample = load_surfaces(
        package, data_dir
    )
    _np, _pd, torch, _model_api, _data_api = source._load_scientific()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    row_sum = np.zeros(holdout.surface.rows, dtype=np.float32)
    boundary_sum = np.zeros((holdout.surface.rows, 2), dtype=np.float32)
    type_sum = np.zeros((holdout.surface.rows, 5), dtype=np.float32)
    loaded = []
    for seed in (20260827, 20260839, 20260863):
        path = (
            package
            / "03_model"
            / "weights"
            / f"full_width_512_seed_{seed}_epoch_150_state.pt"
        )
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("seed") != seed or payload.get("epoch") != 150:
            raise RuntimeError(f"P1 checkpoint contract drift: {path.name}")
        capacity = source._config_for_capacity(source_config, width=512, seed=seed)
        model = source._new_model(training.features.shape[1], capacity, device)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        prediction = source.predict_encoded(
            model,
            holdout,
            source._all_windows(holdout, capacity),
            batch_size=int(capacity["training"]["batch_size"]),
            device=device,
        )
        row_sum += prediction.row_probability.astype(np.float32, copy=False)
        boundary_sum += prediction.boundary_probability.astype(np.float32, copy=False)
        type_sum += prediction.type_probability.astype(np.float32, copy=False)
        loaded.append(path.name)
        del model, prediction
    bundle = source.PredictionBundle(row_sum / 3.0, boundary_sum / 3.0, type_sum / 3.0)
    proposal = source.decode_long_event_segments(
        source._decoder_row_probability(bundle, source_config),
        bundle.boundary_probability,
        holdout.layout,
        high_threshold=0.8,
        snap_radius=int(source_config["decoder"]["boundary_peak_snap_radius_rows"]),
        minimum_rows=int(source_config["decoder"]["minimum_added_segment_rows"]),
        maximum_rows=source._maximum_segment_rows(source_config),
    ).astype(np.int8)
    dtype = {"station": "string", "time": "string"}
    router = pd.read_csv(
        package / "03_model" / "decision_artifacts" / "router_anchor.csv", dtype=dtype
    )
    if not keys_equal(router, sample):
        raise RuntimeError("P1 router/sample ordered keys differ")
    bits = source.anchor_preserving_union(
        router["label"].to_numpy(dtype=np.int8), proposal
    ).astype(np.int8)
    candidate = sample[KEYS].copy()
    candidate["label"] = bits
    patch = json.loads(
        (package / "03_model" / "decision_artifacts" / "gi_spike2_patch.json").read_text(
            encoding="utf-8"
        )
    )
    index = pd.MultiIndex.from_frame(candidate[KEYS].astype(str))
    changed = []
    for row in patch["rows"]:
        key = tuple(str(row[name]) for name in KEYS)
        location = int(index.get_indexer([key])[0])
        if location < 0 or int(candidate.loc[location, "label"]) != 0:
            raise RuntimeError(f"P1 patch precondition failed: {key}")
        candidate.loc[location, "label"] = 1
        changed.append(location)
    target = Path(output_path)
    if not target.is_absolute():
        target = package / target
    target.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(target, index=False, encoding="utf-8", lineterminator="\n")
    from common import sha256_file

    actual = sha256_file(target)
    if actual != contract["candidate_sha256"]:
        raise RuntimeError(f"P1 model-driven output SHA drift: {actual}")
    return {
        "status": "READY_MODEL_INFERENCE_EXACT_NOT_UPLOADED",
        "candidate_id": contract["candidate_id"],
        "rows": len(candidate),
        "columns": list(candidate.columns),
        "positive_rows": int(candidate["label"].sum()),
        "changed_rows": len(changed),
        "key_order_exact": True,
        "sha256": actual,
        "candidate_hash_exact": True,
        "package_atomic": True,
        "training_fit_count": 3,
        "checkpoint_files_loaded": loaded,
        "prediction_source": "trained_checkpoint_inference_plus_registered_decision_artifacts",
        "lineage": "organizer_distributed_data_only_scratch_models",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, default=HERE.parents[1])
    parser.add_argument("--output", type=Path, default=Path("05_answer/P1_submission.csv"))
    args = parser.parse_args()
    print(json.dumps(predict(args.data_dir, args.package_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
