"""Replay original P1 O/B/MS weights with general rules, without archived answers.

Diagnostic only: no fitting of predictive models, no CSV writes or uploads.
The encoder is refitted to historical TRAIN features as in the original loader.
This proves saved-weight replay, not clean-room source-only retraining.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from p1_historical_path_audit_20260906_v1 import (
    ADD_CELLS,
    REMOVE_CELLS,
    compose_mstcn_spike,
    compose_tree,
    digest,
)
from p1_historical_saved_replay_20260906_v1 import EXPECTED, PINS, TEST_SHA

PACKAGE = Path("artifacts/official_final_submission_20260905/P1")
PACKAGE_PINS = {
    "04_predict/p1_pipeline.py":
        "41936b2f0cd408207ed83c83d176ea3389703d218840615aa9158c65ad6d9480",
    "07_source/scripts/run_p1_incumbent_preserving_mstcn_asrf_v2.py":
        "78df1bd3b4777560b0134bc69ad45839c8c0093da39d2c1d3d458b3a3ba9b87a",
    "07_source/configs/experiments/p1_incumbent_preserving_mstcn_asrf_v2.json":
        "1f8940d29ea6b047273e4f53445f62230e7d72bf1f0b14abe9fb18476f0345f0",
    "07_source/src/p1_qc/ms_tcn_asrf.py":
        "57c135bfe746d06a53e5c3b83517cb96f8262a765febefbf43e1d3a3c344fd7f",
    "07_source/src/p1_qc/ms_tcn_asrf_data.py":
        "cf5dc2dbbb3ecf05c489b661f5427ff225caaf28af5fb44d292e3289c7bb9adf",
    "01_data/derived/train_features.parquet":
        "f37c56ff016e90fb9a8d86299b4d9528c8f2e03181d326169b561fe3b27bc912",
    "01_data/derived/train_features.json":
        "396386d48bf3588bd765d26d66625184a5af9a41312703ffbcfd58da9dbe0abd",
    "01_data/derived/test_features.parquet":
        "9442ebcda96cd626185718e17a139f9649eb741c3126d97200fdf6dc9ba399c6",
    "01_data/derived/test_features.json":
        "0cefaf5c5a5a898a0ae436b67993f5fa42f6f08050c9d20160dda5507178c10d",
    "03_model/weights/full_width_512_seed_20260827_epoch_150_state.pt":
        "4660eff7708f1acd77f44651040575d6776d64a80fd60b17835a36a3296a5e7a",
    "03_model/weights/full_width_512_seed_20260839_epoch_150_state.pt":
        "83e46d2cd49749d7ad491146f1e5efd2caf38f47d620c4352cd22141ae3e96c0",
    "03_model/weights/full_width_512_seed_20260863_epoch_150_state.pt":
        "fbcbd10b1b8b9df811ba678d56bfe830ea7598d023ada9048c921696f461ca61",
}
RAW_PINS = {
    "test.csv": TEST_SHA,
    "train.csv": "20b656b0cbd524ad9da0bae8ecb6e0bacfc006e05810b37e83f29a5fa8e65cd2",
    "sample_submission.csv":
        "e7027bcb56836587715e5cd818c6d595ebef4a8518538fa7de7bdcba2b7fc1e4",
}
BASE_SHA = "a52dc49c5f522ae92eb67805a8a567dc04d62791725a5de622f544af7a3ce33b"
FINAL_SHA = "57844ef235f987059d17bddb035cc0a98b92bb0d53263cf878c68b31bbc53687"


def csv_digest(frame):
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def trees(root, data):
    import joblib
    import pandas as pd

    from p1_qc.data import load_dataset
    from p1_qc.features import FeatureBundle
    from p1_qc.pipeline import apply_postprocess, load_model, predict_submission
    from p1_qc.rules import detect_plateaus, detect_singleton_spikes
    from p1_qc.submission import build_submission

    test = load_dataset(data / "test.csv", kind="test")
    meta = json.loads((root / list(PINS)[3]).read_text(encoding="utf-8"))
    bundle = FeatureBundle(pd.read_parquet(root / list(PINS)[2]),
                           tuple(meta["feature_columns"]), tuple(meta["categorical_columns"]))
    o, _ = predict_submission(load_model(root / list(PINS)[0]), test, bundle)
    b_model = joblib.load(root / list(PINS)[1])
    if b_model["branch"] != "event_day_balanced_binary_lgbm":
        raise RuntimeError("Unexpected original B branch")
    features = b_model["encoder"].transform(bundle)
    probability = np.mean(np.vstack([
        p["global"].predict_proba(features)[:, 1] for p in b_model["packages"]
    ]), axis=0).astype(np.float64)
    plateau = detect_plateaus(test).to_numpy(dtype=bool)
    spike = detect_singleton_spikes(test).to_numpy(dtype=bool)
    labels = apply_postprocess(test, probability, plateau, spike, b_model["postprocess"])
    types = np.full(len(test), "", dtype=object)
    types[plateau & labels.astype(bool)] = "flatline"
    types[spike & labels.astype(bool)] = "spike"
    b = build_submission(test, labels, types)
    router_bits, gi_bits = compose_tree(test.station, test.layer, o.label, b.label,
                                       add_cells=ADD_CELLS, remove_cells=REMOVE_CELLS)
    gi_types = b.anomaly_type.to_numpy().copy()
    added = (gi_bits == 1) & b.label.eq(0).to_numpy()
    gi_types[added] = o.anomaly_type.to_numpy()[added]
    router = build_submission(test, router_bits, gi_types)
    checks = {name: csv_digest(frame) for name, frame in {"O": o, "B": b,
                                                        "router": router}.items()}
    if checks != EXPECTED:
        raise RuntimeError(f"Original tree replay mismatch: {checks}")
    return test, router_bits, gi_bits, gi_types, checks


def run(root, data, output, check_only):
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    package = root / PACKAGE
    pins = {**{str(root / p): h for p, h in PINS.items()},
            **{str(package / p): h for p, h in PACKAGE_PINS.items()},
            **{str(data / p): h for p, h in RAW_PINS.items()}}
    for name, sha in pins.items():
        if digest(Path(name)) != sha:
            raise RuntimeError(f"Pin mismatch before inference: {name}")
    # The already-imported P1 tree package and the packaged MS source must agree.
    for name in ("ms_tcn_asrf.py", "ms_tcn_asrf_data.py"):
        if digest(root / "src/p1_qc" / name) != PACKAGE_PINS[f"07_source/src/p1_qc/{name}"]:
            raise RuntimeError("Live and owned MS modules differ")
    result = {"status": "PREFLIGHT_PASS", "pinned_sha256": pins,
              "script_sha256": digest(Path(__file__)), "model_fits": 0,
              "archived_answer_csv_reads": 0, "row_patch_json_reads": 0,
              "hidden_truth_reads": 0, "candidate_csv_writes": 0, "uploads": 0}
    if not check_only:
        sys.path.insert(0, str(root / "src"))
        test, router, gi, gi_types, tree_hashes = trees(root, data)
        print("Original O/B/router replay exact; preparing MS inference", flush=True)
        loader_path = package / "04_predict/p1_pipeline.py"
        spec = importlib.util.spec_from_file_location("original_p1_surfaces", loader_path)
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        source, config, _encoder, train, holdout, sample = loader.load_surfaces(package, data)
        if not loader.keys_equal(test, sample):
            raise RuntimeError("Historical input order mismatch")
        _np, _pd, torch, _model, _data = source._load_scientific()
        if not torch.cuda.is_available():
            raise RuntimeError("Original GPU inference device unavailable; no fallback")
        device = torch.device("cuda")
        torch.set_num_threads(8)
        row = np.zeros(holdout.surface.rows, dtype=np.float32)
        boundary = np.zeros((holdout.surface.rows, 2), dtype=np.float32)
        kinds = np.zeros((holdout.surface.rows, 5), dtype=np.float32)
        for seed in (20260827, 20260839, 20260863):
            path = package / "03_model/weights" / f"full_width_512_seed_{seed}_epoch_150_state.pt"
            payload = torch.load(path, map_location=device, weights_only=False)
            if payload["seed"] != seed or payload["epoch"] != 150:
                raise RuntimeError("Original checkpoint metadata drift")
            capacity = source._config_for_capacity(config, width=512, seed=seed)
            model = source._new_model(train.features.shape[1], capacity, device)
            model.load_state_dict(payload["state_dict"], strict=True)
            prediction = source.predict_encoded(model, holdout, source._all_windows(holdout, capacity),
                                                batch_size=int(capacity["training"]["batch_size"]),
                                                device=device)
            row += prediction.row_probability.astype(np.float32, copy=False)
            boundary += prediction.boundary_probability.astype(np.float32, copy=False)
            kinds += prediction.type_probability.astype(np.float32, copy=False)
            del model, prediction, payload
            print(f"Original MS seed {seed}: inference complete", flush=True)
        bundle = source.PredictionBundle(row / 3.0, boundary / 3.0, kinds / 3.0)
        proposal = source.decode_long_event_segments(
            source._decoder_row_probability(bundle, config), bundle.boundary_probability,
            holdout.layout, high_threshold=0.8,
            snap_radius=int(config["decoder"]["boundary_peak_snap_radius_rows"]),
            minimum_rows=int(config["decoder"]["minimum_added_segment_rows"]),
            maximum_rows=source._maximum_segment_rows(config),
        ).astype(np.int8)
        base = sample[loader.KEYS].copy()
        base["label"] = (router.astype(bool) | proposal.astype(bool)).astype(np.int8)
        final = base.copy()
        final["label"] = compose_mstcn_spike(router, gi, proposal, gi_types)
        from p1_qc.submission import validate_submission

        checks = {"tree_sha256": tree_hashes, "base_sha256": csv_digest(base),
                  "final_sha256": csv_digest(final),
                  "base_hash_match": csv_digest(base) == BASE_SHA,
                  "final_hash_match": csv_digest(final) == FINAL_SHA,
                  "base_validator": validate_submission(base, test),
                  "final_validator": validate_submission(final, test),
                  "general_GI_rule_additions": int((final.label != base.label).sum())}
        result.update({
            "status": "FULL_ORIGINAL_SAVED_WEIGHT_REPLAY_EXACT_PASS"
                      if checks["base_hash_match"] and checks["final_hash_match"]
                      else "FULL_ORIGINAL_REPLAY_HASH_MISMATCH",
            "checks": checks, "encoder_fit_scope": "TRAIN_ONLY_ORIGINAL_776706_ROWS",
            "training_source_rows_read": train.surface.rows,
            "official_observation_rows_read": len(test),
            "sample_dummy_schema_rows_read": len(sample),
            "predictive_model_fits": 0, "saved_predictive_models": 7,
            "scope": "Original saved weights and derived caches. No original answer CSV or "
                     "fixed-key patch is used. NOT a source-only scratch-training proof. "
                     "Original OOF cell-selector provenance remains incomplete.",
        })
    result["runtime_seconds"] = time.perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "checks": result.get("checks"),
                      "runtime_seconds": result["runtime_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    run(args.root.resolve(), args.data.resolve(), args.output.resolve(), args.check_only)
