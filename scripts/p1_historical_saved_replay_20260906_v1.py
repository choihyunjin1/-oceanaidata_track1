"""Zero-fit provenance diagnostic, not a new submission or scratch regeneration.

Read only pinned original O/B weights and organizer-derived feature cache/input.
Recreate original general composition in memory. Compare against pre-existing
SHA receipts, never load historical answer CSVs or change rules after mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from p1_historical_path_audit_20260906_v1 import (
    ADD_CELLS,
    REMOVE_CELLS,
    compose_tree,
    digest,
)

PINS = {
    "output/2026-08-20/retrain/P1/model.joblib":
        "d4b60c543691f51526b3909101763309a6560cc274386a8fb3323773ad469d12",
    "artifacts/p1_target_covariate_density_ratio_xgb_v1/models/"
    "P1_EVENT_DAY_BALANCED_LGBM_FALLBACK_V1.joblib":
        "5f7933a6fa7e2c03a84ce255538a74fa4e697052c546e87af2d8c8a3982f4895",
    "artifacts/cache/test_offline_c2a3877bdecea937.parquet":
        "9442ebcda96cd626185718e17a139f9649eb741c3126d97200fdf6dc9ba399c6",
    "artifacts/cache/test_offline_c2a3877bdecea937.json":
        "0cefaf5c5a5a898a0ae436b67993f5fa42f6f08050c9d20160dda5507178c10d",
    "src/p1_qc/pipeline.py":
        "389a905abbaf4b62e7d862c44fa25bba2e58dae7b7a7f5bcb4e1e8438d914669",
    "src/p1_qc/rules.py":
        "ec921139f210f3b264c519346547fa0e17b094f54ce83780f21cf48f5287069c",
    "src/p1_qc/postprocess.py":
        "2066d8a45c71cdd1b77365a2334c16efb075c7e6d8feb7808d76aaef22cb5bf5",
    "src/p1_qc/data.py":
        "5dc1dac588faa2f50323d15ab2d83231eb06c3a709845d539ba7efbb99addd69",
}
TEST_SHA = "6d5c6522c282651b99f4261ffa803cf99950596028e996de1e7714db77408387"
EXPECTED = {
    "O": "28243fda9bc56e25a698366823dfab3198cda21bfaec04f30fda6a899eaf0cd3",
    "B": "decedb8a9b3df7d955ae9b3848cd8f985c5228e6727accbb514516507755adbf",
    "router": "1b04e81c18d5a5cac3115c3a256e8d5a38a9493a32478a184df81fd99f9f6e5f",
}


def run(root: Path, test_path: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    hashes = {name: digest(root / name) for name in PINS}
    if hashes != PINS or digest(test_path) != TEST_SHA:
        raise RuntimeError("Pinned original artifact/input drift; no replay")
    sys.path.insert(0, str(root / "src"))
    import joblib
    import pandas as pd

    from p1_qc.data import load_dataset
    from p1_qc.features import FeatureBundle
    from p1_qc.pipeline import apply_postprocess, load_model, predict_submission
    from p1_qc.rules import detect_plateaus, detect_singleton_spikes
    from p1_qc.submission import build_submission, validate_submission

    paths = list(PINS)
    meta = json.loads((root / paths[3]).read_text(encoding="utf-8"))
    test = load_dataset(test_path, kind="test")
    frame = pd.read_parquet(root / paths[2])
    if len(test) != 169011 or len(frame) != len(test):
        raise RuntimeError("Original row count mismatch")
    # Historical cache stores features only, not year/time keys. Verify its
    # sealed byte/source lineage and ordered station/raw observations instead.
    if not frame.station.astype(str).equals(test.station.astype(str)):
        raise RuntimeError("Ordered cached stations differ")
    for col in ("temp", "psal", "depth"):
        if not np.allclose(frame[f"{col}_raw"].to_numpy(), test[col].to_numpy(),
                           rtol=1e-6, atol=1e-6, equal_nan=True):
            raise RuntimeError(f"Ordered cache raw observations differ: {col}")
    if meta["source_sha256"] != TEST_SHA or meta["parquet_sha256"] != hashes[paths[2]]:
        raise RuntimeError("Cache lineage mismatch")
    bundle = FeatureBundle(frame, tuple(meta["feature_columns"]),
                           tuple(meta["categorical_columns"]))
    original = load_model(root / paths[0])
    o, _ = predict_submission(original, test, bundle)
    print("Original O saved-model inference finished", flush=True)
    saved_b = joblib.load(root / paths[1])
    if saved_b["branch"] != "event_day_balanced_binary_lgbm":
        raise RuntimeError("Unexpected original B branch")
    features = saved_b["encoder"].transform(bundle)
    packages = saved_b["packages"]
    if len(packages) != 3 or [p["seed"] for p in packages] != saved_b["seeds"]:
        raise RuntimeError("Unexpected original B ensemble")
    probability = np.mean(np.vstack([
        p["global"].predict_proba(features)[:, 1] for p in packages
    ]), axis=0).astype(np.float64)
    if not np.isfinite(probability).all():
        raise RuntimeError("Nonfinite original B probability")
    plateau = detect_plateaus(test).to_numpy(dtype=bool)
    spike = detect_singleton_spikes(test).to_numpy(dtype=bool)
    prediction = apply_postprocess(test, probability, plateau, spike, saved_b["postprocess"])
    types = np.full(len(test), "", dtype=object)
    types[plateau & prediction.astype(bool)] = "flatline"
    types[spike & prediction.astype(bool)] = "spike"
    b = build_submission(test, prediction, types)
    router_bits, gi_bits = compose_tree(test.station, test.layer, o.label, b.label,
                                       add_cells=ADD_CELLS, remove_cells=REMOVE_CELLS)
    gi_types = b.anomaly_type.to_numpy().copy()
    additions = (gi_bits == 1) & b.label.eq(0).to_numpy()
    gi_types[additions] = o.anomaly_type.to_numpy()[additions]
    router = build_submission(test, router_bits, gi_types)
    gi = build_submission(test, gi_bits, gi_types)
    checks = {}
    for name, answer in {"O": o, "B": b, "router": router, "GI_no_removals": gi}.items():
        payload = answer.to_csv(index=False, lineterminator="\n").encode("utf-8")
        sha = hashlib.sha256(payload).hexdigest()
        checks[name] = {"sha256": sha, "bytes": len(payload),
                        "expected_sha256": EXPECTED.get(name),
                        "hash_match": sha == EXPECTED[name] if name in EXPECTED else None,
                        "validator": validate_submission(answer, test)}
    match = all(checks[name]["hash_match"] for name in EXPECTED)
    result = {
        "status": "SAVED_O_B_ROUTER_EXACT_REPLAY_PASS" if match else "REPLAY_HASH_MISMATCH",
        "scope": "Diagnostic uses original saved models and cached features, NOT scratch training. "
                 "No final eligibility claim, no full MS-TCN/GI final-answer replay yet.",
        "checks": checks, "pinned_sha256": hashes, "test_sha256": TEST_SHA,
        "script_sha256": digest(Path(__file__)),
        "helper_sha256": digest(root / "scripts/p1_historical_path_audit_20260906_v1.py"),
        "O_postprocess": original.postprocess, "B_postprocess": saved_b["postprocess"],
        "B_seeds": saved_b["seeds"],
        "original_O_receipt": "artifacts/runs/20260813T155254+0900_train_378a4e89/"
                              "model_metadata.json",
        "official_observation_rows_read": len(test), "cached_feature_rows_read": len(frame),
        "hidden_truth_reads": 0, "archived_answer_csv_reads": 0,
        "model_fits": 0, "candidate_csv_writes": 0, "uploads": 0,
        "runtime_seconds": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.test.resolve(), args.output.resolve())
