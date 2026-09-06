"""Clock-independent saved-model replay; no training or stored answer input."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
PACKAGE = CODE.parent
sys.path.insert(0, str(CODE))
import run as source


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_contract():
    manifest = json.loads((PACKAGE / "portable-manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        path = (PACKAGE / relative).resolve()
        if not path.is_relative_to(PACKAGE) or sha(path) != expected:
            raise ValueError("archive source/model/receipt changed: " + relative)
    recipe = json.loads((PACKAGE / "03_model/frozen_recipe.json").read_text(encoding="utf-8"))
    if sha(PACKAGE / "03_model/frozen_recipe.json") != manifest["contract"]["source_recipe_sha256"]:
        raise ValueError("source recipe mismatch")
    # This adapter does not call or patch the historical workflow timer/lock.
    source.assert_sources(recipe["source_hashes"])
    for arm, digest in recipe["model_hashes"].items():
        if sha(PACKAGE / "03_model" / ("full_" + arm + ".joblib")) != digest:
            raise ValueError("full model mismatch")
    return manifest, recipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-approved", action="store_true", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source.install_boundary_guard("infer")
    manifest, recipe = checked_contract()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError("new replay output root required; existing artifacts are immutable")
    output.mkdir(parents=True)
    for name in ("05_answer", "06_docs"):
        (output / name).mkdir()
    started = time.monotonic()
    timer = threading.Timer(manifest["contract"]["saved_replay_max_seconds"], lambda: os._exit(124))
    timer.daemon = True
    timer.start()
    cfg, frozen = recipe["candidate_contract"], recipe["feature_contract"]
    data = Path(os.environ["P1_DATA_DIR"])
    frame = source.pd.read_csv(data / "test.csv", usecols=source.screen.RAW)
    keys = source.pd.read_csv(data / "sample_submission.csv", usecols=source.screen.KEYS)
    if len(frame) != 169011 or len(keys) != 169011 or not frame[source.screen.KEYS].equals(keys):
        raise ValueError("official schema/order population mismatch")
    frame.sort_values(["station", "layer", "time"], kind="stable", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    probabilities, stats = {}, None
    for arm in ("original", "balanced"):
        package = source.joblib.load(PACKAGE / "03_model" / ("full_" + arm + ".joblib"))
        model = package["model"]
        backend = model.model if arm == "original" else model
        backend.set_params(n_jobs=manifest["contract"]["inference_threads"])
        stats = package["train_stats"]
        bundle = source.arm_bundle(frame, stats, frozen, cfg, arm)
        probabilities[arm] = source.checked_probability(
            model.predict_proba(package["encoder"].transform(bundle))[:, 1], len(frame))
    bits, _, _ = source.canonical.decoder.control_components(frame, probabilities,
        source.screen.rule_masks(frame, stats), frozen, recipe["selection"], recipe["calibrations"], 1e-6)
    answer = source.align_answer(frame, bits, keys)
    payload = answer.to_csv(index=False, lineterminator="\n").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    target = output / "05_answer/P1_submission.csv"
    with target.open("xb") as stream:
        stream.write(payload)
    equal = digest == manifest["contract"]["expected_answer_sha256"]
    receipt = {"status": "SAVED_MODEL_EXACT_REPLAY_PASS" if equal else "ANSWER_MISMATCH",
               "pid": os.getpid(), "sha256": digest, "expected_sha256": manifest["contract"]["expected_answer_sha256"],
               "rows": len(answer), "positive_rows": int(answer.label.sum()),
               "new_fits": 0, "threads": 2, "gpu": False,
               "source_manifest_sha256": sha(PACKAGE / "portable-manifest.json"),
               "source_recipe_sha256": sha(PACKAGE / "03_model/frozen_recipe.json"),
               "official_test_rows": len(frame), "sample_key_rows": len(keys),
               "sample_prediction_values_read": 0, "hidden_rows": 0, "prior_answer_values_read": 0,
               "uploads": 0, "schema_key_order_unique_binary_finite": True,
               "seconds": time.monotonic() - started,
               "scope": "fresh process from extracted saved-model ZIP, not new training"}
    source.write(output / "06_docs/saved-replay-qa.json", receipt)
    timer.cancel()
    print(json.dumps(receipt), flush=True)
    if not equal:
        raise ValueError("answer mismatch; preserved, no retuning or repeat")


if __name__ == "__main__":
    main()
