"""QA-gated separate-process inference; never fits or reads historical answers.

Prepare this code before results. Execution requires a later, explicit root
decision linking completed internal QA and both newly trained components.
"""
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

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4" if len(sys.argv) > 1 and sys.argv[1] == "tree" else "2"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import composition
import numpy as np
import pandas as pd

TEST_SHA = "6d5c6522c282651b99f4261ffa803cf99950596028e996de1e7714db77408387"
SAMPLE_SHA = "e7027bcb56836587715e5cd818c6d595ebef4a8518538fa7de7bdcba2b7fc1e4"
RAW = [*composition.KEYS, "temp", "psal", "depth"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, payload):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def decision(path):
    value = read(path)
    if value.get("official_materialization_authorized") is not True:
        raise PermissionError("root post-QA materialization decision required")
    if value.get("tree_arm") not in ("O", "B", "union", "router"):
        raise ValueError("tree arm must be chosen and documented before official inference")
    receipts = {}
    expected_statuses = {
        "internal_qa": "PASS", "tree_historical": "COMPLETE",
        "tree_training": "COMPLETE", "tree_replay": "PASS", "tree_full_qa": "PASS",
        "mstcn_training": "TRAINING_COMPLETE_REPLAY_PENDING", "mstcn_replay": "PASS", "mstcn_qa": "PASS",
        "union_result": "COMPLETE_RETROSPECTIVE_EVALUATION",
        "union_qa": "PASS", "union_bootstrap_qa": "PASS",
    }
    for name, expected in expected_statuses.items():
        pin = value["evidence"][name]
        if sha(pin["path"]) != pin["sha256"]:
            raise ValueError(f"decision evidence changed: {name}")
        receipt = read(pin["path"])
        receipts[name] = receipt
        if receipt["status"] != expected:
            raise ValueError(f"decision evidence not complete: {name}")
    for model, field in (("tree", "terminal_result_sha256"), ("mstcn", "training_result_sha256")):
        if receipts[f"{model}_replay"][field] != value["evidence"][f"{model}_training"]["sha256"]:
            raise ValueError("replay belongs to another training run")
    historical_sha = value["evidence"]["tree_historical"]["sha256"]
    if receipts["internal_qa"]["terminal_result_sha256"] != historical_sha:
        raise ValueError("internal QA belongs to another historical run")
    if receipts["tree_training"]["historical_result_sha256"] != historical_sha:
        raise ValueError("full tree training belongs to another historical run")
    if receipts["tree_full_qa"]["terminal_result_sha256"] != value["evidence"]["tree_training"]["sha256"]:
        raise ValueError("full tree QA belongs to another training run")
    if (receipts["mstcn_qa"]["training_result_sha256"] != value["evidence"]["mstcn_training"]["sha256"]
            or receipts["mstcn_qa"]["fresh_replay_sha256"] != value["evidence"]["mstcn_replay"]["sha256"]):
        raise ValueError("MS-TCN independent QA belongs to another training or replay")
    if receipts["union_result"]["tree_terminal_sha256"] != historical_sha:
        raise ValueError("union evaluation belongs to another historical run")
    for name in ("union_qa", "union_bootstrap_qa"):
        if receipts[name]["result_sha256"] != value["evidence"]["union_result"]["sha256"]:
            raise ValueError("union QA belongs to another evaluation")
    earliest = min(receipts["tree_historical"]["started_unix"], receipts["mstcn_training"]["started_at"])
    if abs(value["earliest_started_unix"] - earliest) > 1e-6:
        raise ValueError("combined clock is not anchored to actual training starts")
    if time.time() - value["earliest_started_unix"] >= 21600:
        raise TimeoutError("combined source-to-answer six-hour budget exhausted")
    return value


def read_official(data_dir, *, include_sample=False):
    directory = Path(data_dir).resolve()
    path = directory / "test.csv"
    if sha(path) != TEST_SHA:
        raise ValueError("distributed official input SHA mismatch")
    frame = pd.read_csv(path, usecols=RAW)
    if sha(path) != TEST_SHA:
        raise ValueError("official input changed during read")
    if len(frame) != 169011:
        raise ValueError("official population changed")
    composition.canonical_keys(frame)
    if include_sample:
        sample_path = directory / "sample_submission.csv"
        if sha(sample_path) != SAMPLE_SHA:
            raise ValueError("distributed sample SHA mismatch")
        sample = pd.read_csv(sample_path, usecols=list(composition.KEYS))
        if sha(sample_path) != SAMPLE_SHA:
            raise ValueError("official sample changed during read")
        if not sample.equals(frame[list(composition.KEYS)]):
            raise ValueError("official sample/test ordered keys differ")
    return frame


def tree_inference(cfg, data_dir):
    import tree

    output = Path(cfg["tree_model_output"]).resolve()
    seal_path = Path(cfg["tree_seal"]).resolve()
    seal = tree.check_seal(seal_path)
    result = read(output / "terminal_result.json")
    if sha(output / "terminal_result.json") != cfg["evidence"]["tree_training"]["sha256"]:
        raise ValueError("tree model directory differs from root decision")
    if result["phase"] != "full" or result["completed_fits"] != 4:
        raise ValueError("own completed four-fit full model required")
    tree.verify_artifacts(output, result)
    model_dir = output / "full/models"
    fitted = tree.core.joblib.load(model_dir / "preprocess.joblib")
    original = read_official(data_dir)
    frame = original.assign(_order=np.arange(len(original))).sort_values(
        ["station", "layer", "time"], kind="stable").reset_index(drop=True)
    bundle = tree.base_features(frame, fitted["stats"], seal["config"])
    x = fitted["encoder"].transform(bundle)
    probabilities = {"O": [], "B": []}
    for filename in fitted["model_files"]:
        model = tree.core.joblib.load(model_dir / filename)
        probabilities[filename[0]].append(np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64))
        del model
    if [len(probabilities[a]) for a in ("O", "B")] != [1, 3]:
        raise ValueError("O1/B3 inference ensemble drift")
    probabilities = {arm: np.mean(values, axis=0) for arm, values in probabilities.items()}
    if any(not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all() for p in probabilities.values()):
        raise ValueError("invalid tree probability")
    rules = tree.core.rule_masks(frame, fitted["stats"])
    arms = tree.predict_policy(frame, probabilities, rules, seal["config"], read(output / "selector.json"))
    restored = np.empty(len(frame), dtype=np.int8)
    restored[frame._order.to_numpy()] = arms[cfg["tree_arm"]]
    return original[list(composition.KEYS)].assign(tree=restored), {
        "tree_arm": cfg["tree_arm"], "fits": 0, "loaded_new_models": 4,
        "official_rows_read": len(original), "hidden_rows_read": 0,
        "selector_sha256": sha(output / "selector.json"),
    }


def mstcn_inference(cfg, data_dir):
    import mstcn

    output = Path(cfg["mstcn_model_output"]).resolve()
    if sha(output / "training-result.json") != cfg["evidence"]["mstcn_training"]["sha256"]:
        raise ValueError("MS-TCN model directory differs from root decision")
    # Verify state/probe provenance before opening official observations.
    mstcn.verify_owned(output)
    mstcn.configure_owned_runtime(output)
    frame = read_official(data_dir)
    keys, bits, receipt = mstcn.predict_proposal(frame, model_dir=output / "03_model")
    receipt["official_rows_read_by_caller"] = len(frame)
    receipt["hidden_rows_read"] = 0
    return keys.assign(proposal=bits), receipt


def merge_stage(cfg, data_dir, stage_dir, destination):
    directory = Path(stage_dir).resolve()
    frames = []
    for stage in ("tree", "mstcn"):
        receipt = read(directory / stage / f"{stage}.json")
        if receipt["decision_sha256"] != cfg["decision_sha256"]:
            raise ValueError("component belongs to a different root decision")
        path = directory / stage / f"{stage}.parquet"
        if sha(path) != receipt["predictions_sha256"]:
            raise ValueError("component prediction SHA mismatch")
        frames.append(pd.read_parquet(path))
    result, receipt = composition.combine(*frames)
    original = read_official(data_dir, include_sample=True)
    sys.path.insert(0, str(ROOT / "src"))
    from p1_qc.submission import validate_submission

    before = validate_submission(result, original)
    path = destination / "P1_submission.csv"
    with path.open("x", encoding="utf-8", newline="") as handle:
        result.to_csv(handle, index=False, lineterminator="\n")
    after = validate_submission(path, original)
    return {"status": "PASS", "decision_sha256": cfg["decision_sha256"],
            "composition": receipt, "validator_before": before,
            "validator_after": after, "csv_sha256": sha(path), "csv_written": 1,
            "official_rows_read": len(original), "sample_key_rows_read": len(original),
            "hidden_rows_read": 0, "uploads": 0,
            "source_to_answer_seconds": time.time() - cfg["earliest_started_unix"]}


def execute(stage, decision_path, data_dir, destination, stage_dir=None):
    cfg = decision(decision_path)  # No official read before evidence gates.
    cfg["decision_sha256"] = sha(decision_path)
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("use a new stage output; no overwrites")
    destination.mkdir(parents=True)
    write_new(destination / "ATTEMPT_LOCK.json", {"pid": os.getpid(), "stage": stage,
              "decision_sha256": cfg["decision_sha256"], "time": time.time()})
    remaining = 21600 - (time.time() - cfg["earliest_started_unix"])

    def expire():
        write_new(destination / "terminal.json", {"status": "RESOURCE_STOP", "stage": stage})
        os._exit(124)

    timer = threading.Timer(max(0, remaining), expire)
    timer.daemon = True
    timer.start()
    try:
        if stage == "combine":
            receipt = merge_stage(cfg, data_dir, stage_dir, destination)
        else:
            frame, receipt = (tree_inference if stage == "tree" else mstcn_inference)(cfg, data_dir)
            path = destination / f"{stage}.parquet"
            frame.to_parquet(path, index=False)
            receipt.update(status="PASS", decision_sha256=cfg["decision_sha256"],
                           predictions_sha256=sha(path), rows=len(frame), pid=os.getpid(),
                           ordered_key_sha256=composition.ordered_key_sha256(frame))
            write_new(destination / f"{stage}.json", receipt)
        if time.time() - cfg["earliest_started_unix"] > 21600:
            raise TimeoutError("six-hour source-to-answer budget exhausted")
        write_new(destination / "terminal.json", receipt)
    except BaseException as error:
        if not (destination / "terminal.json").exists():
            write_new(destination / "terminal.json", {"status": "TECHNICAL_FAILURE",
                      "error_type": type(error).__name__, "error": str(error), "stage": stage})
        raise
    finally:
        timer.cancel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("tree", "mstcn", "combine"))
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path)
    args = parser.parse_args()
    if args.stage == "combine" and args.stage_dir is None:
        parser.error("combine needs --stage-dir containing tree/ and mstcn/ stages")
    execute(args.stage, args.decision, args.data_dir, args.output, args.stage_dir)
