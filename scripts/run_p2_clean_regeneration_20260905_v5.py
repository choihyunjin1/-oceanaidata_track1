"""Authorized guard-only repair; v4 evidence and all frozen recipes preserved."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_p2_clean_regeneration_20260905_v4 as original  # noqa: E402

ID = "p2_clean_regeneration_20260905_v5"
original.ID = ID
original.CONFIG = ROOT / "configs/experiments" / f"{ID}.json"
original.OUT = ROOT / "artifacts" / ID
original.REPORT = ROOT / "reports" / ID
original.MODELS = original.OUT / "03_model"
original.ANSWERS = original.OUT / "05_answer"
original.SEAL = original.REPORT / "preregistration-seal.json"
CREATED_MODELS = set()
old_path_allowed = original.path_allowed
old_install_guard = original.install_guard
old_fingerprint = original.fingerprints


def path_allowed(path, mode, writing, source):
    path = Path(path).resolve()
    if path.suffix.lower() == ".pt" and mode == "RUN_TRAINING":
        if original.MODELS not in path.parents:
            return False
        if writing:
            CREATED_MODELS.add(path)
            return True
        return path in CREATED_MODELS
    return old_path_allowed(path, mode, writing, source)


def deny_training_model_load(*args, **kwargs):
    raise PermissionError("torch.load is prohibited during scratch training, including self-generated weights")


def install_guard(mode, source):
    CREATED_MODELS.clear()
    if mode == "RUN_TRAINING":
        original.torch.load = deny_training_model_load
    return old_install_guard(mode, source)


def fingerprints():
    value = old_fingerprint()
    value["technical_repair_driver_sha256"] = original.research.file_hash(Path(__file__))
    value["frozen_v4_driver_sha256"] = original.research.file_hash(ROOT / "scripts/run_p2_clean_regeneration_20260905_v4.py")
    return value


original.path_allowed = path_allowed
original.install_guard = install_guard
original.fingerprints = fingerprints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("seal", "RUN_TRAINING", "RUN_INFERENCE"))
    args = parser.parse_args()
    original.lineage_source_contract()
    original.torch.set_num_threads(2)
    try:
        with original.threadpool_limits(limits=2):
            if args.mode == "seal":
                original.save(original.SEAL, {"experiment_id": ID, "hashes": fingerprints(), "maximum_new_full_fits": 3, "technical_change_only": True, "old_attempt_reused": False})
                print("SEALED")
            elif args.mode == "RUN_TRAINING":
                original.run_training()
            else:
                original.run_inference()
    except Exception as exc:
        original.save(original.REPORT / f"{args.mode}-failure.json", {"experiment_id": ID, "status": "TERMINAL_TECHNICAL_FAILURE", "exception": type(exc).__name__, "message": str(exc), "old_attempt_reused": False, "automatic_restart": False})
        raise
    if args.mode != "seal":
        print(json.dumps({"experiment_id": ID, "stage": args.mode, "complete": True}))


if __name__ == "__main__":
    main()
