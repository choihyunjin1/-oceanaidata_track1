"""Native-writer-aware empty-output checksum guard; immutable C3 recipe."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_p2_clean_regeneration_20260905_v4 as original  # noqa: E402

ID = "p2_clean_regeneration_20260905_v6"
original.ID = ID
original.CONFIG = ROOT / "configs/experiments" / f"{ID}.json"
original.OUT = ROOT / "artifacts" / ID
original.REPORT = ROOT / "reports" / ID
original.MODELS = original.OUT / "03_model"
original.ANSWERS = original.OUT / "05_answer"
original.SEAL = original.REPORT / "preregistration-seal.json"
old_path_allowed = original.path_allowed
old_install_guard = original.install_guard
old_fingerprint = original.fingerprints


def path_allowed(path, mode, writing, source):
    path = Path(path).resolve()
    if path.suffix.lower() == ".pt" and mode == "RUN_TRAINING":
        # RUN_TRAINING proves this new directory is empty before any fit. Native
        # PyTorch writes can bypass Python open hooks, so no write-event registry.
        # Reads here are checksum reads; model loading is separately prohibited.
        return original.MODELS in path.parents
    return old_path_allowed(path, mode, writing, source)


def deny_training_model_load(*args, **kwargs):
    raise PermissionError("torch.load forbidden during scratch training")


def install_guard(mode, source):
    if mode == "RUN_TRAINING":
        original.torch.load = deny_training_model_load
    return old_install_guard(mode, source)


def fingerprints():
    result = old_fingerprint()
    result["technical_repair_driver_sha256"] = original.research.file_hash(Path(__file__))
    result["frozen_v4_driver_sha256"] = original.research.file_hash(ROOT / "scripts/run_p2_clean_regeneration_20260905_v4.py")
    return result


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
                original.save(original.SEAL, {"experiment_id": ID, "hashes": fingerprints(), "maximum_new_full_fits": 3, "native_writer_subprocess_synthetic_required": True, "old_attempt_reused": False})
                print("SEALED")
            elif args.mode == "RUN_TRAINING":
                original.run_training()
            else:
                original.run_inference()
    except Exception as exc:
        original.save(original.REPORT / f"{args.mode}-failure.json", {"experiment_id": ID, "status": "TERMINAL_TECHNICAL_FAILURE", "exception": type(exc).__name__, "message": str(exc), "automatic_restart": False})
        raise
    if args.mode != "seal":
        print(json.dumps({"experiment_id": ID, "stage": args.mode, "complete": True}))


if __name__ == "__main__":
    main()
