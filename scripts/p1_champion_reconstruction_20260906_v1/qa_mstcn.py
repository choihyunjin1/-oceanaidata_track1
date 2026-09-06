"""Independent post-terminal MS-TCN QA: CPU metadata only, no model inference/fit.

Readiness gates precede any binary read. This does not import the producer's
validation functions, construct a network, open a raw CSV or load an old model.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path, PurePosixPath

SEEDS = (20260827, 20260839, 20260863)
PARAMETERS = 52568587
ID = "p1_champion_reconstruction_20260906_v1_mstcn"
TRAIN_SHA = "20b656b0cbd524ad9da0bae8ecb6e0bacfc006e05810b37e83f29a5fa8e65cd2"
RUNNER_SHA = "d46b8a63d12459ef565edb03eb9bc8fe38d745624f192ca502c775a29743bbf8"
SOURCE_PINS = {
    "scripts/run_p1_incumbent_preserving_mstcn_asrf_v2.py": "78df1bd3b4777560b0134bc69ad45839c8c0093da39d2c1d3d458b3a3ba9b87a",
    "configs/experiments/p1_incumbent_preserving_mstcn_asrf_v2.json": "1f8940d29ea6b047273e4f53445f62230e7d72bf1f0b14abe9fb18476f0345f0",
    "src/p1_qc/ms_tcn_asrf.py": "57c135bfe746d06a53e5c3b83517cb96f8262a765febefbf43e1d3a3c344fd7f",
    "src/p1_qc/ms_tcn_asrf_data.py": "cf5dc2dbbb3ecf05c489b661f5427ff225caaf28af5fb44d292e3289c7bb9adf",
    "src/p1_qc/features.py": "e769dbe187eeaa5061047a634bb2bcac6ea9a2ad9a359d48d98f49ba4b4b7162",
    "src/p1_qc/config.py": "adf2e04123b95e51bcc7989e2c80b9faee776cb32fe63179b984a48cba2651f1",
    "src/p1_qc/data.py": "5dc1dac588faa2f50323d15ab2d83231eb06c3a709845d539ba7efbb99addd69",
    "src/p1_qc/__init__.py": "5a7743ab77b2f9bd6851f12ebe694313c4fb361611821ed68dfd3b7574a82088",
    "configs/p1.toml": "d04c8a5759645f11bcf0d987fff44829e47f4092f310a48704ef644ac6bc3e80",
}
EXPECTED = {
    "id": ID, "seeds": list(SEEDS), "epochs": 150, "schedule_horizon_epochs": 300,
    "warmup_epochs": 10, "width": 512, "input_features": 165, "parameters": PARAMETERS,
    "batch_size": 64, "accumulation": 1, "cpu_threads": 2, "device": "cuda:0",
    "precision": "original CUDA bfloat16 autocast", "learning_rate": .0003,
    "weight_decay": .0001, "window_rows": 2048, "stride_rows": 512,
    "threshold": .8, "low_threshold": .4, "snap_radius": 12,
    "minimum_added_rows": 19, "maximum_added_rows": None,
    "raw_three_seed_mean_before_type_conditioning": True,
    "full_fit_budget": 3, "end_to_end_cap_seconds": 21600,
    "historical_fit_seconds_excluding_prepare_qa": 6584.228795399889,
    "scratch_byte_determinism": "UNVERIFIED", "official_io": 0,
    "old_cache_or_model_reads": 0, "anchor_csv_patch_or_probability_input_reads": 0,
    "proposal_only_not_final_union": True,
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def owned_path(output, relative):
    relative = PurePosixPath(relative)
    if relative.is_absolute() or ".." in relative.parts or ":" in str(relative) or "\\" in str(relative):
        raise ValueError("Non-owned path in file manifest")
    result = (Path(output).resolve() / str(relative)).resolve()
    if not result.is_relative_to(Path(output).resolve()):
        raise ValueError("Manifest path escapes output")
    return result


def expected_files():
    return {f"02_code/{p}" for p in SOURCE_PINS} | {
        "02_code/mstcn.py", "03_model/encoder.json", "04_validation/own_train_probe.npz",
        *(f"03_model/seed_{s}.pt" for s in SEEDS),
        *(f"04_validation/probe_expected_{s}.npz" for s in SEEDS),
        *(f"04_validation/fit_{s}.json" for s in SEEDS),
    }


def expected_tensor_shapes():
    """Independent explicit Conv1d topology inventory; creates no model/tensor."""
    shapes = {}

    def conv(prefix, outputs, inputs, kernel):
        shapes[prefix + ".weight"] = (outputs, inputs, kernel)
        shapes[prefix + ".bias"] = (outputs,)

    conv("prediction_generator.stem", 512, 165, 1)
    for index in range(10):
        for side in ("left", "right"):
            conv(f"prediction_generator.layers.{index}.{side}", 512, 512, 3)
        conv(f"prediction_generator.layers.{index}.fuse", 512, 1024, 1)
    conv("prediction_generator.row_head", 1, 512, 1)
    for stage in range(3):
        prefix = f"refinement_stages.{stage}"
        conv(prefix + ".stem", 512, 1, 1)
        for index in range(10):
            conv(f"{prefix}.layers.{index}.dilated", 512, 512, 3)
            conv(f"{prefix}.layers.{index}.project", 512, 512, 1)
        conv(prefix + ".row_head", 1, 512, 1)
    conv("boundary_head", 2, 512, 1)
    conv("type_head", 5, 512, 1)
    if sum(math.prod(shape) for shape in shapes.values()) != PARAMETERS:
        raise AssertionError("Independent architecture arithmetic wrong")
    return shapes


def gate(output):
    """Only aggregate JSON is read here; no binary IO until every gate passes."""
    output = Path(output)
    training = read(output / "training-result.json")
    replay = read(output / "fresh-process-replay.json")
    terminal = read(output / "terminal.json")
    if training.get("status") != "TRAINING_COMPLETE_REPLAY_PENDING" or training.get("completed_fits") != 3:
        raise RuntimeError("Three-fit training completion required before QA")
    if replay.get("status") != "PASS" or terminal.get("status") != "PASS":
        raise RuntimeError("Successful fresh-process replay required before QA")
    if terminal != replay:
        raise RuntimeError("Terminal and replay receipts disagree")
    if replay.get("training_result_sha256") != sha(output / "training-result.json"):
        raise RuntimeError("Replay is not linked to this training receipt")
    if replay.get("pid") == training.get("training_pid") or replay.get("training_pid") != training.get("training_pid"):
        raise RuntimeError("Replay/training process lineage invalid")
    if os.getpid() in (replay.get("pid"), training.get("training_pid")):
        raise RuntimeError("Independent QA must use another PID")
    required_checks = {f"{seed}_{head}_exact" for seed in SEEDS for head in ("row", "boundary", "type")}
    required_checks |= {"fresh_pid", "within_original_6h"}
    checks = replay.get("checks", {})
    if not required_checks.issubset(checks) or not all(value is True for value in checks.values()):
        raise RuntimeError("Required replay checks absent or failed")
    if training.get("contract") != EXPECTED or training.get("source_pins") != SOURCE_PINS:
        raise RuntimeError("Training recipe/source contract differs from independently fixed QA")
    if set(training.get("files_sha256", {})) != expected_files():
        raise RuntimeError("Exact own-code/model/encoder/probe manifest required")
    return training, replay


def state_metadata(path, seed):
    """One checkpoint at a time, weights_only CPU load; inspect shapes not values."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    if set(state) != {"seed", "epoch", "input_features", "state_dict"}:
        raise ValueError("Unexpected state envelope")
    if (state["seed"], state["epoch"], state["input_features"]) != (seed, 150, 165):
        raise ValueError("State seed/epoch/input width invalid")
    tensors = state["state_dict"]
    shapes = expected_tensor_shapes()
    if set(tensors) != set(shapes):
        raise ValueError("State tensor inventory differs from fixed architecture")
    metadata = {}
    for name, shape in shapes.items():
        tensor = tensors[name]
        if not torch.is_tensor(tensor) or tensor.device.type != "cpu" or tensor.dtype != torch.float32:
            raise ValueError("Expected CPU float32 state tensor")
        if tuple(tensor.shape) != shape:
            raise ValueError("State tensor shape differs from fixed architecture")
        metadata[name] = {"shape": list(shape), "dtype": str(tensor.dtype), "numel": tensor.numel()}
    count = sum(value["numel"] for value in metadata.values())
    if count != PARAMETERS:
        raise ValueError("State parameter total differs from fixed architecture")
    receipt = {"seed": seed, "parameter_count": count, "tensor_count": len(metadata),
               "shape_dtype_manifest_sha256": hashlib.sha256(
                   json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
               "map_location": "cpu", "weights_only": True, "new_forward_calls": 0,
               "tensor_value_statistics_computed": False}
    del tensors, state
    gc.collect()
    return receipt


def verify(output):
    output = Path(output).resolve()
    training, replay = gate(output)
    started = time.monotonic()
    checks = {}

    def check(name, passed):
        checks[name] = bool(passed)

    hashes = training["files_sha256"]
    # Verify every allowlisted own file before any model deserialization.
    for relative, expected in hashes.items():
        if not is_sha(expected):
            raise ValueError("Invalid SHA in own-file manifest")
        observed = sha(owned_path(output, relative))
        check(f"file_sha:{relative}", observed == expected)
    for relative, expected in SOURCE_PINS.items():
        check(f"original_source_pin:{relative}", hashes[f"02_code/{relative}"] == expected)
    check("executed_frozen_runner_pin", hashes["02_code/mstcn.py"] == RUNNER_SHA)
    if not all(checks.values()):
        raise ValueError("Own source/model/probe hashes failed before model loading")

    import numpy as np
    import torch

    torch.set_num_threads(2)
    prepare = read(output / "prepare.json")
    encoder = read(output / "03_model/encoder.json")
    lock = read(output / "ATTEMPT_LOCK.json")
    replay_lock = read(output / "REPLAY_ATTEMPT_LOCK.json")
    check("train_source_hash", prepare["train_sha256"] == TRAIN_SHA)
    check("prepare_rows_and_tensor_shape", prepare["rows"] == 776706 and prepare["input_shape"] == [776706, 165])
    check("prepare_feature_and_key_hash_syntax", is_sha(prepare["feature_sha256"]) and is_sha(prepare["ordered_key_sha256"]))
    check("raw_label_columns_not_numeric_features", not {"label", "anomaly_type", "year"}.intersection(prepare["numeric_names"]))
    check("numeric_names_exact_unique_74", len(prepare["numeric_names"]) == len(set(prepare["numeric_names"])) == 74)
    check("encoder_numeric_names_match_prepare", encoder["numeric_names"] == prepare["numeric_names"])
    check("encoder_center_scale_74", len(encoder["center"]) == len(encoder["scale"]) == 74)
    check("encoder_center_finite", bool(np.isfinite(encoder["center"]).all()))
    scales = np.asarray(encoder["scale"])
    check("encoder_scale_positive_finite", bool(np.isfinite(scales).all() and (scales > 0).all()))
    check("station_vocab", encoder["station_vocab"] == ["G-ORS", "I-ORS", "S-ORS"])
    check("layer_vocab", encoder["layer_vocab"] == [str(i) for i in range(1, 9)])
    check("depth_vocab", encoder["depth_regime_vocab"] == ["deep", "mid", "missing", "shallow"])
    check("derived_input_features_165", 2 * len(encoder["center"]) + 2 + sum(
        len(encoder[key]) for key in ("station_vocab", "layer_vocab", "depth_regime_vocab")) == 165)
    thresholds = encoder["depth_thresholds"]
    check("depth_thresholds_finite_ordered", len(thresholds) == 2 and np.isfinite(thresholds).all() and thresholds[0] <= thresholds[1])
    check("current_depth_not_supplied_fullrun_regime", encoder["uses_supplied_depth_regime"] is False)
    check("preprocessor_holdout_rows_zero", encoder["preprocessing_fit_uses_holdout_rows"] is False)
    # Original fit-ID hash is little-endian int64 bytes, independently recomputed.
    expected_ids = hashlib.sha256(np.arange(776706, dtype=np.int64).tobytes()).hexdigest()
    check("encoder_full_train_fit_ids_hash", encoder["fit_ids_sha256"] == expected_ids)
    dependency = prepare["dependency"]
    check("safe_projection_width", dependency["cache_feature_count"] == 80 and dependency["model_numeric_feature_count"] == 74)
    check("no_unbounded_projection", dependency["unbounded_features_projected"] == 0 and dependency["all_cache_features_classified"] is True)
    check("explicit_unbounded_exclusions", set(dependency["excluded_unbounded"]) == {
        "nominal_depth_m", "plateau_full_length", "plateau_count", "depth_regime"})
    check("bounded_not_past_only", dependency["bounded_future_support_hours_max"] == 168)

    metadata = []
    fit_seconds = []
    check("fit_receipts_exact_three_seed_order", [fit["seed"] for fit in training["fit_receipts"]] == list(SEEDS))
    for seed, fit in zip(SEEDS, training["fit_receipts"], strict=True):
        saved_fit = read(output / f"04_validation/fit_{seed}.json")
        check(f"fit_{seed}_standalone_matches", fit == saved_fit)
        for key, expected in {"seed": seed, "epoch": 150, "optimizer_steps": 4050,
                              "windows": 1707, "nonfinite_count": 0}.items():
            check(f"fit_{seed}_{key}", fit[key] == expected)
        check(f"fit_{seed}_model_link", fit["model_sha256"] == hashes[f"03_model/seed_{seed}.pt"])
        check(f"fit_{seed}_wall_positive", math.isfinite(fit["wall_seconds"]) and fit["wall_seconds"] > 0)
        fit_seconds.append(fit["wall_seconds"])
        metadata.append(state_metadata(output / f"03_model/seed_{seed}.pt", seed))
        check(f"state_{seed}_parameters", metadata[-1]["parameter_count"] == PARAMETERS)

    with np.load(output / "04_validation/own_train_probe.npz", allow_pickle=False) as probe:
        check("probe_inventory", set(probe.files) == {"values", "valid"})
        values, valid = probe["values"], probe["valid"]
        check("probe_shape", values.shape == (2, 2048, 165) and valid.shape == (2, 2048))
        check("probe_dtype_finite", values.dtype == np.float32 and valid.dtype == np.float32 and np.isfinite(values).all())
        check("probe_valid_binary", bool(np.isin(valid, [0, 1]).all()))
        check("probe_zero_padding", bool(np.all(values[~valid.astype(bool)] == 0)))
    for seed in SEEDS:
        with np.load(output / f"04_validation/probe_expected_{seed}.npz", allow_pickle=False) as probe:
            check(f"probe_{seed}_head_inventory", set(probe.files) == {"row", "boundary", "type"})
            for head, shape in {"row": (2, 2048), "boundary": (2, 2048, 2), "type": (2, 2048, 5)}.items():
                values = probe[head]
                check(f"probe_{seed}_{head}_shape", values.shape == shape)
                check(f"probe_{seed}_{head}_finite_probability", values.dtype == np.float32 and
                      np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all())

    check("lock_training_pid", lock["pid"] == training["training_pid"])
    check("lock_started_at", lock["started_at"] == training["started_at"])
    check("lock_contract", lock["contract"] == EXPECTED)
    check("replay_lock_pid", replay_lock["pid"] == replay["pid"])
    check("original_clock_positive", math.isfinite(training["started_at"]) and training["started_at"] > 0)
    summed_seconds = math.fsum(fit_seconds)
    check("fit_sum_fits_inside_training_wall", 0 < summed_seconds <= training["wall_seconds"] + 1e-6)
    check("prepare_plus_fits_inside_training_wall", 0 <= prepare["wall_seconds"] + summed_seconds <= training["wall_seconds"] + 1e-6)
    check("recorded_training_under_six_hours", 0 < training["wall_seconds"] <= 21600)
    check("recorded_replay_after_training_under_six_hours", training["wall_seconds"] <= replay["source_to_replay_seconds"] <= 21600)
    check("independent_qa_inside_original_six_hours", 0 < time.time() - training["started_at"] <= 21600)
    for key in ("official_access_rows", "old_models_cache_answer_reads"):
        check(f"training_{key}_zero", training[key] == 0)
    check("prepare_official_and_old_cache_zero", prepare["official_access_rows"] == prepare["old_cache_reads"] == 0)
    check("replay_official_zero", replay["official_access_rows"] == 0)
    check("scratch_determinism_not_falsely_claimed", replay["scratch_retraining_determinism"] == "NOT_TESTED")
    for key, value in training["runtime"].items():
        if key != "pid":
            check(f"runtime_replay_{key}", replay["runtime"][key] == value)
            check(f"runtime_recorded_check_{key}", replay["checks"].get(f"runtime_{key}") is True)
    check("runtime_threads_two", training["runtime"]["cpu_threads"] == 2)
    check("runtime_bfloat16", training["runtime"]["precision"] == "bfloat16")
    # Post-read hashes detect mutations while metadata QA ran; never read external paths.
    for relative, expected in hashes.items():
        check(f"post_read_sha:{relative}", sha(owned_path(output, relative)) == expected)
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": "p1.champion.mstcn.independent_qa.v1",
        "status": "PASS" if not failed else "FAIL", "check_count": len(checks),
        "failed_check_count": len(failed), "failed_checks": failed, "checks": checks,
        "training_result_sha256": sha(output / "training-result.json"),
        "fresh_process_replay_sha256": sha(output / "fresh-process-replay.json"),
        "fresh_replay_sha256": sha(output / "fresh-process-replay.json"),
        "terminal_sha256": sha(output / "terminal.json"), "prepare_sha256": sha(output / "prepare.json"),
        "qa_source_sha256": sha(__file__), "pid": os.getpid(),
        "training_pid": training["training_pid"], "replay_pid": replay["pid"],
        "model_metadata": metadata, "fit_wall_seconds_sum": summed_seconds,
        "training_wall_seconds": training["wall_seconds"],
        "source_to_qa_seconds": time.time() - training["started_at"],
        "qa_wall_seconds": time.monotonic() - started,
        "new_model_fits": 0, "new_forward_calls": 0, "gpu_calls": 0,
        "raw_csv_rows_read": 0, "official_rows_read": 0, "hidden_rows_read": 0,
        "old_model_or_answer_reads": 0, "csv_written": 0, "uploads": 0,
        "scope": "state metadata and saved-probe QA; not another training or final-answer replay",
        "zero_access_evidence_limit": "producer receipts and pinned code boundary, not OS-level historical access tracing",
        "root_combined_tree_mstcn_final_answer_6h_still_required": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Independent QA report is append-only; no automatic retries")
    report = verify(args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps({"status": report["status"], "checks": report["check_count"],
                      "failed": report["failed_check_count"], "report_sha256": sha(args.report)}))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
