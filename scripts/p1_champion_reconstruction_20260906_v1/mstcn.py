"""Source-only e150 MS-TCN reconstruction; no old cache, anchor or answer input.

Implementation is prepared, not training authorization. CLI training requires the
two explicit approval flags. predict_frame consumes an already-authorized frame
and returns only the MS-TCN proposal; the independently regenerated tree union is
the caller's responsibility. This is NOT the historical union's score claim.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

ID = "p1_champion_reconstruction_20260906_v1_mstcn"
SEEDS = (20260827, 20260839, 20260863)
BASE = ("station", "year", "layer", "time", "temp", "psal", "depth")
KEYS = BASE[:4]
TRAIN_SHA = "20b656b0cbd524ad9da0bae8ecb6e0bacfc006e05810b37e83f29a5fa8e65cd2"
SOURCE_RUNNER = "scripts/run_p1_incumbent_preserving_mstcn_asrf_v2.py"
SOURCE_CONFIG = "configs/experiments/p1_incumbent_preserving_mstcn_asrf_v2.json"
PINS = {
    SOURCE_RUNNER: "78df1bd3b4777560b0134bc69ad45839c8c0093da39d2c1d3d458b3a3ba9b87a",
    SOURCE_CONFIG: "1f8940d29ea6b047273e4f53445f62230e7d72bf1f0b14abe9fb18476f0345f0",
    "src/p1_qc/ms_tcn_asrf.py": "57c135bfe746d06a53e5c3b83517cb96f8262a765febefbf43e1d3a3c344fd7f",
    "src/p1_qc/ms_tcn_asrf_data.py": "cf5dc2dbbb3ecf05c489b661f5427ff225caaf28af5fb44d292e3289c7bb9adf",
    "src/p1_qc/features.py": "e769dbe187eeaa5061047a634bb2bcac6ea9a2ad9a359d48d98f49ba4b4b7162",
    "src/p1_qc/config.py": "adf2e04123b95e51bcc7989e2c80b9faee776cb32fe63179b984a48cba2651f1",
    "src/p1_qc/data.py": "5dc1dac588faa2f50323d15ab2d83231eb06c3a709845d539ba7efbb99addd69",
    "src/p1_qc/__init__.py": "5a7743ab77b2f9bd6851f12ebe694313c4fb361611821ed68dfd3b7574a82088",
    "configs/p1.toml": "d04c8a5759645f11bcf0d987fff44829e47f4092f310a48704ef644ac6bc3e80",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, payload, *, progress=False):
    path = Path(path)
    if progress:
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    else:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)


def contract():
    return {
        "id": ID, "seeds": list(SEEDS), "epochs": 150,
        "schedule_horizon_epochs": 300, "warmup_epochs": 10,
        "width": 512, "input_features": 165, "parameters": 52568587,
        "batch_size": 64, "accumulation": 1, "cpu_threads": 2,
        "device": "cuda:0", "precision": "original CUDA bfloat16 autocast",
        "learning_rate": 0.0003, "weight_decay": 0.0001,
        "window_rows": 2048, "stride_rows": 512,
        "threshold": 0.8, "low_threshold": 0.4, "snap_radius": 12,
        "minimum_added_rows": 19, "maximum_added_rows": None,
        "raw_three_seed_mean_before_type_conditioning": True,
        "full_fit_budget": 3, "end_to_end_cap_seconds": 21600,
        "historical_fit_seconds_excluding_prepare_qa": 6584.228795399889,
        "scratch_byte_determinism": "UNVERIFIED",
        "official_io": 0, "old_cache_or_model_reads": 0,
        "anchor_csv_patch_or_probability_input_reads": 0,
        "proposal_only_not_final_union": True,
    }


def verify_sources(root):
    root = Path(root).resolve()
    for relative, expected in PINS.items():
        if sha(root / relative) != expected:
            raise ValueError(f"Source SHA mismatch: {relative}")
    return dict(PINS)


def load_source(root):
    root = Path(root).resolve()
    verify_sources(root)
    # Prevent mixing this exact source snapshot with an earlier imported package.
    for name, module in tuple(sys.modules.items()):
        if name == "p1_qc" or name.startswith("p1_qc."):
            location = getattr(module, "__file__", None)
            if location and not Path(location).resolve().is_relative_to(root / "src"):
                raise RuntimeError("Foreign p1_qc module already loaded; use a fresh process")
    spec = importlib.util.spec_from_file_location("_p1_mstcn_frozen_source", root / SOURCE_RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.ROOT = root
    module._load_scientific()
    config = read_json(root / SOURCE_CONFIG)
    capacity = module._config_for_capacity(config, width=512, seed=SEEDS[0])
    assert capacity["training"]["maximum_epochs"] == 300
    assert capacity["training"]["batch_size"] == 64
    assert capacity["training"]["gradient_accumulation_steps"] == 1
    return module, config


def fresh_features(frame, root, source, config):
    """Only seven covariates reach the original feature builder, never labels."""
    import numpy as np

    from p1_qc.config import load_config
    from p1_qc.features import build_features

    feature_config = load_config(Path(root) / "configs/p1.toml", env={})
    if feature_config.features.rolling_hours != (3, 6, 12, 24, 48, 72, 168):
        raise ValueError("Historical rolling geometry changed")
    bundle = build_features(frame.loc[:, BASE].copy(), config=feature_config)
    numeric, _projection, dependency = source._feature_dependency_audit(
        {"feature_columns": bundle.feature_columns,
         "categorical_columns": bundle.categorical_columns}, config)
    surface = source.RowSurface(
        keys=frame.loc[:, KEYS].copy(),
        numeric=bundle.frame.loc[:, numeric].to_numpy(dtype=np.float32, copy=True),
        station=bundle.frame["station"].astype(str).to_numpy(),
        layer_category=bundle.frame["layer_category"].astype(str).to_numpy(),
        depth_regime=None,
        depth=bundle.frame["depth_raw"].to_numpy(dtype=np.float32, copy=True),
    )
    if not bundle.frame.index.equals(frame.index):
        raise ValueError("Fresh features changed row order")
    return surface, numeric, dependency


def encoder_from_receipt(payload):
    import numpy as np

    from p1_qc.ms_tcn_asrf_data import RobustRowEncoder

    values = dict(payload)
    values.pop("preprocessing_fit_uses_holdout_rows", None)
    for key in ("center", "scale"):
        values[key] = np.asarray(values[key], dtype=np.float32)
    for key in ("station_vocab", "layer_vocab", "depth_regime_vocab", "numeric_names"):
        values[key] = tuple(values[key])
    if values["depth_thresholds"] is not None:
        values["depth_thresholds"] = tuple(values["depth_thresholds"])
    return RobustRowEncoder(**values)


def encode_with_saved(surface, encoder, source):
    import numpy as np

    from p1_qc.ms_tcn_asrf_data import SegmentLayout

    layout = SegmentLayout.from_aligned(*(surface.keys[key] for key in KEYS))
    features = encoder.transform(
        surface.numeric, surface.station, surface.layer_category,
        depth=surface.depth, row_valid=np.ones(surface.rows, dtype=bool),
        gap=layout.gap_by_row.astype(bool), one_hot_categories=True).dense
    if features.shape[1] != 165:
        raise ValueError("Saved encoder input width changed")
    return source.EncodedSurface(surface, features, layout, None)


def array_sha(array):
    import numpy as np

    value = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((value.shape, value.dtype.str)).encode("ascii"))
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def runtime(torch):
    return {
        "pid": os.getpid(), "python": sys.version, "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "cpu_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "precision": "bfloat16", "no_backend_control_changes": True,
    }


def configure_device(source):
    _np, _pd, torch, _model, _data = source._load_scientific()
    torch.set_num_threads(2)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Original CUDA bf16 recipe unavailable; no silent CPU/fp32 fallback")
    return torch, torch.device("cuda:0")


def configure_owned_runtime(output):
    """Route runtime scratch under the new run before scientific imports."""
    folder = Path(output).resolve() / "04_runtime"
    folder.mkdir(exist_ok=True)
    for key in ("TEMP", "TMP", "TMPDIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        os.environ[key] = str(folder)
    tempfile.tempdir = str(folder)


def install_io_guard(own_root, train_path=None):
    """Python audited file/network boundary; not an OS/network sandbox claim."""
    own = Path(own_root).resolve()
    train = None if train_path is None else Path(train_path).resolve()
    libraries = (Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve())

    def guard(event, args):
        if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo"}:
            raise PermissionError("Network access prohibited")
        if event != "open" or isinstance(args[0], int):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        mode = args[1] or ""
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        writing = any(flag in str(mode) for flag in "wax+") or bool(
            flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if path.is_relative_to(own):
            return
        if not writing and (path == train or any(path.is_relative_to(p) for p in libraries)):
            return
        raise PermissionError(f"File outside source-only boundary: {path.name}")

    sys.addaudithook(guard)


def arm_deadline(output, started_at):
    remaining = 21600 - (time.time() - started_at)
    if remaining <= 0:
        raise TimeoutError("Original source-to-QA deadline exhausted")

    def expire():
        terminal = Path(output) / "terminal.json"
        if not terminal.exists():
            save_json(terminal, {"status": "RESOURCE_STOP", "pid": os.getpid(),
                                 "started_at": started_at, "time": time.time()})
        os._exit(124)

    timer = threading.Timer(remaining, expire)
    timer.daemon = True
    timer.start()
    return timer


def probe_prediction(model, values, valid, source, device):
    import torch

    model.eval()
    with torch.no_grad(), source._autocast(device):
        output = model(torch.from_numpy(values).to(device),
                       valid_mask=torch.from_numpy(valid).to(device, dtype=torch.bool))
    return {"row": torch.sigmoid(output.final_logits).float().cpu().numpy(),
            "boundary": torch.sigmoid(output.boundary_logits).float().cpu().numpy(),
            "type": torch.sigmoid(output.type_logits).float().cpu().numpy()}


def train(source_root, train_path, output, *, training_approved=False, gpu_approved=False):
    if not (training_approved and gpu_approved):
        raise PermissionError("Training and exclusive GPU allocation require explicit approval")
    source_root, train_path, output = map(lambda p: Path(p).resolve(),
                                         (source_root, train_path, output))
    if train_path.name != "train.csv" or output.exists():
        raise ValueError("Use train.csv and a completely new output directory; no restart")
    verify_sources(source_root)
    output.mkdir(parents=True)
    started_at = time.time()
    save_json(output / "ATTEMPT_LOCK.json", {
        "id": ID, "pid": os.getpid(), "started_at": started_at,
        "contract": contract(), "train_path": str(train_path)})
    timer = arm_deadline(output, started_at)
    completed = 0
    try:
        code = output / "02_code"
        for relative in PINS:
            destination = code / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_root / relative, destination)
        shutil.copyfile(Path(__file__), code / "mstcn.py")
        configure_owned_runtime(output)
        source, config = load_source(code)
        import numpy as np
        import pandas as pd

        torch, device = configure_device(source)
        models = output / "03_model"
        models.mkdir()
        validation = output / "04_validation"
        validation.mkdir()
        install_io_guard(output, train_path)
        if sha(train_path) != TRAIN_SHA:
            raise ValueError("Organizer train source SHA changed")
        frame = pd.read_csv(train_path, usecols=[*BASE, "label", "anomaly_type"], low_memory=False)
        if len(frame) != 776706 or frame.duplicated(list(KEYS)).any():
            raise ValueError("Historical training population/key contract changed")
        surface, names, dependency = fresh_features(frame, code, source, config)
        surface.labels = frame["label"].to_numpy(dtype=np.int8, copy=True)
        surface.anomaly_type = frame["anomaly_type"].fillna("").astype(str).to_numpy()
        if not np.isin(surface.labels, [0, 1]).all():
            raise ValueError("Nonbinary training target")
        ids = np.arange(surface.rows, dtype=np.int64)
        encoder, (encoded,) = source._fit_encoder_and_transform(
            surface, [], fit_ids=ids, forbidden_ids=np.asarray([], dtype=np.int64),
            numeric_names=names)
        if encoded.features.shape != (776706, 165):
            raise ValueError("Fresh full training tensor shape changed")
        save_json(models / "encoder.json", source._encoder_receipt(encoder))
        save_json(output / "prepare.json", {
            "train_sha256": TRAIN_SHA, "rows": len(frame),
            "input_shape": list(encoded.features.shape), "feature_sha256": array_sha(encoded.features),
            "ordered_key_sha256": source._ordered_key_sha(surface.keys),
            "numeric_names": list(names), "dependency": dependency,
            "wall_seconds": time.time() - started_at,
            "old_cache_reads": 0, "official_access_rows": 0})
        first_windows = source._all_windows(encoded, config)[:2]
        values, valid = source._load_scientific()[4].materialize_windows(encoded.features, first_windows)
        np.savez_compressed(validation / "own_train_probe.npz", values=values, valid=valid)
        receipts = []
        for seed in SEEDS:
            capacity = source._config_for_capacity(config, width=512, seed=seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model = source._new_model(165, capacity, device)
            if model.trainable_parameter_count != 52568587:
                raise ValueError("Parameter count mismatch")
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=capacity["training"]["learning_rate"],
                weight_decay=capacity["training"]["weight_decay"])
            windows = source._selected_windows(encoded, capacity)
            steps, total_steps, warmup = source._schedule_geometry(capacity, window_count=len(windows))
            if len(windows) != 1707 or steps != 27 or total_steps != 8100 or warmup != 270:
                raise ValueError("Historical full training window/schedule geometry changed")
            positive_weight = source._positive_weight(surface.labels)
            step, nonfinite = 0, 0
            started_seed = time.time()
            for epoch in range(1, 151):
                telemetry, step, _lr = source._train_epoch(
                    model, optimizer, encoded, windows, config=capacity,
                    positive_weight=positive_weight, device=device, epoch=epoch,
                    global_step=step, total_steps=total_steps)
                nonfinite += int(telemetry.get("nonfinite_count", 0))
                if epoch == 1 or epoch % 5 == 0:
                    save_json(output / "progress.json", {
                        "status": "RUNNING", "pid": os.getpid(), "seed": seed,
                        "epoch": epoch, "epochs": 150, "completed_fits": completed,
                        "maximum_fits": 3, "elapsed_seconds": time.time() - started_at}, progress=True)
            if step != 4050 or nonfinite:
                raise ValueError("Optimizer/nonfinite contract failed")
            model_path = models / f"seed_{seed}.pt"
            with model_path.open("xb") as handle:
                torch.save({"seed": seed, "epoch": 150, "input_features": 165,
                            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, handle)
                handle.flush()
                os.fsync(handle.fileno())
            np.savez_compressed(validation / f"probe_expected_{seed}.npz",
                                **probe_prediction(model, values, valid, source, device))
            completed += 1
            receipt = {"seed": seed, "epoch": 150, "optimizer_steps": step,
                       "windows": len(windows), "model_sha256": sha(model_path),
                       "wall_seconds": time.time() - started_seed, "nonfinite_count": nonfinite}
            receipts.append(receipt)
            save_json(validation / f"fit_{seed}.json", receipt)
            del optimizer, model
            torch.cuda.empty_cache()
        if sha(train_path) != TRAIN_SHA:
            raise ValueError("Train source changed during fit")
        files = {p.relative_to(output).as_posix(): sha(p)
                 for folder in (code, models, validation)
                 for p in sorted(folder.rglob("*"))
                 if p.is_file() and "__pycache__" not in p.parts}
        save_json(output / "training-result.json", {
            "status": "TRAINING_COMPLETE_REPLAY_PENDING", "id": ID,
            "started_at": started_at, "training_pid": os.getpid(), "contract": contract(),
            "source_pins": PINS, "files_sha256": files, "fit_receipts": receipts,
            "completed_fits": completed, "wall_seconds": time.time() - started_at,
            "runtime": runtime(torch), "official_access_rows": 0,
            "old_models_cache_answer_reads": 0, "fresh_process_replay": "PENDING"})
    except BaseException as error:
        if not (output / "terminal.json").exists():
            save_json(output / "terminal.json", {
                "status": "TERMINAL_TECHNICAL_FAILURE", "error_type": type(error).__name__,
                "error": str(error), "completed_fits": completed,
                "pid": os.getpid(), "elapsed_seconds": time.time() - started_at})
        raise
    finally:
        timer.cancel()


def verify_owned(output):
    output = Path(output).resolve()
    receipt = read_json(output / "training-result.json")
    if receipt["contract"] != contract() or receipt["completed_fits"] != 3:
        raise ValueError("Training receipt contract mismatch")
    for relative, expected in receipt["files_sha256"].items():
        path = (output / relative).resolve()
        if not path.is_relative_to(output) or sha(path) != expected:
            raise ValueError(f"Own generated file SHA mismatch: {relative}")
    if sha(__file__) != receipt["files_sha256"]["02_code/mstcn.py"]:
        raise ValueError("Executing reconstruction module differs from the training snapshot")
    return receipt


def load_model(model_dir, seed, source, config, device):
    import torch

    state = torch.load(Path(model_dir) / f"seed_{seed}.pt", map_location="cpu", weights_only=True)
    if (state["seed"], state["epoch"], state["input_features"]) != (seed, 150, 165):
        raise ValueError("Own state contract mismatch")
    model = source._new_model(165, source._config_for_capacity(config, width=512, seed=seed), device)
    model.load_state_dict(state["state_dict"], strict=True)
    return model


def replay(output):
    output = Path(output).resolve()
    receipt = verify_owned(output)
    if receipt["training_pid"] == os.getpid() or (output / "terminal.json").exists():
        raise RuntimeError("Requires a fresh PID and an unconsumed terminal stage")
    save_json(output / "REPLAY_ATTEMPT_LOCK.json", {"pid": os.getpid(), "time": time.time()})
    timer = arm_deadline(output, receipt["started_at"])
    try:
        configure_owned_runtime(output)
        source, config = load_source(output / "02_code")
        import numpy as np

        torch, device = configure_device(source)
        install_io_guard(output)
        checks = {}
        with np.load(output / "04_validation/own_train_probe.npz", allow_pickle=False) as probe:
            values, valid = probe["values"], probe["valid"]
        for seed in SEEDS:
            model = load_model(output / "03_model", seed, source, config, device)
            observed = probe_prediction(model, values, valid, source, device)
            with np.load(output / f"04_validation/probe_expected_{seed}.npz", allow_pickle=False) as expected:
                for name in ("row", "boundary", "type"):
                    checks[f"{seed}_{name}_exact"] = bool(np.array_equal(observed[name], expected[name]))
            del model
            torch.cuda.empty_cache()
        checks["fresh_pid"] = receipt["training_pid"] != os.getpid()
        checks["within_original_6h"] = time.time() - receipt["started_at"] <= 21600
        current_runtime = runtime(torch)
        for key, value in receipt["runtime"].items():
            if key != "pid":
                checks[f"runtime_{key}"] = current_runtime[key] == value
        verify_owned(output)
        result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
                  "pid": os.getpid(), "training_pid": receipt["training_pid"],
                  "training_result_sha256": sha(output / "training-result.json"),
                  "runtime": current_runtime, "official_access_rows": 0,
                  "scratch_retraining_determinism": "NOT_TESTED",
                  "source_to_replay_seconds": time.time() - receipt["started_at"]}
        save_json(output / "fresh-process-replay.json", result)
        save_json(output / "terminal.json", result)
        if result["status"] != "PASS":
            raise ValueError("Saved-model exact replay failed; do not retry or retune")
    except BaseException as error:
        if not (output / "terminal.json").exists():
            save_json(output / "terminal.json", {
                "status": "TERMINAL_REPLAY_FAILURE", "pid": os.getpid(),
                "error_type": type(error).__name__, "error": str(error),
                "training_result_sha256": sha(output / "training-result.json")})
        raise
    finally:
        timer.cancel()


def predict_frame(frame, *, model_dir):
    """Explicit new-model directory -> same-key MS-TCN proposal, in memory only.

    Caller supplies an authorized covariate frame. No CSV/old anchor is opened;
    encoder is loaded, never refitted. This is not the complete tree+MS union.
    """
    model_dir = Path(model_dir).resolve()
    output = model_dir.parent
    if model_dir.name != "03_model":
        raise ValueError("Explicit own 03_model directory required")
    verify_owned(output)
    replay_receipt = read_json(output / "fresh-process-replay.json")
    if replay_receipt["status"] != "PASS" or replay_receipt["training_result_sha256"] != sha(output / "training-result.json"):
        raise ValueError("Independent saved-model replay required before inference")
    source, config = load_source(output / "02_code")
    import numpy as np

    torch, device = configure_device(source)
    surface, _names, _dependency = fresh_features(frame, output / "02_code", source, config)
    encoder = encoder_from_receipt(read_json(model_dir / "encoder.json"))
    encoded = encode_with_saved(surface, encoder, source)
    windows = source._all_windows(encoded, config)
    predictions = []
    for seed in SEEDS:
        model = load_model(model_dir, seed, source, config, device)
        prediction = source.predict_encoded(model, encoded, windows, batch_size=64, device=device)
        predictions.append(prediction)
        del model
        torch.cuda.empty_cache()
    mean = source.PredictionBundle(*(
        np.mean(np.stack([getattr(p, name) for p in predictions]), axis=0).astype(np.float32)
        for name in ("row_probability", "boundary_probability", "type_probability")))
    proposal = source.decode_long_event_segments(
        source._decoder_row_probability(mean, config), mean.boundary_probability, encoded.layout,
        high_threshold=0.8, low_threshold=0.4, snap_radius=12, minimum_rows=19, maximum_rows=None)
    return surface.keys.copy(), proposal, mean


def predict_proposal(frame, *, model_dir):
    """Composition interface: (ordered keys, .8 proposal bits, provenance receipt)."""
    keys, bits, _mean = predict_frame(frame, model_dir=model_dir)
    output = Path(model_dir).resolve().parent
    source, _config = load_source(output / "02_code")
    receipt = {
        "id": ID, "rows": len(keys), "ordered_key_sha256": source._ordered_key_sha(keys),
        "proposal_bits_sha256": array_sha(bits), "positive_rows": int(bits.sum()),
        "training_result_sha256": sha(output / "training-result.json"),
        "fresh_process_replay_sha256": sha(output / "fresh-process-replay.json"),
        "encoder_sha256": sha(Path(model_dir) / "encoder.json"),
        "model_sha256": {str(seed): sha(Path(model_dir) / f"seed_{seed}.pt") for seed in SEEDS},
        "source_sha256": PINS, "threshold": 0.8, "anchor_input_rows": 0,
        "full_policy_union_score": "NOT_COMPUTED", "file_io_official_rows": 0,
        "input_frame_authorization": "caller responsibility; no labels used",
    }
    return keys, bits, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("contract", "train", "replay"))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--training-approved", action="store_true")
    parser.add_argument("--gpu-approved", action="store_true")
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "2"
    if args.mode == "contract":
        print(json.dumps(contract(), indent=2))
    elif args.mode == "train":
        if not all((args.source_root, args.train_csv, args.output)):
            parser.error("train requires --source-root, --train-csv, --output")
        train(args.source_root, args.train_csv, args.output,
              training_approved=args.training_approved, gpu_approved=args.gpu_approved)
    else:
        if args.output is None:
            parser.error("replay requires --output")
        replay(args.output)


if __name__ == "__main__":
    main()
