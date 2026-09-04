"""One-shot runner for the preregistered P1 TE-TAD-lite local experiment.

The runner only consumes frozen local training/OOF artefacts.  It deliberately
has no deployment-output path.  All scientific imports are lazy so the
Windows CUDA:PTX bootstrap is installed before tinygrad is imported.
"""

from __future__ import annotations

import argparse
import ctypes.util
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "p1_tetad_lite_direct_interval_set_v1"
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
SEAL_PATH = ROOT / "artifacts" / f"{EXPERIMENT_ID}_execution_seal.json"
ATTEMPT_LOCK = ROOT / "artifacts" / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
TERMINAL_PATH = ARTIFACT_DIR / "terminal_result.json"

MODEL_MODULE = ROOT / "src" / "p1_qc" / "tetad_lite_tinygrad.py"
EXPERIMENT_MODULE = ROOT / "src" / "p1_qc" / "tetad_lite_experiment.py"
MODEL_TEST = ROOT / "tests" / "test_p1_tetad_lite_tinygrad.py"
EXPERIMENT_TEST = ROOT / "tests" / "test_p1_tetad_lite_experiment.py"
RUNNER_TEST = ROOT / "tests" / "test_run_p1_tetad_lite_direct_interval_set_v1.py"


class ContractError(RuntimeError):
    """The preregistration or one-shot execution contract was violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _exclusive_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _exclusive_json(path: Path, value: Any) -> None:
    _exclusive_bytes(path, _json_bytes(value))


def _atomic_npz(path: Path, **arrays: Any) -> str:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(path)


def _ordered_key_sha256(keys: Any) -> str:
    """Hash ordered key columns without emitting any row values."""

    digest = hashlib.sha256()
    for column in ("station", "year", "layer", "time"):
        digest.update(column.encode("ascii") + b"\0")
        for value in keys.get_column(column).to_list():
            raw = str(value).encode("utf-8")
            digest.update(len(raw).to_bytes(4, "little"))
            digest.update(raw)
    return digest.hexdigest()


def _load_config() -> dict[str, Any]:
    if ROOT.resolve() != Path(__file__).resolve().parents[1]:
        raise ContractError("runner is not under its canonical repository root")
    raw = CONFIG_PATH.read_bytes()
    config = json.loads(raw.decode("utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("experiment identity changed")
    if len(config.get("selected_numeric_features", ())) != 46:
        raise ContractError("registered numeric feature count is not 46")
    if config["preprocessing"].get("expected_input_features") != 97:
        raise ContractError("registered model input dimension is not 97")
    if config["windowing"].get("patch_rows") != 8:
        raise ContractError("registered patch size is not 8")
    if config["windowing"]["rows"] % config["windowing"]["patch_rows"]:
        raise ContractError("window length is not exactly divisible by patch size")
    return config


def _bootstrap_gpu_runtime() -> None:
    """Set tinygrad's backend and make the Windows driver library discoverable."""

    os.environ["DEVICE"] = "CUDA:PTX"
    os.environ["DISKCACHE"] = "0"
    original = ctypes.util.find_library
    if not getattr(original, "_p1_tetad_cuda_patch", False):

        def find_library(name: str) -> str | None:
            return "nvcuda.dll" if name == "cuda" else original(name)

        find_library._p1_tetad_cuda_patch = True  # type: ignore[attr-defined]
        ctypes.util.find_library = find_library


def _load_scientific_modules() -> tuple[Any, Any, Any, Any]:
    _bootstrap_gpu_runtime()
    source = str(ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    import numpy as np
    import polars as pl

    from p1_qc import tetad_lite_experiment as engine
    from p1_qc import tetad_lite_tinygrad as model_module

    return np, pl, engine, model_module


def _parse_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _verify_anchor_identity(config: dict[str, Any], engine: Any, pl: Any, anchor: Any) -> None:
    frozen = config["immutable_inputs"]["frozen_round_b_anchor"]
    path = ROOT / frozen["path"]
    if (
        not path.is_file()
        or path.stat().st_size != frozen["bytes"]
        or _sha256(path) != frozen["sha256"]
    ):
        raise ContractError("frozen Round-B identity surface changed")
    comparison = pl.read_parquet(path, columns=[*engine.KEY_COLUMNS, "fold", frozen["column"]])
    if comparison.height != frozen["rows"]:
        raise ContractError("frozen Round-B identity row count changed")
    for column in (*engine.KEY_COLUMNS, "fold"):
        if not (
            comparison.get_column(column).cast(pl.String).to_numpy()
            == anchor.get_column(column).cast(pl.String).to_numpy()
        ).all():
            raise ContractError(f"Round-B identity key differs: {column}")
    left = comparison.get_column(frozen["column"]).to_numpy()
    right = anchor.get_column("anchor_prediction").to_numpy()
    if not (left == right).all():
        raise ContractError("p100 anchor bits differ from the frozen Round-B identity surface")


def build_preflight(config: dict[str, Any]) -> dict[str, Any]:
    """Run the zero-fit, prefix-only input/query audit."""

    np, pl, engine, _model_module = _load_scientific_modules()
    training_inputs, truth_oof, round_b_parts = engine.canonical_frozen_inputs(ROOT)
    training_inputs.feature_cache.verify()
    training_inputs.key_sidecar.verify()
    training_inputs.label_cache.verify()
    truth_oof.verify()
    part_contract = config["immutable_inputs"]["frozen_round_b_prediction_parts"]
    for part in round_b_parts:
        part.verify()
        record = part_contract[part.fold]
        if (
            part.path.relative_to(ROOT).as_posix() != record["path"]
            or part.rows != record["rows"]
            or part.path.stat().st_size != record["bytes"]
            or part.sha256 != record["sha256"]
        ):
            raise ContractError(f"registered p100 part changed: {part.fold}")
    membership = engine.load_validation_membership(truth_oof)
    anchor = engine.load_frozen_anchor_surface(
        truth_oof,
        round_b_parts,
        prediction_column=part_contract["prediction_column"],
    )
    _verify_anchor_identity(config, engine, pl, anchor)

    expected_folds = tuple(config["chronological_protocol"]["fold_order"])
    observed_folds = tuple(dict.fromkeys(membership.get_column("fold").cast(pl.String).to_list()))
    if observed_folds != expected_folds:
        raise ContractError("exact OOF fold ordering changed")
    if membership.height != config["immutable_inputs"]["frozen_truth_and_folds"]["rows"]:
        raise ContractError("exact OOF membership size changed")

    features = tuple(config["selected_numeric_features"])
    window_length = int(config["windowing"]["rows"])
    stride = int(config["windowing"]["stride_rows"])
    max_queries = int(config["architecture"]["learned_queries"])
    purge = timedelta(
        days=int(config["windowing"]["train_validation_raw_dependency_guard"]["purge_days"])
    )
    registered = config["query_preflight"]
    fold_receipts: dict[str, Any] = {}
    for fold in expected_folds:
        cutoff_text = config["chronological_protocol"][fold.split("_")[-1]][
            "training_rows_time_before"
        ]
        cutoff = _parse_time(cutoff_text)
        fold_membership = membership.filter(pl.col("fold") == fold)
        validation_start = min(_parse_time(value) for value in fold_membership.get_column("time"))
        engine.assert_split_dependency_safe(
            cutoff,
            validation_start,
            purge=purge,
            window_length=window_length,
            maximum_feature_lookahead=timedelta(
                days=int(
                    config["windowing"]["train_validation_raw_dependency_guard"][
                        "offline_centered_feature_radius_days"
                    ]
                )
            ),
        )
        bundle = engine.load_training_prefix_bundle(training_inputs, features, cutoff=cutoff)
        station_count = bundle.keys.get_column("station").cast(pl.String).n_unique()
        input_features = 2 * len(features) + station_count + 2
        if input_features != config["preprocessing"]["expected_input_features"]:
            raise ContractError(f"{fold}: effective input dimension changed")
        segments = engine.exact_cadence_segments(bundle.keys)
        events = engine.eligible_target_events(
            segments,
            bundle.labels,
            minimum_original_rows=int(config["target_definition"]["minimum_original_event_rows"]),
            right_censor_cutoff=cutoff,
        )
        windows = engine.build_windows(
            segments,
            events,
            window_length=window_length,
            stride=stride,
            max_queries=max_queries,
        )
        maximum = max((len(window.targets) for window in windows), default=0)
        expected_windows = int(
            registered["registered_training_windows_including_short_padding"][fold]
        )
        expected_maximum = int(registered["observed_max_targets_per_window"][fold])
        if len(windows) != expected_windows or maximum != expected_maximum:
            raise ContractError(f"{fold}: registered query preflight aggregate changed")
        if maximum > int(registered["hard_fail_above"]):
            raise ContractError(f"{fold}: query budget overflow")
        fold_receipts[fold] = {
            "training_windows": len(windows),
            "maximum_targets_per_window": maximum,
            "training_segments": len(segments),
            "qualifying_events": len(events),
            "effective_input_features": input_features,
            "dependency_gap_seconds": int((validation_start - cutoff).total_seconds()),
        }
    frozen_specs = (
        training_inputs.feature_cache,
        training_inputs.key_sidecar,
        training_inputs.label_cache,
        truth_oof,
        *round_b_parts,
    )
    input_records = {
        spec.path.relative_to(ROOT).as_posix(): {
            "bytes": spec.path.stat().st_size,
            "sha256": spec.sha256,
        }
        for spec in frozen_specs
    }
    anchor_record = config["immutable_inputs"]["frozen_round_b_anchor"]
    input_records[anchor_record["path"]] = {
        "bytes": anchor_record["bytes"],
        "sha256": anchor_record["sha256"],
    }
    return {
        "schema_version": "p1.tetad_lite.query_preflight.v1",
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": _sha256(CONFIG_PATH),
        "feature_count": len(features),
        "input_features": config["preprocessing"]["expected_input_features"],
        "effective_patch_rows": config["windowing"]["patch_rows"],
        "oof_rows": membership.height,
        "frozen_inputs": input_records,
        "folds": fold_receipts,
        "anchor_identity": "PASS",
        "result": "PASS",
    }


def _compiler_support_path() -> Path:
    path = (
        ROOT
        / ".venv-p1"
        / "Lib"
        / "site-packages"
        / "tinygrad"
        / "runtime"
        / "support"
        / "compiler_cuda.py"
    )
    if not path.is_file():
        raise ContractError("patched tinygrad CUDA compiler support file is unavailable")
    return path


def _bound_files() -> dict[str, Path]:
    return {
        "config": CONFIG_PATH,
        "runner": Path(__file__).resolve(),
        "model_module": MODEL_MODULE,
        "experiment_module": EXPERIMENT_MODULE,
        "model_test": MODEL_TEST,
        "experiment_test": EXPERIMENT_TEST,
        "runner_test": RUNNER_TEST,
        "tinygrad_compiler_support": _compiler_support_path(),
    }


def create_execution_seal(config: dict[str, Any], preflight: dict[str, Any]) -> dict[str, Any]:
    if SEAL_PATH.exists() or ATTEMPT_LOCK.exists() or TERMINAL_PATH.exists():
        raise ContractError("seal, attempt, or terminal artefact already exists")
    files = _bound_files()
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise ContractError(f"seal inputs missing: {missing}")
    compiler_record = config["runtime"]["windows_cuda_loader_patch"]
    compiler_path = files["tinygrad_compiler_support"]
    if (
        compiler_path.relative_to(ROOT).as_posix() != compiler_record["path"]
        or _sha256(compiler_path) != compiler_record["sha256"]
    ):
        raise ContractError("patched CUDA compiler support no longer matches preregistration")
    dependencies: dict[str, str] = {}
    for name in ("numpy", "polars", "scipy", "tinygrad"):
        dependencies[name] = importlib.metadata.version(name)
        if dependencies[name] != config["runtime"][name]:
            raise ContractError(f"registered dependency version changed: {name}")
    if platform.python_version() != config["runtime"]["python"]:
        raise ContractError("registered Python version changed")
    seal = {
        "schema_version": "p1.tetad_lite.execution_seal.v1",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "files": {
            name: {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in files.items()
        },
        "dependencies": {
            "python": platform.python_version(),
            **dependencies,
        },
        "gpu_runtime": {"device": "CUDA:PTX", "diskcache": 0},
        "query_preflight": preflight,
    }
    _exclusive_json(SEAL_PATH, seal)
    return seal


def _load_verified_seal() -> dict[str, Any]:
    if not SEAL_PATH.is_file():
        raise ContractError("execution seal is absent")
    seal = json.loads(SEAL_PATH.read_text(encoding="utf-8"))
    if seal.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("execution seal identity changed")
    current = _bound_files()
    if set(seal.get("files", {})) != set(current):
        raise ContractError("execution seal file inventory changed")
    for name, path in current.items():
        record = seal["files"][name]
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise ContractError(f"sealed runtime file changed: {name}")
    if seal["query_preflight"].get("result") != "PASS":
        raise ContractError("sealed query preflight did not pass")
    for relative, record in seal["query_preflight"].get("frozen_inputs", {}).items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()):
            raise ContractError("sealed input path escapes repository")
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or _sha256(path) != record["sha256"]
        ):
            raise ContractError(f"sealed frozen input changed: {relative}")
    return seal


@dataclass
class BlindFold:
    fold: str
    confidence: Any
    anchor: Any
    keys: Any


class ScientificRuntime:
    """Actual tinygrad implementation behind the testable one-shot orchestrator."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.np, self.pl, self.engine, self.model_module = _load_scientific_modules()
        self.inputs, self.truth_oof, parts = self.engine.canonical_frozen_inputs(ROOT)
        self.anchor = self.engine.load_frozen_anchor_surface(
            self.truth_oof,
            parts,
            prediction_column=config["immutable_inputs"]["frozen_round_b_prediction_parts"][
                "prediction_column"
            ],
        )
        _verify_anchor_identity(config, self.engine, self.pl, self.anchor)
        self.metadata: dict[str, Any] = {}

    def _new_model(self, seed: int) -> Any:
        from tinygrad import Tensor
        from tinygrad.nn.state import get_parameters

        architecture = self.config["architecture"]
        Tensor.manual_seed(seed)
        model_config = self.model_module.TETADLiteConfig(
            input_features=self.config["preprocessing"]["expected_input_features"],
            patch_size=architecture["patch_size"],
            d_model=architecture["d_model"],
            num_heads=architecture["num_heads"],
            ff_multiplier=architecture["feedforward_multiplier"],
            num_encoder_layers=architecture["bidirectional_encoder_layers"],
            num_queries=architecture["learned_queries"],
            max_patches=architecture["maximum_patches"],
        )
        model = self.model_module.TETADLiteTinygrad(model_config)
        parameters = get_parameters(model)
        if not parameters or not all(
            str(parameter.device).upper().startswith("CUDA") for parameter in parameters
        ):
            raise ContractError("model parameters are not resident on a CUDA device")
        return model

    def _prefix_training_surface(self, fold: str) -> tuple[Any, list[Any], Any]:
        phase = self.config["chronological_protocol"][fold.split("_")[-1]]
        cutoff = _parse_time(phase["training_rows_time_before"])
        bundle = self.engine.load_training_prefix_bundle(
            self.inputs, self.config["selected_numeric_features"], cutoff=cutoff
        )
        segments = self.engine.exact_cadence_segments(bundle.keys)
        events = self.engine.eligible_target_events(
            segments,
            bundle.labels,
            minimum_original_rows=self.config["target_definition"]["minimum_original_event_rows"],
            right_censor_cutoff=cutoff,
        )
        windows = self.engine.build_windows(
            segments,
            events,
            window_length=self.config["windowing"]["rows"],
            stride=self.config["windowing"]["stride_rows"],
            max_queries=self.config["architecture"]["learned_queries"],
        )
        all_indices = self.np.arange(bundle.rows, dtype=self.np.int64)
        preprocessor = self.engine.RobustPreprocessor.fit(
            bundle,
            all_indices,
            self.config["selected_numeric_features"],
            train_end=cutoff,
        )
        if preprocessor.output_features != self.config["preprocessing"]["expected_input_features"]:
            raise ContractError("fitted preprocessor dimension changed")
        return bundle, windows, preprocessor

    def _window_rank(self, bundle: Any, window: Any, fold: str, seed: int) -> bytes:
        first = int(window.row_indices[0])
        keys = bundle.keys.row(first, named=True)
        identity = "|".join(
            (
                str(seed),
                fold,
                str(keys["station"]),
                str(keys["year"]),
                str(keys["layer"]),
                str(keys["time"]),
            )
        )
        return hashlib.sha256(identity.encode("utf-8")).digest()

    def _registered_training_sample(
        self, bundle: Any, windows: list[Any], fold: str, seed: int
    ) -> list[Any]:
        positive = [window for window in windows if len(window.targets)]
        empty = sorted(
            (window for window in windows if not len(window.targets)),
            key=lambda window: self._window_rank(bundle, window, fold, seed),
        )
        selected = positive + empty[: min(len(empty), 2 * len(positive))]
        return sorted(selected, key=lambda window: (window.segment_id, window.start))

    def sanity_fit(self) -> dict[str, Any]:
        from tinygrad.nn.state import get_parameters

        config = self.config["implementation_sanity_gate"]
        bundle, windows, preprocessor = self._prefix_training_surface("2025_q2")
        seed = int(config["fresh_seed"])

        def rank(window: Any) -> bytes:
            return self._window_rank(bundle, window, "2025_q2", seed)

        positive = sorted((window for window in windows if len(window.targets)), key=rank)[:16]
        empty = sorted((window for window in windows if not len(window.targets)), key=rank)[:16]
        selected = positive + empty
        if len(positive) != 16 or len(empty) != 16:
            raise ContractError("registered 32-window sanity sample is unavailable")
        features = self.engine.materialize_windows(
            bundle, selected, preprocessor, window_length=self.config["windowing"]["rows"]
        )
        model = self._new_model(seed)
        trained = self.engine.train_tetad(
            model,
            features,
            [window.targets for window in selected],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            positive_class_weight=self.config["training"]["positive_query_weight"],
            seed=seed,
            **_loss_arguments(self.config),
        )
        intervals, scores = self.engine.predict_window_proposals(
            trained.model, features, batch_size=config["batch_size"]
        )
        metrics = self.engine.evaluate_overfit_sanity(
            intervals,
            scores,
            [window.targets for window in selected],
            score_threshold=config["threshold"],
            match_iou=config["pass_if_all"]["matched_target_iou_cutoff"],
        )
        gradients = [
            parameter.grad
            for parameter in get_parameters(trained.model)
            if parameter.grad is not None
        ]
        metrics["finite_gradients"] = bool(
            gradients and all(self.np.isfinite(gradient.numpy()).all() for gradient in gradients)
        )
        metrics["final_loss"] = float(trained.epoch_losses[-1])
        return metrics

    def sanity_passes(self, metrics: dict[str, Any]) -> bool:
        gates = self.config["implementation_sanity_gate"]["pass_if_all"]
        return bool(
            metrics.get("finite")
            and metrics.get("finite_gradients")
            and self.np.isfinite(metrics.get("final_loss", self.np.nan))
            and metrics.get("target_recall", 0.0)
            >= gates["matched_target_probability_and_iou_recall_min"]
            and metrics.get("median_matched_iou", 0.0) >= gates["median_matched_iou_min"]
            and metrics.get("negative_window_fp_windows", 999)
            <= gates["negative_windows_with_any_active_query_max"]
        )

    def fit_predict_blind(self, fold: str) -> BlindFold:
        training = self.config["training"]
        bundle, windows, preprocessor = self._prefix_training_surface(fold)
        sampled = self._registered_training_sample(bundle, windows, fold, int(training["seed"]))
        train_features = self.engine.materialize_windows(
            bundle, sampled, preprocessor, window_length=self.config["windowing"]["rows"]
        )
        model = self._new_model(int(training["seed"]))
        trained = self.engine.train_tetad(
            model,
            train_features,
            [window.targets for window in sampled],
            epochs=training["epochs"],
            batch_size=training["batch_size"],
            learning_rate=training["learning_rate"],
            weight_decay=training["weight_decay"],
            positive_class_weight=training["positive_query_weight"],
            seed=training["seed"],
            **_loss_arguments(self.config),
        )
        validation = self.engine.load_validation_feature_bundle(
            self.inputs,
            self.truth_oof,
            self.config["selected_numeric_features"],
            fold=fold,
        )
        validation_segments = self.engine.exact_cadence_segments(validation.keys)
        validation_windows = self.engine.build_windows(
            validation_segments,
            (),
            window_length=self.config["windowing"]["rows"],
            stride=self.config["windowing"]["stride_rows"],
            max_queries=self.config["architecture"]["learned_queries"],
        )
        validation_features = self.engine.materialize_windows(
            validation,
            validation_windows,
            preprocessor,
            window_length=self.config["windowing"]["rows"],
        )
        intervals, scores = self.engine.predict_window_proposals(
            trained.model, validation_features, batch_size=training["batch_size"]
        )
        confidence, _ignored = self.engine.stitch_proposals(
            validation_windows,
            intervals,
            scores,
            total_rows=validation.rows,
            threshold=0.0,
            minimum_decoded_rows=self.config["decode_and_stitch"][
                "minimum_decoded_proposal_rows_before_outer_clip"
            ],
            coordinate_length=self.config["windowing"]["rows"],
        )
        anchor = self.anchor.filter(self.pl.col("fold") == fold)
        for column in self.engine.KEY_COLUMNS:
            if not (
                anchor.get_column(column).cast(self.pl.String).to_numpy()
                == validation.keys.get_column(column).cast(self.pl.String).to_numpy()
            ).all():
                raise ContractError(f"blind validation/anchor binding differs: {fold}/{column}")
        self.metadata[fold] = {
            "keys": validation.keys,
            "segments": validation_segments,
            "epoch_losses": tuple(trained.epoch_losses),
            "train_windows": len(sampled),
            "validation_windows": len(validation_windows),
        }
        return BlindFold(
            fold=fold,
            confidence=confidence.astype(self.np.float32),
            anchor=anchor.get_column("anchor_prediction").to_numpy().astype(self.np.int8),
            keys=validation.keys,
        )

    def load_truth_after_receipt(self, blind: BlindFold, receipt_path: Path) -> Any:
        receipt = _verify_blind_receipt(receipt_path)
        if receipt.get("ordered_key_sha256") != _ordered_key_sha256(blind.keys):
            raise ContractError("blind-score receipt key digest changed before truth access")
        truth = self.engine.load_validation_fold_truth(self.truth_oof, blind.fold)
        for column in self.engine.KEY_COLUMNS:
            if not (
                truth.get_column(column).cast(self.pl.String).to_numpy()
                == blind.keys.get_column(column).cast(self.pl.String).to_numpy()
            ).all():
                raise ContractError(f"blind/truth binding differs: {blind.fold}/{column}")
        return truth

    def training_receipt(self, fold: str) -> dict[str, int | float]:
        metadata = self.metadata[fold]
        losses = metadata["epoch_losses"]
        return {
            "sampled_training_windows": int(metadata["train_windows"]),
            "validation_windows": int(metadata["validation_windows"]),
            "first_epoch_loss": float(losses[0]),
            "final_epoch_loss": float(losses[-1]),
        }


def _loss_arguments(config: dict[str, Any]) -> dict[str, float]:
    weights = config["training"]["loss_weights"]
    return {
        "classification_weight": weights["classification"],
        "endpoint_weight": weights["endpoint_l1"],
        "iou_weight": weights["one_minus_iou"],
    }


def _commit_blind(blind: BlindFold) -> Path:
    np, _pl, _engine, _model = _load_scientific_modules()
    score_path = ARTIFACT_DIR / f"{blind.fold}_blind_scores.npz"
    receipt_path = ARTIFACT_DIR / f"{blind.fold}_blind_scores.receipt.json"
    digest = _atomic_npz(
        score_path,
        detector_confidence=np.asarray(blind.confidence, dtype=np.float32),
        anchor_prediction=np.asarray(blind.anchor, dtype=np.int8),
    )
    receipt = {
        "schema_version": "p1.tetad_lite.blind_score_receipt.v1",
        "experiment_id": EXPERIMENT_ID,
        "fold": blind.fold,
        "rows": int(len(blind.anchor)),
        "ordered_key_sha256": _ordered_key_sha256(blind.keys),
        "score_path": score_path.relative_to(ROOT).as_posix(),
        "score_sha256": digest,
        "created_before_truth_access": True,
    }
    _exclusive_json(receipt_path, receipt)
    return receipt_path


def _verify_blind_receipt(path: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    score_path = ROOT / receipt["score_path"]
    if receipt.get("experiment_id") != EXPERIMENT_ID or not score_path.is_file():
        raise ContractError("blind-score receipt is invalid")
    if _sha256(score_path) != receipt.get("score_sha256"):
        raise ContractError("blind-score bytes changed before truth access")
    return receipt


def _score_fold(
    config: dict[str, Any], runtime: Any, blind: BlindFold, truth: Any, threshold: float
) -> dict[str, Any]:
    np = runtime.np
    engine = runtime.engine
    label = truth.get_column("label").to_numpy().astype(np.int8)
    proposal = (np.asarray(blind.confidence) >= threshold).astype(np.int8)
    candidate = engine.anchor_preserving_union(blind.anchor, proposal)
    comparison = engine.compare_anchor_candidate(label, blind.anchor, candidate)["overall"]
    segments = engine.exact_cadence_segments(truth.select(list(engine.KEY_COLUMNS)))
    events = engine.eligible_target_events(
        segments,
        truth,
        minimum_original_rows=config["target_definition"]["minimum_original_event_rows"],
    )
    anchor_hits = sum(
        bool(np.any(blind.anchor[segment_index(segments, event)])) for event in events
    )
    candidate_hits = sum(
        bool(np.any(candidate[segment_index(segments, event)])) for event in events
    )
    return {
        "aggregate": comparison,
        "long_event_count": len(events),
        "anchor_long_event_recall": anchor_hits / len(events) if events else 0.0,
        "candidate_long_event_recall": candidate_hits / len(events) if events else 0.0,
        "truth": label,
        "candidate": candidate,
        "anchor": np.asarray(blind.anchor, dtype=np.int8),
        "keys": truth.select(list(engine.KEY_COLUMNS)),
    }


def segment_index(segments: list[Any], event: Any) -> Any:
    segment = next(item for item in segments if item.segment_id == event.segment_id)
    return segment.row_indices[event.start : event.end]


def _public_metrics(score: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in score.items()
        if key not in {"truth", "candidate", "anchor", "keys"}
    }


def _training_receipt(runtime: Any, fold: str) -> dict[str, Any]:
    method = getattr(runtime, "training_receipt", None)
    return method(fold) if method is not None else {}


def _select_q2_threshold(
    config: dict[str, Any], runtime: Any, blind: BlindFold, truth: Any
) -> tuple[float, dict[str, Any], bool]:
    outcomes: list[tuple[float, dict[str, Any]]] = []
    for threshold in config["chronological_protocol"]["q2"]["threshold_grid"]:
        score = _score_fold(config, runtime, blind, truth, float(threshold))
        outcomes.append((float(threshold), score))
    threshold, best = max(
        outcomes,
        key=lambda item: (item[1]["aggregate"]["candidate"]["f1"], item[0]),
    )
    aggregate = best["aggregate"]
    kill = bool(
        all(score["aggregate"]["additional_rows"] == 0 for _, score in outcomes)
        or aggregate["f1_delta"] <= 0.0
        or aggregate["added_precision"] <= aggregate["anchor"]["f1"] / 2.0
    )
    summary = {
        "selected_threshold": threshold,
        "kill": kill,
        "grid": [{"threshold": value, **_public_metrics(score)} for value, score in outcomes],
    }
    return threshold, summary, kill


def _paired_segment_bootstrap(
    runtime: Any, scores: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    np, engine = runtime.np, runtime.engine
    truth = np.concatenate([score["truth"] for score in scores])
    anchor = np.concatenate([score["anchor"] for score in scores])
    candidate = np.concatenate([score["candidate"] for score in scores])
    cluster_parts = []
    for fold_index, score in enumerate(scores):
        segments = engine.exact_cadence_segments(score["keys"])
        cluster = engine.exact_segment_cluster_ids(segments, total_rows=len(score["truth"]))
        cluster_parts.append(
            np.asarray([f"{fold_index}|{value}" for value in cluster], dtype=object)
        )
    clusters = np.concatenate(cluster_parts)
    return engine.paired_cluster_bootstrap_ci90(
        truth,
        anchor,
        candidate,
        clusters,
        replicates=config["outer_decision"]["bootstrap_replicates"],
        seed=config["outer_decision"]["bootstrap_seed"],
    )


def _outer_decision(
    runtime: Any, fold_scores: dict[str, dict[str, Any]], config: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    np, engine = runtime.np, runtime.engine
    names = config["outer_decision"]["confirmation_folds"]
    scores = [fold_scores[name] for name in names]
    truth = np.concatenate([score["truth"] for score in scores])
    anchor = np.concatenate([score["anchor"] for score in scores])
    candidate = np.concatenate([score["candidate"] for score in scores])
    keys = runtime.pl.concat([score["keys"] for score in scores], how="vertical")
    pooled = engine.compare_anchor_candidate(truth, anchor, candidate)["overall"]
    bootstrap = _paired_segment_bootstrap(runtime, scores, config)
    anchor_fp = int(pooled["anchor"]["fp"])
    candidate_fp = int(pooled["candidate"]["fp"])
    exposure_days = int(np.sum(truth == 0)) / 144.0
    if anchor_fp == 0:
        fp_relative_for_gate = 0.0 if candidate_fp == 0 else float("inf")
        fp_relative = 0.0 if candidate_fp == 0 else None
    else:
        fp_relative_for_gate = candidate_fp / anchor_fp - 1.0
        fp_relative = fp_relative_for_gate
    supported_min = config["outer_decision"]["go_replication_if_all"][
        "supported_station_layer_positive_rows_min"
    ]
    station = keys.get_column("station").cast(runtime.pl.String).to_numpy()
    layer = keys.get_column("layer").to_numpy()
    slice_deltas: list[float] = []
    for group in sorted(set(zip(station.tolist(), layer.tolist(), strict=True))):
        mask = (station == group[0]) & (layer == group[1])
        if int(np.sum(truth[mask])) < supported_min:
            continue
        comparison = engine.compare_anchor_candidate(truth[mask], anchor[mask], candidate[mask])[
            "overall"
        ]
        slice_deltas.append(float(comparison["f1_delta"]))
    worst_delta_for_gate = min(slice_deltas) if slice_deltas else float("-inf")
    worst_delta = min(slice_deltas) if slice_deltas else None
    gates = config["outer_decision"]["go_replication_if_all"]
    checks = {
        "pooled_delta": pooled["f1_delta"] >= gates["pooled_delta_f1_min"],
        "each_fold_nonnegative": all(
            score["aggregate"]["f1_delta"] >= gates["each_fold_delta_f1_min"] for score in scores
        ),
        "bootstrap_lower": bootstrap["ci90_low"]
        > gates["paired_bootstrap_delta_f1_ci90_lower_strictly_above"],
        "added_precision": pooled["added_precision"] > pooled["anchor"]["f1"] / 2.0,
        "fp_exposure": fp_relative_for_gate
        <= gates["normal_false_positive_per_day_relative_increase_max"],
        "worst_slice": worst_delta_for_gate >= -gates["worst_supported_station_layer_f1_drop_max"],
    }
    result = (
        "GO_REPLICATION_ONLY_NOT_SUBMISSION_READY"
        if all(checks.values())
        else "NO_GO_NEW_ARCHITECTURE"
    )
    return result, {
        "pooled": pooled,
        "bootstrap": bootstrap,
        "normal_exposure_days": exposure_days,
        "anchor_false_positive_per_day": anchor_fp / exposure_days if exposure_days else 0.0,
        "candidate_false_positive_per_day": candidate_fp / exposure_days if exposure_days else 0.0,
        "false_positive_per_day_relative_increase": fp_relative,
        "supported_station_layer_groups": len(slice_deltas),
        "worst_supported_station_layer_f1_delta": worst_delta,
        "gate_checks": checks,
    }


def acquire_attempt_lock(seal: dict[str, Any]) -> None:
    if TERMINAL_PATH.exists():
        raise ContractError("terminal artefact already exists")
    _exclusive_json(
        ATTEMPT_LOCK,
        {
            "schema_version": "p1.tetad_lite.attempt_lock.v1",
            "experiment_id": EXPERIMENT_ID,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "execution_seal_sha256": _sha256(SEAL_PATH),
            "config_sha256": seal["files"]["config"]["sha256"],
        },
    )


def execute_one_shot(config: dict[str, Any], runtime: Any) -> dict[str, Any]:
    """Execute the registered state machine.  Tests inject a no-compute runtime."""

    seal = _load_verified_seal()
    acquire_attempt_lock(seal)
    state = "ATTEMPT_LOCKED"
    try:
        sanity = runtime.sanity_fit()
        if not runtime.sanity_passes(sanity):
            result = {
                "experiment_id": EXPERIMENT_ID,
                "status": "NO_GO_IMPLEMENTATION_GATE",
                "state": "TERMINAL_QA",
                "sanity": sanity,
            }
            _exclusive_json(TERMINAL_PATH, result)
            return result
        state = "SANITY_GATE_PASSED"

        q2_blind = runtime.fit_predict_blind("2025_q2")
        q2_receipt = _commit_blind(q2_blind)
        state = "Q2_BLIND_PREDICTION_COMMITTED"
        q2_truth = runtime.load_truth_after_receipt(q2_blind, q2_receipt)
        threshold, q2_summary, killed = _select_q2_threshold(config, runtime, q2_blind, q2_truth)
        q2_summary["training_receipt"] = _training_receipt(runtime, "2025_q2")
        threshold_path = ARTIFACT_DIR / "q2_threshold_seal.json"
        _exclusive_json(
            threshold_path,
            {
                "schema_version": "p1.tetad_lite.threshold_seal.v1",
                "experiment_id": EXPERIMENT_ID,
                **q2_summary,
            },
        )
        state = "Q2_THRESHOLD_SEALED"
        if killed:
            result = {
                "experiment_id": EXPERIMENT_ID,
                "status": "NO_GO_Q2_REGISTERED_KILL",
                "state": "TERMINAL_QA",
                "sanity": sanity,
                "q2": q2_summary,
            }
            _exclusive_json(TERMINAL_PATH, result)
            return result

        fold_scores: dict[str, dict[str, Any]] = {}
        # Q4 is deliberately not conditioned on the Q3 score.
        for fold in ("2025_q3", "2025_q4"):
            phase = fold.split("_")[-1].upper()
            blind = runtime.fit_predict_blind(fold)
            receipt = _commit_blind(blind)
            state = f"{phase}_BLIND_PREDICTION_COMMITTED"
            truth = runtime.load_truth_after_receipt(blind, receipt)
            fold_scores[fold] = _score_fold(config, runtime, blind, truth, threshold)
            fold_scores[fold]["training_receipt"] = _training_receipt(runtime, fold)
            state = f"{phase}_SCORED"
        status, outer = _outer_decision(runtime, fold_scores, config)
        result = {
            "experiment_id": EXPERIMENT_ID,
            "status": status,
            "state": "TERMINAL_QA",
            "sanity": sanity,
            "q2": q2_summary,
            "confirmation_folds": {
                name: _public_metrics(score) for name, score in fold_scores.items()
            },
            "outer_decision": outer,
            "optimizer_runs": 4,
            "generated_deployment_outputs": 0,
        }
        _exclusive_json(TERMINAL_PATH, result)
        return result
    except BaseException as error:
        technical = {
            "experiment_id": EXPERIMENT_ID,
            "status": "TERMINAL_TECHNICAL_FAILURE_NO_RETRY",
            "state_at_failure": state,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        if not TERMINAL_PATH.exists():
            _exclusive_json(TERMINAL_PATH, technical)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check-only", action="store_true")
    modes.add_argument("--seal-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    config = _load_config()
    if args.check_only:
        print(json.dumps(build_preflight(config), ensure_ascii=False, sort_keys=True))
        return 0
    if args.seal_only:
        preflight = build_preflight(config)
        print(
            json.dumps(create_execution_seal(config, preflight), ensure_ascii=False, sort_keys=True)
        )
        return 0
    runtime = ScientificRuntime(config)
    result = execute_one_shot(config, runtime)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
