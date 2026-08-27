"""Build and independently QA the fixed P1 MS-TCN e150 deployment set.

This is a build-only runner.  It never opens hidden labels and never uploads.
The selected width/epoch/threshold/seeds are frozen by the completed historical
diagnostic.  All supplied 2024-2025 labels fit three fresh models; the supplied
2026 test covariates are used only for label-free inference.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


EXPERIMENT_ID = "p1_mstcn_e150_full_deployment_20260827_v1"
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
KEY_COLUMNS = ("station", "year", "layer", "time")


class ContractError(RuntimeError):
    """Raised when a frozen input, recipe, or output invariant changes."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not replace:
            raise FileExistsError(path)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: Any, *, replace: bool = False) -> None:
    _atomic_bytes(path, _json_bytes(value), replace=replace)


def _atomic_npz(path: Path, **arrays: Any) -> str:
    import numpy as np

    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return _sha256(path)


def _atomic_torch_save(path: Path, value: Any, torch: Any) -> str:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return _sha256(path)


def _file_identity(path: Path, *, base: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    shown = (
        resolved.relative_to(base.resolve()).as_posix()
        if base is not None and resolved.is_relative_to(base.resolve())
        else str(resolved)
    )
    return {
        "path": shown,
        "bytes": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }


def _load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("deployment config experiment identity changed")
    recipe = config.get("recipe", {})
    if recipe != {
        "width": 512,
        "epoch": 150,
        "threshold": 0.8,
        "seeds": [20260827, 20260839, 20260863],
        "source_schedule_horizon_epochs": 300,
        "representation": "raw_three_seed_ensemble_mean",
    }:
        raise ContractError("fixed e150 recipe changed")
    if not all(config.get("prohibitions", {}).values()):
        raise ContractError("a deployment prohibition was disabled")
    if config["deployment"].get("upload_authorized") is not False:
        raise ContractError("this runner cannot authorize upload")
    return config


def _verify_relative_inputs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    for name, expected in config["relative_inputs"].items():
        path = (ROOT / expected["path"]).resolve(strict=True)
        if not path.is_relative_to(ROOT.resolve()):
            raise ContractError(f"relative input escapes repository: {name}")
        identity = _file_identity(path, base=ROOT)
        for field in ("path", "sha256"):
            if identity[field] != expected[field]:
                raise ContractError(f"pinned relative input changed: {name}.{field}")
        if "rows" in expected:
            import pyarrow.parquet as pq

            rows = int(pq.ParquetFile(path).metadata.num_rows)
            if rows != int(expected["rows"]):
                raise ContractError(f"pinned row count changed: {name}")
            identity["rows"] = rows
        observed[name] = identity
    return observed


def _verify_external(
    path: Path,
    expected: dict[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    identity = _file_identity(resolved)
    if identity["sha256"] != expected["sha256"]:
        raise ContractError(f"pinned external input changed: {name}")
    return identity


def _load_source(config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    record = config["relative_inputs"]["source_runner"]
    path = ROOT / record["path"]
    name = f"{EXPERIMENT_ID}_source"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ContractError("cannot load frozen MS-TCN source runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    source_config = json.loads(
        (ROOT / config["relative_inputs"]["source_config"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    return module, source_config


def _keys_equal(left: Any, right: Any) -> bool:
    import pandas as pd

    if len(left) != len(right):
        return False
    return pd.MultiIndex.from_frame(left.loc[:, KEY_COLUMNS].astype(str)).equals(
        pd.MultiIndex.from_frame(right.loc[:, KEY_COLUMNS].astype(str))
    )


def _preflight(
    *,
    data_dir: Path,
    current_router: Path,
    third_candidate: Path,
    delivery_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import pandas as pd
    import pyarrow.parquet as pq

    config = _load_config()
    relative = _verify_relative_inputs(config)
    expected_external = config["external_inputs"]
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"
    external = {
        "test_csv": _verify_external(test_path, expected_external["test_csv"], name="test_csv"),
        "sample_submission": _verify_external(
            sample_path, expected_external["sample_submission"], name="sample_submission"
        ),
        "current_router": _verify_external(
            current_router, expected_external["current_router"], name="current_router"
        ),
        "third_candidate": _verify_external(
            third_candidate, expected_external["third_candidate"], name="third_candidate"
        ),
    }
    for name, path in {
        "test_csv": test_path,
        "sample_submission": sample_path,
        "current_router": current_router,
        "third_candidate": third_candidate,
    }.items():
        rows = sum(1 for _ in path.open("rb")) - 1
        if rows != int(expected_external[name]["rows"]):
            raise ContractError(f"external row count changed: {name}")
        external[name]["rows"] = rows

    source, source_config = _load_source(config)
    train_meta = json.loads(
        (ROOT / config["relative_inputs"]["train_feature_metadata"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    test_meta = json.loads(
        (ROOT / config["relative_inputs"]["test_feature_metadata"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    if train_meta["feature_columns"] != test_meta["feature_columns"]:
        raise ContractError("train/test feature cache columns differ")
    if train_meta["parquet_sha256"] != relative["train_features"]["sha256"]:
        raise ContractError("train metadata/cache digest mismatch")
    if test_meta["parquet_sha256"] != relative["test_features"]["sha256"]:
        raise ContractError("test metadata/cache digest mismatch")
    if test_meta["source_sha256"] != external["test_csv"]["sha256"]:
        raise ContractError("test cache was not built from the pinned test input")
    numeric_names, projected_columns, dependency = source._feature_dependency_audit(
        train_meta, source_config
    )
    if len(numeric_names) != 74 or len(projected_columns) != 76:
        raise ContractError("projected feature contract changed")
    test_schema = pq.ParquetFile(ROOT / config["relative_inputs"]["test_features"]["path"]).schema_arrow
    if "label" in test_schema.names or "anomaly_type" in test_schema.names:
        raise ContractError("test feature cache unexpectedly contains target columns")

    delivery_set = delivery_root / config["outputs"]["delivery_set_name"]
    preflight = {
        "schema_version": "p1.mstcn_e150_full_deployment.preflight.v1",
        "experiment_id": EXPERIMENT_ID,
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "config": _file_identity(CONFIG_PATH, base=ROOT),
        "runner": _file_identity(Path(__file__), base=ROOT),
        "relative_inputs": relative,
        "external_inputs": external,
        "numeric_feature_count": len(numeric_names),
        "projected_columns": list(projected_columns),
        "dependency_receipt": dependency,
        "test_target_columns_read": 0,
        "upload_performed": False,
        "artifact_dir": config["outputs"]["artifact_dir"],
        "delivery_set": str(delivery_set.resolve()),
        "delivery_set_available": not delivery_set.exists(),
        "result": "PASS",
    }
    context = {
        "config": config,
        "source": source,
        "source_config": source_config,
        "numeric_names": numeric_names,
        "projected_columns": projected_columns,
        "test_path": test_path,
        "sample_path": sample_path,
        "delivery_set": delivery_set,
    }
    return preflight, context


def check_only(
    *, data_dir: Path, current_router: Path, third_candidate: Path, delivery_root: Path
) -> dict[str, Any]:
    preflight, _context = _preflight(
        data_dir=data_dir,
        current_router=current_router,
        third_candidate=third_candidate,
        delivery_root=delivery_root,
    )
    return preflight


def _load_surfaces(context: dict[str, Any], current_router: Path) -> tuple[Any, Any, Any, Any, Any]:
    import numpy as np
    import pandas as pd

    config = context["config"]
    source = context["source"]
    projected = list(context["projected_columns"])
    numeric_names = tuple(context["numeric_names"])
    train_features = pd.read_parquet(
        ROOT / config["relative_inputs"]["train_features"]["path"], columns=projected
    )
    test_features = pd.read_parquet(
        ROOT / config["relative_inputs"]["test_features"]["path"], columns=projected
    )
    train_targets = pd.read_parquet(
        ROOT / config["relative_inputs"]["train_targets"]["path"],
        columns=[*KEY_COLUMNS, "label", "anomaly_type"],
    )
    sidecar_record = context["source_config"]["immutable_inputs"]["feature_key_sidecar"]
    sidecar_path = ROOT / sidecar_record["path"]
    if _sha256(sidecar_path) != sidecar_record["sha256"]:
        raise ContractError("train feature key sidecar changed")
    sidecar = pd.read_parquet(sidecar_path, columns=["ordinal", *KEY_COLUMNS])
    if not np.array_equal(sidecar["ordinal"].to_numpy(), np.arange(len(sidecar))):
        raise ContractError("train feature sidecar ordinal changed")
    if not _keys_equal(sidecar, train_targets):
        raise ContractError("train target/feature keys are not aligned")
    labels = train_targets["label"].to_numpy(dtype=np.int8)
    if not np.isin(labels, [0, 1]).all():
        raise ContractError("training labels are not binary")

    test_keys = pd.read_csv(context["test_path"], usecols=list(KEY_COLUMNS))
    sample = pd.read_csv(context["sample_path"])
    router = pd.read_csv(current_router)
    if not _keys_equal(test_keys, sample) or not _keys_equal(test_keys, router):
        raise ContractError("test/sample/current-Router ordered keys differ")
    if not np.isin(router["label"].to_numpy(), [0, 1]).all():
        raise ContractError("current Router labels are not binary")
    if not np.array_equal(
        train_features["station"].astype(str).to_numpy(),
        train_targets["station"].astype(str).to_numpy(),
    ):
        raise ContractError("train cached station is not aligned")
    if not np.array_equal(
        test_features["station"].astype(str).to_numpy(), test_keys["station"].astype(str).to_numpy()
    ):
        raise ContractError("test cached station is not aligned")

    def make_surface(features: Any, keys: Any, **targets: Any) -> Any:
        return source.RowSurface(
            keys=keys.loc[:, KEY_COLUMNS].reset_index(drop=True),
            numeric=features.loc[:, numeric_names].to_numpy(dtype=np.float32),
            station=features["station"].astype(str).to_numpy(),
            layer_category=features["layer_category"].astype(str).to_numpy(),
            depth_regime=None,
            depth=features["depth_raw"].to_numpy(dtype=np.float32),
            **targets,
        )

    train = make_surface(
        train_features,
        train_targets,
        labels=labels,
        anomaly_type=train_targets["anomaly_type"].fillna("").astype(str).to_numpy(),
    )
    test = make_surface(
        test_features,
        test_keys,
        anchor=router["label"].to_numpy(dtype=np.int8),
    )
    encoder, encoded = source._fit_encoder_and_transform(
        train,
        [test],
        fit_ids=np.arange(train.rows, dtype=np.int64),
        forbidden_ids=np.asarray([], dtype=np.int64),
        numeric_names=numeric_names,
    )
    training, holdout = encoded
    if training.features.shape[1] != 165 or holdout.features.shape[1] != 165:
        raise ContractError("full deployment input width changed")
    return encoder, training, holdout, sample, router


def _seed_paths(artifact_dir: Path, seed: int) -> dict[str, Path]:
    stem = f"full_width_512_seed_{seed}_epoch_150"
    return {
        "checkpoint": artifact_dir / f"{stem}_state.pt",
        "prediction": artifact_dir / f"{stem}_test_prediction.npz",
        "history": artifact_dir / f"{stem}_history.json",
        "receipt": artifact_dir / f"{stem}_receipt.json",
    }


def _load_completed_seed(
    paths: dict[str, Path], *, seed: int, rows: int, config_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    import numpy as np

    if not paths["receipt"].is_file():
        return None
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    if not (
        receipt.get("schema_version") == "p1.mstcn_e150_full_deployment.seed.v1"
        and receipt.get("seed") == seed
        and receipt.get("epoch") == 150
        and receipt.get("config_sha256") == config_sha256
    ):
        raise ContractError(f"completed seed receipt changed: {seed}")
    for name in ("checkpoint", "prediction", "history"):
        identity = receipt[f"{name}_artifact"]
        path = paths[name]
        if not path.is_file() or _sha256(path) != identity["sha256"]:
            raise ContractError(f"completed seed artifact changed: {seed}.{name}")
    with np.load(paths["prediction"], allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    expected = {
        "row_probability": (rows,),
        "boundary_probability": (rows, 2),
        "type_probability": (rows, 5),
    }
    if {name: tuple(value.shape) for name, value in arrays.items()} != expected:
        raise ContractError(f"completed seed prediction shape changed: {seed}")
    return arrays, receipt


def _fit_seed(
    source: Any,
    source_config: dict[str, Any],
    training: Any,
    holdout: Any,
    *,
    seed: int,
    device: Any,
    artifact_dir: Path,
    config_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    _np, _pd, torch, _model_api, _data_api = source._load_scientific()
    paths = _seed_paths(artifact_dir, seed)
    completed = _load_completed_seed(
        paths, seed=seed, rows=holdout.surface.rows, config_sha256=config_sha256
    )
    if completed is not None:
        print(json.dumps({"event": "seed_reused", "seed": seed}), flush=True)
        return completed

    capacity = source._config_for_capacity(source_config, width=512, seed=seed)
    if int(capacity["training"]["maximum_epochs"]) != 300:
        raise ContractError("source schedule horizon changed")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    source._reset_cuda_peak_memory(torch, device)
    model = source._new_model(training.features.shape[1], capacity, device)
    expected_parameters = int(
        source_config["architecture"]["exact_parameter_count_by_width_at_input_165"]["512"]
    )
    if int(model.trainable_parameter_count) != expected_parameters:
        raise ContractError("width-512 parameter count changed")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(capacity["training"]["learning_rate"]),
        weight_decay=float(capacity["training"]["weight_decay"]),
    )
    windows = source._selected_windows(training, capacity)
    holdout_windows = source._all_windows(holdout, capacity)
    positive_weight = source._positive_weight(training.surface.labels)
    _steps, total_steps, _warmup = source._schedule_geometry(
        capacity, window_count=len(windows)
    )
    global_step = 0
    history: list[dict[str, Any]] = []
    started_seed = time.perf_counter()
    for epoch in range(1, 151):
        started = time.perf_counter()
        telemetry, global_step, learning_rate = source._train_epoch(
            model,
            optimizer,
            training,
            windows,
            config=capacity,
            positive_weight=positive_weight,
            device=device,
            epoch=epoch,
            global_step=global_step,
            total_steps=total_steps,
        )
        record = source._history_record(
            epoch=epoch,
            telemetry=telemetry,
            global_step=global_step,
            learning_rate=learning_rate,
            elapsed_seconds=time.perf_counter() - started,
        )
        history.append(record)
        if epoch == 1 or epoch % 5 == 0:
            _atomic_json(paths["history"], history, replace=True)
            print(
                json.dumps(
                    {
                        "event": "training_progress",
                        "seed": seed,
                        "epoch": epoch,
                        "epochs": 150,
                        "loss": record["total_loss"],
                        "epoch_seconds": record["epoch_wall_seconds"],
                    },
                    allow_nan=False,
                ),
                flush=True,
            )
    prediction = source.predict_encoded(
        model,
        holdout,
        holdout_windows,
        batch_size=int(capacity["training"]["batch_size"]),
        device=device,
    )
    checkpoint_sha = _atomic_torch_save(
        paths["checkpoint"],
        {
            "schema_version": "p1.mstcn_e150_full_deployment.state.v1",
            "experiment_id": EXPERIMENT_ID,
            "width": 512,
            "seed": seed,
            "epoch": 150,
            "source_schedule_horizon_epochs": 300,
            "input_features": int(training.features.shape[1]),
            "parameter_count": int(model.trainable_parameter_count),
            "state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
        },
        torch,
    )
    arrays = {
        "row_probability": prediction.row_probability.astype(np.float32, copy=False),
        "boundary_probability": prediction.boundary_probability.astype(np.float32, copy=False),
        "type_probability": prediction.type_probability.astype(np.float32, copy=False),
    }
    prediction_sha = _atomic_npz(paths["prediction"], **arrays)
    _atomic_json(paths["history"], history, replace=True)
    receipt = {
        "schema_version": "p1.mstcn_e150_full_deployment.seed.v1",
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": config_sha256,
        "seed": seed,
        "width": 512,
        "epoch": 150,
        "source_schedule_horizon_epochs": 300,
        "training_rows": int(training.surface.rows),
        "test_rows": int(holdout.surface.rows),
        "training_windows": len(windows),
        "test_windows": len(holdout_windows),
        "positive_weight": float(positive_weight),
        "optimizer_steps": int(global_step),
        "parameter_count": int(model.trainable_parameter_count),
        "wall_seconds": float(time.perf_counter() - started_seed),
        "nonfinite_count_total": int(sum(int(row["nonfinite_count"]) for row in history)),
        "checkpoint_artifact": {
            **_file_identity(paths["checkpoint"], base=artifact_dir),
            "sha256": checkpoint_sha,
        },
        "prediction_artifact": {
            **_file_identity(paths["prediction"], base=artifact_dir),
            "sha256": prediction_sha,
        },
        "history_artifact": _file_identity(paths["history"], base=artifact_dir),
        **source._cuda_peak_memory_receipt(torch, device),
    }
    _atomic_json(paths["receipt"], receipt)
    del optimizer, model, prediction
    gc.collect()
    torch.cuda.empty_cache()
    return arrays, receipt


def _write_csv_atomic(frame: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", prefix=f".{path.name}.",
            suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            frame.to_csv(handle, index=False, lineterminator="\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _qa_candidate(candidate: Any, sample: Any, anchor: Any, *, name: str) -> dict[str, Any]:
    import numpy as np

    if list(candidate.columns) != [*KEY_COLUMNS, "label"]:
        raise ContractError(f"candidate schema changed: {name}")
    if len(candidate) != 169011 or not _keys_equal(candidate, sample):
        raise ContractError(f"candidate keys changed: {name}")
    labels = candidate["label"].to_numpy()
    anchor_bits = anchor["label"].to_numpy(dtype=np.int8)
    if not np.isin(labels, [0, 1]).all():
        raise ContractError(f"candidate labels are not binary: {name}")
    bits = labels.astype(np.int8)
    removed = int(np.sum((anchor_bits == 1) & (bits == 0)))
    if removed:
        raise ContractError(f"candidate removed Router positives: {name}")
    stations = candidate["station"].astype(str).to_numpy()
    return {
        "rows": len(candidate),
        "positive_rows": int(bits.sum()),
        "anchor_positive_rows": int(anchor_bits.sum()),
        "added_rows": int(np.sum((anchor_bits == 0) & (bits == 1))),
        "anchor_positive_removed_rows": removed,
        "added_rows_by_station": {
            station: int(np.sum((stations == station) & (anchor_bits == 0) & (bits == 1)))
            for station in sorted(set(stations))
        },
    }


def _package(
    *,
    config: dict[str, Any],
    delivery_set: Path,
    candidate_all_path: Path,
    candidate_gs_path: Path,
    third_candidate: Path,
    current_router: Path,
    qa: dict[str, Any],
) -> dict[str, Any]:
    if delivery_set.exists():
        raise FileExistsError(delivery_set)
    delivery_set.mkdir(parents=True)
    entries = [
        (
            config["outputs"]["candidate_directories"][0],
            candidate_all_path,
            "P1 MS-TCN e150 전 정점 Router 보존 결합 v1",
            "전체 학습 3-seed MS-TCN++/ASRF의 장기 이상 제안을 현 최고 Router에 OR 결합해 기존 양성을 모두 보존합니다.",
        ),
        (
            config["outputs"]["candidate_directories"][1],
            candidate_gs_path,
            "P1 MS-TCN e150 G·S 선택 Router 보존 결합 v1",
            "동일 e150 앙상블 추가를 G-ORS·S-ORS에만 적용하고 I-ORS는 현 최고 Router 판정을 유지합니다.",
        ),
        (
            config["outputs"]["candidate_directories"][2],
            third_candidate,
            "P1 G·I 추가양성 결합·무제거 v1",
            "현 베스트 B에 G/I 추가양성 217행을 함께 복원하되 Router의 12행 제거는 적용하지 않습니다.",
        ),
    ]
    manifest_entries: list[dict[str, Any]] = []
    for index, (directory, source, title, summary) in enumerate(entries, start=1):
        target_dir = delivery_set / directory
        target_dir.mkdir()
        target = target_dir / "P1_submission.csv"
        shutil.copy2(source, target)
        identity = _file_identity(target, base=delivery_set)
        info = (
            f"제출물 제목: {title}\n"
            f"한줄요약(접근방식): {summary}\n"
            "문제: P1\n"
            f"제출 순서: {index}/3\n"
            "CSV 파일: P1_submission.csv\n"
            f"행 수: 169011\n"
            f"SHA-256: {identity['sha256']}\n"
            "승인 경계: 이 패키지 생성은 업로드 승인이 아니며 실제 업로드 직전에 별도 명시 승인이 필요합니다.\n"
        )
        _atomic_bytes(target_dir / "P1_제출정보.txt", info.encode("utf-8"))
        manifest_entries.append(
            {"order": index, "directory": directory, "title": title, "summary": summary, **identity}
        )

    backup = delivery_set / "backup_best_before_round_F"
    backup.mkdir()
    backup_target = backup / "P1_submission.csv"
    shutil.copy2(current_router, backup_target)
    manifest = {
        "schema_version": "p1.mstcn_e150_full_deployment.delivery.v1",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "submission_order": manifest_entries,
        "backup_current_router": _file_identity(backup_target, base=delivery_set),
        "independent_qa": qa,
        "upload_authorized": False,
        "upload_performed": False,
    }
    _atomic_json(delivery_set / "SET_MANIFEST.json", manifest)
    _atomic_json(delivery_set / "INDEPENDENT_QA.json", qa)
    return manifest


def execute(
    *,
    expected_runner_sha256: str,
    data_dir: Path,
    current_router: Path,
    third_candidate: Path,
    delivery_root: Path,
) -> dict[str, Any]:
    import numpy as np

    runner_sha = _sha256(Path(__file__))
    if expected_runner_sha256.casefold() != runner_sha:
        raise ContractError("--expected-runner-sha256 does not match reviewed runner")
    preflight, context = _preflight(
        data_dir=data_dir,
        current_router=current_router,
        third_candidate=third_candidate,
        delivery_root=delivery_root,
    )
    config = context["config"]
    artifact_dir = ROOT / config["outputs"]["artifact_dir"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    lock_path = artifact_dir / "execution_lock.json"
    lock = {
        "schema_version": "p1.mstcn_e150_full_deployment.lock.v1",
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": preflight["config"]["sha256"],
        "runner_sha256": runner_sha,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "upload_authorized": False,
    }
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if any(existing.get(key) != lock[key] for key in ("experiment_id", "config_sha256", "runner_sha256")):
            raise ContractError("existing execution lock differs from current build")
    else:
        _atomic_json(lock_path, lock)
    _atomic_json(artifact_dir / "preflight.json", preflight, replace=True)

    source = context["source"]
    _np, _pd, torch, _model_api, _data_api = source._load_scientific()
    if not torch.cuda.is_available():
        raise ContractError("CUDA is required for the full deployment refit")
    device = torch.device("cuda")
    print(json.dumps({"event": "loading_surfaces"}), flush=True)
    encoder, training, holdout, sample, router = _load_surfaces(context, current_router)
    encoder_receipt = source._encoder_receipt(encoder)
    _atomic_json(artifact_dir / "full_encoder.json", encoder_receipt, replace=True)
    print(
        json.dumps(
            {
                "event": "surfaces_ready",
                "training_rows": training.surface.rows,
                "test_rows": holdout.surface.rows,
                "input_features": int(training.features.shape[1]),
                "device": torch.cuda.get_device_name(device),
            }
        ),
        flush=True,
    )
    row_sum = np.zeros(holdout.surface.rows, dtype=np.float32)
    boundary_sum = np.zeros((holdout.surface.rows, 2), dtype=np.float32)
    type_sum = np.zeros((holdout.surface.rows, 5), dtype=np.float32)
    receipts: list[dict[str, Any]] = []
    for seed in config["recipe"]["seeds"]:
        arrays, receipt = _fit_seed(
            source,
            context["source_config"],
            training,
            holdout,
            seed=int(seed),
            device=device,
            artifact_dir=artifact_dir,
            config_sha256=preflight["config"]["sha256"],
        )
        row_sum += arrays["row_probability"]
        boundary_sum += arrays["boundary_probability"]
        type_sum += arrays["type_probability"]
        receipts.append(receipt)
    divisor = float(len(receipts))
    bundle = source.PredictionBundle(row_sum / divisor, boundary_sum / divisor, type_sum / divisor)
    proposal = source.decode_long_event_segments(
        source._decoder_row_probability(bundle, context["source_config"]),
        bundle.boundary_probability,
        holdout.layout,
        high_threshold=float(config["recipe"]["threshold"]),
        snap_radius=int(context["source_config"]["decoder"]["boundary_peak_snap_radius_rows"]),
        minimum_rows=int(context["source_config"]["decoder"]["minimum_added_segment_rows"]),
        maximum_rows=source._maximum_segment_rows(context["source_config"]),
    ).astype(np.int8)
    anchor = router["label"].to_numpy(dtype=np.int8)
    all_bits = source.anchor_preserving_union(anchor, proposal).astype(np.int8)
    allowed = np.isin(
        sample["station"].astype(str).to_numpy(), config["deployment"]["allowed_gs_stations"]
    )
    gs_bits = source.anchor_preserving_union(anchor, proposal & allowed).astype(np.int8)
    base = sample.loc[:, KEY_COLUMNS].copy()
    candidate_all = base.assign(label=all_bits)
    candidate_gs = base.assign(label=gs_bits)
    qa = {
        "schema_version": "p1.mstcn_e150_full_deployment.qa.v1",
        "experiment_id": EXPERIMENT_ID,
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "recipe": config["recipe"],
        "training_rows": int(training.surface.rows),
        "test_rows": int(holdout.surface.rows),
        "test_target_columns_read": 0,
        "proposal_rows": int(proposal.sum()),
        "candidate_all": _qa_candidate(candidate_all, sample, router, name="all"),
        "candidate_gs": _qa_candidate(candidate_gs, sample, router, name="gs"),
        "seed_receipts": receipts,
        "upload_performed": False,
    }
    candidate_all_path = artifact_dir / "P1_MSTCN_E150_ROUTER_UNION_ALL.csv"
    candidate_gs_path = artifact_dir / "P1_MSTCN_E150_ROUTER_UNION_GS_ONLY.csv"
    if not candidate_all_path.exists():
        _write_csv_atomic(candidate_all, candidate_all_path)
    if not candidate_gs_path.exists():
        _write_csv_atomic(candidate_gs, candidate_gs_path)
    qa["candidate_all"]["artifact"] = _file_identity(candidate_all_path, base=artifact_dir)
    qa["candidate_gs"]["artifact"] = _file_identity(candidate_gs_path, base=artifact_dir)
    _atomic_json(artifact_dir / "independent_qa.json", qa, replace=True)
    manifest = _package(
        config=config,
        delivery_set=context["delivery_set"],
        candidate_all_path=candidate_all_path,
        candidate_gs_path=candidate_gs_path,
        third_candidate=third_candidate,
        current_router=current_router,
        qa=qa,
    )
    terminal = {
        "schema_version": "p1.mstcn_e150_full_deployment.terminal.v1",
        "experiment_id": EXPERIMENT_ID,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "status": "BUILD_AND_QA_PASS_NOT_UPLOADED",
        "device": torch.cuda.get_device_name(device),
        "recipe": config["recipe"],
        "artifact_dir": str(artifact_dir.resolve()),
        "delivery_set": str(context["delivery_set"].resolve()),
        "candidate_hashes": {
            "all": qa["candidate_all"]["artifact"]["sha256"],
            "gs": qa["candidate_gs"]["artifact"]["sha256"],
            "third": manifest["submission_order"][2]["sha256"],
        },
        "upload_authorized": False,
        "upload_performed": False,
    }
    _atomic_json(artifact_dir / "terminal_result.json", terminal)
    print(json.dumps({"event": "completed", **terminal}, ensure_ascii=False), flush=True)
    return terminal


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-runner-sha256")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--current-router", type=Path, required=True)
    parser.add_argument("--third-candidate", type=Path, required=True)
    parser.add_argument("--delivery-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    common = {
        "data_dir": args.data_dir,
        "current_router": args.current_router,
        "third_candidate": args.third_candidate,
        "delivery_root": args.delivery_root,
    }
    if args.check_only:
        result = check_only(**common)
    else:
        if not args.expected_runner_sha256:
            raise ContractError("--execute requires --expected-runner-sha256")
        result = execute(expected_runner_sha256=args.expected_runner_sha256, **common)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
