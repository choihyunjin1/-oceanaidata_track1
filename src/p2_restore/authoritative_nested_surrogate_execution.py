"""Sealed execution engine for the P2 authoritative nested surrogate.

The module is import-safe: importing or dry-running it never trains a model.
Actual training requires a separate command/source/config seal and an explicit
authorization receipt.  The engine reads only ``observations.csv`` and never
has a submission-generation path.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor

from p2_restore.architecture_matched_stage_a_execution_v2 import (
    RouterContext,
    _build_router_context,
    _compose_prediction,
    _gate_receipt,
    _joint_masked_panel,
)
from p2_restore.authoritative_nested_surrogate_conformance import (
    COMPONENTS,
    DEEP_COMPONENTS,
    KEY_COLUMNS,
    TARGET_LAYERS,
    InnerFoldPlan,
    PrefixPlan,
    adapt_panel_for_full_prefix,
    adapt_panel_for_inner_fold,
    build_all_prefix_plans,
    child_seed,
    fit_nnls_stack,
    joint_mask_target_observations,
    merge_component_oof,
    middle_epoch,
)
from p2_restore.deep_data import P2Panel, PanelNormalizer, make_chunk_bounds, time_block_mask
from p2_restore.deep_models import ConditionalDiffusion, build_model, count_parameters
from p2_restore.deep_training import (
    FoldTrainingResult,
    FullTrainingResult,
    TrainingConfig,
    _device,
    _materialize_chunks,
    _predict_chunks,
    _rmse,
    _validation_oof,
    set_deterministic_seed,
    train_full_model,
)
from p2_restore.features import FeatureTable
from p2_restore.final_inference import csv_float_roundtrip
from p2_restore.matched_budget_compare import (
    LocalContext,
    build_bootstrap_plan,
    build_local_context,
    complementarity_report,
    materialize_settings,
    metric_report,
    paired_day_bootstrap,
    prepare_forward_surrogate_surface,
)
from p2_restore.max_rounds import MaxRoundRouterModel
from p2_restore.model import P2Model
from p2_restore.profile_projection import public_endpoint_frame
from p2_restore.public_layer_causal_residual import CausalResidualSpec
from p2_restore.regime_gate import STATE_FEATURES, SoftRegimeGate, fit_soft_gate
from p2_restore.state_conditional import (
    StateConditionalLeanModel,
    compute_state_partition,
)

PREDICTION_COLUMNS = tuple(f"pred_{component}" for component in COMPONENTS)
OUTER_KEY_COLUMNS = ("fold", "station", "layer", "time")
FAMILY_SETTINGS = (
    "INCUMBENT_NOOP",
    "STACK_W0500",
    "STACK_W0625",
    "STACK_W0750",
    "CAUSAL_RESIDUAL_SCALE025",
    "FALLBACK_BLEND50_A0625",
    "CAUSAL_ON_FALLBACK",
)
ROUTER_FEATURE_COUNTS = {"base": 41, "lean": 61, "phase": 81}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _dataframe_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def _joblib_bytes(value: Any) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(value, buffer, compress=3)
    return buffer.getvalue()


def _torch_bytes(value: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _safe_job_id(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")
    _require(value != "" and all(character in allowed for character in value), "unsafe job id")
    return value


def atomic_write_or_verify(path: Path, payload: bytes) -> dict[str, Any]:
    """Publish deterministic final bytes without ever exposing a partial final.

    A pre-existing final is reused only after exact byte/hash verification.  A
    failed publication deliberately preserves its uniquely named partial for
    audit; later resumes ignore old partials and create a fresh one.  Successful
    ``os.rename`` consumes the current partial atomically.
    """

    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_sha = hashlib.sha256(payload).hexdigest()
    expected_bytes = len(payload)

    def verify_final() -> None:
        _require(target.is_file(), "atomic output target is not a regular file")
        _require(target.stat().st_size == expected_bytes, "atomic final size changed")
        _require(sha256_file(target) == expected_sha, "atomic final hash changed")

    if target.exists():
        verify_final()
        return {
            "status": "REUSED_VERIFIED_FINAL",
            "sha256": expected_sha,
            "bytes": expected_bytes,
            "partial_created": False,
            "partial_policy": "PREEXISTING_PARTIALS_IGNORED_AND_PRESERVED_FOR_AUDIT",
        }
    partial = target.parent / (
        f".{target.name}.partial.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        with partial.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.rename(partial, target)
        except FileExistsError:
            verify_final()
            return {
                "status": "RACING_FINAL_MATCHED_PARTIAL_PRESERVED",
                "sha256": expected_sha,
                "bytes": expected_bytes,
                "partial_created": True,
                "partial_preserved": str(partial.name),
                "partial_policy": "FAILED_OR_RACING_PARTIAL_PRESERVED_FOR_AUDIT",
            }
        verify_final()
        return {
            "status": "COMMITTED_BY_FSYNC_AND_ATOMIC_RENAME",
            "sha256": expected_sha,
            "bytes": expected_bytes,
            "partial_created": True,
            "partial_consumed_by_rename": True,
            "partial_policy": "SUCCESSFUL_CURRENT_PARTIAL_CONSUMED_OLD_PARTIALS_PRESERVED",
        }
    except Exception:
        # Never unlink a failed partial: its unique path is an audit receipt and
        # is never considered a completed final on resume.
        raise


@dataclass(frozen=True)
class JobProduct:
    frame: pd.DataFrame
    receipt: dict[str, Any]
    artifacts: dict[str, bytes]


class JobStore:
    """Content-addressed complete-job reuse with atomic directory publication."""

    def __init__(self, root: Path, *, contract_sha256: str) -> None:
        self.root = root.resolve()
        self.contract_sha256 = contract_sha256
        self.root.mkdir(parents=True, exist_ok=True)
        self.new_jobs = 0
        self.reused_jobs = 0

    def _load(self, directory: Path, job_id: str) -> JobProduct:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        _require(manifest["job_id"] == job_id, "job id changed")
        _require(manifest["contract_sha256"] == self.contract_sha256, "job contract changed")
        for name, pin in manifest["files"].items():
            path = (directory / name).resolve(strict=True)
            _require(path.parent == directory.resolve(), "job file escaped directory")
            _require(sha256_file(path) == pin["sha256"], f"job file hash changed: {name}")
            _require(path.stat().st_size == int(pin["bytes"]), f"job file size changed: {name}")
        frame = pd.read_parquet(directory / "prediction.parquet")
        receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
        artifacts = {name: (directory / name).read_bytes() for name in manifest["payload_files"]}
        return JobProduct(frame, receipt, artifacts)

    def materialize(self, job_id: str, factory: Callable[[], JobProduct]) -> JobProduct:
        identifier = _safe_job_id(job_id)
        target = self.root / identifier
        if target.is_dir():
            self.reused_jobs += 1
            return self._load(target, identifier)
        if target.exists():
            raise FileExistsError(f"job target is not a directory: {identifier}")
        partial = self.root / f".{identifier}.partial.{os.getpid()}.{uuid.uuid4().hex}"
        os.mkdir(partial)
        try:
            product = factory()
            files = {
                "prediction.parquet": _dataframe_bytes(product.frame),
                "receipt.json": _json_bytes(product.receipt),
                **product.artifacts,
            }
            pins: dict[str, Any] = {}
            for name, payload in files.items():
                _require(Path(name).name == name, "nested job payload name is forbidden")
                path = partial / name
                with path.open("xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                pins[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            manifest = {
                "schema_version": "p2_authoritative_nested_surrogate_job.v1",
                "job_id": identifier,
                "contract_sha256": self.contract_sha256,
                "files": pins,
                "payload_files": sorted(product.artifacts),
                "complete": True,
            }
            with (partial / "manifest.json").open("xb") as handle:
                handle.write(_json_bytes(manifest))
                handle.flush()
                os.fsync(handle.fileno())
            os.rename(partial, target)
            self.new_jobs += 1
            return self._load(target, identifier)
        except Exception:
            # The unique partial directory is deliberately preserved for audit.
            raise


@contextmanager
def process_lock(path: Path):
    """Hold a non-destructive OS lock; the lock file itself is never deleted."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, io.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("another P2 authoritative execution holds the lock") from error
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError("another P2 authoritative execution holds the lock") from error
        yield
    finally:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _normalize_component_frame(
    frame: pd.DataFrame,
    *,
    component: str,
    inner_fold: str | None,
    fold: str,
) -> pd.DataFrame:
    required = {"station", "layer", "time", "truth", "prediction"}
    _require(required.issubset(frame), f"{component} component frame is incomplete")
    result = frame.loc[:, ["station", "layer", "time", "truth", "prediction"]].copy()
    result["station"] = result["station"].astype(str)
    result["layer"] = result["layer"].astype(int)
    result["time"] = pd.to_datetime(result["time"], utc=True).astype(str)
    result["truth"] = result["truth"].astype(float)
    result["prediction"] = result["prediction"].astype(float)
    _require(np.isfinite(result[["truth", "prediction"]]).all().all(), "nonfinite component")
    if inner_fold is not None:
        result.insert(0, "inner_fold", inner_fold)
        order = [*KEY_COLUMNS]
    else:
        result.insert(0, "fold", fold)
        order = [*OUTER_KEY_COLUMNS]
    return result.sort_values(order, kind="stable").reset_index(drop=True)


def _layer_stack_mapping(oof: pd.DataFrame) -> dict[str, dict[str, float]]:
    arrays = fit_nnls_stack(oof, COMPONENTS)
    return {
        str(layer): {
            component: float(weight)
            for component, weight in zip(COMPONENTS, arrays[layer], strict=True)
        }
        for layer in TARGET_LAYERS
    }


def fit_current_scope_meta(
    oof: pd.DataFrame,
    *,
    scope_id: str,
    regularization: float,
) -> tuple[dict[str, dict[str, float]], SoftRegimeGate, dict[str, Any]]:
    required = {"layer", "truth", *COMPONENTS, *STATE_FEATURES, *KEY_COLUMNS}
    _require(required.issubset(oof), "current-scope meta OOF is incomplete")
    stack = _layer_stack_mapping(oof)
    gate = fit_soft_gate(
        oof,
        prediction_columns=COMPONENTS,
        feature_names=STATE_FEATURES,
        regularization=float(regularization),
    )
    receipt = {
        "scope_id": scope_id,
        "oof_rows": len(oof),
        "oof_key_truth_sha256": canonical_sha256(
            oof.loc[:, [*KEY_COLUMNS, "truth"]].astype(str).to_dict("records")
        ),
        "stack_method": "SCIPY_NNLS_THEN_SUM_NORMALIZE_UNIFORM_IF_ALL_ZERO",
        "stack_weights": stack,
        "gate": _gate_receipt(gate),
        "parameter_source": "CURRENT_SCOPE_INNER_OOF_ONLY",
        "frozen_epoch_reused": False,
        "frozen_stack_reused": False,
        "frozen_gate_reused": False,
    }
    return stack, gate, receipt


class ComponentBackend(Protocol):
    def fit_inner(
        self,
        component: str,
        plan: PrefixPlan,
        inner: InnerFoldPlan,
        pipeline_seed: int,
    ) -> JobProduct: ...

    def fit_full(
        self,
        component: str,
        plan: PrefixPlan,
        pipeline_seed: int,
        full_epochs: int | None,
    ) -> JobProduct: ...

    def compose(
        self,
        frame: pd.DataFrame,
        stack: Mapping[str, Mapping[str, float]],
        gate: SoftRegimeGate,
    ) -> tuple[np.ndarray, dict[str, Any]]: ...


def _inner_job_id(plan: PrefixPlan, seed: int, inner: str, component: str) -> str:
    return f"{plan.scope_id}__s{seed}__{inner}__{component}"


def _full_job_id(plan: PrefixPlan, seed: int, component: str) -> str:
    return f"{plan.scope_id}__s{seed}__full__{component}"


def execute_cell_seed(
    *,
    plan: PrefixPlan,
    pipeline_seed: int,
    backend: ComponentBackend,
    jobs: JobStore,
    gate_regularization: float,
) -> JobProduct:
    """Execute or resume one of the 45 seeded cells."""

    inner_by_component: dict[str, list[pd.DataFrame]] = {component: [] for component in COMPONENTS}
    best_epochs: dict[str, list[int]] = {component: [] for component in DEEP_COMPONENTS}
    component_receipts: list[dict[str, Any]] = []
    router_features: list[pd.DataFrame] = []
    for inner in plan.inner_folds:
        for component in COMPONENTS:
            product = jobs.materialize(
                _inner_job_id(plan, pipeline_seed, inner.inner_fold, component),
                lambda component=component, inner=inner: backend.fit_inner(
                    component, plan, inner, pipeline_seed
                ),
            )
            normalized = _normalize_component_frame(
                product.frame,
                component=component,
                inner_fold=inner.inner_fold,
                fold=plan.outer_fold,
            )
            inner_by_component[component].append(normalized)
            component_receipts.append(product.receipt)
            if component in DEEP_COMPONENTS:
                best_epochs[component].append(int(product.receipt["best_epoch"]))
            if component == "router_400":
                feature_columns = [*KEY_COLUMNS, *STATE_FEATURES]
                required_state = {"station", "layer", "time", *STATE_FEATURES}
                _require(required_state.issubset(product.frame), "router state missing")
                state = product.frame.copy()
                state.insert(0, "inner_fold", inner.inner_fold)
                state["time"] = pd.to_datetime(state["time"], utc=True).astype(str)
                router_features.append(
                    state.loc[:, feature_columns]
                    .sort_values(list(KEY_COLUMNS), kind="stable")
                    .reset_index(drop=True)
                )
    component_frames = {
        component: pd.concat(parts, ignore_index=True)
        for component, parts in inner_by_component.items()
    }
    merged, ledger = merge_component_oof(component_frames)
    state = pd.concat(router_features, ignore_index=True)
    _require(
        merged.loc[:, list(KEY_COLUMNS)].equals(state.loc[:, list(KEY_COLUMNS)]),
        "router state does not share the OOF ledger keys",
    )
    for component in COMPONENTS:
        merged[component] = merged.pop(f"pred_{component}")
    merged = pd.concat([merged, state.loc[:, STATE_FEATURES].reset_index(drop=True)], axis=1)
    stack, gate, meta_receipt = fit_current_scope_meta(
        merged,
        scope_id=f"{plan.scope_id}__s{pipeline_seed}",
        regularization=gate_regularization,
    )
    full_epochs = {component: middle_epoch(values) for component, values in best_epochs.items()}
    outer_by_component: dict[str, pd.DataFrame] = {}
    full_receipts: list[dict[str, Any]] = []
    outer_state: pd.DataFrame | None = None
    for component in COMPONENTS:
        product = jobs.materialize(
            _full_job_id(plan, pipeline_seed, component),
            lambda component=component: backend.fit_full(
                component,
                plan,
                pipeline_seed,
                full_epochs.get(component),
            ),
        )
        outer_by_component[component] = _normalize_component_frame(
            product.frame,
            component=component,
            inner_fold=None,
            fold=plan.outer_fold,
        )
        full_receipts.append(product.receipt)
        if component == "router_400":
            needed = [*OUTER_KEY_COLUMNS, *STATE_FEATURES]
            _require(set(needed).issubset(product.frame), "outer router state missing")
            outer_state = product.frame.loc[:, needed].copy()
            outer_state["time"] = pd.to_datetime(outer_state["time"], utc=True).astype(str)
            outer_state = outer_state.sort_values(
                list(OUTER_KEY_COLUMNS), kind="stable"
            ).reset_index(drop=True)
    reference = outer_by_component[COMPONENTS[0]].loc[:, [*OUTER_KEY_COLUMNS, "truth"]]
    outer = reference.copy()
    for component in COMPONENTS:
        current = outer_by_component[component]
        _require(
            current.loc[:, [*OUTER_KEY_COLUMNS, "truth"]].equals(reference),
            f"outer {component} key/truth surface differs",
        )
        outer[component] = current["prediction"].to_numpy(float)
    assert outer_state is not None
    _require(
        outer.loc[:, list(OUTER_KEY_COLUMNS)].equals(outer_state.loc[:, list(OUTER_KEY_COLUMNS)]),
        "outer router state key surface differs",
    )
    outer = pd.concat([outer, outer_state.loc[:, STATE_FEATURES].reset_index(drop=True)], axis=1)
    prediction, postprocess = backend.compose(outer, stack, gate)
    result = outer.loc[:, [*OUTER_KEY_COLUMNS, "truth"]].copy()
    result["prediction"] = prediction
    receipt = {
        "schema_version": "p2_authoritative_nested_surrogate_cell.v1",
        "cell_id": f"{plan.scope_id}__s{pipeline_seed}",
        "outer_fold": plan.outer_fold,
        "prefix_fraction": plan.fraction,
        "pipeline_seed": pipeline_seed,
        "prefix_cutoff_kst": plan.summary()["cutoff_kst"],
        "inner_oof_ledger": ledger,
        "selected_inner_epochs": best_epochs,
        "full_prefix_epochs": full_epochs,
        "meta": meta_receipt,
        "postprocess": postprocess,
        "component_receipts": component_receipts,
        "full_receipts": full_receipts,
        "guards": {
            "joint_temp_psal_mask": True,
            "seven_day_embargo": True,
            "future_or_outer_labels_in_fit": False,
            "current_scope_meta_only": True,
            "frozen_epoch_stack_gate_reuse": False,
        },
    }
    return JobProduct(
        result,
        receipt,
        {"meta.joblib": _joblib_bytes({"stack": stack, "gate": gate})},
    )


def _fit_p2_model_4threads(table: FeatureTable, rows: np.ndarray, *, seed: int) -> P2Model:
    selected = np.asarray(rows, dtype=bool)
    _require(selected.shape == (len(table.frame),) and selected.any(), "router rows invalid")
    estimator = LGBMRegressor(
        objective="regression_l2",
        n_estimators=400,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=7,
        min_child_samples=200,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=1.0,
        random_state=int(seed),
        n_jobs=4,
        verbosity=-1,
        deterministic=True,
        force_row_wise=True,
    )
    estimator.fit(
        table.frame.loc[selected, table.feature_columns],
        table.frame.loc[selected, "residual"],
    )
    return P2Model(estimator, table.feature_columns)


def _fit_state_model_4threads(
    lean: FeatureTable, rows: np.ndarray, *, seed: int
) -> StateConditionalLeanModel:
    partition = compute_state_partition(lean, rows)
    return StateConditionalLeanModel(
        mixed_model=_fit_p2_model_4threads(lean, partition.mixed_rows, seed=seed + 101),
        stratified_model=_fit_p2_model_4threads(lean, partition.stratified_rows, seed=seed + 202),
        q40=partition.q40,
        q60=partition.q60,
        mixed_training_rows=int(partition.mixed_rows.sum()),
        stratified_training_rows=int(partition.stratified_rows.sum()),
    )


def _all_rows(table: FeatureTable) -> np.ndarray:
    return np.ones(len(table.frame), dtype=bool)


def _subset_feature_table(table: FeatureTable, rows: np.ndarray) -> FeatureTable:
    selected = np.asarray(rows, dtype=bool)
    _require(selected.shape == (len(table.frame),), "feature row mask is misaligned")
    _require(bool(selected.any()), "feature row mask is empty")
    return FeatureTable(
        table.frame.loc[selected].reset_index(drop=True),
        table.feature_columns,
    )


def _joint_router_context(context: RouterContext) -> RouterContext:
    rows = np.asarray(context.joint_rows, dtype=bool)
    _require(rows.shape == (len(context.base.frame),), "router joint mask misaligned")
    _require(bool(rows.any()), "router joint mask is empty")
    positions = np.flatnonzero(rows)
    return RouterContext(
        _subset_feature_table(context.base, rows),
        _subset_feature_table(context.lean, rows),
        _subset_feature_table(context.phase, rows),
        context.public_state.iloc[positions].reset_index(drop=True),
        np.ones(len(positions), dtype=bool),
        pd.DatetimeIndex(context.times[positions]),
    )


def _fit_router_model_4threads(
    context: RouterContext,
    *,
    seed: int,
    layer_arms: Mapping[str, str],
) -> MaxRoundRouterModel:
    available = _joint_router_context(context)
    observed_counts = {
        "base": len(available.base.feature_columns),
        "lean": len(available.lean.feature_columns),
        "phase": len(available.phase.feature_columns),
    }
    _require(observed_counts == ROUTER_FEATURE_COUNTS, "router feature roles changed")
    rows = _all_rows(available.base)
    return MaxRoundRouterModel(
        base_model=_fit_p2_model_4threads(available.base, rows, seed=seed),
        phase_model=_fit_p2_model_4threads(available.phase, rows, seed=seed),
        state_model=_fit_state_model_4threads(available.lean, rows, seed=seed),
        layer_arms={int(layer): str(arm) for layer, arm in layer_arms.items()},
        max_rounds=400,
    )


def _predict_router(
    model: MaxRoundRouterModel,
    context: RouterContext,
    *,
    fold: str,
) -> pd.DataFrame:
    available = _joint_router_context(context)
    prediction = model.predict_components_at(
        available.base,
        available.lean,
        available.phase,
        400,
    )["router"]
    frame = available.base.frame.loc[:, ["station", "layer", "time", "target"]].rename(
        columns={"target": "truth"}
    )
    frame["prediction"] = csv_float_roundtrip(prediction)
    frame.insert(0, "fold", fold)
    state = available.public_state.loc[:, STATE_FEATURES].reset_index(drop=True)
    return pd.concat([frame.reset_index(drop=True), state], axis=1)


def _subset_panel(panel: P2Panel, times: pd.DatetimeIndex, *, expose_targets: bool) -> P2Panel:
    selected = panel.times.isin(times)
    _require(bool(selected.any()), "panel prediction subset is empty")
    kept_times = panel.times[selected]
    delta = kept_times.to_series().diff().dt.total_seconds().div(60).to_numpy()
    segment = np.cumsum(np.r_[True, ~np.isclose(delta[1:], 10)]).astype(np.int32) - 1
    target = panel.target[selected].copy()
    mask = panel.target_mask[selected].copy()
    if not expose_targets:
        target[:] = np.nan
        mask[:] = False
    return P2Panel(
        kept_times,
        panel.inputs[selected].copy(),
        panel.input_names,
        panel.baseline[selected].copy(),
        target,
        mask,
        segment,
    )


def _predict_full_result(result: FullTrainingResult, panel: P2Panel) -> np.ndarray:
    inputs = result.normalizer.transform_inputs(panel.inputs)
    model = build_model(result.config.model, inputs.shape[1]).to(_device())
    model.load_state_dict(result.state_dict)
    bounds = make_chunk_bounds(
        panel.segment_ids,
        length=result.config.chunk_length,
        stride=result.config.chunk_stride,
    )
    normalized = _predict_chunks(
        model,
        inputs,
        bounds,
        length=result.config.chunk_length,
        batch_size=result.config.batch_size,
        diffusion_samples=result.config.diffusion_samples,
        seed=result.config.seed + result.epochs,
    )
    prediction = result.normalizer.inverse_predictions(panel, normalized)
    del model
    torch.cuda.empty_cache()
    return prediction


def _deep_checkpoint_payload(result: Any, *, epochs: int, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "model": result.config.model,
        "config": asdict(result.config),
        "epochs": int(epochs),
        "input_center": result.normalizer.input_center,
        "input_scale": result.normalizer.input_scale,
        "residual_center": result.normalizer.residual_center,
        "residual_scale": result.normalizer.residual_scale,
        "state_dict": result.state_dict,
    }


def exact_checkpoint_key(score: float, epoch: int) -> tuple[float, int]:
    """Return the parent-contract ordering key for an evaluated checkpoint."""

    value = float(score)
    step = int(epoch)
    _require(np.isfinite(value) and step >= 1, "checkpoint key is invalid")
    return value, step


def train_fold_exact_min_rmse(
    panel: P2Panel,
    *,
    block: str,
    start: str,
    stop: str,
    config: TrainingConfig,
) -> FoldTrainingResult:
    """Train one inner fold with the sealed exact minimum/earliest tie rule.

    The historical helper uses a ``1e-6`` improvement threshold.  That is not
    equivalent to the parent contract, so the actual runner owns this loop and
    compares the exact ``(RMSE, epoch)`` tuple at every registered checkpoint.
    """

    set_deterministic_seed(config.seed)
    validation_times = time_block_mask(panel, start, stop)
    train_times = ~validation_times
    normalizer = PanelNormalizer.fit(panel, train_times)
    inputs = normalizer.transform_inputs(panel.inputs)
    target, target_mask = normalizer.transform_targets(panel)
    training_mask = target_mask & train_times[:, None]
    all_bounds = make_chunk_bounds(
        panel.segment_ids,
        length=config.chunk_length,
        stride=config.chunk_stride,
    )
    train_bounds = tuple(
        bound for bound in all_bounds if training_mask[bound[0] : bound[1]].sum() >= 24
    )
    _require(bool(train_bounds), "no exact-fold local training chunks")
    chunk_x, chunk_y, chunk_mask = _materialize_chunks(
        inputs,
        target,
        training_mask,
        train_bounds,
        config.chunk_length,
    )
    device = _device()
    model = build_model(config.model, inputs.shape[1]).to(device)
    parameter_count = count_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.max_epochs, 1),
        eta_min=config.learning_rate * 0.05,
    )
    layer_weights = torch.tensor(
        normalizer.residual_scale**2,
        device=device,
        dtype=torch.float32,
    )
    best_key = (float("inf"), 2**31 - 1)
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    stale = 0
    rng = np.random.default_rng(config.seed)
    for epoch in range(1, config.max_epochs + 1):
        order = rng.permutation(len(train_bounds))
        model.train()
        loss_sum = 0.0
        weight_sum = 0.0
        for begin in range(0, len(order), config.batch_size):
            ids = torch.from_numpy(order[begin : begin + config.batch_size]).long()
            x = chunk_x[ids].to(device, non_blocking=True)
            y = chunk_y[ids].to(device, non_blocking=True)
            mask = chunk_mask[ids].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if isinstance(model, ConditionalDiffusion):
                    loss = model.training_loss(x, y, mask, layer_weights)
                else:
                    predicted = model(x)
                    squared = (predicted - y).square() * layer_weights.view(1, 1, 3)
                    loss = (squared * mask).sum() / mask.sum().clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_weight = float(mask.sum().item())
            loss_sum += float(loss.detach().item()) * batch_weight
            weight_sum += batch_weight
        scheduler.step()
        should_score = (
            epoch == 1 or epoch % config.evaluation_interval == 0 or epoch == config.max_epochs
        )
        if not should_score:
            continue
        normalized_prediction = _predict_chunks(
            model,
            inputs,
            all_bounds,
            length=config.chunk_length,
            batch_size=config.batch_size,
            diffusion_samples=config.diffusion_samples,
            seed=config.seed + epoch,
        )
        prediction = normalizer.inverse_predictions(panel, normalized_prediction)
        oof = _validation_oof(panel, prediction, validation_times, block)
        score = _rmse(
            oof["truth"].to_numpy(float),
            oof["prediction"].to_numpy(float),
        )
        _require(np.isfinite(score), "inner checkpoint RMSE is not finite")
        history.append(
            {
                "epoch": epoch,
                "train_mse_c": loss_sum / max(weight_sum, 1.0),
                "validation_rmse": score,
                "learning_rate": float(scheduler.get_last_lr()[0]),
            }
        )
        current_key = exact_checkpoint_key(score, epoch)
        if current_key < best_key:
            best_key = current_key
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += config.evaluation_interval
        if stale >= config.patience:
            break
    _require(best_state is not None, "exact inner training produced no checkpoint")
    best_rmse, best_epoch = best_key
    model.load_state_dict(best_state)
    normalized_prediction = _predict_chunks(
        model,
        inputs,
        all_bounds,
        length=config.chunk_length,
        batch_size=config.batch_size,
        diffusion_samples=config.diffusion_samples,
        seed=config.seed + best_epoch,
    )
    prediction = normalizer.inverse_predictions(panel, normalized_prediction)
    final_oof = _validation_oof(panel, prediction, validation_times, block)
    final_rmse = _rmse(
        final_oof["truth"].to_numpy(float),
        final_oof["prediction"].to_numpy(float),
    )
    _require(
        np.isclose(final_rmse, best_rmse, rtol=0.0, atol=1e-12),
        "reloaded exact best checkpoint changed RMSE",
    )
    del model
    torch.cuda.empty_cache()
    return FoldTrainingResult(
        block,
        config,
        parameter_count,
        best_epoch,
        final_rmse,
        history,
        normalizer,
        best_state,
        final_oof,
    )


class ActualTrainingBackend:
    """Five-component backend; instantiate only after authorization verification."""

    def __init__(self, observations: pd.DataFrame, config: Mapping[str, Any]) -> None:
        self.observations = observations
        self.config = config
        self.panel = _joint_masked_panel(observations)
        _require(len(self.panel.input_names) == 43, "deep public-only input schema changed")
        self.endpoints = public_endpoint_frame(observations)
        self.layer_factors = config["postprocess"]["layer_extrapolation_factors"]

    def _masked_router_context(self, times: pd.DatetimeIndex) -> RouterContext:
        masked, receipt = joint_mask_target_observations(self.observations, times)
        _require(receipt["joint_mask_equal"], "router target joint mask failed")
        return _build_router_context(masked)

    def _deep_config(self, component: str, seed: int) -> TrainingConfig:
        current = self.config["component_hyperparameters"]["deep"][component]
        return TrainingConfig(
            model=component,
            learning_rate=float(current["learning_rate"]),
            weight_decay=float(current["weight_decay"]),
            max_epochs=int(current["max_epochs"]),
            patience=int(current["patience"]),
            chunk_length=int(current["chunk_length"]),
            chunk_stride=int(current["chunk_stride"]),
            batch_size=int(current["batch_size"]),
            seed=int(seed),
            evaluation_interval=int(current["evaluation_interval"]),
            diffusion_samples=int(current["diffusion_samples"]),
        )

    def _router_product(
        self,
        *,
        plan: PrefixPlan,
        train_times: pd.DatetimeIndex,
        prediction_times: pd.DatetimeIndex,
        seed: int,
        phase: str,
    ) -> JobProduct:
        train_context = self._masked_router_context(train_times)
        prediction_context = self._masked_router_context(prediction_times)
        model = _fit_router_model_4threads(
            train_context,
            seed=seed,
            layer_arms=self.config["component_hyperparameters"]["router_400"]["layer_arms"],
        )
        frame = _predict_router(model, prediction_context, fold=plan.outer_fold)
        receipt = {
            "component": "router_400",
            "phase": phase,
            "seed": seed,
            "composite_lightgbm_estimators": 4,
            "rounds_per_estimator": 400,
            "cpu_threads_per_estimator": 4,
            "training_timestamp_count": len(train_times),
            "prediction_timestamp_count": len(prediction_times),
            "future_or_outer_labels_in_fit": False,
        }
        return JobProduct(frame, receipt, {"model.joblib": _joblib_bytes(model)})

    def fit_inner(
        self,
        component: str,
        plan: PrefixPlan,
        inner: InnerFoldPlan,
        pipeline_seed: int,
    ) -> JobProduct:
        seed = child_seed(
            pipeline_seed,
            component,
            plan.outer_fold,
            plan.fraction,
            inner.inner_fold,
        )
        if component == "router_400":
            return self._router_product(
                plan=plan,
                train_times=inner.train_times,
                prediction_times=inner.validation_times,
                seed=seed,
                phase=inner.inner_fold,
            )
        adapted, adapter = adapt_panel_for_inner_fold(self.panel, inner)
        config = self._deep_config(component, seed)
        start = inner.validation_start_utc.tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S")
        stop = inner.validation_stop_utc.tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S")
        result = train_fold_exact_min_rmse(
            adapted,
            block=inner.inner_fold,
            start=start,
            stop=stop,
            config=config,
        )
        frame = result.oof.copy()
        frame.insert(0, "station", "S-ORS")
        frame = frame.rename(columns={"prediction": "prediction"})
        receipt = {
            "component": component,
            "phase": inner.inner_fold,
            "seed": seed,
            "best_epoch": int(result.best_epoch),
            "best_rmse_c": float(result.best_rmse),
            "parameter_count": int(result.parameter_count),
            "history": result.history,
            "adapter": adapter,
            "future_or_outer_labels_in_fit": False,
        }
        checkpoint = _deep_checkpoint_payload(
            result, epochs=result.best_epoch, role="inner_best_checkpoint"
        )
        return JobProduct(frame, receipt, {"checkpoint.pt": _torch_bytes(checkpoint)})

    def fit_full(
        self,
        component: str,
        plan: PrefixPlan,
        pipeline_seed: int,
        full_epochs: int | None,
    ) -> JobProduct:
        seed = child_seed(
            pipeline_seed,
            component,
            plan.outer_fold,
            plan.fraction,
            "full",
        )
        outer_times = self.panel.times[
            (self.panel.times >= plan.outer_validation_start_utc)
            & (self.panel.times < plan.outer_validation_stop_utc)
        ]
        if component == "router_400":
            return self._router_product(
                plan=plan,
                train_times=plan.prefix_times,
                prediction_times=outer_times,
                seed=seed,
                phase="full",
            )
        _require(full_epochs is not None and full_epochs >= 1, "deep full epochs missing")
        training_panel, adapter = adapt_panel_for_full_prefix(self.panel, plan)
        config = self._deep_config(component, seed)
        config = replace(config, max_epochs=int(full_epochs), patience=int(full_epochs))
        set_deterministic_seed(seed)
        result = train_full_model(
            training_panel,
            config,
            epochs=int(full_epochs),
        )
        prediction_panel = _subset_panel(self.panel, outer_times, expose_targets=False)
        prediction = _predict_full_result(result, prediction_panel)
        truth_panel = _subset_panel(self.panel, outer_times, expose_targets=True)
        rows: list[pd.DataFrame] = []
        for offset, layer in enumerate(TARGET_LAYERS):
            keep = truth_panel.target_mask[:, offset]
            rows.append(
                pd.DataFrame(
                    {
                        "station": "S-ORS",
                        "layer": layer,
                        "time": truth_panel.times[keep].astype(str),
                        "truth": truth_panel.target[keep, offset],
                        "prediction": prediction[keep, offset],
                    }
                )
            )
        frame = pd.concat(rows, ignore_index=True)
        receipt = {
            "component": component,
            "phase": "full",
            "seed": seed,
            "epochs": int(full_epochs),
            "parameter_count": int(result.parameter_count),
            "final_train_mse_c": float(result.final_train_mse_c),
            "adapter": adapter,
            "future_or_outer_labels_in_fit": False,
        }
        checkpoint = _deep_checkpoint_payload(
            result, epochs=int(full_epochs), role="full_prefix_refit"
        )
        return JobProduct(frame, receipt, {"checkpoint.pt": _torch_bytes(checkpoint)})

    def compose(
        self,
        frame: pd.DataFrame,
        stack: Mapping[str, Mapping[str, float]],
        gate: SoftRegimeGate,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        return _compose_prediction(
            frame,
            endpoints=self.endpoints,
            stack_weights=stack,
            gate=gate,
            layer_factors=self.layer_factors,
        )


class TinyBackend:
    """Deterministic synthetic backend; its callbacks are not model fits."""

    def __init__(self) -> None:
        self.callback_count = 0

    @staticmethod
    def _rows(times: pd.DatetimeIndex, fold: str) -> pd.DataFrame:
        rows = [
            {
                "fold": fold,
                "station": "SYNTHETIC",
                "layer": layer,
                "time": time.isoformat(),
                "truth": 12.0 + layer * 0.2 + index * 0.001,
            }
            for index, time in enumerate(times)
            for layer in TARGET_LAYERS
        ]
        return pd.DataFrame(rows)

    def _product(
        self,
        component: str,
        times: pd.DatetimeIndex,
        fold: str,
        phase: str,
        seed: int,
    ) -> JobProduct:
        self.callback_count += 1
        frame = self._rows(times, fold)
        number = COMPONENTS.index(component)
        x = np.arange(len(frame), dtype=float)
        frame["prediction"] = frame["truth"] + (number - 2) * 0.01 + np.sin(x / (7 + number)) * 0.02
        if component == "router_400":
            for index, feature in enumerate(STATE_FEATURES):
                frame[feature] = np.sin(x / (4 + index))
        receipt: dict[str, Any] = {
            "component": component,
            "phase": phase,
            "seed": seed,
            "synthetic_callback_not_a_model_fit": True,
        }
        if component in DEEP_COMPONENTS and phase != "full":
            receipt["best_epoch"] = 4 + 2 * int(phase.split("_")[-1]) + number
        return JobProduct(frame, receipt, {"synthetic.bin": b"synthetic-only"})

    def fit_inner(
        self,
        component: str,
        plan: PrefixPlan,
        inner: InnerFoldPlan,
        pipeline_seed: int,
    ) -> JobProduct:
        return self._product(
            component,
            inner.validation_times[:4],
            plan.outer_fold,
            inner.inner_fold,
            pipeline_seed,
        )

    def fit_full(
        self,
        component: str,
        plan: PrefixPlan,
        pipeline_seed: int,
        full_epochs: int | None,
    ) -> JobProduct:
        times = pd.date_range(
            plan.outer_validation_start_utc,
            periods=4,
            freq="12h",
        )
        product = self._product(component, times, plan.outer_fold, "full", pipeline_seed)
        product.receipt["epochs"] = full_epochs
        return product

    def compose(
        self,
        frame: pd.DataFrame,
        stack: Mapping[str, Mapping[str, float]],
        gate: SoftRegimeGate,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        prediction = np.empty(len(frame), dtype=float)
        for layer in TARGET_LAYERS:
            keep = frame["layer"].to_numpy(int) == layer
            weights = np.array([stack[str(layer)][name] for name in COMPONENTS])
            prediction[keep] = frame.loc[keep, COMPONENTS].to_numpy(float) @ weights
        return prediction, {
            "synthetic_only": True,
            "rows": len(prediction),
            "gate_layers_fitted": len(gate.layers),
        }


def run_tiny_fixture(root: Path, plan: PrefixPlan) -> dict[str, Any]:
    """Exercise one complete seeded cell twice and prove job-level resume."""

    contract = "tiny-contract-v1"
    jobs = JobStore(root / "jobs", contract_sha256=contract)
    backend = TinyBackend()
    first = execute_cell_seed(
        plan=plan,
        pipeline_seed=20260823,
        backend=backend,
        jobs=jobs,
        gate_regularization=10.0,
    )
    first_callbacks = backend.callback_count
    first_new = jobs.new_jobs
    resumed_jobs = JobStore(root / "jobs", contract_sha256=contract)
    resumed_backend = TinyBackend()
    second = execute_cell_seed(
        plan=plan,
        pipeline_seed=20260823,
        backend=resumed_backend,
        jobs=resumed_jobs,
        gate_regularization=10.0,
    )
    _require(first.frame.equals(second.frame), "tiny resume frame differs")
    _require(first.receipt == second.receipt, "tiny resume receipt differs")
    _require(resumed_backend.callback_count == 0, "tiny resume reran a callback")
    _require(first_callbacks == 20 and first_new == 20, "tiny fit graph changed")
    _require(resumed_jobs.reused_jobs == 20, "tiny resume reuse count changed")
    return {
        "status": "PASS_TINY_FULL_CELL_AND_RESUME",
        "synthetic_component_callbacks": first_callbacks,
        "actual_model_fits": 0,
        "first_pass_new_jobs": first_new,
        "second_pass_reused_jobs": resumed_jobs.reused_jobs,
        "second_pass_callbacks": resumed_backend.callback_count,
        "cell_rows": len(first.frame),
        "cell_receipt_sha256": canonical_sha256(first.receipt),
    }


def _causal_spec(config: Mapping[str, Any]) -> CausalResidualSpec:
    current = config["family_views"]["causal_residual_contract"]
    _require(tuple(current["required_anchors"]) == (1, 5), "causal anchors changed")
    _require(current["coherent_sign_required"] is True, "coherence gate changed")
    return CausalResidualSpec(
        public_layers=tuple(int(value) for value in current["public_layers"]),
        rolling_hours=int(current["rolling_hours"]),
        cadence_minutes=int(current["cadence_minutes"]),
        minimum_samples=int(current["minimum_samples"]),
        residual_clip_c=float(current["residual_clip_c"]),
        minimum_anchors=int(current["minimum_anchors"]),
        ridge_slope_lambda=float(current["ridge_slope_lambda"]),
        correction_scale=float(current["correction_scale"]),
        correction_clip_c=float(current["correction_clip_c"]),
        maximum_anchor_span_c=float(current["maximum_anchor_span_c"]),
        depth_scale_m=float(current["depth_scale_m"]),
        nonzero_epsilon=float(current["nonzero_epsilon"]),
    )


def _merge_seed_cells(
    frames: Mapping[int, pd.DataFrame],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    seeds = tuple(sorted(frames))
    reference: pd.DataFrame | None = None
    merged: pd.DataFrame | None = None
    seed_columns: list[str] = []
    for seed in seeds:
        current = (
            frames[seed].sort_values(list(OUTER_KEY_COLUMNS), kind="stable").reset_index(drop=True)
        )
        keys = current.loc[:, [*OUTER_KEY_COLUMNS, "truth"]]
        if reference is None:
            reference = keys
            merged = keys.copy()
        else:
            _require(keys.equals(reference), "seed outer key/truth surface differs")
        column = f"seed_{seed}"
        seed_columns.append(column)
        assert merged is not None
        merged[column] = current["prediction"].to_numpy(float)
    assert merged is not None
    merged["prediction_mean"] = merged.loc[:, seed_columns].mean(axis=1)
    return merged, tuple(seed_columns)


def evaluate_fraction(
    *,
    frames: Mapping[int, pd.DataFrame],
    context: LocalContext,
    spec: CausalResidualSpec,
    bootstrap: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    merged, seed_columns = _merge_seed_cells(frames)
    prepared = prepare_forward_surrogate_surface(merged, context, seed_columns)
    means, per_seed, diagnostics = materialize_settings(
        prepared,
        seed_columns,
        context,
        spec,
    )
    _require(tuple(means) == FAMILY_SETTINGS, "sealed family setting surface changed")
    metrics = {name: metric_report(prepared, value) for name, value in means.items()}
    incumbent = means["INCUMBENT_NOOP"]
    plan = build_bootstrap_plan(
        prepared,
        replicates=int(bootstrap["replicates"]),
        seed=int(bootstrap["seed"]),
    )
    uncertainty = {
        setting: paired_day_bootstrap(
            prepared,
            incumbent,
            prediction,
            plan,
            interval=float(bootstrap["interval"]),
        )
        for setting, prediction in means.items()
    }
    complementarity = {
        setting: complementarity_report(
            prepared,
            incumbent,
            prediction,
            context.endpoints,
        )
        for setting, prediction in means.items()
    }
    output = prepared.loc[:, [*OUTER_KEY_COLUMNS, "truth", *seed_columns]].copy()
    for setting, prediction in means.items():
        output[setting] = prediction
    return output, {
        "rows": len(prepared),
        "seed_columns": list(seed_columns),
        "metrics_by_setting": metrics,
        "paired_day_bootstrap_vs_incumbent": uncertainty,
        "complementarity": complementarity,
        "materialization_diagnostics": diagnostics,
        "per_seed_setting_count": {setting: len(values) for setting, values in per_seed.items()},
    }


def verify_preexecution_seal(
    seal_path: Path,
    *,
    config_sha256: str,
    module_sha256: str,
    runner_sha256: str,
    exact_command: str,
) -> dict[str, Any]:
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    _require(seal["status"] == "EXECUTION_READY_NOT_AUTHORIZED", "preexecution seal status changed")
    _require(seal["config_sha256"] == config_sha256, "preexecution config pin changed")
    _require(seal["module_sha256"] == module_sha256, "preexecution module pin changed")
    _require(seal["runner_sha256"] == runner_sha256, "preexecution runner pin changed")
    _require(
        seal["exact_command_sha256"] == hashlib.sha256(exact_command.encode()).hexdigest(),
        "exact command pin changed",
    )
    _require(seal["actual_model_fits"] == 0, "seal reports a model fit")
    return seal


def verify_authorization(
    authorization_path: Path,
    *,
    preexecution_seal_sha256: str,
    exact_command: str,
) -> dict[str, Any]:
    value = json.loads(authorization_path.read_text(encoding="utf-8"))
    _require(value["status"] == "APPROVED_EXACT_P2_45_CELL_COMMAND", "authorization status invalid")
    _require(value["training_authorized"] is True, "training is not authorized")
    _require(
        value["preexecution_seal_sha256"] == preexecution_seal_sha256,
        "authorization seal pin differs",
    )
    _require(
        value["exact_command_sha256"] == hashlib.sha256(exact_command.encode()).hexdigest(),
        "authorization command differs",
    )
    for key in (
        "official_test_access_authorized",
        "sample_submission_access_authorized",
        "submission_generation_authorized",
        "public_score_selection_authorized",
        "upload_authorized",
        "p3_process_mutation_authorized",
    ):
        _require(value[key] is False, f"authorization expands forbidden scope: {key}")
    return value


def execute_authorized_curve(
    *,
    observations: pd.DataFrame,
    parent_recipe: Mapping[str, Any],
    config: Mapping[str, Any],
    output_dir: Path,
    contract_sha256: str,
) -> dict[str, Any]:
    """Execute/resume all 45 cells after the caller has verified authorization."""

    metadata = observations.loc[:, ["station", "layer", "time"]]
    plans = build_all_prefix_plans(metadata, parent_recipe)
    seeds = tuple(
        int(value)
        for value in parent_recipe["authoritative_nested_surrogate_recipe"][
            "complete_pipeline_seed_contract"
        ]["seeds"]
    )
    _require(len(plans) == 15 and len(seeds) == 3, "45-cell graph changed")
    backend = ActualTrainingBackend(observations, config)
    jobs = JobStore(output_dir / "jobs", contract_sha256=contract_sha256)
    cell_store = JobStore(output_dir / "cells", contract_sha256=contract_sha256)
    by_fraction: dict[float, dict[int, list[pd.DataFrame]]] = {
        fraction: {seed: [] for seed in seeds}
        for fraction in sorted({plan.fraction for plan in plans})
    }
    for plan in plans:
        for seed in seeds:
            cell_id = f"cell__{plan.scope_id}__s{seed}"
            product = cell_store.materialize(
                cell_id,
                lambda plan=plan, seed=seed: execute_cell_seed(
                    plan=plan,
                    pipeline_seed=seed,
                    backend=backend,
                    jobs=jobs,
                    gate_regularization=float(config["meta_refit"]["gate_regularization"]),
                ),
            )
            by_fraction[plan.fraction][seed].append(product.frame)
    spec = _causal_spec(config)
    context = build_local_context(observations, spec)
    result: dict[str, Any] = {}
    population_digest: str | None = None
    expected_rows = int(config["metrics"]["expected_evaluation_rows_per_fraction"])
    for fraction, by_seed in by_fraction.items():
        frames = {seed: pd.concat(parts, ignore_index=True) for seed, parts in by_seed.items()}
        output, metrics = evaluate_fraction(
            frames=frames,
            context=context,
            spec=spec,
            bootstrap=config["metrics"]["bootstrap"],
        )
        _require(len(output) == expected_rows, "outer evaluation population changed")
        current_digest = canonical_sha256(
            output.loc[:, [*OUTER_KEY_COLUMNS, "truth"]].astype(str).to_dict("records")
        )
        if population_digest is None:
            population_digest = current_digest
        else:
            _require(
                current_digest == population_digest,
                "outer population differs across prefix fractions",
            )
        token = f"{int(round(fraction * 100)):03d}"
        path = output_dir / f"evaluated_oof_{token}.parquet"
        payload = _dataframe_bytes(output)
        publish_receipt = atomic_write_or_verify(path, payload)
        result[token] = {
            **metrics,
            "ordered_key_truth_sha256": current_digest,
            "evaluated_oof_publish": publish_receipt,
        }
    return {
        "status": "COMPLETE_LOCAL_AUTHORITATIVE_SURROGATE_NO_PROMOTION",
        "outer_prefix_cells": 15,
        "seeded_cells": 45,
        "component_jobs_new_this_invocation": jobs.new_jobs,
        "component_jobs_reused_this_invocation": jobs.reused_jobs,
        "cell_jobs_new_this_invocation": cell_store.new_jobs,
        "cell_jobs_reused_this_invocation": cell_store.reused_jobs,
        "metrics_by_prefix": result,
        "top_level_component_jobs_total": 900,
        "underlying_base_estimator_fits_total": 1440,
        "underlying_deep_fits_total": 720,
        "underlying_lightgbm_fits_total": 720,
        "meta_optimizations_total": 405,
        "same_population_digest_across_fractions": population_digest,
        "submission_files_generated": 0,
        "uploads": 0,
    }


def temporary_tiny_fixture(plan: PrefixPlan) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="p2_auth_tiny_") as directory:
        return run_tiny_fixture(Path(directory), plan)
