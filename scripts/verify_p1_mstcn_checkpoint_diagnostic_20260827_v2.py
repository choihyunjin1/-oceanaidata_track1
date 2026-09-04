"""Independent, read-only QA for the sealed P1 checkpoint diagnostic.

The verifier deliberately does not import the experiment runner or its source
runner.  It accepts only a completed append-only artifact namespace, verifies
every manifest identity, replays the artifact contracts, and independently
recomputes the reported binary metrics from the pinned historical truth and
current-Router anchor.  It never writes a receipt or submission artifact.

Exit codes:
  0  complete artifact verified
  2  complete-looking artifact failed verification
  3  artifact is absent or still incomplete; wait and retry
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "p1_mstcn_checkpoint_diagnostic_20260827_v2"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
EXPECTED_CONFIG_SHA256 = "437c5c0aa2d2c1508518c48c0a469aa2ae08b9b927a918fc7c7144784c2c0d0c"
EXPECTED_RUNNER_SHA256 = "aa4f9c65b5bcdb62a342480e3b21c80f4c9276f57ba198c7b5e50363ffd9a4cf"
PHASES = ("q3", "q4")
FOLDS = {"q3": "2025_q3", "q4": "2025_q4"}
KEY_COLUMNS = ("station", "year", "layer", "time")
CHECKPOINT_EPOCHS = (120, 125, 130, 145, 150)
ORACLE_EPOCHS = (120, 125, 130, 145)
STATE_EPOCHS = (145, 150)
SEEDS = (20260827, 20260839, 20260863)
TYPE_COUNT = 5
MODEL_NUMERIC_FEATURE_COUNT = 74
RUNTIME_INPUT_FEATURE_COUNT = 165
FLOAT_TOLERANCE = 1.0e-12


class IncompleteArtifactError(RuntimeError):
    """Raised when a one-shot artifact has not reached its terminal seal."""


class VerificationError(RuntimeError):
    """Raised when an immutable artifact contract does not replay."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, *, expected_type: type = dict) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"invalid JSON: {path.name}: {error}") from error
    _require(isinstance(value, expected_type), f"unexpected JSON root type: {path.name}")
    return value


def _parse_utc(value: Any, *, label: str) -> datetime:
    _require(isinstance(value, str), f"{label} is not an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise VerificationError(f"invalid ISO timestamp: {label}") from error
    _require(parsed.tzinfo is not None, f"{label} must be timezone-aware")
    return parsed.astimezone(UTC)


def _safe_child(parent: Path, name: Any, *, label: str) -> Path:
    _require(isinstance(name, str) and bool(name), f"{label} path is invalid")
    relative = Path(name)
    _require(relative.name == name and not relative.is_absolute(), f"{label} escapes namespace")
    path = (parent / relative).resolve()
    _require(path.parent == parent.resolve(), f"{label} escapes namespace")
    return path


def _safe_project_path(project_root: Path, relative_text: Any, *, label: str) -> Path:
    _require(isinstance(relative_text, str) and bool(relative_text), f"{label} path is invalid")
    relative = Path(relative_text)
    _require(not relative.is_absolute(), f"{label} path must be project-relative")
    root = project_root.resolve()
    path = (root / relative).resolve()
    _require(path.is_relative_to(root), f"{label} path escapes project root")
    return path


def _identity(path: Path, *, shown_path: str | None = None) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"artifact is absent or unsafe: {path}")
    return {
        "path": shown_path if shown_path is not None else path.name,
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _verify_child_identity(parent: Path, expected: Any, *, label: str) -> Path:
    _require(isinstance(expected, Mapping), f"{label} identity is invalid")
    path = _safe_child(parent, expected.get("path"), label=label)
    observed = _identity(path)
    wanted = {
        "path": expected.get("path"),
        "bytes": expected.get("bytes"),
        "sha256": expected.get("sha256"),
    }
    _require(observed == wanted, f"{label} identity changed")
    return path


def _verify_project_identity(project_root: Path, expected: Any, *, label: str) -> Path:
    _require(isinstance(expected, Mapping), f"{label} identity is invalid")
    path = _safe_project_path(project_root, expected.get("path"), label=label)
    observed = _identity(path, shown_path=str(expected.get("path")))
    wanted = {
        "path": expected.get("path"),
        "bytes": expected.get("bytes"),
        "sha256": expected.get("sha256"),
    }
    _require(observed == wanted, f"{label} source identity changed")
    return path


def _expected_artifact_names() -> set[str]:
    names = {
        "preflight.json",
        "q2_plateau_revalidation.json",
        "selected_recipe.json",
        "blind_semantic_replays.json",
        "fixed_epoch_150_metrics.json",
        "fixed_epoch_150_decision.json",
        "same_truth_oracle_diagnostic.json",
        "terminal_result.json",
    }
    for phase in PHASES:
        names.update(
            {
                f"{phase}_split.json",
                f"{phase}_encoder.json",
                f"{phase}_blind_checkpoint_curve.npz",
                f"{phase}_blind_checkpoint_curve_receipt.json",
            }
        )
        for seed in SEEDS:
            names.add(f"{phase}_width_512_seed_{seed}_training_history.json")
            for epoch in STATE_EPOCHS:
                names.add(f"{phase}_width_512_seed_{seed}_epoch_{epoch}_state.pt")
    return names


def _read_and_verify_manifest(artifact_dir: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    if not artifact_dir.is_dir():
        raise IncompleteArtifactError("artifact namespace does not exist yet")
    manifest_path = artifact_dir / "manifest.json"
    terminal_path = artifact_dir / "terminal_result.json"
    if not manifest_path.is_file() or not terminal_path.is_file():
        raise IncompleteArtifactError("terminal_result.json or manifest.json is not sealed yet")

    manifest = _load_json(manifest_path)
    _require(
        manifest.get("schema_version") == "p1.mstcn_checkpoint_diagnostic.manifest.v1",
        "manifest schema changed",
    )
    _require(manifest.get("experiment_id") == EXPERIMENT_ID, "manifest experiment changed")
    _require(manifest.get("official_interface_reads") == 0, "manifest records official reads")
    _require(manifest.get("submission_created") is False, "manifest records submission output")
    _require(manifest.get("upload_performed") is False, "manifest records upload")
    entries = manifest.get("files")
    _require(isinstance(entries, list), "manifest file inventory is absent")
    _require(
        manifest.get("file_count_excluding_manifest") == len(entries),
        "manifest file count changed",
    )
    names = [row.get("path") for row in entries if isinstance(row, Mapping)]
    _require(len(names) == len(entries), "manifest contains invalid identity rows")
    _require(len(set(names)) == len(names), "manifest contains duplicate paths")
    _require(names == sorted(names), "manifest paths are not canonically ordered")
    expected_names = _expected_artifact_names()
    _require(set(names) == expected_names, "manifest required file inventory changed")

    actual_files = {
        path.name
        for path in artifact_dir.iterdir()
        if path.is_file() and path.name != "manifest.json"
    }
    unsafe_entries = [
        path.name for path in artifact_dir.iterdir() if path.is_symlink() or path.is_dir()
    ]
    _require(not unsafe_entries, "artifact namespace contains a directory or symlink")
    _require(actual_files == expected_names, "manifest and on-disk inventory differ")
    forbidden = ("submission", "sample", "official_test", "upload")
    _require(
        not any(token in name.casefold() for token in forbidden for name in actual_files),
        "artifact namespace contains an official-interface output name",
    )
    _require(
        not any(Path(name).suffix.casefold() == ".csv" for name in actual_files), "CSV output found"
    )

    paths: dict[str, Path] = {}
    for entry in entries:
        path = _verify_child_identity(artifact_dir, entry, label=f"manifest/{entry['path']}")
        paths[path.name] = path
    return manifest, paths


def _verify_preflight(path: Path, *, project_root: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    preflight = _load_json(path)
    _require(
        preflight.get("schema_version") == "p1.mstcn_checkpoint_diagnostic.preflight.v1",
        "preflight schema changed",
    )
    _require(preflight.get("experiment_id") == EXPERIMENT_ID, "preflight experiment changed")
    _require(preflight.get("result") == "PASS", "preflight did not pass")
    _require(preflight.get("config_sha256") == EXPECTED_CONFIG_SHA256, "config hash changed")
    _require(preflight.get("runner_sha256") == EXPECTED_RUNNER_SHA256, "runner hash changed")
    _require(
        preflight.get("source_q2_blind_receipt_verified") is True, "Q2 receipt was not verified"
    )
    _require(preflight.get("q3_q4_truth_columns_read") == 0, "preflight opened Q3/Q4 truth")
    _require(preflight.get("official_interface_reads") == 0, "preflight records official reads")
    _require(
        preflight.get("artifact_namespace_available") is True, "preflight namespace was not fresh"
    )
    pins = preflight.get("source_pins")
    _require(isinstance(pins, Mapping), "preflight source pins are absent")
    required_pins = {
        "diagnostic_v1_runner",
        "runner",
        "config",
        "model",
        "data",
        "q2_grid",
        "q2_receipt",
        "current_router_anchor",
        "frozen_oof",
    }
    _require(set(pins) == required_pins, "source pin inventory changed")
    verified_paths = {
        name: _verify_project_identity(project_root, pins[name], label=f"source pin/{name}")
        for name in sorted(required_pins)
    }
    runtime = preflight.get("source_runtime_identity")
    _require(
        isinstance(runtime, Mapping) and runtime.get("result") == "PASS_EXACT_RUNTIME_IDENTITY",
        "runtime identity did not pass",
    )
    return preflight, verified_paths


def _verify_q2_and_recipe(
    q2_path: Path,
    recipe_path: Path,
    *,
    project_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    q2 = _load_json(q2_path)
    _require(
        q2.get("schema_version") == "p1.mstcn_checkpoint_diagnostic.q2_revalidation.v1",
        "Q2 revalidation schema changed",
    )
    _require(q2.get("result") == "PASS", "Q2 revalidation did not pass")
    _require(q2.get("role") == "development_only_not_promotion_evidence", "Q2 role changed")
    _require(q2.get("q3_q4_truth_columns_read") == 0, "Q2 seal opened Q3/Q4 truth")
    _verify_project_identity(
        project_root,
        q2.get("source_grid_receipt"),
        label="Q2 source grid receipt",
    )
    plateau = q2.get("plateau_selection")
    _require(isinstance(plateau, Mapping), "Q2 plateau is absent")
    _require(
        [plateau.get("width"), plateau.get("epoch"), plateau.get("threshold")] == [512, 150, 0.8],
        "Q2 plateau recipe changed",
    )
    _require(plateau.get("neighbor_epochs") == [145, 150, 155], "Q2 plateau neighbors changed")
    monthly = q2.get("monthly_development_metrics")
    _require(
        isinstance(monthly, Mapping) and set(monthly) == {"2025-04", "2025-05", "2025-06"},
        "Q2 monthly window inventory changed",
    )
    for month, row in monthly.items():
        _require(
            isinstance(row, Mapping)
            and row.get("delta_positive") is True
            and row.get("added_precision_gate") is True,
            f"Q2 monthly gate failed: {month}",
        )

    recipe = _load_json(recipe_path)
    exact = {
        "width": 512,
        "epoch": 150,
        "threshold": 0.8,
        "representation": "raw_three_seed_ensemble_mean",
        "seeds": list(SEEDS),
        "blind_prediction_epochs": list(CHECKPOINT_EPOCHS),
        "saved_state_epochs": list(STATE_EPOCHS),
    }
    _require(
        recipe.get("schema_version") == "p1.mstcn_checkpoint_diagnostic.fixed_recipe.v1",
        "recipe schema changed",
    )
    _require(all(recipe.get(key) == value for key, value in exact.items()), "fixed recipe changed")
    _require(recipe.get("source_schedule_horizon_epochs") == 300, "LR schedule horizon changed")
    _require(recipe.get("fresh_refit_stop_epoch") == 150, "fresh-refit stop epoch changed")
    _require(
        recipe.get("sealed_before_q3_q4_training") is True, "recipe was not sealed before training"
    )
    _require(
        recipe.get("sealed_before_q3_q4_truth_access") is True, "recipe was not sealed before truth"
    )
    _require(recipe.get("q3_q4_truth_columns_read") == 0, "recipe seal opened Q3/Q4 truth")
    _require(
        recipe.get("q3_q4_result_driven_changes_authorized") is False,
        "result-driven recipe changes were authorized",
    )
    observed_source = _identity(q2_path)
    _require(
        recipe.get("source_revalidation") == observed_source, "recipe Q2 source identity changed"
    )
    return q2, recipe


def _array_inventory(arrays: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in arrays.items()
    }


def _verify_fit_receipts(
    artifact_dir: Path,
    receipt: Mapping[str, Any],
    *,
    phase: str,
) -> None:
    fit_receipts = receipt.get("fit_receipts")
    _require(
        isinstance(fit_receipts, list) and len(fit_receipts) == 3, f"{phase} seed count changed"
    )
    _require(
        [row.get("seed") for row in fit_receipts] == list(SEEDS), f"{phase} seed order changed"
    )
    for fit in fit_receipts:
        seed = fit.get("seed")
        _require(
            fit.get("phase") == phase
            and fit.get("width") == 512
            and fit.get("fresh_refit") is True
            and fit.get("epochs_trained") == 150
            and fit.get("source_schedule_horizon_epochs") == 300,
            f"{phase}/{seed} fresh-refit contract changed",
        )
        _require(
            fit.get("checkpoint_epochs") == list(CHECKPOINT_EPOCHS),
            f"{phase}/{seed} checkpoints changed",
        )
        _require(
            fit.get("saved_state_epochs") == list(STATE_EPOCHS),
            f"{phase}/{seed} state epochs changed",
        )
        history_path = _verify_child_identity(
            artifact_dir,
            fit.get("history_artifact"),
            label=f"{phase}/{seed} training history",
        )
        expected_history = f"{phase}_width_512_seed_{seed}_training_history.json"
        _require(history_path.name == expected_history, f"{phase}/{seed} history name changed")
        history = _load_json(history_path, expected_type=list)
        _require(len(history) == 150, f"{phase}/{seed} history length changed")
        _require(
            [row.get("epoch") for row in history] == list(range(1, 151)),
            f"{phase}/{seed} history epoch sequence changed",
        )
        captured = [
            row.get("epoch") for row in history if row.get("blind_checkpoint_captured") is True
        ]
        saved = [row.get("epoch") for row in history if row.get("state_saved") is True]
        _require(
            captured == list(CHECKPOINT_EPOCHS), f"{phase}/{seed} blind capture history changed"
        )
        _require(saved == list(STATE_EPOCHS), f"{phase}/{seed} saved-state history changed")
        states = fit.get("state_artifacts")
        _require(isinstance(states, list), f"{phase}/{seed} state inventory is absent")
        _require(
            [row.get("epoch") for row in states] == list(STATE_EPOCHS),
            f"{phase}/{seed} state inventory changed",
        )
        for state in states:
            epoch = state.get("epoch")
            state_path = _verify_child_identity(
                artifact_dir,
                state,
                label=f"{phase}/{seed}/epoch{epoch} state",
            )
            expected_state = f"{phase}_width_512_seed_{seed}_epoch_{epoch}_state.pt"
            _require(
                state_path.name == expected_state, f"{phase}/{seed}/epoch{epoch} state name changed"
            )


def _verify_phase(
    artifact_dir: Path,
    receipt_path: Path,
    recipe_path: Path,
    *,
    phase: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    receipt = _load_json(receipt_path)
    _require(
        receipt.get("schema_version") == "p1.mstcn_checkpoint_diagnostic.phase_blind.v1",
        f"{phase} receipt schema changed",
    )
    _require(receipt.get("experiment_id") == EXPERIMENT_ID, f"{phase} experiment changed")
    _require(
        receipt.get("phase") == phase and receipt.get("fold") == FOLDS[phase],
        f"{phase} fold changed",
    )
    _parse_utc(receipt.get("created_at_utc"), label=f"{phase} receipt created_at_utc")
    _require(receipt.get("config_sha256") == EXPECTED_CONFIG_SHA256, f"{phase} config hash changed")
    _require(receipt.get("recipe_sha256") == _sha256(recipe_path), f"{phase} recipe hash changed")
    _require(
        receipt.get("checkpoint_epochs") == list(CHECKPOINT_EPOCHS), f"{phase} checkpoints changed"
    )
    _require(receipt.get("scientific_metric_epoch") == 150, f"{phase} metric epoch changed")
    _require(
        receipt.get("same_truth_oracle_diagnostic_epochs") == list(ORACLE_EPOCHS),
        f"{phase} oracle epochs changed",
    )
    _require(
        receipt.get("same_truth_oracle_promotion_evidence") is False,
        f"{phase} oracle became promotion evidence",
    )
    _require(
        receipt.get("same_truth_oracle_recipe_mutation_allowed") is False,
        f"{phase} oracle mutation enabled",
    )
    _require(
        receipt.get("same_fold_holdout_truth_columns_opened_before_receipt") == 0,
        f"{phase} truth firewall failed",
    )
    _require(
        receipt.get("prior_fold_metrics_computed_before_both_phase_seals") is False,
        f"{phase} prior metrics leaked",
    )
    _require(receipt.get("official_interface_reads") == 0, f"{phase} records official reads")
    rows = receipt.get("holdout_rows")
    _require(isinstance(rows, int) and rows > 0, f"{phase} row count is invalid")

    score_expected = {
        "path": receipt.get("score_path"),
        "bytes": receipt.get("score_bytes"),
        "sha256": receipt.get("score_sha256"),
    }
    score_path = _verify_child_identity(artifact_dir, score_expected, label=f"{phase} blind NPZ")
    _require(
        score_path.name == f"{phase}_blind_checkpoint_curve.npz", f"{phase} blind NPZ name changed"
    )
    with np.load(score_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    expected_names = {
        "epochs",
        "row_probability",
        "boundary_probability",
        "type_probability",
        "proposal",
        "candidate",
    }
    _require(set(arrays) == expected_names, f"{phase} blind NPZ array inventory changed")
    _require(
        _array_inventory(arrays) == receipt.get("array_inventory"), f"{phase} array receipt changed"
    )
    expected_shapes = {
        "epochs": (5,),
        "row_probability": (5, rows),
        "boundary_probability": (5, rows, 2),
        "type_probability": (5, rows, TYPE_COUNT),
        "proposal": (5, rows),
        "candidate": (5, rows),
    }
    _require(
        all(arrays[name].shape == shape for name, shape in expected_shapes.items()),
        f"{phase} blind NPZ shapes changed",
    )
    expected_dtypes = {
        "epochs": "int16",
        "row_probability": "float32",
        "boundary_probability": "float32",
        "type_probability": "float32",
        "proposal": "int8",
        "candidate": "int8",
    }
    _require(
        all(str(arrays[name].dtype) == dtype for name, dtype in expected_dtypes.items()),
        f"{phase} blind NPZ dtypes changed",
    )
    _require(arrays["epochs"].tolist() == list(CHECKPOINT_EPOCHS), f"{phase} epoch order changed")
    for name in ("row_probability", "boundary_probability", "type_probability"):
        values = arrays[name]
        _require(
            bool(np.isfinite(values).all()) and bool(((values >= 0.0) & (values <= 1.0)).all()),
            f"{phase}/{name} is not finite probability data",
        )
    for name in ("proposal", "candidate"):
        _require(bool(np.isin(arrays[name], [0, 1]).all()), f"{phase}/{name} is not binary")
    _require(
        bool((arrays["candidate"] >= arrays["proposal"]).all()),
        f"{phase} candidate removed a proposal",
    )
    _verify_fit_receipts(artifact_dir, receipt, phase=phase)
    return receipt, arrays


def _ordered_key_sha(frame: Any) -> str:
    digest = hashlib.sha256()
    for column in KEY_COLUMNS:
        digest.update(column.encode("ascii") + b"\0")
        for value in frame[column].tolist():
            raw = str(value).encode("utf-8")
            digest.update(len(raw).to_bytes(4, "little"))
            digest.update(raw)
    return digest.hexdigest()


def _keys_equal(left: Any, right: Any) -> bool:
    if len(left) != len(right):
        return False
    return all(
        left[column]
        .astype("string")
        .fillna("")
        .reset_index(drop=True)
        .equals(right[column].astype("string").fillna("").reset_index(drop=True))
        for column in KEY_COLUMNS
    )


def _read_fold(path: Path, *, fold: str, columns: Sequence[str]) -> Any:
    try:
        import pyarrow.dataset as dataset
    except ImportError as error:
        raise VerificationError("pyarrow is required for independent metric replay") from error
    try:
        scanner = dataset.dataset(path, format="parquet").scanner(
            columns=list(columns),
            filter=dataset.field("fold") == fold,
            use_threads=True,
        )
        return scanner.to_table().to_pandas().reset_index(drop=True)
    except Exception as error:
        raise VerificationError(f"cannot read pinned historical fold {fold}: {error}") from error


def _binary_metrics(truth: Any, prediction: Any) -> dict[str, float | int]:
    import numpy as np

    y = np.asarray(truth, dtype=np.int8)
    pred = np.asarray(prediction, dtype=np.int8)
    _require(y.shape == pred.shape, "binary metric arrays are not aligned")
    _require(bool(np.isin(y, [0, 1]).all()), "historical truth is not binary")
    _require(bool(np.isin(pred, [0, 1]).all()), "historical prediction is not binary")
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * tp / (2.0 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _same_number(left: Any, right: Any, *, tolerance: float = FLOAT_TOLERANCE) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, int) and isinstance(right, int):
        return left == right
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _verify_metric_mapping(observed: Any, expected: Mapping[str, Any], *, label: str) -> None:
    _require(isinstance(observed, Mapping), f"{label} metrics are absent")
    _require(set(observed) == set(expected), f"{label} metric fields changed")
    for name, value in expected.items():
        _require(_same_number(observed.get(name), value), f"{label}/{name} changed")


def _load_historical_surfaces(
    source_paths: Mapping[str, Path],
    receipts: Mapping[str, Mapping[str, Any]],
    arrays: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    import numpy as np

    oof_path = source_paths["frozen_oof"]
    anchor_path = source_paths["current_router_anchor"]
    surfaces: dict[str, dict[str, Any]] = {}
    for phase in PHASES:
        fold = FOLDS[phase]
        truth_frame = _read_fold(
            oof_path,
            fold=fold,
            columns=(*KEY_COLUMNS, "label", "fold"),
        )
        anchor_frame = _read_fold(
            anchor_path,
            fold=fold,
            columns=(*KEY_COLUMNS, "fold", "current_router_prediction"),
        )
        receipt = receipts[phase]
        rows = int(receipt["holdout_rows"])
        _require(
            len(truth_frame) == rows and len(anchor_frame) == rows,
            f"{phase} historical row count changed",
        )
        _require(_keys_equal(truth_frame, anchor_frame), f"{phase} truth/anchor keys differ")
        _require(
            _ordered_key_sha(truth_frame) == receipt.get("ordered_holdout_key_sha256"),
            f"{phase} ordered holdout key hash changed",
        )
        truth = truth_frame["label"].to_numpy(dtype=np.int8)
        anchor = anchor_frame["current_router_prediction"].to_numpy(dtype=np.int8)
        _require(bool(np.isin(truth, [0, 1]).all()), f"{phase} historical truth is not binary")
        _require(bool(np.isin(anchor, [0, 1]).all()), f"{phase} historical anchor is not binary")
        for index, epoch in enumerate(CHECKPOINT_EPOCHS):
            proposal = arrays[phase]["proposal"][index]
            candidate = arrays[phase]["candidate"][index]
            replay = np.maximum(anchor, proposal).astype(np.int8, copy=False)
            _require(
                bool(np.array_equal(candidate, replay)),
                f"{phase}/epoch{epoch} anchor union changed",
            )
        surfaces[phase] = {
            "keys": truth_frame.loc[:, KEY_COLUMNS].copy(),
            "truth": truth,
            "anchor": anchor,
        }
    return surfaces


def _recompute_fixed_metrics(
    reported: Mapping[str, Any],
    surfaces: Mapping[str, Mapping[str, Any]],
    arrays: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    import numpy as np

    _require(
        reported.get("schema_version") == "p1.mstcn_checkpoint_diagnostic.fixed_metrics.v1",
        "fixed metric schema changed",
    )
    _require(reported.get("scientific_metric_epoch") == 150, "fixed metric epoch changed")
    _require(reported.get("truth_scored_epochs") == [150], "fixed truth-scored epochs changed")
    _require(
        reported.get("same_truth_oracle_epochs_pending") == list(ORACLE_EPOCHS),
        "fixed oracle pending set changed",
    )
    _require(
        reported.get("official_probe_authorized") is False,
        "fixed metrics authorize official probing",
    )
    _require(
        reported.get("three_official_points_claimed") is False,
        "fixed metrics claim official points",
    )
    epoch_index = CHECKPOINT_EPOCHS.index(150)
    truth_parts: list[Any] = []
    anchor_parts: list[Any] = []
    candidate_parts: list[Any] = []
    station_parts: list[Any] = []
    fold_deltas: list[float] = []
    for phase in PHASES:
        truth = surfaces[phase]["truth"]
        anchor = surfaces[phase]["anchor"]
        candidate = arrays[phase]["candidate"][epoch_index]
        anchor_score = _binary_metrics(truth, anchor)
        candidate_score = _binary_metrics(truth, candidate)
        delta = float(candidate_score["f1"] - anchor_score["f1"])
        observed = reported.get("folds", {}).get(phase)
        _require(isinstance(observed, Mapping), f"fixed metrics/{phase} is absent")
        _verify_metric_mapping(observed.get("anchor"), anchor_score, label=f"fixed/{phase}/anchor")
        _verify_metric_mapping(
            observed.get("candidate"), candidate_score, label=f"fixed/{phase}/candidate"
        )
        _require(_same_number(observed.get("delta_f1"), delta), f"fixed/{phase}/delta_f1 changed")
        truth_parts.append(truth)
        anchor_parts.append(anchor)
        candidate_parts.append(candidate)
        station_parts.append(surfaces[phase]["keys"]["station"].astype(str).to_numpy())
        fold_deltas.append(delta)

    truth = np.concatenate(truth_parts)
    anchor = np.concatenate(anchor_parts)
    candidate = np.concatenate(candidate_parts)
    stations = np.concatenate(station_parts)
    anchor_score = _binary_metrics(truth, anchor)
    candidate_score = _binary_metrics(truth, candidate)
    delta = float(candidate_score["f1"] - anchor_score["f1"])
    added = (candidate == 1) & (anchor == 0)
    removed = int(np.sum((anchor == 1) & (candidate == 0)))
    pooled = reported.get("pooled")
    _require(isinstance(pooled, Mapping), "fixed pooled metrics are absent")
    _require(pooled.get("rows") == len(truth), "fixed pooled row count changed")
    _verify_metric_mapping(pooled.get("anchor"), anchor_score, label="fixed/pooled/anchor")
    _verify_metric_mapping(pooled.get("candidate"), candidate_score, label="fixed/pooled/candidate")
    _require(_same_number(pooled.get("delta_f1"), delta), "fixed pooled delta changed")
    _require(pooled.get("added_rows") == int(added.sum()), "fixed added row count changed")
    expected_precision = float(truth[added].mean()) if added.any() else 0.0
    _require(
        _same_number(pooled.get("added_row_precision"), expected_precision),
        "fixed added precision changed",
    )
    _require(pooled.get("anchor_positive_removed_rows") == removed, "fixed removal count changed")

    by_station = reported.get("by_station")
    _require(isinstance(by_station, Mapping), "fixed station metrics are absent")
    expected_stations = sorted(set(stations.tolist()))
    _require(sorted(by_station) == expected_stations, "fixed station inventory changed")
    for station in expected_stations:
        mask = stations == station
        station_anchor = _binary_metrics(truth[mask], anchor[mask])
        station_candidate = _binary_metrics(truth[mask], candidate[mask])
        station_delta = float(station_candidate["f1"] - station_anchor["f1"])
        observed = by_station[station]
        _verify_metric_mapping(
            observed.get("anchor"), station_anchor, label=f"fixed/{station}/anchor"
        )
        _verify_metric_mapping(
            observed.get("candidate"), station_candidate, label=f"fixed/{station}/candidate"
        )
        _require(
            _same_number(observed.get("delta_f1"), station_delta), f"fixed/{station}/delta changed"
        )
    _require(
        reported.get("fixed_recipe_improved_pooled") is (delta > 0.0),
        "fixed improvement flag changed",
    )
    _require(
        reported.get("both_fold_deltas_positive") is all(value > 0.0 for value in fold_deltas),
        "fixed fold sign flag changed",
    )
    bootstrap = reported.get("bootstrap")
    _require(isinstance(bootstrap, Mapping), "fixed bootstrap receipt is absent")
    return {
        "epoch": 150,
        "pooled_anchor_f1": float(anchor_score["f1"]),
        "pooled_candidate_f1": float(candidate_score["f1"]),
        "pooled_delta_f1": delta,
        "added_rows": int(added.sum()),
        "added_precision": expected_precision,
        "anchor_positive_removed_rows": removed,
    }


def _recompute_oracle(
    reported: Mapping[str, Any],
    surfaces: Mapping[str, Mapping[str, Any]],
    arrays: Mapping[str, Mapping[str, Any]],
    *,
    recipe_path: Path,
    decision_path: Path,
) -> dict[str, Any]:
    import numpy as np

    _require(
        reported.get("schema_version") == "p1.mstcn_checkpoint_diagnostic.same_truth_oracle.v1",
        "oracle schema changed",
    )
    _require(reported.get("oracle_epochs") == list(ORACLE_EPOCHS), "oracle epoch set changed")
    _require(reported.get("recipe_mutated") is False, "oracle mutated recipe")
    _require(reported.get("scientific_decision_mutated") is False, "oracle mutated decision")
    _require(reported.get("promotion_evidence") is False, "oracle became promotion evidence")
    _require(
        reported.get("official_probe_authorized") is False, "oracle authorizes official probing"
    )
    recipe_sha = _sha256(recipe_path)
    _require(
        reported.get("fixed_recipe_sha256_before") == recipe_sha, "oracle pre-recipe hash changed"
    )
    _require(
        reported.get("fixed_recipe_sha256_after") == recipe_sha, "oracle post-recipe hash changed"
    )
    _require(
        reported.get("fixed_scientific_decision") == _identity(decision_path),
        "oracle fixed-decision identity changed",
    )
    rows = reported.get("rows")
    _require(isinstance(rows, list) and len(rows) == len(ORACLE_EPOCHS), "oracle rows changed")
    _require([row.get("epoch") for row in rows] == list(ORACLE_EPOCHS), "oracle row order changed")
    recomputed: list[dict[str, Any]] = []
    for observed, epoch in zip(rows, ORACLE_EPOCHS, strict=True):
        index = CHECKPOINT_EPOCHS.index(epoch)
        truth_parts: list[Any] = []
        anchor_parts: list[Any] = []
        candidate_parts: list[Any] = []
        for phase in PHASES:
            truth = surfaces[phase]["truth"]
            anchor = surfaces[phase]["anchor"]
            candidate = arrays[phase]["candidate"][index]
            anchor_score = _binary_metrics(truth, anchor)
            candidate_score = _binary_metrics(truth, candidate)
            fold = observed.get("folds", {}).get(phase)
            _require(isinstance(fold, Mapping), f"oracle/{epoch}/{phase} is absent")
            _require(
                _same_number(fold.get("anchor_f1"), anchor_score["f1"]),
                f"oracle/{epoch}/{phase}/anchor changed",
            )
            _require(
                _same_number(fold.get("candidate_f1"), candidate_score["f1"]),
                f"oracle/{epoch}/{phase}/candidate changed",
            )
            _require(
                _same_number(fold.get("delta_f1"), candidate_score["f1"] - anchor_score["f1"]),
                f"oracle/{epoch}/{phase}/delta changed",
            )
            truth_parts.append(truth)
            anchor_parts.append(anchor)
            candidate_parts.append(candidate)
        truth = np.concatenate(truth_parts)
        anchor = np.concatenate(anchor_parts)
        candidate = np.concatenate(candidate_parts)
        anchor_score = _binary_metrics(truth, anchor)
        candidate_score = _binary_metrics(truth, candidate)
        delta = float(candidate_score["f1"] - anchor_score["f1"])
        added = (candidate == 1) & (anchor == 0)
        added_precision = float(truth[added].mean()) if added.any() else 0.0
        removed = int(np.sum((anchor == 1) & (candidate == 0)))
        expected = {
            "pooled_anchor_f1": float(anchor_score["f1"]),
            "pooled_candidate_f1": float(candidate_score["f1"]),
            "pooled_delta_f1": delta,
            "added_rows": int(added.sum()),
            "added_precision": added_precision,
            "anchor_positive_removed_rows": removed,
        }
        for name, value in expected.items():
            _require(_same_number(observed.get(name), value), f"oracle/{epoch}/{name} changed")
        recomputed.append({"epoch": epoch, **expected})
    best = max(recomputed, key=lambda row: (row["pooled_delta_f1"], -row["epoch"]))
    reported_best = reported.get("same_truth_oracle_best")
    _require(isinstance(reported_best, Mapping), "oracle best row is absent")
    _require(reported_best.get("epoch") == best["epoch"], "oracle best epoch changed")
    _require(
        _same_number(reported_best.get("pooled_delta_f1"), best["pooled_delta_f1"]),
        "oracle best delta changed",
    )
    return {
        "rows": recomputed,
        "best_epoch": best["epoch"],
        "best_delta_f1": best["pooled_delta_f1"],
    }


def _verify_split_and_encoder(
    split_path: Path,
    encoder_path: Path,
    receipt: Mapping[str, Any],
    *,
    phase: str,
) -> None:
    split = _load_json(split_path)
    _require(
        split.get("schema_version") == "p1.mstcn_asrf.phase_split.v1",
        f"{phase} split schema changed",
    )
    _require(
        split.get("phase") == phase and split.get("fold") == FOLDS[phase],
        f"{phase} split identity changed",
    )
    _require(
        split.get("holdout_rows") == receipt.get("holdout_rows"), f"{phase} split row count changed"
    )
    _require(
        split.get("holdout_key_sha256") == receipt.get("ordered_holdout_key_sha256"),
        f"{phase} split key hash changed",
    )
    _require(
        split.get("holdout_membership_sha256") == receipt.get("ordered_holdout_key_sha256"),
        f"{phase} membership hash changed",
    )
    _require(split.get("split_before_windowing") is True, f"{phase} was not split before windowing")
    _require(split.get("cross_split_window_count") == 0, f"{phase} has cross-split windows")
    _require(
        split.get("holdout_rows_used_to_fit_preprocessing") == 0,
        f"{phase} preprocessing leaked holdout",
    )
    _require(split.get("holdout_rows_used_to_train") == 0, f"{phase} training leaked holdout")
    _require(split.get("holdout_truth_columns_read") == 0, f"{phase} split opened truth")
    _require(
        split.get("runtime_input_features") == RUNTIME_INPUT_FEATURE_COUNT,
        f"{phase} input width changed",
    )
    _require(
        float(split.get("feature_non_overlap_slack_hours")) > 0.0
        and float(split.get("actual_separation_hours"))
        > float(split.get("required_feature_non_overlap_hours")),
        f"{phase} feature dependency supports overlap",
    )
    encoder = _load_json(encoder_path)
    required = {
        "center",
        "scale",
        "station_vocab",
        "layer_vocab",
        "depth_regime_vocab",
        "numeric_names",
        "fit_ids_sha256",
        "uses_supplied_depth_regime",
        "depth_thresholds",
        "preprocessing_fit_uses_holdout_rows",
    }
    _require(set(encoder) == required, f"{phase} encoder receipt fields changed")
    _require(
        encoder.get("preprocessing_fit_uses_holdout_rows") is False, f"{phase} encoder used holdout"
    )
    center = encoder.get("center")
    scale = encoder.get("scale")
    names = encoder.get("numeric_names")
    _require(
        isinstance(center, list) and isinstance(scale, list) and isinstance(names, list),
        f"{phase} encoder vectors are absent",
    )
    _require(
        len(center) == len(scale) == len(names) == MODEL_NUMERIC_FEATURE_COUNT,
        f"{phase} numeric encoder width changed",
    )
    _require(
        all(math.isfinite(float(value)) for value in center), f"{phase} encoder center is nonfinite"
    )
    _require(
        all(math.isfinite(float(value)) and float(value) > 0.0 for value in scale),
        f"{phase} encoder scale is invalid",
    )


def _verify_semantic_replays(path: Path) -> None:
    replays = _load_json(path)
    _require(set(replays) == set(PHASES), "semantic replay phase inventory changed")
    for phase in PHASES:
        row = replays[phase]
        _require(
            isinstance(row, Mapping)
            and row.get("schema_version") == "p1.mstcn_checkpoint_diagnostic.semantic_replay.v1"
            and row.get("phase") == phase
            and row.get("checkpoint_epochs") == list(CHECKPOINT_EPOCHS)
            and row.get("all_checkpoint_decoder_replays") is True
            and row.get("all_checkpoint_anchor_union_replays") is True
            and row.get("truth_columns_read") == 0
            and row.get("result") == "PASS",
            f"{phase} semantic replay changed",
        )


def _verify_control_flags(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "official_interface_reads":
                _require(child == 0, f"{label} records official interface reads")
            elif key in {
                "official_probe_authorized",
                "submission_created",
                "upload_performed",
                "fresh_promotion_evidence",
                "same_truth_oracle_promotion_evidence",
                "three_official_points_claimed",
            }:
                _require(child is False, f"{label}/{key} must remain false")
            _verify_control_flags(child, label=f"{label}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _verify_control_flags(child, label=f"{label}/{index}")


def _verify_decision_and_terminal(
    decision_path: Path,
    fixed_metrics_path: Path,
    oracle_path: Path,
    terminal_path: Path,
    recipe: Mapping[str, Any],
    *,
    recomputed_fixed: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    decision = _load_json(decision_path)
    expected_status = (
        "RETROSPECTIVE_FIXED_E150_IMPROVED"
        if float(recomputed_fixed["pooled_delta_f1"]) > 0.0
        else "RETROSPECTIVE_FIXED_E150_NOT_IMPROVED"
    )
    _require(
        decision.get("schema_version") == "p1.mstcn_checkpoint_diagnostic.fixed_decision.v1",
        "fixed decision schema changed",
    )
    _require(decision.get("experiment_id") == EXPERIMENT_ID, "fixed decision experiment changed")
    _require(decision.get("status") == expected_status, "fixed decision status changed")
    _require(decision.get("scientific_metric_epoch") == 150, "fixed decision epoch changed")
    recipe_path = decision_path.parent / "selected_recipe.json"
    _require(
        decision.get("selected_recipe_sha256") == _sha256(recipe_path),
        "fixed decision recipe hash changed",
    )
    _require(
        decision.get("fixed_metrics") == _identity(fixed_metrics_path),
        "fixed metric identity changed",
    )
    _require(
        decision.get("same_truth_oracle_computed_before_decision") is False,
        "oracle was marked before decision",
    )
    _require(
        decision.get("same_truth_oracle_may_mutate_decision") is False, "oracle may mutate decision"
    )

    terminal = _load_json(terminal_path)
    _require(
        terminal.get("schema_version") == "p1.mstcn_checkpoint_diagnostic.terminal.v1",
        "terminal schema changed",
    )
    _require(terminal.get("experiment_id") == EXPERIMENT_ID, "terminal experiment changed")
    _require(terminal.get("status") == expected_status, "terminal status differs from decision")
    _require(terminal.get("selected_recipe") == recipe, "terminal embedded recipe changed")
    _require(terminal.get("scientific_metric_epoch") == 150, "terminal metric epoch changed")
    _require(
        terminal.get("truth_scored_epochs") == [150], "terminal scientific truth epoch changed"
    )
    _require(
        terminal.get("same_truth_oracle_scored_epochs") == list(ORACLE_EPOCHS),
        "terminal oracle epochs changed",
    )
    _require(
        terminal.get("same_truth_oracle") == _identity(oracle_path),
        "terminal oracle identity changed",
    )
    _require(
        terminal.get("same_truth_oracle_mutated_fixed_decision") is False,
        "terminal records oracle mutation",
    )
    _parse_utc(terminal.get("started_at_utc"), label="terminal started_at_utc")
    _parse_utc(terminal.get("completed_at_utc"), label="terminal completed_at_utc")
    _require(
        _parse_utc(terminal["started_at_utc"], label="terminal started")
        <= _parse_utc(terminal["completed_at_utc"], label="terminal completed"),
        "terminal timestamps are reversed",
    )
    _verify_control_flags(decision, label="fixed decision")
    _verify_control_flags(terminal, label="terminal")
    return decision, terminal


def _verify_ordering(
    artifact_dir: Path,
    manifest: Mapping[str, Any],
    receipts: Mapping[str, Mapping[str, Any]],
    terminal: Mapping[str, Any],
) -> dict[str, Any]:
    names = [
        "q2_plateau_revalidation.json",
        "selected_recipe.json",
        "q3_blind_checkpoint_curve_receipt.json",
        "q4_blind_checkpoint_curve_receipt.json",
        "blind_semantic_replays.json",
        "fixed_epoch_150_metrics.json",
        "fixed_epoch_150_decision.json",
        "same_truth_oracle_diagnostic.json",
        "terminal_result.json",
        "manifest.json",
    ]
    mtimes = [(artifact_dir / name).stat().st_mtime_ns for name in names]
    _require(mtimes == sorted(mtimes), "artifact filesystem ordering is inconsistent")
    started = _parse_utc(terminal["started_at_utc"], label="terminal started")
    completed = _parse_utc(terminal["completed_at_utc"], label="terminal completed")
    q3_created = _parse_utc(receipts["q3"]["created_at_utc"], label="q3 receipt created")
    q4_created = _parse_utc(receipts["q4"]["created_at_utc"], label="q4 receipt created")
    manifest_created = _parse_utc(manifest.get("created_at_utc"), label="manifest created")
    _require(
        started <= q3_created <= q4_created <= completed <= manifest_created,
        "embedded timestamps are inconsistent",
    )
    recipe_mtime = datetime.fromtimestamp(
        (artifact_dir / "selected_recipe.json").stat().st_mtime, tz=UTC
    )
    _require(recipe_mtime <= q3_created, "recipe was not sealed before Q3 receipt")
    return {
        "filesystem_sequence": names,
        "embedded_sequence": [
            "terminal_started",
            "q3_receipt_created",
            "q4_receipt_created",
            "terminal_completed",
            "manifest_created",
        ],
        "post_hoc_limit": "BLIND_CHRONOLOGY_NOT_CRYPTOGRAPHICALLY_PROVABLE_POST_HOC",
    }


def _verify_attempt_lock(artifact_dir: Path, terminal: Mapping[str, Any]) -> dict[str, Any]:
    lock_path = artifact_dir.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
    _require(
        lock_path.is_file() and not lock_path.is_symlink(),
        "persistent one-shot attempt lock is absent",
    )
    lock = _load_json(lock_path)
    _require(
        lock.get("schema_version") == "p1.mstcn_checkpoint_diagnostic.attempt.v1",
        "attempt lock schema changed",
    )
    _require(lock.get("experiment_id") == EXPERIMENT_ID, "attempt lock experiment changed")
    _require(
        lock.get("config_sha256") == EXPECTED_CONFIG_SHA256, "attempt lock config hash changed"
    )
    _require(
        lock.get("runner_sha256") == EXPECTED_RUNNER_SHA256, "attempt lock runner hash changed"
    )
    _require(
        lock.get("one_shot") is True and lock.get("automatic_retry") is False,
        "attempt lock retry contract changed",
    )
    lock_created = _parse_utc(lock.get("created_at_utc"), label="attempt lock created")
    started = _parse_utc(terminal.get("started_at_utc"), label="terminal started")
    _require(lock_created <= started, "attempt lock was not created before execution start")
    return {
        "path": lock_path.name,
        "sha256": _sha256(lock_path),
        "created_at_utc": lock_created.isoformat(),
    }


def verify_artifact(
    *,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    project_root: Path = ROOT,
) -> dict[str, Any]:
    """Verify one complete namespace without writing to it or its project."""

    artifact_dir = artifact_dir.resolve()
    project_root = project_root.resolve()
    manifest, paths = _read_and_verify_manifest(artifact_dir)
    preflight, source_paths = _verify_preflight(paths["preflight.json"], project_root=project_root)
    q2, recipe = _verify_q2_and_recipe(
        paths["q2_plateau_revalidation.json"],
        paths["selected_recipe.json"],
        project_root=project_root,
    )
    receipts: dict[str, dict[str, Any]] = {}
    arrays: dict[str, dict[str, Any]] = {}
    for phase in PHASES:
        receipts[phase], arrays[phase] = _verify_phase(
            artifact_dir,
            paths[f"{phase}_blind_checkpoint_curve_receipt.json"],
            paths["selected_recipe.json"],
            phase=phase,
        )
        _verify_split_and_encoder(
            paths[f"{phase}_split.json"],
            paths[f"{phase}_encoder.json"],
            receipts[phase],
            phase=phase,
        )
    _verify_semantic_replays(paths["blind_semantic_replays.json"])
    surfaces = _load_historical_surfaces(source_paths, receipts, arrays)
    fixed_reported = _load_json(paths["fixed_epoch_150_metrics.json"])
    fixed_recomputed = _recompute_fixed_metrics(fixed_reported, surfaces, arrays)
    oracle_reported = _load_json(paths["same_truth_oracle_diagnostic.json"])
    oracle_recomputed = _recompute_oracle(
        oracle_reported,
        surfaces,
        arrays,
        recipe_path=paths["selected_recipe.json"],
        decision_path=paths["fixed_epoch_150_decision.json"],
    )
    _verify_control_flags(preflight, label="preflight")
    _verify_control_flags(q2, label="Q2 revalidation")
    _verify_control_flags(oracle_reported, label="same-truth oracle")
    _decision, terminal = _verify_decision_and_terminal(
        paths["fixed_epoch_150_decision.json"],
        paths["fixed_epoch_150_metrics.json"],
        paths["same_truth_oracle_diagnostic.json"],
        paths["terminal_result.json"],
        recipe,
        recomputed_fixed=fixed_recomputed,
    )
    ordering = _verify_ordering(artifact_dir, manifest, receipts, terminal)
    attempt_lock = _verify_attempt_lock(artifact_dir, terminal)
    return {
        "schema_version": "p1.mstcn_checkpoint_diagnostic.independent_qa.v2",
        "experiment_id": EXPERIMENT_ID,
        "artifact_directory": str(artifact_dir),
        "manifest_sha256": _sha256(artifact_dir / "manifest.json"),
        "manifest_files_verified": int(manifest["file_count_excluding_manifest"]),
        "attempt_lock": attempt_lock,
        "recipe": {"width": 512, "epoch": 150, "threshold": 0.8},
        "phase_receipts_verified": list(PHASES),
        "state_files_verified": len(PHASES) * len(SEEDS) * len(STATE_EPOCHS),
        "blind_npz_files_verified": len(PHASES),
        "fixed_epoch_150_recomputed": fixed_recomputed,
        "same_truth_oracle_recomputed": oracle_recomputed,
        "ordering": ordering,
        "official_interface_reads": 0,
        "submission_created": False,
        "upload_performed": False,
        "writes_performed": 0,
        "result": "ARTIFACT_INTERNAL_CONSISTENCY_PASS",
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = verify_artifact(
            artifact_dir=args.artifact_dir,
            project_root=args.project_root,
        )
    except IncompleteArtifactError as error:
        print(
            json.dumps(
                {
                    "schema_version": "p1.mstcn_checkpoint_diagnostic.independent_qa.v2",
                    "experiment_id": EXPERIMENT_ID,
                    "result": "WAIT_INCOMPLETE",
                    "reason": str(error),
                    "writes_performed": 0,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 3
    except VerificationError as error:
        print(
            json.dumps(
                {
                    "schema_version": "p1.mstcn_checkpoint_diagnostic.independent_qa.v2",
                    "experiment_id": EXPERIMENT_ID,
                    "result": "VERIFICATION_FAIL",
                    "reason": str(error),
                    "writes_performed": 0,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
