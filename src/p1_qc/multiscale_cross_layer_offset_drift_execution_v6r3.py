"""Process-isolated execution engine for P1 multiscale Gen6r3.

The parent process never receives inner labels.  Exactly fifteen fresh Windows
processes each own one cell and one empty target vault.  Persisted O_EXCL
commitments, rather than a cross-cell Python object, carry ordering state.
"""

from __future__ import annotations

import hashlib
import io
from array import array
from pathlib import Path
from typing import Any

try:
    _CONTEXT = _P1_V6R3_BOOTSTRAP_CONTEXT  # type: ignore[name-defined]  # noqa: F821
    contract = _P1_V6R3_AUTH_CONTRACT  # type: ignore[name-defined]  # noqa: F821
    science = _P1_V6R3_AUTH_SCIENCE  # type: ignore[name-defined]  # noqa: F821
    legacy = _P1_V6R3_AUTH_LEGACY_ENGINE  # type: ignore[name-defined]  # noqa: F821
    verifier = _P1_V6R3_AUTH_VERIFIER  # type: ignore[name-defined]  # noqa: F821
except NameError as exc:  # pragma: no cover - direct-import guard
    raise RuntimeError("P1 Gen6r3 engine requires the authenticated bootstrap") from exc

if (
    not isinstance(_CONTEXT, dict)
    or _CONTEXT.get("mode") not in {"parent", "cell_worker"}
    or _CONTEXT.get("all_owner_roles_authenticated") is not True
    or _CONTEXT.get("bootstrap_documents_prevalidated") is not True
):
    raise RuntimeError("P1 Gen6r3 engine loaded before execution authorization")

GENERATION = contract.GENERATION
TRAIN_COLUMNS = legacy.TRAIN_COLUMNS
TARGET_COLUMNS = legacy.TARGET_COLUMNS
GENESIS_SHA256 = hashlib.sha256(b"p1_v6r3_process_isolated_commitment_genesis").hexdigest()


class ExecutionError(RuntimeError):
    """The authenticated Gen6r3 execution failed closed."""


def _np_pd() -> tuple[Any, Any]:
    import numpy as np
    import pandas as pd

    verifier = _CONTEXT.get("verify_numerical_runtime")
    if not callable(verifier):
        raise ExecutionError("numerical runtime verifier is unavailable")
    verifier()
    return np, pd


def _ids_sha(values: Any) -> str:
    np, _pd = _np_pd()
    ids = np.asarray(values)
    if ids.dtype != np.dtype("int64") or ids.ndim != 1:
        raise ExecutionError("row identity must be exact int64")
    return hashlib.sha256(ids.astype("<i8", copy=False).tobytes(order="C")).hexdigest()


def _strict_ids(values: Any, *, size: int, label: str) -> Any:
    np, _pd = _np_pd()
    ids = np.asarray(values)
    if (
        ids.dtype != np.dtype("int64")
        or ids.ndim != 1
        or len(np.unique(ids)) != len(ids)
        or (len(ids) and (int(ids.min()) < 0 or int(ids.max()) >= size))
    ):
        raise ExecutionError(f"{label} IDs differ")
    return np.ascontiguousarray(ids)


class CellTargetVault:
    """One-cell target decoder whose state dies with its worker process."""

    def __init__(self, capability: object, raw_train: bytes, expected_rows: int, cell: int) -> None:
        live = contract.capability_snapshot(capability)
        if live["role"] != "cell_worker" or live["cell"] != cell or type(raw_train) is not bytes:
            raise PermissionError("cell target vault capability differs")
        self._capability = capability
        self._raw = raw_train
        self._cell = cell
        self._expected_rows = expected_rows
        self._offsets = array("Q")
        self._lengths = array("I")
        self._columns: dict[str, int] = {}
        self._labels: dict[int, int] = {}
        self._anomalies: dict[int, str] = {}
        self._events: list[dict[str, Any]] = []
        self._blocks: dict[int, tuple[Any, Any]] = {}
        self._outer_train: Any | None = None
        self._outer_validation: Any | None = None
        self._stage = "unconfigured"
        self._build_index()

    def _build_index(self) -> None:
        with io.BytesIO(self._raw) as stream:
            line, spans = legacy._csv_field_spans(stream.readline(), len(TRAIN_COLUMNS))
            header = tuple(legacy._decode_field(line, span) for span in spans)
            if header != TRAIN_COLUMNS:
                raise ExecutionError("cell vault train header differs")
            self._columns = {name: index for index, name in enumerate(header)}
            while True:
                offset = stream.tell()
                raw = stream.readline()
                if not raw:
                    break
                legacy._csv_field_spans(raw, len(TRAIN_COLUMNS))
                self._offsets.append(offset)
                self._lengths.append(len(raw))
        if len(self._offsets) != self._expected_rows:
            raise ExecutionError("cell vault row count differs")

    def configure(self, planned: list[Any], outer_train: Any, outer_validation: Any) -> None:
        if self._stage != "unconfigured" or len(planned) != 3:
            raise ExecutionError("cell vault configuration order differs")
        self._blocks = {
            split.block: (
                _strict_ids(split.train_ids, size=self._expected_rows, label="inner train"),
                _strict_ids(split.prediction_ids, size=self._expected_rows, label="inner holdout"),
            )
            for split in planned
        }
        if tuple(self._blocks) != contract.BLOCKS:
            raise ExecutionError("cell vault block identity differs")
        self._outer_train = _strict_ids(
            outer_train, size=self._expected_rows, label="outer train"
        )
        self._outer_validation = _strict_ids(
            outer_validation, size=self._expected_rows, label="outer validation"
        )
        np, _pd = _np_pd()
        if np.intersect1d(self._outer_train, self._outer_validation).size:
            raise ExecutionError("outer train/validation IDs overlap")
        self._stage = "inner_1_train"

    def _release(self) -> tuple[Any, Any]:
        """Decode only the one exact ID set authorized by the current phase.

        There is no ID or purpose argument to abuse through a direct call.  A
        commitment phase has no decode transition and therefore fails closed.
        """

        np, _pd = _np_pd()
        if self._stage.startswith("inner_") and self._stage.endswith("_train"):
            block = int(self._stage.split("_")[1])
            ids = self._blocks[block][0]
            purpose = f"cell_{self._cell:02d}_inner_{block}_train"
            next_stage = f"inner_{block}_commit"
        elif self._stage.startswith("inner_") and self._stage.endswith("_gate"):
            block = int(self._stage.split("_")[1])
            ids = self._blocks[block][1]
            purpose = f"cell_{self._cell:02d}_inner_{block}_gate"
            next_stage = "outer_train" if block == 3 else f"inner_{block + 1}_train"
        elif self._stage == "outer_train" and self._outer_train is not None:
            ids = self._outer_train
            purpose = f"cell_{self._cell:02d}_outer_train"
            next_stage = "outer_commit"
        else:
            raise PermissionError("current cell-vault phase has no target decode capability")
        ids = _strict_ids(ids, size=self._expected_rows, label=purpose)
        missing_labels = np.asarray(
            [item for item in ids if int(item) not in self._labels], dtype=np.int64
        )
        missing_anomalies = np.asarray(
            [item for item in ids if int(item) not in self._anomalies], dtype=np.int64
        )
        if len(missing_labels):
            decoded: dict[int, str] = {}
            index = self._columns["label"]
            with io.BytesIO(self._raw) as stream:
                for row_id in missing_labels.tolist():
                    stream.seek(int(self._offsets[row_id]))
                    line, spans = legacy._csv_field_spans(
                        stream.read(int(self._lengths[row_id])), len(TRAIN_COLUMNS)
                    )
                    decoded[int(row_id)] = legacy._decode_field(line, spans[index])
            if any(value not in {"0", "1"} for value in decoded.values()):
                raise ExecutionError("decoded label is not binary")
            self._labels.update({row: int(value) for row, value in decoded.items()})
        if len(missing_anomalies):
            decoded_anomaly: dict[int, str] = {}
            index = self._columns["anomaly_type"]
            with io.BytesIO(self._raw) as stream:
                for row_id in missing_anomalies.tolist():
                    stream.seek(int(self._offsets[row_id]))
                    line, spans = legacy._csv_field_spans(
                        stream.read(int(self._lengths[row_id])), len(TRAIN_COLUMNS)
                    )
                    decoded_anomaly[int(row_id)] = legacy._decode_field(line, spans[index])
            self._anomalies.update(decoded_anomaly)
        contract.bump_counter(
            self._capability, "target_decodes", len(missing_labels) + len(missing_anomalies)
        )
        self._events.append(
            {
                "purpose": purpose,
                "row_count": len(ids),
                "row_ids_sha256": _ids_sha(ids),
                "decoded_label_rows_new": len(missing_labels),
                "decoded_anomaly_rows_new": len(missing_anomalies),
            }
        )
        self._stage = next_stage
        return (
            np.asarray([self._labels[int(item)] for item in ids], dtype=np.int8),
            np.asarray([self._anomalies[int(item)] for item in ids], dtype=object),
        )

    def release_inner_train(self, block: int) -> tuple[Any, Any]:
        expected = f"inner_{block}_train"
        if self._stage != expected or block not in self._blocks:
            raise PermissionError("inner training release order differs")
        train_ids, prediction_ids = self._blocks[block]
        np, _pd = _np_pd()
        if np.intersect1d(train_ids, prediction_ids).size:
            raise ExecutionError("inner train/holdout IDs overlap")
        return self._release()

    def seal_inner(self, block: int, prediction_ids: Any, commitment_sha256: str) -> dict[str, Any]:
        if self._stage != f"inner_{block}_commit" or not contract._is_sha(commitment_sha256):
            raise PermissionError("inner commitment order differs")
        _train_ids, expected = self._blocks[block]
        ids = _strict_ids(prediction_ids, size=self._expected_rows, label="inner commitment")
        np, _pd = _np_pd()
        if not np.array_equal(ids, expected):
            raise ExecutionError("inner commitment IDs differ from planned holdout")
        already_decoded = int(sum(int(item) in self._labels or int(item) in self._anomalies for item in ids))
        if already_decoded != 0:
            raise PermissionError("current inner holdout was decoded before commitment")
        proof = {
            "row_count": len(ids),
            "row_ids_sha256": _ids_sha(ids),
            "already_decoded_row_scope_exposures_before_commitment": already_decoded,
            "target_scalars_decoded_before_commitment": 0,
            "commitment_sha256": commitment_sha256,
        }
        self._events.append({"purpose": f"cell_{self._cell:02d}_inner_{block}_sealed", **proof})
        self._stage = f"inner_{block}_gate"
        return proof

    def release_inner_gate(self, block: int) -> tuple[Any, Any]:
        if self._stage != f"inner_{block}_gate":
            raise PermissionError("inner gate release preceded commitment")
        _train_ids, prediction_ids = self._blocks[block]
        return self._release()

    def release_outer_train(self) -> tuple[Any, Any]:
        if self._stage != "outer_train" or self._outer_train is None:
            raise PermissionError("outer training release order differs")
        return self._release()

    def seal_outer(self, validation_ids: Any, commitment_sha256: str) -> dict[str, Any]:
        if self._stage != "outer_commit" or self._outer_validation is None:
            raise PermissionError("outer commitment order differs")
        ids = _strict_ids(validation_ids, size=self._expected_rows, label="outer commitment")
        np, _pd = _np_pd()
        if not np.array_equal(ids, self._outer_validation):
            raise ExecutionError("outer commitment IDs differ")
        already_decoded = int(sum(int(item) in self._labels or int(item) in self._anomalies for item in ids))
        if already_decoded != 0:
            raise PermissionError("active outer validation targets were decoded before commitment")
        proof = {
            "row_count": len(ids),
            "row_ids_sha256": _ids_sha(ids),
            "already_decoded_row_scope_exposures_before_commitment": already_decoded,
            "active_outer_target_scalars_decoded_before_commitment": 0,
            "commitment_sha256": commitment_sha256,
        }
        self._events.append({"purpose": f"cell_{self._cell:02d}_outer_sealed", **proof})
        self._stage = "sealed"
        return proof

    def audit(self) -> dict[str, Any]:
        if self._stage != "sealed":
            raise ExecutionError("cell target vault did not reach sealed state")
        return {
            "schema_version": "p1_v6r3_cell_target_vault_audit.v1",
            "cell": self._cell,
            "fresh_process_empty_at_start": True,
            "opaque_index_rows": len(self._offsets),
            "target_fields_decoded_while_indexing": 0,
            "decoded_label_rows": len(self._labels),
            "decoded_anomaly_rows": len(self._anomalies),
            "events": list(self._events),
            "outer_validation_decoded_in_worker": False,
        }


class ParentOuterTargetVault:
    """Parent-only outer decoder; it has no inner-target release method."""

    def __init__(
        self,
        capability: object,
        raw_train: bytes,
        expected_rows: int,
        validation_rows: dict[str, Any],
    ) -> None:
        live = contract.capability_snapshot(capability)
        if live["role"] != "parent":
            raise PermissionError("parent outer vault capability differs")
        self._capability = capability
        self._raw = raw_train
        self._expected_rows = expected_rows
        if type(validation_rows) is not dict or tuple(validation_rows) != contract.FOLDS:
            raise ExecutionError("parent outer validation map differs")
        self._validation = {
            fold: _strict_ids(values, size=expected_rows, label=f"{fold} outer validation")
            for fold, values in validation_rows.items()
        }
        self._released: set[str] = set()
        self._events: list[dict[str, Any]] = []
        self._offsets = array("Q")
        self._lengths = array("I")
        self._columns: dict[str, int] = {}
        with io.BytesIO(raw_train) as stream:
            line, spans = legacy._csv_field_spans(stream.readline(), len(TRAIN_COLUMNS))
            header = tuple(legacy._decode_field(line, span) for span in spans)
            if header != TRAIN_COLUMNS:
                raise ExecutionError("parent vault train header differs")
            self._columns = {name: index for index, name in enumerate(header)}
            while True:
                offset = stream.tell()
                raw = stream.readline()
                if not raw:
                    break
                legacy._csv_field_spans(raw, len(TRAIN_COLUMNS))
                self._offsets.append(offset)
                self._lengths.append(len(raw))
        if len(self._offsets) != expected_rows:
            raise ExecutionError("parent vault row count differs")

    def release_fold(self, fold: str, ids: Any, fold_commitment_sha256: str) -> tuple[Any, Any]:
        if fold not in contract.FOLDS or fold in self._released or not contract._is_sha(fold_commitment_sha256):
            raise PermissionError("parent outer target release scope differs")
        committed = _CONTEXT.get("is_fold_committed")
        if not callable(committed) or committed(fold, fold_commitment_sha256) is not True:
            raise PermissionError("outer truth release preceded persisted fold commitment")
        ids = _strict_ids(ids, size=self._expected_rows, label=f"{fold} outer validation")
        np, _pd = _np_pd()
        if not np.array_equal(ids, self._validation[fold]):
            raise PermissionError("parent outer release IDs differ from the frozen fold validation")
        labels_raw: list[str] = []
        anomalies: list[str] = []
        with io.BytesIO(self._raw) as stream:
            for row_id in ids.tolist():
                stream.seek(int(self._offsets[row_id]))
                line, spans = legacy._csv_field_spans(
                    stream.read(int(self._lengths[row_id])), len(TRAIN_COLUMNS)
                )
                labels_raw.append(legacy._decode_field(line, spans[self._columns["label"]]))
                anomalies.append(
                    legacy._decode_field(line, spans[self._columns["anomaly_type"]])
                )
        if any(value not in {"0", "1"} for value in labels_raw):
            raise ExecutionError("parent decoded label is not binary")
        labels = np.asarray([int(value) for value in labels_raw], dtype=np.int8)
        anomaly = np.asarray(anomalies, dtype=object)
        contract.bump_counter(self._capability, "target_decodes", len(ids) * 2)
        self._released.add(fold)
        self._events.append(
            {
                "fold": fold,
                "row_count": len(ids),
                "row_ids_sha256": _ids_sha(ids),
                "fold_commitment_sha256": fold_commitment_sha256,
                "released_after_fold_commitment": True,
            }
        )
        return labels, anomaly

    def audit(self) -> dict[str, Any]:
        if self._released != set(contract.FOLDS):
            raise ExecutionError("parent outer target releases are incomplete")
        return {
            "schema_version": "p1_v6r3_parent_outer_target_audit.v1",
            "inner_labels_received_from_workers": False,
            "fold_release_events": list(self._events),
        }


def _event_body(schema: str, prior: str, body: dict[str, Any]) -> dict[str, Any]:
    if not contract._is_sha(prior):
        raise ExecutionError("prior commitment hash differs")
    event = {
        "schema_version": schema,
        "generation": GENERATION,
        "prior_event_sha256": prior,
        **body,
    }
    event["event_sha256"] = contract.deep_sha256(event)
    return event


def _teacher_probability(
    capability: object,
    *,
    catalog: dict[str, Any],
    frame: Any,
    fold: str,
    fraction_tag: str,
    block: int,
    train_ids: Any,
    prediction_ids: Any,
) -> tuple[Any, Any]:
    receipt_catalog = _CONTEXT.get("teacher_receipt_catalog")
    if type(receipt_catalog) is not dict:
        raise ExecutionError("authenticated teacher receipt catalog is unavailable")
    artifact_root = contract.contained_path(
        Path(_CONTEXT["workspace"]),
        "artifacts/p1_incumbent_rule_distillation_neural_residual_v5r6",
        must_exist=True,
        kind="directory",
    )
    arrays: list[Any] = []
    for seed in contract.SEEDS:
        receipt = contract.verify_teacher_request(
            receipt_catalog,
            fold=fold,
            fraction_tag=fraction_tag,
            block=block,
            seed=seed,
            prediction_ids_sha256=_ids_sha(prediction_ids),
            train_ids_sha256=_ids_sha(train_ids),
            prediction_rows=len(prediction_ids),
            train_rows=len(train_ids),
        )
        relative = receipt["blind_prediction_relative_path"]
        expected = catalog["manifest"]["artifacts"][relative]
        if expected["sha256"] != receipt["blind_prediction_sha256"]:
            raise ExecutionError("teacher receipt/content hash differs")
        arrays.append(legacy._read_pinned_npy(artifact_root, relative, expected))
    probability = science.mean_seed_incumbent_probability(
        capability=capability, seed_probabilities=tuple(arrays)
    )
    if len(probability) != len(prediction_ids):
        raise ExecutionError("teacher probability row count differs")
    prediction_frame = frame.iloc[prediction_ids].loc[:, list(science.INPUT_ONLY_COLUMNS)].copy()
    prediction = science.fixed_incumbent_postprocess(
        capability=capability, frame=prediction_frame, probabilities=probability, fold=fold
    )
    return probability, prediction


def _commit_inner(
    capability: object,
    *,
    cell: int,
    block: int,
    fold: str,
    fraction: float,
    prediction_ids: Any,
    incumbent_probability: Any,
    incumbent_prediction: Any,
    candidate_probability: Any,
    candidate_prediction: Any,
    model_pin: dict[str, Any],
    split_proof_sha256: str,
    prior: str,
) -> tuple[dict[str, Any], str]:
    np, _pd = _np_pd()
    arrays = {
        "candidate_prediction": np.ascontiguousarray(candidate_prediction, dtype=np.int8),
        "candidate_probability": np.ascontiguousarray(candidate_probability, dtype=np.float32),
        "incumbent_prediction": np.ascontiguousarray(incumbent_prediction, dtype=np.int8),
        "incumbent_probability": np.ascontiguousarray(incumbent_probability, dtype=np.float32),
        "row_ids": np.ascontiguousarray(prediction_ids, dtype=np.int64),
    }
    payload = legacy._array_bundle_bytes(arrays)
    prediction_pin = contract.write_output_exclusive(
        capability, f"cells/cell_{cell:02d}/inner_predictions/block_{block}.bin", payload
    )
    event = _event_body(
        "p1_v6r3_inner_commitment.v1",
        prior,
        {
            "cell": cell,
            "block": block,
            "fold": fold,
            "fraction": fraction,
            "row_ids_sha256": _ids_sha(arrays["row_ids"]),
            "prediction_bundle": prediction_pin,
            "model": model_pin,
            "split_proof_sha256": split_proof_sha256,
        },
    )
    pin = contract.write_output_exclusive(
        capability, f"cells/cell_{cell:02d}/commitments/inner_{block}.json", event
    )
    contract.bump_counter(capability, "inner_commitments")
    return pin, event["event_sha256"]


def run_cell(capability: object, *, cell: int, prior_event_sha256: str) -> dict[str, Any]:
    """Run exactly one cell in one fresh process and return its sealed receipt."""

    live = contract.capability_snapshot(capability)
    if live["role"] != "cell_worker" or live["cell"] != cell or _CONTEXT.get("mode") != "cell_worker":
        raise PermissionError("run_cell requires the matching one-cell worker capability")
    fold, fraction, fraction_tag = contract.cell_identity(cell)
    np, _pd = _np_pd()
    base_config = _CONTEXT["base_config"]
    train_raw = _CONTEXT["authenticated_train_bytes_for_pin"](
        base_config["source_pins"]["train.csv"], "train.csv cell buffer"
    )
    frame = legacy.load_input_only_train(capability, train_raw)
    segment_ids = science.exact_gap_safe_segment_ids(frame)
    catalog = legacy._load_incumbent_catalog(capability)
    outer = legacy._load_outer_cell(
        capability, frame=frame, catalog=catalog, fold=fold, fraction=fraction
    )
    planned = science.build_three_block_inner_splits(
        capability=capability, metadata=frame, outer_prefix_ids=outer["prefix_ids"]
    )
    vault = CellTargetVault(capability, train_raw, len(frame), cell)
    vault.configure(planned, outer["prefix_ids"], outer["validation_ids"])
    inner_results: list[dict[str, Any]] = []
    split_audits: list[dict[str, Any]] = []
    model_audits: list[dict[str, Any]] = []
    prior = prior_event_sha256
    for split in planned:
        train_labels, train_anomaly = vault.release_inner_train(split.block)
        incumbent_probability, incumbent_prediction = _teacher_probability(
            capability,
            catalog=catalog,
            frame=frame,
            fold=fold,
            fraction_tag=fraction_tag,
            block=split.block,
            train_ids=split.train_ids,
            prediction_ids=split.prediction_ids,
        )
        slow_probability, model_pin, model_audit = legacy._fit_and_predict(
            capability,
            frame=frame,
            segment_ids=segment_ids,
            train_ids=split.train_ids,
            prediction_ids=split.prediction_ids,
            labels=train_labels,
            anomaly_types=train_anomaly,
            model_relative=f"cells/cell_{cell:02d}/models/inner_{split.block}.json",
        )
        candidate_probability, candidate_prediction, additions = science.protected_incumbent_union(
            capability=capability,
            incumbent_probability=incumbent_probability,
            incumbent_prediction=incumbent_prediction,
            gate_passed=slow_probability is not None,
            slow_probability=slow_probability,
            segment_ids=segment_ids[split.prediction_ids],
        )
        commitment_pin, event_sha = _commit_inner(
            capability,
            cell=cell,
            block=split.block,
            fold=fold,
            fraction=fraction,
            prediction_ids=split.prediction_ids,
            incumbent_probability=incumbent_probability,
            incumbent_prediction=incumbent_prediction,
            candidate_probability=candidate_probability,
            candidate_prediction=candidate_prediction,
            model_pin=model_pin,
            split_proof_sha256=contract.deep_sha256(model_audit["split_proof"]),
            prior=prior,
        )
        seal_proof = vault.seal_inner(split.block, split.prediction_ids, commitment_pin["sha256"])
        holdout_labels, holdout_anomaly = vault.release_inner_gate(split.block)
        score = science.score_candidate_delta(
            capability=capability,
            truth=holdout_labels,
            anomaly_type=holdout_anomaly,
            station_layer=legacy._station_layer(frame, split.prediction_ids),
            segment_ids=segment_ids[split.prediction_ids],
            incumbent_prediction=incumbent_prediction,
            candidate_prediction=candidate_prediction,
        )
        contract.bump_counter(capability, "scores")
        inner_results.append(
            {
                "truth": holdout_labels,
                "anomaly": holdout_anomaly,
                "station_layer": legacy._station_layer(frame, split.prediction_ids),
                "segment_ids": segment_ids[split.prediction_ids],
                "incumbent_prediction": incumbent_prediction,
                "candidate_prediction": candidate_prediction,
                "score": score,
                "additions": int(np.count_nonzero(additions)),
                "sealed_before_labels": True,
            }
        )
        split_audits.append(
            {
                **split.as_audit(),
                "dependency_proof": model_audit["split_proof"],
                "precommit_target_proof": seal_proof,
            }
        )
        model_audits.append({"role": "inner", "block": split.block, **model_audit})
        prior = event_sha
    gate = legacy._inner_gate(inner_results, capability)
    contract.bump_counter(capability, "scores")  # the 15 aggregate calls omitted by Gen6r2
    outer_labels, outer_anomaly = vault.release_outer_train()
    slow_probability, outer_model_pin, outer_model_audit = legacy._fit_and_predict(
        capability,
        frame=frame,
        segment_ids=segment_ids,
        train_ids=outer["prefix_ids"],
        prediction_ids=outer["validation_ids"],
        labels=outer_labels,
        anomaly_types=outer_anomaly,
        model_relative=f"cells/cell_{cell:02d}/models/outer.json",
    )
    candidate_probability, candidate_prediction, additions = science.protected_incumbent_union(
        capability=capability,
        incumbent_probability=outer["incumbent_probability"],
        incumbent_prediction=outer["incumbent_prediction"],
        gate_passed=bool(gate["passed"] and slow_probability is not None),
        slow_probability=slow_probability,
        segment_ids=segment_ids[outer["validation_ids"]],
    )
    arrays = {
        "candidate_prediction": np.ascontiguousarray(candidate_prediction, dtype=np.int8),
        "candidate_probability": np.ascontiguousarray(candidate_probability, dtype=np.float32),
        "incumbent_prediction": np.ascontiguousarray(outer["incumbent_prediction"], dtype=np.int8),
        "incumbent_probability": np.ascontiguousarray(outer["incumbent_probability"], dtype=np.float32),
        "row_ids": np.ascontiguousarray(outer["validation_ids"], dtype=np.int64),
    }
    outer_prediction_pin = contract.write_output_exclusive(
        capability,
        f"cells/cell_{cell:02d}/outer_prediction.bin",
        legacy._array_bundle_bytes(arrays),
    )
    outer_proof = vault.seal_outer(
        outer["validation_ids"], outer_prediction_pin["sha256"]
    )
    contract.bump_counter(capability, "cell_commitments")
    receipt = _event_body(
        "p1_v6r3_cell_receipt.v1",
        prior,
        {
            "session_sha256": live["session_sha256"],
            "cell": cell,
            "fold": fold,
            "fraction": fraction,
            "fraction_tag": fraction_tag,
            "row_ids_sha256": _ids_sha(outer["validation_ids"]),
            "outer_prediction_bundle": outer_prediction_pin,
            "outer_model": outer_model_pin,
            "train_only_gate": gate,
            "outer_additions": int(np.count_nonzero(additions)),
            "split_audits": split_audits,
            "model_audits": [*model_audits, {"role": "outer", **outer_model_audit}],
            "outer_precommit_target_proof": outer_proof,
            "target_vault_audit": vault.audit(),
            "worker_counters": contract.capability_snapshot(capability)["counters"],
        },
    )
    pin = contract.write_output_exclusive(
        capability, f"cells/cell_{cell:02d}/cell_receipt.json", receipt
    )
    return {
        "cell": cell,
        "receipt": pin,
        "event_sha256": receipt["event_sha256"],
        "worker_counters": contract.capability_snapshot(capability)["counters"],
    }


def _load_outer_result(
    capability: object,
    *,
    frame: Any,
    catalog: dict[str, Any],
    cell: int,
    worker_result: dict[str, Any],
) -> dict[str, Any]:
    fold, fraction, _tag = contract.cell_identity(cell)
    receipt_pin = worker_result.get("receipt")
    reader = _CONTEXT.get("authenticated_output_bytes")
    if not callable(reader) or type(receipt_pin) is not dict:
        raise ExecutionError("worker receipt reader is unavailable")
    receipt = contract.parse_json_bytes(reader(receipt_pin, "cell receipt"), label="cell receipt")
    if (
        receipt.get("cell") != cell
        or receipt.get("fold") != fold
        or receipt.get("fraction") != fraction
        or receipt.get("session_sha256")
        != contract.capability_snapshot(capability)["session_sha256"]
        or receipt.get("event_sha256") != worker_result.get("event_sha256")
        or contract.deep_sha256({key: value for key, value in receipt.items() if key != "event_sha256"})
        != receipt.get("event_sha256")
    ):
        raise ExecutionError("worker receipt identity differs")
    bundle_pin = receipt["outer_prediction_bundle"]
    arrays = legacy._load_array_bundle_bytes(reader(bundle_pin, "outer prediction bundle"))
    outer = legacy._load_outer_cell(
        capability, frame=frame, catalog=catalog, fold=fold, fraction=fraction
    )
    np, _pd = _np_pd()
    if (
        not np.array_equal(arrays["row_ids"], outer["validation_ids"])
        or not np.array_equal(arrays["incumbent_prediction"], outer["incumbent_prediction"])
        or arrays["candidate_prediction"].dtype != np.dtype("int8")
        or len(arrays["candidate_prediction"]) != len(outer["validation_ids"])
    ):
        raise ExecutionError("worker outer prediction binding differs")
    return {
        **outer,
        "cell": cell,
        "candidate_probability": arrays["candidate_probability"],
        "candidate_prediction": arrays["candidate_prediction"],
        "gate": receipt["train_only_gate"],
        "receipt": receipt,
    }


def _write_parent_event(
    capability: object,
    *,
    relative: str,
    schema: str,
    prior: str,
    body: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    event = _event_body(schema, prior, body)
    pin = contract.write_output_exclusive(capability, relative, event)
    return pin, event["event_sha256"]


def run_parent(capability: object) -> dict[str, Any]:
    """Launch the sole authenticated 15-worker research curve."""

    live = contract.capability_snapshot(capability)
    if live["role"] != "parent" or _CONTEXT.get("mode") != "parent":
        raise PermissionError("run_parent requires the parent capability")
    np, _pd = _np_pd()
    base_config = _CONTEXT["base_config"]
    train_raw = _CONTEXT["authenticated_train_bytes_for_pin"](
        base_config["source_pins"]["train.csv"], "train.csv parent buffer"
    )
    frame = legacy.load_input_only_train(capability, train_raw)
    catalog = legacy._load_incumbent_catalog(capability)
    validation_rows = {
        fold: legacy._load_outer_cell(
            capability,
            frame=frame,
            catalog=catalog,
            fold=fold,
            fraction=contract.FRACTIONS[0],
        )["validation_ids"]
        for fold in contract.FOLDS
    }
    target_vault = ParentOuterTargetVault(
        capability, train_raw, len(frame), validation_rows
    )
    launch = _CONTEXT.get("launch_cell_worker")
    validate_chain = _CONTEXT.get("validate_persisted_chain")
    if not callable(launch) or not callable(validate_chain):
        raise ExecutionError("cell worker launcher/chain validator is unavailable")
    session_pin, head = _write_parent_event(
        capability,
        relative="commitments/session.json",
        schema="p1_v6r3_session.v1",
        prior=GENESIS_SHA256,
        body={
            "session_sha256": live["session_sha256"],
            "workers": 15,
            "process_model": "fresh_windows_spawn_one_cell_per_worker",
            "parent_never_receives_inner_labels": True,
        },
    )
    all_cells: list[dict[str, Any]] = []
    worker_counter_rows: list[dict[str, int]] = []
    fold_deltas: dict[str, float] = {}
    fold_commitments: dict[str, dict[str, Any]] = {}
    for cell in range(1, contract.CELL_COUNT + 1):
        validate_chain(expected_next_cell=cell, expected_head=head, session_sha256=live["session_sha256"])
        worker_result = launch(
            cell=cell, session_sha256=live["session_sha256"], prior_event_sha256=head
        )
        if type(worker_result) is not dict or worker_result.get("cell") != cell:
            raise ExecutionError("cell worker result differs")
        graph = verifier.verify_cell_artifact_graph(
            capability,
            cell=cell,
            worker_result=worker_result,
            prior_event_sha256=head,
            session_sha256=live["session_sha256"],
        )
        if graph.get("head_event_sha256") != worker_result.get("event_sha256"):
            raise ExecutionError("independent cell artifact graph head differs")
        loaded = _load_outer_result(
            capability, frame=frame, catalog=catalog, cell=cell, worker_result=worker_result
        )
        head = worker_result["event_sha256"]
        validate_chain(
            expected_next_cell=cell + 1,
            expected_head=head,
            session_sha256=live["session_sha256"],
        )
        all_cells.append(loaded)
        worker_counter_rows.append(worker_result["worker_counters"])
        contract.bump_counter(capability, "worker_processes")
        if cell % 5 == 0:
            fold = contract.FOLDS[cell // 5 - 1]
            fold_cells = [item for item in all_cells if item["fold"] == fold]
            if len(fold_cells) != 5:
                raise ExecutionError("fold cell cardinality differs")
            fold_pin, head = _write_parent_event(
                capability,
                relative=f"commitments/fold_{fold}.json",
                schema="p1_v6r3_fold_commitment.v1",
                prior=head,
                body={
                    "session_sha256": live["session_sha256"],
                    "fold": fold,
                    "cells": [item["cell"] for item in fold_cells],
                    "cell_receipt_sha256": [item["receipt"]["event_sha256"] for item in fold_cells],
                    "active_fold_target_scalars_decoded_before_commitment": 0,
                },
            )
            contract.bump_counter(capability, "fold_commitments")
            fold_commitments[fold] = fold_pin
            truth, anomaly = target_vault.release_fold(
                fold, fold_cells[0]["validation_ids"], fold_pin["sha256"]
            )
            for item in fold_cells:
                if not np.array_equal(item["validation_ids"], fold_cells[0]["validation_ids"]):
                    raise ExecutionError("validation IDs differ across fold fractions")
                item["truth"] = truth
                item["anomaly"] = anomaly
            full_cell = [item for item in fold_cells if item["fraction"] == 1.0]
            if len(full_cell) != 1:
                raise ExecutionError("full-fraction fold cell differs")
            aggregate = legacy._aggregate_score(capability, frame, full_cell)
            fold_deltas[fold] = aggregate["metrics"]["micro_f1_delta"]
            contract.write_output_exclusive(
                capability,
                f"metrics/fold_{fold}.json",
                {
                    "schema_version": "p1_v6r3_fold_metrics.v1",
                    "fold": fold,
                    "score_call_ordinal": 60 + len(fold_deltas),
                    "metrics": aggregate["metrics"],
                },
            )
    completion_pin, head = _write_parent_event(
        capability,
        relative="commitments/predictions_complete.json",
        schema="p1_v6r3_predictions_complete.v1",
        prior=head,
        body={
            "session_sha256": live["session_sha256"],
            "inner_commitments": 45,
            "cell_commitments": 15,
            "fold_commitments": 3,
            "worker_processes": 15,
            "parent_received_inner_labels": False,
            "candidate_created": False,
            "test_prediction_created": False,
            "ledger_appended": False,
            "uploaded": False,
        },
    )
    contract.bump_counter(capability, "predictions_complete")
    fraction_points: list[dict[str, Any]] = []
    for index, (fraction, tag) in enumerate(zip(contract.FRACTIONS, contract.FRACTION_TAGS, strict=True)):
        cells = [item for item in all_cells if item["fraction"] == fraction]
        aggregate = legacy._aggregate_score(capability, frame, cells)
        ci90 = science.paired_bootstrap_f1_delta_ci90(
            capability=capability,
            truth=aggregate["truth"],
            incumbent_prediction=aggregate["incumbent_prediction"],
            candidate_prediction=aggregate["candidate_prediction"],
            bootstrap_unit_ids=aggregate["bootstrap_units"],
            replicates=5000,
            seed=20260823 + index,
        )
        contract.bump_counter(capability, "bootstrap_replicates", 5000)
        point = {
            "fraction": fraction,
            "micro_f1_delta": aggregate["metrics"]["micro_f1_delta"],
            "ci90": ci90,
            "offset_recall_delta": aggregate["metrics"]["offset_recall_delta"],
            "drift_recall_delta": aggregate["metrics"]["drift_recall_delta"],
            "spike_f1_delta": aggregate["metrics"]["spike_f1_delta"],
            "worst_station_layer_f1_delta": aggregate["metrics"]["worst_station_layer_f1_delta"],
            "bootstrap_replicates": 5000,
            "offset_observed": aggregate["offset_observed"],
            "drift_observed": aggregate["drift_observed"],
            "spike_observed": aggregate["spike_observed"],
            "all_required_station_layers_observed": aggregate[
                "all_required_station_layers_observed"
            ],
        }
        fraction_points.append(point)
        contract.write_output_exclusive(
            capability,
            f"metrics/fraction_{tag}.json",
            {
                "schema_version": "p1_v6r3_fraction_metrics.v1",
                "score_call_ordinal": 63 + index + 1,
                **point,
            },
        )
    split_audits = [
        {"cell": item["cell"], "proofs": item["receipt"]["split_audits"]} for item in all_cells
    ]
    model_audits = [
        {"cell": item["cell"], "models": item["receipt"]["model_audits"]} for item in all_cells
    ]
    reproducible = all(
        audit.get("available") is False or audit.get("reload_inference_byte_exact") is True
        for cell in model_audits
        for audit in cell["models"]
    )
    final_input = {
        "fraction_metrics": fraction_points,
        "fold_full_micro_f1_deltas": fold_deltas,
        "all_leakage_checks": all(
            proof["dependency_proof"]["passed"]
            and proof["precommit_target_proof"][
                "already_decoded_row_scope_exposures_before_commitment"
            ]
            == 0
            for item in split_audits
            for proof in item["proofs"]
        ),
        "all_reproducibility_checks": reproducible,
        "all_commitments_verified": True,
    }
    final_gate = science.strict_final_curve_gate(final_input)
    merged_counters: dict[str, int] = {}
    for row in [*worker_counter_rows, contract.capability_snapshot(capability)["counters"]]:
        for key, value in row.items():
            if key != "files_written":
                merged_counters[key] = merged_counters.get(key, 0) + int(value)
    for key in ("test_value_reads", "candidate_files", "ledger_appends", "uploads"):
        merged_counters.setdefault(key, 0)
    contract.verify_completion_counters(merged_counters)
    contract.write_output_exclusive(
        capability,
        "split_audit.json",
        {
            "schema_version": "p1_v6r3_split_audit.v1",
            "worker_processes": 15,
            "proofs": split_audits,
            "all_passed": final_input["all_leakage_checks"],
        },
    )
    contract.write_output_exclusive(capability, "selective_target_audit.json", target_vault.audit())
    metrics = {
        "schema_version": "p1_v6r3_metrics.v1",
        "generation": GENERATION,
        **final_input,
        "final_gate": final_gate,
        "score_call_decomposition": contract.SCORE_DECOMPOSITION,
    }
    evidence = {
        "schema_version": "p1_v6r3_learning_curve_evidence.v1",
        "generation": GENERATION,
        "points": fraction_points,
        "fold_full_micro_f1_deltas": fold_deltas,
        "cell_gates": [item["gate"] for item in all_cells],
        "session": session_pin,
        "fold_commitments": fold_commitments,
        "predictions_complete": completion_pin,
        "commitment_chain_head_sha256": head,
        "worker_process_isolation": True,
        "candidate_creation_allowed": False,
        "test_prediction_allowed": False,
        "ledger_append_allowed": False,
        "upload_allowed": False,
    }
    result = {
        "schema_version": "p1_v6r3_result.v1",
        "generation": GENERATION,
        "status": (
            "RESEARCH_ONLY_CURVE_GATE_PASS_NO_CANDIDATE"
            if final_gate["passed"]
            else "RESEARCH_ONLY_NO_PASS"
        ),
        "passed": bool(final_gate["passed"]),
        "fallback": final_gate["fallback"],
        "candidate": None,
        "test_prediction": None,
        "ledger_event": None,
        "upload": None,
    }
    metrics_pin = contract.write_output_exclusive(capability, "metrics.json", metrics)
    evidence_pin = contract.write_output_exclusive(
        capability, "learning_curve_evidence.json", evidence
    )
    result_pin = contract.write_output_exclusive(capability, "result.json", result)
    resource_pin = contract.write_output_exclusive(
        capability,
        "resource_audit.json",
        {
            "schema_version": "p1_v6r3_resource_audit.v1",
            "exact_counters": merged_counters,
            "model_audits": model_audits,
            "worker_processes": 15,
            "cross_worker_memory_inheritance": False,
        },
    )
    inventory = _CONTEXT["output_inventory"](final=False)
    manifest = {
        "schema_version": "p1_v6r3_manifest.v1",
        "generation": GENERATION,
        "pre_manifest_inventory_sha256": inventory["inventory_sha256"],
        "commitment_chain_head_sha256": head,
        "metrics": metrics_pin,
        "learning_curve_evidence": evidence_pin,
        "result": result_pin,
        "resource_audit": resource_pin,
        "candidate_created": False,
        "test_prediction_created": False,
        "ledger_appended": False,
        "uploaded": False,
    }
    manifest_pin = contract.write_output_exclusive(capability, "manifest.json", manifest)
    contract.write_output_exclusive(
        capability, "manifest.sha256", (manifest_pin["sha256"] + "\n").encode("ascii")
    )
    pre_seal_inventory = _CONTEXT["output_inventory"](final=False)
    if pre_seal_inventory.get("files") != 201:
        raise ExecutionError("pre-seal Gen6r3 file count differs")
    final_seal_pin = contract.write_output_exclusive(
        capability,
        "final_seal.json",
        {
            "schema_version": "p1_v6r3_final_seal.v1",
            "generation": GENERATION,
            "pre_seal_files": 201,
            "pre_seal_inventory_sha256": pre_seal_inventory["inventory_sha256"],
            "expected_final_files": 202,
            "commitment_chain_head_sha256": head,
            "manifest": manifest_pin,
            "exact_counters": merged_counters,
            "candidate_created": False,
            "test_prediction_created": False,
            "ledger_appended": False,
            "uploaded": False,
        },
    )
    final_inventory = verifier.verify_final_output(
        capability,
        final_seal_pin=final_seal_pin,
        commitment_chain_head_sha256=head,
    )
    contract.close_capability(capability)
    return {
        "status": result["status"],
        "passed": result["passed"],
        "manifest_sha256": manifest_pin["sha256"],
        "files": final_inventory["files"],
        "scores": merged_counters["scores"],
        "candidate_created": False,
        "test_prediction_created": False,
        "ledger_appended": False,
        "uploaded": False,
    }


__all__ = ["CellTargetVault", "ExecutionError", "ParentOuterTargetVault", "run_cell", "run_parent"]
