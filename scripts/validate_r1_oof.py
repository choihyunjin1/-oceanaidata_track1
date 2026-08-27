"""Independently validate an R1 candidate OOF against its frozen baseline OOF.

Only aggregate metrics, counts, provenance hashes, and validation diagnostics
are written.  No station-layer-time row or observed value is exported.
Outer labels are used for one-time evaluation only and must never feed model or
post-processing selection.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from p1_qc.data import ANOMALY_TYPES, KEY_COLUMNS, parse_anomaly_types, sha256_file
from p1_qc.metrics import BinaryCounts, binary_counts, group_row_shares, weighted_group_counts
from p1_qc.validation import normal_station_layer_day_fp, paired_block_bootstrap

BASELINE_REQUIRED = (*KEY_COLUMNS, "fold", "label", "probability", "prediction")
CANDIDATE_REQUIRED = (
    *KEY_COLUMNS,
    "fold",
    "label",
    "probability",
    "base_prediction",
    "prediction",
)
TEST_REQUIRED = ("station", "layer")
COUNT_FIELDS = ("tp", "fp", "fn", "tn")
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260813
LONG_EVENT_ROWS = 288


class R1OOFValidationError(ValueError):
    """Raised when an R1 OOF comparison violates the independent-QA contract."""


def _read_table(path: str | Path) -> tuple[Path, pd.DataFrame]:
    source = Path(path).expanduser().resolve(strict=True)
    suffix = source.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(source)
    elif suffix == ".csv":
        frame = pd.read_csv(source, keep_default_na=False, low_memory=False)
    else:
        raise R1OOFValidationError(f"unsupported table format {suffix!r}; expected CSV or Parquet")
    return source, frame


def _require_columns(frame: pd.DataFrame, required: Sequence[str], *, role: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise R1OOFValidationError(f"{role} is missing required columns: {missing}")
    if len(frame) == 0:
        raise R1OOFValidationError(f"{role} is empty")


def _binary_array(values: Sequence[Any], *, role: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise R1OOFValidationError(f"{role} must contain numeric 0/1 values") from exc
    if array.ndim != 1 or not np.isfinite(array).all() or not np.isin(array, [0, 1]).all():
        raise R1OOFValidationError(f"{role} must contain finite numeric 0/1 values")
    return array.astype(np.int8)


def _probability_array(values: Sequence[Any], *, role: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise R1OOFValidationError(f"{role} probability must be numeric") from exc
    if array.ndim != 1 or not np.isfinite(array).all():
        raise R1OOFValidationError(f"{role} probability must be a finite vector")
    if ((array < 0) | (array > 1)).any():
        raise R1OOFValidationError(f"{role} probability must lie in [0, 1]")
    return array


def _validate_oof(frame: pd.DataFrame, *, role: str, candidate: bool) -> dict[str, np.ndarray]:
    _require_columns(frame, CANDIDATE_REQUIRED if candidate else BASELINE_REQUIRED, role=role)
    if frame.loc[:, list(KEY_COLUMNS)].isna().any().any():
        raise R1OOFValidationError(f"{role} has null key values")
    if frame["fold"].isna().any() or frame["label"].isna().any():
        raise R1OOFValidationError(f"{role} has null fold/label values")
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise R1OOFValidationError(f"{role} contains duplicate OOF keys")
    result = {
        "label": _binary_array(frame["label"], role=f"{role} label"),
        "probability": _probability_array(frame["probability"], role=role),
        "prediction": _binary_array(frame["prediction"], role=f"{role} prediction"),
    }
    if candidate:
        result["base_prediction"] = _binary_array(
            frame["base_prediction"], role="candidate base_prediction"
        )
    return result


def _assert_exact_alignment(candidate: pd.DataFrame, baseline: pd.DataFrame) -> None:
    if len(candidate) != len(baseline):
        raise R1OOFValidationError(
            f"candidate/baseline OOF row counts differ: {len(candidate)} != {len(baseline)}"
        )
    for column in (*KEY_COLUMNS, "fold", "label"):
        left = candidate[column].reset_index(drop=True)
        right = baseline[column].reset_index(drop=True)
        if not left.equals(right):
            mismatch_count = int((~left.eq(right)).sum())
            raise R1OOFValidationError(
                f"candidate/baseline {column} order differs in {mismatch_count} rows"
            )


def _counts_payload(counts: BinaryCounts, *, weighted: bool = False) -> dict[str, Any]:
    convert = float if weighted else lambda value: int(round(float(value)))
    return {
        "tp": convert(counts.tp),
        "fp": convert(counts.fp),
        "fn": convert(counts.fn),
        "tn": convert(counts.tn),
        "precision": float(counts.precision),
        "recall": float(counts.recall),
        "f1": float(counts.f1),
        "support": convert(counts.support),
        "predicted_positive": convert(counts.predicted_positive),
    }


def _paired_metrics(
    truth: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    metadata: pd.DataFrame | None = None,
    group_weights: Mapping[Any, float] | None = None,
) -> dict[str, Any]:
    if group_weights is None:
        candidate_counts = binary_counts(truth, candidate)
        baseline_counts = binary_counts(truth, baseline)
        weighted = False
    else:
        if metadata is None:
            raise R1OOFValidationError("weighted metrics require metadata")
        candidate_counts = weighted_group_counts(truth, candidate, metadata, group_weights)
        baseline_counts = weighted_group_counts(truth, baseline, metadata, group_weights)
        weighted = True
    candidate_payload = _counts_payload(candidate_counts, weighted=weighted)
    baseline_payload = _counts_payload(baseline_counts, weighted=weighted)
    return {
        "candidate": candidate_payload,
        "baseline": baseline_payload,
        "delta": {
            "f1": candidate_payload["f1"] - baseline_payload["f1"],
            "precision": candidate_payload["precision"] - baseline_payload["precision"],
            "recall": candidate_payload["recall"] - baseline_payload["recall"],
        },
    }


def _dimension_metrics(
    frame: pd.DataFrame,
    truth: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    work = frame.loc[:, list(columns)].reset_index(drop=True).copy()
    work["__position"] = np.arange(len(work), dtype=np.int64)
    grouper: str | list[str] = list(columns)
    if len(columns) == 1:
        grouper = columns[0]
    rows: list[dict[str, Any]] = []
    for key, part in work.groupby(grouper, sort=True, observed=True, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        positions = part["__position"].to_numpy(dtype=np.int64)
        record = {
            column: _json_scalar(value) for column, value in zip(columns, key_tuple, strict=True)
        }
        record.update(
            {
                "rows": len(positions),
                "positive_rows": int(truth[positions].sum()),
                **_paired_metrics(truth[positions], candidate[positions], baseline[positions]),
            }
        )
        rows.append(record)
    return rows


def _type_metrics(
    anomaly_type: pd.Series,
    truth: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
) -> list[dict[str, Any]]:
    membership = parse_anomaly_types(anomaly_type, strict=True)
    text = anomaly_type.astype("string").fillna("").str.strip()
    if text[truth == 0].ne("").any():
        raise R1OOFValidationError("baseline anomaly_type is populated on normal OOF rows")
    if text[truth == 1].eq("").any():
        raise R1OOFValidationError("baseline anomaly_type is blank on positive OOF rows")
    rows: list[dict[str, Any]] = []
    for anomaly in ANOMALY_TYPES:
        mask = membership[anomaly].to_numpy(dtype=bool) & (truth == 1)
        support = int(mask.sum())
        candidate_tp = int(candidate[mask].sum())
        baseline_tp = int(baseline[mask].sum())
        candidate_recall = candidate_tp / support if support else None
        baseline_recall = baseline_tp / support if support else None
        rows.append(
            {
                "anomaly_type": anomaly,
                "positive_rows": support,
                "candidate_tp": candidate_tp,
                "candidate_fn": support - candidate_tp,
                "candidate_recall": candidate_recall,
                "baseline_tp": baseline_tp,
                "baseline_fn": support - baseline_tp,
                "baseline_recall": baseline_recall,
                "delta_recall": (candidate_recall - baseline_recall if support else None),
            }
        )
    return rows


def _positive_event_ids(frame: pd.DataFrame, truth: np.ndarray) -> np.ndarray:
    work = frame.loc[:, ["station", "layer", "time"]].reset_index(drop=True).copy()
    work["__position"] = np.arange(len(work), dtype=np.int64)
    work["__truth"] = truth.astype(bool)
    work["__time"] = pd.to_datetime(work["time"], errors="coerce", utc=True, format="mixed")
    if work["__time"].isna().any():
        raise R1OOFValidationError("OOF time contains unparseable timestamps")
    ordered = work.sort_values(["station", "layer", "__time", "__position"], kind="mergesort")
    grouped = ordered.groupby(["station", "layer"], sort=False, observed=True)
    contiguous = grouped["__time"].diff().dt.total_seconds().eq(10 * 60)
    prior = grouped["__truth"].shift(1).fillna(False).astype(bool)
    starts = ordered["__truth"] & (~contiguous | ~prior)
    ordered["__event"] = starts.cumsum().where(ordered["__truth"], -1).astype(np.int64)
    return ordered.sort_values("__position", kind="mergesort")["__event"].to_numpy()


def _long_event_recall(
    frame: pd.DataFrame,
    truth: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, Any]:
    event = _positive_event_ids(frame, truth)
    positive_ids = event[event >= 0]
    if len(positive_ids):
        event_ids, event_sizes = np.unique(positive_ids, return_counts=True)
        long_ids = event_ids[event_sizes >= LONG_EVENT_ROWS]
        mask = np.isin(event, long_ids)
    else:
        long_ids = np.empty(0, dtype=np.int64)
        mask = np.zeros(len(frame), dtype=bool)
    rows = int(mask.sum())
    candidate_tp = int(candidate[mask].sum())
    baseline_tp = int(baseline[mask].sum())
    candidate_recall = candidate_tp / rows if rows else None
    baseline_recall = baseline_tp / rows if rows else None
    return {
        "minimum_rows": LONG_EVENT_ROWS,
        "minimum_hours_at_10_min_cadence": 48,
        "events": int(len(long_ids)),
        "positive_rows": rows,
        "candidate_tp": candidate_tp,
        "candidate_fn": rows - candidate_tp,
        "candidate_row_recall": candidate_recall,
        "baseline_tp": baseline_tp,
        "baseline_fn": rows - baseline_tp,
        "baseline_row_recall": baseline_recall,
        "delta_row_recall": (candidate_recall - baseline_recall if rows else None),
    }


def _normal_day_fp_rate(
    frame: pd.DataFrame,
    truth: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, Any]:
    try:
        return normal_station_layer_day_fp(truth, candidate, baseline, frame)
    except (KeyError, TypeError, ValueError) as exc:
        raise R1OOFValidationError(str(exc)) from exc


def _source_payload(path: Path) -> dict[str, Any]:
    return {
        "format": path.suffix.lower().lstrip("."),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if pd.isna(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _assert_additive_breakdown(
    overall: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    dimension: str,
) -> None:
    if sum(int(record["rows"]) for record in records) != int(
        overall["candidate"]["tp"]
        + overall["candidate"]["fp"]
        + overall["candidate"]["fn"]
        + overall["candidate"]["tn"]
    ):
        raise RuntimeError(f"{dimension} rows do not reconcile to overall population")
    for model in ("candidate", "baseline"):
        for field in COUNT_FIELDS:
            subtotal = sum(int(record[model][field]) for record in records)
            if subtotal != int(overall[model][field]):
                raise RuntimeError(
                    f"{dimension}/{model}/{field} does not reconcile to overall counts"
                )


def build_r1_oof_report(
    candidate_oof: str | Path,
    baseline_oof: str | Path,
    test_csv: str | Path,
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Build an aggregate-only, deterministic R1-vs-baseline OOF QA report."""

    if bootstrap_replicates < 1:
        raise R1OOFValidationError("bootstrap_replicates must be positive")
    candidate_path, candidate_frame = _read_table(candidate_oof)
    baseline_path, baseline_frame = _read_table(baseline_oof)
    test_path, test_frame = _read_table(test_csv)
    _require_columns(test_frame, TEST_REQUIRED, role="test CSV")
    if test_frame.loc[:, list(TEST_REQUIRED)].isna().any().any():
        raise R1OOFValidationError("test CSV has null station/layer values")

    candidate_arrays = _validate_oof(candidate_frame, role="candidate OOF", candidate=True)
    baseline_arrays = _validate_oof(baseline_frame, role="baseline OOF", candidate=False)
    _assert_exact_alignment(candidate_frame, baseline_frame)
    if not np.array_equal(candidate_arrays["label"], baseline_arrays["label"]):
        raise R1OOFValidationError("candidate/baseline labels differ after numeric validation")
    base_match = np.array_equal(candidate_arrays["base_prediction"], baseline_arrays["prediction"])
    if not base_match:
        mismatch_count = int(
            np.sum(candidate_arrays["base_prediction"] != baseline_arrays["prediction"])
        )
        raise R1OOFValidationError(
            f"candidate base_prediction differs from baseline prediction in {mismatch_count} rows"
        )

    if "anomaly_type" not in baseline_frame:
        raise R1OOFValidationError("baseline OOF is missing anomaly_type required for type QA")
    if "anomaly_type" in candidate_frame:
        candidate_type = candidate_frame["anomaly_type"].astype("string").fillna("")
        baseline_type = baseline_frame["anomaly_type"].astype("string").fillna("")
        if not candidate_type.reset_index(drop=True).equals(baseline_type.reset_index(drop=True)):
            raise R1OOFValidationError("candidate/baseline anomaly_type order differs")

    truth = baseline_arrays["label"]
    candidate_prediction = candidate_arrays["prediction"]
    baseline_prediction = baseline_arrays["prediction"]
    test_shares = group_row_shares(test_frame)
    oof_groups = set(
        tuple(value)
        for value in baseline_frame.loc[:, ["station", "layer"]].itertuples(index=False, name=None)
    )
    covered_weight = float(sum(test_shares.get(group, 0.0) for group in oof_groups))
    missing_test_groups = set(test_shares).difference(oof_groups)

    overall = _paired_metrics(truth, candidate_prediction, baseline_prediction)
    weighted = _paired_metrics(
        truth,
        candidate_prediction,
        baseline_prediction,
        metadata=baseline_frame,
        group_weights=test_shares,
    )
    by_fold = _dimension_metrics(
        baseline_frame,
        truth,
        candidate_prediction,
        baseline_prediction,
        ("fold",),
    )
    by_group = _dimension_metrics(
        baseline_frame,
        truth,
        candidate_prediction,
        baseline_prediction,
        ("station", "layer"),
    )
    _assert_additive_breakdown(overall, by_fold, dimension="fold")
    _assert_additive_breakdown(overall, by_group, dimension="station_layer")

    probability_difference = np.abs(
        candidate_arrays["probability"] - baseline_arrays["probability"]
    )
    report: dict[str, Any] = {
        "contract_version": 1,
        "status": "passed",
        "warning": (
            "OUTER-LABEL EVALUATION ONLY. This report must not be used to tune, select, "
            "or fit the R1 candidate and is not an official hidden-test score."
        ),
        "scope": {
            "grain": "one aligned outer-validation station-layer-time row",
            "timezone": "Asia/Seoul (+09:00)",
            "official_primary_metric": "row-level binary micro F1",
            "rows": len(baseline_frame),
            "folds": int(baseline_frame["fold"].nunique(dropna=False)),
            "groups": int(baseline_frame.loc[:, ["station", "layer"]].drop_duplicates().shape[0]),
        },
        "use_policy": {
            "outer_validation_labels_used": True,
            "evaluation_only": True,
            "candidate_selection_allowed": False,
            "hidden_test_or_official_score_claim_allowed": False,
        },
        "sources": {
            "candidate_oof": _source_payload(candidate_path),
            "baseline_oof": _source_payload(baseline_path),
            "test_csv": _source_payload(test_path),
        },
        "alignment": {
            "key_fold_label_order_exact_match": True,
            "candidate_base_prediction_exact_baseline_prediction": True,
            "candidate_vs_baseline_probability_max_abs_difference": float(
                probability_difference.max(initial=0.0)
            ),
            "candidate_vs_baseline_probability_exact_match": bool(
                np.array_equal(candidate_arrays["probability"], baseline_arrays["probability"])
            ),
        },
        "official_row_metrics": overall,
        "test_share_weighted_metrics": {
            **weighted,
            "test_group_count": len(test_shares),
            "oof_group_count": len(oof_groups),
            "covered_test_row_share": covered_weight,
            "missing_test_groups_in_oof": len(missing_test_groups),
            "weights_renormalized_over_observed_oof_groups": covered_weight < 1.0 - 1.0e-12,
        },
        "by_fold": by_fold,
        "by_station_layer": by_group,
        "by_anomaly_type_membership_non_additive": _type_metrics(
            baseline_frame["anomaly_type"],
            truth,
            candidate_prediction,
            baseline_prediction,
        ),
        "long_positive_events": _long_event_recall(
            baseline_frame,
            truth,
            candidate_prediction,
            baseline_prediction,
        ),
        "normal_station_layer_day_fp": _normal_day_fp_rate(
            baseline_frame,
            truth,
            candidate_prediction,
            baseline_prediction,
        ),
        "paired_block_bootstrap": paired_block_bootstrap(
            truth,
            candidate_prediction,
            baseline_prediction,
            baseline_frame,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "checks": [
            "candidate and baseline keys/fold/label match exactly in row order",
            "candidate base_prediction exactly matches baseline prediction",
            "OOF keys are unique and labels/predictions/probabilities are valid",
            "fold and station-layer count subtotals reconcile to overall counts",
            "only aggregate metrics and SHA-256 provenance are emitted",
        ],
    }
    # Reject accidental NaN/Infinity or non-JSON data before any file is written.
    json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return report


def write_r1_oof_report(report: Mapping[str, Any], output: str | Path) -> dict[str, Any]:
    """Write deterministic JSON and a detached SHA-256 sidecar."""

    destination = Path(output).expanduser().resolve()
    if destination.suffix.lower() != ".json":
        raise R1OOFValidationError("--output must end in .json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    destination.write_text(payload + "\n", encoding="utf-8", newline="\n")
    digest = sha256_file(destination)
    sidecar = destination.with_suffix(destination.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {destination.name}\n", encoding="ascii", newline="\n")
    return {
        "output": destination,
        "output_sha256": digest,
        "sha256_sidecar": sidecar,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-oof", type=Path, required=True)
    parser.add_argument("--baseline-oof", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_r1_oof_report(
            args.candidate_oof,
            args.baseline_oof,
            args.test_csv,
            bootstrap_replicates=BOOTSTRAP_REPLICATES,
            bootstrap_seed=BOOTSTRAP_SEED,
        )
        written = write_r1_oof_report(report, args.output)
    except (OSError, R1OOFValidationError, ValueError, KeyError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print("PASS: R1 candidate OOF is aligned and aggregate metrics were validated")
    print(f"output={written['output']}")
    print(f"sha256={written['output_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
