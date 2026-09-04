"""Past-only, anchor-preserving long-event change-point rescue primitives.

The module is a clean successor to the sealed proposal-rescore experiments.
It deliberately does not import or mutate their scientific constants.  Every
physical feature at row ``t`` uses only observations at or before ``t``.
Proposal features aggregate those causal row features only after a candidate
interval closes, which is valid for the competition's offline QC setting.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

KEY_COLUMNS = ("station", "year", "layer", "time")
TARGET_CELLS = (("G-ORS", 1), ("I-ORS", 1), ("S-ORS", 2))
CADENCE_MINUTES = 10


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    station: str
    layer: int
    row_ids: np.ndarray
    end_time: pd.Timestamp
    features: np.ndarray


@dataclass(frozen=True)
class FittedScorer:
    model: Any | None
    constant_probability: float | None

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if len(features) == 0:
            return np.empty(0, dtype=np.float64)
        if self.model is None:
            if self.constant_probability is None:
                raise RuntimeError("constant scorer is incomplete")
            return np.full(len(features), self.constant_probability, dtype=np.float64)
        return self.model.predict_proba(features)[:, 1].astype(np.float64, copy=False)


def stable_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def binary_metrics(y_true: Sequence[int], prediction: Sequence[int]) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=np.int8)
    pred = np.asarray(prediction, dtype=np.int8)
    if y.shape != pred.shape or not np.isin(y, [0, 1]).all() or not np.isin(pred, [0, 1]).all():
        raise ValueError("binary metric arrays are invalid")
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def anchor_preserving_union(anchor: Sequence[int], additions: Sequence[int]) -> np.ndarray:
    anchor_bits = np.asarray(anchor, dtype=np.int8)
    add_bits = np.asarray(additions, dtype=np.int8)
    if anchor_bits.shape != add_bits.shape:
        raise ValueError("anchor/addition shape mismatch")
    if not np.isin(anchor_bits, [0, 1]).all() or not np.isin(add_bits, [0, 1]).all():
        raise ValueError("anchor/addition must be binary")
    return np.maximum(anchor_bits, add_bits).astype(np.int8, copy=False)


def _rolling_history(values: pd.Series, rows: int, minimum: int) -> tuple[pd.Series, pd.Series]:
    history = values.shift(1)
    median = history.rolling(rows, min_periods=minimum).median()
    scale = history.rolling(rows, min_periods=minimum).std(ddof=0).clip(lower=0.05)
    return median, scale


def build_past_only_row_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return causal physical features for the three preregistered cells.

    Appending later rows cannot alter any already-computed row.  Same-time
    peer layers are contemporaneous covariates, not future observations.
    """

    required = {*KEY_COLUMNS, "temp_raw", "psal_raw", "depth_raw", "anchor_probability", "anchor"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"missing feature inputs: {missing}")
    work = frame.copy()
    work["_row_id"] = np.arange(len(work), dtype=np.int64)
    work["_time"] = pd.to_datetime(work["time"], utc=True, format="mixed", errors="raise")
    work["station"] = work["station"].astype(str)
    work["layer"] = work["layer"].astype(int)
    work["temp_raw"] = pd.to_numeric(work["temp_raw"], errors="coerce")

    peer_group = work.groupby(["station", "_time"], sort=False, observed=True)["temp_raw"]
    peer_sum = peer_group.transform("sum")
    peer_count = peer_group.transform("count")
    work["_peer_temp"] = (peer_sum - work["temp_raw"]) / (peer_count - 1).replace(0, np.nan)
    work["_peer_residual"] = work["temp_raw"] - work["_peer_temp"]

    selected = []
    for station, layer in TARGET_CELLS:
        cell = work.loc[(work["station"] == station) & (work["layer"] == layer)].copy()
        cell.sort_values("_time", inplace=True, kind="mergesort")
        if cell.empty:
            continue
        delta = cell["_time"].diff().dt.total_seconds().div(60.0)
        cell["_segment"] = (delta.ne(CADENCE_MINUTES) | delta.isna()).cumsum().astype(int)
        parts = []
        for _segment_id, part in cell.groupby("_segment", sort=False, observed=True):
            part = part.copy()
            temp = part["temp_raw"].astype(float)
            median72, scale72 = _rolling_history(temp, 72 * 6, 12 * 6)
            median168, scale168 = _rolling_history(temp, 168 * 6, 24 * 6)
            part["temp_z_72h"] = ((temp - median72) / scale72).clip(-30.0, 30.0)
            part["temp_z_168h"] = ((temp - median168) / scale168).clip(-30.0, 30.0)

            peer = part["_peer_residual"].astype(float)
            peer_median, peer_scale = _rolling_history(peer, 72 * 6, 12 * 6)
            part["peer_z_72h"] = ((peer - peer_median) / peer_scale).clip(-30.0, 30.0)
            part["slope_6h"] = ((temp - temp.shift(36)) / scale72).clip(-30.0, 30.0)
            part["slope_24h"] = ((temp - temp.shift(144)) / scale168).clip(-30.0, 30.0)
            physical_columns = ["temp_z_72h", "temp_z_168h", "peer_z_72h", "slope_6h", "slope_24h"]
            physical_matrix = np.nan_to_num(
                np.abs(part[physical_columns].to_numpy(dtype=np.float64)),
                nan=0.0,
                posinf=30.0,
                neginf=30.0,
            )
            physical = np.max(physical_matrix, axis=1)
            part["physical_score"] = np.clip(physical, 0.0, 30.0)
            part["persistent_score_3h"] = (
                part["physical_score"].rolling(18, min_periods=6).mean().fillna(0.0)
            )
            part["temp_missing"] = temp.isna().astype(float)
            part["peer_missing"] = peer.isna().astype(float)
            parts.append(part)
        selected.append(pd.concat(parts, ignore_index=False))
    if not selected:
        return work.iloc[0:0].copy()
    result = pd.concat(selected, ignore_index=False)
    result.sort_values(["station", "layer", "_time"], inplace=True, kind="mergesort")
    return result.reset_index(drop=True)


def _runs_with_small_gaps(mask: np.ndarray, maximum_gap_rows: int) -> list[tuple[int, int]]:
    positive = np.flatnonzero(mask)
    if len(positive) == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(positive[0])
    previous = start
    for value in positive[1:]:
        current = int(value)
        if current - previous - 1 > maximum_gap_rows:
            runs.append((start, previous + 1))
            start = current
        previous = current
    runs.append((start, previous + 1))
    return runs


def generate_proposals(
    row_features: pd.DataFrame,
    *,
    score_thresholds: Sequence[float],
    minimum_support_rows: Sequence[int],
    maximum_gap_rows: int,
    padding_rows: int,
    minimum_interval_rows: int,
    maximum_interval_rows: int,
) -> tuple[list[Proposal], tuple[str, ...]]:
    """Build a deterministic, label-free multi-threshold proposal bank."""

    feature_names = (
        "duration_rows",
        "support_fraction",
        "physical_mean",
        "physical_q90",
        "physical_max",
        "persistent_mean",
        "temp_z72_abs_mean",
        "temp_z168_abs_mean",
        "peer_z72_abs_mean",
        "slope6_abs_mean",
        "slope24_abs_mean",
        "anchor_probability_mean",
        "anchor_probability_max",
        "anchor_positive_fraction",
        "temp_missing_fraction",
        "peer_missing_fraction",
        "station_g",
        "station_i",
        "station_s",
    )
    proposals: list[Proposal] = []
    seen: set[tuple[str, int, int, int]] = set()
    for (station, layer, _segment), part in row_features.groupby(
        ["station", "layer", "_segment"], sort=False, observed=True
    ):
        part = part.sort_values("_time", kind="mergesort").reset_index(drop=True)
        physical = part["physical_score"].to_numpy(dtype=np.float64)
        for threshold in score_thresholds:
            evidence = physical >= float(threshold)
            for support in minimum_support_rows:
                for start, stop in _runs_with_small_gaps(evidence, maximum_gap_rows):
                    evidence_count = int(evidence[start:stop].sum())
                    if evidence_count < int(support):
                        continue
                    left = max(0, start - int(padding_rows))
                    right = min(len(part), stop + int(padding_rows))
                    while right - left > maximum_interval_rows:
                        sub_right = min(right, left + maximum_interval_rows)
                        if sub_right - left >= minimum_interval_rows:
                            key = (str(station), int(layer), int(part.loc[left, "_row_id"]), int(part.loc[sub_right - 1, "_row_id"]))
                            if key not in seen:
                                proposals.append(_proposal_from_slice(part, left, sub_right, threshold, feature_names))
                                seen.add(key)
                        left = sub_right
                    if right - left < minimum_interval_rows:
                        continue
                    key = (str(station), int(layer), int(part.loc[left, "_row_id"]), int(part.loc[right - 1, "_row_id"]))
                    if key in seen:
                        continue
                    proposals.append(_proposal_from_slice(part, left, right, threshold, feature_names))
                    seen.add(key)
    proposals.sort(key=lambda item: (item.end_time, item.station, item.layer, item.proposal_id))
    return proposals, feature_names


def _proposal_from_slice(
    part: pd.DataFrame,
    start: int,
    stop: int,
    threshold: float,
    feature_names: Sequence[str],
) -> Proposal:
    window = part.iloc[start:stop]
    station = str(window["station"].iloc[0])
    layer = int(window["layer"].iloc[0])
    row_ids = window["_row_id"].to_numpy(dtype=np.int64)
    physical = window["physical_score"].to_numpy(dtype=np.float64)
    values = {
        "duration_rows": float(len(window)),
        "support_fraction": float(np.mean(physical >= threshold)),
        "physical_mean": float(np.mean(physical)),
        "physical_q90": float(np.quantile(physical, 0.9)),
        "physical_max": float(np.max(physical)),
        "persistent_mean": float(window["persistent_score_3h"].mean()),
        "temp_z72_abs_mean": float(window["temp_z_72h"].abs().fillna(0.0).mean()),
        "temp_z168_abs_mean": float(window["temp_z_168h"].abs().fillna(0.0).mean()),
        "peer_z72_abs_mean": float(window["peer_z_72h"].abs().fillna(0.0).mean()),
        "slope6_abs_mean": float(window["slope_6h"].abs().fillna(0.0).mean()),
        "slope24_abs_mean": float(window["slope_24h"].abs().fillna(0.0).mean()),
        "anchor_probability_mean": float(window["anchor_probability"].mean()),
        "anchor_probability_max": float(window["anchor_probability"].max()),
        "anchor_positive_fraction": float(window["anchor"].mean()),
        "temp_missing_fraction": float(window["temp_missing"].mean()),
        "peer_missing_fraction": float(window["peer_missing"].mean()),
        "station_g": float(station == "G-ORS"),
        "station_i": float(station == "I-ORS"),
        "station_s": float(station == "S-ORS"),
    }
    payload = {
        "station": station,
        "layer": layer,
        "first_row": int(row_ids[0]),
        "last_row": int(row_ids[-1]),
        "threshold": float(threshold),
    }
    return Proposal(
        proposal_id=stable_sha256(payload)[:20],
        station=station,
        layer=layer,
        row_ids=row_ids,
        end_time=pd.Timestamp(window["_time"].iloc[-1]),
        features=np.asarray([values[name] for name in feature_names], dtype=np.float64),
    )


def proposal_targets(
    proposals: Sequence[Proposal],
    y_true: Sequence[int],
    anchor: Sequence[int],
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    y = np.asarray(y_true, dtype=np.int8)
    base = np.asarray(anchor, dtype=np.int8)
    base_metrics = binary_metrics(y, base)
    targets = np.zeros(len(proposals), dtype=np.int8)
    diagnostics: list[dict[str, float | int]] = []
    for index, proposal in enumerate(proposals):
        rows = proposal.row_ids[base[proposal.row_ids] == 0]
        tp_added = int(np.sum(y[rows] == 1))
        fp_added = int(np.sum(y[rows] == 0))
        candidate_tp = int(base_metrics["tp"]) + tp_added
        candidate_fp = int(base_metrics["fp"]) + fp_added
        candidate_fn = int(base_metrics["fn"]) - tp_added
        denominator = 2 * candidate_tp + candidate_fp + candidate_fn
        candidate_f1 = float(2 * candidate_tp / denominator) if denominator else 0.0
        delta = candidate_f1 - float(base_metrics["f1"])
        targets[index] = int(tp_added > 0 and delta > 0.0)
        diagnostics.append(
            {
                "tp_added": tp_added,
                "fp_added": fp_added,
                "individual_delta_f1": delta,
                "beneficial": int(targets[index]),
            }
        )
    return targets, diagnostics


def fit_scorer(features: np.ndarray, targets: Sequence[int], *, seed: int) -> FittedScorer:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int8)
    if len(x) != len(y) or x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("proposal training matrix is invalid")
    unique = np.unique(y)
    if len(unique) < 2:
        return FittedScorer(None, float(unique[0]) if len(unique) else 0.0)
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=0.25,
                    class_weight="balanced",
                    max_iter=4000,
                    random_state=int(seed),
                    solver="liblinear",
                ),
            ),
        ]
    )
    model.fit(x, y)
    return FittedScorer(model, None)


def expanding_cross_fit_scores(
    proposals: Sequence[Proposal],
    targets: Sequence[int],
    *,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Score each month only with proposal labels from strictly earlier months."""

    y = np.asarray(targets, dtype=np.int8)
    features = np.vstack([proposal.features for proposal in proposals]) if proposals else np.empty((0, 0))
    scores = np.full(len(proposals), np.nan, dtype=np.float64)
    months = pd.PeriodIndex([proposal.end_time.tz_convert("Asia/Seoul").tz_localize(None) for proposal in proposals], freq="M")
    receipts: list[dict[str, Any]] = []
    for month in sorted(months.unique()):
        validation = np.flatnonzero(months == month)
        training = np.flatnonzero(months < month)
        if len(training) == 0:
            receipts.append({"month": str(month), "train_rows": 0, "validation_rows": int(len(validation)), "arm": "ZERO_ADD_NO_HISTORY"})
            continue
        scorer = fit_scorer(features[training], y[training], seed=seed)
        scores[validation] = scorer.predict_proba(features[validation])
        receipts.append(
            {
                "month": str(month),
                "train_rows": int(len(training)),
                "validation_rows": int(len(validation)),
                "training_latest_end": max(proposals[index].end_time for index in training).isoformat(),
                "validation_earliest_end": min(proposals[index].end_time for index in validation).isoformat(),
                "strictly_past": bool(max(proposals[index].end_time for index in training) < min(proposals[index].end_time for index in validation)),
            }
        )
    return scores, receipts


def additions_from_scores(
    rows: int,
    proposals: Sequence[Proposal],
    scores: Sequence[float],
    threshold: float,
    anchor: Sequence[int],
) -> np.ndarray:
    base = np.asarray(anchor, dtype=np.int8)
    if len(base) != rows or len(proposals) != len(scores):
        raise ValueError("proposal decode shape mismatch")
    additions = np.zeros(rows, dtype=np.int8)
    for proposal, score in zip(proposals, scores, strict=True):
        if np.isfinite(score) and float(score) >= float(threshold):
            additions[proposal.row_ids] = 1
    additions[base == 1] = 0
    return additions


def cell_fp_diagnostics(
    keys: pd.DataFrame,
    y_true: Sequence[int],
    anchor: Sequence[int],
    candidate: Sequence[int],
) -> dict[str, dict[str, float | int]]:
    y = np.asarray(y_true, dtype=np.int8)
    base = np.asarray(anchor, dtype=np.int8)
    pred = np.asarray(candidate, dtype=np.int8)
    time = pd.to_datetime(keys["time"], utc=True, format="mixed")
    result: dict[str, dict[str, float | int]] = {}
    for station, layer in TARGET_CELLS:
        mask = (keys["station"].astype(str).to_numpy() == station) & (keys["layer"].to_numpy(dtype=int) == layer)
        days = max(1, int(time.loc[mask].dt.tz_convert("Asia/Seoul").dt.date.nunique()))
        anchor_fp = int(np.sum(mask & (y == 0) & (base == 1)))
        candidate_fp = int(np.sum(mask & (y == 0) & (pred == 1)))
        added_fp = int(np.sum(mask & (y == 0) & (base == 0) & (pred == 1)))
        result[f"{station}/L{layer}"] = {
            "rows": int(mask.sum()),
            "days": days,
            "anchor_fp": anchor_fp,
            "candidate_fp": candidate_fp,
            "added_fp": added_fp,
            "anchor_fp_per_day": float(anchor_fp / days),
            "candidate_fp_per_day": float(candidate_fp / days),
            "added_fp_per_day": float(added_fp / days),
        }
    return result


def select_threshold_arm(
    keys: pd.DataFrame,
    y_true: Sequence[int],
    anchor: Sequence[int],
    proposals: Sequence[Proposal],
    scores: Sequence[float],
    *,
    threshold_candidates: Sequence[float],
    maximum_added_fp_per_day: float,
    minimum_added_precision: float,
) -> tuple[dict[str, Any], np.ndarray]:
    y = np.asarray(y_true, dtype=np.int8)
    base = np.asarray(anchor, dtype=np.int8)
    base_metrics = binary_metrics(y, base)
    arms: list[tuple[dict[str, Any], np.ndarray]] = []
    no_op = np.zeros(len(base), dtype=np.int8)
    arms.append(
        (
            {
                "arm": "ZERO_ADD_NO_OP",
                "threshold": None,
                "delta_f1": 0.0,
                "added_rows": 0,
                "added_precision": 1.0,
                "caps_passed": True,
                "cell_fp": cell_fp_diagnostics(keys, y, base, base),
            },
            no_op,
        )
    )
    for threshold in threshold_candidates:
        additions = additions_from_scores(len(base), proposals, scores, threshold, base)
        candidate = anchor_preserving_union(base, additions)
        metrics = binary_metrics(y, candidate)
        changed = (base == 0) & (candidate == 1)
        added_tp = int(np.sum(changed & (y == 1)))
        added_fp = int(np.sum(changed & (y == 0)))
        added_precision = float(added_tp / (added_tp + added_fp)) if added_tp + added_fp else 1.0
        cells = cell_fp_diagnostics(keys, y, base, candidate)
        caps_passed = all(float(value["added_fp_per_day"]) <= maximum_added_fp_per_day for value in cells.values())
        caps_passed = caps_passed and added_precision >= minimum_added_precision
        arms.append(
            (
                {
                    "arm": f"SCORE_GTE_{float(threshold):.3f}",
                    "threshold": float(threshold),
                    "delta_f1": float(metrics["f1"] - base_metrics["f1"]),
                    "added_rows": int(changed.sum()),
                    "added_tp": added_tp,
                    "added_fp": added_fp,
                    "added_precision": added_precision,
                    "caps_passed": bool(caps_passed),
                    "cell_fp": cells,
                },
                additions,
            )
        )
    eligible = [item for item in arms if item[0]["caps_passed"]]
    selected = max(
        eligible,
        key=lambda item: (
            float(item[0]["delta_f1"]),
            float(item[0]["added_precision"]),
            -int(item[0]["added_rows"]),
            str(item[0]["arm"]),
        ),
    )
    selected_summary = {**selected[0], "all_arms": [item[0] for item in arms]}
    return selected_summary, selected[1]


def ordered_changed_key_sha(keys: pd.DataFrame, changed: np.ndarray) -> str:
    selected = keys.loc[np.asarray(changed, dtype=bool), list(KEY_COLUMNS)].astype(str)
    digest = hashlib.sha256()
    for row in selected.itertuples(index=False, name=None):
        digest.update("\x1f".join(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def proposal_bank_sha(proposals: Iterable[Proposal]) -> str:
    digest = hashlib.sha256()
    for proposal in proposals:
        digest.update(proposal.proposal_id.encode("ascii"))
        digest.update(proposal.row_ids.astype("<i8", copy=False).tobytes())
        digest.update(proposal.features.astype("<f8", copy=False).tobytes())
    return digest.hexdigest()
