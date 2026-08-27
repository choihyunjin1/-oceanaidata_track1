"""Leakage-safe primitives for the P3 causal forcing analog outer v4."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as pyarrow_dataset

from p3_wave.episode_distinct_analog import LEADS

PAIR_KEYS = ("fold", "anchor_id", "station", "lead_h")
COMPONENT_BLIND_COLUMNS = (
    *PAIR_KEYS,
    "current_hs",
    "history_eligible",
    "conditioning_used",
    "fallback_reason",
    "query_mad_scale",
    "neighbor_anchor_ids_sha256",
    "neighbor_episode_ids_sha256",
    "neighbor_distance_mean",
    "neighbor_distance_max",
    "analog_prediction",
)
FINAL_BLIND_COLUMNS = (
    *COMPONENT_BLIND_COLUMNS,
    "incumbent_final",
    "candidate_final",
)
EVALUATED_ADDITIONAL_COLUMNS = (
    "target_hs",
    "incumbent_squared_error",
    "candidate_squared_error",
)


class OuterResearchError(ValueError):
    """Raised when the fixed v4 outer-research contract is violated."""


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_integer_array(values: np.ndarray | Sequence[int]) -> str:
    array = np.asarray(values, dtype="<i8")
    return sha256(array.tobytes()).hexdigest()


def canonical_membership_hashes(keys: pd.DataFrame) -> dict[str, str]:
    _require_columns(keys, PAIR_KEYS, role="membership keys")
    ordered = keys.loc[:, PAIR_KEYS].sort_values(list(PAIR_KEYS)).reset_index(drop=True)
    rows = "".join(
        f"{row.fold}|{int(row.anchor_id)}|{row.station}|{int(row.lead_h)}\n"
        for row in ordered.itertuples(index=False)
    ).encode("utf-8")
    cases = (
        ordered[["fold", "anchor_id", "station"]]
        .drop_duplicates()
        .sort_values(["fold", "anchor_id", "station"])
    )
    case_rows = "".join(
        f"{row.fold}|{int(row.anchor_id)}|{row.station}\n"
        for row in cases.itertuples(index=False)
    ).encode("utf-8")
    return {
        "canonical_row_key_sha256": sha256(rows).hexdigest(),
        "canonical_case_key_sha256": sha256(case_rows).hexdigest(),
    }


def validate_membership_keys(
    keys: pd.DataFrame,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if list(keys.columns) != list(PAIR_KEYS):
        raise OuterResearchError("membership reader opened a non-key OOF column")
    if keys.duplicated(list(PAIR_KEYS)).any():
        raise OuterResearchError("frozen outer keys are duplicated")
    if len(keys) != int(contract["expected_rows"]):
        raise OuterResearchError("frozen outer row count changed")
    cases = keys[["fold", "anchor_id", "station"]].drop_duplicates()
    if len(cases) != int(contract["expected_cases"]):
        raise OuterResearchError("frozen outer case count changed")
    if cases.duplicated(["anchor_id"]).any():
        raise OuterResearchError("an outer anchor belongs to multiple folds")
    lead_sets = keys.groupby(["fold", "anchor_id"], observed=True)["lead_h"].agg(
        lambda values: tuple(sorted(int(value) for value in values))
    )
    if not lead_sets.map(lambda values: values == tuple(LEADS)).all():
        raise OuterResearchError("an outer case does not contain exactly six leads")
    station_counts = keys.groupby(["fold", "anchor_id"], observed=True)["station"].nunique()
    if not station_counts.eq(1).all():
        raise OuterResearchError("an outer case maps to multiple stations")
    fold_cases = {
        str(key): int(value)
        for key, value in cases.groupby("fold", observed=True).size().items()
    }
    station_cases = {
        str(key): int(value)
        for key, value in cases.groupby("station", observed=True).size().items()
    }
    if fold_cases != {str(key): int(value) for key, value in contract["expected_fold_cases"].items()}:
        raise OuterResearchError("frozen outer fold membership changed")
    if station_cases != {
        str(key): int(value) for key, value in contract["expected_station_cases"].items()
    }:
        raise OuterResearchError("frozen outer station membership changed")
    hashes = canonical_membership_hashes(keys)
    for name, actual in hashes.items():
        if actual != str(contract[name]):
            raise OuterResearchError(f"frozen outer {name} changed")
    return {
        "rows": int(len(keys)),
        "cases": int(len(cases)),
        "fold_cases": fold_cases,
        "station_cases": station_cases,
        "leads": list(LEADS),
        **hashes,
        "columns_read": list(PAIR_KEYS),
        "incumbent_prediction_values_read": 0,
        "designated_target_values_read": 0,
    }


def read_membership_keys_only(
    path: str | Path,
    contract: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    """Read only the four frozen OOF key columns; never infer membership anew."""

    keys = pd.read_parquet(path, columns=list(PAIR_KEYS))
    audit = validate_membership_keys(keys, contract)
    ordered = keys.sort_values(list(PAIR_KEYS)).reset_index(drop=True)
    membership = {
        str(name): np.sort(
            ordered.loc[ordered["fold"].astype(str).eq(str(name)), "anchor_id"]
            .unique()
            .astype(np.int64)
        )
        for name in contract["expected_fold_cases"]
    }
    return ordered, membership, audit


@dataclass(frozen=True)
class FoldLibraryScope:
    name: str
    library_ids: np.ndarray
    validation_ids: np.ndarray


@dataclass
class TrainingTargetVault:
    """Open only fold-library targets, excluding current and future outer cases."""

    path: Path
    scopes: Sequence[FoldLibraryScope]
    access_log: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.scopes) != 3 or len({scope.name for scope in self.scopes}) != 3:
            raise OuterResearchError("training target vault requires three ordered folds")
        validation = np.concatenate(
            [np.asarray(scope.validation_ids, dtype=np.int64) for scope in self.scopes]
        )
        if len(np.unique(validation)) != len(validation):
            raise OuterResearchError("outer validation anchor repeats across folds")

    def read_library(self, fold: str, anchor_ids: Sequence[int]) -> np.ndarray:
        names = [scope.name for scope in self.scopes]
        if fold not in names:
            raise OuterResearchError(f"unknown outer fold: {fold}")
        index = names.index(fold)
        scope = self.scopes[index]
        ids = np.asarray(anchor_ids, dtype=np.int64)
        expected = np.asarray(scope.library_ids, dtype=np.int64)
        if not np.array_equal(ids, expected):
            raise PermissionError(f"{fold} target request differs from the sealed library")
        current_future = np.concatenate(
            [
                np.asarray(item.validation_ids, dtype=np.int64)
                for item in self.scopes[index:]
            ]
        )
        overlap = np.intersect1d(ids, current_future)
        if overlap.size:
            raise PermissionError(f"{fold} requested current/future outer targets")
        prior = (
            np.concatenate(
                [np.asarray(item.validation_ids, dtype=np.int64) for item in self.scopes[:index]]
            )
            if index
            else np.empty(0, dtype=np.int64)
        )
        columns = ["anchor_id", *[f"target_{lead}" for lead in LEADS]]
        dataset = pyarrow_dataset.dataset(self.path, format="parquet")
        table = dataset.to_table(
            columns=columns,
            filter=pyarrow_dataset.field("anchor_id").isin(ids.tolist()),
        )
        frame = table.to_pandas().set_index("anchor_id")
        if not np.isin(ids, frame.index.to_numpy(dtype=np.int64)).all():
            raise OuterResearchError("training target vault read is incomplete")
        values = frame.loc[ids, [f"target_{lead}" for lead in LEADS]].to_numpy(
            dtype=np.float64
        )
        if values.shape != (len(ids), len(LEADS)) or not np.isfinite(values).all():
            raise OuterResearchError("training target matrix is invalid")
        self.access_log.append(
            {
                "fold": fold,
                "purpose": "causal_analog_library_targets",
                "anchor_count": int(len(ids)),
                "anchor_ids_sha256": hash_integer_array(ids),
                "current_or_future_outer_overlap_count": 0,
                "allowed_prior_outer_overlap_count": int(np.intersect1d(ids, prior).size),
            }
        )
        return values


@dataclass
class FrozenOOFStageVault:
    """Enforce component seal -> incumbent read -> final seal -> target read."""

    path: Path
    expected_keys: pd.DataFrame
    incumbent_column: str = "prediction"
    target_column: str = "target_hs"
    component_seal_sha256: str | None = None
    final_seal_sha256: str | None = None
    incumbent_prediction_read_count: int = 0
    designated_target_read_count: int = 0
    access_log: list[dict[str, Any]] = field(default_factory=list)

    def register_component_seal(
        self,
        seal_path: str | Path,
        component_path: str | Path,
    ) -> None:
        if self.component_seal_sha256 is not None:
            raise PermissionError("component blind seal was already registered")
        seal = _read_seal(seal_path)
        if seal.get("stage") != "analog_component_before_incumbent_prediction":
            raise PermissionError("component seal stage is invalid")
        if seal.get("incumbent_prediction_read_count") != 0:
            raise PermissionError("component seal was created after incumbent exposure")
        if seal.get("designated_target_read_count") != 0:
            raise PermissionError("component seal was created after target exposure")
        if seal.get("component_blind_sha256") != sha256_file(component_path):
            raise PermissionError("component seal is not bound to the component parquet")
        self.component_seal_sha256 = sha256_file(seal_path)

    def read_incumbent_once(self) -> pd.DataFrame:
        if self.component_seal_sha256 is None:
            raise PermissionError("incumbent prediction requires the component blind seal")
        if self.incumbent_prediction_read_count:
            raise PermissionError("incumbent prediction may be read only once")
        columns = [*PAIR_KEYS, self.incumbent_column]
        frame = pd.read_parquet(self.path, columns=columns)
        if list(frame.columns) != columns:
            raise OuterResearchError("incumbent read opened an unexpected OOF column")
        ordered = frame.sort_values(list(PAIR_KEYS)).reset_index(drop=True)
        self._validate_keys(ordered)
        prediction = ordered[self.incumbent_column].to_numpy(dtype=np.float64)
        if not np.isfinite(prediction).all() or (prediction < 0.0).any() or (prediction > 30.0).any():
            raise OuterResearchError("exact incumbent prediction is invalid")
        self.incumbent_prediction_read_count += 1
        self.access_log.append(
            {
                "purpose": "exact_incumbent_after_component_blind_seal",
                "columns_read": columns,
                "rows": int(len(ordered)),
                "component_seal_sha256": self.component_seal_sha256,
                "target_values_read": 0,
            }
        )
        return ordered.rename(columns={self.incumbent_column: "incumbent_final"})

    def register_final_seal(self, seal_path: str | Path, blind_path: str | Path) -> None:
        if self.component_seal_sha256 is None or self.incumbent_prediction_read_count != 1:
            raise PermissionError("final seal requires exactly one post-component incumbent read")
        if self.final_seal_sha256 is not None:
            raise PermissionError("final blind seal was already registered")
        seal = _read_seal(seal_path)
        if seal.get("stage") != "all_final_blind_predictions_before_designated_target":
            raise PermissionError("final seal stage is invalid")
        if seal.get("component_seal_sha256") != self.component_seal_sha256:
            raise PermissionError("final seal is not bound to the component seal")
        if seal.get("incumbent_prediction_read_count") != 1:
            raise PermissionError("final seal incumbent read count changed")
        if seal.get("designated_target_read_count") != 0:
            raise PermissionError("final seal was created after target exposure")
        if seal.get("final_blind_sha256") != sha256_file(blind_path):
            raise PermissionError("final seal is not bound to the blind parquet")
        self.final_seal_sha256 = sha256_file(seal_path)

    def read_designated_target_once(
        self,
        *,
        scoring_lock_paths: Sequence[str | Path],
        ledger_receipt: Mapping[str, Any],
    ) -> pd.DataFrame:
        if self.final_seal_sha256 is None:
            raise PermissionError("designated target requires the final blind seal")
        if self.designated_target_read_count:
            raise PermissionError("designated target may be read only once")
        if len(scoring_lock_paths) != 2 or not all(Path(path).is_file() for path in scoring_lock_paths):
            raise PermissionError("both O_EXCL scoring locks are required")
        if ledger_receipt.get("final_seal_sha256") != self.final_seal_sha256:
            raise PermissionError("ledger receipt is not bound to the final seal")
        columns = [*PAIR_KEYS, self.target_column]
        frame = pd.read_parquet(self.path, columns=columns)
        if list(frame.columns) != columns:
            raise OuterResearchError("designated target read opened an unexpected OOF column")
        ordered = frame.sort_values(list(PAIR_KEYS)).reset_index(drop=True)
        self._validate_keys(ordered)
        target = ordered[self.target_column].to_numpy(dtype=np.float64)
        if not np.isfinite(target).all():
            raise OuterResearchError("designated target contains a non-finite value")
        self.designated_target_read_count += 1
        self.access_log.append(
            {
                "purpose": "designated_outer_target_after_final_seal_and_O_EXCL_ledger",
                "columns_read": columns,
                "rows": int(len(ordered)),
                "final_seal_sha256": self.final_seal_sha256,
            }
        )
        return ordered.rename(columns={self.target_column: "target_hs"})

    def _validate_keys(self, frame: pd.DataFrame) -> None:
        actual = frame.loc[:, PAIR_KEYS]
        expected = self.expected_keys.loc[:, PAIR_KEYS]
        if not actual.equals(expected):
            raise OuterResearchError("staged OOF read keys differ from frozen membership")


def extract_native_20m_histories(wave: pd.DataFrame, anchors: pd.DataFrame) -> np.ndarray:
    """Extract exactly 145 past/current hs points without touching future values."""

    required_wave = {"station", "time", "hs"}
    required_anchor = {"anchor_id", "station", "anchor_time", "current_hs"}
    if not required_wave.issubset(wave) or not required_anchor.issubset(anchors):
        raise OuterResearchError("history extraction input schema changed")
    ordered_anchors = anchors.sort_values("anchor_id").reset_index(drop=True)
    if not np.array_equal(
        ordered_anchors["anchor_id"].to_numpy(dtype=np.int64),
        np.arange(len(ordered_anchors), dtype=np.int64),
    ):
        raise OuterResearchError("history extraction requires contiguous anchor IDs")
    history = np.full((len(ordered_anchors), 145), np.nan, dtype=np.float64)
    for station, current in ordered_anchors.groupby("station", sort=True, observed=True):
        source = wave.loc[wave["station"].astype(str).eq(str(station))].copy()
        source["time"] = pd.to_datetime(source["time"], utc=True, errors="raise")
        source = source.sort_values("time").reset_index(drop=True)
        cadence = source["time"].diff().dropna()
        if not cadence.eq(pd.Timedelta(minutes=20)).all():
            raise OuterResearchError(f"native 20m cadence changed at {station}")
        anchor_time = pd.to_datetime(current["anchor_time"], utc=True, errors="raise")
        positions = pd.Index(source["time"]).get_indexer(anchor_time)
        if (positions < 144).any():
            raise OuterResearchError(f"an anchor at {station} lacks 48h history")
        values = pd.to_numeric(source["hs"], errors="coerce").to_numpy(dtype=np.float64)
        current_indices = current.index.to_numpy(dtype=np.int64)
        for output_row, position in zip(current_indices, positions, strict=True):
            history[output_row] = values[position - 144 : position + 1]
    if not np.allclose(
        history[:, -1],
        ordered_anchors["current_hs"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
        equal_nan=False,
    ):
        raise OuterResearchError("native history current value differs from anchor metadata")
    return history


def validate_component_blind(
    frame: pd.DataFrame,
    expected_keys: pd.DataFrame,
) -> dict[str, Any]:
    if list(frame.columns) != list(COMPONENT_BLIND_COLUMNS):
        raise OuterResearchError("component blind schema changed")
    forbidden = [
        column
        for column in frame
        if "target" in column.lower() or "truth" in column.lower() or "incumbent" in column.lower()
    ]
    if forbidden:
        raise OuterResearchError(f"component blind exposes a sealed OOF value: {forbidden}")
    _validate_prediction_keys(frame, expected_keys)
    eligible = frame["history_eligible"].astype(bool)
    applicable = frame["analog_prediction"].notna()
    if not applicable.equals(eligible):
        raise OuterResearchError("analog applicability differs from history eligibility")
    if not np.isfinite(frame.loc[eligible, "analog_prediction"]).all():
        raise OuterResearchError("eligible analog prediction is non-finite")
    if not frame.loc[~eligible, "analog_prediction"].isna().all():
        raise OuterResearchError("history-ineligible analog must be missing")
    finite = frame.loc[eligible, ["current_hs", "query_mad_scale"]].to_numpy(dtype=np.float64)
    if not np.isfinite(finite).all():
        raise OuterResearchError("eligible component lacks finite current or MAD scale")
    if not frame.loc[~eligible, "query_mad_scale"].isna().all():
        raise OuterResearchError("history-ineligible component unexpectedly has a MAD scale")
    for _, group in frame.groupby(["fold", "anchor_id"], sort=False, observed=True):
        for column in COMPONENT_BLIND_COLUMNS[5:-1]:
            if group[column].nunique(dropna=False) != 1:
                raise OuterResearchError(f"case-level component field varies by lead: {column}")
    cases = frame[["fold", "anchor_id"]].drop_duplicates()
    return {
        "rows": int(len(frame)),
        "cases": int(len(cases)),
        "eligible_cases": int(frame.loc[eligible, ["fold", "anchor_id"]].drop_duplicates().shape[0]),
        "conditioning_used_cases": int(
            frame.loc[frame["conditioning_used"].astype(bool), ["fold", "anchor_id"]]
            .drop_duplicates()
            .shape[0]
        ),
    }


def compose_final_blind(
    component: pd.DataFrame,
    incumbent: pd.DataFrame,
    *,
    alpha: float = 0.2,
) -> pd.DataFrame:
    if float(alpha) != 0.2:
        raise OuterResearchError("v4 alpha is frozen at 0.2")
    expected_incumbent = [*PAIR_KEYS, "incumbent_final"]
    if list(incumbent.columns) != expected_incumbent:
        raise OuterResearchError("staged incumbent schema changed")
    merged = component.merge(
        incumbent,
        on=list(PAIR_KEYS),
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if len(merged) != len(component) or len(merged) != len(incumbent):
        raise OuterResearchError("component and incumbent keys differ")
    incumbent_values = merged["incumbent_final"].to_numpy(dtype=np.float64)
    analog = merged["analog_prediction"].to_numpy(dtype=np.float64)
    leads = merged["lead_h"].to_numpy(dtype=np.int64)
    active = np.isin(leads, [12, 18, 24]) & np.isfinite(analog)
    candidate = incumbent_values.copy()
    candidate[active] = 0.8 * incumbent_values[active] + 0.2 * analog[active]
    merged["candidate_final"] = np.clip(candidate, 0.0, 30.0)
    return merged.loc[:, FINAL_BLIND_COLUMNS]


def validate_final_blind(
    frame: pd.DataFrame,
    expected_keys: pd.DataFrame,
) -> dict[str, Any]:
    if list(frame.columns) != list(FINAL_BLIND_COLUMNS):
        raise OuterResearchError("final blind schema changed")
    forbidden = [column for column in frame if "target" in column.lower() or "truth" in column.lower()]
    if forbidden:
        raise OuterResearchError(f"final blind exposes designated targets: {forbidden}")
    _validate_prediction_keys(frame, expected_keys)
    incumbent = frame["incumbent_final"].to_numpy(dtype=np.float64)
    candidate = frame["candidate_final"].to_numpy(dtype=np.float64)
    analog = frame["analog_prediction"].to_numpy(dtype=np.float64)
    leads = frame["lead_h"].to_numpy(dtype=np.int64)
    if not np.isfinite(np.column_stack([incumbent, candidate])).all():
        raise OuterResearchError("final blind contains a non-finite required prediction")
    if (incumbent < 0.0).any() or (incumbent > 30.0).any() or (candidate < 0.0).any() or (
        candidate > 30.0
    ).any():
        raise OuterResearchError("final blind prediction lies outside [0,30]m")
    no_op = np.isin(leads, [3, 6, 9]) | ~np.isfinite(analog)
    if not np.array_equal(candidate[no_op], incumbent[no_op]):
        raise OuterResearchError("protected final blind row is not an exact no-op")
    active = np.isin(leads, [12, 18, 24]) & np.isfinite(analog)
    expected = np.clip(0.8 * incumbent[active] + 0.2 * analog[active], 0.0, 30.0)
    if not np.array_equal(candidate[active], expected):
        raise OuterResearchError("active final blind row differs from fixed alpha 0.2")
    return {
        "rows": int(len(frame)),
        "cases": int(frame[["fold", "anchor_id"]].drop_duplicates().shape[0]),
        "active_rows": int(active.sum()),
        "exact_no_op_rows": int(no_op.sum()),
    }


def attach_designated_targets(blind: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    expected_target = [*PAIR_KEYS, "target_hs"]
    if list(targets.columns) != expected_target:
        raise OuterResearchError("designated target schema changed")
    evaluated = blind.merge(
        targets,
        on=list(PAIR_KEYS),
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if len(evaluated) != len(blind) or len(evaluated) != len(targets):
        raise OuterResearchError("designated target coverage is incomplete")
    evaluated["incumbent_squared_error"] = np.square(
        evaluated["incumbent_final"] - evaluated["target_hs"]
    )
    evaluated["candidate_squared_error"] = np.square(
        evaluated["candidate_final"] - evaluated["target_hs"]
    )
    expected = (*FINAL_BLIND_COLUMNS, *EVALUATED_ADDITIONAL_COLUMNS)
    if tuple(evaluated.columns) != expected:
        raise OuterResearchError("evaluated outer schema changed")
    return evaluated


def paired_case_bootstrap(
    evaluated: pd.DataFrame,
    *,
    replicates: int = 5000,
    seed: int = 20260822,
) -> dict[str, float | int]:
    grouped = list(evaluated.groupby(["fold", "anchor_id"], sort=True, observed=True))
    if not grouped or int(replicates) != 5000 or int(seed) != 20260822:
        raise OuterResearchError("paired bootstrap contract changed")
    incumbent_sse = np.asarray(
        [np.square(group["incumbent_final"] - group["target_hs"]).sum() for _, group in grouped],
        dtype=np.float64,
    )
    candidate_sse = np.asarray(
        [np.square(group["candidate_final"] - group["target_hs"]).sum() for _, group in grouped],
        dtype=np.float64,
    )
    counts = np.asarray([len(group) for _, group in grouped], dtype=np.float64)
    generator = np.random.default_rng(int(seed))
    deltas = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        draw = generator.integers(0, len(grouped), size=len(grouped))
        denominator = counts[draw].sum()
        deltas[index] = np.sqrt(candidate_sse[draw].sum() / denominator) - np.sqrt(
            incumbent_sse[draw].sum() / denominator
        )
    return {
        "replicates": int(replicates),
        "case_count": int(len(grouped)),
        "delta_rmse_mean_m": float(deltas.mean()),
        "ci90_lower_m": float(np.quantile(deltas, 0.05)),
        "ci90_upper_m": float(np.quantile(deltas, 0.95)),
    }


def evaluate_outer_gate(evaluated: pd.DataFrame) -> dict[str, Any]:
    required = {*FINAL_BLIND_COLUMNS, *EVALUATED_ADDITIONAL_COLUMNS}
    if set(evaluated.columns) != required:
        raise OuterResearchError("outer gate evaluated schema changed")
    incumbent_error = np.square(
        evaluated["incumbent_final"].to_numpy(dtype=np.float64)
        - evaluated["target_hs"].to_numpy(dtype=np.float64)
    )
    candidate_error = np.square(
        evaluated["candidate_final"].to_numpy(dtype=np.float64)
        - evaluated["target_hs"].to_numpy(dtype=np.float64)
    )
    if not np.array_equal(
        incumbent_error, evaluated["incumbent_squared_error"].to_numpy(dtype=np.float64)
    ) or not np.array_equal(
        candidate_error, evaluated["candidate_squared_error"].to_numpy(dtype=np.float64)
    ):
        raise OuterResearchError("stored outer squared errors are not independently reproducible")
    incumbent_rmse = float(np.sqrt(incumbent_error.mean()))
    candidate_rmse = float(np.sqrt(candidate_error.mean()))
    delta = candidate_rmse - incumbent_rmse
    by_fold = _group_deltas(evaluated, "fold")
    by_station = _group_deltas(evaluated, "station")
    by_lead = _group_deltas(evaluated, "lead_h")
    bootstrap = paired_case_bootstrap(evaluated)
    checks = {
        "pooled_delta_at_most_minus_0_010m": delta <= -0.01,
        "paired_case_bootstrap_CI90_upper_below_zero": bootstrap["ci90_upper_m"] < 0.0,
        "at_least_two_of_three_folds_improve": sum(
            item["delta_m"] < 0.0 for item in by_fold.values()
        )
        >= 2,
        "no_station_degrades_over_0_010m": max(
            item["delta_m"] for item in by_station.values()
        )
        <= 0.01,
        "lead_18_non_degrading": by_lead["18"]["delta_m"] <= 0.0,
        "lead_24_non_degrading": by_lead["24"]["delta_m"] <= 0.0,
    }
    passed = bool(all(checks.values()))
    return {
        "pass": passed,
        "decision": (
            "PASS_OUTER_RESEARCH_REQUIRES_HIDDEN_SCORE_NO_PROMOTION"
            if passed
            else "NO_GO_KEEP_FROZEN_INCUMBENT"
        ),
        "checks": checks,
        "incumbent_rmse_m": incumbent_rmse,
        "candidate_rmse_m": candidate_rmse,
        "candidate_minus_incumbent_rmse_m": delta,
        "by_fold": by_fold,
        "by_station": by_station,
        "by_lead": by_lead,
        "paired_case_bootstrap": bootstrap,
        "rows": int(len(evaluated)),
        "cases": int(evaluated[["fold", "anchor_id"]].drop_duplicates().shape[0]),
        "adaptive_research": True,
        "independent_holdout": False,
        "promotion_permitted_without_hidden_score": False,
    }


def validate_qa_go_receipt(
    receipt: Mapping[str, Any],
    *,
    experiment_id: str,
    dry_receipt_sha256: str,
    implementation_sha256: Mapping[str, str],
) -> None:
    expected = {
        "experiment_id": experiment_id,
        "decision": "QA_GO_OUTER_V4",
        "dry_receipt_sha256": dry_receipt_sha256,
        "implementation_sha256": dict(implementation_sha256),
        "outer_model_execution_count": 0,
        "incumbent_prediction_read_count": 0,
        "designated_target_read_count": 0,
        "test_context_read_count": 0,
        "submission_write_count": 0,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise PermissionError(f"QA GO receipt differs at {key}")
    if not str(receipt.get("reviewer", "")).strip():
        raise PermissionError("QA GO receipt lacks a reviewer identity")


def _group_deltas(frame: pd.DataFrame, column: str) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for value, group in frame.groupby(column, sort=True, observed=True):
        incumbent = float(
            np.sqrt(np.mean(np.square(group["incumbent_final"] - group["target_hs"])))
        )
        candidate = float(
            np.sqrt(np.mean(np.square(group["candidate_final"] - group["target_hs"])))
        )
        result[str(value)] = {
            "incumbent_rmse_m": incumbent,
            "candidate_rmse_m": candidate,
            "delta_m": candidate - incumbent,
        }
    return result


def _validate_prediction_keys(frame: pd.DataFrame, expected_keys: pd.DataFrame) -> None:
    if frame.duplicated(list(PAIR_KEYS)).any():
        raise OuterResearchError("blind prediction keys are duplicated")
    actual = frame.loc[:, PAIR_KEYS].sort_values(list(PAIR_KEYS)).reset_index(drop=True)
    expected = expected_keys.loc[:, PAIR_KEYS].sort_values(list(PAIR_KEYS)).reset_index(drop=True)
    if not actual.equals(expected):
        raise OuterResearchError("blind prediction keys differ from frozen membership")


def _read_seal(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_file():
        raise PermissionError("required blind seal is missing")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("sealed") is not True:
        raise PermissionError("blind seal is not marked sealed")
    return payload


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, role: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise OuterResearchError(f"{role} lacks columns: {sorted(missing)}")


__all__ = [
    "COMPONENT_BLIND_COLUMNS",
    "EVALUATED_ADDITIONAL_COLUMNS",
    "FINAL_BLIND_COLUMNS",
    "FoldLibraryScope",
    "FrozenOOFStageVault",
    "OuterResearchError",
    "PAIR_KEYS",
    "TrainingTargetVault",
    "attach_designated_targets",
    "canonical_membership_hashes",
    "compose_final_blind",
    "evaluate_outer_gate",
    "extract_native_20m_histories",
    "hash_integer_array",
    "paired_case_bootstrap",
    "read_membership_keys_only",
    "sha256_file",
    "validate_component_blind",
    "validate_final_blind",
    "validate_membership_keys",
    "validate_qa_go_receipt",
]
