"""Audit whether existing P3 bases add information beyond the champion alpha axis.

Only historical OOF predictions and their historical targets are used.  The
orthogonal projection coefficient is prediction-only; targets are used solely
for the preregistered historical evaluation and clustered confidence interval.
No official test, sample submission, hidden answer, or upload path is read.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "reports"
    / "parallel_breakthrough_deep_research_20260828_v14"
    / "p3_orthogonal_basis_audit.json"
)
KEYS = ["fold", "anchor_id", "station", "lead_h"]
ALPHA = -10.21743189862218
WEIGHT = 0.25
SEED = 20260828
REPLICATES = 5000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def rmse(actual: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(prediction)) ** 2)))


def grouped_delta(frame: pd.DataFrame, prediction: np.ndarray, column: str) -> dict[str, float]:
    output: dict[str, float] = {}
    for value, index in frame.groupby(column, sort=True).groups.items():
        rows = np.asarray(list(index), dtype=int)
        output[str(value)] = rmse(frame.loc[rows, "target_hs"], prediction[rows]) - rmse(
            frame.loc[rows, "target_hs"], frame.loc[rows, "champion_prediction"]
        )
    return output


def bootstrap_delta(
    frame: pd.DataFrame, prediction: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    case_keys = frame[["fold", "anchor_id", "station"]].drop_duplicates().reset_index(drop=True)
    case_rows = []
    for row in case_keys.itertuples(index=False):
        mask = (
            (frame["fold"].astype(str).to_numpy() == str(row.fold))
            & (frame["anchor_id"].astype(str).to_numpy() == str(row.anchor_id))
            & (frame["station"].astype(str).to_numpy() == str(row.station))
        )
        rows = np.flatnonzero(mask)
        require(len(rows) == 6, "each case must contain six leads")
        case_rows.append(rows)
    require(len(case_rows) == 181, "expected 181 historical cases")
    truth = frame["target_hs"].to_numpy(float)
    champion = frame["champion_prediction"].to_numpy(float)
    draws = np.empty(REPLICATES, dtype=float)
    for replicate in range(REPLICATES):
        sampled = rng.integers(0, len(case_rows), size=len(case_rows))
        rows = np.concatenate([case_rows[index] for index in sampled])
        draws[replicate] = rmse(truth[rows], prediction[rows]) - rmse(
            truth[rows], champion[rows]
        )
    lower, upper = np.quantile(draws, [0.05, 0.95])
    return float(lower), float(upper)


def load_basis(path: Path, prediction_column: str, *, prefix: bool = False) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if prefix:
        frame = frame.loc[np.isclose(frame["prefix_fraction"].to_numpy(float), 1.0)]
    require(not frame.duplicated(KEYS).any(), f"duplicate basis keys: {path}")
    return frame[KEYS + [prediction_column]].rename(columns={prediction_column: "basis_prediction"})


def main() -> None:
    canonical_path = (
        ROOT
        / "artifacts"
        / "p3_champion_lineage_matched_energy_residual_replay_20260828_v1"
        / "sealed_candidate_predictions.parquet"
    )
    truth_path = ROOT / "artifacts" / "p3_corrected_repeated_forward_catboost_v2" / "oof.parquet"
    canonical = pd.read_parquet(canonical_path)
    truth = pd.read_parquet(truth_path, columns=KEYS + ["target_hs"])
    require(len(canonical) == 1086 and not canonical.duplicated(KEYS).any(), "canonical surface")
    require(not truth.duplicated(KEYS).any(), "truth keys duplicate")
    frame = canonical.merge(truth, on=KEYS, how="left", validate="one_to_one")
    require(frame["target_hs"].notna().all(), "historical target merge is incomplete")
    require(frame.index.equals(pd.RangeIndex(len(frame))), "canonical index changed")
    axis_prediction = frame["o_prediction"] + ALPHA * (
        frame["a_prediction"] - frame["o_prediction"]
    )
    active = frame["champion_axis_active"].astype(bool).to_numpy()
    require(
        np.max(
            np.abs(
                axis_prediction.to_numpy(float)[active]
                - frame["champion_prediction"].to_numpy(float)[active]
            )
        )
        <= 1e-12,
        "champion alpha identity changed",
    )

    basis_specs = [
        (
            "ERA5",
            canonical_path,
            "transfer_prediction",
            False,
        ),
        (
            "KMA",
            ROOT
            / "artifacts"
            / "p3_kma_calibrated_longlead_blend_v2"
            / "one_shot"
            / "blind_predictions.parquet",
            "calibrated_source",
            False,
        ),
        (
            "energy_state_space",
            ROOT
            / "artifacts"
            / "p3_station_stable_energy_state_space_20260823_v1"
            / "oof"
            / "learning_curve_oof.parquet",
            "challenger_prediction",
            True,
        ),
        (
            "sequence_checkpoint",
            ROOT
            / "artifacts"
            / "p3_causal_forcing_sequence_checkpoint_nested_20260827_v2"
            / "oof"
            / "learning_curve_oof.parquet",
            "checkpoint_nested_prediction",
            False,
        ),
    ]
    champion = frame["champion_prediction"].to_numpy(float)
    target = frame["target_hs"].to_numpy(float)
    x_all = frame["a_prediction"].to_numpy(float) - frame["o_prediction"].to_numpy(float)
    long_lead = frame["lead_h"].isin([18, 24]).to_numpy()
    incumbent_rmse = rmse(target, champion)
    rng = np.random.default_rng(SEED)
    results = {}
    direction_cache = {}
    for name, path, prediction_column, prefix in basis_specs:
        if path == canonical_path:
            basis = canonical[KEYS + [prediction_column]].rename(
                columns={prediction_column: "basis_prediction"}
            )
        else:
            basis = load_basis(path, prediction_column, prefix=prefix)
        merged = frame[KEYS].merge(basis, on=KEYS, how="left", validate="one_to_one")
        basis_values = merged["basis_prediction"].to_numpy(float)
        support = long_lead & np.isfinite(basis_values)
        d = basis_values[support] - champion[support]
        x = x_all[support]
        beta = float(np.dot(d, x) / np.dot(x, x))
        perpendicular = d - beta * x
        orthogonality_cosine = float(
            np.dot(perpendicular, x)
            / max(np.linalg.norm(perpendicular) * np.linalg.norm(x), np.finfo(float).eps)
        )
        raw_prediction = champion.copy()
        raw_prediction[support] = champion[support] + WEIGHT * d
        orth_prediction = champion.copy()
        orth_prediction[support] = champion[support] + WEIGHT * perpendicular
        ci90 = bootstrap_delta(frame, orth_prediction, rng)
        full_direction = np.full(len(frame), np.nan, dtype=float)
        full_direction[support] = d
        direction_cache[name] = full_direction
        result = {
            "path": str(path.relative_to(ROOT)),
            "path_sha256": sha256(path),
            "prediction_column": prediction_column,
            "supported_cases": int(
                frame.loc[support, ["fold", "anchor_id", "station"]].drop_duplicates().shape[0]
            ),
            "supported_rows": int(support.sum()),
            "raw_blend_delta_rmse": rmse(target, raw_prediction) - incumbent_rmse,
            "prediction_only_beta": beta,
            "orthogonality_cosine": orthogonality_cosine,
            "orthogonal_blend_delta_rmse": rmse(target, orth_prediction) - incumbent_rmse,
            "orthogonal_blend_ci90": list(ci90),
            "orthogonal_delta_by_fold": grouped_delta(frame, orth_prediction, "fold"),
            "orthogonal_delta_by_station": grouped_delta(frame, orth_prediction, "station"),
            "orthogonal_delta_by_lead": grouped_delta(frame, orth_prediction, "lead_h"),
        }
        group_values = [
            *result["orthogonal_delta_by_fold"].values(),
            *result["orthogonal_delta_by_station"].values(),
            *result["orthogonal_delta_by_lead"].values(),
        ]
        result["promotion_gate"] = {
            "support_at_least_95_percent": result["supported_cases"] >= int(np.ceil(181 * 0.95)),
            "ci90_upper_below_zero": ci90[1] < 0.0,
            "all_groups_nonworse": max(group_values) <= 0.0,
            "pooled_delta_at_most_minus_0_003": result["orthogonal_blend_delta_rmse"] <= -0.003,
        }
        result["promotion_pass"] = all(result["promotion_gate"].values())
        results[name] = result

    correlations = {}
    names = list(direction_cache)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            left_values = direction_cache[left]
            right_values = direction_cache[right]
            common = np.isfinite(left_values) & np.isfinite(right_values)
            correlations[f"{left}__{right}"] = float(
                np.corrcoef(left_values[common], right_values[common])[0, 1]
            )

    output = {
        "schema_version": "p3.champion_orthogonal_prediction_basis.audit.20260828.v1",
        "status": "NO_GO_NO_OFFICIAL_CANDIDATE",
        "completed_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "historical_rows": len(frame),
        "historical_cases": 181,
        "champion_rmse": incumbent_rmse,
        "alpha": ALPHA,
        "blend_weight": WEIGHT,
        "bootstrap": {"seed": SEED, "replicates": REPLICATES, "cluster": "case"},
        "results": results,
        "direction_correlations": correlations,
        "decision": {
            "promoted": [name for name, result in results.items() if result["promotion_pass"]],
            "official_candidate_created": False,
            "upload_performed": False,
            "same_family_veto": "ERA5 is also vetoed because its family failed the official probe.",
        },
        "input_hashes": {
            "canonical": sha256(canonical_path),
            "truth": sha256(truth_path),
        },
        "leakage_contract": {
            "official_test_read": False,
            "official_sample_or_submission_read": False,
            "hidden_answer_read": False,
            "historical_oof_truth_used_for_evaluation_only": True,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, OUTPUT)
    print(
        json.dumps(
            {
                name: {
                    "orth_delta": result["orthogonal_blend_delta_rmse"],
                    "ci90": result["orthogonal_blend_ci90"],
                    "pass": result["promotion_pass"],
                }
                for name, result in results.items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
