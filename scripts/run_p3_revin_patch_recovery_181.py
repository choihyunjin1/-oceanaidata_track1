"""One-time 181-case recovery for the sealed P3 RevIN Patch blind predictions.

This runner never trains or predicts. It intersects the already-sealed blind predictions with
the frozen incumbent OOF keys, seals every target-free artifact and a durable pre-open receipt,
then permits exactly one final filtered target read for metric generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as pyarrow_dataset

EXPECTED_BLIND_MANIFEST_SHA256 = "61337f2044de39b90e249c6b4a422140792badef821a554cf7fe28e301232232"
EXPECTED_PRIOR_EXPOSURE_RECEIPT_SHA256 = (
    "cc6089927ec0abb7573a46c6d508df3bbbc5211c3e87e743ca4aa7a249b60c1d"
)
EXPECTED_INCUMBENT_OOF_SHA256 = "2850ac32676b7425935b5a3dd40892c589f4c48b1a6d2f66cdf184b39744bedd"
EXPECTED_INCUMBENT_SUBMISSION_SHA256 = (
    "d89e69b940c90ea1fbecf1e882bee69136255fffb12601d2fc853d032900e5b7"
)
EXPECTED_ANCHOR_CACHE_SHA256 = "07452389a19efd63121f4465a9c08cf7f9ef9e58cf1e3ea1f577e2dca5d8611a"
PAIR_KEYS = ["fold", "anchor_id", "station", "lead_h"]
LEADS = (3, 6, 9, 12, 18, 24)
SEEDS = (20260821, 20260822, 20260823)
EXCLUDED_CANDIDATE_ONLY = {3818}
EXCLUDED_INCUMBENT_ONLY = {3739}
BOOTSTRAP_REPLICATES = 5000


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    with temporary.open("rb+") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def update_status(
    path: Path,
    *,
    state: str,
    phase: str,
    progress: float,
    detail: str,
    result: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "title": "P3 sealed-blind 181-case recovery",
        "status": state,
        "phase": phase,
        "progress": float(progress),
        "detail": detail,
        "updated_at": _now(),
    }
    if result is not None:
        payload["result"] = result
    atomic_json(path, payload)


def _verify_source_manifest(manifest_path: Path) -> dict[str, Any]:
    if sha256_file(manifest_path) != EXPECTED_BLIND_MANIFEST_SHA256:
        raise ValueError("sealed blind manifest SHA differs from the approved recovery source")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("sealed") is not True or len(payload.get("prediction_files", {})) != 9:
        raise ValueError("approved blind manifest is not sealed for nine files")
    for relative, record in payload["prediction_files"].items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("unsafe source path in sealed blind manifest")
        path = manifest_path.parent / relative_path
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"sealed blind prediction changed: {relative}")
    return payload


def build_intersection_artifacts(
    blind: pd.DataFrame,
    incumbent: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    expected_blind = [
        "fold",
        "seed",
        "anchor_id",
        "station",
        "episode_id",
        "lead_h",
        "current_hs",
        "patch_prediction",
    ]
    if list(blind.columns) != expected_blind:
        raise ValueError("sealed blind prediction schema mismatch")
    if list(incumbent.columns) != [*PAIR_KEYS, "incumbent_prediction"]:
        raise ValueError("incumbent key/prediction schema mismatch")
    if set(blind["seed"].unique()) != set(SEEDS):
        raise ValueError("sealed blind prediction seed set differs from the fixed three seeds")
    if blind.duplicated([*PAIR_KEYS, "seed"]).any() or incumbent.duplicated(PAIR_KEYS).any():
        raise ValueError("duplicate sealed blind or incumbent key")

    candidate_keys = blind[PAIR_KEYS].drop_duplicates()
    key_audit = candidate_keys.merge(
        incumbent[PAIR_KEYS], on=PAIR_KEYS, how="outer", indicator=True
    )
    candidate_only = key_audit.loc[key_audit["_merge"].eq("left_only")]
    incumbent_only = key_audit.loc[key_audit["_merge"].eq("right_only")]
    if set(candidate_only["anchor_id"].astype(int)) != EXCLUDED_CANDIDATE_ONLY:
        raise ValueError("candidate-only exclusion differs from approved anchor 3818")
    if set(incumbent_only["anchor_id"].astype(int)) != EXCLUDED_INCUMBENT_ONLY:
        raise ValueError("incumbent-only exclusion differs from approved anchor 3739")

    intersection_keys = key_audit.loc[key_audit["_merge"].eq("both"), PAIR_KEYS].copy()
    intersection_keys = intersection_keys.sort_values(PAIR_KEYS).reset_index(drop=True)
    if len(intersection_keys) != 1086 or intersection_keys["anchor_id"].nunique() != 181:
        raise ValueError("approved intersection must contain 1,086 rows and 181 cases")
    if intersection_keys["anchor_id"].isin([3818, 3739]).any():
        raise ValueError("excluded mismatch anchor entered the intersection")

    selected_blind = blind.merge(
        intersection_keys, on=PAIR_KEYS, how="inner", validate="many_to_one"
    )
    seed_count = selected_blind.groupby(PAIR_KEYS, observed=True)["seed"].nunique()
    row_count = selected_blind.groupby(PAIR_KEYS, observed=True).size()
    if not seed_count.eq(3).all() or not row_count.eq(3).all():
        raise ValueError("each intersection key must contain exactly the three fixed seeds")
    seed_mean = (
        selected_blind.groupby(
            [*PAIR_KEYS, "episode_id", "current_hs"], as_index=False, observed=True
        )["patch_prediction"]
        .mean()
        .sort_values(PAIR_KEYS)
        .reset_index(drop=True)
    )
    incumbent_intersection = (
        incumbent.merge(intersection_keys, on=PAIR_KEYS, how="inner", validate="one_to_one")
        .sort_values(PAIR_KEYS)
        .reset_index(drop=True)
    )
    blind_blend = seed_mean.merge(
        incumbent_intersection,
        on=PAIR_KEYS,
        how="inner",
        validate="one_to_one",
    ).sort_values(PAIR_KEYS)
    lead_tuple = blind_blend.groupby(["fold", "anchor_id"], observed=True)["lead_h"].agg(tuple)
    if not lead_tuple.map(lambda value: value == LEADS).all():
        raise ValueError("intersection cases do not contain six ordered leads")
    incumbent_matrix = blind_blend["incumbent_prediction"].to_numpy(dtype=np.float64).reshape(-1, 6)
    patch_matrix = blind_blend["patch_prediction"].to_numpy(dtype=np.float64).reshape(-1, 6)
    candidate = incumbent_matrix.copy()
    candidate[:, 3:] = 0.8 * incumbent_matrix[:, 3:] + 0.2 * patch_matrix[:, 3:]
    blind_blend["candidate_prediction"] = candidate.reshape(-1)
    protected = blind_blend["lead_h"].isin([3, 6, 9])
    if not np.array_equal(
        blind_blend.loc[protected, "candidate_prediction"].to_numpy(),
        blind_blend.loc[protected, "incumbent_prediction"].to_numpy(),
    ):
        raise AssertionError("3/6/9h incumbent predictions are not bit-exact")
    if blind_blend.duplicated(PAIR_KEYS).any() or len(blind_blend) != 1086:
        raise ValueError("fixed blend artifact is not one-to-one on 1,086 keys")
    if any("target" in column.lower() for column in blind_blend.columns):
        raise ValueError("target column entered a pre-open recovery artifact")

    exclusions = {
        "candidate_only": candidate_only.sort_values(PAIR_KEYS).reset_index(drop=True),
        "incumbent_only": incumbent_only.sort_values(PAIR_KEYS).reset_index(drop=True),
    }
    audit = {
        "intersection_rows": 1086,
        "intersection_cases": 181,
        "seed_rows_before_mean": int(len(selected_blind)),
        "fixed_seed_count_per_key": 3,
        "candidate_only_anchor_ids": [3818],
        "incumbent_only_anchor_ids": [3739],
        "protected_3_6_9_bit_exact": True,
        "one_to_one_keys": True,
    }
    return {
        "intersection_keys": intersection_keys,
        "seed_mean_patch": seed_mean,
        "incumbent_key_prediction": incumbent_intersection,
        "fixed_blend_blind": blind_blend,
        **exclusions,
    }, audit


@dataclass
class FinalIntersectionTargetVault:
    path: Path
    open_count: int = 0
    access_log: list[dict[str, Any]] = field(default_factory=list)

    def open_final_once(
        self,
        anchor_ids: np.ndarray,
        *,
        pre_open_receipt_path: Path,
        pre_open_manifest_path: Path,
    ) -> pd.DataFrame:
        if self.open_count:
            raise PermissionError("third target opening is forbidden")
        receipt = json.loads(pre_open_receipt_path.read_text(encoding="utf-8"))
        if receipt.get("family_cumulative_target_open_before") != 1:
            raise PermissionError("pre-open receipt does not record the prior single opening")
        if receipt.get("prior_metrics_generation_count") != 0:
            raise PermissionError("pre-open receipt does not record zero prior metrics")
        if receipt.get("approved_current_target_open_max") != 1:
            raise PermissionError("recovery target-open maximum differs from one")
        if receipt.get("approved_current_metrics_generation_max") != 1:
            raise PermissionError("recovery metric-generation maximum differs from one")
        if receipt.get("third_target_open_forbidden") is not True:
            raise PermissionError("pre-open receipt does not forbid a third family opening")
        if receipt.get("fsync_completed_before_final_target_open") is not True:
            raise PermissionError("pre-open receipt does not confirm durable fsync")
        if receipt.get("pre_open_manifest_sha256") != sha256_file(pre_open_manifest_path):
            raise PermissionError("pre-open receipt and manifest SHA differ")
        ids = np.asarray(anchor_ids, dtype=np.int64)
        if len(ids) != 181 or len(np.unique(ids)) != 181:
            raise ValueError("final target request must contain 181 unique intersection IDs")
        if np.isin(ids, [3818, 3739]).any():
            raise PermissionError("excluded mismatch anchor entered final target request")
        columns = ["anchor_id", *[f"target_{lead}" for lead in LEADS]]
        dataset = pyarrow_dataset.dataset(self.path, format="parquet")
        table = dataset.to_table(
            columns=columns,
            filter=pyarrow_dataset.field("anchor_id").isin(ids.tolist()),
        )
        frame = table.to_pandas().set_index("anchor_id").loc[ids].reset_index()
        if len(frame) != 181 or frame["anchor_id"].duplicated().any():
            raise ValueError("final filtered target read does not cover 181 cases exactly")
        if not np.isfinite(frame.drop(columns="anchor_id").to_numpy(dtype=np.float64)).all():
            raise ValueError("final filtered target read contains non-finite labels")
        self.open_count += 1
        self.access_log.append(
            {
                "purpose": "second_and_final_family_target_open_intersection_only",
                "cases": 181,
                "family_cumulative_target_open_after": 2,
            }
        )
        return frame


def _target_long(frame: pd.DataFrame, fold_by_anchor: dict[int, str]) -> pd.DataFrame:
    result = frame.melt(
        id_vars="anchor_id",
        value_vars=[f"target_{lead}" for lead in LEADS],
        var_name="lead_h",
        value_name="target_hs",
    )
    result["lead_h"] = result["lead_h"].str.removeprefix("target_").astype(np.int64)
    result["fold"] = result["anchor_id"].map(fold_by_anchor)
    if result["fold"].isna().any():
        raise ValueError("intersection target could not be mapped to a fold")
    return result


def rmse(truth: pd.Series | np.ndarray, prediction: pd.Series | np.ndarray) -> float:
    return float(
        np.sqrt(
            np.mean(
                np.square(
                    np.asarray(prediction, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
                )
            )
        )
    )


def metric_slices(frame: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "rmse": rmse(frame["target_hs"], frame[prediction_column]),
        "rows": int(len(frame)),
        "cases": int(frame[["fold", "anchor_id"]].drop_duplicates().shape[0]),
    }
    for column, name in (("lead_h", "by_lead"), ("station", "by_station"), ("fold", "by_fold")):
        result[name] = {
            str(value): rmse(group["target_hs"], group[prediction_column])
            for value, group in frame.groupby(column, sort=True, observed=True)
        }
    return result


def paired_block_bootstrap(
    frame: pd.DataFrame,
    *,
    block_columns: list[str],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    work = frame.reset_index(drop=True)
    blocks = [
        group.index.to_numpy(dtype=np.int64)
        for _, group in work.groupby(block_columns, sort=False, observed=True)
    ]
    if not blocks or replicates != BOOTSTRAP_REPLICATES:
        raise ValueError("recovery bootstrap requires non-empty blocks and 5,000 replicates")
    truth = work["target_hs"].to_numpy(dtype=np.float64)
    candidate = work["candidate_prediction"].to_numpy(dtype=np.float64)
    incumbent = work["incumbent_prediction"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    delta = np.empty(replicates, dtype=np.float64)
    for number in range(replicates):
        index = np.concatenate(
            [blocks[item] for item in rng.integers(0, len(blocks), size=len(blocks))]
        )
        delta[number] = rmse(truth[index], candidate[index]) - rmse(truth[index], incumbent[index])
    return {
        "replicates": replicates,
        "seed": seed,
        "blocks": int(len(blocks)),
        "ci90": np.quantile(delta, [0.05, 0.95]).tolist(),
        "median": float(np.median(delta)),
        "probability_improved": float(np.mean(delta < 0.0)),
    }


def evaluate_recovery_gate(
    evaluated: pd.DataFrame,
    case_bootstrap: dict[str, Any],
    episode_bootstrap: dict[str, Any],
) -> dict[str, Any]:
    candidate = metric_slices(evaluated, "candidate_prediction")
    incumbent = metric_slices(evaluated, "incumbent_prediction")
    delta = float(candidate["rmse"] - incumbent["rmse"])
    lead_checks = {
        str(lead): candidate["by_lead"][str(lead)] <= incumbent["by_lead"][str(lead)]
        for lead in (18, 24)
    }
    station_checks = {
        station: candidate["by_station"][station] <= incumbent["by_station"][station] + 0.01
        for station in candidate["by_station"]
    }
    checks = {
        "delta_rmse_at_most_minus_0p010": delta <= -0.010,
        "case_bootstrap_ci90_upper_below_zero": float(case_bootstrap["ci90"][1]) < 0.0,
        "episode_bootstrap_ci90_upper_below_zero": float(episode_bootstrap["ci90"][1]) < 0.0,
        "lead_18_non_degrading": bool(lead_checks["18"]),
        "lead_24_non_degrading": bool(lead_checks["24"]),
        "all_station_degradation_at_most_0p010": bool(all(station_checks.values())),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "candidate": candidate,
        "incumbent": incumbent,
        "delta_rmse": delta,
        "lead_checks": lead_checks,
        "station_checks": station_checks,
        "legacy_182_case_absolute_thresholds": {
            "status": "reference_only_not_a_recovery_gate",
            "incumbent_local_rmse": 0.7801609198910191,
            "maximum_candidate_rmse": 0.7701609198910191,
        },
    }


def _write_pre_open_artifacts(
    output: Path,
    artifacts: dict[str, pd.DataFrame],
    audit: dict[str, Any],
    *,
    source_manifest_path: Path,
    prior_receipt_path: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    pre_open = output / "pre_open"
    erratum = {
        "created_at": _now(),
        "scope": "approved 181-case exact-key recovery only",
        "prior_attempt": {
            "target_open_count": 1,
            "metrics_generation_count": 0,
            "failure": "one outer case key mismatch before metrics",
        },
        "approved_exclusions": {
            "candidate_only_anchor_id": 3818,
            "incumbent_only_anchor_id": 3739,
        },
        "prohibitions": {
            "retraining": True,
            "reprediction": True,
            "corrected_code_predictions": True,
            "checkpoint_or_submission_mutation": True,
            "third_target_open": True,
            "second_metrics_generation": True,
        },
    }
    erratum_path = pre_open / "erratum.json"
    atomic_json(erratum_path, erratum)
    artifact_paths: dict[str, Path] = {}
    for name in (
        "intersection_keys",
        "seed_mean_patch",
        "incumbent_key_prediction",
        "fixed_blend_blind",
        "candidate_only",
        "incumbent_only",
    ):
        path = pre_open / f"{name}.parquet"
        atomic_parquet(path, artifacts[name])
        artifact_paths[name] = path
    manifest = {
        "created_at": _now(),
        "sealed": True,
        "source_blind_manifest_sha256": sha256_file(source_manifest_path),
        "prior_exposure_receipt_sha256": sha256_file(prior_receipt_path),
        "audit": audit,
        "files": {
            "erratum.json": sha256_file(erratum_path),
            **{path.name: sha256_file(path) for path in artifact_paths.values()},
        },
        "target_columns_opened": False,
        "fsync_completed": True,
    }
    manifest_path = pre_open / "manifest.json"
    atomic_json(manifest_path, manifest)
    receipt = {
        "created_at": _now(),
        "pre_open_manifest_sha256": sha256_file(manifest_path),
        "source_blind_manifest_sha256": EXPECTED_BLIND_MANIFEST_SHA256,
        "family_cumulative_target_open_before": 1,
        "prior_metrics_generation_count": 0,
        "approved_current_target_open_max": 1,
        "approved_current_metrics_generation_max": 1,
        "family_cumulative_target_open_after_if_success": 2,
        "third_target_open_forbidden": True,
        "intersection_cases": 181,
        "intersection_rows": 1086,
        "excluded_anchor_ids": [3818, 3739],
        "fsync_completed_before_final_target_open": True,
    }
    receipt_path = pre_open / "receipt.json"
    atomic_json(receipt_path, receipt)
    # Re-open from disk and rehash immediately before returning authorization to the vault.
    persisted = json.loads(receipt_path.read_text(encoding="utf-8"))
    if persisted != receipt or sha256_file(manifest_path) != receipt["pre_open_manifest_sha256"]:
        raise RuntimeError("durable pre-open receipt verification failed")
    return manifest_path, receipt_path, manifest


def run_recovery(
    *,
    root: Path,
    output: Path,
    status_path: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("immutable recovery folder already exists; retry is forbidden")
    source = root / "artifacts/p3_revin_patch_v1/full_one_shot"
    source_manifest_path = source / "blind_prediction_manifest.json"
    prior_receipt_path = source / "outer_label_exposure_receipt.json"
    incumbent_path = root / "artifacts/p3/long_persistence_shrink/oof.parquet"
    incumbent_submission_path = root / "submissions/p3_long_persistence_shrink/submission.csv"
    target_cache_path = root / "artifacts/p3/features_all20_v1/train_anchors.parquet"
    if sha256_file(prior_receipt_path) != EXPECTED_PRIOR_EXPOSURE_RECEIPT_SHA256:
        raise ValueError("prior exposure receipt SHA changed")
    if sha256_file(incumbent_path) != EXPECTED_INCUMBENT_OOF_SHA256:
        raise ValueError("frozen incumbent OOF SHA changed")
    if sha256_file(incumbent_submission_path) != EXPECTED_INCUMBENT_SUBMISSION_SHA256:
        raise ValueError("frozen incumbent submission SHA changed")
    if sha256_file(target_cache_path) != EXPECTED_ANCHOR_CACHE_SHA256:
        raise ValueError("frozen target cache SHA changed")
    source_manifest = _verify_source_manifest(source_manifest_path)
    output.mkdir(parents=True, exist_ok=False)
    update_status(
        status_path,
        state="running",
        phase="seal_181_case_target_free_intersection",
        progress=10,
        detail="기존 sealed blind만 사용해 181-case pre-open artifact SHA+fsync 중",
    )
    blind = pd.concat(
        [pd.read_parquet(source / relative) for relative in source_manifest["prediction_files"]],
        ignore_index=True,
    )
    incumbent = pd.read_parquet(
        incumbent_path,
        columns=[*PAIR_KEYS, "prediction"],
    ).rename(columns={"prediction": "incumbent_prediction"})
    artifacts, audit = build_intersection_artifacts(blind, incumbent)
    manifest_path, receipt_path, pre_open_manifest = _write_pre_open_artifacts(
        output,
        artifacts,
        audit,
        source_manifest_path=source_manifest_path,
        prior_receipt_path=prior_receipt_path,
    )
    update_status(
        status_path,
        state="running",
        phase="second_and_final_intersection_target_open",
        progress=55,
        detail="pre-open receipt fsync 검증 완료 · intersection 181개 target 마지막 1회 개방",
    )

    blind_blend = artifacts["fixed_blend_blind"]
    intersection_ids = np.sort(blind_blend["anchor_id"].unique().astype(np.int64))
    fold_by_anchor = {
        int(row.anchor_id): str(row.fold)
        for row in blind_blend[["fold", "anchor_id"]].drop_duplicates().itertuples(index=False)
    }
    vault = FinalIntersectionTargetVault(target_cache_path)
    target_wide = vault.open_final_once(
        intersection_ids,
        pre_open_receipt_path=receipt_path,
        pre_open_manifest_path=manifest_path,
    )
    post_open_receipt = {
        "created_at": _now(),
        "family_cumulative_target_open_count": 2,
        "current_recovery_target_open_count": 1,
        "opened_cases": 181,
        "opened_anchor_ids_sha256": hashlib.sha256(intersection_ids.tobytes()).hexdigest(),
        "excluded_anchor_ids_not_opened": [3818, 3739],
        "third_target_open_forbidden": True,
        "metrics_generation_count_before_current": 0,
        "approved_current_metrics_generation_max": 1,
        "pre_open_receipt_sha256": sha256_file(receipt_path),
        "fsync_completed_immediately_after_open": True,
    }
    post_open_receipt_path = output / "post_open" / "receipt.json"
    atomic_json(post_open_receipt_path, post_open_receipt)
    if vault.open_count != 1:
        raise AssertionError("final recovery target vault open count differs from one")

    target_long = _target_long(target_wide, fold_by_anchor)
    evaluated = blind_blend.merge(
        target_long,
        on=["fold", "anchor_id", "lead_h"],
        how="inner",
        validate="one_to_one",
    ).sort_values(PAIR_KEYS)
    if len(evaluated) != 1086 or evaluated["anchor_id"].nunique() != 181:
        raise ValueError("evaluated recovery grain differs from 1,086 rows / 181 cases")
    case_bootstrap = paired_block_bootstrap(
        evaluated,
        block_columns=["fold", "anchor_id"],
        replicates=BOOTSTRAP_REPLICATES,
        seed=20260821,
    )
    episode_bootstrap = paired_block_bootstrap(
        evaluated,
        block_columns=["fold", "station", "episode_id"],
        replicates=BOOTSTRAP_REPLICATES,
        seed=20260822,
    )
    gate = evaluate_recovery_gate(evaluated, case_bootstrap, episode_bootstrap)
    evaluated_path = output / "post_open" / "evaluated.parquet"
    atomic_parquet(evaluated_path, evaluated)
    metrics = {
        "created_at": _now(),
        "experiment": "p3_revin_patch_sealed_blind_recovery_181",
        "status": "gate_passed" if gate["passed"] else "gate_failed",
        "grain": {"rows": 1086, "cases": 181, "leads": list(LEADS)},
        "source": {
            "sealed_blind_manifest_sha256": EXPECTED_BLIND_MANIFEST_SHA256,
            "prior_exposure_receipt_sha256": EXPECTED_PRIOR_EXPOSURE_RECEIPT_SHA256,
            "incumbent_oof_sha256": EXPECTED_INCUMBENT_OOF_SHA256,
        },
        "exclusions": {"candidate_only": 3818, "incumbent_only": 3739},
        "case_bootstrap": case_bootstrap,
        "episode_bootstrap": episode_bootstrap,
        "gate": gate,
        "target_access": {
            "family_cumulative_open_count": 2,
            "current_recovery_open_count": 1,
            "third_open_forbidden": True,
            "access_log": vault.access_log,
        },
        "invariants": {
            "retraining_or_reprediction": False,
            "corrected_code_predictions_used": False,
            "three_seed_mean": True,
            "patch_blend_weight": 0.2,
            "protected_3_6_9_bit_exact": True,
            "one_to_one_keys": True,
            "checkpoint_model_or_submission_changed": False,
            "metrics_generation_count": 1,
        },
    }
    metrics_path = output / "post_open" / "metrics.json"
    if metrics_path.exists():
        raise FileExistsError("recovery metrics already exist; second generation is forbidden")
    atomic_json(metrics_path, metrics)
    final_receipt = {
        "created_at": _now(),
        "immutable_recovery_complete": True,
        "gate_passed": bool(gate["passed"]),
        "family_cumulative_target_open_count": 2,
        "metrics_generation_count": 1,
        "third_target_open_and_second_metrics_generation_forbidden": True,
        "pre_open_manifest_sha256": sha256_file(manifest_path),
        "pre_open_receipt_sha256": sha256_file(receipt_path),
        "post_open_receipt_sha256": sha256_file(post_open_receipt_path),
        "evaluated_sha256": sha256_file(evaluated_path),
        "metrics_sha256": sha256_file(metrics_path),
        "source_blind_manifest_sha256": EXPECTED_BLIND_MANIFEST_SHA256,
        "incumbent_submission_sha256_before_after": sha256_file(incumbent_submission_path),
        "incumbent_oof_sha256_before_after": sha256_file(incumbent_path),
        "anchor_cache_sha256_before_after": sha256_file(target_cache_path),
        "pre_open_artifact_manifest": pre_open_manifest,
    }
    final_receipt_path = output / "FINAL_IMMUTABLE_RECEIPT.json"
    atomic_json(final_receipt_path, final_receipt)
    result = {
        "gate": gate,
        "case_bootstrap": case_bootstrap,
        "episode_bootstrap": episode_bootstrap,
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "final_receipt_path": str(final_receipt_path),
        "final_receipt_sha256": sha256_file(final_receipt_path),
    }
    update_status(
        status_path,
        state="completed_gate_pass" if gate["passed"] else "completed_gate_fail",
        phase="immutable_recovery_complete_no_retry",
        progress=100,
        detail=f"181-case delta RMSE {gate['delta_rmse']:+.6f} · gate {'PASS' if gate['passed'] else 'FAIL'}",
        result=result,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="artifacts/p3_revin_patch_v1/recovery_181_sealed_blind",
    )
    parser.add_argument(
        "--status-path",
        default="artifacts/status/p3_revin_patch_recovery_181.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (root / args.output_dir).resolve()
    status_path = (root / args.status_path).resolve()
    try:
        result = run_recovery(root=root, output=output, status_path=status_path)
    except Exception as error:
        update_status(
            status_path,
            state="failed_no_retry",
            phase="recovery_stopped",
            progress=100,
            detail=f"{type(error).__name__}: {error}",
        )
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
