"""Train-only P1-B provenance and fixed P1-C audit, no model backbone fitting."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import run_p1_depth_contract_repair_20260905_v2 as a  # noqa: E402
import run_p1_score_repair_decoder_20260905_v1 as dec  # noqa: E402

REPORT = ROOT / "reports" / a.RUN
ARTIFACT = ROOT / "artifacts" / a.RUN


def provenance():
    """Read historical key-only surfaces and source metadata, never checkpoints."""
    REPORT.mkdir(parents=True, exist_ok=True)
    output = REPORT / "provenance-audit.json"
    if output.exists():
        raise FileExistsError("provenance audit already recorded")
    old_keys_path = ROOT / "artifacts/p1_current_router_oof_anchor_v1/anchor.parquet"
    if (
        a.old.sha(old_keys_path)
        != "fc5c594aadabec98e0fbf032ff33422691d94d8525d134495a81a47586024536"
    ):
        raise ValueError("router historical surface hash mismatch")
    old_keys = pd.read_parquet(old_keys_path, columns=a.old.KEYS + ["fold"])
    current = pd.concat(
        [
            pd.read_parquet(
                ROOT / "artifacts/p1_score_repair_20260905_v1" / (name + "_intact_oof.parquet"),
                columns=a.old.KEYS + ["fold"],
            )
            for name in ("2025_q2", "2025_q3", "2025_q4")
        ],
        ignore_index=True,
    )
    merged = old_keys.merge(
        current,
        on=a.old.KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_old", "_new"),
    )
    common = merged._merge.eq("both")
    moved = common & merged.fold_old.ne(merged.fold_new)
    receipt = {
        "status": "B_BLOCKED_EXACT_CONTRACT_MISMATCH_NOT_SCIENTIFIC_NO_GO",
        "source_policy": "distributed train-only lineage exists; no blanket P1 invalidation",
        "total_unique_old_keys": len(old_keys),
        "total_unique_new_keys": len(current),
        "same_global_key_set": bool(common.all()),
        "changed_fold_rows": int(moved.sum()),
        "old_fold_counts": old_keys.fold.value_counts().to_dict(),
        "new_fold_counts": current.fold.value_counts().to_dict(),
        "e150_purge_days": 21,
        "old_router_tree_purge_days": 7,
        "new_tree_purge_days": 21,
        "eligible_exact_zero_fit_combination": False,
        "zero_fit_combination_executed": False,
        "gpu_fit_started": False,
        "recovery_requires": "new exact-contract e150 9 historical plus 3 full fits subject to root GPU/time approval",
        "historical_e150_6fit_runtime_seconds": 9438.412428,
        "historical_e150_runtime_is_not_current_runtime_guarantee": True,
        "router_provenance": "general station/layer disagreement rules exist in pre-registered candidate builder; old prediction CSV is not a reproducible training input",
        "gi_provenance": "legacy builder uses general novel AND predicted-type-spike condition but depends on frozen official CSV ancestors; no row patch or CSV copied",
        "fixed_e150_not_trial18_or_epoch125": True,
        "old_unknown_features": "MS-TCN excludes cached full-year nominal depth and computes current-row depth bins using phase train thresholds",
        "official_rows": 0,
        "hidden_rows": 0,
        "raw_observation_rows": 0,
        "csv_written": 0,
        "upload": 0,
        "model_fits": 0,
        "sources": {},
    }
    source_paths = [
        "scripts/build_p1_current_router_oof_anchor_v1.py",
        "scripts/package_preregistered_submission_20260826.py",
        "scripts/build_deadline_probe_set_20260828.py",
        "scripts/run_p1_meaningful_learning_curve_generation_v1.py",
        "configs/p1.toml",
        "configs/experiments/p1_mstcn_checkpoint_diagnostic_20260827_v2.json",
        "configs/experiments/p1_incumbent_preserving_mstcn_asrf_v2.json",
        "artifacts/p1_mstcn_checkpoint_diagnostic_20260827_v2/q3_split.json",
        "artifacts/p1_mstcn_checkpoint_diagnostic_20260827_v2/q4_split.json",
        "artifacts/p1_mstcn_checkpoint_diagnostic_20260827_v2/terminal_result.json",
        "artifacts/p1_current_router_oof_anchor_v1/manifest.json",
    ]
    for path in source_paths:
        receipt["sources"][path] = a.old.sha(ROOT / path)
    if not common.all() or not moved.any():
        raise ValueError("unexpected historical split comparison; do not infer equivalence")
    a.old.write_json(output, receipt)
    print(json.dumps(receipt, ensure_ascii=False))


def paired_block_interval(frame, prediction, reference, replicates=2000, seed=20260905):
    """Same seven-day block weights for both arms, all stations retained together."""
    work = frame.copy()
    dates = pd.to_datetime(work.time, utc=True).dt.tz_convert("Asia/Seoul").dt.floor("D")
    work["_block"] = (dates - dates.min()).dt.days // 7
    counts = []
    for _, part in work.groupby(["fold", "_block"], sort=True):
        pair = []
        for name in (prediction, reference):
            y, p = part.label.to_numpy(bool), part[name].to_numpy(bool)
            pair.extend([np.sum(y & p), np.sum(~y & p), np.sum(y & ~p)])
        counts.append(pair)
    values = np.asarray(counts, dtype=np.int64)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(replicates):
        totals = values[rng.integers(0, len(values), len(values))].sum(axis=0)
        p = 2 * totals[0] / max(1, 2 * totals[0] + totals[1] + totals[2])
        r = 2 * totals[3] / max(1, 2 * totals[3] + totals[4] + totals[5])
        differences.append(p - r)
    return {
        "unit": "7-day shared station blocks within historical fold",
        "blocks": len(values),
        "replicates": replicates,
        "seed": seed,
        "ci90": np.quantile(differences, [0.05, 0.95]).tolist(),
        "role": "retrospective development uncertainty, not fresh confirmation or hard gate",
    }


def decoder():
    """A then B are complete before fixed OFF/ON comparison; no tuning grid."""
    if (
        a.old.sha(dec.__file__)
        != "d5ed9ff15745b7e400066eff2331c533d6fd3a420e941d04cf317ef40b530db9"
    ):
        raise ValueError("fixed decoder source drift")
    result = json.loads((REPORT / "result.json").read_text(encoding="utf-8"))
    qa = json.loads((REPORT / "independent-qa.json").read_text(encoding="utf-8"))
    b = json.loads((REPORT / "provenance-audit.json").read_text(encoding="utf-8"))
    if (
        result["status"] != "TERMINAL_MODELS_FROZEN_OFFICIAL_UNREAD"
        or qa["status"] != "PASS"
        or b["eligible_exact_zero_fit_combination"]
        or result["runner_sha256"] != a.old.sha(a.__file__)
        or qa["result_sha256"] != a.old.sha(REPORT / "result.json")
    ):
        raise ValueError("A/B predecessor integrity failure")
    directory = ARTIFACT / "p1_c_fixed_decoder"
    directory.mkdir(exist_ok=False)
    started = time.monotonic()
    receipt = {
        "status": "RUNNING",
        "runner_sha256": a.old.sha(__file__),
        "source_a_result_sha256": a.old.sha(REPORT / "result.json"),
        "source_b_result_sha256": a.old.sha(REPORT / "provenance-audit.json"),
        "decoder_source_sha256": a.old.sha(dec.__file__),
        "lambda": 1.0,
        "laplace": 1.0,
        "probability_clip": 1e-6,
        "new_backbone_fits": 0,
        "transition_estimates": 0,
        "official_rows": 0,
        "hidden_rows": 0,
        "csv_written": 0,
        "upload": 0,
    }
    a.old.write_json(directory / "ATTEMPT_LOCK.json", receipt)
    train_path = Path(os.environ["P1_DATA_DIR"]) / "train.csv"
    if a.old.sha(train_path) != result["train_sha256"]:
        raise ValueError("distributed train identity mismatch")
    train = pd.read_csv(train_path, usecols=a.old.RAW + ["label", "anomaly_type"])
    train.sort_values(["station", "layer", "time"], kind="stable", inplace=True)
    train.reset_index(drop=True, inplace=True)
    times = pd.to_datetime(train.time, utc=True)
    frozen = result["final_recipe"]["feature_contract"]
    parts, folds = [], []
    for config, fold in zip(frozen["folds"], result["folds"], strict=True):
        name = config["name"]
        part = pd.read_parquet(ARTIFACT / "03_training" / f"{name}_oof.parquet")
        evaluation = train.loc[
            (times >= pd.Timestamp(config["start"])) & (times < pd.Timestamp(config["end"]))
        ].reset_index(drop=True)
        if not evaluation[a.old.KEYS].equals(part[a.old.KEYS]):
            raise ValueError("fixed decoder key mismatch")
        training = a.old.train_slice(train, pd.Timestamp(config["start"]) - pd.Timedelta(days=21))
        probabilities = {m: part["candidate_" + m + "_probability"].to_numpy() for m in a.MODELS}
        rules = a.old.rule_masks(evaluation, a.old.stats_fit(training))
        spec = fold["selections"]["candidate"]
        reference, unary, hard = dec.control_components(
            evaluation, probabilities, rules, frozen, spec["policy"], spec["calibrations"]
        )
        if not np.array_equal(reference, part.candidate):
            raise ValueError("OFF decoder identity mismatch")
        transition = dec.transition_fit(training, laplace=1.0)
        part["candidate_decoder_on"] = dec.decode_viterbi(
            evaluation, unary, transition, hard, strength=1.0
        )
        receipt["transition_estimates"] += 1
        if np.any(hard & part.candidate_decoder_on.eq(0).to_numpy()):
            raise ValueError("fixed decoder hard mask mismatch")
        part.to_parquet(directory / f"{name}_oof.parquet", index=False)
        fold_metrics = {
            k: a.summarize(part, k, "control")
            for k in ("control", "candidate", "candidate_decoder_on")
        }
        folds.append(
            {"fold": name, "metrics": fold_metrics, "transition": transition, "hard_removed": 0}
        )
        parts.append(part)
    pooled = pd.concat(parts, ignore_index=True)
    pooled.to_parquet(directory / "oof.parquet", index=False)
    names = ("control", "candidate", "candidate_decoder_on")
    receipt["pooled"] = {k: a.summarize(pooled, k, "control") for k in names}
    receipt["folds"] = folds
    receipt["chosen_development_policy"] = max(names, key=lambda k: receipt["pooled"][k]["f1"])
    receipt["selection_rule"] = (
        "same-key pooled F1; exact tie retains earlier simpler policy; retrospective development only"
    )
    receipt["delta_f1_vs_control"] = (
        receipt["pooled"][receipt["chosen_development_policy"]]["f1"]
        - receipt["pooled"]["control"]["f1"]
    )
    receipt["intervals_vs_control"] = {
        k: paired_block_interval(pooled, k, "control") for k in names[1:]
    }
    months = pd.to_datetime(pooled.time, utc=True).dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m")
    receipt["month_metrics"] = {
        m: {k: a.old.metric(rows.label, rows[k]) for k in names}
        for m, rows in pooled.groupby(months)
    }
    full_transition = dec.transition_fit(train, laplace=1.0)
    receipt["transition_estimates"] += 1
    a.old.write_json(directory / "full_transition.json", full_transition)
    receipt["full_transition_sha256"] = a.old.sha(directory / "full_transition.json")
    receipt["oof_sha256"] = a.old.sha(directory / "oof.parquet")
    receipt["status"] = "FIXED_ON_OFF_DEVELOPMENT_COMPLETE_OFFICIAL_UNREAD"
    receipt["runtime_seconds"] = time.monotonic() - started
    receipt["expected_official_score"] = None
    a.old.write_json(REPORT / "decoder-result.json", receipt)
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in (
                    "status",
                    "chosen_development_policy",
                    "delta_f1_vs_control",
                    "runtime_seconds",
                )
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--provenance", action="store_true")
    group.add_argument("--decoder", action="store_true")
    args = parser.parse_args()
    provenance() if args.provenance else decoder()
