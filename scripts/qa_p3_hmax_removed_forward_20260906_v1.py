"""Independent saved-OOF arithmetic/chronology/hash QA; no fitting or official I/O."""

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from p3_forward_independent_statistics_20260906_v2 import independent_bootstrap
from run_p3_hmax_removed_forward_20260906_v1 import (
    CONFIG,
    KEYS,
    LEADS,
    OUT,
    REPORT,
    ROOT,
    access_guard,
    sha,
    verify_pins,
    write_json,
    zero_access,
)


def independent_metrics(truth, control, candidate):
    y, b, c = map(lambda array: np.asarray(array, float), (truth, control, candidate))
    if y.ndim != 1 or y.shape != b.shape or y.shape != c.shape or not len(y):
        raise ValueError("aligned nonempty arrays required")
    if not np.isfinite(np.r_[y, b, c]).all():
        raise ValueError("nonfinite evaluation")
    sse_b = math.fsum(float(item) ** 2 for item in b - y)
    sse_c = math.fsum(float(item) ** 2 for item in c - y)
    rmse_b, rmse_c = math.sqrt(sse_b / len(y)), math.sqrt(sse_c / len(y))
    return {
        "rows": len(y),
        "control_sse_m2": sse_b,
        "candidate_sse_m2": sse_c,
        "control_rmse_m": rmse_b,
        "candidate_rmse_m": rmse_c,
        "delta_rmse_m": rmse_c - rmse_b,
    }


def run():
    cfg = json.loads(CONFIG.read_text())
    source = Path(os.environ["P3_DATA_DIR"]).resolve()
    access_guard(source, cfg)
    verify_pins(cfg, source, sealed=True)
    result = json.loads((REPORT / "result.json").read_text())
    pre = json.loads((REPORT / "preflight.json").read_text())
    replay = json.loads((REPORT / "fresh-process-replay.json").read_text())
    evaluation = json.loads((ROOT / cfg["contract"]).read_text())
    recipe = json.loads((ROOT / cfg["recipe"]).read_text())
    anchors = pd.read_parquet(OUT / "anchors.parquet")
    paired = pd.read_parquet(OUT / "paired_oof.parquet")
    base = pd.read_parquet(OUT / "baseline_oof.parquet")
    candidate = pd.read_parquet(OUT / "candidate_oof.parquet")
    checks = []

    def check(name, ok):
        checks.append({"check": name, "pass": bool(ok)})

    def close(first, second, tolerance=1e-10):
        return np.isclose(first, second, rtol=0, atol=tolerance)

    for name, digest in result["artifacts"].items():
        check(f"artifact_hash/{name}", sha(OUT / name) == digest)
    check("preflight_link", sha(REPORT / "preflight.json") == result["preflight_sha256"])
    check("source_hash_link", result["source_sha256"] == cfg["source_files"])
    check(
        "complete_actual_fit_counts",
        len(result["fit_receipts"]) == result["fit_count"] == 10
        and result["router_fit_count"] == 4,
    )
    check(
        "actual_models_10_plus4",
        len(list((OUT / "models").rglob("*.cbm"))) == 10
        and len(list((OUT / "models").rglob("*.joblib"))) == 4,
    )
    check(
        "no_existing_prediction_or_model_input",
        all(
            not any(mark in name for mark in ("oof.parquet", "/models/", ".cbm", "submission.csv"))
            for name in cfg["inputs"]
        ),
    )
    check(
        "complete_key_population",
        len(paired) == len(base) == len(candidate) == 103602
        and paired.anchor_id.nunique() == 17267
        and not paired.duplicated(KEYS).any(),
    )
    check(
        "paired_key_order", paired[KEYS].equals(base[KEYS]) and paired[KEYS].equals(candidate[KEYS])
    )
    check(
        "paired_target_exact",
        np.array_equal(paired.target_hs, base.target_hs)
        and np.array_equal(paired.target_hs, candidate.target_hs),
    )
    check(
        "component_to_final_exact",
        np.array_equal(paired.control, base.final_prediction)
        and np.array_equal(paired.candidate, candidate.final_prediction),
    )
    check(
        "same_cycle_baseline_exact_hash",
        sha(ROOT / cfg["baseline"]["out"] / "baseline_oof.parquet")
        == result["baseline_pins"][cfg["baseline"]["out"] + "/baseline_oof.parquet"],
    )
    check("no_numeric_combination", result["numeric_lead_used"] is False)
    check(
        "candidate_feature_removal",
        len(result["candidate_features"]) == 527
        and all(not c.startswith("hmax_") for c in result["candidate_features"])
        and result["router_removed_features"] == ["hmax_current"],
    )
    check(
        "raw_hmax_perturbation_no_retained_effect",
        pre["hmax_raw_perturbation_retained_features_exact"],
    )
    check(
        "finite_range",
        np.isfinite(paired[["control", "candidate"]]).all().all()
        and paired[["control", "candidate"]].ge(0).all().all()
        and paired[["control", "candidate"]].le(30).all().all(),
    )
    check(
        "six_exact_leads",
        paired.groupby("anchor_id").lead_h.agg(tuple).map(lambda value: value == LEADS).all(),
    )
    wave = pd.read_csv(source / "train_wave.csv", usecols=["station", "time", "hs"])
    wave.time = pd.to_datetime(wave.time, utc=True)
    wave_lookup = wave.set_index(["station", "time"]).hs
    times = pd.to_datetime(paired.anchor_time, utc=True) + pd.to_timedelta(paired.lead_h, unit="h")
    target = wave_lookup.reindex(pd.MultiIndex.from_arrays([paired.station, times])).to_numpy()
    check(
        "all_evaluation_targets_reconstructed_direct_source",
        np.array_equal(target, paired.target_hs),
    )
    observed = independent_metrics(paired.target_hs, paired.control, paired.candidate)
    cluster_qa = independent_bootstrap(paired, evaluation["common"]["bootstrap"])
    for name in ("n_rows", "n_clusters", "p_improve", "resamples", "seed"):
        check("independent_bootstrap/" + name, cluster_qa[name] == result["comparison"][name])
    check(
        "independent_bootstrap/ci90",
        np.allclose(cluster_qa["ci90"], result["comparison"]["ci90"], atol=1e-12, rtol=0),
    )
    for ours, theirs in (
        ("control_sse_m2", "control_sse_m2"),
        ("candidate_sse_m2", "candidate_sse_m2"),
        ("control_rmse_m", "control"),
        ("candidate_rmse_m", "candidate"),
        ("delta_rmse_m", "delta_candidate_minus_control"),
    ):
        check(
            f"pooled_arithmetic/{ours}",
            close(observed[ours], result["comparison"][theirs], 1e-8 if "sse" in ours else 1e-12),
        )
    check(
        "decision_mean_only",
        (observed["delta_rmse_m"] < 0) == result["comparison"]["candidate_retained"],
    )
    check("no_probability_gate", result["comparison"]["probability_hard_gate"] is None)
    check(
        "onset_greedy_disabled",
        result["comparison"]["onset_diagnostic"]
        == result["comparison"]["greedy_diagnostic"]
        == "NOT_ENABLED",
    )
    for slice_row in result["comparison"]["slices"]:
        part = paired.loc[paired[slice_row["dimension"]].astype(str) == slice_row["key"]]
        metrics = independent_metrics(part.target_hs, part.control, part.candidate)
        check(
            f"slice/{slice_row['dimension']}/{slice_row['key']}",
            len(part) == slice_row["rows"]
            and close(metrics["delta_rmse_m"], slice_row["delta_candidate_minus_control_m"], 1e-12),
        )
    for receipt in result["fit_receipts"]:
        model_path = OUT / receipt["model_path"]
        model = CatBoostRegressor().load_model(model_path)
        multi = receipt["kind"] == "multi"
        expected = dict(recipe["model"]["multi" if multi else "single"])
        expected.pop("devices", None)
        expected.update(
            task_type="GPU" if multi else "CPU",
            thread_count=2,
            random_seed=receipt["seed"],
            verbose=False,
            allow_writing_files=False,
        )
        if multi:
            expected.update(devices="0", boosting_type="Plain")
        check(f"model_hash/{receipt['model_path']}", sha(model_path) == receipt["model_sha256"])
        check(
            f"recipe/{receipt['model_path']}",
            receipt["parameters"] == expected
            and receipt["iterations"] == model.tree_count_ == (1200 if multi else 700),
        )
        cats = [0, 1] if receipt["kind"] == "categorical_single" else [0]
        check(
            f"native_cat_indices/{receipt['model_path']}", model.get_cat_feature_indices() == cats
        )
        check(
            f"native_no_hmax/{receipt['model_path']}",
            all(not c.startswith("hmax_") for c in model.feature_names_),
        )
    folds = [fold for fold in evaluation["P3"]["folds"] if not fold["warmup"]]
    completed = []
    for index, fold in enumerate(folds):
        start, end = pd.Timestamp(fold["start"]), pd.Timestamp(fold["end"])
        in_fold = anchors.anchor_time.ge(start) & anchors.anchor_time.lt(end)
        validation = anchors.loc[in_fold]
        held = set(zip(validation.station, validation.episode_id, strict=True))
        safe = anchors.anchor_time.lt(start - pd.Timedelta(hours=78)) & np.array(
            [(row.station, row.episode_id) not in held for row in anchors.itertuples(index=False)]
        )
        check(
            f"base_fold_support/{fold['id']}",
            int(safe.sum()) == cfg["expected_train"][index]
            and len(validation) == cfg["expected_validation"][index],
        )
        for station, part in validation.groupby("station"):
            train = anchors.loc[safe & anchors.station.eq(station)]
            check(
                f"base_footprints/{fold['id']}/{station}",
                train.anchor_time.max() + pd.Timedelta(hours=24)
                < part.anchor_time.min() - pd.Timedelta(hours=48),
            )
        prior = paired.fold.isin(completed) & paired.anchor_id.isin(anchors.loc[safe, "anchor_id"])
        payload = paired.loc[prior, KEYS].to_json(orient="split", index=False)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        for arm in ("candidate",):
            item = result["router_receipts"][arm][index]
            check(
                f"meta_key_count/{arm}/{fold['id']}",
                item["prior_key_sha256"] == digest
                and item["past_fit_rows"] == int(prior.sum())
                and item["past_fit_anchors"] == paired.loc[prior, "anchor_id"].nunique(),
            )
            if index:
                check(
                    f"meta_model_hash/{arm}/{fold['id']}",
                    sha(OUT / item["model_path"]) == item["model_sha256"],
                )
                check(
                    f"meta_footprint/{arm}/{fold['id']}",
                    (
                        paired.loc[prior, "anchor_time"] + pd.Timedelta(hours=24)
                        < start - pd.Timedelta(hours=48)
                    ).all(),
                )
        check(
            f"meta_preflight_counts/{fold['id']}",
            pre["folds"][index]["meta_prior_safe_anchors"]
            == paired.loc[prior, "anchor_id"].nunique(),
        )
        completed.append(fold["id"])
    check("fresh_pid", replay["fresh_pid"] != result["pid"] == replay["training_pid"])
    check("replay_result_link", replay["result_sha256"] == sha(REPORT / "result.json"))
    check(
        "replay_exact10plus4",
        replay["status"] == "PASS"
        and replay["backbone_models"] == 10
        and replay["router_models"] == 4
        and replay["max_abs_prediction_error_m"] == 0,
    )
    check("under60min", result["execution_seconds"] <= cfg["budget"]["wall_seconds"])
    check(
        "no_full_regeneration_claim",
        result["baseline_regeneration_check"] == "NOT_RUN_FULL_MATERIALIZATION_OUT_OF_SCOPE",
    )
    check("access_zero", all(result[key] == replay[key] == 0 for key in zero_access()))
    failed = [item["check"] for item in checks if not item["pass"]]
    receipt = {
        "status": "PASS" if not failed else "FAIL",
        "checks_count": len(checks),
        "failed_checks": failed,
        "checks": checks,
        "independently_recomputed": observed,
        "result_sha256": sha(REPORT / "result.json"),
        "replay_sha256": sha(REPORT / "fresh-process-replay.json"),
        "qa_runner_sha256": sha(__file__),
        "baseline_regeneration_check": "NOT_RUN_FULL_MATERIALIZATION_OUT_OF_SCOPE",
        **zero_access(),
    }
    write_json(REPORT / "independent-qa.json", receipt)
    print(
        json.dumps({"status": receipt["status"], "checks": len(checks), "failed": failed}),
        flush=True,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
