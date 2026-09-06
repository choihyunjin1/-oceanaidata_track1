"""Independent zero-fit aggregate/bootstrap verification of a terminal union result."""
# ruff: noqa: E402
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import composition
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

ARMS = ("O", "B", "union", "router")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def independent_metric(part, column):
    tn, fp, fn, tp = confusion_matrix(part.label, part[column], labels=[0, 1]).ravel()
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "rows": len(part), "f1": float(f1_score(part.label, part[column], zero_division=0))}


def independent_bootstrap(part, arm, cfg):
    """Independent sklearn block counts and replicate-by-replicate pooled F1."""
    bs = cfg["bootstrap"]
    rng = np.random.default_rng(bs["seed"])
    plans, clusters = [], {}
    for fold, f in part.groupby("fold", sort=True):
        vectors = []
        for _, block in f.groupby("block", sort=True):
            pair = []
            for name in (arm, arm + "_OR_e150"):
                cm = confusion_matrix(block.label, block[name], labels=[0, 1])
                pair.append([cm[1, 1], cm[0, 1], cm[1, 0]])
            vectors.append(pair)
        clusters[fold] = len(vectors)
        if len(vectors) < bs["minimum_clusters_per_fold"]:
            return {"status": "NOT_ESTIMABLE_TOO_FEW_BLOCKS", "clusters": clusters}
        sample = rng.integers(0, len(vectors), size=(bs["resamples"], len(vectors)))
        plans.append((np.asarray(vectors, dtype=np.int64), sample))
    differences = []
    for iteration in range(bs["resamples"]):
        pooled = np.zeros((2, 3), dtype=np.int64)
        for vectors, samples in plans:
            pooled += vectors[samples[iteration]].sum(axis=0)
        scores = []
        for tp, fp, fn in pooled:
            denominator = 2 * int(tp) + int(fp) + int(fn)
            scores.append(2 * int(tp) / denominator if denominator else 0.)
        differences.append(scores[1] - scores[0])
    differences = np.asarray(differences)
    return {"status": "DESCRIPTIVE_PAIRED_BLOCK_BOOTSTRAP", "clusters": clusters,
            "resamples": bs["resamples"], "seed": bs["seed"],
            "mean_delta_f1": float(np.mean(differences)),
            "ci90": np.quantile(differences, bs["ci_quantiles"]).tolist(),
            "fraction_delta_positive": float(np.count_nonzero(differences > 0) / len(differences)),
            "fraction_delta_zero": float(np.count_nonzero(differences == 0) / len(differences)),
            "not_fresh_test_or_posterior_probability": True}


def verify_statistics(frame, result, cfg):
    checked = []

    def check(name, value):
        if not value:
            raise AssertionError(name)
        checked.append(name)

    stamp = pd.to_datetime(frame.time, utc=True).dt.tz_convert("Asia/Seoul")
    days = (stamp.dt.date - pd.Timestamp(cfg["bootstrap"]["anchor_kst"]).date()).apply(lambda v: v.days)
    block = days.to_numpy() // cfg["bootstrap"]["block_days"]
    check("independently_rederived_KST_7day_blocks", np.array_equal(block, frame.block))
    for arm in ARMS:
        check(arm + "/Boolean_OR_exact", np.array_equal(
            np.logical_or(frame[arm], frame.proposal).astype(np.int8), frame[arm + "_OR_e150"]))
    scopes = {"primary_Q3_Q4": frame}
    scopes.update({key: part for key, part in frame.groupby("fold", sort=True)})
    check("scope_names_exact", set(scopes) == set(result["scopes"]))
    for name, part in scopes.items():
        for arm in ARMS:
            reported = result["scopes"][name][arm]
            for side, column in (("control", arm), ("candidate", arm + "_OR_e150")):
                actual = independent_metric(part, column)
                check(name + "/" + arm + "/" + side, all(
                    abs(actual[k] - reported[side][k]) <= 1e-12 for k in actual))
            expected_delta = independent_metric(part, arm + "_OR_e150")["f1"] - independent_metric(part, arm)["f1"]
            check(name + "/" + arm + "/delta", abs(expected_delta - reported["delta_f1"]) <= 1e-12)
            calculated = independent_bootstrap(part, arm, cfg)
            observed = reported["bootstrap"]
            check(name + "/" + arm + "/bootstrap_shape", set(calculated) == set(observed))
            for key, value in calculated.items():
                equal = np.allclose(value, observed[key], rtol=0, atol=1e-12) if key in (
                    "mean_delta_f1", "ci90", "fraction_delta_positive", "fraction_delta_zero") else value == observed[key]
                check(name + "/" + arm + "/bootstrap/" + key, equal)
    expected_slices = []
    for by in (("fold",), ("station",), ("station", "layer"), ("fold", "block")):
        for key, part in frame.groupby(list(by), sort=True):
            values = key if isinstance(key, tuple) else (key,)
            keylist = [v.item() if isinstance(v, np.generic) else v for v in values]
            matched = [s for s in result["slices"] if s["by"] == list(by) and s["key"] == keylist]
            check("unique_slice/" + str(keylist), len(matched) == 1 and matched[0]["rows"] == len(part))
            current = matched[0]
            expected_slices.append(current)
            for arm in ARMS:
                before, after = independent_metric(part, arm), independent_metric(part, arm + "_OR_e150")
                reported = current["arms"][arm]
                check("slice_counts/" + arm + "/" + str(keylist), all(
                    abs(value - reported[side][metric]) <= 1e-12
                    for side, actual in (("control", before), ("candidate", after)) for metric, value in actual.items()))
                check("slice_delta/" + arm + "/" + str(keylist), abs(after["f1"] - before["f1"] - reported["delta_f1"]) <= 1e-12)
                check("slice_changes/" + arm + "/" + str(keylist), reported["removed_rows"] == 0 and
                    reported["added_rows"] == int((part[arm].eq(0) & part[arm + "_OR_e150"].eq(1)).sum()))
    check("no_extra_or_missing_slices", len(expected_slices) == len(result["slices"]))
    for arm in ARMS:
        for by in (("fold",), ("station",), ("station", "layer"), ("fold", "block")):
            group = [s for s in expected_slices if s["by"] == list(by)]
            worst = min(group, key=lambda s: s["arms"][arm]["delta_f1"])
            expected = {"slice_count": len(group),
                "negative_slice_count": sum(s["arms"][arm]["delta_f1"] < 0 for s in group),
                "worst_delta_f1": worst["arms"][arm]["delta_f1"],
                "worst_key": worst["key"], "worst_rows": worst["rows"]}
            check("risk/" + arm + "/" + "/".join(by), expected == result["risk"][arm]["/".join(by)])
    return checked


def main():
    report_dir = ROOT / "reports/p1_champion_reconstruction_20260906_v1"
    cfg = read(report_dir / "union-contract.json")
    output = ROOT / "artifacts" / cfg["id"]
    target = output / "independent-bootstrap-qa.json"
    if target.exists():
        raise FileExistsError("preserve existing independent QA")
    result = read(output / "result.json")
    if result["status"] != "COMPLETE_RETROSPECTIVE_EVALUATION":
        raise ValueError("complete exact-key performance result required; a mismatch is not a performance failure")
    seal = read(report_dir / "union-seal.json")
    if any(sha(ROOT / p) != expected for p, expected in seal["source_pins"].items()):
        raise ValueError("union sealed code/contract changed")
    if result["union_seal_sha256"] != sha(report_dir / "union-seal.json"):
        raise ValueError("result belongs to another union seal")
    if result["paired_artifact_sha256"] != sha(output / "paired-evaluation.parquet"):
        raise ValueError("paired artifact SHA mismatch")
    if not result["coverage"]["same_keys_and_folds"] or any(result[k] != 0 for k in (
            "new_fits", "official_rows", "hidden_rows", "csv_outputs", "upload")):
        raise ValueError("exact coverage or scope counters failed")
    frame = pd.read_parquet(output / "paired-evaluation.parquet")
    if result["rows"] != len(frame) or result["ordered_keys_sha256"] != composition.ordered_key_sha256(frame):
        raise ValueError("paired schema/population/key SHA mismatch")
    checks = verify_statistics(frame, result, cfg)
    payload = {"status": "PASS", "pid": os.getpid(), "checks": len(checks), "check_names": checks,
               "method": "independent sklearn confusion counts plus replicate-loop fold-stratified block bootstrap",
               "result_sha256": sha(output / "result.json"), "paired_sha256": result["paired_artifact_sha256"],
               "qa_source_sha256": sha(__file__), "new_fits": 0, "official_rows": 0, "hidden_rows": 0,
               "rtol": 0, "atol": 1e-12, "confidence": "share_with_retrospective_and_block_dependence_caveats"}
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "PASS", "checks": len(checks), "fits": 0}))


if __name__ == "__main__":
    main()
