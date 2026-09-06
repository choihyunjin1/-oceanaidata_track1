"""Root audit of exact candidate files and aggregate historical metrics; no fits."""

import argparse
import hashlib
import json
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/parallel_core_training_20260906_v1"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_audit(problem, value):
    with (REPORT / f"{problem}-root-independent-qa.json").open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(value, ensure_ascii=False))


def check_equal_metric(actual, expected):
    for name, value in actual.items():
        if name in expected and not np.isclose(value, expected[name], rtol=1e-12, atol=1e-12):
            raise ValueError("Independent metric differs: " + name)


def binary_metric(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    if y.shape != prediction.shape or not np.isin(y, [0, 1]).all() or not np.isin(prediction, [0, 1]).all():
        raise ValueError("same-row binary inputs required")
    tp = int(((y == 1) & (prediction == 1)).sum())
    fp = int(((y == 0) & (prediction == 1)).sum())
    fn = int(((y == 1) & (prediction == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def p1_audit():
    out = ROOT / "artifacts/p1_core_tuning_20260906_v1"
    result = read(out / "terminal_result.json")
    replay = read(out / "fresh-replay-qa.json")
    if result["status"] != "COMPLETE" or replay["status"] != "PASS":
        raise ValueError("terminal and replay PASS required")
    if replay["terminal_result_sha256"] != digest(out / "terminal_result.json"):
        raise ValueError("replay terminal lineage differs")
    if result["worker_pid"] == replay["pid"]:
        raise ValueError("new replay process required")
    if result["completed_fits"] != result["attempted_fits"] or not 39 <= result["completed_fits"] <= 48:
        raise ValueError("actual fit count differs")
    if any(result[name] != 0 for name in ["official_rows", "hidden_rows", "csv_written", "upload"]):
        raise ValueError("scope zero counter differs")
    if digest(out / "oof.parquet") != result["oof_sha256"]:
        raise ValueError("OOF hash differs")
    frame = pd.read_parquet(out / "oof.parquet")
    if len(frame) != 421032 or frame.duplicated(["station", "year", "layer", "time"]).any():
        raise ValueError("OOF population or keys differ")
    if frame.groupby("fold").size().to_dict() != {"2025_q2": 133170, "2025_q3": 176738, "2025_q4": 111124}:
        raise ValueError("fixed fold population differs")
    metrics = {}
    for name, mask in {"primary_all_Q2_Q3_Q4": np.ones(len(frame), dtype=bool),
                       "secondary_old_Q3_Q4": frame.fold.isin(["2025_q3", "2025_q4"])}.items():
        part = frame.loc[mask]
        actual = {arm: binary_metric(part.label, part[arm]) for arm in ["control", "candidate"]}
        for arm in actual:
            check_equal_metric(actual[arm], result["evaluation"][name]["metrics"][arm])
        actual["delta_f1"] = actual["candidate"]["f1"] - actual["control"]["f1"]
        metrics[name] = actual
    folds = {str(fold): {arm: binary_metric(part.label, part[arm]) for arm in ["control", "candidate"]}
             for fold, part in frame.groupby("fold")}
    for fit in result["fit_receipts"]:
        if digest(out / fit["path"]) != fit["model_sha256"]:
            raise ValueError("model hash differs: " + fit["path"])
    write_audit("p1", {"status": "PASS", "metrics_recomputed": metrics, "folds": folds,
                      "completed_fits": result["completed_fits"], "runtime_seconds": result["runtime_seconds"],
                      "terminal_sha256": digest(out / "terminal_result.json"),
                      "replay_sha256": digest(out / "fresh-replay-qa.json"),
                      "official_rows_read_by_root": 0, "new_fits_by_root": 0})


def p2_audit():
    out = ROOT / "artifacts/p2_c3_training_comparison_20260906_v1"
    result = read(out / "terminal_result.json")
    replay = read(out / "fresh-replay.json")
    if result["status"] != "COMPLETE_24_SCREEN_PLUS_4_PRIMARY_SEED_FITS" or replay["status"] != "PASS":
        raise ValueError("terminal and replay PASS required")
    if result["completed_fits"] != result["attempted_fits"] or result["completed_fits"] != 28:
        raise ValueError("28 actual fits required")
    if replay["terminal_result_sha256"] != digest(out / "terminal_result.json") or replay["pid"] == result["worker_pid"]:
        raise ValueError("replay terminal/process lineage differs")
    if digest(out / "evaluation.npz") != result["evaluation_sha256"]:
        raise ValueError("evaluation hash differs")
    if any(result[name] != 0 for name in ["official_rows", "hidden_rows", "csv_written", "upload"]):
        raise ValueError("scope zero counter differs")
    metrics = {}

    def measure(y, p):
        if y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
            raise ValueError("same-row finite predictions required")
        sse = float(np.square(y - p).sum())
        return {"n": len(y), "sse": sse, "rmse_C": float(np.sqrt(sse / len(y)))}

    with np.load(out / "evaluation.npz", allow_pickle=False) as values:
        y, fold = values["truth"], values["fold"]
        if len(y) != 166268 or len(np.unique(values["key"])) != len(y):
            raise ValueError("full population differs")
        for arm in ["C60", "L120", "D60"]:
            for scope, mask in [("all8_pooled", np.ones(len(y), bool)), ("B3_primary", fold == "B3")]:
                actual = measure(y[mask], values["natural_" + arm][mask])
                check_equal_metric(actual, result["screen"]["natural"][arm][scope])
                metrics[arm + "_" + scope] = actual
        for arm in ["C60", result["selected_challenger"]]:
            for suffix in ["all3", "heldback2"]:
                actual = measure(y[fold == "B3"], values[f"B3_{arm}_natural_{suffix}"])
                check_equal_metric(actual, result["confirmation"][arm]["natural_" + suffix])
                metrics[arm + "_B3_" + suffix] = actual
    for fit in result["fits"]:
        if digest(out / fit["model_file"]) != fit["model_sha256"]:
            raise ValueError("model hash differs")
    for name in {fit["fold"] for fit in result["fits"]}:
        matched = [fit for fit in result["fits"] if fit["fold"] == name]
        for field in ["train_keys_sha256", "train_truth_sha256", "training_arrays_sha256"]:
            if len({json.dumps(fit[field], sort_keys=True) for fit in matched}) != 1:
                raise ValueError("same-fold feature/target/weight mismatch: " + name + ":" + field)
    write_audit("p2", {"status": "PASS", "metrics_recomputed": metrics,
                      "completed_fits": 28, "runtime_seconds": result["runtime_seconds"],
                      "terminal_sha256": digest(out / "terminal_result.json"),
                      "replay_sha256": digest(out / "fresh-replay.json"),
                      "official_rows_read_by_root": 0, "new_fits_by_root": 0,
                      "same_fold_training_keys_truth_arrays_all_arms_identical": True,
                      "caveat": "full8 firstseed and B3 seed sensitivity, not full8 three-seed independent holdout"})


def p3_audit():
    out = ROOT / "artifacts/p3_numeric_candidate_20260906_v1"
    logs = out / "04_logs"
    answer = logs / "answer-qa.json"
    receipt = read(answer)
    replay = read(logs / "answer-replay-qa.json")
    training = read(logs / "training-result.json")
    internal = read(logs / "numeric-independent-qa.json")
    history_path = ROOT / "reports/p3_numeric_lead_forward_gpu_20260906_v2/result.json"
    history = read(history_path)
    paired_path = ROOT / "artifacts/p3_numeric_lead_forward_gpu_20260906_v2/paired_oof.parquet"
    checks = {}

    def check(name, valid):
        checks[name] = bool(valid)
        if not valid:
            raise ValueError("Root QA failed: " + name)

    check("historical_result_pin", digest(history_path) == receipt["experiment_result_sha256"])
    check("historical_paired_pin", digest(paired_path) == history["artifacts"]["paired_oof.parquet"])
    paired = pd.read_parquet(paired_path)
    check("historical_count", len(paired) == 103602)
    check("historical_keys_unique", not paired.duplicated(["anchor_id", "station", "lead_h"]).any())
    check("historical_finite", np.isfinite(paired[["target_hs", "control", "candidate"]]).all().all())
    metrics = {}
    for arm in ("control", "candidate"):
        errors = paired[arm].to_numpy(float) - paired.target_hs.to_numpy(float)
        sse = float(np.square(errors).sum())
        rmse = float(np.sqrt(sse / len(errors)))
        check(arm + "_sse", np.isclose(sse, history["comparison"][arm + "_sse_m2"], rtol=1e-12))
        check(arm + "_rmse", abs(rmse - history["comparison"][arm]) < 1e-12)
        metrics[arm] = {"n": len(errors), "sse_m2": sse, "rmse_m": rmse}
    metrics["delta_rmse_m"] = metrics["candidate"]["rmse_m"] - metrics["control"]["rmse_m"]

    csv = out / "05_answer/submission.csv"
    frame = pd.read_csv(csv)
    keys = ["case_id", "station", "lead_h"]
    check("answer_schema", list(frame) == keys + ["hs_pred"])
    check("answer_count", len(frame) == 1200 and frame.case_id.nunique() == 200)
    check("answer_keys", not frame[keys].isna().any().any() and not frame.duplicated(keys).any())
    # Distributed P3 README: +3/+6/+9/+12/+18/+24 hours.
    check("answer_leads", set(frame.lead_h) == {3, 6, 9, 12, 18, 24})
    check("answer_six_per_case", frame.groupby("case_id").size().eq(6).all())
    check("answer_finite_range", np.isfinite(frame.hs_pred).all() and frame.hs_pred.between(0, 30).all())
    check("answer_hash", digest(csv) == receipt["sha256"] == replay["sha256"])
    index_path = Path("C:/Users/cedis/Downloads/p3/데이터셋_P3/P3_wave_forecast/test_index.csv")
    check("public_index_hash", digest(index_path) == receipt["public_input_sha256"]["test_index.csv"])
    index = pd.read_csv(index_path, usecols=keys)
    check("answer_exact_index_order", frame[keys].equals(index[keys]))
    check("fresh_pid", len({training["pid"], receipt["pid"], replay["pid"]}) == 3)
    check("replay_pass", replay["status"] == "EXACT_ANSWER_REPLAY_PASS")
    check("training_pin", digest(logs / "training-result.json") == receipt["source_training_result_sha256"])
    check("internal_qa", internal["status"] == "PASS" and len(internal["checks"]) == 24
          and all(value is True for value in internal["checks"].values()))
    check("fits", training["new_backbone_fits"] == 2 and training["new_router_fits"] == 1)
    for name, expected in training["model_sha256"].items():
        check("model_" + name, digest(out / "03_model" / name) == expected)
    check("zero_hidden_upload", all(row["hidden_rows"] == 0 and row["uploads"] == 0
                                   for row in (training, receipt, replay)))
    check("official_after_qa", internal["official_rows"] == training["official_rows"] == 0)
    result = {"status": "PASS_LOCAL_CANDIDATE_NOT_UPLOADED", "checks": checks,
              "metrics_recomputed": metrics, "new_fits": 3,
              "fit_seconds": training["seconds"], "answer_rows": len(frame),
              "answer_path": str(csv), "answer_sha256": digest(csv),
              "official_score": None, "hidden_rows_read_by_root": 0,
              "public_index_rows_read_by_root": len(index), "new_fits_by_root": 0,
              "whole_cold_retraining_twice": False,
              "audit_correction": "initial root check mistakenly expected lead1 instead of lead9; corrected from distributed README before PASS; candidate unchanged",
              "limits": ["historical retrospective metrics, not fresh official score",
                         "saved-model exact replay, not two scratch trainings",
                         "browser connection unavailable; no upload claimed"]}
    with (REPORT / "p3-root-independent-qa.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False))


def p2_answer_audit():
    out = ROOT / "artifacts/p2_c3_training_comparison_full_20260906_v1"
    logs = out / "04_logs"
    qa = read(logs / "independent-qa.json")
    train = read(logs / "training-result.json")
    replay = read(logs / "replay-result.json")
    train_qa = read(logs / "training-independent-qa.json")
    inference = read(logs / "inference-result.json")
    terminal = read(logs / "terminal_result.json")
    if qa["status"] != "PASS" or not all(value is True for value in qa["checks"].values()):
        raise ValueError("all final QA checks required")
    if terminal["qa_sha256"] != digest(logs / "independent-qa.json"):
        raise ValueError("terminal QA lineage differs")
    for name in ["training", "inference", "replay"]:
        if qa[name + "_sha256"] != digest(logs / (name + "-result.json")):
            raise ValueError("stage lineage differs")
    if len({train["pid"], train_qa["pid"], inference["pid"], replay["pid"]}) != 4:
        raise ValueError("fresh stage PIDs required")
    path = out / qa["answer"]
    if digest(path) != qa["answer_sha256"] or digest(path) != replay["answer_sha256"]:
        raise ValueError("answer SHA/replay differs")
    frame = pd.read_csv(path)
    columns = ["station", "layer", "time"]
    if list(frame) != columns + ["temp"] or len(frame) != 26061 or not np.isfinite(frame.temp).all():
        raise ValueError("answer schema/count/finite differs")

    def keys(data):
        values = data[columns].copy()
        if values.isna().any().any():
            raise ValueError("null key")
        values["time"] = pd.to_datetime(values.time, utc=True)
        return pd.MultiIndex.from_frame(values)

    source = Path("C:/Users/cedis/Downloads/p2/데이터셋_P2/P2_profile_restore")
    index = pd.read_csv(source / "test_index.csv", usecols=columns)
    sample = pd.read_csv(source / "sample_submission.csv", usecols=columns)
    if not keys(frame).is_unique or not keys(frame).equals(keys(sample)):
        raise ValueError("answer uniqueness/sample key order differs")
    if len(index) != len(frame) or set(keys(frame)) != set(keys(index)):
        raise ValueError("index population differs")
    for name in ["index", "sample"]:
        filename = "test_index.csv" if name == "index" else "sample_submission.csv"
        if digest(source / filename) != qa["official_" + name + "_sha256"]:
            raise ValueError("public key-source file changed")
    if train["new_full_fits"] != 3 or not train["empty_model_directory_before_training"]:
        raise ValueError("fresh full3 required")
    if train["official_access_rows"] != 0 or train["csv_written"] != 0:
        raise ValueError("official use before training QA")
    for fit in train["fits"]:
        if digest(out / "03_model" / fit["file"]) != fit["sha256"] or fit["epochs"] != 120:
            raise ValueError("frozen full model/recipe differs")
    if set(frame.layer) != {2, 3, 4} or b"\r\n" in path.read_bytes():
        raise ValueError("layer/LF format differs")
    write_audit("p2-answer", {"status": "PASS_LOCAL_CANDIDATE_NOT_UPLOADED",
              "answer_path": str(path), "rows": len(frame), "sha256": digest(path),
              "new_full_fits": 3, "historical_plus_full_fits": 31,
              "full_cycle_seconds": terminal["runtime_seconds"],
              "candidate_qa_sha256": digest(logs / "independent-qa.json"),
              "sample_value_rows_read_by_root": 0, "public_key_rows_read_by_root": len(index) + len(sample),
              "official_score": None, "uploads_by_root": 0,
              "whole_cold_twice_claim": False})


def p1_bridge_audit():
    out = ROOT / "artifacts/p1_core_tuning_ms_bridge_20260906_v1"
    result = read(out / "result.json")
    tree_out = ROOT / "artifacts/p1_core_tuning_20260906_v1"
    proposal_path = ROOT / "artifacts/p1_champion_reconstruction_20260906_v1_historical_proposals/proposals.parquet"
    if digest(out / "paired.parquet") != result["paired_sha256"]:
        raise ValueError("bridge paired hash differs")
    if digest(tree_out / "oof.parquet") != result["input_sha256"]["tree_OOF"]:
        raise ValueError("new tree OOF pin differs")
    if digest(proposal_path) != result["input_sha256"]["fixed_MS_proposal"]:
        raise ValueError("fixed proposal pin differs")
    keys = ["station", "year", "layer", "time"]
    frame = pd.read_parquet(out / "paired.parquet")
    tree = pd.read_parquet(tree_out / "oof.parquet")
    proposal = pd.read_parquet(proposal_path)

    def canonical(data):
        data = data.copy()
        data["time"] = pd.to_datetime(data.time, utc=True)
        for column in ["year", "layer"]:
            data[column] = data[column].astype(int)
        return data.set_index(keys)

    tree = canonical(tree.loc[tree.fold.isin(["2025_q3", "2025_q4"])])
    proposal = canonical(proposal)
    joined = canonical(frame)
    if len(frame) != 287862 or not joined.index.is_unique or set(joined.index) != set(tree.index):
        raise ValueError("complete bridge/tree key set required")
    if set(joined.index) != set(proposal.index):
        raise ValueError("complete fixed MS keys required")
    ordered_tree, ordered_proposal = tree.loc[joined.index], proposal.loc[joined.index]
    for column in ["fold", "label", "control", "candidate"]:
        if not np.array_equal(joined[column], ordered_tree[column]):
            raise ValueError("tree bits/labels/fold owner differs")
    if not np.array_equal(joined.proposal, ordered_proposal.proposal):
        raise ValueError("fixed MS proposal changed")
    metrics = {}
    for arm in ["control", "candidate"]:
        bit = joined[arm].to_numpy(dtype=np.int8) | joined.proposal.to_numpy(dtype=np.int8)
        if not np.array_equal(bit, joined[arm + "_plus_ms"]):
            raise ValueError("same proposal union differs")
        actual = binary_metric(joined.label, bit)
        check_equal_metric(actual, result["primary_Q3_Q4"]["metrics"][arm + "_plus_ms"])
        metrics[arm + "_plus_ms"] = actual
    metrics["delta_f1"] = metrics["candidate_plus_ms"]["f1"] - metrics["control_plus_ms"]["f1"]
    write_audit("p1-ms-bridge", {"status": "PASS", "rows": len(frame), "metrics_recomputed": metrics,
              "all_tree_and_proposal_keys_preserved": True, "original_fold_owner_preserved": True,
              "result_sha256": digest(out / "result.json"), "new_fits": 0, "official_rows_read": 0,
              "audit_correction": "raw timestamp representations differed; UTC canonical keys required, with no row deletion or prediction edits",
              "decision": "keep existing composed candidate; new full13 not executed because mean composition did not improve"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", choices=["p1", "p2", "p3", "p2-answer", "p1-ms-bridge"])
    args = parser.parse_args()
    {"p1": p1_audit, "p2": p2_audit, "p3": p3_audit, "p2-answer": p2_answer_audit,
     "p1-ms-bridge": p1_bridge_audit}[args.problem]()
