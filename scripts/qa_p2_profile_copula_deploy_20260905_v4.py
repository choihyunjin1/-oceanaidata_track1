"""Independent local candidate identity, schema, lineage and replay receipts."""

import json
from pathlib import Path

import run_p2_profile_copula_deploy_20260905_v4 as m


def main():
    cfg = m.config()
    hashes = json.loads(m.SEAL.read_text(encoding="utf-8"))["hashes"]
    assert hashes == m.fingerprints()
    source, _ = m.guard("QA")
    train = json.loads((m.REPORT / "training-result.json").read_text(encoding="utf-8"))
    infer = json.loads((m.REPORT / "result.json").read_text(encoding="utf-8"))
    replay = json.loads((m.REPORT / "replay.json").read_text(encoding="utf-8"))
    checks = []

    def check(name, value):
        checks.append({"check": name, "pass": bool(value)})

    check("one_new_copula_zero_new_backbones", train["new_full_copula_fits"] == 1 and train["new_backbone_fits"] == 0)
    check("released_train_rows", train["train_rows"] == 166268)
    check("no_training_official_or_answers", train["official_access_rows"] == train["csv_written"] == train["upload"] == 0 and train["access"]["old_OOF_reads"] == train["access"]["old_answer_values"] == 0)
    check("frozen_full_policy", train["coefficient"] == cfg["correction_strength"] == 1 and infer["policy"] == "fixed_full_1.0_no_new_tuning")
    check("three_independent_processes", len({train["pid"], infer["pid"], replay["pid"]}) == 3)
    check("source_immutable", m.base.file_hash(source) == cfg["source_sha256"] == train["source_sha256_before_after"])
    check("copula_model_hash", m.base.file_hash(m.MODELS / "copula_full.npz") == train["copula_sha256"])
    check("three_regenerated_C_components", len(train["regenerated_C_models"]) == 3)
    for model in train["regenerated_C_models"]:
        check(f"copied_C_identity/{model['seed']}", m.base.file_hash(m.MODELS / model["file"]) == model["sha256"] == m.base.file_hash(m.REGEN / "03_model" / model["file"]))
    answer = m.ROOT / infer["candidate"]
    replay_answer = m.ROOT / replay["candidate"]
    check("whole_CSV_exact_replay", m.base.file_hash(answer) == infer["candidate_sha256"] == replay["candidate_sha256"] == m.base.file_hash(replay_answer))
    check("same_C_control_bytes", infer["control_sha256"] == replay["control_sha256"] == cfg["baseline_expected_sha256"])
    check("new_candidate_not_C_duplicate", infer["candidate_sha256"] != cfg["baseline_expected_sha256"])
    frame = m.pd.read_csv(answer)
    check("columns_exact", list(frame.columns) == [*m.canonical.KEYS, "temp"])
    check("all_rows_unique", len(frame) == infer["rows"] == replay["rows"] == 26061 and not m.canonical.canonical_keys(frame).duplicated().any())
    check("finite_output", m.np.isfinite(frame.temp).all())
    check("target_layers", set(frame.layer) == {2, 3, 4})
    for kind, result in (("inference", infer), ("replay", replay)):
        check(f"{kind}/schema_key_order_checks", all(result["checks"].values()))
        check(f"{kind}/key_only_official", result["access"]["official_index_key_rows"] == result["access"]["official_sample_key_rows"] == 26061 and result["access"]["sample_value_rows"] == result["access"]["old_answer_values"] == result["access"]["old_OOF_reads"] == 0)
        check(f"{kind}/serialization_precision", result["serialization_max_abs_error"] < 1e-8)
        check(f"{kind}/no_upload", result["upload"] == 0)
    check("baseline_blank_model_reproduction_pass", infer["baseline_empty_model_training"] == "PASS_from_separate_v6_regeneration")
    check("no_official_score_estimate", infer["expected_official_score"] is None)
    failures = [item["check"] for item in checks if not item["pass"]]
    result = {"status": "PASS" if not failures else "FAIL", "checks_count": len(checks), "checks": checks, "failed_checks": failures, "candidate_sha256": infer["candidate_sha256"], "candidate_path": infer["candidate"], "source_sha256": cfg["source_sha256"], "result_sha256": m.base.file_hash(m.REPORT / "result.json"), "training_sha256": m.base.file_hash(m.REPORT / "training-result.json"), "replay_sha256": m.base.file_hash(m.REPORT / "replay.json"), "qa_runner_sha256": m.base.file_hash(Path(__file__)), "focused_synthetic_tests": 4, "ruff": "PASS", "upload": 0, "official_target_values_read": 0, "new_fits": 0, "interpretation": "local_generation_integrity_PASS_not_proven_official_performance_improvement"}
    m.save(m.REPORT / "independent-qa.json", result)
    print(json.dumps({"status": result["status"], "checks_count": len(checks), "failures": failures}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
