"""Preserve the actually scored numeric full models separately from new cold models."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from build_final_release_20260907_v1 import archive, copy_pins, read, save, sha

EXPECTED = "ff42a6a08c76f0d58ed2f3a9ea31a08819ada5fa6e9007af942fe0b891937960"


def build(repo, verified_cold_saved, destination):
    scored = repo / "artifacts/p3_numeric_candidate_20260906_v1"
    report = read(repo / "reports/p3_numeric_candidate_20260906_v1/result.json")
    training = read(scored / "04_logs/training-result.json")
    qa = read(scored / "04_logs/numeric-independent-qa.json")
    replay = read(scored / "04_logs/answer-replay-qa.json")
    if (report["candidate"]["sha256"] != EXPECTED or replay["sha256"] != EXPECTED
            or sha(scored / "05_answer/submission.csv") != EXPECTED):
        raise ValueError("scored answer pin mismatch")
    if (training["status"] != "FULL_TRAINING_COMPLETE" or (training["new_backbone_fits"], training["new_router_fits"]) != (2, 1)
            or qa["status"] != "PASS" or qa["failed_checks"]
            or qa["training_result_sha256"] != sha(scored / "04_logs/training-result.json")
            or sha(scored / "04_logs/numeric-independent-qa.json") != report["validation"]["training_qa_sha256"]
            or sha(scored / "04_logs/answer-replay-qa.json") != report["validation"]["answer_replay_qa_sha256"]
            or replay["status"] != "EXACT_ANSWER_REPLAY_PASS"):
        raise ValueError("scored training/QA/replay evidence differs")
    # The cold recipe is a distinct completed validation, not the source of these weights.
    old = read(verified_cold_saved / "02_code/frozen.json")
    pins = read(verified_cold_saved / "02_code/manifest.json")
    for name, expected in pins.items():
        if sha(verified_cold_saved / name) != expected:
            raise ValueError("validated cold saved source drift")
    destination.mkdir(parents=True, exist_ok=False)
    for folder in ("01_data", "02_code", "03_model", "04_logs", "05_answer", "06_docs"):
        (destination / folder).mkdir()
    source_files = {"02_code/" + name: expected for name, expected in old["reviewed_source_sha256"].items()}
    copy_pins(verified_cold_saved, destination, source_files)
    for name in ("02_code/requirements.txt", "06_docs/shrink-provenance.json"):
        shutil.copy2(verified_cold_saved / name, destination / name)
    model_pins = {}
    for original in ("single.cbm", "multi.cbm", "router.joblib"):
        expected = report["new_training"]["models_sha256"][original]
        if sha(scored / "03_model" / original) != expected:
            raise ValueError("scored model bytes changed")
        name = "full_" + original
        shutil.copy2(scored / "03_model" / original, destination / "03_model" / name)
        model_pins[name] = expected
    c = {key: old[key] for key in ("purpose", "variant", "selected_columns", "recipe", "resource_overrides", "worker_timeout_seconds", "reviewed_source_sha256")}
    c.update(full_models_sha256=model_pins, expected_answer_sha256=EXPECTED,
             public_input_sha256=report["public_input_sha256"], training_qa_status="PASS",
             scored_report_sha256=sha(repo / "reports/p3_numeric_candidate_20260906_v1/result.json"),
             training_result_sha256=sha(scored / "04_logs/training-result.json"),
             training_qa_sha256=sha(scored / "04_logs/numeric-independent-qa.json"),
             new_cold_answer_sha256=old["expected_answer_sha256"],
             same_recipe_cold_whole_seconds=old["cold_whole_elapsed_seconds"],
             cold_answer_equals_scored=False,
             provenance="Scored full 2+1 models trained using authorized same-cycle OOF; separate whole-cold re-execution produced a different answer. No byte determinism claim.")
    save(destination / "02_code/frozen.json", c)
    save(destination / "06_docs/training-provenance.json", c)
    # Pure provenance naming correction only. Prediction code and fixed parameters unchanged.
    text = (verified_cold_saved / "02_code/infer.py").read_text(encoding="utf-8")
    replacements = {
        "cold_training_qa_status": "training_qa_status",
        "expected_cold_answer_sha256": "expected_scored_answer_sha256",
        "cold_provenance_sha256": "training_provenance_sha256",
        "06_docs/cold-provenance.json": "06_docs/training-provenance.json",
        "accepted cold": "accepted scored",
        "cold_training_reexecuted": "training_reexecuted",
    }
    for old_text, new_text in replacements.items():
        if old_text not in text:
            raise ValueError("expected adapter provenance text absent")
        text = text.replace(old_text, new_text)
    (destination / "02_code/infer.py").write_text(text, encoding="utf-8", newline="\n")
    (destination / "README.md").write_text(
        "# P3 scored numeric saved models\n\n"
        "These are the actual scored ff42 full models, not the new 56e2 cold weights. "
        "The separate SOURCE_ONLY archive contains full training. New cold training passed but changed 660 predictions "
        "(maximum 0.003506091m); it must not inherit the scored answer's official score.\n\n"
        "Set P3_DATA_DIR to the distributed P3_wave_forecast directory. From a fresh extraction run:\n\n"
        "`python -I 02_code/infer.py --preflight`\n\n"
        "`python -I 02_code/infer.py --infer --official-approved --output-dir <new absolute output directory>`\n\n"
        "The new output submission.csv must match ff42. No fits, old CSV inputs, hidden truth, or uploads. "
        "Inference is CPU2 with a 600s cap. The whole-cold and scored saved claims are separately recorded. "
        "The only inference-adapter edits rename provenance fields from cold to scored training; numerical logic is unchanged. "
        "See the top-level final guide for original fixed local-OOF shrink provenance and environment limitations.\n",
        encoding="utf-8")
    files = {p.relative_to(destination).as_posix(): sha(p) for p in destination.rglob("*") if p.is_file()}
    save(destination / "02_code/manifest.json", files)
    receipt = archive(destination, destination.parent / "P3_SAVED_MODELS.zip")
    receipt.update(status="SCORED_ARCHIVE_BUILT_REPLAY_PENDING", adapter_numerical_logic_changed=False,
                   model_files=3, answer_sha256=EXPECTED, new_fits=0)
    save(destination.parent / "P3_SCORED_BUILD_QA.json", receipt)
    print(receipt["status"], receipt["sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--verified-cold-saved", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    build(args.repo.resolve(), args.verified_cold_saved.resolve(), args.destination.resolve())
