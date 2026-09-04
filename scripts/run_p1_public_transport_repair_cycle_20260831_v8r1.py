"""Deterministic import-only recovery of P1 v8 under a new output prefix."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v8 as cycle  # noqa: E402

EXPERIMENT_ID = "p1_public_transport_repair_cycle_20260831_v8r1"
ORIGINAL_ARTIFACT = ROOT / "artifacts/p1_public_transport_repair_cycle_20260831_v8"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
DELIVERY = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용"
    r"\20260831_P1_PUBLIC_TRANSPORT_REPAIR_CYCLE_V8R1"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    original_lock = json.loads(
        (ORIGINAL_ARTIFACT / "attempt_lock.json").read_text(encoding="utf-8")
    )
    original_failure = json.loads(
        (ORIGINAL_ARTIFACT / "terminal_failure.json").read_text(encoding="utf-8")
    )
    if original_failure["error_type"] != "NameError" or "f1_score" not in original_failure[
        "error"
    ]:
        raise RuntimeError("unexpected v8 failure; recovery not authorized")
    cycle.EXPERIMENT_ID = EXPERIMENT_ID
    cycle.ARTIFACT = ARTIFACT
    cycle.REPORT = REPORT
    cycle.DELIVERY = DELIVERY
    result = cycle.execute()
    receipt = {
        "recovery_scope": "one missing sklearn.metrics.f1_score import only",
        "candidate_config_gate_unchanged": True,
        "config_sha256_unchanged": original_lock["config_sha256"]
        == sha256_file(cycle.CONFIG_PATH),
        "original_runner_sha256": original_lock["runner_sha256"],
        "corrected_runner_sha256": sha256_file(Path(cycle.__file__)),
        "recovery_wrapper_sha256": sha256_file(Path(__file__)),
        "original_consumed_model_fits": 3,
        "original_metric_rows_scored": 0,
        "original_official_covariate_reads": 0,
        "original_hidden_truth_reads": 0,
        "original_uploads": 0,
        "automatic_same_prefix_restart": False,
    }
    cycle.write_json(REPORT / "technical-recovery-receipt.json", receipt)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
