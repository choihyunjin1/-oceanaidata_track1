from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_v54_historical_promotable_candidate_registry_audit_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
RUNNER = ROOT / f"scripts/run_{EXPERIMENT_ID}.py"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _corpus() -> list[Path]:
    found: list[Path] = []
    for base in (ROOT / "artifacts", ROOT / "reports"):
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".json", ".md"}:
                continue
            rel = path.relative_to(ROOT).as_posix()
            low = rel.lower()
            if EXPERIMENT_ID in low:
                continue
            if "/p1" not in low and not path.name.lower().startswith("p1"):
                continue
            if "/p2" in low or "/p3" in low:
                continue
            if re.search(r"(result|terminal|qa|manifest|report-source)", path.name, re.I):
                found.append(path)
    return sorted(found, key=lambda item: item.relative_to(ROOT).as_posix().lower())


def test_frozen_metadata_corpus_count_and_digest() -> None:
    config = _config()["corpus_contract"]
    corpus = _corpus()
    records = []
    for path in corpus:
        payload = path.read_bytes()
        records.append(f"{path.relative_to(ROOT).as_posix()}|{len(payload)}|{hashlib.sha256(payload).hexdigest()}")
    digest = hashlib.sha256(("\n".join(records) + "\n").encode()).hexdigest()
    assert len(corpus) == config["files"] == 470
    assert sum(path.suffix.lower() == ".json" for path in corpus) == config["json_files"] == 374
    assert sum(path.suffix.lower() == ".md" for path in corpus) == config["markdown_files"] == 96
    assert digest == config["ordered_path_bytes_sha256_corpus_digest"]


def test_all_pinned_evidence_hashes_match() -> None:
    manifest = json.loads(
        (ROOT / f"reports/{EXPERIMENT_ID}/evidence-manifest.json").read_text(encoding="utf-8")
    )
    for item in manifest["pinned_evidence"]:
        path = ROOT / item["path"]
        assert path.stat().st_size == item["bytes"]
        assert _sha(path) == item["sha256"]


def test_promotion_funnel_counts_are_exact() -> None:
    config = _config()
    rows = config["reviewed_lineages"]
    counts = config["counts"]
    assert len(rows) == counts["terminal_positive_or_pass_lineages_reviewed"] == 9
    assert sum(row["b"] is True for row in rows) == counts["passed_unchanged_prospective_promotion_gate_b"] == 3
    assert sum(row["b"] is False for row in rows) == counts["failed_gate_b"] == 6
    assert sum(row["d"] is True for row in rows) == counts["exact_reproducible_commitment_d"] == 6
    assert sum(all(row[key] is True for key in "abcd") for row in rows) == 0


def test_every_gate_b_pass_is_exactly_already_submitted() -> None:
    config = _config()
    ledger = json.loads((ROOT / config["official_sha_ledger"]["path"]).read_text(encoding="utf-8"))
    submitted = {row["sha256"] for row in ledger["official_records"] if row.get("sha256")}
    gate_passes = [row for row in config["reviewed_lineages"] if row["b"] is True]
    assert len(gate_passes) == 3
    assert all(row["candidate_sha256"] in submitted for row in gate_passes)
    assert all(row["c"] is False and row["d"] is True for row in gate_passes)


def test_failed_historical_gates_are_not_reinterpreted() -> None:
    rows = {row["id"]: row for row in _config()["reviewed_lineages"]}
    assert rows["p1_public_transport_repair_cycle_20260831_v28"]["b"] is False
    assert "safety gates" in rows["p1_public_transport_repair_cycle_20260831_v28"]["exclusion"]
    assert rows["p1_iors_e150_microfragment_veto_20260901_v34a"]["b"] is False
    assert "metric-geometry" in rows["p1_iors_e150_microfragment_veto_20260901_v34a"]["exclusion"]


def test_terminal_has_no_candidate_contract_runner_or_lock() -> None:
    config = _config()
    assert config["status"] == "NO_OVERLOOKED_PROMOTABLE_P1_CANDIDATE"
    assert config["decision"]["candidate_pointer"] is None
    assert config["decision"]["fresh_confirmation_contract"] is None
    assert all(value == 0 for value in config["operations"].values())
    assert not RUNNER.exists()
    assert not LOCK.exists()


def test_official_ledger_is_used_only_for_sha_membership() -> None:
    ledger = _config()["official_sha_ledger"]
    assert ledger["records"] == 19 and ledger["distinct_non_null_sha256"] == 18
    assert ledger["performance_fields_used_for_v54"] == 0
    assert ledger["row_or_candidate_selection_uses"] == 0
    assert ledger["candidate_sha_membership_checks"] == 6
