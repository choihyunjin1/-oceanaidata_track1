from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v1 = _load("build_validation_system_audit_report_v1_for_r2_test", ROOT / "scripts/build_validation_system_audit_report_v1.py")
v2 = _load("build_validation_system_audit_report_v2", ROOT / "scripts/build_validation_system_audit_report_v2.py")


def _artifacts() -> tuple[dict[str, object], dict[str, object]]:
    evidence = v1.collect_evidence(ROOT)
    generated_at = "2026-08-22T18:00:00+09:00"
    return (
        v1.build_artifact(evidence, generated_at=generated_at),
        v2.build_artifact(evidence, generated_at=generated_at),
    )


def test_r2_changes_only_scope_metric_label_and_generation_identity() -> None:
    r1, r2 = _artifacts()
    normalized = copy.deepcopy(r2)
    scope = next(
        block
        for block in normalized["manifest"]["blocks"]
        if block["id"] == "scope_definitions"
    )
    assert v2.NEW_SCOPE_LABEL in scope["body"]
    assert v2.OLD_SCOPE_LABEL not in scope["body"]
    scope["body"] = scope["body"].replace(v2.NEW_SCOPE_LABEL, v2.OLD_SCOPE_LABEL, 1)
    normalized["package_info"]["originUrl"] = r1["package_info"]["originUrl"]
    assert normalized == r1


def test_r2_preserves_sources_ids_notes_and_evidence_pins() -> None:
    r1, r2 = _artifacts()
    assert r2["manifest"]["sources"] == r1["manifest"]["sources"]
    assert r2["sources"] == r1["sources"]
    assert [block["id"] for block in r2["manifest"]["blocks"]] == [
        block["id"] for block in r1["manifest"]["blocks"]
    ]
    assert r2["manifest"]["charts"] == r1["manifest"]["charts"]
    assert r2["manifest"]["tables"] == r1["manifest"]["tables"]
    assert r2["snapshot"] == r1["snapshot"]
    assert v2.base.EXPECTED_SHA256 == v1.EXPECTED_SHA256


def test_r2_output_contract_is_append_only() -> None:
    assert v2.DEFAULT_OUTPUT.as_posix() == (
        "reports/generated/validation_system_audit_2026-08-22_r2/artifact.json"
    )
    assert v2.REPORT_ID.endswith("-r2")
    assert v2.DEFAULT_OUTPUT != v1.DEFAULT_OUTPUT
