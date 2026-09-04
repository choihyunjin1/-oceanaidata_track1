from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v2 = _load("build_validation_system_audit_report_v2_for_r3_test", ROOT / "scripts/build_validation_system_audit_report_v2.py")
v3 = _load("build_validation_system_audit_report_v3", ROOT / "scripts/build_validation_system_audit_report_v3.py")


def _artifacts() -> tuple[dict[str, object], dict[str, object]]:
    evidence = v2.base.collect_evidence(ROOT)
    generated_at = "2026-08-22T18:00:00+09:00"
    return (
        v2.build_artifact(evidence, generated_at=generated_at),
        v3.build_artifact(evidence, generated_at=generated_at),
    )


def test_r3_changes_only_dependent_note_and_generation_identity() -> None:
    r2, r3 = _artifacts()
    normalized = copy.deepcopy(r3)
    method = next(
        source
        for source in normalized["manifest"]["sources"]
        if source["id"] == "method_note"
    )
    assert v3.NEW_NOTE_LABEL in method["note"]
    assert v3.OLD_NOTE_LABEL not in method["note"]
    method["note"] = method["note"].replace(v3.NEW_NOTE_LABEL, v3.OLD_NOTE_LABEL, 1)
    normalized["package_info"]["originUrl"] = r2["package_info"]["originUrl"]
    assert normalized == r2


def test_r3_has_no_water_level_target_label_and_preserves_hs_scope() -> None:
    _, r3 = _artifacts()
    serialized = json.dumps(r3, ensure_ascii=False)
    assert "water-level RMSE" not in serialized
    assert "P3는 유의파고(hs) RMSE(m)" in serialized
    assert "significant-wave-height (hs) RMSE/case sampling" in serialized


def test_r3_output_contract_is_append_only() -> None:
    assert v3.DEFAULT_OUTPUT.as_posix() == (
        "reports/generated/validation_system_audit_2026-08-22_r3/artifact.json"
    )
    assert v3.REPORT_ID.endswith("-r3")
    assert v3.DEFAULT_OUTPUT != v2.DEFAULT_OUTPUT
