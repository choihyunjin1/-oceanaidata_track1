from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_full_improvement_cycle_report_v2.py"
SPEC = importlib.util.spec_from_file_location("build_full_improvement_cycle_report_v2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def _inputs():
    return builder.collect_inputs(ROOT)


def test_r1_and_registry_exact_pins() -> None:
    _, _, r1 = _inputs()
    assert hashlib.sha256((ROOT / builder.R1_ARTIFACT).read_bytes()).hexdigest() == (
        builder.EXPECTED_R1_ARTIFACT_SHA256
    )
    assert hashlib.sha256((ROOT / builder.base.DEFAULT_REGISTRY).read_bytes()).hexdigest() == (
        builder.EXPECTED_REGISTRY_SHA256
    )
    assert r1["package_info"]["originUrl"] == "artifact://full-improvement-cycle-2026-08-22-r1"


def test_r2_adds_exact_visible_visual_blocks() -> None:
    _, _, r1 = _inputs()
    artifact = builder.build_artifact(r1)
    chart_blocks = [block for block in artifact["manifest"]["blocks"] if block["type"] == "chart"]
    table_blocks = [block for block in artifact["manifest"]["blocks"] if block["type"] == "table"]
    assert chart_blocks == builder.ADDED_BLOCKS[1:]
    assert table_blocks == builder.ADDED_BLOCKS[:1]
    assert [block["chartId"] for block in chart_blocks] == [
        chart["id"] for chart in artifact["manifest"]["charts"]
    ]
    assert [block["tableId"] for block in table_blocks] == [
        table["id"] for table in artifact["manifest"]["tables"]
    ]


def test_r2_normalizes_exactly_to_r1() -> None:
    _, _, r1 = _inputs()
    artifact = builder.build_artifact(r1)
    normalized = copy.deepcopy(artifact)
    added_ids = {block["id"] for block in builder.ADDED_BLOCKS}
    normalized["manifest"]["blocks"] = [
        block for block in normalized["manifest"]["blocks"] if block["id"] not in added_ids
    ]
    normalized["package_info"]["originUrl"] = r1["package_info"]["originUrl"]
    assert normalized == r1


def test_r2_preserves_registry_sources_datasets_and_narrative() -> None:
    _, _, r1 = _inputs()
    artifact = builder.build_artifact(r1)
    for field in ("charts", "tables", "sources"):
        assert artifact["manifest"][field] == r1["manifest"][field]
    assert artifact["snapshot"] == r1["snapshot"]
    r1_markdown = [block for block in r1["manifest"]["blocks"] if block["type"] == "markdown"]
    r2_markdown = [block for block in artifact["manifest"]["blocks"] if block["type"] == "markdown"]
    assert r2_markdown == r1_markdown


def test_r2_check_only_preserves_output() -> None:
    output = ROOT / builder.DEFAULT_OUTPUT
    before = (
        output.exists(),
        hashlib.sha256(output.read_bytes()).hexdigest() if output.exists() else None,
    )
    assert builder.main(["--root", str(ROOT), "--check-only"]) == 0
    after = (
        output.exists(),
        hashlib.sha256(output.read_bytes()).hexdigest() if output.exists() else None,
    )
    assert after == before
