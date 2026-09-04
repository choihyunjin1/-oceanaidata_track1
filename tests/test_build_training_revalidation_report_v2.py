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


v2 = _load(
    "build_training_revalidation_report_v2",
    ROOT / "scripts/build_training_revalidation_report_v2.py",
)


def _artifacts() -> tuple[dict[str, object], dict[str, object]]:
    evidence, registry, r1 = v2.collect_inputs(ROOT)
    return r1, v2.build_artifact(evidence, registry, r1)


def test_r2_changes_only_two_sort_fields_and_generation_identity() -> None:
    r1, r2 = _artifacts()
    normalized = copy.deepcopy(r2)
    for table in normalized["manifest"]["tables"]:
        assert table["defaultSort"]["field"] == "problem"
        table["defaultSort"]["field"] = "sequence"
    normalized["package_info"]["originUrl"] = r1["package_info"]["originUrl"]
    assert normalized == r1


def test_r2_default_sort_fields_are_declared_columns() -> None:
    _, r2 = _artifacts()
    for table in r2["manifest"]["tables"]:
        declared = {column["field"] for column in table["columns"]}
        assert table["defaultSort"]["field"] in declared


def test_r2_output_is_append_only_and_registry_is_unchanged() -> None:
    assert v2.DEFAULT_OUTPUT.as_posix() == (
        "reports/generated/training_revalidation_2026-08-22_r2/artifact.json"
    )
    assert v2.DEFAULT_OUTPUT != v2.R1_ARTIFACT
    assert v2.EXPECTED_REGISTRY_SHA256 == (
        "907c9f5b2df2a4ae70799ef1fadd04737fb619d0d9bc3c30f7009e1201f19117"
    )
