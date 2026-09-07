"""Aggregate-only consistency tests for the final-day read-only audit."""
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/p3_numeric_cpudet_noshrink_20260907_v1"


def read_result(name):
    return json.loads((REPORT / name).read_text(encoding="utf-8"))


def test_row_bin_weight_identity():
    data = read_result("test-mix-check.json")
    assert data["context"]["bin_counts"] == [80, 40, 33, 27, 20]
    for source in ["cpu", "gpu"]:
        run = data[source]
        assert sum(b["oof_rows"] for b in run["bins"]) == 103602
        assert run["oof_below_1_5_rows"] == 0
        for b in run["bins"]:
            assert b["weight"] * b["oof_share"] == pytest.approx(b["test_share"])
        for kind in ["base", "candidate"]:
            rmse = math.sqrt(sum(b["test_share"] * b[f"{kind}_rmse"]**2 for b in run["bins"]))
            assert rmse == pytest.approx(run[f"weighted_{kind}_rmse"], abs=1e-12)
        assert run["weighted_delta"] > 0
        assert run["short_leads_exact_unchanged"]


def test_sse_and_cluster_order_explanation():
    data = read_result("test-mix-check.json")
    lexical = read_result("test-mix-bootstrap-order.json")
    assert data["cpu"]["base_sse"] == pytest.approx(48091.9523451907, abs=1e-7)
    assert data["cpu"]["candidate_sse"] == pytest.approx(47546.9931632454, abs=1e-7)
    assert lexical["input_sha256"] == data["cpu"]["sha256"]
    assert lexical["ci90"] == pytest.approx([.002354, .007736], abs=5.1e-7)
    assert lexical["p_improve_exact"] == 1 / 2000
    assert lexical["p_improve_formatted_3dp"] == "0.001"
    assert data["new_fits"] == lexical["new_fits"] == 0
    assert data["thread_limit"] == lexical["thread_limit"] == 2


def test_deletion_table_count_and_unique_ui_keys():
    doc = (ROOT / "docs/ocean_v2_codex/USER_HANDOFF_DELETION_CHECK_20260907.md").read_text(encoding="utf-8")
    rows = [line for line in doc.splitlines()
            if line.startswith(("| 08-", "| 09-01")) and " | 0." in line]
    assert len(rows) == 39
    keys = [tuple(row.split("|")[1:3]) for row in rows]
    assert len(set(keys)) == 39
    assert sum("**미대조**" in row for row in rows) == 3
