from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "build_external_data_breakthrough_report.py"
)
SPEC = importlib.util.spec_from_file_location("build_external_data_breakthrough_report", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def _evidence(*, point_complete: bool = False) -> dict[str, object]:
    per_layer = {
        "1": (0.4907715307977714, 0.4641319248823433, 0.05428107427528293, 32573),
        "2": (0.5047967110435361, 0.5163769636826555, -0.02294042806891565, 27122),
        "3": (0.896602162733059, 0.7140873505463048, 0.2035627614709351, 18072),
        "4": (1.5930874428975468, 1.4015962544961793, 0.12020130423793848, 32622),
        "5": (1.9417152421849495, 0.9778362048190873, 0.4964059695391999, 11695),
        "7": (3.6047729893501446, 1.158314276742657, 0.6786720605805822, 32621),
    }
    p1_v1 = {
        "metrics": {
            "baseline": {"rmse": 1.9363519368704971},
            "candidate": {"rmse": 0.9597394211774888},
            "rmse_relative_improvement": 0.5043569286642152,
            "q10_q90": {"coverage": 0.6930222035486895},
            "per_layer": {
                layer: {
                    "baseline": {"rmse": values[0]},
                    "candidate": {"rmse": values[1]},
                    "rmse_relative_improvement": values[2],
                    "rows": values[3],
                }
                for layer, values in per_layer.items()
            },
        }
    }
    point = {
        "status": "running",
        "decision": "RUNNING — P1 OOF 결과 대기",
        "path": None,
        "sha256": None,
        "duplicate_count": 0,
        "result": None,
    }
    if point_complete:
        point = {
            "status": "complete",
            "decision": "NO_GO_POINT_RESIDUAL",
            "passed": False,
            "weighted_f1_delta": -0.001,
            "path": "artifacts/p1_iors_external_point_residual_oof_v1/result.json",
            "sha256": "a" * 64,
            "duplicate_count": 1,
            "result": {},
        }
    return {
        "p1_v1": p1_v1,
        "p1_v2": {
            "point_metrics": {
                "candidate": {"rmse": 1.171231330440692},
                "rmse_relative_improvement": 0.3951350949488973,
            },
            "conformal_test": {"coverage": 0.8112148928606057},
        },
        "p1_point": point,
        "p2_nasa": {
            "metrics": {
                "external_incremental_candidate_vs_control": {
                    "delta_rmse": 0.0,
                }
            }
        },
        "p2_era5": {
            "transfer_gate": {
                "estimated_full_gib": 112.98942985013127,
                "estimated_full_hours_at_50mbps": 5.142274496290419,
            }
        },
        "p3_kma": {"status": "awaiting_credential"},
        "hashes": dict(report.EXPECTED_SHA256),
    }


def test_builds_one_horizontal_chart_and_three_exact_tables() -> None:
    artifact = report.build_artifact(_evidence(), generated_at="2026-08-21T18:00:00+09:00")

    assert artifact["surface"] == "report"
    assert len(artifact["manifest"]["charts"]) == 1
    assert artifact["manifest"]["charts"][0]["type"] == "horizontalBar"
    assert len(artifact["manifest"]["tables"]) == 3
    rows = artifact["snapshot"]["datasets"]["p1_v1_layer_rmse_improvement"]
    assert len(rows) == 6
    assert rows[0]["layer"] == "Layer 7"
    assert rows[-1]["layer"] == "Layer 2"
    assert sum(row["rmse_relative_improvement"] >= 0 for row in rows) == 5


def test_decisions_are_scoped_and_missing_point_result_is_running() -> None:
    artifact = report.build_artifact(_evidence(), generated_at="2026-08-21T18:00:00+09:00")
    rows = artifact["snapshot"]["datasets"]["decision_register"]

    assert rows[2]["decision"].startswith("RUNNING")
    assert rows[3]["decision"] == "NO_GO_NASA_POWER_RESIDUAL_META_V1"
    assert "ERA5 모델 미평가" in rows[4]["decision"]
    assert rows[5]["decision"] == "AWAITING_CREDENTIAL"


def test_completed_point_result_is_content_addressed_and_included() -> None:
    artifact = report.build_artifact(
        _evidence(point_complete=True), generated_at="2026-08-21T18:00:00+09:00"
    )
    rows = artifact["snapshot"]["datasets"]["decision_register"]
    integrity = artifact["snapshot"]["datasets"]["evidence_integrity"]

    assert rows[2]["decision"] == "NO_GO_POINT_RESIDUAL"
    assert rows[2]["headline"] == "overall weighted F1 delta -0.001000"
    assert integrity[-1] == {
        "evidence": "p1_point",
        "path": "artifacts/p1_iors_external_point_residual_oof_v1/result.json",
        "sha256": "a" * 64,
        "seal": "build-time content address; single unique result",
    }


def test_report_is_aggregate_only_portable_and_has_technical_structure() -> None:
    artifact = report.build_artifact(_evidence(), generated_at="2026-08-21T18:00:00+09:00")
    serialized = json.dumps(artifact, ensure_ascii=False)
    ids = {block["id"] for block in artifact["manifest"]["blocks"]}

    assert {
        "technical_summary",
        "visual_finding",
        "scope_definitions",
        "methodology",
        "limitations",
        "next_steps",
        "further_questions",
    } <= ids
    assert "C:\\Users\\" not in serialized
    assert "station,time,temp" not in serialized
    assert "incumbent_probability" not in serialized
    assert all(not Path(source["path"]).is_absolute() for source in artifact["sources"])


def test_discovery_refuses_ambiguous_nonidentical_optional_results(tmp_path: Path) -> None:
    result_dir = tmp_path / report.POINT_RESULT_DIR
    (result_dir / "run_a").mkdir(parents=True)
    (result_dir / "run_b").mkdir(parents=True)
    base = {
        "experiment_id": "p1_iors_external_point_residual_oof_v1",
        "decision": "NO_GO",
        "gate": {"passed": False},
        "submission_created": False,
    }
    (result_dir / "run_a" / "result.json").write_text(json.dumps(base), encoding="utf-8")
    base["decision"] = "GO"
    (result_dir / "run_b" / "result.json").write_text(json.dumps(base), encoding="utf-8")

    with pytest.raises(report.ReportEvidenceError, match="ambiguous"):
        report._discover_point_result(tmp_path)


def test_sealed_reader_fails_closed_on_sha_mismatch(tmp_path: Path) -> None:
    evidence = tmp_path / "result.json"
    evidence.write_text('{"decision":"NO_GO"}', encoding="utf-8")
    actual = hashlib.sha256(evidence.read_bytes()).hexdigest()

    assert report._read_sealed_json(evidence, actual, "fixture")["decision"] == "NO_GO"
    with pytest.raises(report.ReportEvidenceError, match="sealed SHA mismatch"):
        report._read_sealed_json(evidence, "0" * 64, "fixture")
