from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_r1_report_artifact.py"
SPEC = importlib.util.spec_from_file_location("build_r1_report_artifact", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def _metric(f1: float) -> dict[str, float]:
    return {
        "f1": f1,
        "precision": f1 + 0.01,
        "recall": f1 - 0.01,
        "tp": 80.0,
        "fp": 10.0,
    }


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _fixtures(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    folds = []
    independent_folds = []
    baseline_folds = []
    for index, name in enumerate(("2025_q2", "2025_q3", "2025_q4")):
        base_micro = 0.80 + index * 0.01
        r1_micro = base_micro + 0.02
        base_weighted = 0.75 + index * 0.01
        r1_weighted = base_weighted + 0.02
        folds.append(
            {
                "fold": name,
                "candidate": {"micro": _metric(r1_micro), "weighted": _metric(r1_weighted)},
                "base": {"micro": _metric(base_micro), "weighted": _metric(base_weighted)},
            }
        )
        independent_folds.append(
            {
                "fold": name,
                "rows": 1000 + index,
                "positive_rows": 50 + index,
                "candidate": _metric(r1_micro),
                "baseline": _metric(base_micro),
                "delta": {"f1": 0.02},
            }
        )
        baseline_folds.append(
            {
                "fold": name,
                "candidate": {"micro": _metric(base_micro), "weighted": _metric(base_weighted)},
            }
        )

    metrics = {
        "outer_labels_used_for_selection": False,
        "folds": folds,
        "aggregate": {"micro": _metric(0.83), "weighted": _metric(0.78)},
        "base_aggregate": {"micro": _metric(0.81), "weighted": _metric(0.76)},
    }
    independent = {
        "status": "passed",
        "use_policy": {"candidate_selection_allowed": False, "evaluation_only": True},
        "scope": {"rows": 3003, "folds": 3, "groups": 2},
        "official_row_metrics": {
            "candidate": _metric(0.83),
            "baseline": _metric(0.81),
            "delta": {"f1": 0.02},
        },
        "test_share_weighted_metrics": {
            "candidate": _metric(0.78),
            "baseline": _metric(0.76),
            "delta": {"f1": 0.02},
        },
        "by_fold": independent_folds,
        "by_station_layer": [
            {"station": "A", "layer": 1, "candidate": _metric(0.81), "baseline": _metric(0.80)},
            {"station": "B", "layer": 2, "candidate": _metric(0.91), "baseline": _metric(0.90)},
        ],
        "paired_block_bootstrap": {
            "replicates": 2000,
            "difference_ci90": [0.01, 0.03],
            "probability_improved": 0.99,
        },
        "normal_station_layer_day_fp": {
            "candidate": {"false_positive_rows_per_normal_station_layer_day": 1.05},
            "baseline": {"false_positive_rows_per_normal_station_layer_day": 1.0},
        },
        "long_positive_events": {"delta_row_recall": 0.04},
    }
    manifest = {"finished_at": "2026-08-13T19:00:00+09:00"}
    preregistration = {
        "created_at_kst": "2026-08-13T18:20:00+09:00",
        "experiment_id": "R1_fixture",
        "baseline": {"run_id": "baseline_fixture"},
        "grid": {"total_candidates_including_no_op": 37},
        "outer_evaluation": {
            "promotion_gate": {
                "bootstrap_90pct_delta_lower_gt": 0.0,
                "folds_non_degrading_min": 2,
                "micro_f1_delta_min": 0.005,
                "normal_fp_day_relative_increase_lt": 0.1,
                "station_group_f1_drop_max": 0.01,
            }
        },
    }
    baseline = {
        "aggregate": {"micro": _metric(0.81), "weighted": _metric(0.76)},
        "folds": baseline_folds,
    }
    return (
        _write(tmp_path / "metrics.json", metrics),
        _write(tmp_path / "independent_validation.json", independent),
        _write(tmp_path / "manifest.json", manifest),
        _write(tmp_path / "preregistration.json", preregistration),
        _write(tmp_path / "baseline_metrics.json", baseline),
    )


def test_builds_complete_aggregate_only_technical_artifact(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    artifact = report.build_artifact(*paths)

    assert artifact["surface"] == "report"
    assert artifact["manifest"]["blocks"][0]["body"].startswith("# ")
    assert len(artifact["snapshot"]["datasets"]["fold_comparison"]) == 12
    assert artifact["snapshot"]["datasets"]["headline"][0]["promotion_passed"] is True
    assert {row["metric"] for row in artifact["snapshot"]["datasets"]["fold_comparison"]} == {
        "Row micro F1",
        "Test-share weighted F1",
    }
    block_text = "\n".join(block.get("body", "") for block in artifact["manifest"]["blocks"])
    for required in (
        "Technical Summary",
        "Scope and Metric Definitions",
        "Nested Selection and Provenance",
        "Robustness Checks",
        "Recommended Next Steps",
        "Further Questions",
    ):
        assert required in block_text
    serialized = json.dumps(artifact, ensure_ascii=False)
    assert str(tmp_path) not in serialized
    assert "candidate_oof" not in artifact["snapshot"]["datasets"]
    assert all(not Path(source["path"]).is_absolute() for source in artifact["sources"])


def test_cross_source_metric_mismatch_fails_closed(tmp_path: Path) -> None:
    paths = list(_fixtures(tmp_path))
    independent = json.loads(paths[1].read_text(encoding="utf-8"))
    independent["official_row_metrics"]["candidate"]["f1"] = 0.84
    _write(paths[1], independent)

    with pytest.raises(report.R1ReportArtifactError, match="does not reconcile"):
        report.build_artifact(*paths)


def test_cli_writes_only_artifact_json(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    output = tmp_path / "report" / "artifact.json"
    assert (
        report.main(
            [
                "--metrics",
                str(paths[0]),
                "--independent-validation",
                str(paths[1]),
                "--manifest",
                str(paths[2]),
                "--preregistration",
                str(paths[3]),
                "--baseline-metrics",
                str(paths[4]),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert output.is_file()
    assert list(output.parent.iterdir()) == [output]
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["surface"] == "report"
