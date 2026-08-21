from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_three_problem_execution_results_report.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_three_problem_execution_results_report", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def _evidence() -> dict[str, object]:
    p1 = {
        "baseline": {
            "weighted": {"f1": 0.5721820403993508},
            "micro": {"f1": 0.5979287928036888},
            "events": {"f1": 0.6377176421229285},
        },
        "candidate": {
            "weighted": {"f1": 0.5747732464951086},
            "micro": {"f1": 0.6126730412444699},
            "events": {"f1": 0.6113853904282115},
        },
        "gates": {
            "weighted_f1_delta": 0.002591206095757803,
            "bootstrap_ci90_lower": -0.01566918281468185,
            "normal_fp_day_relative_increase": 0.27867830423940143,
            "worst_station_layer_f1_delta": -0.05887445887445887,
        },
        "normal_station_layer_day_fp": {
            "baseline": {"false_positive_rows_per_normal_station_layer_day": 1.5247148288973384},
            "candidate": {"false_positive_rows_per_normal_station_layer_day": 1.9496197718631179},
        },
    }
    block = lambda r2, condition, support, coverage: {  # noqa: E731
        "residual_skill": {"pooled_temperature_r2": r2},
        "observability": {"condition": condition},
        "support": {"validation_supported_share": support},
        "coverage": {"two_public_temperature_share": coverage},
    }
    p2 = {
        "reference_metrics": {"adaptive_proxy_rmse": 0.7683674566216134},
        "precheck": {
            "aggregate_residual_r2": {
                "temp_l2": -0.5211931134976522,
                "temp_l3": -0.9307885097005606,
                "temp_l4": -1.014211564341955,
            },
            "by_block": {
                "2024_sep_oct": block(-1.6290207747806913, 21.5158, 0.9998, 0.9994),
                "2025_jul_aug": block(-0.04227664446042856, 61.1733, 1.0, 1.0),
                "2025_nov_dec": block(-3.6125000128008526, 56.3648, 0.1169171, 0.6410519),
            },
        },
    }
    p3 = {
        "gate": {
            "incumbent": {
                "rmse": 0.779748041094144,
                "by_lead": {"18": 0.8929582438638863, "24": 0.8434355226442385},
                "by_station": {
                    "G-ORS": 0.7275076409752465,
                    "I-ORS": 0.8867091368020714,
                    "S-ORS": 0.751671788202155,
                },
            },
            "candidate": {
                "rmse": 0.7840617300585763,
                "by_lead": {"18": 0.9048184374379117, "24": 0.8426147029779989},
                "by_station": {
                    "G-ORS": 0.7316541226142421,
                    "I-ORS": 0.8902603200200747,
                    "S-ORS": 0.7567932391056145,
                },
            },
            "delta_rmse": 0.004313688964432294,
        },
        "case_bootstrap": {"ci90": [0.00036830500573408755, 0.008118750224081208]},
        "episode_bootstrap": {"ci90": [0.0002657996067025215, 0.008229261402149963]},
    }
    return {
        "p1": {"metrics": p1, "manifest": {}},
        "p2": {"result": p2, "manifest": {}},
        "p3": {
            "metrics": p3,
            "receipt": {"family_cumulative_target_open_count": 2, "metrics_generation_count": 1},
        },
        "hashes": dict(report.EXPECTED_SHA256),
    }


def test_builds_one_chart_and_exact_stage_contracts() -> None:
    artifact = report.build_artifact(_evidence(), generated_at="2026-08-21T17:00:00+09:00")

    assert artifact["surface"] == "report"
    assert len(artifact["manifest"]["charts"]) == 1
    summary = artifact["snapshot"]["datasets"]["experiment_summary"]
    assert summary[0]["decision"] == "REJECT; outer evaluation count = 0"
    assert summary[1]["candidate"] == "not produced"
    assert summary[2]["decision"] == "REJECT; family opens = 2, metrics = 1"


def test_chart_uses_directionally_aligned_aggregate_evidence() -> None:
    artifact = report.build_artifact(_evidence(), generated_at="2026-08-21T17:00:00+09:00")
    rows = artifact["snapshot"]["datasets"]["normalized_signed_improvement"]

    assert rows[0]["normalized_signed_improvement"] > 0  # P1 higher F1 is better.
    assert all(row["normalized_signed_improvement"] < 0 for row in rows[1:4])
    assert rows[4]["normalized_signed_improvement"] < 0  # P3 higher RMSE is worse.
    assert {row["evidence"] for row in rows[1:4]} == {
        "P2 tide-RTS L2",
        "P2 tide-RTS L3",
        "P2 tide-RTS L4",
    }


def test_exact_gate_table_and_artifact_are_aggregate_only() -> None:
    artifact = report.build_artifact(_evidence(), generated_at="2026-08-21T17:00:00+09:00")
    gates = artifact["snapshot"]["datasets"]["exact_gate_table"]
    serialized = json.dumps(artifact, ensure_ascii=False)

    assert len(gates) == 17
    assert sum(row["status"] == "PASS" for row in gates) == 4
    assert sum(row["status"] == "NOT RUN" for row in gates) == 1
    assert "C:\\Users\\" not in serialized
    assert "target_hs" not in serialized
    assert all(not Path(source["path"]).is_absolute() for source in artifact["sources"])
