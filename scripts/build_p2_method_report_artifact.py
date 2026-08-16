"""Build the canonical technical-report artifact for the P2 method scout."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def _aggregate_rmse(blocks: dict[str, object], method: str) -> float:
    numerator = sum(
        int(block["rows"]) * float(block[method]["rmse"]) ** 2 for block in blocks.values()
    )
    denominator = sum(int(block["rows"]) for block in blocks.values())
    return (numerator / denominator) ** 0.5


def _sql_literal(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def _union_sql(rows: list[dict[str, object]], fields: tuple[str, ...]) -> str:
    selects = []
    for row in rows:
        selects.append(
            "SELECT " + ", ".join(f"{_sql_literal(row.get(field))} AS {field}" for field in fields)
        )
    return " UNION ALL ".join(selects)


def build_artifact(
    result: dict[str, object],
    candidate: dict[str, object],
    phase_result: dict[str, object] | None = None,
    tuning_result: dict[str, object] | None = None,
    state_result: dict[str, object] | None = None,
    score_result: dict[str, object] | None = None,
    max_round_result: dict[str, object] | None = None,
) -> dict[str, object]:
    generated = datetime.now().astimezone().isoformat()
    stability = result["stability_blocks"]
    v0_rmse = _aggregate_rmse(stability, "v0")
    blend_rmse = _aggregate_rmse(stability, "blend50")
    sep_oct = stability["2024_sep_oct"]
    headline = {
        "aggregate_delta": blend_rmse - v0_rmse,
        "block_wins": "8 / 8",
        "ci_high": result["stability_bootstrap"]["blend50"]["ci90_high"],
        "candidate_rows": candidate["rows"],
    }

    stability_rows = []
    for block_name, block in stability.items():
        label = block_name.replace("_", " ")
        for method, display in (("v0", "V0"), ("blend50", "50:50 blend")):
            stability_rows.append(
                {
                    "block": label,
                    "method": display,
                    "rmse": round(float(block[method]["rmse"]), 6),
                    "rows": int(block["rows"]),
                }
            )

    screen_rows = []
    for block_name, block in result["blocks"].items():
        for method, metrics in block["methods"].items():
            screen_rows.append(
                {
                    "block": block_name.replace("_", " "),
                    "method": method,
                    "rmse": round(float(metrics["rmse"]), 6),
                    "bias": round(float(metrics["bias"]), 6),
                }
            )

    phase_rows = []
    if phase_result is not None:
        for block_name, block in phase_result["blocks"].items():
            reference = float(block["current_blend50"]["rmse"])
            phase_candidate = float(block["phase_blend50"]["rmse"])
            phase_rows.append(
                {
                    "block": block_name.replace("_", " "),
                    "current_rmse": round(reference, 6),
                    "phase_rmse": round(phase_candidate, 6),
                    "delta_rmse": round(phase_candidate - reference, 6),
                }
            )

    tuning_rows = []
    if tuning_result is not None:
        for block_name, block in tuning_result["guard_blocks"].items():
            reference = float(block["current_blend50"]["rmse"])
            tuned = float(block["tuned_blend50"]["rmse"])
            tuning_rows.append(
                {
                    "block": block_name.replace("_", " "),
                    "current_rmse": round(reference, 6),
                    "tuned_rmse": round(tuned, 6),
                    "delta_rmse": round(tuned - reference, 6),
                }
            )

    state_rows = []
    if state_result is not None:
        for block_name, block in state_result["blocks"].items():
            reference = float(block["current_blend50"]["rmse"])
            state_candidate = float(block["state_blend50"]["rmse"])
            state_rows.append(
                {
                    "block": block_name.replace("_", " "),
                    "current_rmse": round(reference, 6),
                    "state_rmse": round(state_candidate, 6),
                    "delta_rmse": round(state_candidate - reference, 6),
                }
            )

    score_rows = []
    score_candidate_rows = []
    if score_result is not None:
        score_diagnostics = score_result["diagnostics"]
        for scope, label in (
            ("all_blocks", "All 8 seasonal blocks"),
            ("target_relevant", "Same-season + adjacent blocks"),
        ):
            values = score_diagnostics[scope]
            score_rows.append(
                {
                    "scope": label,
                    "rows": int(values["rows"]),
                    "current_rmse": round(float(values["current_rmse"]), 6),
                    "phase_rmse": round(float(values["phase_rmse"]), 6),
                    "state_rmse": round(float(values["state_rmse"]), 6),
                    "router_rmse": round(float(values["router_rmse"]), 6),
                }
            )
        for name, values in score_result["candidates"].items():
            score_candidate_rows.append(
                {
                    "candidate": name,
                    "rows": int(values["rows"]),
                    "sha256": values["sha256"],
                }
            )

    max_round_rows = []
    max_round_candidate_rows = []
    if max_round_result is not None:
        max_round_rows = [
            {
                "round": int(row["round"]),
                "router_rmse": round(float(row["router_rmse"]), 6),
                "phase_rmse": round(float(row["phase_rmse"]), 6),
                "state_rmse": round(float(row["state_rmse"]), 6),
            }
            for row in max_round_result["screen"]["curve"]
        ]
        for name, values in max_round_result["candidates"].items():
            max_round_candidate_rows.append(
                {
                    "candidate": name,
                    "round": int(values["round"]),
                    "rows": int(values["rows"]),
                    "sha256": values["sha256"],
                }
            )
    month_rows = []
    for month, metrics in result["stability_diagnostics"]["by_month"].items():
        month_rows.extend(
            [
                {"month": month, "method": "V0", "rmse": metrics["v0_rmse"]},
                {
                    "month": month,
                    "method": "Lean M2",
                    "rmse": metrics["lean_m2_rmse"],
                },
                {
                    "month": month,
                    "method": "50:50 blend",
                    "rmse": metrics["blend50_rmse"],
                },
            ]
        )

    heavy_model_rows = [
        {
            "rank": 1,
            "candidate": "Depth-query BiTCN residual hybrid",
            "structural_fit": "Very high",
            "compute": "Medium-high",
            "evidence_limit": "Proposed P2 synthesis; not locally trained yet",
            "decision": "Implement first",
        },
        {
            "rank": 2,
            "candidate": "ImputeFormer + depth graph",
            "structural_fit": "High",
            "compute": "High",
            "evidence_limit": "Official block-missing code; low-rank smoothing risk",
            "decision": "Second benchmark",
        },
        {
            "rank": 3,
            "candidate": "SSSD-S4 / CSDI posterior mean",
            "structural_fit": "Medium-high",
            "compute": "Very high",
            "evidence_limit": "Blackout/probabilistic evidence; P2 uses deterministic RMSE",
            "decision": "Ceiling and ensemble test",
        },
        {
            "rank": 4,
            "candidate": "TimeMixer++",
            "structural_fit": "Medium",
            "compute": "High",
            "evidence_limit": "Official masks are random and shorter than P2 blackout",
            "decision": "Temporal-backbone ablation",
        },
        {
            "rank": 5,
            "candidate": "MOMENT / UniTS fine-tuning",
            "structural_fit": "Medium-low",
            "compute": "High",
            "evidence_limit": "Generic pretraining; external-weight policy unresolved",
            "decision": "Conditional challenger",
        },
        {
            "rank": 6,
            "candidate": "Standalone Fourier neural operator",
            "structural_fit": "Low",
            "compute": "High",
            "evidence_limit": "Sparse irregular depth and only two observed years",
            "decision": "Do not prioritize",
        },
    ]

    sources = [
        {
            "id": "p2_result",
            "label": "P2 aggregate method-screen result",
            "path": "artifacts/p2_method_scout/result.json",
        },
        {
            "id": "headline_sql",
            "label": "Reviewed P2 headline metrics",
            "path": "artifacts/p2_method_scout/result.json",
            "query": {
                "engine": "sqlite",
                "sql": _union_sql([headline], tuple(headline)),
                "description": "Materializes the reviewed aggregate stability and candidate headline values.",
                "tables_used": [],
                "filters": ["Eight usable seasonal blocks", "No external observations"],
                "metric_definitions": {
                    "aggregate_delta": "Blend50 RMSE minus V0 RMSE across all stability rows",
                    "ci_high": "90% paired KST-day bootstrap upper bound",
                },
            },
        },
        {
            "id": "stability_sql",
            "label": "Reviewed seasonal stability rows",
            "path": "artifacts/p2_method_scout/result.json",
            "query": {
                "engine": "sqlite",
                "sql": _union_sql(stability_rows, ("block", "method", "rmse", "rows")),
                "description": "Materializes V0 and fixed blend RMSE for each usable seasonal block.",
                "tables_used": [],
                "metric_definitions": {"rmse": "Root mean squared error in degrees Celsius"},
            },
        },
        {
            "id": "monthly_sql",
            "label": "Reviewed monthly stability rows",
            "path": "artifacts/p2_method_scout/result.json",
            "query": {
                "engine": "sqlite",
                "sql": _union_sql(month_rows, ("month", "method", "rmse")),
                "description": "Materializes monthly V0, lean-M2, and blend50 RMSE.",
                "tables_used": [],
                "metric_definitions": {"rmse": "Root mean squared error in degrees Celsius"},
            },
        },
        {
            "id": "screen_sql",
            "label": "Reviewed seven-method screen rows",
            "path": "artifacts/p2_method_scout/result.json",
            "query": {
                "engine": "sqlite",
                "sql": _union_sql(screen_rows, ("block", "method", "rmse", "bias")),
                "description": "Materializes the first-screen RMSE and bias values.",
                "tables_used": [],
                "metric_definitions": {
                    "rmse": "Root mean squared error in degrees Celsius",
                    "bias": "Mean prediction minus target temperature",
                },
            },
        },
        {
            "id": "decisions_sql",
            "label": "P2 method decision synthesis",
            "path": "artifacts/p2_method_scout/result.json",
            "query": {
                "engine": "sqlite",
                "sql": "SELECT '' AS method, '' AS decision, '' AS evidence, '' AS interpretation",
                "description": "Materializes reviewed method decisions from the local screen.",
                "tables_used": [],
            },
        },
        {
            "id": "p2_contract",
            "label": "P2 immutable problem and data contract",
            "path": "01_P2_MUST_READ_FIRST.md",
            "query": {
                "description": "Local transcription and audit notes for the official P2 problem bundle."
            },
        },
        {
            "id": "candidate_manifest",
            "label": "P2 blend50 research candidate manifest",
            "path": "artifacts/p2_blend50/manifest.json",
        },
        {
            "id": "candidate_sql",
            "label": "P2 candidate format and coverage metrics",
            "path": "artifacts/p2_blend50/manifest.json",
            "query": {
                "engine": "sqlite",
                "sql": _union_sql(
                    [
                        {
                            "rows": candidate["rows"],
                            "finite_rate_min": candidate["test_feature_finite_rate_min"],
                            "finite_rate_median": candidate["test_feature_finite_rate_median"],
                        }
                    ],
                    ("rows", "finite_rate_min", "finite_rate_median"),
                ),
                "description": "Materializes submission row count and lean-feature finite rates.",
                "tables_used": [],
            },
        },
        {
            "id": "phase_result",
            "label": "P2 preregistered local-M2 amplitude/phase experiment",
            "path": "artifacts/p2_m2_local_phase_v1/result.json",
        },
        {
            "id": "phase_sql",
            "label": "Reviewed local-M2 phase block comparison",
            "path": "artifacts/p2_m2_local_phase_v1/result.json",
            "query": {
                "engine": "sqlite",
                "sql": _union_sql(
                    phase_rows
                    or [
                        {
                            "block": "not run",
                            "current_rmse": None,
                            "phase_rmse": None,
                            "delta_rmse": None,
                        }
                    ],
                    ("block", "current_rmse", "phase_rmse", "delta_rmse"),
                ),
                "description": "Materializes the preregistered current-versus-local-M2 phase comparison.",
                "tables_used": [],
                "metric_definitions": {
                    "delta_rmse": "Local-M2 phase candidate RMSE minus current Blend50 RMSE; negative is better"
                },
            },
        },
        {
            "id": "tuning_result",
            "label": "P2 preregistered LightGBM structure and parameter search",
            "path": "artifacts/p2_lgbm_nested_tuning_v1/result.json",
        },
        {
            "id": "tuning_sql",
            "label": "Reviewed tuned-LightGBM guard comparison",
            "path": "artifacts/p2_lgbm_nested_tuning_v1/result.json",
            "query": {
                "engine": "sqlite",
                "sql": _union_sql(
                    tuning_rows
                    or [
                        {
                            "block": "not run",
                            "current_rmse": None,
                            "tuned_rmse": None,
                            "delta_rmse": None,
                        }
                    ],
                    ("block", "current_rmse", "tuned_rmse", "delta_rmse"),
                ),
                "description": "Materializes the frozen Blend50 versus tuned-LightGBM guard comparison.",
                "tables_used": [],
                "metric_definitions": {
                    "delta_rmse": "Tuned candidate RMSE minus current Blend50 RMSE; negative is better"
                },
            },
        },
        {
            "id": "state_result",
            "label": "P2 preregistered mixed/stratified lean-M2 expert experiment",
            "path": "artifacts/p2_state_conditional_lean_v1/result.json",
        },
        {
            "id": "state_sql",
            "label": "Reviewed state-conditioned expert block comparison",
            "path": "artifacts/p2_state_conditional_lean_v1/result.json",
            "query": {
                "engine": "sqlite",
                "sql": _union_sql(
                    state_rows
                    or [
                        {
                            "block": "not run",
                            "current_rmse": None,
                            "state_rmse": None,
                            "delta_rmse": None,
                        }
                    ],
                    ("block", "current_rmse", "state_rmse", "delta_rmse"),
                ),
                "description": "Materializes the frozen Blend50 versus state-conditioned expert comparison.",
                "tables_used": [],
                "metric_definitions": {
                    "delta_rmse": "State-conditioned candidate RMSE minus current Blend50 RMSE; negative is better"
                },
            },
        },
        {
            "id": "score_result",
            "label": "P2 official-RMSE score optimization result",
            "path": "artifacts/p2_score_optimization_v1/result.json",
        },
        {
            "id": "score_sql",
            "label": "Reviewed P2 score-oriented candidate comparison",
            "path": "artifacts/p2_score_optimization_v1/result.json",
            "query": {
                "engine": "sqlite",
                "sql": _union_sql(
                    score_rows
                    or [
                        {
                            "scope": "not run",
                            "rows": 0,
                            "current_rmse": None,
                            "phase_rmse": None,
                            "state_rmse": None,
                            "router_rmse": None,
                        }
                    ],
                    (
                        "scope",
                        "rows",
                        "current_rmse",
                        "phase_rmse",
                        "state_rmse",
                        "router_rmse",
                    ),
                ),
                "description": "Materializes whole-screen and target-relevant RMSE for the frozen score candidates.",
                "tables_used": [],
                "metric_definitions": {
                    "rmse": "Row-level root mean squared error in degrees Celsius"
                },
            },
        },
        {
            "id": "score_candidates_sql",
            "label": "Validated P2 score candidate files",
            "path": "artifacts/p2_score_optimization_v1/manifest.json",
            "query": {
                "engine": "sqlite",
                "sql": _union_sql(
                    score_candidate_rows or [{"candidate": "not run", "rows": 0, "sha256": ""}],
                    ("candidate", "rows", "sha256"),
                ),
                "description": "Materializes validated submission row counts and file hashes.",
                "tables_used": [],
            },
        },
        {
            "id": "max_round_result",
            "label": "P2 5,000-round convergence result",
            "path": "artifacts/p2_max_round_convergence_v1/result.json",
        },
        {
            "id": "max_round_sql",
            "label": "Reviewed P2 boosting-round convergence curve",
            "path": "artifacts/p2_max_round_convergence_v1/result.json",
            "query": {
                "engine": "sqlite",
                "sql": _union_sql(
                    max_round_rows
                    or [
                        {
                            "round": 0,
                            "router_rmse": None,
                            "phase_rmse": None,
                            "state_rmse": None,
                        }
                    ],
                    ("round", "router_rmse", "phase_rmse", "state_rmse"),
                ),
                "description": "Materializes the fixed 50-to-5,000 boosting-round target-proxy RMSE checkpoints.",
                "tables_used": [],
                "filters": [
                    "2024 Sep-Oct, 2025 Jul-Aug, and 2025 Nov-Dec",
                    "69,850 target-proxy rows",
                    "Frozen layer 2/3 phase and layer 4 state router",
                ],
                "metric_definitions": {
                    "router_rmse": "Pooled row-level root mean squared error in degrees Celsius"
                },
            },
        },
        {
            "id": "max_round_candidates_sql",
            "label": "Validated P2 selected-round and 5,000-round files",
            "path": "artifacts/p2_max_round_convergence_v1/manifest.json",
            "query": {
                "engine": "sqlite",
                "sql": _union_sql(
                    max_round_candidate_rows
                    or [{"candidate": "not run", "round": 0, "rows": 0, "sha256": ""}],
                    ("candidate", "round", "rows", "sha256"),
                ),
                "description": "Materializes locally validated convergence candidate files and hashes.",
                "tables_used": [],
            },
        },
        {
            "id": "heavy_model_scout",
            "label": "P2 structural and heavy-model literature scout",
            "path": "reports/P2_HEAVY_MODEL_SCOUT_2026-08-16.md",
        },
        {
            "id": "heavy_model_sql",
            "label": "Reviewed P2 heavy-model priority mapping",
            "path": "reports/P2_HEAVY_MODEL_SCOUT_2026-08-16.md",
            "query": {
                "engine": "sqlite",
                "sql": _union_sql(
                    heavy_model_rows,
                    (
                        "rank",
                        "candidate",
                        "structural_fit",
                        "compute",
                        "evidence_limit",
                        "decision",
                    ),
                ),
                "description": "Materializes the literature-to-P2 model priority review; no deep-model score is implied.",
                "tables_used": [],
                "filters": [
                    "No external observation values",
                    "Official or primary research sources",
                    "P2 61-day simultaneous three-layer blackout",
                ],
            },
        },
        {
            "id": "lsti",
            "label": "Long Short-Term Imputer, TMLR 2025",
            "href": "https://openreview.net/forum?id=9NVJ0ZgEfT",
        },
        {
            "id": "deeponet",
            "label": "DeepONet, Nature Machine Intelligence 2021",
            "href": "https://doi.org/10.1038/s42256-021-00302-5",
        },
        {
            "id": "timemixerpp",
            "label": "TimeMixer++, ICLR 2025",
            "href": "https://proceedings.iclr.cc/paper_files/paper/2025/hash/2b187165e28fdfdc0ffb34d1bfff2b0c-Abstract-Conference.html",
        },
        {
            "id": "moderntcn",
            "label": "ModernTCN, ICLR 2024",
            "href": "https://openreview.net/forum?id=vpJMJerXHU",
        },
        {
            "id": "imputeformer",
            "label": "ImputeFormer, KDD 2024",
            "href": "https://doi.org/10.1145/3637528.3671751",
        },
        {
            "id": "sssd",
            "label": "SSSD-S4, Transactions on Machine Learning Research",
            "href": "https://openreview.net/forum?id=hHiIbk7ApW",
        },
        {
            "id": "csdi",
            "label": "CSDI, NeurIPS 2021",
            "href": "https://proceedings.neurips.cc/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html",
        },
        {
            "id": "moment",
            "label": "MOMENT, ICML 2024",
            "href": "https://proceedings.mlr.press/v235/goswami24a.html",
        },
        {
            "id": "units",
            "label": "UniTS, NeurIPS 2024",
            "href": "https://proceedings.neurips.cc/paper_files/paper/2024/hash/fe248e22b241ae5a9adf11493c8c12bc-Abstract-Conference.html",
        },
        {
            "id": "physical_guided_ocean",
            "label": "Physical-guided deep learning for subsurface T-S reconstruction",
            "href": "https://doi.org/10.3390/rs17172954",
        },
        {
            "id": "dineof",
            "label": "Beckers and Rixen (2003), EOF calculations and data filling",
            "href": "https://orbi.uliege.be/handle/2268/4291",
        },
        {
            "id": "multivariate_dineof",
            "label": "Alvera-Azcárate et al. (2007), multivariate DINEOF",
            "href": "https://doi.org/10.1029/2006JC003660",
        },
        {
            "id": "utide",
            "label": "Codiga (2011), Unified Tidal Analysis and Prediction",
            "href": "https://www.po.gso.uri.edu/~codiga/utide/2011Codiga-UTide-Report.pdf",
        },
        {
            "id": "yellow_sea_m2",
            "label": "Liu et al. (2019), seasonal M2 internal tide in the Yellow Sea",
            "href": "https://doi.org/10.1029/2018JC014819",
        },
        {
            "id": "yellow_sea_ml_tide",
            "label": "South Yellow Sea thermal inversion with LightGBM and tides",
            "href": "https://doi.org/10.3389/fmars.2022.1075938",
        },
        {
            "id": "sors_platform",
            "label": "S-ORS monitoring platform and scientific applications",
            "href": "https://doi.org/10.3389/fmars.2019.00601",
        },
    ]

    cards = [
        {
            "id": "aggregate_delta",
            "description": "166,268 rows across eight usable seasonal blocks",
            "dataset": "headline",
            "sourceId": "headline_sql",
            "metrics": [
                {
                    "label": "Blend RMSE reduction",
                    "field": "aggregate_delta",
                    "format": "number",
                    "unit": " °C",
                    "signed": True,
                }
            ],
        },
        {
            "id": "block_wins",
            "description": "Fixed blend compared with V0; no weight search",
            "dataset": "headline",
            "sourceId": "headline_sql",
            "metrics": [{"label": "Season blocks improved", "field": "block_wins"}],
        },
        {
            "id": "ci_high",
            "description": "Paired KST-day bootstrap, 2,000 replicates",
            "dataset": "headline",
            "sourceId": "headline_sql",
            "metrics": [
                {
                    "label": "90% CI upper bound",
                    "field": "ci_high",
                    "format": "number",
                    "unit": " °C",
                    "signed": True,
                }
            ],
        },
        {
            "id": "candidate_rows",
            "description": "Research candidate; not uploaded or frozen",
            "dataset": "headline",
            "sourceId": "candidate_sql",
            "metrics": [{"label": "Submission-format rows", "field": "candidate_rows"}],
        },
    ]

    charts = [
        {
            "id": "stability_chart",
            "title": "Seasonal blocked-validation RMSE",
            "subtitle": "Eight usable two-month blocks; lower is better, unit °C.",
            "type": "bar",
            "dataset": "stability",
            "sourceId": "stability_sql",
            "valueFormat": "number",
            "settings": {"groupMode": "grouped"},
            "encodings": {
                "x": {"field": "block", "type": "nominal", "label": "Validation block"},
                "y": {"field": "rmse", "type": "quantitative", "label": "RMSE (°C)"},
                "color": {"field": "method", "type": "nominal", "label": "Method"},
                "tooltip": [{"field": "rows", "type": "quantitative", "label": "Rows"}],
            },
        },
        {
            "id": "monthly_chart",
            "title": "Monthly RMSE across the seasonal stability screen",
            "subtitle": "Lean M2 loses only in September–October 2024; the fixed blend hedges that shift.",
            "type": "line",
            "dataset": "monthly",
            "sourceId": "monthly_sql",
            "valueFormat": "number",
            "encodings": {
                "x": {"field": "month", "type": "temporal", "label": "Month"},
                "y": {"field": "rmse", "type": "quantitative", "label": "RMSE (°C)"},
                "color": {"field": "method", "type": "nominal", "label": "Method"},
            },
        },
        {
            "id": "max_round_chart",
            "title": "Target-proxy RMSE by boosting round",
            "subtitle": "Three relevant seasonal blocks, 69,850 rows; lower is better and 400 rounds is the minimum checkpoint.",
            "type": "line",
            "dataset": "max_round_curve",
            "sourceId": "max_round_sql",
            "valueFormat": "number",
            "encodings": {
                "x": {"field": "round", "type": "quantitative", "label": "Boosting round"},
                "y": {"field": "router_rmse", "type": "quantitative", "label": "RMSE (°C)"},
                "tooltip": [
                    {"field": "phase_rmse", "type": "quantitative", "label": "Phase RMSE"},
                    {"field": "state_rmse", "type": "quantitative", "label": "State RMSE"},
                ],
            },
        },
    ]

    tables = [
        {
            "id": "screen_table",
            "title": "Seven-method first screen",
            "subtitle": "Three predeclared proxy windows; lower RMSE is better.",
            "dataset": "method_screen",
            "sourceId": "screen_sql",
            "columns": [
                {"field": "block", "label": "Block", "type": "text"},
                {"field": "method", "label": "Method", "type": "text"},
                {"field": "rmse", "label": "RMSE (°C)", "type": "number"},
                {"field": "bias", "label": "Bias (°C)", "type": "number"},
            ],
        },
        {
            "id": "method_decisions",
            "title": "Method decisions and failure modes",
            "subtitle": "Literature fit was necessary but not sufficient; local blocked validation controls adoption.",
            "dataset": "decisions",
            "sourceId": "decisions_sql",
            "columns": [
                {"field": "method", "label": "Method", "type": "text"},
                {"field": "decision", "label": "Decision", "type": "text"},
                {"field": "evidence", "label": "Local evidence", "type": "text"},
                {"field": "interpretation", "label": "Interpretation", "type": "text"},
            ],
        },
        {
            "id": "phase_comparison",
            "title": "Local M2 amplitude/phase one-shot result",
            "subtitle": "Eight fixed seasonal blocks; negative delta favors the phase candidate.",
            "dataset": "phase_comparison",
            "sourceId": "phase_sql",
            "columns": [
                {"field": "block", "label": "Block", "type": "text"},
                {"field": "current_rmse", "label": "Current RMSE (°C)", "type": "number"},
                {"field": "phase_rmse", "label": "Phase RMSE (°C)", "type": "number"},
                {
                    "field": "delta_rmse",
                    "label": "Delta RMSE (°C)",
                    "type": "number",
                    "semantic": "movement",
                },
            ],
        },
        {
            "id": "tuning_comparison",
            "title": "Bounded LightGBM tuning guard result",
            "subtitle": "Four frozen guard blocks; negative delta favors the tuned candidate.",
            "dataset": "tuning_comparison",
            "sourceId": "tuning_sql",
            "columns": [
                {"field": "block", "label": "Guard block", "type": "text"},
                {"field": "current_rmse", "label": "Current RMSE (°C)", "type": "number"},
                {"field": "tuned_rmse", "label": "Tuned RMSE (°C)", "type": "number"},
                {
                    "field": "delta_rmse",
                    "label": "Delta RMSE (°C)",
                    "type": "number",
                    "semantic": "movement",
                },
            ],
        },
        {
            "id": "state_comparison",
            "title": "Mixed/stratified expert one-shot result",
            "subtitle": "Eight fixed seasonal blocks; negative delta favors the state-conditioned arm.",
            "dataset": "state_comparison",
            "sourceId": "state_sql",
            "columns": [
                {"field": "block", "label": "Validation block", "type": "text"},
                {"field": "current_rmse", "label": "Current RMSE (°C)", "type": "number"},
                {"field": "state_rmse", "label": "State RMSE (°C)", "type": "number"},
                {
                    "field": "delta_rmse",
                    "label": "Delta RMSE (°C)",
                    "type": "number",
                    "semantic": "movement",
                },
            ],
        },
        {
            "id": "score_comparison",
            "title": "Official-RMSE candidate comparison",
            "subtitle": "All exposed blocks versus the same-season and adjacent target proxies; lower is better.",
            "dataset": "score_comparison",
            "sourceId": "score_sql",
            "columns": [
                {"field": "scope", "label": "Evaluation scope", "type": "text"},
                {"field": "rows", "label": "Rows", "type": "number"},
                {"field": "current_rmse", "label": "Current RMSE (°C)", "type": "number"},
                {"field": "phase_rmse", "label": "Phase RMSE (°C)", "type": "number"},
                {"field": "state_rmse", "label": "State RMSE (°C)", "type": "number"},
                {"field": "router_rmse", "label": "Router RMSE (°C)", "type": "number"},
            ],
        },
        {
            "id": "score_candidates",
            "title": "Validated score candidate files",
            "subtitle": "Three local-only 26,061-row files; none has been uploaded.",
            "dataset": "score_candidates",
            "sourceId": "score_candidates_sql",
            "columns": [
                {"field": "candidate", "label": "Candidate", "type": "text"},
                {"field": "rows", "label": "Rows", "type": "number"},
                {"field": "sha256", "label": "SHA-256", "type": "text"},
            ],
        },
        {
            "id": "max_round_candidates",
            "title": "Validated convergence candidates",
            "subtitle": "Selected 400-round and diagnostic 5,000-round files; neither has been uploaded.",
            "dataset": "max_round_candidates",
            "sourceId": "max_round_candidates_sql",
            "columns": [
                {"field": "candidate", "label": "Candidate", "type": "text"},
                {"field": "round", "label": "Boosting round", "type": "number"},
                {"field": "rows", "label": "Rows", "type": "number"},
                {"field": "sha256", "label": "SHA-256", "type": "text"},
            ],
        },
        {
            "id": "heavy_model_priorities",
            "title": "Structural and heavy-model priority",
            "subtitle": "Qualitative P2 fit from primary literature; these are not local RMSE results.",
            "dataset": "heavy_model_priorities",
            "sourceId": "heavy_model_sql",
            "columns": [
                {"field": "rank", "label": "Rank", "type": "number"},
                {"field": "candidate", "label": "Candidate", "type": "text"},
                {"field": "structural_fit", "label": "P2 structural fit", "type": "text"},
                {"field": "compute", "label": "Compute", "type": "text"},
                {"field": "evidence_limit", "label": "Evidence boundary", "type": "text"},
                {"field": "decision", "label": "Decision", "type": "text"},
            ],
        },
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": "# P2 연직 수온 구조 복원 방법 정찰"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": (
                "## Technical Summary\n\n"
                "공개층 수온의 **6시간 반주기와 12.42시간 M2 전후 변화 20개**를 추가한 "
                "LightGBM은 8개 유효 계절 블록 중 7개에서 V0를 개선했다. 정확한 2024년 "
                "9–10월만 소폭 악화했기 때문에 가중치를 탐색하지 않고 V0와 50:50으로 평균했다. "
                f"그 결과 {sum(int(v['rows']) for v in stability.values()):,}행에서 모든 8개 블록이 "
                f"개선됐고 aggregate RMSE는 {v0_rmse:.4f}→{blend_rmse:.4f}°C였다. "
                "후속 사전등록 local-M2 amplitude/phase arm은 평균 RMSE를 더 낮췄지만 "
                "계절 안정성 2개 게이트를 위반해 기각했다. 이어서 shared/layerwise 구조와 40회 "
                "LightGBM 탐색을 수행했으나 guard RMSE가 0.7939→0.8026°C로 악화해 역시 기각했다. "
                "마지막으로 공개 layer 1–5 수온차의 fold-train 분위수만으로 lean arm을 혼합·성층 "
                "전문가로 나눈 후보는 aggregate RMSE를 1.1234→1.0997°C로 낮췄지만, 가림 직전 "
                "2025년 7–8월이 +0.0079°C 악화해 고정 +0.005°C 안전 한도를 넘었다. "
                "공식 RMSE만을 최적화하는 후속 단계에서는 개별 블록 veto를 제거하고, 같은 계절과 "
                "가림 전후 69,850행에서 layer 2·3은 phase, layer 4는 상태조건 arm을 쓰는 router를 "
                "선택했다. 이 구조는 관련 proxy RMSE 0.7889°C와 leave-one-block-out 0.7961°C로 "
                "두 단일 arm보다 낮았다. "
                "동일 모델을 5,000 boosting round까지 연장한 수렴 실험에서는 400 round가 "
                "0.7889°C로 최저였고 5,000 round는 0.8665°C로 악화했다. 따라서 현재 "
                "learning rate 0.04에서 400 round를 유지한다. "
                "추가 문헌 정찰에서는 단순 대형 Transformer보다 수직 depth query, 양방향 61일 문맥, "
                "multi-scale M2 encoder를 결합한 residual network를 다음 1순위로 선정했다. "
                "ImputeFormer는 두 번째 deterministic benchmark, SSSD-S4/CSDI는 가장 무거운 "
                "상한선 후보로 두되 아직 로컬 deep RMSE는 없다. "
                "이는 hidden test 점수가 아니며 "
                "세 파일 모두 제출 형식으로 동결했지만 업로드하지 않았다."
            ),
        },
        {
            "id": "headline_strip",
            "type": "metric-strip",
            "cardIds": ["aggregate_delta", "block_wins", "ci_high", "candidate_rows"],
        },
        {
            "id": "stability_finding",
            "type": "markdown",
            "sourceId": "p2_result",
            "body": (
                "## 고정 50:50 앙상블은 계절 전이 위험을 줄였다\n\n"
                f"Lean M2 단독은 2024년 같은 계절에서 RMSE가 "
                f"{sep_oct['v0']['rmse']:.4f}→{sep_oct['lean_m2']['rmse']:.4f}°C로 악화했다. "
                f"50:50 평균은 {sep_oct['blend50']['rmse']:.4f}°C로 오히려 V0보다 낮았다. "
                "다른 계절의 큰 동역학 이득을 유지하면서 동일 계절의 분포 이동을 완화한 것이다."
            ),
        },
        {"id": "stability_visual", "type": "chart", "chartId": "stability_chart"},
        {
            "id": "month_finding",
            "type": "markdown",
            "sourceId": "p2_result",
            "body": (
                "## M2 동역학의 이득은 대부분의 관측 월에 재현됐다\n\n"
                "Lean M2는 관측 가능한 15개월 중 13개월에서 개선됐고 2024년 9월과 10월에서만 "
                "악화했다. 50:50 평균은 그 두 달도 개선했다. 이 패턴은 조석 특징을 월 고정 규칙으로 "
                "사용하라는 뜻이 아니라, 계절에 따라 달라지는 오차 공분산을 단일 모델 교체보다 "
                "보수적 앙상블이 더 안정적으로 흡수한다는 증거다."
            ),
        },
        {"id": "monthly_visual", "type": "chart", "chartId": "monthly_chart"},
        {
            "id": "phase_finding",
            "type": "markdown",
            "sourceId": "phase_result",
            "body": (
                "## 국소 M2 진폭·위상은 평균 이득에도 계절 안정성 때문에 기각했다\n\n"
                "고정 7일 공개층 조화 특징 20개를 추가한 one-shot 후보는 166,268행 aggregate "
                "RMSE를 1.1234→1.0787°C로 낮췄고 paired KST-day bootstrap 90% CI도 "
                "[-0.0551, -0.0347]°C였다. 그러나 8개 블록 중 5개만 개선했고 2025년 "
                "3–4월은 +0.0469°C 악화해 사전등록된 최대 +0.02°C 블록 회귀 한도를 넘었다. "
                "따라서 평균 성능만 보고 창·가중치를 재탐색하지 않고 계열을 종료했다."
            ),
        },
        {"id": "phase_table_block", "type": "table", "tableId": "phase_comparison"},
        {
            "id": "tuning_finding",
            "type": "markdown",
            "sourceId": "tuning_result",
            "body": (
                "## 최적 파라미터는 개발 구간에 맞았지만 guard로 전이되지 않았다\n\n"
                "층별 모델은 shared 모델보다 개발 RMSE가 0.0017°C 높아 탈락했다. shared 구조의 "
                "40-trial Optuna 탐색은 개발 score-month RMSE를 1.6402→1.5834°C로 낮췄고, "
                "최대 5,000 round early stopping에서 블록별 best iteration은 91·269·1,249·2,038이었다. "
                "median 759 round와 최적 파라미터를 고정한 guard에서는 RMSE가 "
                "0.7939→0.8026°C로 +0.0087°C 악화했고 90% CI도 [+0.0048, +0.0127]°C였다. "
                "따라서 단일 epoch와 파라미터가 계절별 최적점을 안정적으로 대표하지 못한다고 판단한다."
            ),
        },
        {"id": "tuning_table_block", "type": "table", "tableId": "tuning_comparison"},
        {
            "id": "state_finding",
            "type": "markdown",
            "sourceId": "state_result",
            "body": (
                "## 상태조건 전문가는 평균 이득을 냈지만 가림 직전 안정성에서 기각됐다\n\n"
                "표층–layer 5 공개 수온차의 fold-train q40·q60만 사용해 lean M2를 두 전문가로 "
                "나눈 one-shot 후보는 166,268행 aggregate RMSE를 1.1234→1.0997°C로 낮췄고, "
                "paired KST-day bootstrap 90% CI는 [-0.0300, -0.0173]°C였다. 8개 블록 중 "
                "6개가 개선되고 모든 층도 개선됐지만, hidden 구간에 가장 인접한 2025년 7–8월이 "
                "+0.0079°C 악화해 사전등록된 +0.005°C veto를 넘었다. 분위수·전문가 overlap·blend "
                "weight를 사후 조정하지 않고 이 정확한 계열을 종료한다."
            ),
        },
        {"id": "state_table_block", "type": "table", "tableId": "state_comparison"},
        {
            "id": "score_finding",
            "type": "markdown",
            "sourceId": "score_result",
            "body": (
                "## 공식 RMSE 중심 선택은 layer 2·3 phase, layer 4 state를 택했다\n\n"
                "개별 계절 회귀를 탈락 조건에서 제외하고, hidden 구간과 직접 연결되는 2024년 "
                "9–10월·2025년 7–8월·11–12월의 69,850행 pooled RMSE만 최소화했다. 8개 가능한 "
                "층별 phase/state router 중 layer 2·3=phase, layer 4=state가 0.7889°C로 phase "
                "0.8064°C와 state 0.7982°C를 모두 앞섰다. 관련 3블록 leave-one-out에서도 "
                "0.7961°C였고 phase 대비 paired KST-day bootstrap 차이는 -0.0175°C, 90% CI "
                "[-0.0230, -0.0122]°C였다. 이는 hidden 정답을 보지 않은 target-proxy 최적화이며 "
                "공식 점수 자체는 제출 전 알 수 없다."
            ),
        },
        {"id": "score_table_block", "type": "table", "tableId": "score_comparison"},
        {"id": "score_candidates_block", "type": "table", "tableId": "score_candidates"},
        {
            "id": "max_round_finding",
            "type": "markdown",
            "sourceId": "max_round_result",
            "body": (
                "## 5,000라운드 학습은 400라운드 이후 과적합을 확인했다\n\n"
                "LightGBM 파라미터·특징·층별 router·학습 행은 고정하고 boosting horizon만 "
                "400에서 5,000으로 늘렸다. 목표 계절 3블록 69,850행에서 router RMSE는 "
                "400라운드 0.788890°C가 14개 체크포인트 중 최저였고, 600라운드부터 상승해 "
                "5,000라운드에서 0.866540°C가 됐다. 최대 학습은 수렴 부족을 해결하지 않았으며 "
                "일반화 오차를 0.077651°C 악화했다. 400라운드 예측은 기존 frozen OOF와 최대 "
                "절대오차 0으로 일치했으므로 비교는 같은 모델 경로의 순수 epoch ablation이다."
            ),
        },
        {"id": "max_round_visual", "type": "chart", "chartId": "max_round_chart"},
        {
            "id": "max_round_interpretation",
            "type": "markdown",
            "sourceId": "max_round_result",
            "body": (
                "400라운드는 단순한 임의 상한이 아니라 이 검증 범위에서의 최적 checkpoint다. "
                "5,000라운드 파일도 요청대로 진단 후보로 보존했지만, 검증 RMSE가 악화했으므로 "
                "제출 우선순위는 400라운드 router다. hidden 2025년 9–10월 정답을 보지 않았기 "
                "때문에 공식 RMSE 자체를 보장하지는 않는다."
            ),
        },
        {"id": "max_round_candidates_block", "type": "table", "tableId": "max_round_candidates"},
        {
            "id": "heavy_model_conclusion",
            "type": "markdown",
            "sourceId": "heavy_model_scout",
            "body": (
                "## 더 무거운 모델보다 먼저 바꿔야 할 것은 수직·시간 구조다\n\n"
                "문헌과 P2의 결측 형태를 대조한 1순위는 **Depth-query Bidirectional "
                "Multi-scale Residual Network**다. 공개 layer 1·5·6·7·8의 동시각 수온·염분·실제 "
                "수심을 수직 encoder로 읽고, 양방향 dilated TCN/TimeMixer++ 계열이 61일 blackout "
                "전후와 12.42시간 조석을 함께 처리한 뒤, DeepONet식 depth query가 7.04·9.44·14.74 m "
                "잔차를 공동 출력한다. 최종 온도는 현재 선형보간값에 이 잔차를 더한다. 이는 단순히 "
                "Transformer 크기를 키우는 것보다 P2의 수온약층 곡률·긴 연속 결측·불규칙 수심에 "
                "맞는 출발점이다. 아직 로컬 deep RMSE는 없으므로 제안과 검증 결과를 혼동하지 않는다."
            ),
        },
        {"id": "heavy_model_table_block", "type": "table", "tableId": "heavy_model_priorities"},
        {
            "id": "heavy_model_evidence",
            "type": "markdown",
            "sourceId": "heavy_model_scout",
            "body": (
                "## 직접 근거와 P2 적용 경계\n\n"
                "[LSTI](https://openreview.net/forum?id=9NVJ0ZgEfT)는 긴 연속 결측에 forward/backward "
                "예측과 consistency를 쓰며, [DeepONet](https://doi.org/10.1038/s42256-021-00302-5)은 "
                "관측 함수와 출력 좌표를 분리해 연속 operator를 학습한다. "
                "[TimeMixer++](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2b187165e28fdfdc0ffb34d1bfff2b0c-Abstract-Conference.html)와 "
                "[ModernTCN](https://openreview.net/forum?id=vpJMJerXHU)은 multi-scale 시간·주파수 및 "
                "긴 convolution 문맥의 근거다. 다만 이들의 일반 imputation benchmark는 P2의 "
                "8,784-step 동시 3층 blackout과 같지 않다. 논문 개선율을 P2 기대 RMSE로 옮기지 않고 "
                "동일 69,850행 target proxy에서 새로 비교한다."
            ),
        },
        {
            "id": "heavy_model_ceiling",
            "type": "markdown",
            "sourceId": "heavy_model_scout",
            "body": (
                "## 무거운 상한선은 ImputeFormer 다음 diffusion 순서가 합리적이다\n\n"
                "[ImputeFormer](https://doi.org/10.1145/3637528.3671751)는 공식 block-missing 구현과 "
                "low-rank/deep 표현의 절충이 있어 두 번째 benchmark로 적합하다. 다만 로컬 rank-3 EOF "
                "실패와 thermocline smoothing 위험 때문에 depth encoding과 baseline residual head가 필요하다. "
                "[SSSD-S4](https://openreview.net/forum?id=hHiIbk7ApW)와 "
                "[CSDI](https://proceedings.neurips.cc/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html)는 "
                "blackout/conditional diffusion의 높은 표현 상한을 제공하지만 sampling 비용이 크고 P2는 "
                "확률 점수가 아니라 단일 RMSE다. 따라서 posterior sample 평균이 deterministic model과 "
                "상보적인 경우에만 마지막 ensemble 후보로 올린다."
            ),
        },
        {
            "id": "heavy_model_protocol",
            "type": "markdown",
            "sourceId": "heavy_model_scout",
            "body": (
                "## 첫 deep 실험 계약\n\n"
                "128 hidden, 8 temporal blocks, kernel 7, dilation 1–128의 약 3–8M parameter 모델을 "
                "RTX 5090 bf16으로 학습한다. AdamW learning rate {1e-4, 3e-4, 1e-3} × weight decay "
                "{1e-4, 1e-3}의 6개 조합을 최대 300 epoch·patience 30으로 screen하고, 선택된 한 구조만 "
                "3 seeds로 재학습한다. 61일 창은 공개층과 정답이 완전한 사례가 없으므로 관측 정답에만 "
                "masked MSE를 적용하고, 목표 세 층을 같은 중앙 구간에서 6시간·1일·7일·30일·61일로 "
                "함께 가린다. 최종 비교는 current 400-round router 0.7888895064°C와 standalone 및 "
                "inner-selected convex blend 모두에서 수행한다. 마지막 epoch가 아니라 최저 validation "
                "RMSE checkpoint를 복원한다."
            ),
        },
        {
            "id": "screen_finding",
            "type": "markdown",
            "sourceId": "p2_result",
            "body": (
                "## 단순 곡선 보간과 전역 EOF는 채택 근거가 없었다\n\n"
                "PCHIP은 2024년 동일 계절에서만 선형보간을 조금 개선했고 2025년 proxy에서 악화했다. "
                "Rank-3 EOF는 2025년 7–8월 8.06°C까지 붕괴했다. 연도별 센서 수심 재배치와 강한 "
                "계절 비정상성 때문에 하나의 전역 저랭크 공간이 맞지 않았다. Ridge는 선형보간보다 "
                "낫지만 LightGBM보다 일관되게 약했다."
            ),
        },
        {"id": "screen_table_block", "type": "table", "tableId": "screen_table"},
        {
            "id": "literature_mapping",
            "type": "markdown",
            "body": (
                "## 문헌 근거와 로컬 적용\n\n"
                "[DINEOF](https://orbi.uliege.be/handle/2268/4291)는 불완전 해양자료를 EOF로 채우고 "
                "모드 수를 교차검증하는 근거를 제공하며, [multivariate DINEOF](https://doi.org/10.1029/2006JC003660)는 "
                "보조변수와 lagged field의 가치를 보였다. 그러나 이 데이터에서는 전역 EOF가 실패했다. "
                "[UTide](https://www.po.gso.uri.edu/~codiga/utide/2011Codiga-UTide-Report.pdf)는 정확한 시간과 "
                "조화성분·강건 회귀를 다루며, [Yellow Sea M2 연구](https://doi.org/10.1029/2018JC014819)는 "
                "12–18 m pycnocline의 10분 관측에서 M2 내부조석 진동을 확인했다. "
                "[South Yellow Sea LightGBM+tide 연구](https://doi.org/10.3389/fmars.2022.1075938)는 "
                "조석 결합이 thermocline 아래 복원을 개선할 수 있음을 보였다. 이 근거 때문에 시간동역학을 "
                "추가했지만, 최종 채택은 논문 성능이 아니라 로컬 blocked validation으로 결정했다."
            ),
        },
        {"id": "decision_table_block", "type": "table", "tableId": "method_decisions"},
        {
            "id": "scope_methods",
            "type": "markdown",
            "sourceId": "p2_contract",
            "body": (
                "## 범위, 데이터, 지표와 검증 설계\n\n"
                "대상은 S-ORS layer 2·3·4의 10분 수온이며, temp가 관측된 학습 행에서만 RMSE를 계산했다. "
                "모든 모델 입력은 공개 layer 1·5·6·7·8의 temp·psal·depth, 목표 공칭수심, 시간 특징으로 "
                "제한했다. 가림층 temp·psal은 특징 생성 API에서 금지한다. 2025년 9–10월은 어떠한 "
                "로컬 모델 선택에도 사용하지 않았고, 외부 관측값도 사용하지 않았다."
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "sourceId": "p2_result",
            "body": (
                "## 한계와 강건성\n\n"
                "8개 블록과 391 KST-day의 bootstrap은 hidden 2025년 9–10월 정답을 대체하지 않는다. "
                "운영 결측으로 11개 예정 블록 중 8개만 평가 가능했다. 같은 데이터로 방법을 반복 비교하면 "
                "연구자 과적합이 누적되므로, 50:50 이후 가중치 탐색은 금지했다. Lean test 특징 20개의 "
                f"최소 finite rate는 {candidate['test_feature_finite_rate_min']:.1%}이며 LightGBM missing routing에 "
                "의존하는 시각이 남아 있다. 상태조건 실험도 동일한 노출 블록을 재사용했으므로 "
                "강한 평균 개선을 독립 일반화 증거로 해석하지 않는다."
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## 권장 다음 단계\n\n"
                "1. layer 2·3 phase, layer 4 state의 400-round router를 첫 공식 점수 후보로 유지한다.\n"
                "2. Depth-query BiTCN residual hybrid의 6개 optimizer 조합을 최대 300 epoch로 screen한다.\n"
                "3. 선택된 deep 구조 하나만 3 seeds로 재학습하고 current router와 OOF convex blend를 비교한다.\n"
                "4. 이 구조가 개선될 때만 동일 masking 계약의 ImputeFormer를 두 번째 benchmark로 실행한다.\n"
                "5. diffusion은 deterministic deep과 오류 상관이 낮을 때 posterior mean ensemble로만 시험한다.\n"
                "6. 기각된 M2·LightGBM·state gate 파라미터 계열은 다시 열지 않는다.\n"
                "7. 5,000-round 파일은 과적합 진단용으로만 보존한다.\n"
                "8. 저장 모델 재추론과 모든 CSV의 26,061행·SHA 검증 결과를 유지한다.\n"
                "9. 정확한 CSV와 SHA를 사용자 승인하기 전에는 업로드하지 않는다."
            ),
        },
        {
            "id": "questions",
            "type": "markdown",
            "body": (
                "## 남은 질문\n\n"
                "- 리더보드 분할이 시간·층·무작위 행 중 어떤 방식인지 공개되는가?\n"
                "- 태풍 통과 시각이나 바람 관측이 대회 허용 입력으로 제공되는가?\n"
                "- 최종 모델 패키지가 양방향 public-layer 문맥 사용을 허용하는가?"
            ),
        },
    ]

    decisions = [
        {
            "method": "PCHIP vertical profile",
            "decision": "Reject",
            "evidence": "Worsened both 2025 proxy blocks",
            "interpretation": "Deep public levels do not identify upper-thermocline curvature reliably",
        },
        {
            "method": "Rank-3 EOF",
            "decision": "Reject",
            "evidence": "1.92–8.06°C RMSE in first screen",
            "interpretation": "Global low-rank basis is unstable under seasonal and deployment shifts",
        },
        {
            "method": "Layerwise Ridge residual",
            "decision": "Keep as sanity check",
            "evidence": "Better than linear interpolation in 2/3 blocks",
            "interpretation": "Public T–S state contains signal but the mapping is nonlinear",
        },
        {
            "method": "Full 120-feature dynamics",
            "decision": "Do not promote",
            "evidence": "Improved 2025 proxies but slightly worsened 2024 same season",
            "interpretation": "Too many correlated lag/lead choices for the evidence budget",
        },
        {
            "method": "Lean M2 20 features",
            "decision": "Useful arm",
            "evidence": "7/8 seasonal blocks improved; aggregate ΔRMSE -0.1093°C",
            "interpretation": "M2-scale public-layer dynamics carry transferable information",
        },
        {
            "method": "Fixed 50:50 V0 + Lean M2",
            "decision": "Research candidate",
            "evidence": "8/8 blocks improved; bootstrap 90% upper bound < 0",
            "interpretation": "Best current robustness/complexity trade-off; hidden score unknown",
        },
        {
            "method": "Local M2 amplitude/phase 20 features",
            "decision": "Reject and close family",
            "evidence": "Aggregate -0.0447°C but only 5/8 block wins; worst block +0.0469°C",
            "interpretation": "Average gain did not satisfy the preregistered seasonal-stability contract",
        },
        {
            "method": "Shared/layerwise + 40-trial LightGBM tuning",
            "decision": "Reject and close generation",
            "evidence": "Dev -0.0568°C; guard +0.0087°C with positive 90% CI",
            "interpretation": "Development optimum and median 759 rounds did not transfer across seasons",
        },
        {
            "method": "Mixed/stratified lean-M2 experts",
            "decision": "Reject and close family",
            "evidence": "Aggregate -0.0238°C and 6/8 block wins, but 2025 Jul–Aug +0.0079°C",
            "interpretation": "The training-only public contrast gate helped on average but missed the pre-gap safety veto",
        },
        {
            "method": "Layer 2/3 phase + layer 4 state router",
            "decision": "Freeze as primary score candidate",
            "evidence": "Target-proxy RMSE 0.7889°C; LOBO 0.7961°C; 90% CI versus phase entirely below zero",
            "interpretation": "Layer-specific error structure improves the metric most relevant to the hidden Sep-Oct transition",
        },
        {
            "method": "Maximum 5,000 boosting rounds",
            "decision": "Reject maximum; retain 400 rounds",
            "evidence": "Target-proxy RMSE 0.7889 at round 400 versus 0.8665 at round 5,000",
            "interpretation": "The incumbent learning rate converges before the maximum and later trees overfit",
        },
    ]
    next(source for source in sources if source["id"] == "decisions_sql")["query"]["sql"] = (
        _union_sql(decisions, ("method", "decision", "evidence", "interpretation"))
    )

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "P2 연직 수온 구조 복원 방법 정찰",
            "description": "프로파일·M2 기반 로컬 검증과 구조적·대형 시계열 모델 문헌을 P2 결측 계약에 매핑한 기술 보고서",
            "generatedAt": generated,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated,
            "status": "ready",
            "datasets": {
                "headline": [headline],
                "stability": stability_rows,
                "monthly": month_rows,
                "method_screen": screen_rows,
                "phase_comparison": phase_rows,
                "tuning_comparison": tuning_rows,
                "state_comparison": state_rows,
                "score_comparison": score_rows,
                "score_candidates": score_candidate_rows,
                "max_round_curve": max_round_rows,
                "max_round_candidates": max_round_candidate_rows,
                "heavy_model_priorities": heavy_model_rows,
                "decisions": decisions,
            },
        },
        "sources": [
            {
                "id": source["id"],
                **({"path": source["path"]} if "path" in source else {}),
                **({"href": source["href"]} if "href" in source else {}),
            }
            for source in sources
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--phase-result", type=Path)
    parser.add_argument("--tuning-result", type=Path)
    parser.add_argument("--state-result", type=Path)
    parser.add_argument("--score-result", type=Path)
    parser.add_argument("--max-round-result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
    phase_result = (
        json.loads(args.phase_result.read_text(encoding="utf-8")) if args.phase_result else None
    )
    tuning_result = (
        json.loads(args.tuning_result.read_text(encoding="utf-8")) if args.tuning_result else None
    )
    state_result = (
        json.loads(args.state_result.read_text(encoding="utf-8")) if args.state_result else None
    )
    score_result = (
        json.loads(args.score_result.read_text(encoding="utf-8")) if args.score_result else None
    )
    max_round_result = (
        json.loads(args.max_round_result.read_text(encoding="utf-8"))
        if args.max_round_result
        else None
    )
    artifact = build_artifact(
        result,
        candidate,
        phase_result,
        tuning_result,
        state_result,
        score_result,
        max_round_result,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
