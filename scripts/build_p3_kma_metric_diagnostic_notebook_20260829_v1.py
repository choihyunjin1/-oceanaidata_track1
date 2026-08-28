"""Build and execute the P3 KMA local-vs-Public metric diagnostic notebook."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "reports" / "p3_kma_metric_diagnostic_20260829_v1"
NOTEBOOK_PATH = OUTPUT_DIR / "analysis.ipynb"


def markdown(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(dedent(source).strip())


def code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(dedent(source).strip())


def build_notebook() -> nbformat.NotebookNode:
    cells = [
        markdown(
            """
            # P3 KMA local OOF vs official Public metric diagnostic

            ## tl;dr

            This notebook reconciles the frozen 182-case local OOF KMA surface with the
            three recorded official Public points. It does **not** read official test
            features, sample submissions, hidden targets, or candidate CSV values.

            The controlling conclusion is that the two surfaces establish a real
            local-to-Public transport gap, but not a scalar conversion rule. The Public
            curve supports a low-dimensional KMA direction; the local cross-fit results
            reject adaptive station/lead weight fitting on the current 182-case OOF set.
            """
        ),
        markdown(
            """
            ## Context & Methods

            ### Key Assumptions

            - The committed `result.json` is the controlling local aggregate artifact.
            - The three Public RMSE values are recorded observations for the same
              frozen prediction lineage at uniform KMA alpha 0.0, 0.2, and 0.4.
            - The local and Public evaluation populations are different; absolute RMSE
              or delta-RMSE is therefore not transported between them.
            - A positive local delta-RMSE means degradation; a negative value means
              improvement.
            - Lead-isolation formulas assume the official scorer averages the same
              number of observations at each of the six horizons.
            """
        ),
        code(
            """
            from pathlib import Path
            import json
            import math

            import numpy as np
            import pandas as pd

            repo_root = Path.cwd()
            result_path = repo_root / "reports" / "p3_kma_alpha_surface_sweep_20260829_v1" / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            str(result_path.relative_to(repo_root))
            """
        ),
        markdown("## Data\n\n### 1. Validate the aggregate evidence contract"),
        code(
            """
            contract = result["data_contract"]
            checks = {
                "rows_equal_cases_times_leads": contract["rows"] == contract["cases"] * len(contract["leads"]),
                "three_temporal_folds": len(contract["folds"]) == 3,
                "three_stations": len(contract["stations"]) == 3,
                "six_leads": contract["leads"] == [3, 6, 9, 12, 18, 24],
                "official_test_context_not_read": contract["official_test_context_read"] is False,
                "hidden_target_not_read": contract["hidden_target_read"] is False,
                "no_new_model_fit": contract["new_model_fit_count"] == 0,
                "no_row_values_in_report": contract["row_level_values_written_to_report"] is False,
                "three_unique_public_alphas": len({p["uniform_alpha"] for p in result["official_public_curve"]["points"]}) == 3,
                "all_candidates_unuploaded": all(not c["uploaded"] for c in result["candidate_manifest"]["candidates"]),
                "all_candidates_have_1200_rows": all(c["rows"] == 1200 for c in result["candidate_manifest"]["candidates"]),
            }
            checks_frame = pd.DataFrame(
                [{"check": name, "passed": passed} for name, passed in checks.items()]
            )
            assert checks_frame["passed"].all(), checks_frame.loc[~checks_frame["passed"]]
            checks_frame
            """
        ),
        code(
            """
            population_frame = pd.DataFrame(
                [
                    {
                        "surface": "Local frozen OOF",
                        "cases": contract["cases"],
                        "rows": contract["rows"],
                        "temporal_groups": len(contract["folds"]),
                        "observed_alpha_points": 301,
                        "role": "historical model-selection diagnostic",
                    },
                    {
                        "surface": "Official Public",
                        "cases": 66,
                        "rows": 66 * 6,
                        "temporal_groups": np.nan,
                        "observed_alpha_points": len(result["official_public_curve"]["points"]),
                        "role": "competition feedback on a different population",
                    },
                ]
            )
            population_frame
            """
        ),
        markdown("## Results\n\n### 2. Establish the local pattern"),
        code(
            """
            uniform = result["same_row_exhaustive"]["uniform"]
            lead_surface = result["same_row_exhaustive"]["lead_surface"]
            minimax = result["same_row_exhaustive"]["fold_robust_lead_surface"]

            local_summary = pd.DataFrame(
                [
                    {
                        "estimate": "same-row uniform optimum",
                        "degrees_of_freedom": 1,
                        "alpha": f'{uniform["best_alpha"]:.2f}',
                        "delta_rmse_m": uniform["metrics"]["delta_rmse"],
                        "selection_status": "post-selection, optimistic",
                    },
                    {
                        "estimate": "same-row lead-specific optimum",
                        "degrees_of_freedom": 2,
                        "alpha": f'{lead_surface["best_alpha_18"]:.2f}/{lead_surface["best_alpha_24"]:.2f}',
                        "delta_rmse_m": lead_surface["best_rmse"] - uniform["metrics"]["base_rmse"],
                        "selection_status": "post-selection, optimistic",
                    },
                    {
                        "estimate": "posthoc fold-minimax pair",
                        "degrees_of_freedom": 2,
                        "alpha": f'{minimax["minimax_pair"]["alpha_18"]:.2f}/{minimax["minimax_pair"]["alpha_24"]:.2f}',
                        "delta_rmse_m": minimax["posthoc_minimax_metrics"]["delta_rmse"],
                        "selection_status": "post-selection; 90% CI crosses zero",
                    },
                    *[
                        {
                            "estimate": item["strategy"] + " cross-fit",
                            "degrees_of_freedom": np.nan,
                            "alpha": "trained outside held fold",
                            "delta_rmse_m": item["delta_rmse"],
                            "selection_status": "held-fold estimate",
                        }
                        for item in result["crossfit_ranked"][:5]
                    ],
                ]
            )
            local_summary.sort_values("delta_rmse_m")
            """
        ),
        code(
            """
            fold_frame = (
                pd.DataFrame.from_dict(uniform["metrics"]["by_fold"], orient="index")
                .rename_axis("fold")
                .reset_index()[["fold", "rows", "base_rmse", "candidate_rmse", "delta_rmse"]]
            )
            station_frame = (
                pd.DataFrame.from_dict(uniform["metrics"]["by_station"], orient="index")
                .rename_axis("station")
                .reset_index()[["station", "rows", "base_rmse", "candidate_rmse", "delta_rmse"]]
            )
            fold_frame, station_frame
            """
        ),
        markdown("### 3. Reproduce the official uniform-alpha pattern"),
        code(
            """
            official = pd.DataFrame(result["official_public_curve"]["points"]).sort_values("uniform_alpha")
            official["delta_rmse_vs_previous"] = official["rmse"].diff()
            official["delta_mse_vs_previous"] = official["rmse"].pow(2).diff()

            coefficients = np.polyfit(official["uniform_alpha"], official["rmse"].pow(2), deg=2)
            optimum_alpha = -coefficients[1] / (2 * coefficients[0])
            optimum_rmse = math.sqrt(np.polyval(coefficients, optimum_alpha))

            assert math.isclose(optimum_alpha, result["official_public_curve"]["quadratic_optimum_alpha"], abs_tol=1e-12)
            assert math.isclose(optimum_rmse, result["official_public_curve"]["quadratic_predicted_rmse"], abs_tol=1e-12)
            official, pd.DataFrame([{"quadratic_alpha_star": optimum_alpha, "predicted_rmse": optimum_rmse}])
            """
        ),
        code(
            """
            public_gain_0_to_02 = official.iloc[0]["rmse"] - official.iloc[1]["rmse"]
            public_gain_02_to_04 = official.iloc[1]["rmse"] - official.iloc[2]["rmse"]
            local_same_row_gain = -uniform["metrics"]["delta_rmse"]
            gap_frame = pd.DataFrame(
                [
                    {
                        "comparison": "Public uniform 0.0 -> 0.2",
                        "rmse_improvement_m": public_gain_0_to_02,
                        "selection": "official observed",
                    },
                    {
                        "comparison": "Public uniform 0.2 -> 0.4",
                        "rmse_improvement_m": public_gain_02_to_04,
                        "selection": "official observed",
                    },
                    {
                        "comparison": f'Local same-row 0.0 -> {uniform["best_alpha"]:.2f}',
                        "rmse_improvement_m": local_same_row_gain,
                        "selection": "optimistic same-row optimum",
                    },
                    {
                        "comparison": "Local best adaptive held-fold strategy",
                        "rmse_improvement_m": -result["crossfit_ranked"][0]["delta_rmse"],
                        "selection": "cross-fit",
                    },
                ]
            )
            effect_ratio = public_gain_0_to_02 / local_same_row_gain
            gap_frame, pd.DataFrame([{"Public_0_to_02_over_local_same_row_gain": effect_ratio}])
            """
        ),
        markdown("### 4. Quantify what the next orthogonal probes can identify"),
        code(
            """
            probe_frame = pd.DataFrame(
                [
                    {
                        "label": "C recorded baseline",
                        "alpha18": 0.4,
                        "alpha24": 0.4,
                        "identifies": "neither lead separately",
                    },
                    {
                        "label": "A proposed",
                        "alpha18": 0.4,
                        "alpha24": 0.6,
                        "identifies": "24h direction versus C",
                    },
                    {
                        "label": "B proposed",
                        "alpha18": 0.2,
                        "alpha24": 0.6,
                        "identifies": "18h direction versus A",
                    },
                ]
            )

            def lead24_delta_mse(overall_rmse_a: float, overall_rmse_c: float) -> float:
                return 6.0 * (overall_rmse_a**2 - overall_rmse_c**2)

            def lead18_delta_mse(overall_rmse_b: float, overall_rmse_a: float) -> float:
                return 6.0 * (overall_rmse_b**2 - overall_rmse_a**2)

            probe_frame
            """
        ),
        code(
            """
            candidate_frame = pd.DataFrame(result["candidate_manifest"]["candidates"])[
                ["id", "alpha_18", "alpha_24", "role", "rows", "changed_rows_vs_current", "uploaded"]
            ]
            candidate_frame
            """
        ),
        markdown("### 5. Driver assessment"),
        code(
            """
            driver_frame = pd.DataFrame(
                [
                    {
                        "driver": "Incumbent lineage mismatch",
                        "evidence": "Report states local OOF incumbent and official champion use different long-axis lineages",
                        "status": "verified limitation",
                        "impact": "blocks absolute delta transport",
                    },
                    {
                        "driver": "Population/regime shift",
                        "evidence": "182 historical cases in three named folds versus 66 Public cases",
                        "status": "verified population mismatch; causal regime details unresolved",
                        "impact": "can change KMA residual correlation and optimal alpha",
                    },
                    {
                        "driver": "Weight-estimation variance",
                        "evidence": "all adaptive cross-fit variants degrade; six station-lead weights are unstable across folds",
                        "status": "strongly supported",
                        "impact": "rejects high-dimensional adaptive alpha on current sample",
                    },
                    {
                        "driver": "Public sampling noise",
                        "evidence": "only 66 Public cases and no hidden row-level residuals",
                        "status": "plausible but not separately estimable",
                        "impact": "limits confidence in fine alpha optimization",
                    },
                    {
                        "driver": "Real KMA signal on Public lineage",
                        "evidence": "RMSE improves at both 0.2 and 0.4; second marginal gain remains positive",
                        "status": "verified on Public sample",
                        "impact": "supports one low-dimensional lead-isolation sequence",
                    },
                ]
            )
            driver_frame
            """
        ),
        markdown(
            """
            ## Takeaways

            1. **Do not build a local-to-Public scalar conversion.** Even the optimistic
               local same-row gain is much smaller than the observed first Public step,
               while the held-fold adaptive estimates have the opposite sign.
            2. **Treat high-dimensional alpha fitting as failed on this sample.** Every
               adaptive cross-fit variant degrades and the best 90% bootstrap interval
               lies entirely above zero delta-RMSE.
            3. **The Public KMA direction is real but low-resolution.** Three points show
               improvement and diminishing returns, but cannot identify 18h and 24h
               effects separately.
            4. **Use orthogonal probes, not a 0.425 micro-step.** Candidate A changes only
               24h relative to the recorded 0.4/0.4 baseline; candidate B then changes
               only 18h relative to A. Their sequential RMSE differences identify the
               lead-specific MSE directions under the equal-count scoring assumption.
            5. **Promotion rule:** official exploitation requires a repeated monotone
               direction or an identified lead effect; local structural candidates still
               require held-fold non-degradation and a prespecified uncertainty check.

            ### Open questions

            - The 66 Public cases' regime mix and station mix are hidden, so the precise
              source of the transport gap cannot be resolved from aggregate RMSE alone.
            - Public improvement does not establish Private generalization.
            - The third daily slot should be chosen only after A and B update the two
              lead-specific directions; it is not preregistered as an automatic upload.
            """
        ),
    ]

    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3.12"}
    return notebook


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    client = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    executed = client.execute()
    nbformat.write(executed, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
