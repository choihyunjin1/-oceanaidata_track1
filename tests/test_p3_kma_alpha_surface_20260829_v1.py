from __future__ import annotations

import numpy as np
import pandas as pd

from p3_wave.kma_alpha_surface import (
    apply_official_correction,
    crossfit_predictions,
    exhaustive_lead_surface,
    fold_robust_lead_surface,
    fit_alpha,
    make_alpha_grid,
    prepare_oof_frame,
)


def _synthetic_oof() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for fold_index, fold in enumerate(("f1", "f2", "f3")):
        for case_index in range(4):
            anchor = f"{fold}_{case_index}"
            station = ("G-ORS", "I-ORS", "S-ORS")[case_index % 3]
            for lead in (3, 6, 9, 12, 18, 24):
                base = 2.0 + 0.01 * lead + 0.02 * case_index
                delta = 0.0 if lead < 18 else (0.2 if lead == 18 else -0.25)
                alpha = 0.3 if lead == 18 else 0.7
                target = base + alpha * delta + 0.001 * (fold_index - 1)
                rows.append(
                    {
                        "fold": fold,
                        "anchor_id": anchor,
                        "station": station,
                        "lead_h": lead,
                        "incumbent_final": base,
                        "calibrated_source": base + delta,
                        "prediction": base,
                        "target_hs": target,
                    }
                )
    frame = pd.DataFrame(rows)
    blind = frame[
        [
            "fold",
            "anchor_id",
            "station",
            "lead_h",
            "incumbent_final",
            "calibrated_source",
        ]
    ].copy()
    evaluated = frame[
        ["fold", "anchor_id", "station", "lead_h", "target_hs", "prediction"]
    ].copy()
    return blind, evaluated


def test_prepare_and_exhaustive_lead_surface_recovers_known_optimum() -> None:
    blind, evaluated = _synthetic_oof()
    frame = prepare_oof_frame(blind, evaluated)
    grid = make_alpha_grid(-1.0, 2.0, 0.01)
    fitted_18 = fit_alpha(frame.loc[frame["lead_h"].eq(18)], grid)
    fitted_24 = fit_alpha(frame.loc[frame["lead_h"].eq(24)], grid)
    assert fitted_18.grid == 0.3
    assert fitted_24.grid == 0.7
    surface = exhaustive_lead_surface(frame, grid)
    assert surface["evaluated_pairs"] == 301 * 301
    assert surface["best_alpha_18"] == 0.3
    assert surface["best_alpha_24"] == 0.7
    robust = fold_robust_lead_surface(frame, grid)
    assert robust["evaluated_pairs"] == 301 * 301
    assert robust["strictly_improves_every_fold_pairs"] > 0


def test_crossfit_lead_strategy_does_not_use_heldout_fold() -> None:
    blind, evaluated = _synthetic_oof()
    frame = prepare_oof_frame(blind, evaluated)
    grid = make_alpha_grid(-1.0, 2.0, 0.01)
    prediction, fits = crossfit_predictions(frame, grid, strategy="lead")
    assert np.isfinite(prediction).all()
    assert set(fits) == {"f1", "f2", "f3"}
    short = ~frame["lead_h"].isin([18, 24]).to_numpy()
    assert np.array_equal(prediction[short], frame.loc[short, "base"].to_numpy(float))


def test_apply_official_correction_is_lead_specific_and_short_lead_exact() -> None:
    current = pd.DataFrame(
        {
            "case_id": ["C1"] * 6,
            "station": ["G-ORS"] * 6,
            "lead_h": [3, 6, 9, 12, 18, 24],
            "hs_pred": [1.0] * 6,
        }
    )
    old = current.copy()
    kma = current.copy()
    kma.loc[kma["lead_h"].eq(18), "hs_pred"] = 1.4
    kma.loc[kma["lead_h"].eq(24), "hs_pred"] = 0.6
    candidate = apply_official_correction(
        current,
        old,
        kma,
        alpha_by_lead={18: 0.2, 24: 0.6},
    )
    assert np.array_equal(candidate.loc[candidate["lead_h"].lt(18), "hs_pred"], np.ones(4))
    assert np.isclose(candidate.loc[candidate["lead_h"].eq(18), "hs_pred"].iloc[0], 1.2)
    assert np.isclose(candidate.loc[candidate["lead_h"].eq(24), "hs_pred"].iloc[0], 0.4)
