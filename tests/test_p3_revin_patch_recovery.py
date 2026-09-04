from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_p3_revin_patch_recovery_181 import (
    LEADS,
    PAIR_KEYS,
    SEEDS,
    FinalIntersectionTargetVault,
    atomic_json,
    atomic_parquet,
    build_intersection_artifacts,
)


def _synthetic_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    incumbent_rows = []
    blind_rows = []
    for anchor_id in range(181):
        station = ("G-ORS", "I-ORS", "S-ORS")[anchor_id % 3]
        for lead in LEADS:
            incumbent_rows.append(
                {
                    "fold": "fold",
                    "anchor_id": anchor_id,
                    "station": station,
                    "lead_h": lead,
                    "incumbent_prediction": 2.0,
                }
            )
            for seed in SEEDS:
                blind_rows.append(
                    {
                        "fold": "fold",
                        "seed": seed,
                        "anchor_id": anchor_id,
                        "station": station,
                        "episode_id": anchor_id,
                        "lead_h": lead,
                        "current_hs": 2.0,
                        "patch_prediction": 1.9 + (seed - SEEDS[0]) * 0.01,
                    }
                )
    # Replace one key on each side with the exact approved mismatch IDs.
    for lead in LEADS:
        incumbent_rows.append(
            {
                "fold": "fold",
                "anchor_id": 3739,
                "station": "G-ORS",
                "lead_h": lead,
                "incumbent_prediction": 2.0,
            }
        )
        for seed in SEEDS:
            blind_rows.append(
                {
                    "fold": "fold",
                    "seed": seed,
                    "anchor_id": 3818,
                    "station": "G-ORS",
                    "episode_id": 3818,
                    "lead_h": lead,
                    "current_hs": 2.0,
                    "patch_prediction": 1.9,
                }
            )
    return pd.DataFrame(blind_rows), pd.DataFrame(incumbent_rows)


def test_intersection_is_181_cases_and_protected_leads_are_bit_exact() -> None:
    blind, incumbent = _synthetic_frames()
    artifacts, audit = build_intersection_artifacts(blind, incumbent)
    blend = artifacts["fixed_blend_blind"]
    assert audit["intersection_rows"] == 1086
    assert blend["anchor_id"].nunique() == 181
    assert set(artifacts["candidate_only"]["anchor_id"]) == {3818}
    assert set(artifacts["incumbent_only"]["anchor_id"]) == {3739}
    protected = blend["lead_h"].isin([3, 6, 9])
    assert np.array_equal(
        blend.loc[protected, "candidate_prediction"].to_numpy(),
        blend.loc[protected, "incumbent_prediction"].to_numpy(),
    )
    assert not blend.duplicated(PAIR_KEYS).any()


def test_final_vault_requires_prior_receipt_and_forbids_third_open(tmp_path: Path) -> None:
    target = pd.DataFrame({"anchor_id": np.arange(181, dtype=np.int64)})
    for lead in LEADS:
        target[f"target_{lead}"] = 2.0
    target_path = tmp_path / "target.parquet"
    atomic_parquet(target_path, target)
    manifest_path = tmp_path / "manifest.json"
    atomic_json(manifest_path, {"sealed": True})
    from scripts.run_p3_revin_patch_recovery_181 import sha256_file

    receipt_path = tmp_path / "receipt.json"
    atomic_json(
        receipt_path,
        {
            "family_cumulative_target_open_before": 1,
            "prior_metrics_generation_count": 0,
            "approved_current_target_open_max": 1,
            "approved_current_metrics_generation_max": 1,
            "third_target_open_forbidden": True,
            "fsync_completed_before_final_target_open": True,
            "pre_open_manifest_sha256": sha256_file(manifest_path),
        },
    )
    vault = FinalIntersectionTargetVault(target_path)
    opened = vault.open_final_once(
        np.arange(181, dtype=np.int64),
        pre_open_receipt_path=receipt_path,
        pre_open_manifest_path=manifest_path,
    )
    assert len(opened) == 181
    with pytest.raises(PermissionError, match="third target"):
        vault.open_final_once(
            np.arange(181, dtype=np.int64),
            pre_open_receipt_path=receipt_path,
            pre_open_manifest_path=manifest_path,
        )
