from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p2_restore import p2_gaussian_copula_v2_exact_frozen_submission_pack_20260830_v1 as pack

EXPECTED_CONFIG_SHA256 = "061e8667085278282d4a23d792ac04c90bef91dcbad3b32395ac00bbcfa6773f"


def test_repaired_profile_mapping_preserves_partial_profiles() -> None:
    complete_time = pd.Timestamp("2025-09-01T00:00:00Z")
    partial_time = pd.Timestamp("2025-09-01T00:10:00Z")
    query = pd.DataFrame(
        {
            "time": [complete_time] * 3 + [partial_time] * 2,
            "layer": [2, 3, 4, 2, 4],
        }
    )
    values = np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64)
    mapped = pack.repair.repaired_row_correction(
        query, pd.DatetimeIndex([complete_time]), values
    )
    np.testing.assert_allclose(mapped, [0.1, 0.2, 0.3, 0.0, 0.0])


def test_query_contract_checks_profile_support() -> None:
    test = pd.DataFrame(
        {
            "station": pd.Series(["S"] * 5, dtype="string"),
            "layer": [2, 3, 4, 2, 4],
            "time": pd.Series(["t0"] * 3 + ["t1"] * 2, dtype="string"),
        }
    )
    valued = test.assign(temp=np.arange(5, dtype=float))
    config = {
        "submission_contract": {
            "required_rows": 5,
            "layer_rows": {"2": 2, "3": 1, "4": 2},
            "complete_three_layer_times": 1,
            "partial_two_layer_times": 1,
            "partial_one_layer_times": 0,
        }
    }
    result = pack.validate_query_sources(
        test, valued, valued, valued, valued, config
    )
    assert result["rows"] == 5
    assert result["profile_times_by_target_layer_count"] == {"3": 1, "2": 1}


def test_duplicate_denylist_includes_recent_live_history() -> None:
    config = {
        "official_history_and_duplicate_denylist": {
            "older_submitted_sha256": ["old"],
            "recent_20260829": [
                {"submission_sha256": "recent", "public_rmse_c": 0.430209}
            ],
            "current_best_public_rmse_c": 0.430209,
            "current_best_points": 27.935277,
            "remaining_p2_submissions_today": 3,
            "semantic_mapping_limitation": "synthetic",
        }
    }
    assert pack.duplicate_receipt("recent", config)["exact_hash_duplicate"] is True
    assert pack.duplicate_receipt("new", config)["exact_hash_duplicate"] is False


def test_submission_staging_must_be_outside_repo(tmp_path: Path) -> None:
    with pytest.raises(pack.SubmissionPackError, match="outside the repository"):
        pack.ensure_external_output_dir(tmp_path, tmp_path / "reports" / "candidate")
    external = tmp_path.parent / f"{tmp_path.name}_external"
    assert pack.ensure_external_output_dir(tmp_path, external) == external.resolve()


def test_canonical_repository_config_hash() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = (
        repo_root
        / "configs"
        / "experiments"
        / "p2_gaussian_copula_v2_exact_frozen_submission_pack_20260830_v1.json"
    )
    assert pack.sha256_file(config_path) == EXPECTED_CONFIG_SHA256
