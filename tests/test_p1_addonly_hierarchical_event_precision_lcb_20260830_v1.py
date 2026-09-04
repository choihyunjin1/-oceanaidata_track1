from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p1_qc import (  # noqa: E402
    p1_addonly_hierarchical_event_precision_lcb_20260830_v1 as core,
)

EXPERIMENT_ID = "p1_addonly_hierarchical_event_precision_lcb_20260830_v1"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
RUNNER = ROOT / "scripts" / f"run_{EXPERIMENT_ID}.py"
SOURCE_NAMES = ["frozen_a", "frozen_b"]


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("p1_addonly_hierarchical_test_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config() -> dict[str, Any]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _synthetic_frame(*, next_event_count: int = 5) -> pd.DataFrame:
    starts = {
        "2025_q2": pd.Timestamp("2025-04-01T00:00:00Z"),
        "2025_q3": pd.Timestamp("2025-07-01T00:00:00Z"),
        "2025_q4": pd.Timestamp("2025-10-01T00:00:00Z"),
    }
    rows: list[dict[str, Any]] = []
    for fold, start in starts.items():
        for index in range(12):
            rows.append(
                {
                    "station": "ANCHOR_TP",
                    "year": 2025,
                    "layer": 1,
                    "time": start - pd.Timedelta(hours=index + 1),
                    "fold": fold,
                    "label": int(index < 8),
                    "anchor": int(index < 10),
                    "frozen_a": 0,
                    "frozen_b": 0,
                }
            )
        event_count = next_event_count if fold == "2025_q3" else 8
        for event_index in range(event_count):
            station = "GOOD" if event_index % 2 == 0 else "WEAK"
            source = "frozen_a" if station == "GOOD" else "frozen_b"
            event_start = start + pd.Timedelta(hours=event_index * 2)
            for row_index in range(12):
                good_label = row_index < 10
                weak_label = row_index < 2
                rows.append(
                    {
                        "station": station,
                        "year": 2025,
                        "layer": 1 if station == "GOOD" else 2,
                        "time": event_start + pd.Timedelta(minutes=10 * row_index),
                        "fold": fold,
                        "label": int(good_label if station == "GOOD" else weak_label),
                        "anchor": 0,
                        "frozen_a": int(source == "frozen_a"),
                        "frozen_b": int(source == "frozen_b"),
                    }
                )
    return pd.DataFrame(rows)


def _events(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], np.ndarray]:
    return core.build_event_bank(
        frame,
        SOURCE_NAMES,
        event_group=["fold", "station", "year", "layer"],
        cadence_minutes=10,
    )


def test_preregistration_is_metric_aligned_one_shot_and_target_preserving() -> None:
    runner = _load_runner()
    config = _config()
    runner._validate_config(config)

    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == runner.CONFIG_SHA256
    assert config["decision_hierarchy"]["level_1_sole_primary"].endswith(
        "directional margin = 0"
    )
    assert set(config["decision_hierarchy"]["forbidden_hard_gates"]) == {
        "minimum event count",
        "minimum improved window count",
        "all-window improvement",
        "station concentration cap",
        "worst-slice veto",
        "post-result numeric delta",
    }
    assert config["execution_contract"]["model_fit_count"] == 2
    assert config["execution_contract"]["threshold_search_count"] == 0
    assert config["execution_contract"]["retry_or_tuning_count"] == 0
    assert config["outlier_policy"]["remove_label_1_rows_or_events"] is False
    assert config["outlier_policy"]["hard_delete_any_input_row"] is False
    assert config["outputs"]["allowed_extensions"] == [".json"]
    assert config["outputs"]["csv_files"] == 0
    assert all(config["prohibitions"].values())


def test_event_geometry_is_label_free_and_exactly_partitions_anchor_negative_bank() -> None:
    frame = _synthetic_frame()
    events, proposal_mask = _events(frame.drop(columns="label"))

    assert len(events) == 21
    assert int(proposal_mask.sum()) == sum(event["rows"] for event in events)
    assert all(event["rows"] == 12 for event in events)
    assert not any("label" in event for event in events)
    covered = [row for event in events for row in event["row_indices"]]
    assert len(covered) == len(set(covered)) == int(proposal_mask.sum())
    assert frame.iloc[covered]["anchor"].eq(0).all()


def test_next_fold_truth_cannot_change_fit_or_blind_acceptance() -> None:
    frame = _synthetic_frame(next_event_count=1)
    events, _ = _events(frame)
    config = _config()
    prefix = config["chronological_prefixes"][0]
    original, original_candidate, _ = core.evaluate_prefix(frame, events, prefix, config["head"])

    changed = frame.copy()
    next_proposals = changed["fold"].eq("2025_q3") & changed["anchor"].eq(0)
    changed.loc[next_proposals, "label"] = 1 - changed.loc[next_proposals, "label"]
    changed_result, changed_candidate, _ = core.evaluate_prefix(
        changed, events, prefix, config["head"]
    )

    assert changed_result["fit"] == original["fit"]
    assert changed_result["blind_acceptance"] == original["blind_acceptance"]
    assert np.array_equal(changed_candidate, original_candidate)
    assert changed_result["next_fold_metrics"] != original["next_fold_metrics"]
    assert original["f1_over_2_hard_sanity"]["pass"] is True
    assert original["f1_over_2_hard_sanity"]["anchor_positive_removed_rows"] == 0


def test_one_next_fold_event_is_allowed_without_an_arbitrary_support_gate() -> None:
    frame = _synthetic_frame(next_event_count=1)
    events, _ = _events(frame)
    config = _config()
    receipt, _, _ = core.evaluate_prefix(
        frame, events, config["chronological_prefixes"][0], config["head"]
    )

    assert receipt["blind_acceptance"]["events_scored"] == 1
    assert receipt["support_diagnostics"]["role"] == "DIAGNOSTIC_ONLY_NO_VETO"
    assert receipt["fit"]["model_fit_count"] == 1
    assert receipt["blind_acceptance"]["threshold_search_count"] == 0


def test_f1_over_2_addonly_identity_is_exact_at_equality() -> None:
    anchor = {"tp": 1, "fp": 0, "fn": 2, "tn": 5, "rows": 8}
    candidate = {"tp": 2, "fp": 3, "fn": 1, "tn": 2, "rows": 8}
    sanity = core.addonly_algebra_sanity(
        anchor, candidate, added_tp=1, added_fp=3, anchor_positive_removed=0
    )

    assert core.f1_from_counts(anchor)[0] == core.f1_from_counts(candidate)[0] == 0.5
    assert sanity["proposal_precision"] == sanity["anchor_f1_over_2"] == 0.25
    assert sanity["utility_numerator_exact"] == 0
    assert sanity["precision_relation_sign"] == sanity["f1_delta_sign"] == 0
    assert sanity["pass"] is True


def test_paired_bootstrap_is_deterministic_and_keeps_events_complete() -> None:
    frame = _synthetic_frame()
    events, _ = _events(frame)
    candidate = frame["anchor"].to_numpy(dtype=np.int8).copy()
    chosen = [event for event in events if event["fold"] in {"2025_q3", "2025_q4"}]
    for event in chosen:
        if event["station"] == "GOOD":
            candidate[list(event["row_indices"])] = 1
    kwargs = {
        "replicates": 200,
        "seed": 20260830,
        "lower_quantile": 0.05,
        "upper_quantile": 0.95,
    }
    first = core.paired_event_preserving_day_bootstrap(
        frame, candidate, events, ["2025_q3", "2025_q4"], **kwargs
    )
    second = core.paired_event_preserving_day_bootstrap(
        frame, candidate, events, ["2025_q3", "2025_q4"], **kwargs
    )

    assert first == second
    assert first["complete_event_assignment_verified"] is True
    assert first["paired_unit"] == "event-preserving_joint_KST_day"
    assert first["block_length_days"] >= 1
    assert first["point_candidate_minus_anchor_f1"] > 0
    assert core.evidence_state(
        first["point_candidate_minus_anchor_f1"],
        first["lower_one_sided_95"],
        first["upper_one_sided_95"],
    ).endswith("RESEARCH_ONLY")


def test_exclusive_json_creation_cannot_overwrite_a_race_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    created = tmp_path / "created.json"
    runner._atomic_json(created, {"writer": "first"})
    with pytest.raises(FileExistsError):
        runner._atomic_json(created, {"writer": "second"})
    assert json.loads(created.read_text(encoding="utf-8")) == {"writer": "first"}

    raced = tmp_path / "raced.json"
    real_link = os.link

    def rival_wins(source: Path, destination: Path) -> None:
        Path(destination).write_text('{"writer":"rival"}\n', encoding="utf-8")
        real_link(source, destination)

    monkeypatch.setattr(runner.os, "link", rival_wins)
    with pytest.raises(FileExistsError):
        runner._atomic_json(raced, {"writer": "experiment"})
    assert json.loads(raced.read_text(encoding="utf-8")) == {"writer": "rival"}
    assert list(tmp_path.glob("*.tmp")) == []


def test_runner_has_no_csv_or_unregistered_fit_entrypoint() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "read_csv(" not in source
    assert "to_csv(" not in source
    assert "--execute-one-shot" in source
    assert "--p1-dir" not in source
    assert "--input" not in source
