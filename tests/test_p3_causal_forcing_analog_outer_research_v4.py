from __future__ import annotations

import copy
import importlib.util
import json
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from p3_wave.causal_forcing_outer_research import (
    COMPONENT_BLIND_COLUMNS,
    PAIR_KEYS,
    FoldLibraryScope,
    FrozenOOFStageVault,
    OuterResearchError,
    TrainingTargetVault,
    attach_designated_targets,
    canonical_membership_hashes,
    compose_final_blind,
    evaluate_outer_gate,
    extract_native_20m_histories,
    read_membership_keys_only,
    sha256_file,
    validate_component_blind,
    validate_final_blind,
    validate_qa_go_receipt,
)
from p3_wave.episode_distinct_analog import LEADS

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_p3_causal_forcing_analog_outer_research_v4.py"


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("p3_outer_v4_runner_test", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the v4 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _component() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    fold_station = (
        ("2024_h2_storm", "G-ORS"),
        ("winter_transition", "I-ORS"),
        ("2025_h1", "S-ORS"),
    )
    for anchor_id, (fold, station) in enumerate(fold_station):
        for lead in LEADS:
            rows.append(
                {
                    "fold": fold,
                    "anchor_id": anchor_id,
                    "station": station,
                    "lead_h": lead,
                    "current_hs": 2.0,
                    "history_eligible": True,
                    "conditioning_used": True,
                    "fallback_reason": "",
                    "query_mad_scale": 0.2,
                    "neighbor_anchor_ids_sha256": "a" * 64,
                    "neighbor_episode_ids_sha256": "b" * 64,
                    "neighbor_distance_mean": 0.5,
                    "neighbor_distance_max": 1.0,
                    "analog_prediction": 1.0,
                }
            )
    return pd.DataFrame(rows, columns=COMPONENT_BLIND_COLUMNS).sort_values(
        list(PAIR_KEYS)
    ).reset_index(drop=True)


def _membership_contract(keys: pd.DataFrame) -> dict[str, object]:
    cases = keys[["fold", "anchor_id", "station"]].drop_duplicates()
    return {
        "expected_rows": len(keys),
        "expected_cases": len(cases),
        "expected_fold_cases": {
            str(key): int(value)
            for key, value in cases.groupby("fold", observed=True).size().items()
        },
        "expected_station_cases": {
            str(key): int(value)
            for key, value in cases.groupby("station", observed=True).size().items()
        },
        **canonical_membership_hashes(keys),
    }


def test_membership_reader_materializes_only_four_key_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    keys = _component().loc[:, PAIR_KEYS]
    source = keys.copy()
    source["prediction"] = 2.0
    source["target_hs"] = 1.0
    path = tmp_path / "oof.parquet"
    source.to_parquet(path, index=False)
    requested: list[list[str] | None] = []
    original = pd.read_parquet

    def audited_read(*args: object, **kwargs: object) -> pd.DataFrame:
        columns = kwargs.get("columns")
        requested.append(None if columns is None else list(columns))
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", audited_read)
    actual, membership, audit = read_membership_keys_only(
        path, _membership_contract(keys)
    )

    assert requested == [list(PAIR_KEYS)]
    assert list(actual.columns) == list(PAIR_KEYS)
    assert sum(len(value) for value in membership.values()) == 3
    assert audit["incumbent_prediction_values_read"] == 0
    assert audit["designated_target_values_read"] == 0


def test_staged_oof_vault_enforces_two_seals_and_one_time_reads(tmp_path: Path) -> None:
    component = _component()
    keys = component.loc[:, PAIR_KEYS]
    oof = keys.copy()
    oof["prediction"] = 2.0
    oof["target_hs"] = 1.0
    oof_path = tmp_path / "oof.parquet"
    oof.to_parquet(oof_path, index=False)
    component_path = tmp_path / "component.parquet"
    component.to_parquet(component_path, index=False)
    component_seal_path = tmp_path / "component_seal.json"
    component_seal_path.write_text(
        json.dumps(
            {
                "sealed": True,
                "stage": "analog_component_before_incumbent_prediction",
                "component_blind_sha256": sha256_file(component_path),
                "incumbent_prediction_read_count": 0,
                "designated_target_read_count": 0,
            }
        ),
        encoding="utf-8",
    )
    vault = FrozenOOFStageVault(oof_path, keys)

    with pytest.raises(PermissionError, match="component blind seal"):
        vault.read_incumbent_once()
    vault.register_component_seal(component_seal_path, component_path)
    incumbent = vault.read_incumbent_once()
    with pytest.raises(PermissionError, match="only once"):
        vault.read_incumbent_once()
    with pytest.raises(PermissionError, match="final blind seal"):
        vault.read_designated_target_once(scoring_lock_paths=[], ledger_receipt={})

    final = compose_final_blind(component, incumbent)
    final_path = tmp_path / "final.parquet"
    final.to_parquet(final_path, index=False)
    final_seal_path = tmp_path / "final_seal.json"
    final_seal_path.write_text(
        json.dumps(
            {
                "sealed": True,
                "stage": "all_final_blind_predictions_before_designated_target",
                "component_seal_sha256": sha256_file(component_seal_path),
                "final_blind_sha256": sha256_file(final_path),
                "incumbent_prediction_read_count": 1,
                "designated_target_read_count": 0,
            }
        ),
        encoding="utf-8",
    )
    vault.register_final_seal(final_seal_path, final_path)
    with pytest.raises(PermissionError, match="O_EXCL"):
        vault.read_designated_target_once(scoring_lock_paths=[], ledger_receipt={})

    first_lock = tmp_path / "first.lock"
    second_lock = tmp_path / "second.lock"
    first_lock.write_text("sealed", encoding="utf-8")
    second_lock.write_text("sealed", encoding="utf-8")
    target = vault.read_designated_target_once(
        scoring_lock_paths=[first_lock, second_lock],
        ledger_receipt={"final_seal_sha256": sha256_file(final_seal_path)},
    )

    assert target["target_hs"].tolist() == pytest.approx([1.0] * len(target))
    assert vault.incumbent_prediction_read_count == 1
    assert vault.designated_target_read_count == 1
    with pytest.raises(PermissionError, match="only once"):
        vault.read_designated_target_once(
            scoring_lock_paths=[first_lock, second_lock],
            ledger_receipt={"final_seal_sha256": sha256_file(final_seal_path)},
        )


def test_training_target_vault_blocks_current_and_future_outer_labels(tmp_path: Path) -> None:
    frame = pd.DataFrame({"anchor_id": np.arange(7, dtype=np.int64)})
    for lead in LEADS:
        frame[f"target_{lead}"] = np.arange(7, dtype=np.float64) + lead
    path = tmp_path / "anchors.parquet"
    frame.to_parquet(path, index=False)
    scopes = (
        FoldLibraryScope("f1", np.asarray([0, 1]), np.asarray([2])),
        FoldLibraryScope("f2", np.asarray([0, 1, 2, 3]), np.asarray([4])),
        FoldLibraryScope("f3", np.asarray([0, 1, 2, 3, 4, 5]), np.asarray([6])),
    )
    vault = TrainingTargetVault(path, scopes)

    actual = vault.read_library("f2", [0, 1, 2, 3])

    assert actual.shape == (4, len(LEADS))
    assert vault.access_log[-1]["allowed_prior_outer_overlap_count"] == 1
    unsafe_scopes = (
        FoldLibraryScope("f1", np.asarray([0, 1, 4]), np.asarray([2])),
        scopes[1],
        scopes[2],
    )
    unsafe = TrainingTargetVault(path, unsafe_scopes)
    with pytest.raises(PermissionError, match="current/future outer targets"):
        unsafe.read_library("f1", [0, 1, 4])


def test_native_history_extraction_is_past_only() -> None:
    times = pd.date_range("2024-01-01", periods=170, freq="20min", tz="UTC")
    values = np.arange(170, dtype=np.float64)
    wave = pd.DataFrame({"station": "G-ORS", "time": times, "hs": values})
    anchors = pd.DataFrame(
        {
            "anchor_id": [0, 1],
            "station": ["G-ORS", "G-ORS"],
            "anchor_time": [times[144], times[150]],
            "current_hs": [values[144], values[150]],
        }
    )

    expected = extract_native_20m_histories(wave, anchors)
    changed_future = wave.copy()
    changed_future.loc[151:, "hs"] = 99999.0
    actual = extract_native_20m_histories(changed_future, anchors)

    assert expected.shape == (2, 145)
    assert np.array_equal(actual, expected)
    assert expected[0].tolist() == pytest.approx(values[:145])
    assert expected[1].tolist() == pytest.approx(values[6:151])


def test_fixed_candidate_and_outer_gate_contract() -> None:
    component = _component()
    keys = component.loc[:, PAIR_KEYS]
    validate_component_blind(component, keys)
    incumbent = keys.copy()
    incumbent["incumbent_final"] = 2.0

    final = compose_final_blind(component, incumbent, alpha=0.2)
    validate_final_blind(final, keys)
    targets = keys.copy()
    targets["target_hs"] = 1.0
    evaluated = attach_designated_targets(final, targets)
    gate = evaluate_outer_gate(evaluated)

    short = final["lead_h"].isin([3, 6, 9])
    assert np.array_equal(
        final.loc[short, "candidate_final"].to_numpy(),
        final.loc[short, "incumbent_final"].to_numpy(),
    )
    assert gate["pass"] is True
    assert gate["decision"] == "PASS_OUTER_RESEARCH_REQUIRES_HIDDEN_SCORE_NO_PROMOTION"
    assert gate["promotion_permitted_without_hidden_score"] is False
    assert gate["paired_case_bootstrap"]["ci90_upper_m"] < 0.0

    corrupted = evaluated.copy()
    corrupted.loc[0, "candidate_squared_error"] += 0.1
    with pytest.raises(OuterResearchError, match="independently reproducible"):
        evaluate_outer_gate(corrupted)


def test_qa_receipt_binds_dry_and_implementation_hashes() -> None:
    implementation = {"config": "a" * 64, "runner": "b" * 64}
    receipt = {
        "experiment_id": "v4",
        "decision": "QA_GO_OUTER_V4",
        "dry_receipt_sha256": "c" * 64,
        "implementation_sha256": implementation,
        "reviewer": "independent-qa",
        "outer_model_execution_count": 0,
        "incumbent_prediction_read_count": 0,
        "designated_target_read_count": 0,
        "test_context_read_count": 0,
        "submission_write_count": 0,
    }

    validate_qa_go_receipt(
        receipt,
        experiment_id="v4",
        dry_receipt_sha256="c" * 64,
        implementation_sha256=implementation,
    )
    receipt["designated_target_read_count"] = 1
    with pytest.raises(PermissionError, match="designated_target_read_count"):
        validate_qa_go_receipt(
            receipt,
            experiment_id="v4",
            dry_receipt_sha256="c" * 64,
            implementation_sha256=implementation,
        )


@pytest.mark.parametrize("mutation", ("fold_window", "candidate_alpha"))
def test_direct_outer_call_rejects_runtime_config_mutation_before_authorization(
    mutation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    mutated = copy.deepcopy(runner._load_config(runner.CONFIG_PATH))
    if mutation == "fold_window":
        mutated["folds"]["windows"][0][1] = "1900-01-01"
    else:
        mutated["candidate"]["alpha"] = 0.3

    authorization_read = False

    def synthetic_authorization_ready() -> dict[str, object]:
        nonlocal authorization_read
        authorization_read = True
        return {"synthetic": "authorized"}

    monkeypatch.setattr(runner, "_load_authorization", synthetic_authorization_ready)
    with pytest.raises(PermissionError, match="differs from canonical config"):
        runner._outer_one_shot(
            mutated,
            tmp_path,
            confirmation=runner.OUTER_CONFIRMATION_TOKEN,
            started=time.perf_counter(),
        )

    assert authorization_read is False
