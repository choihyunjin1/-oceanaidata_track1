from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p3_wave.catboost_confirmation_repair_v3 import (
    CONTRACT_COLUMNS,
    ConfirmationContractError,
    build_single_blind,
    validate_confirmation_contract,
    validate_selection_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiments/p3_catboost_confirmation_contract_repair_20260830_v3.json"
RUNNER_PATH = ROOT / "scripts/run_p3_catboost_confirmation_contract_repair_20260830_v3.py"
SPEC = importlib.util.spec_from_file_location("p3_confirmation_repair_v3_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _contract() -> pd.DataFrame:
    rows = []
    for lead in (3, 6, 9, 12, 18, 24):
        rows.append(
            {
                "fold": "fold_a",
                "anchor_id": 1,
                "station": "S-ORS",
                "lead_h": lead,
                "current_hs": 2.0,
                "single_prediction": 2.1,
                "multi_prediction": 2.2,
                "persistence": 2.0,
                "weight_single": 0.5,
                "weight_multi": 0.5,
                "weight_persistence": 0.0,
                "second_stage_persistence_weight": 0.2 if lead >= 12 else 0.0,
                "prediction": 2.15,
            }
        )
    return pd.DataFrame(rows, columns=list(CONTRACT_COLUMNS))


def test_config_freezes_v2_selection_and_zero_search_fits() -> None:
    config = RUNNER.load_config(CONFIG_PATH)
    assert config["frozen_v2"]["selected_candidate_id"] == "challenger_21"
    assert config["frozen_v2"]["selected_iteration"] == 138
    assert config["frozen_v2"]["selection_search_rerun_allowed"] is False
    assert config["execution"]["search_fit_count"] == 0
    assert config["confirmation"]["challenger_fit_count"] == 3
    assert config["confirmation"]["maximum_historical_fit_count"] == 6


def test_selection_receipt_is_exactly_frozen() -> None:
    config = RUNNER.load_config(CONFIG_PATH)
    selection = json.loads(
        (ROOT / config["frozen_v2"]["selection_path"]).read_text(encoding="utf-8")
    )
    audit = validate_selection_receipt(selection, config["frozen_v2"])
    assert audit["selected_candidate_id"] == "challenger_21"
    assert audit["selected_iteration"] == 138
    assert audit["selection_gate_pass"] is True


def test_confirmation_contract_requires_exact_columns_and_six_leads() -> None:
    frame = _contract()
    audit = validate_confirmation_contract(frame)
    assert audit["rows"] == 6
    assert audit["cases"] == 1
    with pytest.raises(ConfirmationContractError, match="column order"):
        validate_confirmation_contract(frame.drop(columns="current_hs"))
    with pytest.raises(ConfirmationContractError, match="exactly six leads"):
        validate_confirmation_contract(frame.iloc[:-1].copy())


def test_confirmation_contract_rejects_duplicate_and_nonfinite() -> None:
    frame = _contract()
    with pytest.raises(ConfirmationContractError, match="duplicated"):
        validate_confirmation_contract(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))
    damaged = frame.copy()
    damaged.loc[0, "single_prediction"] = np.nan
    with pytest.raises(ConfirmationContractError, match="non-finite"):
        validate_confirmation_contract(damaged)


def test_single_blind_reconstructs_exact_control_contract() -> None:
    frame = _contract()
    challenger = frame[["anchor_id", "station", "lead_h"]].copy()
    challenger["prediction"] = 2.3
    single = build_single_blind(frame, challenger)
    assert list(single.columns) == [
        "fold",
        "anchor_id",
        "station",
        "lead_h",
        "current_hs",
        "control_single_prediction",
        "challenger_single_prediction",
    ]
    assert single["control_single_prediction"].eq(2.1).all()
    assert single["challenger_single_prediction"].eq(2.3).all()


def test_single_blind_rejects_key_loss() -> None:
    frame = _contract()
    challenger = frame[["anchor_id", "station", "lead_h"]].iloc[:-1].copy()
    challenger["prediction"] = 2.3
    with pytest.raises(ConfirmationContractError, match="keys differ"):
        build_single_blind(frame, challenger)


def test_wrong_token_stops_before_preflight_or_attempt_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = RUNNER.load_config(CONFIG_PATH)
    lock = ROOT / config["outputs"]["attempt_lock"]
    before = lock.read_bytes() if lock.exists() else None

    def forbidden_preflight(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("preflight must not run for a wrong token")

    monkeypatch.setattr(RUNNER, "static_preflight", forbidden_preflight)
    with pytest.raises(ConfirmationContractError, match="authorization token differs"):
        RUNNER.execute_confirmation(CONFIG_PATH, tmp_path, "WRONG_TOKEN")
    after = lock.read_bytes() if lock.exists() else None
    assert after == before
