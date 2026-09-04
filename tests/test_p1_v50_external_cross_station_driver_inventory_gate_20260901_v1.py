from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_v50_external_cross_station_driver_inventory_gate_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
RUNNER = ROOT / f"scripts/run_{EXPERIMENT_ID}.py"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"


def test_inventory_closes_external_driver_unavailable() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "EXTERNAL_DRIVER_UNAVAILABLE"
    assert config["candidate_selection"]["qualifying_families"] == 0
    assert config["candidate_selection"]["selected"] is None
    assert config["candidate_selection"]["ready"] is False
    assert config["terminal"] == "EXTERNAL_DRIVER_UNAVAILABLE"


def test_multi_station_drivers_end_before_p1_train() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    inventory = {item["family"]: item for item in config["inventory"]}
    era5 = inventory["ERA5 multi-station context pretrain"]
    kma = inventory["KMA ocean buoy meteorology and waves"]
    assert len(era5["stations"]) == 3 and era5["p1_train_overlap_rows"] == 0
    assert len(kma["station_proxies"]) == 3 and kma["p1_train_overlap_rows"] == 0
    assert era5["observed_end_utc"].startswith("2023-12-31")
    assert kma["observed_end_kst"].startswith("2023-12-31")


def test_concurrent_external_drivers_are_single_station() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    inventory = {item["family"]: item for item in config["inventory"]}
    nasa = inventory["NASA POWER hourly meteorology"]
    era5 = inventory["ERA5 S-ORS surface forcing exact historical blocks"]
    assert nasa["concurrent_station_count"] == 1
    assert era5["concurrent_station_count"] == 1
    assert nasa["stations"] == ["S-ORS"]
    assert era5["stations"] == ["S-ORS"]


def test_tide_is_reconstructible_not_external_observation() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    tide = next(item for item in config["inventory"] if item["family"].startswith("astronomical tide"))
    assert tide["external_values_used"] is False
    assert tide["nonreconstructible"] is False
    assert "time" in tide["reconstructible_from"]


def test_no_network_runner_lock_fit_or_forbidden_access() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert all(value == 0 for value in config["operations"].values())
    assert not RUNNER.exists()
    assert not LOCK.exists()


def test_v28_v33_add_only_contract_and_bounded_recipe() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    contract = config["contracts_preserved"]
    recipe = config["prospective_contract_if_a_driver_had_qualified"]
    assert contract["add_only"] is True and contract["anchor_removals"] == 0
    assert contract["maximum_fits"] == 9
    assert recipe["fits"] == 3 and recipe["maximum_fits"] == 9
    assert recipe["threshold_quantiles"] == [0.995, 0.9975, 0.999]
    assert recipe["sweep"] == 0
