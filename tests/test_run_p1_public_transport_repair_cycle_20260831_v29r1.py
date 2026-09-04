from scripts import run_p1_public_transport_repair_cycle_20260831_v29r1 as runner


def test_recovery_adds_only_missing_metric_field():
    amendment, repaired = runner.load_contract()
    assert amendment["repair"]["additional_fits"] == 0
    assert amendment["repair"]["prediction_changes"] == 0
    assert repaired["decision_policy"]["bootstrap_probability_improved_minimum_inclusive"] == 0.8


def test_validation_passes_without_artifact_write():
    assert runner.validate()["status"] == "PASS"
