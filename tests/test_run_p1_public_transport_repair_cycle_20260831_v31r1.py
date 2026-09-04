import numpy as np

from scripts import run_p1_public_transport_repair_cycle_20260831_v31r1 as runner


def test_contract_is_exact_authorization_only():
    amendment, base = runner.load_contract()
    assert base["candidate"] == "P1_1_PREFIX_LOGIT_SHRUNK_LABEL_SHIFT_EM"
    assert amendment["amendment"]["model_parameter_changes"] == 0
    assert amendment["amendment"]["maximum_historical_fits"] == 2


def test_remainder_split_is_chronological_and_exhaustive():
    times = np.repeat(np.arange(8, dtype=np.int64), 2)
    remainder = np.ones(len(times), dtype=bool)
    shrink, selection, cutoff = runner.split_remainder(times, remainder)
    assert np.array_equal(shrink | selection, remainder)
    assert not np.any(shrink & selection)
    assert times[shrink].max() == cutoff
    assert times[shrink].max() < times[selection].min()


def test_validation_reports_no_external_access():
    result = runner.validate_only()
    assert result["status"] == "VALID"
    assert result["official_reads"] == 0
    assert result["hidden_truth_reads"] == 0
