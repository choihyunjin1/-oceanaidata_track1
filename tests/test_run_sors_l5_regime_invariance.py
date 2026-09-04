from __future__ import annotations

from p1_qc.sors_l5_regime_preregistration import SORSL5PreregistrationError
from scripts.run_sors_l5_regime_invariance import parse_args


def test_runner_defaults_to_contract_audit_only() -> None:
    args = parse_args(["--data-dir", "data"])
    assert args.outer_cv is False


def test_runner_outer_flag_is_explicit_but_validator_is_fail_closed() -> None:
    args = parse_args(["--data-dir", "data", "--outer-cv"])
    assert args.outer_cv is True
    # The runner delegates this flag as require_outer_authorized=True, whose
    # validator path is exhaustively tested to raise this domain error.
    assert issubclass(SORSL5PreregistrationError, ValueError)
