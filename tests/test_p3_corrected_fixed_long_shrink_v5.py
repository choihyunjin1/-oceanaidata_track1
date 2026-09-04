from __future__ import annotations

import inspect
from pathlib import Path

import scripts.run_p3_corrected_fixed_long_shrink_v5_full_refit as runner


def test_v5_has_actual_fit_calls_before_candidate_inference() -> None:
    source = inspect.getsource(runner._fit_full_models)
    assert source.count(".fit(") == 2
    assert ".fit(" in inspect.getsource(runner._fit_router)
    run_source = inspect.getsource(runner.run_experiment)
    assert run_source.index("_fit_full_models") < run_source.index("_infer_candidate")


def test_v5_api_has_no_config_output_or_hyperparameter_override() -> None:
    assert set(inspect.signature(runner.run_experiment).parameters) == {"root", "data_dir"}


def test_v5_config_and_output_are_append_only_identities() -> None:
    root = Path(__file__).resolve().parents[1]
    assert runner.v4.sha256_file(root / runner.CONFIG_RELATIVE) == runner.CONFIG_SHA256
    assert runner.OUTPUT_RELATIVE.endswith("p3_corrected_fixed_long_shrink_v5_full_refit")
