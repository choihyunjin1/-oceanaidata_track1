from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import materialize_p2_v52_score_priority_20260901_v1 as deploy  # noqa: E402


def test_v52_deployment_config_is_frozen_and_authorized() -> None:
    config = deploy.load_config()
    frozen = config["frozen_candidate"]
    assert frozen["required_status"] == "SCORE_PRIORITY_PASS_EXPLICIT_STABILITY_RISK"
    assert frozen["required_qa_status"] == "PASS"
    assert frozen["seeds"] == [20260901, 20260902, 20260903]
    assert frozen["epochs"] == 60
    assert frozen["full_history_fit_count"] == 3
    assert frozen["blend"] == {"anchor_weight": 0.8, "model_weight": 0.2}
    assert frozen["tuning"] == 0
    assert frozen["automatic_retry_count"] == 0


def test_v52_deployment_preflight_is_ready_not_materialized() -> None:
    value = deploy.preflight()
    assert value["status"] == "READY_USER_APPROVED_EXACT_V52_MATERIALIZATION"
    assert value["planned_fits"] == 3
    assert value["tuning"] == 0
    assert value["automatic_retries"] == 0
    assert value["hidden_truth_rows_read"] == 0
    assert value["score_file_rows_read"] == 0
    assert value["sample_submission_temp_values_read"] == 0
    assert value["submission_csv_created"] == 0
    assert value["uploads"] == 0


def test_v52_deployment_uses_sample_keys_not_sample_values() -> None:
    source = inspect.getsource(deploy._load_frames)
    assert 'nrows=0' in source
    assert 'usecols=KEYS' in source
    assert 'sample_submission"], dtype=' not in source


def test_v52_deployment_uses_full_history_v52_not_oof_or_v23_csv() -> None:
    source = inspect.getsource(deploy.execute)
    assert "v52.train_predict_seed" in source
    assert "MaskedThirdCentralMomentProfileVerticalDeepSet" in source
    assert "source_config = v52.load_config()" in source
    assert "prediction_commitment" not in source
    assert "P2_V23" not in source


def test_v52_deployment_is_exactly_once_and_no_retry() -> None:
    source = inspect.getsource(deploy.execute)
    assert source.index("ARTIFACT.exists()") < source.index("ARTIFACT.mkdir")
    assert source.index("output_dir.exists()") < source.index("ARTIFACT.mkdir")
    assert '"automatic_retry_count": 0' in source
    assert '"uploads": 0' in source


def test_v52_deployment_runner_has_no_external_or_pretrained_loader() -> None:
    source = deploy.RUNNER.read_text(encoding="utf-8").lower()
    forbidden = (
        "import requests",
        "from requests",
        "import urllib",
        "from urllib",
        "torch.load(",
        "load_state_dict(",
        "from_pretrained(",
        "huggingface_hub",
        "cdsapi",
        "ecmwf.datastores",
    )
    assert all(value not in source for value in forbidden)
