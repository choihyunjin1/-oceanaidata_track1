from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_p3_era5_context_transfer_static_v1.py"


def _module():
    spec = importlib.util.spec_from_file_location("p3_era5_static_validator", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_metadata_reads_json_only(tmp_path: Path) -> None:
    module = _module()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "created_at_utc": "fixed",
                "stage": "combine",
                "row_count": 262917,
                "observed_start": "2014-01-01T00:00:00+00:00",
                "observed_end": "2023-12-31T14:00:00+00:00",
                "local_file": "quarantine/derived/final.parquet",
                "file_sha256": "a" * 64,
                "selected_cells": [{}, {}, {}],
                "requests": {"selected_single_cell_years": [{}] * 363},
                "files": [],
                "official_test_or_submission_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    observed = module._manifest_metadata(manifest)
    assert observed["row_count"] == 262917
    assert observed["selected_cells"] == 3
    assert observed["year_requests"] == 363
    assert observed["local_file_present"] is True
    assert observed["file_sha256_present"] is True


def test_validator_source_has_no_era5_value_or_operational_reader() -> None:
    source = SCRIPT.read_text(encoding="utf-8").casefold()
    forbidden = (
        "pd.read_parquet",
        "pyarrow",
        "netcdf",
        "ecmwf.datastores",
        "get_jobs",
        "psutil",
        "get-process",
    )
    assert not any(token in source for token in forbidden)


def test_audit_config_pins_common_recon_and_prohibits_execution() -> None:
    config = json.loads(
        (ROOT / "configs/experiments/p3_era5_static_contract_audit_20260825_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert (
        config["common_secondary_recon"]["sha256"]
        == "6ead6ca745a0c204f04377c2510a2206b61fe6ccf976963fdb7a7cccc070b073"
    )
    assert config["prohibitions"]["era5_value_file_reads"] is True
    assert config["prohibitions"]["running_process_inspection_or_mutation"] is True
    assert config["prohibitions"]["context_runner_execution"] is True
    assert config["prohibitions"]["model_fits"] is True
