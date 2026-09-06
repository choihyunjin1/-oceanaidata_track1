import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/verify_final_candidate_bundle_20260906_v1.py"
SPEC = importlib.util.spec_from_file_location("bundle_check", SOURCE)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def test_hash_inventory_does_not_claim_retraining(tmp_path):
    (tmp_path / "answer.csv").write_text("x\n1\n")
    (tmp_path / "qa.json").write_text(json.dumps({"status": "PASS"}))
    spec = {"csv": "answer.csv", "sha256": m.sha(tmp_path / "answer.csv"),
            "rows": 1, "receipt": "qa.json", "qa": "qa.json"}
    assert m.candidate_check(tmp_path, spec)["status"] == "PINNED_CSV_HASH_MATCH"
    (tmp_path / "answer.csv").write_text("x\n2\n")
    with pytest.raises(ValueError, match="SHA"):
        m.candidate_check(tmp_path, spec)


@pytest.mark.parametrize("name", ["../escape.py", "C:/escape.py", "x\\escape.py", "03_model/model.pt",
                                   "01_data/train.csv", "04_logs/ATTEMPT_LOCK.json", ".env", "a/credentials.json"])
def test_unsafe_cold_entries(tmp_path, name):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as archive:
        info = zipfile.ZipInfo("placeholder")
        info.filename = name
        archive.writestr(info, "synthetic")
    with pytest.raises(ValueError):
        m.cold_archive_check(path)


def test_clean_archive_and_path_boundary(tmp_path):
    path = tmp_path / "good.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("02_code/run.py", "print('synthetic')")
        archive.writestr("README.md", "no raw data")
    result = m.cold_archive_check(path)
    assert result["status"] == "SOURCE_ARCHIVE_INVENTORY_PASS_NOT_EXECUTION"
    with pytest.raises(ValueError, match="outside"):
        m.owned(tmp_path, "../outside")
