"""No-fit archive boundaries and minimal CPU2 resource-amendment checks."""
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("portable_archives", HERE / "build_archives.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


@pytest.fixture(scope="module")
def archives(tmp_path_factory):
    path = tmp_path_factory.mktemp("p1_archive_test") / "new"
    b.build(path)
    return path


def test_frozen_source_hashes_stay_exact(archives):
    c = json.loads((HERE / "contract.json").read_text())
    b.validate_source(c)
    assert b.sha(archives / "saved/P1/02_code/run.py") == c["source_run_sha256"]
    assert b.sha(archives / "saved/P1/02_code/source-manifest.json") == c["source_manifest_sha256"]


def test_new_destinations_only(archives):
    with pytest.raises(FileExistsError):
        b.build(archives)


@pytest.mark.parametrize("mode,model_count", [("saved", 2), ("cold", 0)])
def test_archive_members_crc_and_explicit_data_exclusions(archives, mode, model_count):
    with zipfile.ZipFile(archives / ("P1_bracket_" + mode + "_v2.zip")) as stream:
        assert stream.testzip() is None
        names = stream.namelist()
        assert sum(n.endswith(".joblib") for n in names) == model_count
        assert not any(n.endswith((".csv", ".npy", ".npz", ".pyc", ".parquet")) for n in names)
        assert not any("ATTEMPT_LOCK" in n or ".." in Path(n).parts for n in names)
        if mode == "cold":
            assert not any(n.endswith(("train_result.json", "model-replay-qa.json", "independent-training-qa.json")) for n in names)


def test_only_id_and_thread_constants_change_in_cold_runner(archives):
    original = (b.SOURCE / "02_code/run.py").read_text(encoding="utf-8")
    cold = (archives / "cold/P1/02_code/run.py").read_text(encoding="utf-8")
    reverse = cold.replace('RUN = "p1_bracket_portable_cold_cpu2_20260906_v2"',
                           'RUN = "p1_bracket_candidate_20260906_v1"').replace("THREADS = 2", "THREADS = 4")
    assert reverse == original
    cfg = json.loads((archives / "cold/P1/02_code/configs/candidate.json").read_text())
    assert cfg["threads"] == 2 and cfg["max_fits"] == 4 and cfg["trees"] == 700
    assert cfg["features"] == {"original": 80, "balanced": 107}
    assert cfg["depth_policy"] == "unchanged station-year-layer lookup; no new fallback"


def test_adapter_has_independent_clock_and_no_training_or_answer_inputs():
    code = (HERE / "archive_infer.py").read_text()
    assert "source.train(" not in code and "source.infer(" not in code
    assert "source.budget_guard(" not in code and "started_unix" not in code
    assert ' / "test.csv"' in code and ' / "sample_submission.csv"' in code
    assert ' / "train.csv"' not in code and 'target.open("xb")' in code
    assert "source.screen.RAW" in code and "source.screen.KEYS" in code
    assert "backend.set_params(n_jobs=" in code


@pytest.mark.parametrize("mode,script", [("saved", "02_code/archive_infer.py"), ("cold", "02_code/run.py"),
                                        ("cold", "06_docs/qa_training_independent.py")])
def test_isolated_entrypoints_from_other_cwd(archives, tmp_path, mode, script):
    proc = subprocess.run([sys.executable, "-I", str(archives / mode / "P1" / script), "--help"],
                          cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr


def test_saved_tamper_rejected_before_official_read(archives):
    code = '''import archive_infer as a
path=a.PACKAGE/"02_code/bracket.py"
path.write_text(path.read_text()+"\\n# tamper")
try:a.checked_contract()
except ValueError as e:assert "changed" in str(e)
else:raise AssertionError("tamper accepted")
'''
    proc = subprocess.run([sys.executable, "-c", code], cwd=archives / "saved/P1/02_code",
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
