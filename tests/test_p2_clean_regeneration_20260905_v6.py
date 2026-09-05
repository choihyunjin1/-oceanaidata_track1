"""Actual native torch.save and checksum path under an active audit hook."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_native_three_seed_save_checksum_and_negative_reads_in_fresh_process(tmp_path):
    code = r'''
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "scripts"))
import run_p2_clean_regeneration_20260905_v6 as r
m = r.original
root = Path(sys.argv[1]).resolve()
m.MODELS = root / "03_model"
m.OUT = root
m.ANSWERS = root / "05_answer"
m.MODELS.mkdir()
assert not list(m.MODELS.iterdir())
source = root / "source" / "observations.csv"
source.parent.mkdir()
source.write_text("value\n1\n", encoding="utf-8")
old = root / "old" / "model.pt"
old.parent.mkdir()
m.torch.save({"weight": m.torch.ones(1)}, old)
outside = root / "outside.csv"
outside.write_text("value\n9\n", encoding="utf-8")
m.install_guard("RUN_TRAINING", source)
assert len(m.pd.read_csv(source)) == 1
checksums = []
for seed in [11, 22, 33]:
    m.torch.manual_seed(seed)
    network = m.torch.nn.Linear(3, 1)
    path = m.MODELS / f"model_seed{seed}.pt"
    m.torch.save(network.state_dict(), path)
    checksums.append(m.research.file_hash(path))
    m.research.atomic_json(m.MODELS / "progress.json", {"seed": seed})
assert len(checksums) == 3 and all(len(x) == 64 for x in checksums)
for forbidden in [old, outside]:
    try:
        forbidden.open("rb")
    except PermissionError:
        pass
    else:
        raise AssertionError("existing model or unrelated source read was allowed")
try:
    m.torch.load(m.MODELS / "model_seed11.pt", weights_only=True)
except PermissionError:
    pass
else:
    raise AssertionError("training torch.load was allowed")
try:
    source.open("w")
except PermissionError:
    pass
else:
    raise AssertionError("source write was allowed")
print("NATIVE_THREE_SAVE_HASH_AND_NEGATIVE_READS_PASS")
'''
    outcome = subprocess.run([sys.executable, "-c", code, str(tmp_path)], cwd=ROOT, capture_output=True, text=True, timeout=40)
    assert outcome.returncode == 0, outcome.stderr
    assert "NATIVE_THREE_SAVE_HASH_AND_NEGATIVE_READS_PASS" in outcome.stdout


def test_training_driver_keeps_empty_directory_proof_and_no_legacy_fit():
    source = (ROOT / "scripts/run_p2_clean_regeneration_20260905_v4.py").read_text(encoding="utf-8")
    assert "assert not list(MODELS.iterdir())" in source
    assert "if OUT.exists():" in source
    assert "canonical.train(True)" in source
    current = (ROOT / "scripts/run_p2_clean_regeneration_20260905_v6.py").read_text(encoding="utf-8")
    assert "original.torch.load = deny_training_model_load" in current
    assert "write-event registry" in current
