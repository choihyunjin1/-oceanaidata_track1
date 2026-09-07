import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/p3_cpudet_unbounded_20260907_v2.py"


def test_only_execution_budget_and_id_changed():
    old = json.loads(
        (ROOT / "configs/experiments/p3_numeric_cpudet_s3_20260907_v1.json").read_text()
    )
    new = json.loads(
        (ROOT / "configs/experiments/p3_numeric_cpudet_unbounded_20260907_v2.json").read_text()
    )
    ignored = {"id", "maximum_seconds_per_cold", "execution_amendment"}
    assert {k: v for k, v in old.items() if k not in ignored} == {
        k: v for k, v in new.items() if k not in ignored
    }
    assert new["maximum_seconds_per_cold"] is None
    assert new["execution_amendment"]["total_new_production_fits"] == (36 - 29) + 5 + 36 + 5 == 53


def test_no_workflow_timer_or_subprocess_timeout():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    selected = [n for n in tree.body if getattr(n, "name", None) in {"NoTimer", "no_watchdog"}]
    scope = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<timer-test>", "exec"), scope)
    assert scope["no_watchdog"]({}, 0).cancel() is None
    supervisor = ast.parse(
        (ROOT / "scripts/build_execute_p3_unbounded_20260907_v2.py").read_text(encoding="utf-8")
    )
    for node in ast.walk(supervisor):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run"
        ):
            assert all(k.arg != "timeout" for k in node.keywords)


def test_reuse_requires_exact_complete_model_set_and_hash(tmp_path):
    modeldir = tmp_path / "03_model"
    docs = tmp_path / "06_docs"
    modeldir.mkdir()
    docs.mkdir()

    def digest(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    def write(p, v):
        p.write_text(json.dumps(v))

    config = {"id": "new", "maximum_seconds_per_cold": None, "seeds": [1, 2, 3]}
    write(docs / "parent-config.json", {**config, "id": "old", "maximum_seconds_per_cold": 14400})
    write(docs / "prepare.json", {"source_sha256": {"train": "source-pin"}})
    fits = []
    pins = {}
    for i in range(29):
        file = modeldir / f"{i}.cbm"
        file.write_bytes(b"synthetic-model-pin")
        relative = file.relative_to(tmp_path).as_posix()
        fits.append({"model": relative})
        pins[relative] = digest(file)
    receipt = {
        "parent_config_sha256": digest(docs / "parent-config.json"),
        "prepare_sha256": digest(docs / "prepare.json"),
        "imported_files": pins,
        "fits": fits,
    }
    write(docs / "reuse-manifest.json", receipt)
    scope = {
        "P": tmp_path,
        "D": docs,
        "M": modeldir,
        "b": SimpleNamespace(read=lambda p: json.loads(p.read_text()), sha=digest),
    }
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if getattr(n, "name", None) == "reuse_fits")
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<reuse-test>", "exec"), scope)
    assert len(scope["reuse_fits"](config, {"source_files": {"train": "source-pin"}})) == 29
    (modeldir / "partial.cbm").write_bytes(b"partial")
    with pytest.raises(ValueError, match="partial"):
        scope["reuse_fits"](config, {"source_files": {"train": "source-pin"}})
