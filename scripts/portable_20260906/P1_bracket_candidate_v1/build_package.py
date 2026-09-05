"""Build an isolated bracket-only candidate from pinned, auditable source copies."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONFIG = "configs/experiments/p1_bracket_candidate_20260906_v1.json"
BASE = HERE.parent / "P1"
spec = importlib.util.spec_from_file_location("clean_p1_builder", BASE / "build_package.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def build(destination):
    # This copies code/config only; the existing baseline and all model files are untouched.
    base.build(destination)
    code = destination / "02_code"
    manifest = json.loads((code / "source-manifest.json").read_text(encoding="utf-8"))

    def put(name, value, sources):
        path = code / name
        path.write_text(value, encoding="utf-8", newline="\n")
        manifest[name] = {"sha256": base.sha(path), "source_sha256": {
            str(p.relative_to(ROOT)).replace("\\", "/"): base.sha(p) for p in sources}}

    bracket = "scripts/run_p1_bracket_forward_20260906_v1.py"
    header = "import core as old\nimport numpy as np\nimport pandas as pd\n\n\n"
    put("bracket.py", header + base.extract(bracket, ["bracket_features"]), [ROOT / bracket])
    put("configs/candidate.json", (ROOT / CONFIG).read_text(encoding="utf-8"), [ROOT / CONFIG])
    driver = "scripts/verify_p1_clean_regeneration_20260905_v5.py"
    header = "# ruff: noqa: E402\n" + (BASE / "runtime_header.txt").read_text(encoding="utf-8")
    header = header.replace("Portable P1 v5-equivalent fresh training and answer generation.",
                            "P1 bracket-only four-fit training, gated local inference and replay.")
    header = header.replace('RUN = "p1_portable_cleanroom_20260906_v1"',
                            'RUN = "p1_bracket_candidate_20260906_v1"')
    header = "\n".join(line for line in header.splitlines()
                       if not line.startswith("EXPECTED_ANSWER_SHA =")) + "\n"
    header = header.replace("    return {}, frozen, hashes",
                            '    return json.loads((ROOT / "configs/candidate.json").read_text()), frozen, hashes')
    header = header.replace("import tempfile", "import threading")
    header = header.replace("from __future__ import annotations\n", "from __future__ import annotations\n\n")
    header = header.replace("from types import SimpleNamespace\n", "from types import SimpleNamespace\n\n")
    header = header.replace("import core as screen", "import bracket\nimport core as screen")
    header = header.replace("import decoder\n", "import decoder\n\n")
    common = base.extract(driver, ["write", "policy_bits", "select_policy",
                                   "ensure_empty_models", "assert_sources"])
    put("run.py", header + common + (HERE / "runtime.txt").read_text(encoding="utf-8"),
        [ROOT / driver, BASE / "runtime_header.txt", HERE / "runtime.txt", Path(__file__)])
    # Include build-time dependencies in the provenance even though runtime is repository-independent.
    manifest["run.py"]["source_sha256"]["scripts/portable_20260906/P1/build_package.py"] = base.sha(BASE / "build_package.py")
    (code / "source-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (destination / "README.md").write_text((HERE / "README.md").read_text(encoding="utf-8"), encoding="utf-8")
    (destination / "06_docs/constant-lineage.md").write_text(
        "# Constants and trained values\n\nO80 is the unchanged clean recipe. B adds only the exact validated 27 bracket features "
        "(6/24/72 h windows, one-hour flanks). CPU4, seed and 700 trees are frozen. "
        "Final-inner policy/thresholds, training statistics, encoders and all four models are fitted anew from distributed train. "
        "No depth fallback, hard-rule/cell patch, historical threshold transfer, old model, answer, external observation or score inversion is an input. "
        "Full train probe is in-sample determinism only. Existing model SHA strings are comparison metadata, not optimization targets.\n",
        encoding="utf-8")
    return {"package": str(destination), "manifest_sha256": base.sha(code / "source-manifest.json"),
            "run_sha256": base.sha(code / "run.py"), "model_files_before_training": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    print(json.dumps(build(parser.parse_args().destination.resolve())))
