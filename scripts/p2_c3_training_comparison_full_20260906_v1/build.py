"""Build an isolated, source-only L120 package; no fits or official reads."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ID = "p2_c3_training_comparison_full_20260906_v1"
HERE = Path(__file__).resolve().parent
CONFIG = ROOT / "configs/experiments" / (ID + ".json")
OUT = ROOT / "artifacts" / ID


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adapted_base(source):
    changes = {
        "submission_p2_clean_C3.csv": "submission_p2_L120_3seed.csv",
        "replay_p2_clean_C3.csv": "replay_p2_L120_3seed.csv",
        'output.to_csv(path, index=False, float_format="%.12g")':
            'output.to_csv(path, index=False, float_format="%.12g", lineterminator="\\n")',
    }
    for before, after in changes.items():
        if source.count(before) != 1:
            raise ValueError("portable base adaptation occurrence changed")
        source = source.replace(before, after)
    return source


def build():
    cfg = json.loads(CONFIG.read_text())
    core = ROOT / "scripts/portable_20260906/P2/02_code/core.py"
    base = ROOT / "scripts/portable_20260906/P2/02_code/run.py"
    if sha(core) != cfg["core_sha256"] or sha(base) != cfg["source_base_sha256"]:
        raise ValueError("C3 pure numerical/base lineage mismatch")
    if OUT.exists():
        raise ValueError("new package directory required")
    for folder in ("01_data", "02_code", "03_model", "04_logs", "05_answer", "06_docs"):
        (OUT / folder).mkdir(parents=True)
    shutil.copy2(core, OUT / "02_code/core.py")
    shutil.copy2(HERE / "run.py", OUT / "02_code/run.py")
    (OUT / "02_code/base.py").write_text(adapted_base(base.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
    shutil.copy2(CONFIG, OUT / "config.json")
    history = {
        "historical_result": ROOT / "artifacts/p2_c3_training_comparison_20260906_v1/terminal_result.json",
        "historical_qa": ROOT / "reports/p2_c3_training_comparison_20260906_v1/independent-qa.json",
        "historical_replay": ROOT / "artifacts/p2_c3_training_comparison_20260906_v1/fresh-replay.json",
        "root_qa": ROOT / "reports/parallel_core_training_20260906_v1/p2-root-independent-qa.json",
    }
    for label, path in history.items():
        if sha(path) != cfg[label + "_sha256"]:
            raise ValueError(label + " provenance drift")
        shutil.copy2(path, OUT / "06_docs" / (label + ".json"))
    shutil.copy2(HERE / "README.md", OUT / "README.md")
    (OUT / "01_data/README.md").write_text("배포 원본은 복사하지 않습니다. P2_DATA_DIR 환경변수를 배포 P2_profile_restore 폴더로 설정하세요.\n", encoding="utf-8")
    paths = [OUT / "config.json", *sorted((OUT / "02_code").glob("*.py")), *sorted((OUT / "06_docs").glob("*.json"))]
    manifest = {"status": "SEALED_BEFORE_FULL_FITS", "fits": 0,
        "files": {p.relative_to(OUT).as_posix(): sha(p) for p in paths},
        "source_base_sha256": sha(base), "core_sha256": sha(core),
        "base_adaptations": ["candidate CSV filenames", "explicit LF CSV serialization only"],
        "authoring_pins": {p.relative_to(ROOT).as_posix(): sha(p) for p in
            (Path(__file__), HERE / "run.py", HERE / "README.md", ROOT / "tests/test_p2_c3_training_comparison_full_20260906_v1.py")}}
    (OUT / "PACKAGE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "SEALED", "path": str(OUT), "fits": 0}))


if __name__ == "__main__":
    build()
