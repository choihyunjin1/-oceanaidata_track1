"""Build a disjoint original-recipe source package; never copy learned artifacts."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

MODULES = ["__init__", "pipeline", "config", "data", "features", "rules", "submission",
           "postprocess", "metrics", "models_tabular", "augment", "splits", "validation",
           "experiment", "audit", "ms_tcn_asrf", "ms_tcn_asrf_data"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def build(repo, target):
    import nbformat

    target.mkdir(parents=True, exist_ok=False)
    for name in ["01_data", "02_code", "03_model", "04_logs", "05_answer", "06_docs"]:
        (target / name).mkdir()
    code = target / "02_code"
    copies = {f"src/p1_qc/{name}.py": f"source/src/p1_qc/{name}.py" for name in MODULES}
    for rel in ["configs/p1.toml", "configs/experiments/p1_incumbent_preserving_mstcn_asrf_v2.json",
                "scripts/run_p1_incumbent_preserving_mstcn_asrf_v2.py"]:
        copies[rel] = f"source/{rel}"
    copies.update({
        "scripts/p1_champion_reconstruction_20260906_v1/mstcn.py": "ms_driver.py",
        "scripts/portable_20260906/P1_original_v1/run.py": "run.py",
        "scripts/p1_historical_path_audit_20260906_v1.py": "composition.py",
        "artifacts/runs/20260813T155254+0900_train_378a4e89/config.toml": "tree_config.toml",
    })
    provenance = {}
    for source, relative in copies.items():
        dest = code / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / source, dest)
        provenance[relative] = {"source": source, "sha256": sha(dest)}
    # Mechanical extraction of the original pure weighting function. No alternate
    # algorithm or historical one-shot CLI is executed.
    source = repo / "scripts/run_p1_meaningful_learning_curve_generation_v1.py"
    text = source.read_text(encoding="utf-8")
    func = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)
                and n.name == "_event_day_weight")
    extracted = ast.get_source_segment(text, func)
    with (code / "event_weight.py").open("x", encoding="utf-8") as handle:
        handle.write('"""Unchanged pure event/day weighting from the historical runner."""\n'
                     'import math\nimport numpy as np\nimport pandas as pd\n\n' + extracted + "\n")
    provenance["event_weight.py"] = {"source": str(source.relative_to(repo)),
                                    "source_sha256": sha(source), "function": func.name,
                                    "sha256": sha(code / "event_weight.py")}
    old_selection = json.loads((repo / "artifacts/runs/20260813T153038+0900_cv_378a4e89/selection.json").read_text())
    old_b = json.loads((repo / "configs/p1_meaningful_learning_curve_generation_v1.json").read_text())
    selection = {k: old_selection[k] for k in ["backend", "feature_mode", "iteration_count", "feature_hash", "postprocess"]}
    write(code / "tree_recipe.json", {"O_selection": selection, "B_parameters": old_b["lightgbm_parameters"],
          "parameter_provenance": "Historical training-only CV selection and preregistered B recipe; "
                                  "frozen for deployment retraining, not re-selected from Public scores"})
    contract = {
        "id": "p1_original_source_20260906_v1", "model_fits": 7, "cap_seconds": 21600,
        "tree_threads": 8, "ms_cpu_threads": 2, "ms_device": "exclusive CUDA0 bf16",
        "train_sha256": "20b656b0cbd524ad9da0bae8ecb6e0bacfc006e05810b37e83f29a5fa8e65cd2",
        "test_sha256": "6d5c6522c282651b99f4261ffa803cf99950596028e996de1e7714db77408387",
        "historical_answer_sha256": "57844ef235f987059d17bddb035cc0a98b92bb0d53263cf878c68b31bbc53687",
        "historical_cells": "Preserved as the original locally selected fixed policy; original "
                            "cell-selector algorithm is not recovered or rerun",
        "covariate_context": "Original released whole-series offline features; sample/year depth "
                             "summaries computed separately on the current supplied observations. "
                             "Learned encoders/model parameters fitted on TRAIN only.",
        "scope": "Exact original recipe recovery, not a new threshold/feature search or the clean "
                 "train-fit-depth ablation. No claim of organizer final acceptance.",
        "uploads": 0, "pretrained_or_archived_model_inputs": 0,
    }
    write(target / "contract.json", contract)
    shutil.copyfile(repo / "scripts/portable_20260906/P1_original_v1/README.md", target / "README.md")
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell("# P1 원형 전체 재학습·추론\n\n"
            "## Goal\n빈 `03_model`에서 7모델 학습→별도 프로세스 QA→답안을 만듭니다. "
            "과거 모델/답안/캐시는 입력이 아닙니다. 셀 정책은 역사적 고정 설정이며 재선택하지 않습니다. "
            "약 2시간 예상, 전체 6시간 제한. 기존 실행 폴더에서는 다시 실행하지 마세요."),
        nbformat.v4.new_markdown_cell("## Setup\n패키지 루트를 작업 폴더로 열고 `P1_DATA_DIR`을 "
                                     "배포 P1 디렉터리로 설정하세요. 검증된 Python 환경을 사용합니다."),
        nbformat.v4.new_code_cell("import os, sys, subprocess\nfrom pathlib import Path\n"
            "package = Path.cwd()\nassert (package / 'contract.json').is_file()\n"
            "data = Path(os.environ['P1_DATA_DIR']).resolve()\n"
            "assert not any((package / '03_model').iterdir())\n"
            "assert not (package / 'ATTEMPT_LOCK.json').exists()"),
        nbformat.v4.new_markdown_cell("## Steps\n학습 로그와 진행률은 `04_logs`, "
                                     "`03_model/mstcn/progress.json`에 기록됩니다."),
        nbformat.v4.new_code_cell("subprocess.run([sys.executable, str(package / '02_code/run.py'), "
                                  "'all', '--data', str(data)], check=True, cwd=package)"),
        nbformat.v4.new_markdown_cell("## Checks\n실제 완료 영수증을 확인합니다. "
                                     "`historical_answer_exact`가 false면 과거 점수를 승계하지 않습니다."),
        nbformat.v4.new_code_cell("import json\nresult = json.loads((package / 'terminal.json').read_text())\n"
            "print(result)\nassert result['status'] == 'TRAIN_TO_ANSWER_COMPLETE'"),
        nbformat.v4.new_markdown_cell("## Next Steps\n공식 제출 전 후보 계보·독립 QA·실측 시간·"
                                     "최신 대회 요건을 확인합니다. 이 노트북은 업로드하지 않습니다."),
    ], metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    nbformat.validate(notebook)
    nbformat.write(notebook, target / "RUN_ALL.ipynb")
    versions = {name: importlib.metadata.version(name) for name in [
        "numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "xgboost", "lightgbm", "torch",
        "joblib", "nbformat", "nbclient", "ipykernel"]}
    write(target / "06_docs/environment.json", versions)
    write(target / "06_docs/source-provenance.json", provenance)
    files = {p.relative_to(target).as_posix(): sha(p) for p in target.rglob("*") if p.is_file()}
    write(target / "source-manifest.json", {"files": files, "models_copied": 0, "data_copied": 0,
                                           "cache_copied": 0, "notebook": "SCHEMA_VALID_EXECUTION_PENDING"})
    print(json.dumps({"status": "SOURCE_PACKAGE_CREATED_NOT_TRAINED", "root": str(target),
                      "files": len(files)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    build(args.repo.resolve(), args.target.resolve())
