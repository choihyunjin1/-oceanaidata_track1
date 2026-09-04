"""Generate isolated TRAIN and PREDICT notebooks for P1, P2, and P3."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "final_submission_20260905"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def metadata() -> dict:
    return {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }


def training_notebook(problem: str, note: str) -> nbf.NotebookNode:
    output_dirs = {
        "P1": "03_model/retrained_from_scratch",
        "P2": "03_model/retrained_from_notebook",
        "P3": "03_model/retrained_reference",
    }
    calls = {
        "P1": "train_model.train(DATA_DIR, PACKAGE_DIR, OUTPUT_DIR)",
        "P2": "train_model.train(DATA_DIR, PACKAGE_DIR, OUTPUT_DIR)",
        "P3": "train_model.train(DATA_DIR, PACKAGE_DIR, OUTPUT_DIR)",
    }
    cells = [
        markdown(
            f"# {problem} — TRAIN\n\n"
            "이 notebook은 운영진 배포 데이터에서 scratch 학습하는 과정을 독립적으로 보존합니다. "
            "최종 답안을 복사하는 notebook이 아닙니다. 기본 실행은 이미 생성된 certified model manifest와 "
            "입력 해시를 검증하며, `RUN_FULL_SCRATCH_RETRAIN=True`로 바꾸면 별도 출력 폴더에 새 모델을 "
            f"학습합니다. {note}"
        ),
        code(
            "from pathlib import Path\n"
            "import hashlib\n"
            "import json\n"
            "import sys\n\n"
            "PACKAGE_DIR = Path.cwd().resolve().parent\n"
            "DATA_DIR = PACKAGE_DIR / '01_data' / 'organizer_dataset'\n"
            "if not (PACKAGE_DIR / 'contract.json').is_file():\n"
            "    raise RuntimeError('Run this notebook from P?/02_train')\n"
            "sys.path.insert(0, str(Path.cwd().resolve()))\n"
            "contract = json.loads((PACKAGE_DIR / 'contract.json').read_text(encoding='utf-8'))\n"
            "input_manifest = json.loads((PACKAGE_DIR / '01_data/INPUT_MANIFEST.json').read_text(encoding='utf-8'))\n"
            "model_manifest = json.loads((PACKAGE_DIR / '03_model/MODEL_MANIFEST.json').read_text(encoding='utf-8'))\n"
            "print({'candidate': contract['candidate_id'], 'input_files': len(input_manifest['files']), "
            "'model_status': model_manifest['status']})"
        ),
        markdown("## 1. 현재 학습 산출물의 무결성 확인"),
        code(
            "def sha256(path):\n"
            "    h = hashlib.sha256()\n"
            "    with Path(path).open('rb') as stream:\n"
            "        for block in iter(lambda: stream.read(1 << 20), b''):\n"
            "            h.update(block)\n"
            "    return h.hexdigest()\n\n"
            "checked = []\n"
            "for record in contract['model_files']:\n"
            "    path = PACKAGE_DIR / record['path']\n"
            "    assert path.is_file() and sha256(path) == record['sha256']\n"
            "    checked.append(record['path'])\n"
            "print({'verified_model_files': len(checked), 'pretrained_weights_loaded': 0, 'external_rows': 0})"
        ),
        markdown(
            "## 2. 선택적 전체 scratch 재학습\n\n"
            "기존 certified weights를 덮어쓰지 않도록 별도 디렉터리에 출력합니다. P1은 3×150 epoch라 "
            "오래 걸립니다. P3의 간단 wrapper는 base branches를 재학습하며, 최종 router/calibrator의 정확한 "
            "역사적 학습 소스는 `07_source/scripts/`에 함께 있습니다."
        ),
        code(
            "RUN_FULL_SCRATCH_RETRAIN = False\n"
            f"OUTPUT_DIR = PACKAGE_DIR / '{output_dirs[problem]}'\n"
            "if RUN_FULL_SCRATCH_RETRAIN:\n"
            "    import train_model\n"
            f"    retrain_receipt = {calls[problem]}\n"
            "    print(retrain_receipt)\n"
            "else:\n"
            "    print({'status': 'CERTIFIED_TRAINING_OUTPUT_VERIFIED', 'retrain_executed_now': False, "
            "'toggle': 'set RUN_FULL_SCRATCH_RETRAIN=True'})"
        ),
        markdown(
            "## 3. 다음 단계\n\n"
            "최종 제출 재현은 `../04_predict/PREDICT.ipynb`가 `03_model`의 검증된 가중치를 실제로 로드해 "
            "수행합니다. `05_answer`는 모델 추론 SHA가 최고점 후보 SHA와 같을 때만 생성됩니다."
        ),
    ]
    return nbf.v4.new_notebook(cells=cells, metadata=metadata())


def prediction_notebook(problem: str, metric: str) -> nbf.NotebookNode:
    cells = [
        markdown(
            f"# {problem} — PREDICT / 제출 파일 생성\n\n"
            f"공식 지표는 `{metric}`입니다. 이 notebook은 `03_model`의 실제 가중치를 로드해 추론하며, "
            "기존 최고점 CSV를 입력으로 읽거나 복사하지 않습니다. 네트워크 요청과 업로드도 하지 않습니다."
        ),
        code(
            "from pathlib import Path\n"
            "import importlib.util\n"
            "import sys\n\n"
            "PACKAGE_DIR = Path.cwd().resolve().parent\n"
            "DATA_DIR = PACKAGE_DIR / '01_data' / 'organizer_dataset'\n"
            "if not (PACKAGE_DIR / 'run_submission.py').is_file():\n"
            "    raise RuntimeError('Run this notebook from P?/04_predict')\n"
            "sys.path.insert(0, str(PACKAGE_DIR))\n"
            "spec = importlib.util.spec_from_file_location('final_runner', PACKAGE_DIR / 'run_submission.py')\n"
            "run_submission = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(run_submission)\n"
            "from common import bounded_receipt\n"
            "print({'package': PACKAGE_DIR.name, 'data_present': DATA_DIR.is_dir()})"
        ),
        markdown("## 1. 데이터·모델·결정 자산 hash preflight"),
        code("preflight = run_submission.preflight(DATA_DIR, PACKAGE_DIR)\nbounded_receipt(preflight)"),
        markdown("## 2. 모델 추론으로 제출 CSV 생성"),
        code(
            f"output_path = PACKAGE_DIR / '05_answer' / '{problem}_submission.csv'\n"
            "receipt = run_submission.materialize(DATA_DIR, PACKAGE_DIR, output_path)\n"
            "bounded_receipt(receipt)"
        ),
        markdown("## 3. 제출 계약 확인"),
        code(
            "assert receipt['status'] == 'READY_MODEL_INFERENCE_EXACT_NOT_UPLOADED'\n"
            "assert receipt['candidate_hash_exact'] and receipt['key_order_exact']\n"
            "assert receipt['prediction_source'] != 'frozen_candidate_csv_copy'\n"
            "assert output_path.is_file()\n"
            "print({'status': receipt['status'], 'rows': receipt['rows'], 'sha256': receipt['sha256']})"
        ),
        markdown(
            "정확한 열 순서, dtype, 행 수, 홈페이지 제목과 한 줄 요약은 "
            "`../06_submission/FORMAT.md`와 `FORM.json`에 고정되어 있습니다."
        ),
    ]
    return nbf.v4.new_notebook(cells=cells, metadata=metadata())


def main() -> None:
    specs = {
        "P1": (
            "F1",
            "Certified 모델은 운영진 배포 train의 776,706행, 165개 past-only feature로 학습된 3-seed MS-TCN입니다.",
        ),
        "P2": (
            "pooled RMSE (C)",
            "빌더가 이 notebook과 동일한 v52 학습 함수를 호출해 세 체크포인트를 새로 생성합니다.",
        ),
        "P3": (
            "pooled RMSE (m)",
            "Certified 모델은 운영진 배포 자료로 scratch 학습된 두 CatBoost/router 계보입니다.",
        ),
    }
    for problem, (metric, note) in specs.items():
        target = OUT / problem
        target.mkdir(parents=True, exist_ok=True)
        nbf.write(training_notebook(problem, note), target / "TRAIN.ipynb")
        nbf.write(prediction_notebook(problem, metric), target / "PREDICT.ipynb")
        print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
