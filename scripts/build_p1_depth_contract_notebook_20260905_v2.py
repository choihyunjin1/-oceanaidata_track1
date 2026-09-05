"""Build and execute an aggregate-only P1 research handoff notebook."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
RUN = "p1_depth_contract_repair_20260905_v2"


def main():
    directory = ROOT / "reports" / RUN
    result = json.loads((directory / "decoder-result.json").read_text(encoding="utf-8"))
    qa = json.loads((directory / "cycle-independent-qa.json").read_text(encoding="utf-8"))
    if qa["status"] != "PASS":
        raise ValueError("verified aggregates required")
    selected = result["chosen_development_policy"]
    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "# P1 수심 계약 v2 — 재사용 가능한 학습·내부검증 기록\n\n"
            "## tl;dr\n\n"
            f"동일 421,032행의 development 정책 순위는 `{selected}`가 가장 높았다. "
            f"기준 control 대비 ΔF1={result['delta_f1_vs_control']:+.8f}. "
            "공식 점수 상승은 미측정이며, 공식 입력/CSV/업로드는 이 사이클에서 0이다."
        ),
        nbformat.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "A: 현재 depth를 2m 반올림하는 연도 독립 특징, 나머지 recipe 동일. "
            "Q2/Q3/Q4 purge21일·earlier-inner60일·seed20260813, 신규12 historical+2 final-inner+2 full fits. "
            "B: legacy OOF 출처/키만 감사, 119행 fold 불일치 및 router purge7일로 무학습 결합 차단. "
            "C: A/B 완료 뒤 고정 OFF/ON(lambda1), 0 backbone fits.\n\n"
            "### Key Assumptions\n\n반복 노출된 historical development다. 월별 악화/CI는 위험 근거이며 "
            "새 hard gate가 아니다. 이 노트북은 집계 QA의 실행 가능한 companion으로 "
            "학습을 다시 시작하거나 공식 값을 읽지 않는다."
        ),
        nbformat.v4.new_markdown_cell(
            "### 학습 실행 경로\n\n"
            "새 isolated checkout에서 `P1_DATA_DIR`를 배포 데이터 폴더로 설정하고 "
            "`.venv-p1/Scripts/python.exe scripts/run_p1_depth_contract_repair_20260905_v2.py --execute`를 실행한다. "
            "기존 artifact가 있는 checkout에서는 one-shot 보호로 재실행되지 않는다. "
            "완료 후 같은 runner `--verify`, postaudit `--provenance`(1회), `--decoder`, "
            "독립 QA runner 순서다. 모델/config/lock를 지우거나 수정하여 재실행하지 않는다. "
            "04_models의 저장 모델만 추론하는 것과 위 학습 절차를 구분한다."
        ),
        nbformat.v4.new_markdown_cell(
            "## Data\n\n배포 train 776,706행의 실행 영수증과 OOF 집계만 읽는다. 원시 관측·행별 정답은 이 노트북에 없다."
        ),
        nbformat.v4.new_code_cell(
            "import json\nfrom pathlib import Path\n"
            "project_root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'configs').is_dir() and (p / 'scripts').is_dir())\n"
            f"report_dir = project_root / 'reports/{RUN}'\n"
            "read = lambda name: json.loads((report_dir / name).read_text(encoding='utf-8'))\n"
            "training = read('result.json')\ndecoder = read('decoder-result.json')\n"
            "audit = read('cycle-independent-qa.json')\n"
            "assert audit['status'] == 'PASS'\n"
            "{key: training[key] for key in ['status', 'train_sha256', 'screen_fits', 'final_inner_fits', 'full_fits', 'runtime_seconds', 'official_rows']}"
        ),
        nbformat.v4.new_markdown_cell(
            "## Results\n\n### 동일 평가키의 주요 지표\n\nF1은 fold 평균이 아니라 pool한 TP/FP/FN에서 산출한다."
        ),
        nbformat.v4.new_code_cell(
            "[{ 'policy': name, **values } for name, values in audit['independent_metrics'].items()]"
        ),
        nbformat.v4.new_markdown_cell("### 불확실성과 분할 계약"),
        nbformat.v4.new_code_cell(
            "provenance = read('provenance-audit.json')\n"
            "{'intervals': decoder['intervals_vs_control'], 'B_status': provenance['status'], "
            "'fold_changed_rows': provenance['changed_fold_rows'], 'QA': [audit['passed'], audit['check_count']]}"
        ),
        nbformat.v4.new_markdown_cell(
            "## Takeaways\n\n"
            "정책의 실제 효과와 수심 결측 계약의 수정 여부를 구분한다. B의 미실행은 성능 실패가 아니다. "
            "공식 예상 점수는 미산정이며 마지막 소수의 Public 반환값으로 환산식을 맞추지 않는다. "
            "자세한 fold/월/정점층 결과는 decoder-result.json과 report-source.md에 보존한다. "
            "현재 모델은 연구 자산이며 기존 최종 제출 패키지를 대체하지 않았다."
        ),
    ]
    NotebookClient(
        notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}
    ).execute()
    path = directory / "analysis-companion.ipynb"
    if path.exists():
        raise FileExistsError("notebook already exists")
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print(
        json.dumps(
            {
                "notebook": str(path.relative_to(ROOT)),
                "cells": len(notebook.cells),
                "status": "EXECUTED_TOP_TO_BOTTOM",
            }
        )
    )


if __name__ == "__main__":
    main()
