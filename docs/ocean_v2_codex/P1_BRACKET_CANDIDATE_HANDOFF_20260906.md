# P1 bracket-only 후보 — 별도 학습 및 답안 준비

상태: **새 P1 후보의 4회 학습, 독립 내부 QA, 로컬 답안 생성 및 별도 프로세스 전체 replay 완료.** 공식 채점은 하지 않았다. 기존 [기준 패키지](PORTABLE_PACKAGE_HANDOFF_20260906.md)는 fallback으로 보존한다. 이 작업은 로컬 후보 생성이며 업로드, 최종 모델 지정, commit/push를 포함하지 않는다.

## 답안 선택과 검증 범위

- 채점용 파일: `artifacts/p1_bracket_candidate_20260906_v1/package/05_answer/P1_submission.csv`.
- 169,011행, SHA256 `9031c84ea72dfa4294406dd995525e89e8975a76983e9f9d7a7b2ba74dbad93a`. 검산용 파일이나 기존 fallback CSV와 혼동하지 않는다.
- 코드 `package/02_code/`, 이번 학습 모델·recipe `package/03_model/`, 내부 probe `package/04_logs/`, 답안 `package/05_answer/`, 검증 기록 `package/06_docs/`. 배포 원본은 복사하지 않고 `P1_DATA_DIR`로 참조한다.
- 검증: synthetic pytest 17 PASS/Ruff PASS, 별도 모델 replay PASS, root train-only QA 49 PASS, root 답안 QA 26 PASS. 변경하지 않은 original 모델 두 개는 기준 모델 SHA와 일치했다.
- 새로 적합한 최종 선택은 **balanced LightGBM 107열, threshold 0.1**이다. original도 검증했지만 최종 답안은 두 모델 평균/합집합이 아니다. 선택값은 이번 final-inner 학습 산출 recipe에서 유래한다.
- 기준 답안과 1,417행이 다르다(0→1: 1,018행, 1→0: 399행). 이는 완료 후 집계한 차이이지 정답을 더 맞혔다는 뜻이 아니다. 신규 공식 점수는 미확인이다.
- 학습 254.907초, 모델 replay 105.828초, root 내부 QA 24.219초, 추론 25.938초, 답안 replay 25.906초. 단계 사이 대기까지 포함해 답안 replay 완료 시 589.317초였다.

실측 내부 지표와 악화 위험은 [후보 보고서](../../reports/p1_bracket_candidate_20260906_v1/report-source.md), 최종 파일 검사는 [root 답안 QA](../../reports/p1_bracket_candidate_20260906_v1/root-answer-qa.json)를 참조한다. 내부 개선·로컬 재현·공식 성적을 구분했다.

### 이후 재현 시 주의

이 runner는 학습 시작 시각부터 3,600초까지로 제한된 **실험 검증본**이다. 완료 폴더에서 다음날 `infer`/`verify`를 다시 실행하면 시간 제한으로 차단된다. 모델이나 lock/영수증을 지워 우회하지 않는다. 새 빈 폴더에서 코드와 검증 도구만 복사한 뒤 새 학습→model-qa→독립 QA→infer→verify가 필요하다. 장기 보관 모델만 바로 추론하는 최종 제출 패키지로 승격하려면 별도의 재생 어댑터·포장 검증이 필요하며 이번 완료 범위가 아니다. 기존 검증된 기준 패키지를 대체하거나 최종 모델로 잠그지 않았다.

## 고정 실행 계약

- 변경 하나: balanced LightGBM에 검증된 구간 모양 특징 27개 추가(80→107열). original XGBoost는 기존 80열을 유지한다.
- 창 6/24/72시간, 외부 flank 1시간, 최대 반경 37시간. 정점·층·10분 연속 구간 경계를 넘지 않는다. 라벨로 특징 창을 선택하지 않는다.
- 총 4 fits: final-inner original/balanced 각 1회, 전체 학습 original/balanced 각 1회. CPU 4 threads/GPU 0, 700 trees, workflow 3,600초 상한. 중단된 실행을 자동 재시작하지 않는다.
- 기존 final-inner 구간 2025-07-12 이상~2025-09-10 미만, 학습 cutoff 2025-06-21 미만, purge 21일 및 끝의 양성 run 제거를 유지한다. 구간을 나눈 뒤 각 구간의 특징을 계산한다.
- 기존 threshold grid와 결합 정책 선택 알고리즘을 그대로 사용하되 이번 inner 학습 결과에서 새로 적합한다. 과거 선택값, outer 최고값, 리더보드 유래 값은 이식하지 않는다.
- depth fallback, 범위 하드 룰, 셀별 조합 정책, 추가 HPO 또는 새 후처리는 포함하지 않는다.

## 완료 조건과 증거 구분

1. synthetic parity·분할·출력 폴더·접근 경계 검사와 focused pytest/Ruff를 먼저 완료한다.
2. 새 빈 모델 폴더에서 4 fits를 수행한다. 변경하지 않은 XGBoost의 inner/full 모델을 기존 기준과 대조한다.
3. training result·모델 hash·fit count·내부 혼동행렬/선택 계산을 검산하고, 별도 프로세스에서 저장 모델의 내부 예측을 재생한다. 이 단계까지 공식 입력 접근 0, 답안 CSV 0이다.
4. 위 검증 통과 후에만 공식 공개 입력으로 169,011행 답안을 로컬 생성한다. sample은 키/순서만 사용하고 hidden truth는 접근하지 않는다.
5. schema·키·순서·중복·binary/finite·SHA를 확인하고 저장 모델로 답안 전체를 다시 생성해 비교한다. 기존 기준 SHA에 억지로 맞추지 않는다.

이전 two-sided/depth stress는 이미 관찰한 내부 근거이며 새 holdout이 아니다. final-inner 수치는 선택에 사용한 수치로 표시한다. 저장 모델 replay는 두 번의 독립 전체 재학습을 뜻하지 않는다. 새 후보의 공식 점수는 실제 채점 전 알 수 없다.

## 연결

- 구현: `scripts/portable_20260906/P1_bracket_candidate_v1/`
- 설정: `configs/experiments/p1_bracket_candidate_20260906_v1.json`
- 실행 및 검증 기록: `reports/p1_bracket_candidate_20260906_v1/`
- 앞선 효과 검증: `reports/p1_tuning_twosided_20260906_v1/`, `reports/p1_tuning_depth_stress_20260906_v2/`

모든 로컬 보존 파일은 Git 제외 대상이다. 실제 업로드는 현재 포털 횟수·마감·첨부 제한과 사용자 승인 범위를 확인한 뒤 별도 수행한다.
