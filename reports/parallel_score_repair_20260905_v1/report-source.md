# 점수 개선 실행 — 2026-09-05

> 후속 상태: 아래 본문은 학습·후보 준비 당시 기록이다. 이후 사용자 승인으로 2026-09-05 19:28~19:29 KST에 네 후보를 공식 제출해 모두 채점 완료했다. [공식 결과와 정확 파일 영수증](../official_score_repair_submissions_20260905_v1/report-source.md)을 우선 확인한다. 기존 학습 결과·해시·선택 절차는 변경하지 않았다.

## 결론

**세 문제의 실제 학습·내부 테스트·로컬 후보 생성을 완료했다. P2의 연속 결측 증강이 가장 뚜렷한 내부 개선을 보였다. P1은 기존 clean control을 전체 학습해 보존했고, P3는 재현된 clean 기준선과 미세한 6h 혼합 후보를 준비했다.** P1의 flank 특징·조건부 decoder와 P3의 기상 결측 증강은 악화되어 채택하지 않았다. 공식 점수 상승이나 최종 제출 완료를 의미하지 않는다.

기준 Git: `535f94a1791f2398f82ad27659b58701513ab327`, `codex/p1-qc`. 승인된 [실행 계획](../../docs/SCORE_IMPROVEMENT_PLAN_20260905.md)을 구현했다. 과거 모델·답안·attempt lock·사용자 변경은 보존했고 기존 실험을 재시작하지 않았다. 이번 작업에는 업로드 및 commit/push가 포함되지 않는다.

## 실측 비교

| 문제 | 같은 평가의 기준 | 후보 | 후보−기준 | 해석 |
|---|---:|---:|---:|---|
| P1, F1 | 0.851174240 | 0.836927881 | −0.014246359 | flank 특징 기각, 421,032행, 앞선 inner에서만 선택/임계값 결정 |
| P2, RMSE ℃ | 0.920345939 | 0.859249914 | −0.061096025 | v23 3seed 대 v23 연속 결측 증강 3seed, 69,850행 |
| P3, RMSE m | 0.779104840 | 0.778339752 | −0.000765088 | clean 기준 대 6h-only TabPFN25, 181사례·1,086행 |

P1은 기존 대비 TP가 70개 늘었지만 FP가 609개 늘어 악화했다. 추가/제거를 나눠 보면 TP +244/−174, FP +611/−2다. fragmentation에서도 0.847735→0.831255로 악화하여 추가 seed 학습은 하지 않는다. 한편 같은 평가의 XGB 단독은 0.843227, inner-selected control은 0.851174이므로 합법적인 기존 모델 조합·선택 경로는 보존한다. 과거 공식 최고와의 직접 비교는 아니다.

P1의 후속 binary decoder도 0.851174240→0.850291424로 악화했다. inner 선택은 Q2/Q3 OFF, Q4 ON이었으며 선택 전략의 outer 결과에서 TP 21개 추가/50개 제거, FP 0개 추가/8개 제거였다. 0 backbone fits, 6개 training-only 전이 추정, 3개 inner on/off 선택을 175.688초에 완료했다. decoder OFF의 기존 control을 전체 학습한다. 전체 과거 평가를 보고 가지를 채택/기각했으므로 최종 선택의 성능에는 선택 편향이 남으며, fresh 검증이 아니다.

참고로 raw always-on decoder는 0.852791538로 control보다 +0.001617298이었다. 이것은 **실패한 선택 절차와 보정 자체를 구분할 단서**다. 사전등록된 inner 선택을 outer 결과를 본 뒤 always-on으로 교체하지 않았고, 새 후보의 검증 성적으로 주장하지 않는다. 후속 실험은 보정의 존재 자체보다 inner→outer 선택 전이 문제를 우선 검토할 수 있다.

P2의 3개 fold와 3개 target layer에서 모두 RMSE가 감소했다. 다만 9~10월 fold는 0.492355→0.488284℃로 개선 폭이 작고, 전체 개선은 여름·연말 fold의 기여가 더 크다. seed별 RMSE 표준편차는 0.04138→0.04741로 늘어 평균 성능 개선을 seed 안정성 개선으로 부르지 않는다. 평가 구간은 재사용된 과거 데이터이며 fresh 검증이 아니다. P3는 6h의 181개 예측만 바뀌었고 2025 H1에서는 소폭 악화했다. seed 평균과 평균 RMSE를 혼동하지 않고, 평균 **예측**에서 SSE/전체 행수로 pooled RMSE를 다시 계산했다.

P3 신규 학습은 control 0.80655461m, weather-blockmask 0.81544321m로 증강 자체가 +0.00888860m 악화했다. 기존 clean 기준과의 고정 25% 결합도 개선되지 않아 이 학습 가지는 채택하지 않았다. 작은 6h 후보는 이 12-fit 학습의 성과가 아니라 별도로 사전 지정한 0-fit 상보성 비교에서 나온 것이다.

## 점수 환산의 한계

**새 후보의 공식 점수는 미확인이다.** 이전 계획서의 P1/P2 표시용 경험적 기울기는 공식 환산식이 아니므로 이번 결론에서 예상 점수로 인용하지 않는다. P2 내부 개선을 그 기울기로 계산한 +0.766점 역시 공식 상승 전망으로 채택하지 않는다. 과거 공식 P2 RMSE 0.424019℃에 이번 내부 개선량을 빼지 않는다. 표시용 환산은 학습 계수·임계값·가중치 선택에 사용하지 않았다. P3의 기존 공개 정책 구간을 따른 조건부 크기는 약 +0.0121점이지만, 내부 효과의 구간이 0을 포함하므로 공식 상승 예측은 아니다.

## 데이터 및 비교 계약

- P1: 배포 train만 사용한다. 21일 purge와 앞선 60일 inner를 두고, 정점/배포 수심 및 spike scale을 학습 split에서만 적합한다. gap을 넘지 않는 24~168h flank 25열만 추가한다. intact와 label-independent fragmentation을 분리한다. 과거 station/layer router의 선택 계보가 확인되지 않아 적격 control에서 제외했다. 새 control은 과거 28.909341점 모델의 exact clone이 아니다.
- P2: 배포 observations만 사용한다. 61/62/61일 복원창과 ±7일 purge, 공개층만의 특징, target temp/psal 격리를 적용한다. 마지막 fold는 자료 끝에 있어 뒤쪽 학습 관측이 없다는 한계가 있다. hidden QC는 제공되지 않아 finite target 및 공개 수온층 2개 이상을 proxy로 사용했다. v23/v52·연속 결측·실제 수심을 비교했으며 과거 bin17/답안 anchor는 입력하지 않았다. 실제 수심 추가는 1.03973275℃로 악화했고 채택하지 않았다.
- P3: 배포 train_wave/train_atmos에서 나온 고정 hash의 학습 특징과 clean OOF를 대조했다. 48h context·6 lead·78h gap, 181개 독립 episode 평가를 유지했다. raw weather block을 가린 뒤 요약한 결과와 가공 특징 masking의 합성 동등성을 검사했다. KMA/ERA5 또는 Public 역산 계수는 입력하지 않았다.

세 과제 모두 이번 내부 평가를 과거 공식 최고 성적 또는 모집단의 진정한 최고 모델로 해석하지 않는다. 최신 9월 2일 규정상 공식 점수로 역산한 계수를 상수로 저장해 재사용하는 것은 적격성 회복이 아니다. [정책 보충](../../configs/compliance/organizer_score_use_policy_20260902.json)에 구분했다.

## 독립 QA와 재현

- [수치 QA](independent-qa.json): 별도 코드로 key 고유성·배열 정렬·유한값·pooled 지표·fold 지표·변경 행을 재계산한다. 미완료는 PENDING으로 표시한다.
- [실행 완료한 QA 노트북](../../notebooks/parallel_score_repair_20260905_v1.ipynb): ignored OOF만 읽으며 학습/공식 입력/행별 관측 출력은 없다. nbclient로 code cell 4개를 순서대로 실행, 오류 0이다. 설치되어 있지 않은 nbconvert를 쓰려던 첫 시도는 실행되지 않았고, 의존성을 추가 설치하지 않고 사용 가능한 nbclient를 사용했다.
- [P1 계약](../p1_score_repair_20260905_v1/preregistration.md), [P2 결과](../p2_score_repair_20260905_v1/result.json), [P3 결과](../p3_score_repair_20260905_v1/result.json).
- P1 12 LightGBM + 6 XGBoost fits, calibration search 15회, 763.422초. P2 24 DeepSets fits + 15 OAS covariance fits, 435.454초. P3 12 LightGBM fits, 526.713초. 배포 fullfit은 별도로 센다.
- 공통 정책·합성 데이터·문제별 focused pytest **78개** 및 Ruff 통과(후속 decoder와 배포 계약 포함). 이 테스트 수는 모델 평가 fit 수와 다르다.
- P2/P3는 저장 모델 reload 일치를 확인했다. 전체 scratch 재학습의 bitwise 동일성을 자동 보장하는 진술은 아니다.
- 학습 단계 공식 입력 0행, CSV 0, upload 0. 배포 단계의 승인된 key/context 접근 및 생성 CSV는 별도 영수증에 기록한다. OS 전체의 모든 프로세스를 감시했다는 주장은 하지 않는다.

## 배포 준비

내부 평가 완료 후 별도 run ID에서 배포 자료 전체 학습 → 저장 모델 → 새 프로세스 추론 → schema/key/order/finite/hash QA로 진행한다. 과거 제출 CSV 또는 historical OOF를 최종 추론의 필수 입력으로 사용하지 않는다. 후보 준비와 공식 업로드·최종 모델 제출은 구분한다. 기존 최종 패키지는 덮어쓰지 않는다.

P1은 fullfit 2회 160.234초, 추론 14.250초, 복제 `02_code`와 해당 `src`만의 PYTHONPATH를 사용한 새 프로세스 replay 14.484초를 완료했다. 후보는 `artifacts/p1_clean_control_fulltrain_20260905_v1/05_answer/P1_submission.csv`, 169,011행/6,505양성, SHA-256 `064ef022faf2a3e8bc7c70633210847aa494060858374aa43f28f4eced84ec43`이다. root가 sample/test 키, schema·순서·이진값·2개 모델 및 봉인 recipe 해시를 독립 확인했다. 전 공식 행에서 2026 nominal-depth 사전 키가 unknown인 위험을 숨기지 않는다. [P1 재현 안내](../p1_clean_control_fulltrain_20260905_v1/README.md), [후보 독립 QA](candidate-qa.json).

P2는 fullfit 3회 66.188초, 별도 추론 11.454초를 완료했다. `artifacts/p2_score_repair_deploy_20260905_v1/submission_p2_v23_blockmask_3seed.csv`의 26,061행, schema/key/order/finite, 3개 저장 모델 SHA를 root가 독립 확인했다. 후보 SHA-256은 `46d194a1ef40a1deaebd084916644d9359433d2e6ce7d5c0b53d9f515bbec071`이다. [재현 및 제출 안내](../p2_score_repair_deploy_20260905_v1/README.md), [후보 영수증](../p2_score_repair_deploy_20260905_v1/predict-result.json).

P2 등록 학습 범위는 2024-05-01 이상/2026-01-01 미만이며 실제 유효 166,268행의 범위는 2024-05-08 18:00~2025-12-10 03:00 KST다. 가중치 영수증의 `month04`는 2025년 4월을 뜻하며 범위 밖 2024년 4월 자료가 아니다. 시작일 이전 유효 학습 행 0을 별도로 재확인했다.

P3는 원본에서 24,360 anchors/591 features를 697.831초에 다시 구성하고, 새 6개 모델로 과거 181사례의 기준선 OOF를 재생성했다. root가 키 정렬 후 검산한 1,086행의 truth·예측은 기존 clean OOF와 모두 동일했고 RMSE도 0.7791048399763751m로 동일했다. 이 비교는 재생성이 끝난 뒤의 읽기 전용 QA이며, 학습 러너가 과거 OOF를 입력한 것은 아니다. 새 평가가 아니라 같은 사례의 재현이다.

P3 배포 단계는 CatBoost 8 fits, router 3 fits, TabPFN fitted-context 1 fit을 완료했다. TabPFN 합성 가중치 자체를 gradient 학습한 것은 아니다. 저장 전후 별도 프로세스 예측 오차는 0이었다. 원본 재구성+학습/저장+fresh replay+공식 형식 추론을 직접 합산한 소요시간은 **1,678.413초(27분 58초)**다. 현재 RTX 5090/CPU 2 threads에서의 실측이며 운영진 장비의 6시간 보장은 아니다.

각 1,200행의 P3 후보 두 개는 root의 schema/key/order/finite/range/hash 검사를 통과했다. 혼합 후보는 6시간 200행만 바뀌고 나머지 1,000행은 기준선과 동일하다. [P3 학습·가중치·재현 안내](../p3_score_repair_deploy_20260905_v1/README.md), [최종 후보 QA](candidate-qa.json).

## 준비된 파일과 제출 상태

| 문제·용도 | 로컬 후보 | SHA-256 |
|---|---|---|
| P1 clean control | [P1_submission.csv](../../artifacts/p1_clean_control_fulltrain_20260905_v1/05_answer/P1_submission.csv) | `064ef022faf2a3e8bc7c70633210847aa494060858374aa43f28f4eced84ec43` |
| P2 내부 개선 후보 | [submission_p2_v23_blockmask_3seed.csv](../../artifacts/p2_score_repair_deploy_20260905_v1/submission_p2_v23_blockmask_3seed.csv) | `46d194a1ef40a1deaebd084916644d9359433d2e6ce7d5c0b53d9f515bbec071` |
| P3 clean 기준선 | [clean_baseline.csv](../../artifacts/p3_score_repair_deploy_20260905_v1/candidates/clean_baseline.csv) | `6bfa23d25f944df4711c11d1fce82978a96df08b58fdc57f666ac792a7da96b7` |
| P3 6h 탐색 후보 | [tabpfn25_6h_only.csv](../../artifacts/p3_score_repair_deploy_20260905_v1/candidates/tabpfn25_6h_only.csv) | `a7c7b247e5d74e7a0b6c8be42a7d4298a220b73881acb99490fcf7fe85f82f29` |

CSV는 각각 OCN-01/02/03의 답안 업로드용 **미채점 로컬 후보**다. 해당 정확 파일의 제출 승인이 있어야 업로드한다. 최종 모델 ZIP 제출과는 별개다. 기존 최종 패키지 불변, 업로드·commit·push·stage 0이다. 위 artifact 링크는 로컬에서만 존재하며 Git에 대용량 데이터·모델·답안 CSV를 넣지 않았다. [검증 실행 기록](qa-execution.json)과 [다음 연구 단서](gap-matrix.md)를 함께 보존했다.
