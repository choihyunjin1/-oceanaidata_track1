# P2 clean restoration repair — 실행 전 계약

2026-09-05. 계획 `docs/SCORE_IMPROVEMENT_PLAN_20260905.md`의 P2 부분 실행.

- 기존 v23의 mean/max DeepSets와 v52의 mean/max/third-moment DeepSets, seasonal OAS의 순수 학습 커널을 재사용한다. 과거 bin17 답안, alpha/U/공식 역산 계수, 외부자료 계보는 읽거나 적용하지 않는다.
- 입력은 `P2_DATA_DIR/observations.csv` 한 개, 고정 SHA 검증. target temp/psal은 DeepSets 입력에서 항상 제외하고 OAS 평가창에서는 두 변수를 함께 NaN으로 치환한다.
- 2024-09~10, 2025-07~08, 2025-11~12의 시간 가림 3fold. 각 61/62/61일. 2024-05-01 이후 배포 관측을 양쪽에서 이용하되 평가창 전후 7일 purge. 원본 특징은 동시각 공개 프로파일과 시간의 결정적 주기함수이므로 의존 범위 0시간. 마지막 fold 이후에는 배포 학습자료가 없어 그 fold는 실제로 앞쪽만 사용한다.
- 유한 target, 공개 수온층 2개 이상만 평가한다. QC 열이 배포되지 않아 이는 proxy이며 hidden QC로 선별한 공식 모집단의 완전 재현이 아니다. 평가에서 손실이 큰 행을 결과에 따라 제거하지 않는다.
- baseline은 nominal depth에 대한 numpy.interp이며 범위 밖은 끝점 고정. 학습과 예측에서 동일 함수를 사용한다. target actual depth는 별도 arm에서 유효 수심/50와 presence를 기존 nominal context 뒤에 추가하며 미관측·비양수는 nominal로 대체한다.
- T5/S5 증강은 label-independent 3/7/14일 연속 calendar block, 고정 RNG와 약30% 시간 coverage. 원본을 보존하고 해당 관측을 가린 복제본의 baseline/scale/token을 재계산한다. 복제 가능 행은 원본/증강에 각0.5 가중치를 주어 원래 행별 총 가중치 보존. 2개 공개 수온 support가 남지 않으면 그 행의 증강 복제만 생략.
- 6fits v23/v52 screen → control의 blockmask 3fits → 같은 control의 actualdepth 3fits. 단일 ablation이 control을 이길 때만 두 모델의 2추가seed×3fold=12fits. 최대24 DeepSets fits. ablation이 모두 악화하면12에서 종료하며 중복 control confirmation은 하지 않는다. 각60epochs, AdamW/gradient penalty 기존 recipe.
- OAS covariance fits와 DeepSets fits를 분리 기록. OAS/선택DeepSets 50:50 하나는 사전고정 secondary diagnostic이며 공식 점수로 weights를 맞추지 않는다. 이번 연구는 재사용 historical folds의 탐색이지 fresh holdout이 아니다.
- CPU/BLAS1thread, loader0, GPU단독. RSS12GiB 상한. 하이퍼파라미터/epoch를 시간이나 결과 때문에 바꾸지 않는다. consumed attempt는 자동재시작하지 않는다.
- raw per-seed 모델예측, 모델가중치, row keys/truth는 ignored artifacts에만 저장한다. 집계·hash·fit receipt만 reports. 공식 입력·CSV·업로드·commit·push0. 최종 fullfit/materialize는 root의 별도 승인 이후.
