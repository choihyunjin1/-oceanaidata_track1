# P2-A 실행 계약 — 절대 ℃ 목적함수 대조

이 실행은 과거 bin17 최고점 복제가 아니라 2026-09-05 공식 `0.455143℃ / 27.622418점`을 받은 clean blockmask 모델 C와 새 목적함수 후보의 내부 비교다. 현재 공식 점수는 새 후보의 성적이 아니다.

- 유일 원본은 `P2_DATA_DIR/observations.csv`, immutable hash `cc5d6fd9ea398a613e485b34fd530c7dc064fa040fa675127a12318856ab178a`.
- 학습 입력/정답은 배포 관측으로 분리하고 target layer2/3/4의 temp·psal은 공개층 특징에 전혀 포함하지 않는다. 공식 test/sample/baseline/hidden 입력은 읽지 않는다.
- C 기존 학습9회는 신규fit에 포함하지 않고 clean raw OOF의 key/truth/fold와 저장모델 재추론을 exact 대조한다. 옛 실험의 lock/config/코드/모델은 수정하지 않는다.
- M/R/MR은 `normalized Huber 대 absolute-C MSE × domain 대 original-row-uniform` factorial의 새3칸이다. 기존 normalized-Huber input-gradient penalty, 그 domain weights와 계수0.01은 네 칸 모두 같다.
- 원본별 증강 전 가중치 총량은 원본/복제에 보존한다. 입력/target scale은 공개 프로파일만 이용하며 ℃ MSE는 `scale² × normalized residual²`다. loss 단위가 바뀌므로 정규화항의 상대 크기 차이는 남아 있고 사후 lambda 튜닝하지 않는다.
- primary는 2024-09/10 intact RMSE로 실행 전 고정. 전체3fold pooled RMSE·layer·가을 outage는 보조 위험표다. stress는 2024-10-18부터11-01 전까지 T5/S5를 같이 가리고 baseline/scale/tokens를 새로 만든다. stress에서 공개 수온2개 미만은 support 부족으로 별도 분모 기록하며 intact 평가행은 삭제하지 않는다.
- 첫seed9fits 후 primary→pooled→단순성 순으로 선택하고, C가 아니면 승자만 추가2seeds×3fold. 최대15new historical fits. 60epochs 고정, GPU단일소유, CPU1thread, DataLoader0. 비정상종료 자동재시작 금지.
- 과거평가면 재사용이므로 development 탐색 결과다. bootstrap CI가0포함하거나 한 계절이 악화됐다는 이유만으로 자동탈락시키지 않는다. 가을3seed평균이 개선되지 않으면 정의된 P2-B로 진행한다.
- 이 연구 단계는 fullfit/공식추론/CSV/upload/commit/push를 포함하지 않는다. root 독립QA 이후 새 frozen deploy ID에서 진행해야 한다.

재현 가능한 코드: `scripts/run_p2_objective_alignment_20260905_v2.py`, `scripts/qa_p2_objective_alignment_20260905_v2.py`, `tests/test_p2_objective_alignment_20260905_v2.py`. 실행 전 고정지문은 [preregistration-seal.json](preregistration-seal.json). 해시를 바꾸거나 기존attempt lock을 지우고 재실행하지 않는다.

`validate-data`와 `analyze-data-quality` 검증지침을 적용해 분모·동일모집단·단위·키·원본별 증강질량·저장모델 재현을 검증한다. 원본 행 값은 tracked 보고서에 기록하지 않는다. 보고서는 요청된 작은 Markdown/JSON 형태이며 결과 노트북/최종 패키징은 root의 통합 경로에서 연결한다.
