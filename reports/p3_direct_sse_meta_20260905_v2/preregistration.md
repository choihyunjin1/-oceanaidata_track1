# P3-A: 저자유도 직접 SSE 보정 사전 고정

상태: 실제 메타 적합 전 고정. 실행 승인은 `docs/SCORE_IMPROVEMENT_PLAN_20260905_V2.md`에 대한 사용자 승인 범위이다. 공식 입력, 제출 CSV, 업로드, Git 작업은 포함하지 않는다.

- 기준: 배포 train만으로 재생성한 181 cases × 6 leads OOF, 기준 RMSE 0.7791048399763751 m. 기존 공식 답안/공식 점수 역산 계수/외부자료 유래 값은 재사용하지 않는다.
- 대안은 정확히 no-op, 12/18/24h 공통 single/multi/persistence simplex(2자유도), 전체 lead global bias(1자유도)이다. 직접 과거 SSE/평균 잔차만 적합한다. 별도 station 또는 lead별 계수 탐색은 하지 않는다.
- 첫 49-case fold는 기존 예측을 그대로 유지한다. 두 번째 79-case fold는 과거 49 cases, 마지막 53-case fold는 과거 128 cases만 적합한다. 미래 fold를 포함하는 leave-one-out은 금지한다.
- 적합 label은 해당 fold 시작 전 모두 관측 완료됨을 검증했다. 같은 정점의 최소 anchor 간격은 112h, 180.667h, episode 중첩 0이다. 78h 규칙은 정점별이며 정점 간 footprint 겹침을 사후 제거하지 않는다.
- 최종 판단은 181 cases pooled post-clip RMSE 최소, 동점은 no-op → bias → simplex 순이다. 0–30m clipping 고정. fold/lead/station/관측 case-peak 상위 10%는 진단만 하고 결과 기반 후보 변경에 쓰지 않는다.
- 예산: historical meta 4 fits, 선택된 비-no-op full meta 최대 1 fit, backbone/GPU 0. CPU 최대 2 threads.
- 5,000회 complete-case bootstrap은 과거 개발자료의 기술적 불확실성 요약이다. 새 독립 검증이나 선택편향 제거로 주장하지 않는다.
- 실행 전 synthetic focused pytest 7 PASS, Ruff PASS, 실제 입력 정렬/완전성/target 동일성/시간격리 read-only preflight PASS. 실측 성능은 아직 열지 않았다.

고정 runner SHA-256: `79e63357e135556e7fb915492530048bcd13614aaa1f8cb74bd2532b0b6bfdb1`

고정 config SHA-256: `0a6f70bcc96d5bb848e125153787f38bf9a8fada7531223ff5761a7d1d8e52d2`

실행 명령: `P3_DATA_DIR`를 공식 배포 P3 폴더로 설정한 뒤 `.venv-p1/Scripts/python.exe scripts/run_p3_direct_sse_meta_20260905_v2.py --execute`. 이미 attempt lock 또는 결과 폴더가 있으면 재실행하지 않는다.
