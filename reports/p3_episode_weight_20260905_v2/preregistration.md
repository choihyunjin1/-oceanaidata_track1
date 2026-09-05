# P3-B 사건가중치 고정 비교

실제 historical fit 전 고정. root가 P2-A 종료 뒤 P3 GPU 독점 사용을 승인했다. CPU2 threads, 단일 GPU0, official input/CSV/upload0, Git0이다.

- 배포 train에서만 생성한 24,360 anchors/591 features, 동일181 cases/1,086 targets 및 split/purge78h/6leads를 해시와 재계산으로 검증했다. 현재 첫 control seed-set은 이 적격 OOF의 single/multi 성분만 재사용한다. 완성 답안, 옛 router 계수, 외부자료 또는 Public 역산 계수는 재사용하지 않는다.
- control과 candidate의 유일한 recipe 차이는 `threshold_weight / sqrt(outer-train station×episode anchor count)`의 평균1정규화다. raw wave에서 ≥1.5m 연속20분 사건을 정의하고, 크기는 해당 outer-train ids 안에서만 센다. background는 사건으로 묶지 않는다.
- 첫 fold seed세트20260816/17/18, 둘째20260916/17/18. control 추가6 + candidate12 = historical CatBoost18 fits로 봉인한다. 실행 순서는 control 추가3fold, candidate 첫3fold, candidate 둘째3fold다. 결과에 따른 중간 범위 확장/재시작은 없다.
- 각 arm의 두 seed single/multi 예측을 먼저 평균하고, 그 arm의 새 OOF로 고정 router(alpha10/temp2/strength0.5)를 이전49/128 cases에서 각각 다시 적합한다. short3/6/9h router no-op, persistence shrink0.2는 보존한다. Router historical4fit 별도. A simplex/bias와 결합하지 않는다.
- paired episode효과는 새 seed-matched control 대비 평가한다. 실용 후보는 legacy/no-op, two-seed control, episode-weight 완성정책의 동일181case pooled postclip RMSE 최저로 고른다. 동점 legacy 우선. fold/lead/station/peak/CI는 위험 진단이며 평균 개선을 사후 hard gate로 차단하지 않는다.
- 이번 stage의 full backbone/router/official inference는0이다. 승자 배포는 root QA 후 별도 승인된 단계에서 seeds20260817/20260917 single/multi4fits와 fullrouter1fit이 필요하다.
- historical 예상30.5분, 별도 full재현·QA 포함45–75분은 과거 receipt 외삽이지 보장시간이 아니다.
- synthetic focused tests7 PASS, 실제 입력 preflight PASS, 실제 CatBoost CPU single/GPU MultiRMSE synthetic3iterations×2fits PASS, 저장/재로딩 오차0이다. Synthetic2fits는 historical18 budget와 분리한다.

runner SHA-256: `54591a09145c1a7552accb27db9c8b102fe629cfbd73cb9e6f2540b78ecb6705`

config SHA-256: `557384ee373b572043ea8394c4aaef82bcf9072f4d7e6488d4d0e161ad5bee3e`

실행: `P3_DATA_DIR`를 공식 배포 P3 경로로 설정하고 `.venv-p1/Scripts/python.exe scripts/run_p3_episode_weight_20260905_v2.py --mode execute --gpu-approved`. Lock 또는 output이 있으면 자동 재실행하지 않는다. 신규 artifacts/checkpoints는 로컬 ignored 폴더에만 둔다.
