# P3 numeric-lead 최종 후보 준비 계약

기존 내부 평균 개선 후보를 새 최종 모델로 구현한다. 새로운 탐색이나 hmax 제거와의 조합이 아니다. 기존 기준 CSV와 hmax 연구 후보는 변경하지 않는다.

- 원 실험: `p3_numeric_lead_forward_gpu_20260906_v2`, 같은 키 103,602행. 내부 RMSE 0.6840385926 → 0.6835381743m, Δ −0.000500418m, paired cluster CI90 [−0.001879861, +0.000809684]m. 작은 평균 개선이며 불확실성/악화 구간이 남는다.
- 원 검증 15 backbone + 8 router fits 및 149-check QA/새 PID replay는 그대로 보존. 결과/QA/OOF/코드/config를 exact SHA로 연결하고, 역사 모델은 읽거나 재학습하지 않는다.
- 새 학습은 24,360개 학습 anchor의 full numeric single(CPU2, 700 trees) + full multi(GPU0, Plain, 1,200 trees) + full OOF router 1개뿐이다. 새 3 fits와 과거 23 fits를 구분한다. 고정 seed 20260817, 591 features 및 hmax 파생 특징 유지.
- router는 동일 실험의 single/multi/persistence OOF와 training-derived 관측 특징만 사용한다. 103,602행 전체 OOF의 final router 학습은 배포용이며, 그 학습 오차/저장 probe를 일반화 성능으로 보고하지 않는다. 역사 평가의 이전-fold-only/purge 계약은 기존 149 QA를 그대로 인용한다.
- 고정 후처리: Ridge alpha 10, temperature 2, strength 0.5; 3/6/9h equal single/multi, 12/18/24h router + persistence 0.2. 0.2는 과거 local adaptive selection 뒤 고정된 재현 recipe이며, 새 virgin 증거나 Public 역산 계수가 아니다.
- full 학습 예상 약 5~8분, QA/두 PID 추론 포함 약 10~15분. 근거는 동일 CPU2 single의 22,808-anchor fit 241.47초와 GPU multi 24.64초다. fit 프로세스에 3,600초 watchdog을 적용하며, 전체 후보 준비 시간은 별도 측정한다.
- 합성 검사/Ruff/해시 preflight → 새 빈 full 모델 폴더 학습 → 새 PID 내부 저장 probe 재생 → 독립 native 모델/계보 QA → 공식 anonymous context/index allowlist 추론 → 별도 PID exact CSV bytes replay 순서다. 학습/QA 전에 official/sample/hidden/CSV/upload 0. sample 값/hidden/external/upload는 전 단계 0.
- 작업 중 원 소스/봉인 lock/기준 패키지는 불변. source-only cache와 역사 OOF를 재사용하므로 이번 실행을 독립 빈 폴더 전체 cold 재생성이나 두 번 scratch 학습으로 주장하지 않는다. 공식 점수/업로드는 root 담당이며 이 모듈은 수행하지 않는다.

실행 환경은 기존 `.venv-p1`; `P3_DATA_DIR`에 배포 P3 데이터 폴더를 지정한다. 새 runner의 `--stage preflight`, `fit`, `replay`, `qa`, `infer`, `verify-answer`를 각각 별도 프로세스로 실행한다. 원 materializer는 변경 없이 exact SHA로 재사용한다.
