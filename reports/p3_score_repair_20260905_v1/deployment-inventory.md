# P3 배포 경로 정찰 — 공식 입력 접근 전

아래는 공식 입력 승인 전의 **배포 계획 정찰 기록**이다. 이후 별도 승인된 source-only 경로가 완료되었으며 최신 상태는 [새 배포 README](../p3_score_repair_deploy_20260905_v1/README.md)와 [완료 보고](../p3_score_repair_deploy_20260905_v1/report-source.md)를 따른다. clean 기준선 및 6h 후보 2개 로컬 CSV, 9 backbone/3 router fit, fresh-process reload 오차0, 총1,678.413초가 확인됐다. 업로드/최종ZIP/격리환경 검증은 하지 않았다. 아래의 '추가 준비 필요' 등은 당시 실행 전의 상태이며 새 완료 증거를 대체하지 않는다.

## 새 LightGBM 단독 또는 2-seed 평균

- train feature 재생성: `src/p3_wave/data.py:197`의 `build_training_grid`, `src/p3_wave/features.py:140`의 `build_training_features`. `P3Data`의 train wave/atmos만 채우고 다른 입력은 빈 frame으로 두면 공식 파일을 열지 않고 재사용 가능하다. 현재 screen cache는 불변 source 및 cache SHA가 대조됐다.
- 학습: 이번 runner의 동일 residual target, 591개 compact features, 700-tree 고정 recipe를 전체 24,360 anchor에 적용한다. 선택된 arm의 2 seeds full fit을 새 별도 영수증으로 센다. 이것은 screen 12 fits에 포함되지 않는다.
- 저장: LightGBM native txt 2개와 열 순서·station/lead categorical schema·clip/mean 정책·입력 hash를 저장한다. 기존 예측 CSV와 historical OOF는 inference 입력이 아니다.
- 별도 predict 프로세스는 승인된 이후 해당 case의 289행만 `summarize_context`로 요약하고, lead별 residual+current를 0~30 clip한 후 두 seeds를 평균한다. 이 순서를 screen과 같게 유지한다.
- 현재 screen은 저장 txt를 새 Booster 객체로 reload하여 최대 절대오차 ≤1e-12를 검사한다. 별도 프로세스 최종 inference와 full-retraining 재현성은 아직 수행하지 않았다.

## Clean corrected fallback을 포함하는 경우

- 과거 구현: `scripts/run_p3_corrected_repeated_forward_catboost_v1.py:251`의 fold single/multi 학습, `:357`의 prequential router/고정 0.2 persistence shrink, `:546`의 full-fit 경로.
- archive exactly-once runner를 다시 실행하지 않는다. `:546`은 학습 뒤 모델 저장 **전에** official feature/index를 읽고 CSV를 쓰므로 현재 권한에서 호출하면 안 된다.
- 순수 재현 패키지는 source로부터 3fold single+multi 6 fits를 재생성해 router용 train OOF를 만들고, full single+multi 2 fits 및 router fit을 수행한다. 총 CatBoost 8 fits이며 router fitting은 별도로 센다. 새 run 내부 OOF는 허용되는 중간 산출물이지만 과거 OOF를 필수 동봉 입력으로 삼지 않는다.
- 모든 모델/라우터를 저장한 뒤 별도 predict 프로세스로 분리해야 한다. 기존 제출 CSV, refined Public-alpha, KMA correction은 어떤 단계에서도 사용하지 않는다.
- 당시 실측: `artifacts/p3_corrected_repeated_forward_catboost_v2/metrics.json`의 fold pairs 88.688/90.146/109.627초, 전체 screen+full-fit+추론+QA 854.828초. 별도 feature cache 생성 기록은 309.347초다. 당시 CPU 8threads+GPU 조건이므로 현재 2threads의 완료시간 보장은 아니다.

## 6h-only TabPFN25를 선택하는 경우

- standalone historical OOF는 `scripts/run_p3_tabpfn3_structural_transition_20260901_v1r1.py:200` 이후 실제 `LEADS=(3,6,9,12,18,24)`를 순회한다. 과거 config의 `leads_h=[1,2,3,6,12,24]`는 실행 영수증과 불일치하는 metadata 결함이며 신규 code/config에서는 그대로 복제하면 안 된다.
- 6h feature 변환은 같은 파일 `lead_specific_matrix`(:70), local-only 모델 생성은 `src/ocean_tabpfn3/offline.py:245`를 참고한다. full 6h fit 1개가 추가되며 GPU 자원 재배분이 필요하다.
- 기존 18 lead×fold fit+prediction 실측은 251.536초다. 최대 fold train case는 20,899; full 24,360case의 1lead 실행시간/저장/reload는 아직 실측되지 않았다. 단순 18분의1로 완료시간을 보장하지 않는다.
- local regressor checkpoint SHA-256 `311ce18d97e9533d8585eaadafe040fbdd8070533209ed8696641dadc97a7301`을 이번 정찰에서 다시 검증했다. `artifacts/tabpfn3/user-license-receipt.json`에는 2026-09-01 사용자 수락 및 synthetic-only provenance review 기록이 있고 credentials는 저장하지 않았다고 명시한다. `configs/compliance/tabpfn3_offline_transition_20260901.json`에 synthetic-only 공식 근거와 offline 조건이 있다.
- 이 기록만으로 새 후보가 6시간 재현을 완주했다고 선언하지 않는다. 정확 weights 동봉, local-only load, README 공개, 별도 predict, 실제 전체 wall time 검증이 필요하다. 현재는 **추가 준비 필요**다.

## 공통 중단선

official test_context/test_index/sample/hidden/답안 CSV 읽기와 upload는 이 screen에서 0이다. full training 준비 및 공식 materialization 권한은 구분한다. GPU는 P2 예약 상태이므로 fallback/TabPFN 단계 전에 root 자원 배분을 받는다. 기존 패키지는 수정하지 않는다.
