# P1 수심 계약 v2 사전등록

2026-09-05 KST. 연구용 배포 train만 읽는 1회 실행이다. 공식 입력/답안/업로드는 포함하지 않는다.

- 실행 코드 SHA-256: `4f6db75d1ba45cba6e5bd4159712a6d1ab1851e691dbb21c371bbed09b5e2a0c`.
- 설정 SHA-256: `f6514ffe6fe868d4d44f01e4b6c57dd6b702404878764e200ac1caec0ff9d972`.
- 주평가: 기존과 정확히 같은 Q2/Q3/Q4 421,032행 pooled TP/FP/FN의 micro F1. 역사 평가면은 이미 노출된 development이며 fresh holdout이 아니다.
- 변경: nominal-depth의 `(station,year,layer)` train lookup 대신 현재 관측 depth의 고정 2m 반올림. explicit missing은 유지. 이외 offline 관측 특징, 모델 recipe, seed, purge21일, inner60일, threshold grid는 고정.
- control: 구 모델 12개 재로딩. 구 결과·설정·import dependency·recipe·각 모델 hash를 실행 전에 확인했고, 실행 중 inner threshold 및 outer key/probability/binary의 exact replay를 확인한다. 불일치는 성능 결과가 아니라 기술 중단이다.
- 신규 학습: O/B × 3fold × inner/outer = 12, 최종 train 끝 기준 inner60일 2, 전체학습 2; 합16 fits. CPU4, GPU0, 45분/12GiB 상한. 캡은 각 단계 경계에서 확인하므로 단일 fit의 중간 강제종료 타이머는 아니다.
- 선택: 각 fold earlier-inner에서만 original/balanced/독립 union/재보정 union 4개 완성정책과 기존 grid 선택. control의 threshold를 그대로 적용한 진단은 별도 열로 구분한다.
- A 평균 개선 여부와 무관하게 final-inner/fulltrain을 완료하여 year-safe 후보를 저장하되, 악화일 때 improvement 후보로 부르지 않는다. 실제 공식 materialization은 별도 root QA 지시 후에만 가능하다.
- B는 기존 router/e150/GI의 출처와 split/purge/key를 별도 정찰한다. 부적격 OOF를 새 tree와 임의 결합하지 않는다. 새 MS-TCN GPU 학습은 root 승인 전 시작하지 않는다.
- C는 A/B 이후에만 고정 OFF/always-ON을 별도 대조한다. A runner에는 decoder 적용이 없다.
- synthetic pytest 9 PASS, Ruff PASS, control의 기록된 18개 fit 중 O/B 12개 hash 및 dependency/recipe 검사 PASS. 실행 코드/config는 이후 변경하지 않는다.
