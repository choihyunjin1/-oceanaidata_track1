# P2 C60/L120 검증 범위 확장 — 학습 전 계약

가을 B3 primary와 기존3seed 수치를 그대로 보존한다. 이번 목적은 이미 준비된 L120 후보를 새로 선택하거나 좋은 표면으로 primary를 바꾸는 것이 아니라 **나머지7fold의 추가2seed로 전체8fold3seed secondary 증거를 완성**하는 것이다. 기존 후보/공식답안/모델/lock은 바꾸지 않는다.

기존 p2_c3_training_comparison_20260906_v1의 C60/L120 각 firstseed8모델+B3추가2모델, 총20개 모델과전체평가 예측을 exact SHA로 재사용한다. 남은 B1/B2/B4/B5/B6/B7/B8에서 seed20260902/03 × C60/L120만 새로28회 학습한다. 합계48개 historical 모델을8fold×2arm×3seed로 연결한다. 기존 D60 모델과완료된 attempts는 재실행하지 않는다.

특징/입력166268키/target temp+psal firewall/7-day 양쪽purge/domain weights/blockmask/gradient coefficient/무projection은 이전 sealed helper/core와 exact 동일하다. 새 fit의 입력 배열SHA는 같은fold firstseed와 같아야만 학습한다. C60은60epoch, L120은120epoch, 둘다wd0.0001·GPU0/CPU2/DataLoader0이다.

평가는 B3 primary와전체8fold secondary의 natural/outage wholefold/outage 실제17일 구간/자연T5결측/layer를 병기한다. 먼저각arm 3seed의예측을평균한후 pooled sqrt(SSE/n)을계산한다. fold RMSE 평균은 금지한다. B4/B8의 outage평가행0은 NOT_ESTIMABLE_NO_ROWS로남긴다. unsupported가있으면scenario전체지원불가이며 행삭제로점수를만들지않는다.

기존 B3 3seed arrays exact equality를 강제하고 새로운후처리/조건부routing/후보선택0이다. 일부계절/outage 악화는 위험으로보고하되 자동veto로추가하지않는다. CI90은 KST2024-01-01원점7day공통block/2000회/seed20260906으로고정하며 공식개선확률로해석하지않는다. 이미노출된검증의반복이라fresh holdout으로표현하지않는다.

예산: 신규28 fits/기존20 exact reuse, GPU0독점CPU2, training wallcap3600초. 이전28fits757.673초와 이번42개의60epoch환산량을고려하면 준비/평가/전체48모델재생포함약15~20분계획이다. 예산초과는기술/예산terminal로보존하며자동재시작하지않는다.

실행: synthetic/Ruff → prefit seal/reuse20SHA확인 → execute → 새PID전체48모델 natural/outage validation행 exact replay → 독립SSE/키/seed평균/불확실성QA. 기존모델을new fit의warmstart로로드하지않는다. 재사용허용은sealed20개의binaryhash/평가예측/후속replay로한정된다. 공식입력/CSV/업로드/Git0이다.

후속 L120 portable/cold ZIP은 별도준비만가능하며 실제cold full3fit은root의새GPU배정전실행하지않는다.
