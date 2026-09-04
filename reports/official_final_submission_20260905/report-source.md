# 공식 최종 제출 원자적 패키지 QA

## 결론

P1/P2/P3를 공유 실행 상태가 없는 독립 Jupyter package로 구성했다. 세 package 모두 운영진 배포 입력의 고정 SHA-256을 확인하고, frozen clean-lineage 후보의 정확한 CSV SHA-256까지 재현해야만 `READY_EXACT_NOT_UPLOADED`가 된다. 네트워크·hidden truth·외부 관측·재분석·예보·pretrained weight 경로는 없다.

## 설계 판단

하나의 notebook에서 세 문제를 순차 처리하면 kernel state, 상대경로, dependency monkey-patch와 산출물 덮어쓰기 위험이 서로 전파된다. 그래서 package마다 별도 `contract.json`, `assets/`, `outputs/`, notebook, runner를 둔다. 공통 guard 파일도 build 시 각 폴더에 복사되므로 실행 중 cross-problem import가 없다.

## 후보와 한계

- P1: `P1_1_E150_PLUS_GI_SPIKE2`, 169,011행, positive 6,396행, anchor 대비 add-only 2행.
- P2: `P2_V52_SCORE_PRIORITY_FULL_HISTORY_BLEND020`, 26,061행. historical scratch run이 checkpoint를 저장하지 않아 exact output을 동결한다. 3-fit training source는 포함한다.
- P3: `P3_REFINED_PUBLIC_OPTIMUM_20260827`, 1,200행. 12/18/24h 600행에만 frozen alpha를 적용하며 short lead는 bitwise no-op이다.

공식 최종 제출 modal의 per-file 50 MB 한도를 build-time hard guard로 넣었다. P1 checkpoint는 45 MB parts로 분할하며 전체 및 part별 SHA를 기록한다.

## 실행 검증

세 notebook은 각각 새 Python kernel에서 code cell 4개를 모두 실행했고 error output은 0개였다. P1/P2/P3 출력 SHA는 frozen 공식 후보와 각각 exact 일치했다. focused pytest 4개와 Ruff가 통과했으며, upload ZIP 18개는 모두 50,000,000 bytes 이하이고 ZIP integrity와 manifest SHA가 일치했다. P1 checkpoint 세 개는 조각을 디스크에 다시 쓰지 않고 순서대로 hashing하여 원본 전체 SHA와 일치함을 확인했다. 패키지에는 운영진 원본 데이터 파일과 일반 credential fingerprint가 없다.
