# P2 OAS alpha 0.50 배포 준비 결과

상태: `OFFICIAL_PUBLIC_SCORE_CONFIRMED_IMPROVEMENT`
실행일: 2026-08-28 KST
업로드: 완료

## 공식 결과

- Public RMSE: `0.431252`
- 점수: `27.922187`
- 이전 최고 alpha 0.40 대비 RMSE 개선: `0.013895 ℃`
- 이전 최고 대비 점수 상승: `0.174340점`
- Private 지표: 마감 후 공개
- 오늘 남은 P2 제출: `1/3`

관측 RMSE는 사전등록 범위 `[0.424378485886, 0.439880115136]` 안에 있으며 중심 예상 `0.432198548994`와의 차이는 `-0.000946548994 ℃`다. 따라서 동일 공식 scoring set에 대한 점수기하 가정과 구현은 이번 한 건에서 확인됐다.

## 정확한 제출 파일

- 파일: `C:\Users\cedis\Downloads\해양 해커톤 제출용\20260828_P2_SEASONAL_OAS_TS50_PROJECTED_READY\P2_submission.csv`
- 바이트: `1,276,414`
- SHA-256: `bd550127cfbab9bcd2df75ad7d3fb65dafdf62568fca628851b0e0ae1dc241d5`
- 제목: `P2 계절 국소 T/S 조건부 프로파일 OAS 50% v1`
- 한줄요약: `기존 U를 50% 유지하고 공개층 T/S의 계절 국소 OAS 조건부 프로파일을 50% 결합한 뒤 endpoint/PAVA 물리 투영을 적용했습니다.`

## 검증 결과

- 26,061행, 열 `station,layer,time,temp`
- 층별 행 수 2/3/4 = 8,713 / 8,712 / 8,636
- 공식 test와 sample의 키·순서 일치
- 키 결측·중복 0, temp 유한·허용범위 통과
- PAVA 재투영 최대오차 0
- 별도 재실행 파일과 byte-identical
- 숨은 정답·mirror 미사용
- 공식 업로드 미실행

## 조건부 공식 RMSE 범위

기존 alpha 0.00/0.10/0.20/0.40 공식 RMSE와 실제 예측 벡터로 계산했다.

- 중심: `0.432198548994`
- 6자리 점수 반올림 강건 범위: `[0.424378485886, 0.439880115136]`
- alpha 0.40 공식 RMSE `0.445147` 대비 조건부 최소 개선: `0.005266884864 ℃`

이 범위는 동일 26,061행·동일 통합 RMSE scorer라는 조건에서만 유효하다. 공식 점수가 범위를 벗어나면 같은 축의 추가 계수 탐색을 즉시 중단한다.

## 근거 파일

- 설정: `configs/experiments/p2_seasonal_oas_alpha50_deploy_20260828.json`
- 생성기: `scripts/build_p2_seasonal_oas_alpha50_20260828.py`
- 독립 QA: `scripts/qa_p2_seasonal_oas_alpha50_20260828.py`
- 실행 receipt: `artifacts/p2_seasonal_oas_submission_20260828_v4_alpha50/receipt.json`
- QA 결과: `artifacts/p2_seasonal_oas_submission_20260828_v4_alpha50/independent_qa.json`

다음 단계는 alpha 0.50 공식 점수를 포함해 기하를 다시 계산하는 것이다. 남은 한 장을 자동으로 사용하지 않는다.
