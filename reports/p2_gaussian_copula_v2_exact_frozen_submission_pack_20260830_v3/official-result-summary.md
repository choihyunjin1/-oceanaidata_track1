# P2 Gaussian copula v2 공식 결과 요약

결론: 이 exact frozen candidate는 공식 Public에서 기존 champion보다 악화되어 `PUBLIC_HARM / DO_NOT_REPEAT`입니다. favorable historical result는 `FROZEN_ALPHA50_PROXY`에 대한 exposed comparison이었고, 공식 분포로 전이되지 않은 `PROXY_COMPARATOR_TRANSFER_FAILURE`로 분류합니다. Private 결과는 아직 알 수 없습니다.

## 공식 결과

| 항목 | 후보 | 이전 champion | 후보 - champion |
|---|---:|---:|---:|
| Public RMSE (°C) | 0.442259 | 0.430209 | +0.012050 |
| 점수 | 27.784078 | 27.935277 | -0.151199 |

- 제출관리 표시: `2026-08-30 22:05 KST`
- 공식 확인 출처: root가 확인한 OCN-02 제출관리 화면
- 제출 후 당일 P2 잔여 횟수: `2/3`
- Private: `UNKNOWN_NOT_RELEASED`

## 후보 무결성

- 경로: `C:\Users\cedis\Downloads\해양 해커톤 제출용\20260830_P2_GAUSSIAN_COPULA_V2_FROZEN_READY_V3\P2_submission.csv`
- bytes: `1,353,537`
- SHA-256: `f498c6e1d7e22d11d5571b971454f0e375247fc2ff5ae3387bfcb4186460c4a3`
- model receipt: `446f70411d646fab4568a15080c09a374184e44bca669a5c954a34c25d867e7d`
- official receipt SHA-256: `198e2162dfe44d65a06468530ea8aa5ecdb71dc7be498e60bb954d4b73fbc248`

기록 과정에서는 후보 CSV의 bytes/hash만 다시 읽었고 행은 파싱하지 않았습니다. 새 학습·예측·CSV 생성·업로드는 모두 0이며, official test/sample/hidden truth와 `score.py` 접근도 0입니다. 원본 QA FAIL chronology와 tolerance adjudication PASS는 모두 그대로 보존했습니다.

## 결정

동일 SHA, 동일 frozen recipe 또는 그 결과에 반응한 근접 재제출은 금지합니다. 이번 Public 결과는 exact candidate의 승격 근거를 반증하며, 별도 사전등록 없는 shrinkage·window·edge 수정의 출발점으로 사용하지 않습니다.
