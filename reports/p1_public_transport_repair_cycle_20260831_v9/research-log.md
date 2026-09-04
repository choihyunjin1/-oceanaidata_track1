# Deep Research log and stop reason

## 질문

과거 add-only 후보가 local improvement에도 Public 동률이었던 상황에서, P1의 다음 후보가 최소 `+0.01` calibrated expected points를 낼 가능성을 어떻게 시간순 내부검증으로 걸러낼 것인가?

## 검색/대조 경로

1. 시간 순서 교차검증의 공식 계약을 scikit-learn 문서에서 확인했다.
2. add-only F1 marginal gain의 precision threshold를 F1 threshold 이론과 대조했다.
3. local/Public covariate shift를 별도 수송 위험으로 취급해야 함을 IWCV 1차 논문과 대조했다.
4. 시간 의존 행의 uncertainty를 행 iid가 아닌 day blocks로 평가하도록 block-bootstrap 문헌과 대조했다.
5. 로컬 v5–v8 실패 artifact를 읽어, 다음 미검증 가설을 row classifier가 아닌 target-free CAPA proposal-level posterior selector로 좁혔다.
6. family-aware calibration v2가 신규 learned router를 `HARD_CONDITIONAL_ROUTER`로 분류함을 봉인했다.

## 중단 기준

방법론 선택에 필요한 1차 출처가 포화되었고, 실제 sealed experiment가 가장 강한 반증(TP 0/FP 2046)을 제공했다. 추가 검색은 이 data-specific support 부재를 바꾸지 못하므로 research browsing을 종료했다. 결과를 본 뒤 threshold, daily budget, prior strength를 변경하지 않았다.

## 사용하지 않은 근거

- 블로그/튜토리얼/leaderboard 추측은 claim ledger에서 제외했다.
- Public score를 역추정해 hidden labels를 추론하지 않았다.
- v9 결과를 이용한 사후 proposal duration/bin 재선택은 하지 않았다.
