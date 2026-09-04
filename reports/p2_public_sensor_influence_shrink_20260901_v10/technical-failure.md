# P2 public-sensor influence sensitivity v10 — terminal technical failure

## 결론

`INVALID_ZERO_FIT_PUBLIC_LAYER8_TRAINING_SUPPORT_ABSENT`다. 사전등록 후보의 공개층 8에 등록 훈련창 exact-10-minute difference가 0건이어서 첫 prediction commitment와 metric 전에 fail-closed 됐다. 따라서 이는 과학적 NO_GO가 아니며 같은 ID 재실행은 금지한다.

두 zero-operation preflight는 byte-identical이었고 SHA-256은 `06d870632d8aa86dead5bfe0dd966f06cedaeef8a0dec59b7a4a99648f72aad3`였다. focused pytest 7/7, Ruff, pycompile도 통과했다. 그러나 preflight는 데이터를 열지 않는 계약이므로 layer 8의 실제 support 부재는 execute 때 발견됐다.

## support 진단

등록 훈련창(2024-05-01~2024-08-25 KST)의 exact-10-minute public-temperature difference 수는 L1 16,211, L5 16,161, L6 15,159, L7 16,221, L8 0이다. attempt lock만 생성됐으며 model fit 0, prediction 0, metric 0이다.

## v9r1 과학적 폐쇄

v9r1의 두 normalized-curvature Ridge 후보는 모두 1/3 fold만 개선하고 L2를 악화했다. 이 family는 `EXPLORATORY_NO_GO_BOTH_SEALED_CANDIDATES`로 닫는다. v10 기술 실패가 이 결론을 바꾸지 않는다.

## 다음 단일 미사용 축 제안

새 ID에서 **supported public-layer change coherence**를 가장 단순한 unused axis로 제안한다. L1/L5/L6/L7의 exact-10-minute signed change를 training-only layer median/MAD로 표준화한 뒤, 한 층만 cross-layer median change에서 크게 이탈한 경우에만 챔피언 보정을 고정 Huber influence로 감쇠한다. 이는 단일층의 큰 변화 자체를 이상치로 간주한 v10과 달리, 실제 수괴 변화처럼 여러 층이 함께 움직이는 경우를 보존하고 고립 sensor inconsistency만 묻는 질문이다.

행 삭제·fit·learned gate·threshold 탐색은 0, 정상 행은 bit-exact champion, 공개층 최소 3개가 없으면 no-op으로 한다. 새 ID와 새 lock이 필요하며 v10 lock은 재사용하지 않는다.

## 접근 경계

official test/sample/baseline/score/query support/hidden/submission CSV/upload 접근은 모두 0이다. v7 ready pack은 값이나 해시를 다시 읽거나 materialize/upload하지 않았고 우선순위도 기존처럼 “챔피언 기본, 슬롯을 반드시 쓸 때만 정보 probe”로 유지한다.
