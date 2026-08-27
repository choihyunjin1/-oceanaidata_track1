# P1 v1r6 독립 실행 후 QA

## 결론

실행·산출물 무결성은 **PASS**지만, 과학적 판정은 **`NO_GO_LOCAL_GATE`**입니다. 저장된 `result.json`의 결론과 독립 재계산 결과 사이에 차이는 없습니다. 이 버전은 연구 승격 대상도, 제출 대상도 아닙니다.

핵심 원인은 명확합니다. 후보 예측이 봉인된 로컬 OOF 421,032행 모두에서 Round-B anchor와 동일했고(`different rows = 0`), 최종 decoder가 구조한 행도 0개였습니다. 세 seed별 후보도 각각의 anchor와 전부 동일했습니다. 따라서 새 residual 모델이 최종 분류 행동을 한 행도 바꾸지 못했습니다.

## 독립 검증 범위와 안전성

- 기존 one-shot을 재실행하지 않았고 QA 과정에서 fit을 수행하지 않았습니다.
- 공식 test, sample, submission 및 candidate-submission 경로를 읽지 않았습니다.
- submission 생성·업로드를 하지 않았습니다.
- 봉인된 실험 artifact는 수정하지 않았습니다.
- 허용된 로컬 frozen truth와 prediction을 읽어 키를 one-to-one 정렬하고, 프로젝트 scorer를 호출하지 않은 별도 구현으로 모든 aggregate metric과 gate를 재계산했습니다.
- 행 단위 데이터는 출력하지 않았습니다.

## 실행 및 산출물 무결성

- manifest record 39개를 파일 크기와 SHA-256으로 전수 검증: PASS
- journal 28개 전체 해시 체인: PASS
- physical fit: reserved 9 / completed 9
- terminal: `998_worker_terminal` 1회 후 `999_completed` 1회
- failure terminal 0, `initialization_failed.json` 없음, `execution.lock` 없음
- 실행 경과 210.29537892341614초, global deadline 잔여 1531.0634002685547초
- Q2/Q3/Q4 left-censor count: 모두 0
- 세 prediction part의 순서 있는 결합과 aggregate prediction 421,032행이 dtype·값까지 정확히 일치

Canonical SHA-256:

- manifest: `0d2c0434b7f6eb4053c6c3935e6b18f6333213f4230673f6885d2f98701e76bb`
- result: `9136acf5c1f7072e29a3ee7456795dd0c0ba8f1785f7a7d8a6aab4588aa8064c`
- metrics: `5d0c6c6858cadf4283d32e6faf2974cc63a8e03a019492adc59d944219e40535`
- predictions: `953415097b2f43421cf40ffe98100f305e3fbd2ba1215656acc4d13bf2b8ec93`
- predictions complete: `7f63f34fb7d17a3f0330c8fcef5bc0278a11f054fcaecbb5d45173e0dc8ef8d1`
- frozen local truth: `d1b9439db6d0d906fa080bd01f1eb8fc21d051c3d056a274e2b02e43c1e55f4a`
- journal 998: `bd8f3c42db52b801d47e958004cd4495999be3cb8002b62a8dbf1fea64c236f2`
- journal 999: `8412ce02f697a8d475a07955eb54499e79b9f0592f69d87a3e7462a929f387e4`

## 독립 재계산 결과

- pooled: TP 12,718 / FP 644 / FN 3,337 / TN 404,333
- F1 `0.8646700887242071`, precision `0.9518036222122437`, recall `0.7921519775770788`
- 후보의 F1·precision·recall delta: 모두 `0.0`
- paired bootstrap: 5,000회, 3,089 blocks(positive events 141 + normal station-layer KST days 2,948)
- bootstrap delta CI90 `[0.0, 0.0]`, 평균·중앙값 `0.0`, 개선 확률 `0.0`
- 3 folds 및 3 stations의 F1 delta: 전부 `0.0`
- equal-weight station-fold 8 cells: delta `0.0`
- adequately-supported 7 cells의 worst delta: `0.0`
- 세 seed F1 delta: 전부 `0.0`
- normal FP/day ratio `1.0`
- spike 61행 recall: anchor와 후보 모두 `0.8852459016393442`
- 제거된 anchor-positive, rescue, 신규 disconnected event, 신규 singleton: 모두 0

저장된 `metrics.json` 전체 구조와 수치를 절대 허용오차 `1e-15`로 재현했습니다.

## Gate 판정

다음 네 개선 gate가 실패했습니다.

- pooled micro F1 delta: 실제 `0.0`, 요구 `>= 0.003`
- bootstrap CI90 lower: 실제 `0.0`, 요구 `> 0`
- equal-weight station-fold F1 delta: 실제 `0.0`, 요구 `>= 0.0015`
- recall delta: 실제 `0.0`, 요구 `>= 0.008`

fold/station 비열화 수, worst supported cell, all-seed, precision, FP/day, spike, disconnected/singleton 안전 gate는 통과했지만, 이는 후보가 anchor와 동일한 데서 나온 보존 결과입니다.

## 최종 조치

`v1r6`에는 leaderboard 제출 기회를 사용하지 않습니다. 모든 사전등록 gate를 통과해야 얻는 상태도 `GO_LOCAL_SCREEN_ONLY`일 뿐 제출 승인이나 공식 점수 보장은 아닌데, 이번 버전은 그보다 앞선 local improvement gate에서 실패했습니다. 이 결과는 유효한 null-result 영수증으로 보존하고, 다음 실험은 rescue threshold만 사후 조정하지 말고 구조 행이 실제로 생기면서도 사전등록 gate를 검증할 수 있도록 별도 설계해야 합니다.

P0/P1/P2 무결성 finding은 없습니다. 과학적 null result만 확인되었습니다.
