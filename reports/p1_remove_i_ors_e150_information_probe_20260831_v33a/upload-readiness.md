# P1 v33a REMOVE_I-ORS 업로드 준비서

## Materialized candidate receipt (22:30 KST)

- 상태: `MATERIALIZED_NOT_UPLOADED`
- 파일: `C:\Users\cedis\Downloads\해양 해커톤 제출용\20260831_P1_REMOVE_I_ORS_E150_INFO_PROBE_READY_V33A\P1_1_REMOVE_I_ORS_E150\P1_submission.csv`
- SHA-256: `c0cc4ad59008eeb39972a4ca1236a254abcc4786e64af750a3c047dc8ef94ab5`
- 행/양성/변경: `169,011 / 6,316 / 80`
- 독립 QA: exact I-ORS E150 removal mask, anchor·GI2 preservation, schema/key/order/binary/hash 모두 PASS
- 외부 업로드: `0` (사용자 action-time 확인 전)

## 결론

이 문서는 `p1_remove_i_ors_e150_information_probe_20260831_v33a`가 terminal 결과에서 **공식 제출 대상으로 선택된 경우에만** 사용하는 조건부 운영 체크리스트다. 이 문서를 작성하면서 후보 CSV를 생성하거나 열지 않았고, 브라우저 업로드·제출·최종 선택도 수행하지 않았다.

- 최신 공식 receipt(2026-08-31 21:22 KST) 기준 오늘 P1 제출 기회는 **1회** 남아 있다.
- 운영 safety line은 **2026-08-31 23:30 KST**다. 그 시각까지 v33a terminal 판정, exact-file 결속, materialization, 독립 QA가 모두 끝나지 않으면 서둘러 미검증 파일을 올리지 않는다.
- v33a가 동점·회귀·무효이거나 점수 확인이 불명확하면 기존 canonical champion `P1_1_E150_PLUS_GI_SPIKE2`를 최종 선택 기준으로 보존한다: Public F1 `0.833548`, points `28.909341`, rows `169011`, SHA-256 `57844ef235f987059d17bddb035cc0a98b92bb0d53263cf878c68b31bbc53687`.

## 제출 제목과 한줄 요약

- 제목: `P1 E150 REMOVE_I-ORS 공식 정보 probe`
- 한줄 요약: `공식 최고 E150 계보를 기준으로 I-ORS 양성만 사전 봉인된 규칙대로 제거해 정점별 수송 효과를 확인합니다.`

개선이 확인되기 전이므로 제목이나 설명에서 최고 성능·개선을 주장하지 않는다.

## A. REMOVE_I 선택 시 필수 QA

아래 항목은 모두 PASS여야 하며, 하나라도 실패하면 업로드를 중단한다.

### 1. 결과와 파일 결속

- [ ] v33a가 terminal 상태이며 기술 실패가 아니다.
- [ ] terminal 결과가 REMOVE_I를 제출 대상으로 명시적으로 선택했거나, 사전 규정된 information-value 기준을 통과했다.
- [ ] experiment ID, exact candidate 경로, 생성 runner/config hash가 result·manifest·QA에 동일하게 기록돼 있다.
- [ ] candidate SHA-256을 업로드 직전에 다시 계산했고 result·manifest·QA의 값과 일치한다.
- [ ] candidate byte size, 행 수, 양성 행 수를 aggregate로 기록했다.
- [ ] hidden truth 접근, 행별 leaderboard probing, outer 결과를 본 뒤의 규칙 수정은 모두 0이다.

### 2. P1 제출 계약

- [ ] 행 수가 정확히 `169011`이다.
- [ ] 컬럼과 순서가 정확히 `station, year, layer, time, label`이다.
- [ ] 공식 key frame과 `station, year, layer, time` 값 및 행 순서가 완전히 동일하다.
- [ ] key 결측 0, key 중복 0, 완전 중복 행 0이다.
- [ ] `label` 결측 0, non-finite 0, 정수형이며 값 집합이 `{0, 1}`의 부분집합이다.
- [ ] 예상하지 않은 extra column이 없다. `anomaly_type`은 ranked 제출본에 포함하지 않는다.
- [ ] 저장소의 공식 validator가 PASS한다: `.venv-p1\Scripts\python.exe scripts\validate_submission.py <exact-candidate.csv>`.

### 3. 봉인된 REMOVE_I 의미 검증

- [ ] candidate와 canonical champion의 key와 행 순서는 동일하다.
- [ ] changed rows는 모두 `station == I-ORS`에만 존재한다.
- [ ] 모든 변경은 champion `1`에서 candidate `0`으로의 제거뿐이다.
- [ ] I-ORS 밖 변경 0, `0 -> 1` 추가 0이다.
- [ ] 제거 행 수와 candidate-vs-champion hash/diff aggregate가 v33a terminal receipt와 일치한다.
- [ ] 위 비교는 정답을 사용하지 않으며 두 prediction의 구조적 차이만 확인한다.

### 4. 업로드 직전 UI 확인

- [ ] 올바른 문제 OCN-01/P1 화면인지 확인한다.
- [ ] 오늘 남은 P1 제출 기회가 여전히 `1`인지 다시 확인한다.
- [ ] UI에 선택된 파일의 이름과 로컬 exact candidate 경로·SHA가 대응한다.
- [ ] 제목과 한줄 요약을 위 문구로 입력한다.
- [ ] 23:30 KST safety line을 넘지 않았다.
- [ ] 업로드 후 timestamp, submission ID, 파일 SHA, Public F1, points, 남은 횟수를 별도 receipt에 기록한다.

이 문서의 범위에는 실제 브라우저 클릭, 업로드, 제출 확정, 최종 모델 선택이 포함되지 않는다.

## B. 점수 확인 후 champion 보존 절차

| v33a 공식 결과 | 운영 판단 | 최종 선택 기준 |
|---|---|---|
| F1 `> 0.833548`이고 points도 `> 28.909341` | receipt·SHA·schema QA가 일치할 때만 새 최고 후보로 등록 | 별도 승인 전 자동 finalization 금지 |
| F1/points가 기존 값과 정확히 동점 | 정보성 동점으로 기록 | canonical champion 유지 |
| F1 또는 points가 더 낮음 | 공식 회귀로 기록 | canonical champion 유지 |
| score 없음, validator/hash/schema 불일치, 업로드 오류 | 무효/미확정으로 기록하고 임의 재시도 금지 | canonical champion 유지 |

Canonical champion 보존 순서:

1. `P1_1_E150_PLUS_GI_SPIKE2`의 기존 파일·manifest·receipt를 이동, 덮어쓰기, 삭제하지 않는다.
2. v33a submission receipt는 별도 실험 ID 아래 기록해 champion 계보와 섞지 않는다.
3. best pointer는 **엄격한 공식 점수 상승과 독립 QA**가 모두 확인될 때만 변경한다.
4. 동점·회귀·모호한 결과에서는 canonical champion을 그대로 최종 선택 대상으로 둔다.
5. 모델 finalization 버튼은 별도 명시적 승인 없이는 누르지 않는다.
6. 최종 보고에는 champion 이름, F1 `0.833548`, points `28.909341`, rows `169011`, SHA-256을 함께 고정 기록한다.

## 근거 원장

- 최신 제출 횟수와 v30 회귀: `reports/parallel_public_transport_repair_cycle_20260831_v1/official-submission-receipt.json`
- P1 구조 QA 계약의 선행 사례: `reports/parallel_public_transport_repair_cycle_20260831_v1/pass-registry.json`
- 독립 QA 항목과 hidden access 0: `reports/parallel_public_transport_repair_cycle_20260831_v1/independent-qa.json`
- 기존 tie-best 공식 결과: `reports/parallel_internal_pass_registry_20260831_v1/official-submission-results-20260831.json`
- canonical champion lineage와 SHA: `C:\Users\cedis\Downloads\해양 해커톤 제출용\20260828_DEADLINE_INFORMATION_PROBES_READY\SET_MANIFEST.json`

작성 시각: 2026-08-31 22:21 KST. 후보 CSV는 읽지 않았다.
