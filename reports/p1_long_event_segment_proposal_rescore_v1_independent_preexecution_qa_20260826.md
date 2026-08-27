# P1 Cycle-1 long-event 독립 사전 감리

## 결론

**현재 봉인된 v1은 `NO_GO (P0)`입니다. 실행 승인하면 안 됩니다.** 과학 설계와 Round-B anchor 선택 자체는 보존됐지만, `execute_worker`가 readiness 확인 직후 무조건 `AuthorizationError`를 발생시키는 zero-fit stub입니다. 따라서 실제 authorization digest를 넣어도 72개 fit, bootstrap, gate 판정, 결과·manifest 게시 중 어느 것도 수행하지 못합니다.

이번 감리에서는 모델 fit, feature/proposal materialization, 공식 test/sample/submission/candidate 열람, 제출 파일 생성·업로드, P3 접근을 모두 0으로 유지했습니다.

## 통과한 부분

- 고정 design SHA-256은 `31b0bde...9563`이며 변경되지 않았습니다.
- trigger resolution은 v1r4의 placeholder 형태를 재구성하고 v1r4→v1r6의 8개 수치 섹션 동일성을 확인한 뒤, v1r6의 terminal scientific `NO_GO_LOCAL_GATE`에 따라 **오직 `FROZEN_ROUND_B` branch**를 선택합니다.
- 정적 예산은 anchor 9 + inner segment 54 + outer segment 9 = 총 72 fits, materialization 최대 21로 일치합니다.
- held outer surface는 구 XGB 열이 아니라 정확한 `event_day_balanced_binary_lgbm__probability`와 `__prediction` 열을 읽도록 되어 있습니다.
- Round-B 원래 3개 seed, LightGBM 고정 파라미터, event/day weight primitive, uniform postprocess 0.2/0.1/0/12가 코드와 amendment에 고정돼 있습니다.
- focused test는 `18 passed`, ruff는 `All checks passed`였습니다. 다만 테스트가 현재의 영구 실행 차단을 성공 조건으로 삼으므로, 이 green 결과는 실행 가능성의 증거가 아닙니다.

## 차단 결함

1. **P0 — 실제 수치 실행 경로 없음.** `scripts/run_p1_long_event_segment_proposal_rescore_v1.py:1195-1204`는 authorization/static/data preflight 뒤 무조건 예외를 던집니다. `AttemptJournal.begin`, 72-fit loop, inner selection, outer one-shot, 5,000회 bootstrap, gate 계산, `result.json`/`manifest.json`/`report_ko.md`/`999_completed.json` 게시가 연결돼 있지 않습니다.

2. **P0 — authorization이 독립 QA에 transitively 묶이지 않음.** 현재 검증기는 authorization이 스스로 제공한 dependency/input/outer map을 그 map 자체에 대해 확인합니다. preexecution seal, strict preflight, 독립 QA JSON/MD의 hash와 PASS verdict, sealed template과의 exact map equality를 요구하지 않습니다. 즉 이름만 `AUTHORIZED_INDEPENDENT_QA_PASS`인 파일이 실제 감리 대상을 바꿀 수 있습니다.

3. **P1 — 선언 시각의 순서가 불가능함.** authorization template은 `08:20:00` 생성이라고 선언하지만, 그것을 읽어 봉인했다는 seal은 `07:52:22`, strict preflight는 `07:52:30`입니다. hash는 맞아도 현재 receipt chronology는 신뢰 증거가 될 수 없습니다.

4. **P1 — segment 학습용 anchor surface의 out-of-sample 계약이 없음.** 추가된 9개 Round-B fit은 각 inner validation용 prefix model을 만들 수 있지만, segment classifier의 라벨된 학습 proposal 행에 어떤 anchor probability를 쓰는지 정의·구현하지 않았습니다. 같은 행의 label로 fit한 anchor probability를 meta-feature로 쓰면 in-sample stacking leakage가 됩니다. 72-fit 예산 안에서 정직한 surface가 불가능하면 조용히 우회하지 말고 새 설계·예산으로 재등록해야 합니다.

5. **P1 — peer 축이 잘못됨.** 새 모듈은 `(time, layer)`로 묶어 다른 **station** 값을 빼지만, P1의 canonical feature는 `(station, time)` 안에서 다른 **layer**를 peer로 사용합니다. 현 구현은 I/S의 수직 구조 대신 멀리 떨어진 정점 간 차이를 넣습니다.

6. **P1 — feature dependency가 context bank보다 길 수 있음.** robust center/scale과 양방향 CUSUM이 전체 continuous segment를 사용합니다. 24/72/168시간 bank 및 7일 purge와 맞는 유한 dependency가 아니며, 입력을 어느 단위로 자르느냐에 따라 값도 달라집니다.

7. **P1 — segment classifier 수치가 원래 prospective authority에 없음.** 400 trees, learning rate 0.03, 31 leaves 등은 구현 모듈에만 있고 design/amendment에는 Round-B anchor 파라미터만 있습니다. Cycle-1 결과를 보기 전에 successor amendment에서 이 값을 그대로 명시하고 비결과 기반 lineage를 고정해야 합니다.

## 반드시 거쳐야 할 수정

현재 봉인본을 덮어쓰지 말고 successor revision을 새로 만들어야 합니다.

- worker에 전체 72-fit orchestration을 구현하고, synthetic/fake-fit 통합 테스트로 reserve/complete 순서와 21 materialization ceiling을 실제 호출 수준에서 검증합니다.
- segment 학습 proposal의 anchor probability가 모든 행에서 out-of-sample임을 증명합니다. 필요한 cross-fit이 72-fit을 넘으면 새 설계로 재등록하고 기존 예산을 사후 확장하지 않습니다.
- 같은 station·time의 다른 layer peer로 축을 바로잡고 G-ORS fallback을 검증합니다.
- proposal 통계를 24/72/168시간에 bounded하게 만들거나 유한 dependency와 purge를 새로 고정합니다.
- exact 80-feature order, per-prefix encoder, event/day weight, 3-seed mean, detector와 postprocess를 매 window마다 검증합니다.
- pooled inner threshold/cell 선택 뒤에만 outer truth를 사용하고, 3개 outer fold를 정확히 한 번 평가하며 event/day 단위 5,000회 paired bootstrap과 모든 gate를 계산합니다.
- actual authorization은 새 seal·preflight·독립 QA 보고서·PASS verdict·user approval 및 canonical sealed template 전체를 hash로 pin하고, authorization이 입력 map을 자체 교체하지 못하게 합니다.
- Windows 성공 게시 순서는 lock 유지 중 aggregate result/report/manifest flush → `999_completed` journal flush → lock unlink 최종 commit → directory flush → 이후 작업 0으로 테스트합니다. 어느 단계 실패든 lock을 유지하고 terminal failure 하나만 남겨야 합니다.
- 공식 test/sample/submission/candidate/P3 경로 접근 0과 candidate 생성 0을 계속 강제합니다.

현재 v1은 fit을 하나도 소비하지 않았으므로, 위 결함을 고친 successor를 다시 독립 감리한 뒤에만 실행 승인을 검토할 수 있습니다.
