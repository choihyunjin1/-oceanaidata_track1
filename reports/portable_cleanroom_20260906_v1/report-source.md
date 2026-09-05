# 재생성 기준 패키지와 P1 최적화 진단 — 진행 통합 기록

후속 업데이트: 이 문서 아래의 기준 패키지 완료 시점 이후, **P1 bracket-only 4-fit 후보의 로컬 답안 및 독립 QA도 완료**했다. 신규 실행은 [별도 보고서](../p1_bracket_candidate_20260906_v1/report-source.md), 파일 선택은 [후보 안내](../../docs/ocean_v2_codex/P1_BRACKET_CANDIDATE_HANDOFF_20260906.md)를 따른다. 아래 기준 모델·과거 진단 기록과 혼합하거나 새 공식 점수로 해석하지 않는다.

## 현재 결론 (2026-09-06 04:17 KST)

**P1/P2/P3 모두 과거 모델·답안 복사 없이 새 학습으로 채점된 기준 SHA를 재현했다. 사용자 일시중지 뒤 P1 깊이 진단도 67/67 QA로 완료했다.** P3 v2 run_a는 backbone 7개 성공 후 8번째 학습 도중 중단됐으며 정상 terminal/failure receipt가 없어 정확한 종료 원인은 미확인이다. pristine run_b는 전체 실행 및 42/42 QA, 별도 recovery는 44/44 QA를 마쳤으며 두 결과의 예측/답안이 exact 일치했다. 최신 저장 모델 ZIP도 실제 추출·추론 검증을 통과했다. 이 작업은 성능 개선/공식 최종 제출 완료와 구분한다. 기존 원본·봉인 실행·최종 패키지를 보존했고 commit/push/upload/final lock은 0이다.

| 문제 | 현재 패키지 검증 | 학습→추론 실측 | 공식 답안 SHA | 증거 |
|---|---|---:|---|---|
| P1 | 2회 독립 full rebuild, 총8fits, 4개 모델/답안 exact, 각각 새 PID replay exact | 194.765 / 215.079초 | `5971e145…128a` | [보고서](P1/report-source.md), [20-check QA](P1/independent-qa-v2.json) |
| P2 | 1회3fits, 새 PID inference/replay exact | 80.203초 | `46d194a1…c071` | [보고서](P2/report-source.md), [71-check QA](P2/independent-qa.json) |
| P3 | pristine run_b 빈 폴더부터 8 backbone+3 router, 42/42 QA PASS; recovery 44/44 PASS. 두 결과의 예측/CSV exact | 계산 1,203.415초; 단계 간 대기 포함 1,256.163초 | `6bfa23d2…96b7` | [비교 QA](P3/repetition-comparison.json), [보존 QA](../../artifacts/portable_cleanroom_20260906_v1/P3/v2_run_b_completed/04_logs/receipts/independent-qa.json) |

같은 PC의 기존 dependency 환경을 사용했다. 원 저장소 밖 실행/로컬 코드 origin/guard를 검사했지만, 새 PC·새 venv 설치·OS 방화벽 차단 시험까지 완료했다고 하지 않는다. 원본 배포 데이터는 ZIP에서 제외했다. 실제 파일 선택과 cold start 방법은 [사용 안내](../../docs/ocean_v2_codex/PORTABLE_PACKAGE_HANDOFF_20260906.md)에 있다.

## 점수 향상 연구를 패키지와 분리

- P1 `p1_tuning_twosided_20260906_v1`: 700회 기준/구간 모양 특징과 inner1400 학습곡선 진단. **새15fit+정확한기존3재사용, 1,443.422초, 별도 PID [63/63 QA PASS](../p1_tuning_twosided_20260906_v1/independent-qa.json)**. 기존 공식 답안이나 기준 portable의 가중치를 대체하지 않는다.
- P1 주평가 H1 2025의 F1 **0.737031619 → 0.757196860 (Δ+0.020165240)**, paired-day CI90 **[+0.005014770,+0.037606590]**, bootstrap 양의 개선 비율 **0.993**. 전체 pooled F1 **0.795140828 → 0.809641411**. O는 동일80열/모델이고 B만80→107열, 같은 inner 선택 알고리즘을 다시 수행한 전체 절차 비교다. 이는 반복 사용한 역사적 양측 평가이며 미래/공식 성공확률이 아니다.
- 위험: H2 2024 fold Δ **−0.021024011**, 그 구간 S-ORS L1 F1 Δ **−0.084545228**(TP579 동일, FP261→425). 안정성 위험은 숨기거나 자동 탈락 기준으로 바꾸지 않고 별도 보고한다. [문제별 원문](../p1_tuning_twosided_20260906_v1/report-source.md)의 day/층별 수치를 함께 본다.
- 학습곡선은 일괄 underfit을 지지하지 않는다. fixed-decoder pooled inner O는700→1400 F1 +0.000849, B는−0.006735이며 모든 arm logloss는악화했다. bracket은1000회 진단이700회보다+0.002374였으나 **outer에 이 선택을 적용하지 않았다**. 큰 epoch/트리 수를 자동 승격하지 않는다.
- P1 train 통계의 station/year/layer 키는 미등록 연도에서 nominal_depth_m NaN / depth_regime unknown 문자열을 만든다. 해당 범주가 encoder 학습에서 없었을 때 −1로 인코딩된다. raw depth는 남는다. [새 ID의 0-fit 스트레스](../p1_tuning_depth_stress_20260906_v2/report-source.md)는 중단 부분 18확률을 재사용하고 9확률을 추가 추론했다. **별도 PID의 전체 27확률 재예측 및 67/67 QA PASS**, 이어가기+QA 551.172초다. 기존 모델/threshold와 두 열 이외 입력은 유지했다.
- Unknown-year 조건에서도 주평가 F1 **0.739425357→0.752901598 (Δ+0.013476241, CI90 [+0.002593516,+0.025738337])**. 동일 208,093행/양성9,911행이며 TP+10, FP−308로 root 산술 검산도 일치한다. 다만 표준 대비 후보 자체 F1은−0.004295262이며 최악 정점·층의 후보 상대 Δ는−0.096710815다. 상대 개선을 수심 문제 해결이나 공식 점수 향상 확정으로 확대하지 않는다. bootstrap 양의 비율 .9795는 공식 성공확률이 아니다. fallback 선택, 추가 HPO, 새 후보 full 학습/공식 materialize는 하지 않았다.
- P2 기존 crossfit copula의 roundoff assertion 실패는 성능 NO_GO가 아니다. [정정 설계](P2/copula-numeric-amendment-design.md)는 보존 모델 재사용 검증과 남은48fits를 구분하며 이번에 재실행하지 않았다.
- P3 hmax 제거는 안전한 개선으로 단정하지 않는다. 우선 현재 코어의 재현 패키지와 시간/계보를 확보한다. 과거 CPU2 90분 계획의 349분 예상과 resource stop을 반복하지 않는다.

## P3 중단 복구와 전체 재현의 구분

Pristine run_b는 빈 모델부터 중단 없는 전체 재생성 1회다. 별도 recovery는 같은 사이클 run_a의 성공 backbone 7개와 OOF를 검증해 재사용하고 full multi1+router1만 추가했다. 중단 전 full single의 save-time digest는 없어 acceptance-time hash임을 명시했고, run_b와 같은 입력에서 full single 예측/모델 파라미터를 별도 대조했다. 두 경로의 1,086행 historical OOF·full-model probe 및 1,200행 답안은 exact, 차이 RMSE/최대차/불일치 행수 모두0이다. CBM 파일 SHA는 서로 다르므로 모델 bytes까지 같다고 하지 않는다. 전체 성공 backbone16/router6 및 중단 backbone 시도1을 기록했다.

Recovery 추론 전 선언된 빈 `05_answer` 폴더 부재로 발생한 FileNotFoundError는 [운영 정정 기록](P3/inference-directory-repair.json)에 남겼다. 기존 failure receipt/모델/코드는 보존했고, 공식 입력 접근 및 inference lock 생성 전 실패였음을 확인한 뒤 빈 폴더만 생성해 학습 추가0으로 완료했다. Recovery를 두 번째 중단 없는 cold-start라고 세지 않는다.

## 완료와 다음 단계

P1 진단과 P3 재현/복구/포장 검증을 완료했다. P3 최신 ZIP은 `archives_v3`의 cold-start 코드와 저장 모델 재생용으로 역할을 구분했다. 원 runner/manifest를 바꾸지 않은 별도 추론 어댑터를 새 ZIP 추출 폴더에서 실행해 3.668초/fit0으로 답안 SHA exact를 확인했다. 현재 선택 파일/실행법은 [사용 안내](../../docs/ocean_v2_codex/PORTABLE_PACKAGE_HANDOFF_20260906.md)를 따른다.

후속 P1 bracket-only 후보는 기존 final-inner 2 fits+full 2 fits의 별도 계약으로 준비할 수 있으나 이번 재개에서는 아직 실행하지 않았다. 다른 답안 SHA에는 과거 점수를 승계하지 않는다. 외부 자료·hidden truth·Public 역산·사후 행 패치는 금지한다.
