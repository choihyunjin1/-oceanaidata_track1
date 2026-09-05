# 점수 개선 실행 및 개발 환경 정리 — 2026-09-05 v2

## 결론

사전 설계된 P1/P2/P3 분기를 실제 학습·내부 평가까지 실행했지만, 각 문제의 고정 주평가에서
기존 기준을 이긴 새 완성 정책은 없었다. 이번 실행의 신규 공식 입력 접근·답안 CSV·업로드는 0이다.
이는 모든 과거 모델의 무가치나 개선 불가능을 입증한 결과가 아니라, 아래 제한된 비교의 결과다.
P3의 마지막 18개 저장 모델도 별도 CPU 프로세스에서 6,516개 성분 예측을 재현했고 최대차이0m였다.
모든 승인 분기가 terminal이며 이번 연구의 GPU·CPU 프로세스는 종료됐다.

실험과 병행하여 활성 안내 7개를 1,061줄에서 217줄로 줄였다. 원문은 보존했고,
새 프로젝트 스킬·집중 검증 도구의 실제 테스트 22개와 Ruff가 통과했다.
안전하게 미사용이라고 입증한 Python 파일은 없어 실행 코드를 임의 삭제하지 않았다.
커밋·푸시는 이번 요청의 실행 범위에 포함하지 않았으며 수행하지 않았다.

## 같은 평가면에서 실제로 확인한 결과

| 문제 / 정책 | 고정 내부 기준 → 후보 | 판단 |
|---|---|---|
| P1 year-safe depth | pooled F1 0.851174 → 0.848961, Δ −0.002213 | 수심 계약은 수정했지만 점수 개선 아님 |
| P1 depth + 기존 fixed decoder ON | 0.851174 → 0.849872, Δ −0.001303 | decoder로 손실 일부 회복, 기준 미달 |
| P2 절대 ℃ MSE / 균등 행 가중치 / 둘 다 | 가을 intact RMSE 0.465330℃ → 0.516946 / 0.505493 / 0.528892℃ | 첫 seed의 동일 기준 C 유지 |
| P2 물리 프로파일 트리 / 고정 50:50 | 같은 0.465330℃ → 0.710383 / 0.518502℃ | 새 주평가 승자 없음 |
| P3 long-simplex / global-bias | pooled RMSE 0.779105m → 0.779467 / 0.779429m | 4개 소규모 메타 적합 후 기존 no-op 유지 |
| P3 두-seed control / episode-weight | 같은 0.779105m → 0.783338 / 0.794558m | 두 정책 모두 악화; episode-weight는 paired control에도 +0.011219m |

P1은 동일 421,032 historical 행의 TP/FP/FN을 합산한 F1이다. P2는 실행 전에 정한 2024년9–10월 intact
주평가의 같은 첫 seed 비교이며, 기존 3-seed 평균 0.488284℃와 섞어 비교하지 않는다.
P3는 동일 181cases×6leads=1,086행이다. 모두 반복 노출된 historical development이지 새 blind confirmation이 아니다.
월별 악화나 CI만으로 탈락시킨 것이 아니라 **고정 주평가 평균이 개선되지 않았다.**

현재 공식 기준은 P1 **27.771400점**, P2 **27.622418점**, P3 **23.696500점**이다.
이 숫자는 [이미 제출한 정확한 SHA의 receipt](../official_score_repair_submissions_20260905_v1/receipt.json)에만 연결된다.
새 후보 예상 공식 점수는 **미산정**이며 내부 Δ를 그대로 공식 점수로 치환하거나 Public 점수로 계수를 역산하지 않았다.

## 실행량과 재현 경계

| 경로 | 새 학습 / 적합 | runner wall 및 후속 처리 |
|---|---:|---|
| P1-A | historical 12 + final-inner 2 + fulltrain 2 = 16 | 765.219초; 기존 control 모델 12개 재사용; 저장한 outer 6개 fresh-process 예측 일치 |
| P1-B / C | backbone 0 | B 계보/분할 점검, C 기존 고정 decoder 평가; C 41.328초 |
| P2-A / B | historical 9 + 3 = 12 | 216.703초 + 39.781초; 기존 C 모델 9개 exact 재사용; 조건부 추가 seed 12fits 미실행 |
| P3-A | backbone 0, historical meta 4 | 0.241초; 원본 OOF 준비·QA·문서 시간은 별도 |
| P3-B | historical backbone 18, historical router 4 | 2,108.859초; 별도 synthetic smoke 2fits; 18모델 fresh-process replay PASS; 조건부 full backbone 4 + router 1 미실행 |

총 신규 base 학습은 **46fits**, 별도 소규모 메타/router는 **8fits**다. synthetic smoke와 QA 재추론은 이 합계에 섞지 않았다.
병렬 작업의 runner 시간을 합쳐 전체 소요시간이나 절약 시간이라고 주장하지 않는다.
모델 저장·새 프로세스 추론 일치는 재학습의 bitwise 결정성이나 최종 전체 패키지 6시간 인증을 뜻하지 않는다.
기존 최종 패키지·frozen runner/config·attempt lock은 덮어쓰거나 재시작하지 않았다.

## 다음 연구에 남긴 구체적인 단서

1. **P1: 깊이 계약 수정과 과거 강한 모델 복구는 별개다.** 기존 deep OOF와 현재 비교면은 119행의 fold 배정이 다르고
   router purge도 7일 대 21일이다. 적격 source라는 이유만으로 0-fit 결합할 수 없다.
   과거 e150 6fits가 약157분이어서 새 동일계약 학습 비용을 따로 확보해야 한다. 이번에는 GPU 장기 작업을 자동 시작하지 않았다.
2. **P2: 개선이 전혀 없는 것은 아니라 조건에 따라 반대였다.** target-free T5/S5 14일 outage 스트레스 6,031행에서는
   균등 행 가중치가 RMSE 0.465796→0.224904℃로 좋아졌다. 그러나 intact 가을에서는 나빠졌다.
   다음 질문은 결측 상태별 보완을 intact 손실 없이 쓸 수 있는지다. 새 target-free rule과 outer-train inner OOF,
   사전에 정한 여러 outage episode가 필요하며 지금 나온 표에서 rule을 정한 뒤 같은 표를 확증으로 쓰면 안 된다.
3. **P3: bias 제거와 사건 재가중이 RMSE 개선으로 이어지지 않았다.** global-bias는 signed error를 줄였지만 계절별 손익이 상쇄됐고,
   two-seed control 자체도 모든 fold에서 기존 single-seed 완성 정책보다 나빴다. `seed 추가=개선`, `긴 사건의 반복 anchor 가중치 완화=개선`을 전제로 반복하지 않는다.
   현재 증거만으로 새 복잡한 모델이나 추가 혼합비 탐색을 자동 승인하지 않는다.

## 재사용할 단일 근거

- P1: [실행 결과](../p1_depth_contract_repair_20260905_v2/report-source.md), [24-check cycle QA](../p1_depth_contract_repair_20260905_v2/cycle-independent-qa.json), [계보 점검](../p1_depth_contract_repair_20260905_v2/provenance-audit.md).
- P2: [A 실행/전체 해석](../p2_objective_alignment_20260905_v2/report-source.md), [A 34-check QA](../p2_objective_alignment_20260905_v2/independent-recalculation.json), [B 결과](../p2_physical_profile_tree_20260905_v2/result.json), [B 30-check QA](../p2_physical_profile_tree_20260905_v2/independent-recalculation.json).
- P3: [A 결과](../p3_direct_sse_meta_20260905_v2/result.json), [A 13-check QA](../p3_direct_sse_meta_20260905_v2/independent-qa.json), [B 실행/해석](../p3_episode_weight_20260905_v2/report-source.md), [B 35-check QA/replay](../p3_episode_weight_20260905_v2/independent-qa.json).
- 개발 환경: [실제 변경/세션 감사/검증](../agent_workflow_cleanup_20260905_v1/report-source.md), [짧은 개발 루프](../../docs/AGENT_WORKFLOW.md).
- 실행 전 설계/Deep Research: [계획](../../docs/SCORE_IMPROVEMENT_PLAN_20260905_V2.md), [주장–출처 원장](../score_improvement_redesign_20260905_v2/claim-source-ledger.md). 문헌 근거와 위 실측 결과를 구분한다.

문제별 focused pytest/Ruff는 P1 11개, P2 27개, P3 29개가 통과했다. 개발 도구/규정 회귀 22개는 별도 범위다.
중복 검사를 합산해 고유 테스트 수나 전체 저장소 통과라고 표현하지 않는다.
Root는 [통합 QA](independent-qa.json)에서 기존 QA의 해시·검사 결과, fit 합계와 집계 산술을 대조했다.
P2 담당자의 별도 문서 리뷰로 모집단·seed·fit 합계와 `1/√n` 가중치 용어도 확인했다.

완료 뒤 같은 ID를 재실행하지 않는다. 다음 실행자는 이 문서와 담당 문제의 result/QA부터 읽고,
전체 과거 로그·설치·전체 테스트를 반복하지 않는다. [gap-matrix.md](gap-matrix.md)에 미해결 문제와 재진입 조건을 분리했다.
