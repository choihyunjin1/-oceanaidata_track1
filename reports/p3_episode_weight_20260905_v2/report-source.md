# P3-B 결론: 사건 가중치와 2-seed control 모두 기존 기준선보다 악화

**기존 clean 1-seed 정책을 유지하며 새 전체학습·공식 제출 파일은 만들지 않는다.** 정해진 18회 historical CatBoost 학습과 4회 과거-only router 재적합을 완료했고 기술적 실패가 아니라 이번 고정 정책의 내부 평균 개선 실패다. 불확실성이 커서 탈락시킨 것이 아니라 사전고정 pooled RMSE 자체가 악화됐다.

| 완성 정책 | 동일181cases 내부 pooled RMSE m | 기존 clean 대비 m | case bootstrap delta 95% 구간 m |
|---|---:|---:|---:|
| 기존 clean 1-seed / A no-op | 0.7791048399763751 | 0 | 기준선 |
| two-seed control | 0.7833381709227665 | +0.0042333309463914 | [+0.0020537989, +0.0063693483] |
| two-seed episode weight | 0.7945575383912831 | +0.0154526984149080 | [+0.0058601185, +0.0250506228] |

사건 가중치의 **seed-matched control 대비 차이**는 +0.0112193674685166m, descriptive 95% 구간은 [+0.0022778559, +0.0201241831]이다. Two-seed control 자체도 후보로 빠짐없이 비교했지만 기존 정책보다 낮았다. 따라서 weighted가 control에 졌다는 이유만으로 전체 판단을 생략한 것이 아니다.

두 새 정책은 기존 기준선보다 세 fold와 여섯 lead 모두 RMSE가 악화됐다. 사건 가중치 후보의 변화는 3h +0.0276893m, 9h +0.0255556m로 짧은 lead에서도 컸다. 정해진 데이터 가중 변화가 실제 사례 대표성을 더 잘 맞췄다는 가설은 이번 표면에서 지지되지 않는다. 이것을 모든 사건 가중치·모든 seed ensemble의 일반적 실패로 확대하지 않는다. 개별 seed를 사후 골라내거나 가중 지수·clip·router·shrink를 재탐색하지 않는다.

## 무엇을 실제 실행했는가

- 적격 배포 train 유래 24,360 anchors, 591 features를 사용했다. 기존 181cases × 6leads = 1,086행의 평가 key·target을 그대로 대조했다.
- control 첫 seed세트20260816/17/18의 성분 OOF는 source-only 재생성 캐시의 해시를 확인해 재사용했다. 추가 control세트20260916/17/18 = 6 fits, 사건 가중치 두 세트 = 12 fits, 총18 backbone fits다.
- 사건 가중은 `threshold_weight / sqrt(해당 outer-train 내 station×episode anchor 수)`를 평균1로 정규화했다. 정점별 배포 wave의 ≥1.5m 연속20분 사건을 사용했고 background/관측 간격 단절을 구분했다. Train anchor 수는 fold별7,912 / 11,754 / 20,899, 사건 수는420 / 673 / 1,012였다. Target 지원 누락으로 같은 물리적 사건을 쪼개지 않았다.
- 각 arm의 two-seed single/multi 성분을 먼저 평균하고 그 arm의 OOF로 고정 router를 이전49/128 cases에서 다시 적합했다. 옛 saved router 계수0, Public 역산계수0, A simplex/bias 결합0이다. 3/6/9h router no-op, 12/18/24h persistence shrink0.2는 보존했다.
- Runtime 2,108.859초(35.15분). 처음8 model pairs는167.689–234.490초, 마지막 pair는557.770초였다. 마지막 구간에 높은 GPU 사용률/VRAM점유와 사용자 앱이 관찰되어 자원 경합 가능성을 기록했으나 인과를 단정하지 않고 어떤 사용자 앱도 변경하지 않았다. 성공한 run/lock/config/seed/schedule은 수정하거나 재시작하지 않았다.
- 연구 학습 프로세스는 종료됐고 GPU를 해제했다. Root가 조건부로 승인한 full4fits+fullrouter1 경로는 준비·synthetic 검증했지만 승자가 기존 no-op이므로 **실행0**이다.

## 독립 검증과 한계

[independent-qa.json](independent-qa.json): 35/35 PASS. 별도 알고리즘의 사건 grouping/counting/weight SHA, 원본·캐시·코드·모델 해시, 같은 key/target, 실제 target 관측완료 시각, 78h/episode 격리, seed 평균 순서, router/shrink 산술과 주평가를 재계산했다. 과거 target가 다음 fold 시작보다 각각10.667h/106.667h 먼저 준비됐고 정점별 anchor 간격은112h/180.667h, episode 중첩0이었다.

별도 CPU 프로세스에서 신규18개 모델을 재로딩해3,258 historical lead행의6,516 성분 예측을 재생했으며 최대차이0m였다. 이는 재생 동등성 검증이지 새 학습이나 새 독립 holdout이 아니다. Focused pytest는 A7+B8+기존split/router9+준비된full정책5=29 PASS, Ruff PASS다.

이 표면은 이미 반복 사용된 historical development다. Case bootstrap은 descriptive이며 선택편향을 제거하지 않는다. 과거 persistence gate의 결과/`consequence`는 기존 helper의 진단 기록으로만 보존한다. 이번 권한이나 승격 판단을 그 옛 문자열이 새로 승인하는 것은 아니다. 실제 hard integrity는 모두 통과했으나 사전고정 평균 개선이 없었다.

새 공식 예상 점수는 **미산정**이다. 내부 RMSE를 Public 모집단으로 그대로 옮기거나 공식 점수로 계수를 역산하지 않는다. 공식test/sample/hidden/CSV/upload0, 외부자료0, source/old artifacts 변경0, Git0이다. Access0은 audit-hook allowlist와 호출경로 검토의 범위이며 OS 전체 감시로 과장하지 않는다.

## 재현 자산

- [사전등록](preregistration.md), [config](../../configs/experiments/p3_episode_weight_20260905_v2.json), [runner](../../scripts/run_p3_episode_weight_20260905_v2.py), [별도 QA/replay](../../scripts/qa_p3_episode_weight_20260905_v2.py), [tests](../../tests/test_p3_episode_weight_20260905_v2.py).
- [A 결과](../p3_direct_sse_meta_20260905_v2/report-source.md), [최상위 규정](../../00_ORGANIZER_DATA_POLICY.md), [고정 실행계획](../../docs/SCORE_IMPROVEMENT_PLAN_20260905_V2.md).
- 결과 SHA-256 `c00d3edc4e6832dfb8fe8c129f5ad38956afc0a330782a9f00f6eacf868784c1`.
- 현재 attempt lock을 삭제하거나 기존 run을 재개하지 않는다. 원본 관측·정답·행별예측·체크포인트는 보고서/버전관리 대상이 아니며 ignored local artifacts에만 있다.

Data Analytics 검증 지침은 동일 평가 grain, train-only 계보, 재생과 일반화의 구분, 위험·실패 기록에 적용했다.
