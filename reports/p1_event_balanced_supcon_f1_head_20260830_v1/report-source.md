# P1 실제 이벤트 균형 SupCon + F1 top-k 스크린

## 결론

**NO_GO_LOW_FIDELITY_SCREEN.** 실제 이벤트 단위 균형 표본, supervised contrastive 보조손실, soft-F1 보조손실, 학습 구간에서만 보정한 station×layer×계절 top-k 제안 헤드를 결합했지만, 동결 incumbent보다 세 역사 창 모두 크게 나빴다. 3-seed 확인 실험은 별도 승인이 필요하기도 하지만, 이번 kill gate가 이미 발동했으므로 실행 가치가 없다.

| 역사 창 | 후보 F1 | incumbent F1 | ΔF1 | 제안 precision | 요구 하한(incumbent F1/2) | 유형 macro ΔF1 |
|---|---:|---:|---:|---:|---:|---:|
| Q2 | 0.692710 | 0.867676 | -0.174966 | 0.153815 | 0.433838 | -0.240401 |
| Q3 | 0.747104 | 0.912188 | -0.165084 | 0.118634 | 0.456094 | -0.337999 |
| Q4 | 0.742868 | 0.898901 | -0.156033 | 0.044517 | 0.449450 | -0.314190 |

pooled 후보 F1은 `0.728424`, incumbent F1은 `0.893298`이고, pooled 유형 macro ΔF1은 `-0.290627`이다. anchor 양성 제거는 0행이고 최대 station 변경 집중도는 `0.584442`로 안전성·분산도 게이트는 통과했지만, 성능 관련 세 게이트는 모두 실패했다.

## 무엇을 고정하고 무엇을 시험했나

- 기존 Sobol 탐색에서 선택된 `trial_18`, 165개 past-only 입력, width 512 MS-TCN 계보, 강한 weight decay `0.001`, anchor-union decoder는 그대로 유지했다.
- Q2/Q3/Q4 각각 seed `20260830`, 25 epoch, 정확히 1회씩 총 3 historical fits만 수행했다.
- 합성 이상을 만들지 않았다. 연속된 실제 양성 run 하나를 이벤트 하나로 삼아 이벤트마다 centered window 1개와 station×layer×기상 계절이 일치하는 정상 window 1개를 배정했다.
- 생성기 hidden representation에 SupCon weight `0.1`, 최종 row logit에 soft-F1 weight `0.25`를 추가했다.
- 전역 threshold를 탐색하지 않았다. 각 phase의 학습 prefix에서만 station×layer×계절별 최적 top-k 비율을 정하고 최대 2%로 제한한 뒤 blind holdout에 적용했다.
- Q2/Q3/Q4 예측을 모두 봉인하고 바이트 검증한 후에만 세 holdout 정답을 열었다. 창 결과에 따른 재시작·재튜닝은 없었다.

## 사전 지원성

0-fit preflight에서 실제 이벤트 수는 Q2/Q3/Q4 학습 prefix별 `116/153/213`, 지원 유형은 모두 5개, station×layer×계절 셀은 `30/38/48`개였다. 최대 이벤트 셀 집중도는 각각 `0.0776/0.0654/0.0563`으로 0.8 제한을 넉넉히 통과했다. 정상 matching은 Q2 전부, Q3 전부, Q4 213개 중 212개가 정확한 station×layer×계절 셀에서 이뤄졌다. 따라서 이번 실패는 이전처럼 “양성 이벤트 지원이 없음”으로 설명되지 않는다.

## 게이트 판정

| 게이트 | 판정 | 근거 |
|---|---|---|
| 모든 창 ΔF1 > 0 | FAIL | 세 창 모두 -0.156 이하 |
| pooled 유형 macro-F1 개선 | FAIL | Δ -0.290627 |
| 각 창 제안 precision > incumbent F1/2 | FAIL | 0.0445~0.1538 대 0.4338~0.4561 |
| anchor 제거 0행 | PASS | 세 창 합계 0행 |
| 변경행 최대 station share ≤ 0.8 | PASS | pooled 0.584442 |

## 해석과 닫히는 가설

학습 구간 F1을 최대화한 top-k 비율이 대부분 상한 2%에 붙었지만 역사 holdout에서는 제안 precision이 급락했다. 표현학습 자체가 25 epoch 저충실도라는 한계는 있으나, 세 창에서 같은 방향으로 `0.156~0.175` F1의 큰 손실이 났고 유형별 macro 성능도 전부 악화했다. 따라서 **이 exact recipe를 epoch·seed만 늘려 확인하는 경로는 닫는다.**

가족 전체를 닫는 결과는 아니다. 다음 P1 연구는 이벤트 균형 자체보다 (1) 학습 top-k rate의 분포 이동을 견디는 외부 품질/위험 제약, (2) incumbent의 false-negative 영역만 학습하는 residual target, (3) proposal 수가 아니라 예상 utility를 직접 보정하는 event-level head를 별도 사전등록해야 한다. 현재 결과를 보고 2% cap이나 loss weight를 바꿔 재실행하는 것은 금지한다.

## 재현·QA

```powershell
.\.venv-p1\Scripts\python.exe -m pytest tests/test_p1_event_balanced_supcon_f1.py -q
.\.venv-p1\Scripts\python.exe -m ruff check src/p1_qc/event_balanced_supcon_f1.py scripts/run_p1_event_balanced_supcon_f1_head_20260830_v1.py scripts/qa_p1_event_balanced_supcon_f1_head_20260830_v1.py tests/test_p1_event_balanced_supcon_f1.py
.\.venv-p1\Scripts\python.exe scripts/qa_p1_event_balanced_supcon_f1_head_20260830_v1.py
```

독립 QA는 세 receipt의 seed/epoch/history hash, 봉인 NPZ hash·inventory, 후보·control·유형 macro·proposal precision·station 집중도·gate vector를 재계산해 PASS했다. 공식 test/sample/submission 접근, 제출 CSV 생성, 업로드는 모두 0이다. checkpoint도 저장하지 않았다.

