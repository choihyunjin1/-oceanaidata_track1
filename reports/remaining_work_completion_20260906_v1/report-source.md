# 남은 순서 실행 결과 — 2026-09-06

**이번에 확정한 남은 순서(비교→학습→내부 QA→패키지 재생→공식 채점)를 P1/P2/P3 모두 완료했다. P1은 개선했고, P2·P3 새 후보는 보존 기준을 넘지 못했다.** 현재 보존 선택은 P1 bracket / P2 기존 CUDA C3 / P3 기존 재생성 기준이다. 모든 옛 제안이나 로드맵 전체를 실행했다고 주장하지 않는다. 기존 기준 파일·실패·봉인 상태는 보존했다.

## 공식 확인과 현재 선택

| 문제 | 이번 공식 확인 | 현재 판단 |
|---|---|---|
| P1 | bracket F1 0.785944 / 27.644124점; 같은 재생성 기준 대비 +0.008195 / +0.217805점 | 9031 답안과 이를 만드는 새 portable 패키지 보존 |
| P2 | 같은 CPU C3 0.489080→copula 0.475174℃ / +0.174493점 | 기존 CUDA C3 0.455143℃보다 나빠 기존 기준 유지 |
| P3 | hmax 제거 새 cold 답안 0.608184m / 23.680619점 | 기존 0.607183m / 23.696500점보다 +0.001001m 악화; 기존 기준 유지 |

공식 성적은 파일 SHA에 귀속된다. P1 옛 0.790733/0.833548, P2 계수 보정 옛 최고 기록 등을 현재 파일로 승계하지 않는다. Private 성적 또는 제출 점수를 내부 지표에서 추정하지 않는다.

## 순서별 실행

1. P1 bracket9031: 사전26QA 재확인 후 06:39 단1회 제출·채점 완료. [receipt](../p1_bracket_official_submission_20260906_v1/receipt.json).
2. P1 portable: 저장 모델 ZIP 실제 추출·새PID 추론29.234초 exact. 별도 빈 모델 폴더 CPU2 cold4fit376.750초, 독립47QA·모델/답안 replay 포함753.030초에 동일9031. [canonical](../p1_bracket_portable_20260906_v2/report-source.md), [root 지문9대조](p1-package-root-check.json).
3. P1 범위/셀정책: 새backbone0fit, train-normal min/max와 inner OOF cell O/B/AND/OR를 각각 평가했다. 주평가 ΔF1 0 / −0.005102573으로 비승격, 새CSV없음. 전체행 pooled의 작은 개선과 주평가를 혼동하지 않는다. 고정0–35℃ 룰 자체와 동일 실험은 아니며 미래 FP0도 보장하지 않는다. 독립190QA/새PID6replay. [canonical](../p1_trainfit_postpolicy_20260906_v1/report-source.md).
4. P2 crossfit 수치정정: 기존40fit 검증 재사용 +48fit 완료(논리전체88fit). 독립1117QA, 새PID 전체replay32, [root산술73](p2-root-arithmetic-qa.json). in-sample만 주평가 개선, crossfit은 크게 악화. [canonical](../p2_crossfit_copula_forward_numeric_20260906_v2/report-source.md).
5. P2 full/portable: 빈 모델4fit167.125초→학습QA17→26,061키 두답안→새PID/ZIP추출replay→독립73QA 및 [root23대조](p2-pre-upload-root-qa.json). 같은 CPU 대조/후보를 07:28 각각1회 공식 채점했다. [학습/패키지](../p2_crossfit_copula_materialization_20260906_v3/report-source.md), [공식receipt](../p2_copula_official_submission_20260906_v1/receipt.json).
6. P3 numeric lead: 15backbone+8router/1955.61초, 독립149QA·103,602행 exact replay. ΔRMSE −0.000500418m. [root산술36](p3-numeric-root-arithmetic-qa.json), [canonical](../p3_numeric_lead_forward_gpu_20260906_v2/report-source.md).
7. P3 hmax 제거: 별도10backbone+4router/903.260초, 독립134QA·새PID 전체replay exact. ΔRMSE −0.001571311m, CI90 [−0.003978547, +0.000703372]. 평균 개선이 numeric보다 크지만 불확실성은 남는다. S-ORS +0.002208646m, wind-missing +0.003634297m 악화는 별도 위험으로 보존한다. [root산술36](p3-hmax-root-arithmetic-qa.json), [canonical](../p3_hmax_removed_forward_20260906_v1/report-source.md).
8. P3 whole-cold: **07:39:13 KST source prepare부터 전체1,611.919초에 완료**. 배포 원자료→591공통 특징→527 hmax제외 특징→5fold10backbone+4router→full2+1→새PID QA→공식CSV→또 새PID exact replay. CPU2/GPU device0단독, 새 전체 학습1회다. 별도 warm full2+1은 중복이라 생략했다. source seal 전 독립 리뷰가 historical single replay의 lead-major/anchor-major 정렬 오류를 발견했고, shuffled 3×6 사례를 포함한16개 합성검사 PASS와 재리뷰 후 시작했다. 오류가 있는 상태로 실제 모델을 학습하지 않았다. [cold 증거](../p3_forward_candidate_cold_20260906_v1/report-source.md).
9. P3 저장 모델 ZIP: full 모델3개만 포함한 ZIP을 원 저장소 밖에 실제 추출하고 새PID27384/CPU2/추가fit0/6.088초에 1,200행 SHA d456 exact를 재현했다. 코드합성35PASS/Ruff·독립정적리뷰 뒤 실행했다. [saved 증거](../p3_forward_saved_20260906_v1/report-source.md).
10. P3 공식: [root22 QA](p3-pre-upload-root-qa.json)와 패키지 재생 후 08:08 단1회 제출·채점 완료. Public 0.608184m, 기존 대비 −0.015881점. 내부의 작은 개선이 Public에서 재현되지 않았다. [영수증](../p3_hmax_official_submission_20260906_v1/receipt.json). 남은 당일기회 P1 1/P2 1/P3 2이며 마지막 실측 기준으로 시간이 지나면 다시 확인한다.

## 검증 범위와 한계

- 평균 개선과 불확실성/최악 구간을 분리했다. 경험적 bootstrap 개선 비율을 공식 점수 개선 확률로 쓰지 않는다. 과거 노출된 검증 기간은 새 virgin holdout이 아니다.
- 동일 환경 saved-model replay, 새 폴더 전체 학습, 새 OS/offline 검증은 다르다. P1/P2 ZIP 추출 replay가 두 번째 전체 학습이라는 뜻은 아니다. 현 패키지에 별도 선언이 없는 clean OS/offline dependency 설치는 미검증이다.
- P2 자연 T5 결측 부분의 악화와 일부 후기 블록 악화를 보존한다. 평가행0인 outage 블록은 NOT_ESTIMABLE이며 성능0으로 해석하지 않는다.
- 외부 관측/hidden truth/리더보드 역산0. 정상 급변 행 삭제·가중축소0. 공개 입력은 해당 후보 학습QA 뒤에만 사용했다.
- 최종 모델 잠금은 하지 않았다. 이번 작업은 로컬 연구·재현·승인된 CSV 채점이며 Git commit/push는 별도 요청 없으므로 하지 않았다.
