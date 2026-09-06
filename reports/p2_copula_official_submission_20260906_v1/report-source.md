# P2 CPU 대조 제출 — 2026-09-06

결론: copula는 같은 CPU 학습본보다 공식 RMSE를 0.013906℃ 줄였지만, 보존된 CUDA C3 기준본(0.455143℃)은 넘지 못했다. 기존 기준 패키지를 유지한다. 공식 결과로 강도·계수·행별 예측을 바꾸지 않는다.

| 파일 | SHA 앞 8자리 | Public RMSE ℃ | 공식 점수 |
|---|---|---:|---:|
| P2_C3_control.csv | ae2df148 | 0.489080 | 27.196585 |
| P2_insample_full.csv | e0b4005b | 0.475174 | 27.371078 |
| 보존 CUDA C3 | 46d194a1 | 0.455143 | 27.622418 |

같은 CPU 대조 대비 +0.174493점, 보존 CUDA 기준 대비 −0.251340점이다. 이는 실제 Public 채점이며 내부 검증 또는 기대점수와 혼합하지 않는다. Private 결과는 알 수 없다.

두 파일 모두 26,061행, 새 빈 모델 학습4fit/167.125초, 학습 QA17·별도 PID 전체 CSV replay·ZIP 추출 replay·독립73QA·root23개 지문/스키마/키/순서/유한성 대조를 통과했다. [학습/패키지 기록](../p2_crossfit_copula_materialization_20260906_v3/report-source.md), [root 사전 QA](../remaining_work_completion_20260906_v1/p2-pre-upload-root-qa.json).

07:28 KST(홈페이지 분 단위) 각1회 제출·채점완료. 오늘 잔여3→1. [정확한 경로·SHA·접수 근거](receipt.json). 최종 모델 잠금·hidden truth·추가 학습·Git commit/push 없음. 이 제출은 CPU 대조 개선을 확인한 것이며 CUDA와 CPU 학습을 동등하다고 증명한 것이 아니다.
