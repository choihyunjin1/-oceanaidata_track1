# Canonical report source — deadline parallel Deep Research v17

## 범위와 의사결정

- 시각: 2026-08-28 23:09 KST부터 시작.
- 사용자 목표: 남은 일일 제출 기회를 무작정 소모하지 않으면서, 오늘 실행 가능한 세 문제의 비중복 구조를 병렬 검증해 제출 후보 또는 다음 날의 발판을 확보.
- 공식 read-only snapshot: 분당독고다이 5위, P1/P2/P3=`28.901363/27.922187/24.066168`, total `80.889718`; 잔여 제출 P1/P2/P3=`3/1/2`.
- 제외: 공식 test/sample/submission CSV 값 접근·생성·업로드, 결과 기반 retry, 기존 terminal experiment의 threshold 완화.

## 직접 답변

오늘 제출할 새 후보는 없다. P1과 P3은 구조 가정이 source/preflight에서 반증됐고, P2는 계절 회귀 차단이라는 유의미한 성공을 얻었지만 기존 α50를 교체할 필수 benefit gate에 미달했다.

## P1 evidence

- candidate: `p1_station_pooled_hierarchical_residual_subset_scan_anchor_union_20260828_v1`.
- statistic: `temp_long_resid_7d` cell median/log-MAD를 station 정상분포로 fixed empirical-Bayes shrinkage.
- block calibration: 7-day cell-block, alpha=.01, 15-day purge, 50/50 time fit/cal.
- support: S Q2/Q3/Q4 normal blocks 107/126/161, fitted layers 7.
- failure: top-10% tail maximum layer share 0.454545/0.461538/0.529412 > 0.4; I fit support 0.
- action: active fold 0, exact e150 no-op, deletion 0, outer truth 0.

## P2 evidence

- candidate: `p2_alpha50_supervised_rank1_threeway_crossfit_regime_veto_20260828_v2`.
- three chronological groups rotate H/M/R roles for held score, correction fit, reference support.
- candidate modifies only bins whose inner OOF benefit CI90 upper<0; others α50 exact no-op.
- aggregate ΔRMSE -0.002453834 C; CI90 [-0.004330819,-0.001569947].
- fold Δ: Sep-Oct -0.002596358, Jul-Aug -0.006200575, Nov-Dec 0.
- layer Δ: L2 -0.007739413, L3 -0.004941947, L4 -0.001440175 C.
- gate fail: pooled target -0.005 and Sep-Oct target -0.003.
- recovery: post-commit pandas formatting error only; sealed prediction hash verified, model/prediction rerun false, finalize-only QA PASS.
- action: HOLD, official 0.

## P3 evidence

- candidate: `p3_era5_joint_wave_state_multitask_transfer_20260828_v1`.
- source variables: frozen 286 features; target state Hs/tp/hmax using CatBoost MultiRMSE.
- comparator: same rows/seed/features with single-Hs objective; 0.20 increment at 18/24h only.
- source ΔRMSE -0.000007093 m; CI90 [-0.000421627,+0.000398041].
- year Δ: 2021 -0.000189926, 2022 +0.000425340, 2023 -0.000193823 m.
- station Δ: G -0.000338933, I +0.000211183, S +0.000103962 m.
- lead Δ: 18h -0.000076624, 24h +0.000041515 m.
- action: source gate fail, shadow truth 0, family closed.

## 종합 해석

모델 크기나 epoch 부족이 실패 원인이 아니다. P1은 pooling group heterogeneity, P2는 selectivity와 benefit의 trade-off, P3는 auxiliary task conflict가 각각 병목이다. 이 세 원인에 더 큰 학습량을 투입해도 직접 해결되지 않는다.

오늘의 확실한 발판은 P2이다. regression veto가 작동했음을 새 구조에서 확인했으므로 다음 candidate는 veto 여부를 다시 묻지 않고, exact no-op과 full correction 사이를 training-only confidence로 단조롭게 연결해야 한다. 공식 α50 historical OOF 부재와 exposed fold 적응 위험은 계속 명시한다.

## 출처와 중단

관련 1차 문헌과 공식 문서는 claim ledger에 기록했다. 세 후보 모두 terminal 판정과 independent QA에 도달했고 추가 탐색은 결과 기반 완화가 되므로 이 사이클을 중단한다.
