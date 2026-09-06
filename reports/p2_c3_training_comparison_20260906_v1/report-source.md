# P2 C3 학습량·정규화 비교 결과

결론: **120 epoch(L120)는 가을 3-seed 평균 RMSE를 0.004779℃ 개선하여 후보로 유지한다. 다만 결측 위험은 뚜렷하게 악화되어 기존 C3를 대체 확정하지 않는다.** 28 historical fits와 저장 모델 전체 재현·독립 QA를 완료했다. 첫 seed만으로는 control보다 나빴지만 추가 seed를 포함하면 평균 개선이 남았다. 새 독립 기간 성능이나 공식 점수 상승을 입증한 것은 아니다.

## 비교 결과와 정확한 분모

단위는 ℃, Δ는 candidate−C60(음수가 개선)이다. 모든 비교는 같은 키/같은 CUDA GPU0/CPU2, 같은 특징·augmentation·domain weights·gradient penalty·무projection 표면이다.

| 평가 | 행 수 | C60 | L120 | Δ |
|---|---:|---:|---:|---:|
| B3 첫 seed | 26,273 | 0.465330203 | 0.481479504 | +0.016149301 |
| B3 추가 2-seed 평균 | 26,273 | 0.533473508 | 0.525777767 | −0.007695741 |
| B3 전체 3-seed 평균 — 주판단 | 26,273 | 0.488284326 | 0.483505057 | **−0.004779270** |
| 전체 8fold 첫 seed pooled | 166,268 | 1.270852306 | 1.281373210 | +0.010520904 |
| B3 3seed, outage 적용 전체 구간 | 26,273 | 0.538760112 | 0.563601402 | +0.024841290 |
| B3 3seed, 실제 추가17일 outage행만 | 7,322 | 0.445059242 | 0.561965765 | **+0.116906524** |
| B3 3seed, 자연 T5 결측행 | 180 | 0.535881156 | 0.765854952 | **+0.229973796** |

가을 주평가 SSE는 C60 6264.050259102666, L120 6142.026795332680이며 각각 26273으로 나누어 제곱근을 취한다. fold RMSE 평균이 아니다. first-seed L120이 이긴 것은 다른 challenger D60에 대해서다. C60보다 첫 seed가 좋았다고 표현하지 않는다.

| 첫 seed, natural | n | C60 | L120 | D60 |
|---|---:|---:|---:|---:|
| B1 | 22,940 | 2.004072014 | 2.063022734 | 2.040893101 |
| B2 | 26,018 | 1.371700519 | 1.380717017 | 1.370609569 |
| B3 | 26,273 | 0.465330203 | 0.481479504 | 0.493296499 |
| B4 | 18,093 | 0.049580551 | 0.050951335 | 0.048180743 |
| B5 | 16,417 | 0.804912777 | 0.968196050 | 0.720559808 |
| B6 | 26,308 | 1.891105276 | 1.842633219 | 1.903779622 |
| B7 | 13,335 | 1.095310409 | 1.036624184 | 1.057655410 |
| B8 | 16,884 | 0.242470408 | 0.185967453 | 0.205258197 |
| pooled | 166,268 | 1.270852306 | 1.281373210 | 1.275178081 |

D60은 고정 first-seed B3 선택에서 L120보다 나빠 추가 seed를 실행하지 않았다. D60의 3-seed 결론은 미확인이지 실패가 확증된 것이 아니다.

## 불확실성·위험

KST 2024-01-01을 원점으로 하는 공통 7-day block, paired 2000회/seed20260906 bootstrap이다. B3 3seed는 10 blocks, Δ CI90 [−0.020369156, +0.010871728]℃, 경험적 개선 비율0.667이다. 6/10 blocks가 악화했고 최악 block Δ+0.029781736℃다. 이것은 공식 개선 확률이 아니다. 추가2seed CI90도 [−0.024281640, +0.015602082]로0을 포함한다.

가을 추가17일 outage행 7322개는 독립 3 blocks뿐이며, 3seed Δ CI90 [+0.070059418, +0.153358551]℃로 악화 위험이 크다. 자연 T5 결측180행의 Δ CI90 [+0.100545691, +0.341185545]℃도 악화다. 전체8fold firstseed outage interval42292행은 1.800540739→1.911620277℃(Δ+0.111079538)로 악화했다. 해당 위험을 숨기거나 사후 조건부 routing으로 고치지 않았다.

B4/B8 마지막17일은 평가 대상 행0으로 NOT_ESTIMABLE_NO_ROWS다. 원래 구간을 이동하지 않았다. 모든 평가키166268행을 보존했고 추가 outage unsupported0이다. B1/B3 target actual-depth 부재는 각각42/34행이지만 원래 nominal-depth metadata fallback으로 유지했다. 이 숫자는 과거 다른 지원표의33행을 복사하지 않고 이번 실제 population에서 계산했다.

## 설계·중복 감사

[사전등록](preregistration.md)과 [새 config](../../configs/experiments/p2_c3_training_comparison_20260906_v1.json) 참조. C60(60epoch/wd1e-4), L120(120epoch/wd1e-4), D60(60epoch/wd1e-3) 외 변수는 고정했다. 이전 absolute-Celsius MSE 2×2 및 실패한 crossfit copula를 중복 실행하지 않았다.

24 screen fits=3recipes×8fold×seed20260901. B3 control/L120 각각 seed20260902/03 추가4fits, 합계28이다. B3는 후보 선택에 이미 사용했으므로 추가 seed는 seed 민감도 검증이지 새로운 독립 holdout이 아니다. 전체8fold3seed라고 주장하지 않는다. 주평가의 엄격한 평균 개선만으로 후보를 유지하고 CI0.8/outage-neutral 같은 사후 hard gate를 추가하지 않았다.

## 검증·실측·계보

- 학습: 2026-09-06 13:15:50 KST 시작, worker38328, 28/28 성공, 757.673434초. pilot B1/C60 15.453초는 실제 fit으로 재사용했고 추가 pilot fit0. GPU0 RTX5090/CPU2, DataLoader0.
- 합성 pytest15 PASS/Ruff PASS. 합성 중 별도 subprocess의 tiny CPU 1epoch×2seed는 historical fit28과 분리된 안전 경로 검사다.
- 새 PID40764 전체28모델의 **모든 natural 및 지원되는 outage validation행** exact replay PASS,25.156초. 최대오차0이며 probe-only 검사가 아니다.
- [독립 QA](independent-qa.json) 821/821 PASS: SSE/n/RMSE/분모·seed 평균·28fit 조합·model/prediction/source/recipe SHA·replay 연결.
- [root 독립 산술 QA](../parallel_core_training_20260906_v1/p2-root-independent-qa.json)도 PASS. 각8fold의 recipe/seed별 학습 keys/truth/arrays SHA unique수는 모두1로 같았다.
- 최종 state_dict28개+전 recipe/seed/학습 input 지문/전체평가 예측을 보존했다. 학습재개용 optimizer checkpoint가 아니라 exactly-once 최종 모델 재생 기록이다.
- 공식 index/sample/hidden/이전답안 읽기0, CSV0, upload0, Git변경0. 배포 observations.csv SHA 불변. sealed core/runner/config/이전모델/attempt lock을 바꾸지 않았다.

| artifact | SHA-256 |
|---|---|
| terminal_result.json | 8b793a62350203eb29ba834251d25c58ff83143ac3c97416bcbcb739663250ba |
| evaluation.npz | 085b6ba5511ef3cb3103097cab7397dcf1d2935f5531788fcfb098b7757f127e |
| fresh-replay.json | eeabb32721aff4a822d0a79ce43c4117784dd039c10967445e07a644e39574c8 |
| independent-qa.json | c5114db0b153a668b558f289f8c593a82241d7df0ed86a3c59183ddcb7f84d46 |

Artifact root: artifacts/p2_c3_training_comparison_20260906_v1/; numerical/replay entrypoint: scripts/p2_c3_training_comparison_20260906_v1/{run.py,qa.py}. 이미 소비된 training/replay lock은 재사용하지 않는다.

## 다음 판단과 미완료

현재 L120은 **작은 가을 평균 개선 + 상당한 결측 위험을 가진 정보가치 후보**다. 예상 공식 점수는 산출하지 않는다. 공식 점수는 해당 exact CSV가 제출된 후에만 붙인다. 기존 적격 CUDA C3 SHA46d194...c071을 보존한다.

root는 위 위험을 확인하고 별도 [후속 full3 범위](deployment-scope-pending.md)를 승인했다. 신규 [p2_c3_training_comparison_full_20260906_v1 보고서](../p2_c3_training_comparison_full_20260906_v1/report-source.md)에서 동일 L120 full3 scratch/내부QA/공식26061행 답안과 새 PID 전체CSV replay까지 완료했다(SHA fee6118...ce4d). 이 historical 실험 자체에는 full model/공식 CSV가 없으며, 합계31 연구·배포 fits 중 추가3은 별도 ID의 산출물이다. 포털 업로드/채점은 미실행이다.
