# P1 CPU2 tuning + fixed MS bridge

**완료: 같은 CPU2 트리에 같은 고정 MS proposal을 OR한 후보는 pooled F1 0.905472150→0.905169867, Δ−0.000302283이다. 실제 제출 구성의 평균 개선이 확인되지 않아 기존 b2f17/fallback을 유지하고 새 full13fit/공식 후보 생성은 이번 사이클에서 실행하지 않는다.** 특정 Q 하락이나 P<0.8 때문이 아니라 같은 구성의 pooled 평균 감소가 판단 근거다.

0-fit, 287,862행 전체키/원래 fold 소유권 PASS, 독립 산식181검사 PASS, runtime4.359초. Synthetic8/Ruff PASS. 공식 입력·CSV·upload·Git0.

| 비교 | 같은 MS + CPU2 기준 | 같은 MS + CPU2 후보 | ΔF1 |
|---|---:|---:|---:|
| Q3/Q4 pooled, 287,862행 | 0.905472150 | 0.905169867 | −0.000302283 |
| Q3, 176,738행 | 0.913129645 | 0.912109698 | −0.001019947 |
| Q4, 111,124행 | 0.894351841 | 0.894929303 | +0.000577462 |

Pooled 양성10,685행: TP9,258→9,192(−66), FP506→433(−73), FN1,427→1,493(+66). Fold-stratified KST7일 block 15+11개, CI90 [−0.005915730,+0.004809858], P(Δ>0)=0.4945. 선언된 동일 CPU2 비교의 작은 음수 결과이지 모집단 열등성이 통계적으로 확증되었다는 주장은 아니다.

Worst station-layer I-ORS5: 7,334행/양성201, F1 0.411067194→0.019704433, Δ−0.391362760; 고정 MS가 여기서 잃은 TP50을 복구하지 않았다. Worst Q4/block44: 15,087행/양성732, Δ−0.043911768. 위험별 후처리나 셀 패치를 추가하지 않았다.

**과거 최고와의 참고 비교는 별개다.** [기존 CPU4 union OR e150 진단](../p1_champion_reconstruction_20260906_v1/union-report.md)의 같은 Q3/Q4 F1은0.9069664686533457이다. 새 후보0.9051698670605614는 이 값보다0.0017966015927843 낮다. 이는 CPU2/CPU4 학습·선택 차이가 섞인 참고 수치로, 학습률 변경의 인과효과나 공식 점수 차이로 해석하지 않는다. 이번 실험이 현재 최고를 갱신했다고 주장하지 않는다. 기존 CPU4 OOF/모델/답안은 읽지 않았고 이 비교에는 작은 집계 보고서만 사용했다.

## 완결 증거

- [result.json](../../artifacts/p1_core_tuning_ms_bridge_20260906_v1/result.json), SHA `5ada025491863783982dcb2ef7e7d80c48423f2f4e158785002a34f4db6ae9a8`.
- [independent QA](../../artifacts/p1_core_tuning_ms_bridge_20260906_v1/independent-qa.json), SHA `e2982510dfd4acdb95c30dff3a55a5e374f43c3d12eebd827af6909a2a4ee36d`,181검사.
- Paired local-only parquet SHA `2b2bca5e32a8704de880b02ee48d6e634b09b9eb857e837daf6711b1f9517ed8`; 행별 값 출력/Git0.
- [seal.json](seal.json), SHA `a916cd732b5395b2261cf84f07ae1529910abbf42deccf4e20caa48b7632450d`. 새 tree terminal/QA/OOF 및 기존 MS proposal/QA SHA는 result의 `input_sha256`에 연결되어 있다.

## 실행 전 고정 범위

This is a separate retrospective **zero-fit diagnostic**, not a replacement of the tuning experiment's overall Q2/Q3/Q4 primary. It compares the same fixed original MS-TCN proposal OR fresh CPU2 tree control against that same proposal OR fresh CPU2 inner-selected tree candidate. It does not compare a CPU2 candidate to a CPU4 baseline and does not read the old CPU4 OOF/model.

The original [union-contract-v2](../p1_champion_reconstruction_20260906_v1/union-contract-v2.json) controls the frozen MS threshold/decoder, original Q3→Q4 ordered key digests, whole-run fold ownership, and fold-stratified calendar seven-day bootstrap (2,000 resamples, seed 20260906, CI90). All 287,862 keys must match bijectively including the Q3-owned October boundary rows. Missing, extra, duplicate or reassigned-fold rows fail closed; no intersection scoring.

The source/recipe and known MS proposal/QA SHA are checked before reading predictions. New tuning terminal + exact OOF SHA + linked independent QA must pass first. The bridge performs no selection, coefficient fitting, threshold changes, full-model training, official input, answer generation or upload. Existing b2f17/champion and original proposal remain immutable.

Artifacts: `artifacts/p1_core_tuning_ms_bridge_20260906_v1/{result.json,independent-qa.json,paired.parquet}`. [seal.json](seal.json) pins the new source and the immutable original ownership contract. Independent checks use sklearn confusion/F1 and an alternate scalar implementation of paired stratified resampling.
