# 결론: 고정 e150 OR는 네 트리 모두 pooled F1을 개선했으나 Q4 위험은 남았다

새 트리 O/B/union/router와 **같은 287,862행**을 대조한 결과, 원형 MS-TCN e150 제안을 OR하면 Q3+Q4 pooled F1이 모두 개선됐다. 최종 절대 F1은 `union OR e150`의 **0.906966469**로 가장 높았고, `O OR e150`은 0.906572006으로 근접했다. 각 arm에 OR를 더한 효과를 평가한 것이며 두 최종 후보 간 우열이 통계적으로 확증되었다는 뜻은 아니다.

**네 arm 모두 Q4 F1은 악화했다.** 과거에 반복 노출된 Q3/Q4를 재사용한 retrospective 복원 진단이므로 새 독립 확인이나 공식 점수 보장이 아니다. 현재 fallback, 전체 재학습 모델, 공식 답안과 이번 내부 점수는 각각 별도 근거다. 이 평가기는 후보 선택·공식 추론·업로드를 실행하지 않았다.

## 동일 행 비교

단위는 F1이며, TP/FP/FN을 먼저 합산한 pooled F1이다. Δ는 `동일 tree arm OR e150 − 동일 tree arm`이다. `union`은 트리 O OR B, `router`는 earlier inner에서 학습한 station-layer B/O/AND/OR 정책이다.

| 트리 arm | 트리 F1 | +e150 OR F1 | pooled ΔF1 | Δ CI90 | 재표집 Δ>0 비율 | Q3 Δ | Q4 Δ |
|---|---:|---:|---:|---|---:|---:|---:|
| O | 0.875598086 | 0.906572006 | +0.030973920 | [0.010260013, 0.055018349] | 0.9920 | +0.062148632 | −0.012442048 |
| B | 0.878668464 | 0.903799595 | +0.025131131 | [0.007015402, 0.044899992] | 0.9865 | +0.051455232 | −0.012327392 |
| union | 0.889264149 | 0.906966469 | +0.017702320 | [0.000335083, 0.036505590] | 0.9540 | +0.039638158 | −0.013630062 |
| router | 0.872615449 | 0.905548705 | +0.032933257 | [0.012366585, 0.056028983] | 0.9945 | +0.064942516 | −0.012319770 |

Q3 176,738행과 Q4 111,124행, 총 287,862행이다. key+fold 누락·추가·중복·암묵적 교집합 제외는 0행이다. 트리의 전체 3-fold 평가 및 Q2는 트리 canonical 결과에 있으며 여기에서는 MS-TCN 지원이 있는 사전고정 Q3/Q4만 비교한다.

`union OR e150`의 confusion counts는 TP 9,237 / FP 447 / FN 1,448 / TN 276,730이며, union 단독은 TP 8,689 / FP 168 / FN 1,996 / TN 277,009다. 827행을 양성으로 추가했고 기존 양성 제거는 0행이다. 모든 행에서 동일 고정 OR를 적용했으며 특정 정점·셀의 수동 패치는 없다.

## 위험과 불확실성

- Q4에서 union F1은 0.912866148 → 0.899236086, Δ −0.013630062다. Q4에 추가한 156행 중 TP 증가 15 / FP 증가 141로, Q3 이득을 Q4에도 동일하게 기대할 수 없다.
- 네 arm 모두 26개 시간블록 중 4개가 악화했다. union의 worst-block Δ는 −0.079944182이며 7,013행이다. Q4 block 47, 즉 KST 2025-11-26~12-02에 해당한다.
- 정점 전체 집계는 세 정점 모두 개선했지만, station-layer 16개 중 union/B는 2개, O/router는 1개가 악화했다. union의 worst station-layer는 I-ORS layer 4, 19,912행, Δ −0.034983941이다. 정점 전체 평균만으로 층별 악화를 숨기지 않는다.
- CI90는 KST 2025-01-01을 기준으로 한 7일 시간블록을 **원래 fold 내에서** paired bootstrap 2,000회 재표집했다(seed 20260906, Q3 15개/Q4 11개 블록). 매 재표집에서도 pooled TP/FP/FN으로 F1을 계산했다. 일별 bootstrap을 사용하는 트리 단독 QA와 다른 사전고정 분석이다.
- Δ>0 비율은 bootstrap 표본에서 양수인 비율이다. fresh test의 p-value, 공식 개선 확률 또는 Bayesian posterior로 해석하지 않는다. 블록 사이 독립성·계절 전이·반복 노출 및 네 arm 비교에 따른 선택 위험도 남는다.
- bootstrap 평균 Δ는 O +0.031962704, B +0.025717800, union +0.018203474, router +0.033786429다. 표의 실측 pooled Δ와 구분한다.

## provenance, 시간 분리, 기술 정정

트리는 새로운 O1+B3 inner/outer 24fits의 terminal OOF를 사용했다. MS-TCN은 이미 노출된 원형 e150 Q3/Q4의 6개 historical fits에서 고정 proposal만 exact-hash 재사용했다. old router/candidate 배열·과거 제출 답안·공식 입력·hidden 값은 읽지 않았다. 새로운 full MS-TCN 3fit의 현재 학습 결과를 미리 평가했다고 주장하지 않는다.

MS-TCN의 마지막 학습 시각은 Q3 2025-06-09 14:50 UTC, Q4 2025-09-09 14:50 UTC다. 각 holdout 최소 시각까지 **504시간 10분**으로, 기존 feature dependency 337시간보다 길다. 원형 MS-TCN은 허용된 offline 양방향 문맥을 포함하며 전부 past-only라고 부르지 않는다.

트리는 train-only year-depth dictionary와 plateau 168h cap의 80열을 유지했다. 원래 batch-depth/full-plateau 특징까지 동일하게 복원한 실험은 아니며, 과거 공식 28.9점 또는 F1 0.833548을 새 후보의 성적으로 승계하지 않는다.

첫 평가 v1은 달력 분기로 fold를 유도하여 Q3 run에 속한 10월 119행을 Q4로 잘못 배정했다. 성능 열을 읽기 전에 중단했고 [실패 기록](union-evaluation-v1-technical-failure.json)을 보존했다. 승인된 v2는 추출기의 Q3→Q4 concatenation 순서 및 각 fold의 **원본 ordered key SHA**를 검증해 원래 소유권을 복구했다. 119행을 삭제하거나 예측값·임계·bootstrap 설정·평가 범위를 바꾸지 않았다. 별도 회귀검사에 이 119행 경계 사례를 포함했다.

## 실행·QA와 다음 판단

- 새 fit 0, 기존 모델 수정/재시작 0, 공식/hidden/CSV/upload 모두 0.
- v2 평가 17.938초. CPU1, GPU 미사용. 트리 실제 학습 시간 및 full 재생성 시간은 별도 실행 receipt를 따른다.
- 합성 검사: v1 평가 13개, 독립 QA 6개, v2 fold 정정 5개 PASS. 각각 해당 코드의 Ruff PASS. 이 검사 수는 fit 수가 아니다.
- 실측 sklearn confusion/F1 산술 384 checks PASS. 별도 검증기의 bootstrap 평균·CI90·양수 비율·slice·worst-risk·원래 key/fold 소유권 **793 checks PASS**.
- 원형 e150 OR의 정보가치는 확인됐지만 Q4 악화와 station-layer 위험을 함께 보고해야 한다. 이 평가에서 자동 승격·새 full fit·규칙 재튜닝을 하지 않았다. root가 새 전체 학습·replay·공식 키 QA를 완료한 후 후보를 명시 결정한다.

## 단일 근거와 재현 명령

- 결과: [`result.json`](../../artifacts/p1_champion_reconstruction_20260906_v2_union_evaluation/result.json), SHA-256 `24ca88594c4cdb5d5dc2c3ae5defa1415823f8cfcebe463820b140dd9376146f`.
- paired artifact 및 각 scope/slice의 전체 분모·TP/FP/FN은 위 결과가 해시로 연결한다. 로컬 train-derived parquet이며 Git 대상이 아니다.
- [산술 QA](../../artifacts/p1_champion_reconstruction_20260906_v2_union_evaluation/independent-qa.json), [별도 bootstrap/risk QA](../../artifacts/p1_champion_reconstruction_20260906_v2_union_evaluation/independent-bootstrap-qa.json), 후자 SHA `679809a607e94df35d2fbee4f3d329b8acbfb1b7a6b62b8ed933cb02ef55c173`.
- [v2 계약](union-contract-v2.json), [봉인](union-seal-v2.json), [원 v1 실패 설명](union-evaluation-v1-failure.md).

이미 실행된 명령이며 기존 output/receipt에 다시 실행하지 않는다.

```powershell
.venv-p1/Scripts/python.exe -I scripts/p1_champion_reconstruction_20260906_v1/evaluate_union_v2.py
.venv-p1/Scripts/python.exe -I scripts/p1_champion_reconstruction_20260906_v1/qa_union_v2.py
```
