# P1 MSTCN 사건 단위 라우터 회고 실험

## 결론

일반 사건 단위 라우터는 **정식 `NO_GO_RETROSPECTIVE_GATE`**다. Q3에서는 현 incumbent보다 F1이 `+0.012105` 높았지만 raw e150보다 `−0.005104` 낮았고, Q4에서는 incumbent보다 `−0.015115` 낮아졌다. Q3·Q4 pooled F1도 `0.903886`으로 raw e150의 `0.906804`를 넘지 못했다. 이 core 라우터를 제출 후보로 승격하지 않는다.

그러나 유형 확률의 쓰임은 좁혀졌다. Q3의 MSTCN 유형·경계 출력을 학습한 type-augmented 라우터는 Q4에서 MSTCN 추가 구간 4개를 모두 거절해 incumbent F1 `0.914342`를 정확히 복원했다. raw e150의 Q4 F1 `0.898901`보다 `+0.015441` 높다. 유형 head는 독립 검출기보다 **불안정한 추가 구간을 차단하는 veto/abstention 신호**로 연구할 가치가 있다.

## 무엇을 중복하지 않았는가

정찰 결과 다음 축은 이미 실행돼 실패 또는 불안정 판정을 받았다.

- five one-vs-rest type heads + factorial semi-Markov decoder
- typed duration semi-Markov decoder
- typed multiclass LightGBM union
- incumbent rule distillation neural residual
- generic change-point proposal scorer와 direct interval set

따라서 이번 실험은 새 유형 모델을 다시 학습하지 않았다. 이미 저장된 e150 MSTCN의 row probability, boundary probability, type probability와 frozen incumbent OOF만 사용해 MSTCN이 추가하려는 연속 구간을 사건 단위로 승인 또는 거절했다.

## 고정 설계

- core: Q2 구간으로 적합해 Q3 평가, 이어 Q2+Q3으로 적합해 Q4 평가
- type-augmented: 유형 확률이 함께 저장된 Q3 구간으로 적합해 Q4 평가
- 모델: 표준화된 저용량 logistic regression, `C=1`, balanced class weight, acceptance probability `0.5`
- target: 추가 구간의 precision이 해당 fold incumbent F1의 절반보다 클 때만 beneficial
- incumbent는 절대 삭제하지 않고 승인된 구간만 union
- 결과 기반 재학습·threshold 변경 없음

## 결과

| 평가 | 학습 | 후보 F1 | incumbent 대비 | raw e150 대비 | 판정 |
|---|---|---:|---:|---:|---|
| Q3 core | Q2 | 0.907085 | +0.012105 | −0.005104 | raw e150 미달 |
| Q4 core | Q2+Q3 | 0.899227 | −0.015115 | +0.000326 | incumbent 붕괴 |
| Q4 type-augmented | Q3 | 0.914342 | 0.000000 | +0.015441 | exact fallback |
| Q3+Q4 core pooled | 순방향 | 0.903886 | +0.000969 | −0.002918 | 정식 NO_GO |

구간 표본도 작고 fold shift가 크다. MSTCN 추가 구간은 Q2 27개 중 18개, Q3 46개 중 34개가 beneficial이었지만 Q4 4개 중 beneficial은 0개였다. core model이 Q4에서 2개 구간 139행을 승인하면서 true positive 1행과 false positive 138행을 추가한 것이 실패의 직접 원인이다.

## 후행 진단

사전 고정한 두 arm을 결과 후 조합한 `Q3 core + Q4 type fallback`은 pooled F1 `0.910009`로 incumbent보다 `+0.007092`, raw e150보다 `+0.003206` 높다. 하지만 이 조합은 Q4 결과를 본 뒤 계산한 산술 진단이므로 승격 증거가 아니다. 작은 양성 추세를 버리지 않기 위한 방향성 기록으로만 남긴다.

## 다음 판단

P1 다음 구조는 더 많은 유형별 검출기를 추가하는 것이 아니다. MSTCN 추가 구간에 대해 type posterior와 모델 불확실성을 사용한 **보수적 veto**를 학습하고, 실패 시 incumbent로 정확히 돌아가는 구조가 우선이다. 다음 실행 전에는 다음을 고정해야 한다.

1. type posterior를 이용할 수 있는 Q3·Q4 50개 구간만으로 표본이 충분한지 bootstrap/event support를 계산한다.
2. 학습 목표를 단순 beneficial 분류가 아니라 add-only F1의 정확한 증분으로 바꾼다.
3. official 후보를 만들 경우 현 e150+GI champion을 보존하고, veto로 삭제되는 행과 구간 수를 먼저 봉인한다.
4. 공식 제출은 별도 사용자 승인 전까지 생성·업로드하지 않는다.

## QA와 제한

- 독립 aggregate QA: `PASS_QA_OF_NO_GO_RESULT`
- Ruff: PASS
- pytest: 2 passed
- 공식 P1 test rows read: 0
- submission files created: 0
- uploads: 0
- 모든 Q2–Q4 label은 과거 연구에서 이미 노출됐으므로 fresh promotion evidence가 아니다.

정량 근거는 [evidence.json](evidence.json)에 보존했다. 원시 행, 모델 체크포인트, 공식 CSV는 보고서나 Git 커밋에 포함하지 않는다.
