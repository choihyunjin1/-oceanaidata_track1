# 연구 원문 — 점수 개선 재설계 2026-09-05 v2

## 결론

새 아키텍처를 늘리는 것보다 P1 feature 계약, P2 손실 단위, P3 결합 목적을 먼저 시험한다. 문제별 세 병렬 read-only 조사와 root의 직접 코드 대조를 종합했다. 독자용 최종 산출물은 [다음 실행 설계](../../docs/SCORE_IMPROVEMENT_PLAN_20260905_V2.md)다. **계획 작성만 완료했으며 학습·코드 변경·공식값 재열람·CSV·업로드·Git 변경은 하지 않았다.**

## 질문과 연구 범위

- 질문: 합법적·완전 재학습 가능한 후보의 점수를 어떻게 올릴 것인가? 무조건 새로운 기법이 필요한가? 과거 실패는 정말 동일한 가설을 반박하는가?
- 사용자 조건: 현재 Astra로 직접 추론하고 Deep Research를 병행하되 실제 실행은 이후 낮은 추론 강도로 진행한다.
- 최신 기준: 2026-09-05 HEAD `535f94a1791f2398f82ad27659b58701513ab327` 및 현재 dirty 상태. 공식 성적은 직전 제출 영수증을 다시 읽었고, 이번 연구에서 portal 또는 원본/공식 행별 관측을 열지 않았다.
- 원본 README 세 개·최상위 운영진 정책·문제별 필독문서는 읽었다. 자료의 과제·접근 계약만 확인했으며 외부 관측을 수집하지 않았다.

## 조사 진행 기록

| 단계 | 상태 | 수행 내용 |
|---|---|---|
| 범위/근거 지도 | 완료 | 계획만, 적격 모델만, 오늘 official 영수증과 source 정책을 우선 |
| 병렬 발견 | 완료 | P1 배포/최고점 부품, P2 objective/season, P3 direct SSE/anchor population |
| 후속 확인 | 완료 | root가 P1 depth 코드, P2 loss/weights, P3 router 목적과 GPU 경로 직접 확인 |
| 문헌 대조 | 완료 | PyTorch losses, sklearn stacking/time split, Cawley & Talbot 선택편향 |
| 종합/검증 | 완료 | bounded A→B 실행, fit 예산, GPU 큐, 불확실성·기회 배분 문서화 |

현재 도구 목록에 `update_plan`이 없어 이 표로 진행 단계를 기록했다. 하위 조사자에게 별도 artifact 작성·실행·재위임을 요청하지 않았다. 실제 Deep Research 스킬을 적용한 자료 조사와 Astra 추론이며 Gemini나 별도 원격 Deep Research 작업을 실행했다고 주장하지 않는다.

## 핵심 재해석

### P1

오늘 모델의 수심 사전은 station/year/layer를 key로 사용한다. CV에서는 2025 prefix가 있지만 배포는 2026이므로 nominal-depth가 모두 missing이라는 배포 보고와 일치한다. 물리 수심이 아니라 단순 layer fallback으로 수리하면 연도별 센서 위치 변화를 무시한다. 현재 관측 depth를 일관되게 표현하는 대조가 필요하다.

또 오늘 모델은 XGB+LGBM 두 개이고 과거 최고 조합은 MS-TCN/일반 router/spike 정책이 포함된다. 전자는 후자의 clean 재현 결과가 아니다. P1 과거 전체를 비적격으로 단정하지 않고 부품별 최초 선택·재생성 경로를 확인한다. 원인·효과는 새 학습 전 미확정이다.

### P2

공식은 절대 ℃ RMSE지만 현재 normalized residual의 SmoothL1를 domain-weighted 학습한다. 작은 오차 영역의 implicit ℃ weight는 scale에 의존한다. 현재 목적이 일반화에 유리할 수도 있으므로 loss×weight 2×2 대조가 적절하다. 입력-gradient regularizer까지 바꾸지 않도록 기존 Huber와 기존 regularizer weights를 보존한다.

내부 pooled 개선−0.061096℃ 중 같은 가을 fold는−0.004071℃뿐이다. 비증강 control 공식 미제출이므로 blockmask의 공식 효과를 부정할 수 없다. 가을 주평가와 다른계절 스트레스를 나누되 역사 재노출 사실을 유지한다.

### P3

현재 clean `.607183m`는 적법한 과거 원형 `.607071m`와 유사하다. 비적격 `.583892m`를 복원하기 위해 Publicα를 다시 쓰면 안 된다. loss router는 component log-loss 모델이지 최종 혼합 SSE 최적화가 아니다. 적격 OOF가 재현되어 있으므로 낮은 자유도의 과거-only 메타 단계가 가장 저렴한 다음 질문이다.

촘촘한 학습 anchor와 희소 평가 사건의 차이는 episode-size weighting으로 별도 시험한다. 모델 결합과 가중치를 동시에 바꾸면 어느 효과인지 알 수 없으므로 분리한다. 단순 전체OOF LOFO는 base 모델을 통한 outer-label 간접 누출 가능성이 있어 채택하지 않는다.

## 문헌이 지지하는 것과 지지하지 않는 것

[Cawley & Talbot 2010](https://www.jmlr.org/papers/v11/cawley10a.html)은 유한 평가면에서 모델 선택을 반복하면 선택 기준 자체를 과적합할 수 있음을 다룬다. 이 때문에 새로운 이름의 실험을 무제한 늘리지 않고 노출 이력·후보 수를 기록한다. 이 논문은 우리 후보의 예상 상승폭을 제공하지 않는다.

[StackingRegressor 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html)는 메타 학습용 교차 예측과 같은 자료의 prefit 예측을 구별한다. [교차검증 안내](https://scikit-learn.org/stable/modules/cross_validation.html)는 시간·그룹 의존 구조를 고려한다. 따라서 여기서는 일반 무작위 KFold가 아니라 P2 nested masked split, P3 과거-only 순차 split을 설계했다.

[PyTorch MSE](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MSELoss.html)와 [SmoothL1](https://docs.pytorch.org/docs/2.14/generated/torch.nn.SmoothL1Loss.html)은 손실 정의의 근거다. normalized target을 대입한 implicit weighting은 root의 대수적 해석이며 실제 성능 효과는 미측정이다.

## 중단 규칙과 열린 질문

유용한 결정은 가능한 상태다. P1 원인 기여도, P2 ℃ 손실의 일반화, P3 저차원 보정의 drift는 더 많은 문헌보다 bounded 실험이 필요하다. 현재 자료는 효과 추정을 뒷받침하지 않으므로 새 점수 예측은 하지 않는다. 기존 기준선 보존과 하드 적격성 검사는 약화하지 않는다.

최종 planning QA는 파일·링크·사실/가설 구분·fit 산술·자원 충돌·금지 계보 제외를 확인하는 문서 검토다. 이번 턴의 focused pytest/Ruff·학습 성능 QA를 수행했다고 표시하지 않는다.

로컬 문서4개와 상대 링크34개를 검사해 깨진 링크0을 확인했다. P3 독립 검토에서 새 backbone에 옛 router 계수를 복사할 수 있는 모호함을 발견해, 각 arm의 평균 OOF로 같은 router 설정을 과거-only 재적합하고 작은 fits를 별도 집계하도록 명확히 했다. 이는 설계 수정이며 기존 실행 artifact를 수정한 것이 아니다.
