# 무엇이 끝났고 무엇이 남았는가

| 질문 | 현재 증거 | 판정 / 다음 경계 |
|---|---|---|
| 긴 사건 외곽 신호를 더하면 좋아지는가 | 동일 seed/split의 12 LGBM+6 XGB screen; selected 후보 F1 0.836928, 강한 control 0.851174 | 해당 25-feature arm 폐기. 추가 seed/HPO 실행 없음 |
| 양성 anchor를 일부 제거하는 이진 디코더는 도움이 되는가 | backbone 0-fit, 전이 추정 6회. inner on/off 전략 F1 0.850291 | 사전등록 전략 폐기. always-on 진단의 긍정 수치로 바꾸지 않음 |
| 완전 재학습 가능한 기존 부품 자체에 가치가 있는가 | 같은 평가행에서 inner-selected O/B control 0.851174, XGB-alone 0.843227 | 새 clean 전체 학습 후보로 보존할 가치가 있음. 역사적 공식 모델보다 우수함을 입증한 것은 아님 |
| 과거 최종 답안/행 패치가 필요하지 않은가 | 새 경로는 distributed train→O/B 두 새 모델→고정 inner threshold→CSV만 사용 | router_anchor/GI2/MS checkpoint 입력 없음 |
| 내부와 공식 입력의 조건이 같은가 | 2026 year-key nominal-depth는 unseen. Q4 inner는 seen-year | covariate shift 가능성 명시. 이번 조건에서 사후 depth fallback/threshold 변경 없음 |
| 저장 모델로 똑같이 답안을 다시 만드는가 | 모델 probe 및 별도 --verify 과정의 QA receipt 참조 | 모델 reload 재현과 전체 새 2-fit 재현을 구분 |
| 공식 최고점을 갱신했는가 | 새 CSV 아직 공식 채점하지 않음 | 불명. 역사적 28.909341 귀속 금지 |
| 최종 제출 파일로 검증됐는가 | 새 로컬 code/model/answer 준비 | ZIP/다운로드 없이 새 환경 실행/6시간 전체 재현/포털 업로드는 별도 미검증 |

현재 실행에는 upload, commit, push 권한이 포함되지 않는다. 후보 보존 뒤 새 정책 실험 또는 최종 제출은 root의 별도 판단을 따른다.
