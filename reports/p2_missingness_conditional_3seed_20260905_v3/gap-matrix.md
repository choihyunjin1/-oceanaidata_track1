# P2 conditional 3-seed — 남은 증거와 재사용 판단

| 질문 | 현재 증거 | 판단/제약 |
|---|---|---|
| 첫 seed의 가을 개선이 배포와 같은 3-seed 비교에서도 남는가? | intact +0.000783517℃, 가을7/14d +0.030801703℃ | 이번 정확한 conditional은 개선 근거 미재현 |
| 원래 C 및 R model/source/recipe를 그대로 재사용했는가? | C historical9/R historical3/C full3 해시와 원래 runner/config/dependencies 일치 | 신규 R9 fits만 수행, old A/B/0-fit 보존 |
| 실제 공식 행에서 몇 행이 바뀌며 점수는 얼마인가? | 공식 index/sample 미접근, CSV0, upload0 | 미측정. 내부 RMSE를 공식 점수로 치환하지 않음 |
| artificial3d support 부족은 실제 공식 결함인가? | 4 unsupported, 전체1,294행 scenario 미산정, key삭제0 | 실제 공식 입력과 별개. 지원제약과 성능 결론 분리 |
| 겨울 세 episode가 새 intervention인가? | 자연적으로 T5/S5가 이미 모두 결측 | no-op이며 독립 세 실험으로 해석하지 않음 |
| R standalone pooled 성능이 좋으니 전체 R을 내면 되는가? | pooled .821974℃는 C보다 좋지만 가을 .498233℃는 C보다 나쁨 | 승인된 rule을 결과 보고 바꾸지 않음. 별도 분기 미실행 |
| 오프라인 수치 재현 가능한가? | 저장 full6모델, 공개 특징128행, 다른 PID exact | 전체 official CSV나 scratch 재학습 bitwise 검증이 아님 |
| 배포 준비 코드는 있는가? | separate adapter+3 synthetic/Ruff PASS | 정보조건 불충족으로 미실행. 공식 승인 없는 실행 금지 |

정확한 수치와 분모는 [canonical report](report-source.md), [result](result.json), [QA](independent-qa.json)에만 연결한다. 다음 실험을 자동 실행하거나 가장 유리했던 첫 seed로 되돌아가지 않는다.
