# 상수·학습 산출물 계보 — 현재 재생성 경로

이 문서는 숫자가 있다는 이유만으로 위반을 판정하는 lint가 아니다. **실행에 쓰이는 값의 출처**와 재생성 결과를 구분하는 감사 원장이다. 아래 문서와 각 실행의 dependency/model SHA, 경로 가드, 별도 PID replay를 함께 확인한다. 아직 모든 실행·최종 패키지가 완료됐다는 의미는 아니다.

| 문제/항목 | 출처와 저장 방식 | 판정/주의 |
|---|---|---|
| P1 XGBoost O / LightGBM B 트리·encoder | 배포 train에서 이번 실행이 fit한 joblib | 새 모델만 추론에 로드. fitted tree를 JSON이 아닌 joblib로 저장했다는 이유로 부적격이 되지 않음 |
| P1 train statistics / decoder 정책·threshold | 새 Q4 earlier-inner fit·평가로 산출한 recipe JSON | 예상 과거 정책은 사후 일치 검사만 수행. 임계값을 답안에서 복사하거나 강제하지 않음 |
| P1 seed / CPU thread / model hyperparameters | 명시한 clean recipe와 실행 환경 | 재학습 결과에 영향을 줄 수 있어 출처·순서·자원 조건을 기록. 과거 공식 score는 새 답안에 전이하지 않음 |
| P2 DeepSets weights / scaling | 배포 observations에서 새 3-seed training의 state dict와 training metadata | 기존 pt/답안 입력 금지. 새 산출물 SHA를 읽는 동작은 weight 재사용과 다름 |
| P2 copula rank CDF / covariance / residual state | 배포 학습 행의 프로파일 및 새 C 예측 잔차에서 산출 | full correction은 새 C 재생성 이후만 실행. historical OOF·공식 답안을 full-fit target으로 사용하지 않음 |
| P2 full/half correction 강도 | 실험 전 선언한 후보 정책 1/0.5/0 | v4에서는 full primary를 유지. stress 결과를 보고 강도를 재튜닝하지 않음 |
| P3 single/multi CatBoost | 배포 train_wave/train_atmos로 이번 실행이 생성한 모델 | 외부 사전학습 및 TabPFN 가중치 미사용 |
| P3 router 적합 계수 | 새 historical OOF를 만들고 이전 완료 fold에서만 학습; 마지막 full router는 새 전체 OOF에서 fit | 과거 router coefficients/answer 복사 아님 |
| P3 Ridge alpha=10 / long-lead shrink=0.2 | 기존 사전등록 clean recipe의 정규화·구조 설정 | 금지된 Public 역산 alpha=-10.217 및 axis 체인과 다름. 값 이름이 alpha라는 이유만으로 같은 계보로 분류하지 않음 |

## 제외해야 할 과거 계보

`router_anchor.csv`, `gi_spike2_patch.json`, `bin17_anchor.csv`, Public 역산 alpha/axis 파일 및 외부 ERA5/KMA/KIOST/실관측 pretrained model은 새 실행 입력이 아니다. 역사적 감사 자료는 삭제하거나 은폐하지 않는다. `02_code`의 넓은 source snapshot에 과거 모듈이 존재하는 것은 해당 모듈의 데이터·계수가 실행에 쓰였다는 증거가 아니지만, 최종 portable 패키지에는 실제 의존성만 선별해야 한다.

## 재현의 서로 다른 조건

1. **새 학습 가능**: 빈 `03_model`에서 배포 데이터만으로 모델과 답안을 생성한다.
2. **저장 모델 replay**: 그 새 모델을 다른 프로세스에서 로드했을 때 답안이 동일하다.
3. **과거 제출본 exact 복원**: 새 학습 답안이 과거 제출 SHA와 같다. 1/2가 통과해도 3은 실패할 수 있다.
4. **최종 재현 패키지**: 독립 경로·환경·차단망에서 정해진 시간 내 1/2를 검증한다. repo 의존 snapshot만으로는 완료가 아니다.

CatBoost는 GPU 학습에서 부동소수점 합산 순서로 인한 비결정성을 명시한다. 그러므로 3의 실패를 자동으로 데이터 계보 위반으로 해석하지 않으며, 반대로 비결정성이라는 이유로 split/target/학습·추론 오류를 면제하지도 않는다. [CatBoost 공식 GPU 문서](https://catboost.ai/docs/en/features/training-on-gpu), 2026-09-05 확인.

최종 README는 “모든 적합값은 이 학습 실행에서 생성한 모델/metadata 파일에서 온다; 리더보드 역산 계수 및 과거 답안 입력 0”처럼 실제 저장 방식과 출처에 맞게 작성해야 한다. 확인하지 않은 독립 환경 PASS나 운영진 적격성 확정은 쓰지 않는다.
