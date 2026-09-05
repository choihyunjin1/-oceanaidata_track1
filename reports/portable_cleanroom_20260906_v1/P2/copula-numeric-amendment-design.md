# 未실행 설계: 기존 cross-fit copula의 수치 불변성 검사 정정

현재 상태는 **DESIGN_ONLY_NOT_AUTHORIZED_TO_EXECUTE_HERE**다. 기존 `p2_crossfit_copula_forward_20260906_v1`의 terminal failure·봉인·40개 실제 fit은 모두 보존한다. 이 문서는 새 실행이나 수정 승인을 대신하지 않는다.

## 정정 범위

제안 ID: `p2_crossfit_copula_forward_numeric_20260906_v2`.

원래 중단은 B2의 비outage 19,402행 중 1행의 `3.552713678800501e-15`℃ 차이였다. 동일 11개 물리 특징·기준선 입력의 bit identity와 동일 shape 부분집합의 출력 exact identity는 이미 입증됐다. 후보 성능에 따른 변경은 아니다.

새 실행 전에 수치 검사를 다음과 같이 별도 봉인한다.

1. 비outage행의 키, C3 값, 11개 물리 특징과 NaN 위치는 여전히 **bit-exact equality**여야 한다. 이 입력 검사는 완화하지 않는다.
2. 같은 입력을 서로 다른 전체 missingness batch shape로 평가한 copula 최종 온도 차이만 `rtol=0, atol=1e-12℃`로 검사한다. 이는 관측 오차/승격 허용치가 아니라 float64 선형대수·15-node quadrature의 수치 roundoff 한계다. 실제 관측된 최대 차이의 약281배이나 기존 CSV serialization 검사 `1e-8℃`보다 10,000배 작다. 성능을 보며 늘리지 않는다.
3. 한계를 넘으면 새 technical terminal로 종료한다. 비outage 출력을 사후 복사하거나 clip/가중치/threshold를 조정하지 않는다.
4. 동일 전체 입력·동일 batch shape의 새 프로세스 재현은 기존대로 **전체 bit-exact**를 요구한다. 자연/마스킹 두 면의 비교만 위 수치 한계를 사용한다.
5. 큰 배치/결측 패턴 변경을 포함한 synthetic 검사로 tolerance 이내 roundoff는 허용하고, 특징 변화·순서 변화·`1e-9℃`의 인위적 출력 변화는 차단함을 먼저 확인한다.

## 재사용 계보와 추가 비용

우선 가능한 다음 단계는 **0-fit provenance/replay 검증 설계**다. 이미 존재하는 36개 신경망과 4개 copula의 hash·calibration·키 계보를 확인하고, 동일 저장 모델에서 기존 예측값을 다시 계산하여 수치 예외를 설명할 수 있다. 예측을 수정하거나 기존모델을 재학습하지 않는다. 이 범위만으로 아직 없는 B3~B8 inner 모델이나 전체 후보 비교를 완성했다고 주장할 수 없다. 아래 48개 신규 fit은 전체 연구를 완성하려면 필요한 별도 비용 산정이며, 현재 승인·자동 실행 대상이 아니다.

- 기존 8-fold C3 24 models, B1/B2 inner 12 models = **36개 신경망**을 exact SHA로 재사용.
- 기존 B1/B2 × in-sample/cross-fit **4개 copula**도 새로 fit하지 않는다. 마지막 B2 copula는 fit/save는 끝났으나 assertion 뒤 영수증을 쓰지 못했으므로 calibration 배열·원래 residual·모델 CDF/covariance 및 수치 진단으로 별도 검증 영수증을 만든 후에만 재사용 가능하다.
- 모델·배열·source·코드 의존성·train keys·seed·epoch·CPU 설정이 원 v1과 맞지 않으면 재사용하지 않고 blocker로 보고한다. 기존파일을 수정하지 않는다.
- 남은 B3~B8 × 2 inner blocks × 3 seeds = **36 neural fits**, 남은 6 folds × 2 copula = **12 small fits**, 총 **48개 신규 fit**만 필요하다. 기존40 + 추가48 = 원래 계획88개이며 반복학습 비용을 추가하지 않는다.
- 외부/public 역산 계보, 옛 답안/기존 다른 실험 OOF 사용 0. v1의 새 source-only 모델·calibration 배열만 read-only 승인 목록에 고정한다.

주평가 B3/8-fold 분모·inner 월·2단계 purge·C3 recipe·11특징·강도1.0·90분 원래 계획·CI90·후보 보존 방침은 그대로다. 0.8 또는 outage 비악화 gate를 새로 추가하지 않는다. 남은 실행 약30분은 앞선 실제 fit속도로부터의 추정이며 성능을 읽기 전에 자원 예산을 다시 산정한다.

이번 portable 패키지 작업에서는 위 수정 runner 작성·fit·공식 입력·CSV·업로드를 전혀 진행하지 않았다.
