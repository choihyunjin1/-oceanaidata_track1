# P1 구현·검증 가이드

작성일: 2026-08-13 KST
상태: 코어 구현과 첫 nested CV 완료. augmentation·deep·SSL·stress 승격 검증 및 최종 후보 재현은 계속 진행한다.

## 1. 목표와 해석

목표는 `station,year,layer,time` 각 행의 수온 센서 이상 여부를 판정하여 공식 binary F1을 최대화하는 것이다. `temp`만 판정 대상이고 `psal`, `depth`는 참고 변수다. `anomaly_type`은 순위 점수에 반영되지 않으므로 binary label을 우선하며, 유형 membership은 오류 분석과 본선 설명에 사용한다.

주최측 규칙 기반 기준값은 F1 0.548255다. 이 값은 공식 평가 집합에서 채점된 값이므로 train의 로컬 fold 점수와 직접 차감하지 않는다. 후보 승격은 반드시 동일한 로컬 split에서 계산한 기준선과 비교한다.

## 2. 고정 운영 원칙

- 배포된 완전한 시계열을 사용하는 양방향 `offline` QC가 메인이다. 공개 문제문에는 실시간/online 판정 제약이 없다.
- 미래 행을 사용하지 않는 `causal` 모드는 실운영 가능성과 규칙 변경 대응을 위한 독립 ablation이다.
- centered·양방향 피처는 동일한 연속 관측 segment 안에서만 계산하며, outer split purge가 최대 피처·후처리 의존 범위를 덮어야 한다.
- test label은 끝까지 미지 상태로 유지한다. 4% 양성률이나 baseline 예측률을 test에 강제로 맞추지 않는다.
- 외부 관측값은 운영진 서면 승인 전 사용하지 않는다. 2024~2026 KORS/KHOA 원자료와 실시간값은 원신호 복원 위험 때문에 승인 여부와 관계없이 제외한다.
- 원본 CSV·ZIP은 읽기 전용이며 Git, 문서, 노트북 출력, 제출 패키지에 복제하지 않는다.
- 하루 1회 기회를 보호하기 위해 사용자가 정확한 파일을 승인하기 전에는 업로드하지 않는다.

offline 문맥의 공식 범위는 문의 초안에 남겨 두지만, 답변 전까지 offline을 금지한다는 뜻은 아니다. 추후 공식 서면 규칙이 online-only라고 확인되면 causal 경로로 전환한다.

## 3. 환경과 재현 계약

- Python: CPython 3.12.10 x64
- 전용 환경: `.venv-p1`; 기존 `.venv`는 보존
- CPU/data 패키지: `requirements.txt`
- 공식 CUDA 13.0 PyTorch wheel: `requirements-dl.txt`
- 프로젝트 설치: `pip install --no-deps -e .`
- 설정: `configs/p1.toml`

표준 설치:

~~~powershell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap_env.ps1
~~~

bootstrap은 `.venv-p1` 생성, 고정 패키지 설치, editable 설치, `pip check`, CUDA smoke test를 실행한다. RTX 5090에서는 CUDA 사용 가능, capability `(12,0)`, `sm_120`, 행렬 연산과 역전파를 확인했다.

입력은 `P1_DATA_DIR` 또는 `--data-dir`로 받는다. 환경변수가 없을 때의 로컬 검색 fallback은 필수 파일 집합을 가진 후보가 정확히 하나일 때만 허용하고, 0개 또는 2개 이상이면 실패한다. 최종 패키지는 영문 상대경로만 사용한다.

## 4. 데이터 계약

| 데이터 | 필수 열 |
|---|---|
| train | `station,year,layer,time,temp,psal,depth,label,anomaly_type` |
| test | `station,year,layer,time,temp,psal,depth` |
| sample | `station,year,layer,time,label,anomaly_type` |

매 실행 전 다음을 검사한다.

- train 776,706행, test/sample 169,011행과 필수 열
- `station,year,layer,time` 결측·중복 없음
- sample/test 키와 순서 완전 일치
- `time` 파싱과 KST `+09:00`, `year` 일치
- label `{0,1}`, 수치 유한성, 값 범위와 결측률
- 입력 SHA-256과 원본 쓰기 금지

같은 station-layer 안에서도 인접 시각 차가 10분이 아니면 새 `segment_id`를 시작한다. rolling, plateau, 변화점, 후처리, 시퀀스 window는 segment 경계를 넘지 않는다.

`anomaly_type`은 `+`로 분리한 `spike`, `noise`, `flatline`, `offset`, `drift` 다중 membership으로 처리하고 중복 토큰은 제거한다. 정상은 빈 집합이다.

layer 번호를 고정 센서나 절대 수심으로 간주하지 않는다. 실제 `depth`의 강건한 중심으로 deployment/depth regime을 만들고, G-ORS 2026처럼 depth가 구조적으로 결측인 경우 station-layer mask와 시간 피처를 사용하는 fallback을 적용한다.

## 5. 피처와 규칙

모든 기준 통계는 station, depth regime, 연속 segment 단위의 median/MAD 계열로 강건화한다. 전역 고정 임계값은 출발점으로만 사용한다.

| 계열 | 핵심 피처 |
|---|---|
| spike | 이전·다음 차분, 3점 복귀도, 2차 차분, Hampel/MAD score |
| flatline | exact/epsilon plateau 전체 길이와 causal 길이, rolling range, 고유값 수 |
| noise | 3·6·12·24·48시간 차분 MAD/분산, 7일 배경 대비 비율 |
| offset/drift | 24·48·72시간·7일 잔차, 좌우 median 차이, CUSUM, robust slope |
| 층간 | 동시각 peer median·잔차·spread·peer 수, 최근 상관, 성층 gate |
| 보조 | psal/depth 변화와 mask, 계절·시각 주기, station/depth-regime category |

offline은 이전·다음 이웃과 centered 통계를 사용한다. causal은 같은 API에서 미래 의존 열을 제거하거나 과거 방향으로만 계산하며, prefix-invariance 테스트로 미래 행 변화가 과거 피처에 영향을 주지 않는지 검사한다.

outer fold 모두에서 `plateau length >= 6`이 사실상 무오탐 규칙인지 재확인한 뒤 plateau 시작점까지 `flatline` hard override한다. 공식 spike는 한 행이므로 어떤 최소길이 후처리도 singleton spike를 일괄 제거할 수 없다.

## 6. 모델 계단과 구현 상태

1. 제공 baseline 구조 감사와 설명 가능한 규칙 expert
2. deterministic LightGBM 주력, XGBoost·CatBoost 비교 API
3. fold train 내부 정상구간에만 다섯 이상과 중첩을 합성하는 augmentation
4. fold-local masked-reconstruction SSL과 embedding 추출
5. 14일 dilated TCN, 1시간 patch Transformer
6. 규칙·최고 tree·최고 deep 모델의 최대 3개 비음수 convex ensemble

LightGBM은 현재 검증된 주력이다. XGBoost와 CatBoost, deep, SSL, ensemble은 구현되었다는 사실만으로 승격하지 않는다. 동일 outer split, bootstrap, 그룹 하락 기준을 통과한 경우에만 test 후보에 반영한다.

deep 학습 구성은 binary BCE + soft-Dice와 다섯 유형 multi-label 보조 BCE, anomaly/normal crop 균형 sampling, 실제 class prior 보정, bf16 AMP, AdamW, 최대 50 epoch, patience 8이다. TCN·Transformer의 사전 정의된 12개 설정을 screening하고 상위 설정만 3개 seed로 재학습한다.

SSL은 outer train 안의 정상구간만 사용하고 masked reconstruction으로 사전학습한다. validation/test 또는 외부 관측값이 pretraining provenance에 섞이면 해당 run을 실패 처리한다.

## 7. nested 검증

outer rolling-origin split은 다음과 같이 고정한다. 각 경계에는 7일 purge를 둔다.

| fold | train 종료 | validation |
|---|---|---|
| 2025 Q2 | 2025-03-24 | 2025-04-01 ~ 2025-06-30 |
| 2025 Q3 | 2025-06-23 | 2025-07-01 ~ 2025-09-30 |
| 2025 Q4 | 2025-09-23 | 2025-10-01 ~ 2025-12-10 |

모델 반복 수, threshold, hysteresis, gap, 최소길이, augmentation 통계, scaling과 SSL은 각 outer train 내부의 과거 blocked inner split에서만 고른다. outer validation 라벨은 최종 fold 평가에 한 번만 사용한다.

공식 주지표는 행 단위 micro F1이다. 함께 보고할 항목은 다음과 같다.

- test station-layer 구성비 재가중 F1
- 유형 membership별 recall과 중첩 이상 성능
- 이벤트 검출률과 정상 1일당 false positive
- station, layer, station-layer F1과 최악 그룹
- positive event와 정상 24시간 block 단위 paired bootstrap 2,000회
- S-ORS 2024→2025 상반기 year-transfer와 supervised G-ORS 전체 holdout stress

## 8. 현재 검증 결과

각 fold의 inner-selected 예측을 합친 honest outer 결과:

| 모드 | micro F1 | test 구성비 재가중 F1 | paired bootstrap plateau 규칙 대비 90% CI |
|---|---:|---:|---:|
| offline | 0.816737 | 0.768804 | [0.369574, 0.565821] |
| causal | 0.757248 | 0.703250 | [0.312208, 0.499846] |

offline fold별 결과:

| fold | micro F1 | weighted F1 |
|---|---:|---:|
| 2025 Q2 | 0.734298 | 0.733669 |
| 2025 Q3 | 0.836400 | 0.805346 |
| 2025 Q4 | 0.904195 | 0.871116 |

현재 로컬 근거에서는 offline이 causal보다 weighted F1 약 0.06555 높다. 이는 완전한 시계열의 양방향 문맥이 이 문제에 유효하다는 로컬 증거이며 official test 성능을 보장하지 않는다.

전체 OOF에 배포 후처리 값을 다시 맞춘 resubstitution 결과는 offline micro 0.837877/weighted 0.784227, causal micro 0.761137/weighted 0.701584다. 이 값은 outer validation에 재적합했으므로 honest 추정치가 아니며 성능 표의 주 결과로 사용하지 않는다.

## 9. 후처리와 유형 문자열

후처리는 다음만 허용한다.

- outer fold에서 일관되게 확인된 plateau hard override
- spike singleton 보존
- 연속형 이상에 대한 high/low hysteresis
- 동일 segment 안의 짧은 gap 결합

threshold·gap·최소길이는 inner OOF의 유한 grid에서 선택한다. 예측 양성률 4%는 경고만 기록하며 값을 맞추지 않는다.

제출 `anomaly_type`의 canonical 순서는 `spike+noise+flatline+offset+drift`다. `label=1`이지만 어떤 유형 threshold도 넘지 않으면 빈 문자열을 유지할 수 있다. 이 열은 선택 사항이며 순위 점수에는 반영되지 않는다.

## 10. 외부자료 정책

현재 `external.enabled=false`다. 운영진 서면 승인 전 외부 관측값을 다운로드하거나 모델, 피처, 임계값, 검증에 사용하지 않는다.

승인 후에도 후보는 명시적 CC BY 4.0인 I-ORS 2014~2023, DOI `10.22808/DATA-2024-6`으로 제한한다. 라이선스가 불명확한 S-ORS와 2024~2026 중첩 자료는 제외한다.

외부자료는 timestamp 교집합 0, regime mapping, 관측률 95% 이상, median KS 0.05 이하, max KS 0.10 이하, domain classifier AUC 0.65 이하를 모두 통과해야 한다. local-only 대비 weighted F1 +0.005, bootstrap 90% CI 하한 > 0, 세 fold 중 두 fold 이상 개선, 정점별 하락 0.01 이하, 정상 FPR 상대 증가 10% 미만일 때만 전체 모델에 채택한다.

문의 초안은 [EXTERNAL_DATA_APPROVAL_DRAFT.md](EXTERNAL_DATA_APPROVAL_DRAFT.md)에 있으며 자동 발송하지 않는다.

## 11. CLI와 run 기록

공개 인터페이스:

~~~text
python -m p1_qc audit|cv|train|predict|validate|reproduce --config configs/p1.toml
~~~

대표 실행:

~~~powershell
$env:P1_DATA_DIR = "사용자 PC의 P1_qc_anomaly 폴더"
.venv-p1\Scripts\python.exe -m p1_qc audit --config configs\p1.toml
.venv-p1\Scripts\python.exe -m p1_qc cv --config configs\p1.toml --mode offline --backend lightgbm --bootstrap-replicates 2000
.venv-p1\Scripts\python.exe -m p1_qc cv --config configs\p1.toml --mode causal --backend lightgbm --bootstrap-replicates 2000
~~~

각 `run_id`는 ignored `artifacts/runs` 아래에 설정, Git SHA와 dirty 상태, 데이터 SHA-256, split·feature hash, seed, 패키지·GPU 정보, OOF, fold·aggregate 지표, 선택 임계값, 모델과 제출 SHA-256을 기록한다. 캐시는 ZSTD Parquet이며 Git에서 제외한다.

## 12. 테스트와 제출 validator

필수 테스트:

- 시간대, 키, 10분 gap segmentation
- composite type parsing
- causal prefix invariance와 centered segment 경계
- plateau 시작점 backfill
- peer 없음, depth 전체 결측 fallback
- fold-local augmentation과 원본 불변성
- spike singleton 보존
- F1, weighted F1, paired bootstrap
- CPU tree 적합과 CUDA forward/backward
- train/test 전체 audit
- notebook 전 셀 실행
- 저장 모델에서 동일 제출 행 단위 재현

validator는 sample과 동일한 169,011행, 키와 순서, label `{0,1}`, 정상 유형 빈칸, 허용 유형 문자열, UTF-8/no-index를 강제한다. 하나라도 어긋나면 후보 생성을 실패 처리한다.

## 13. 승격과 제출 게이트

첫 후보는 동일 fold의 가장 강한 로컬 기준선보다 weighted F1 +0.01 이상이고 세 outer fold 모두에서 개선되어야 한다. 이후 후보는 현재 최선 대비 +0.005, bootstrap 90% CI 하한 > 0, 세 fold 중 두 fold 이상 비열화 없음, 정점별 하락 0.01 이하를 만족해야 한다.

후보마다 비교 보고서, 재현 명령, 절대경로, 바이트 수와 SHA-256을 사용자에게 제시한다. 사용자가 그 정확한 파일을 승인하기 전에는 업로드하지 않는다.

최종 모델 지정 전에는 다음을 별도로 확인한다.

1. clean environment 재설치와 `pip check`
2. 전체 train 재학습과 test 재추론
3. 저장 모델로 제출 CSV 완전 일치 재현
4. ASCII 상대경로 패키징과 원본 미포함
5. 최종 모델 지정 뒤 예측 업로드 잠금 가능성 경고
6. 사용자의 재확인

## 14. 일정과 패키지

2026-08-13에 확인한 최신 대회 UI는 문제 공개 8월 13일, 제출 마감·최종 모델 9월 7일을 표시한다. 초기 KIMST PDF의 8월 10일~9월 4일 해커톤과 9월 4일 10:00 ZIP 마감은 이전 일정 증거로만 보존한다. 정확한 9월 7일 마감 시각과 ZIP 업로드 관계는 최종 행동 전에 다시 확인한다.

공고의 코드·데이터셋 ZIP 요구와 배포 README의 재배포 금지가 충돌하므로 운영진 답변 전 원본 ZIP/CSV를 패키지에 넣지 않는다. 패키지는 코드, 설정, 고정 의존성, 환경 manifest, 입력 위치 계약, 원본 해시 검사, 모델·피처 설명과 집계 보고서만 allowlist 방식으로 구성한다.

## 15. 현재 다음 작업

1. fold-local augmentation을 동일 nested split에서 완료하고 기본 offline LightGBM과 비교한다.
2. deep/SSL 후보를 사전 고정된 screening과 3-seed 절차로 평가한다.
3. year-transfer와 G-ORS holdout stress 결과를 후보 카드에 결합한다.
4. 최선 설정으로 전체 train 적합, test 추론, validator와 reproduce를 실행한다.
5. 결과를 Git에 코드·문서만 백업하고, 사용자에게 검증된 후보 파일을 제시하되 업로드는 기다린다.
