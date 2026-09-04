# P1·P2·P3 외부 데이터 및 대안 구조 정찰

작성일: 2026-08-17 KST  
상태: 규정·메타데이터 정찰 완료, 외부 관측값 다운로드·학습 사용 0건, 제출 0건

## 결론

현재 공개 공고와 참가자 전용 제출 안내에는 외부 데이터, 공개 사전학습 가중치, 외부자료 공개 의무가 명시되어 있지 않다. 따라서 현 상태는 **허용**이 아니라 **미규정**이다. 공고는 명시되지 않은 사항을 대회규정·관련법령·전문위원회 판단 대상으로 두므로, 운영진 서면 승인 전에는 외부 값을 모델·특징·검증·임계값에 넣지 않는다.

외부 데이터의 기대효과 순위는 다음과 같다.

1. **P2: 2015~2023 S-ORS 동일계절 다층 CTD** — 가장 높은 구조적 가치. 로컬에는 목표 계절의 독립 연도가 사실상 2024년 하나뿐이므로, 성층 붕괴의 연도별 변이를 직접 늘릴 수 있다. 단, ScienceWatch 페이지에 개별 라이선스가 표시되지 않아 KIOST 권리 허락과 대회 운영진 승인이 모두 필요하다.
2. **P3: 2023년 이전 파랑·기상 폭풍 사례** — 높은 가치. 로컬 학습기간이 18개월뿐이고 긴 리드 폭풍 발달·감쇠 사례가 적다. KMA 해양기상부이, ERA5, NOAA WAVEWATCH III를 이용한 사전학습이 후보지만, 평가기간 2025-07-04~2026-06-30 값은 사례시각 복원 위험 때문에 영구 제외한다.
3. **P1: 2023년 이전 정상 수온·염분** — 중간 이하 가치. 이미 정상행이 약 74만 행이고 병목은 합성 offset/drift의 구간 경계이므로, 외부 정상행 자체보다 clean-signal 모델과 주입 역문제 구조가 더 중요하다.

즉, 모든 문제에 외부 데이터를 일괄 투입하지 않는다. **P2 한 계열과 P3 한 계열만 사전등록된 단일 ablation으로 검증**하고 P1은 후순위로 둔다.

## 1. 공식 규정 판독

### 확인된 사실

- KIMST 공고 제2026000111호 23쪽 전체에는 외부 데이터 또는 사전학습 가중치의 허용·금지 조항이 없다.
- 공고는 코드·데이터셋 결과물, 재현성 검증, 부정행위 실격, 실행 가능한 환경을 요구한다.
- 공고에 명시되지 않은 사항은 대회규정·관련법령·전문위원회 개최 등을 통해 판단한다고 적혀 있다.
- 2026-08-12 참가자 전용 제출 안내는 답안 업로드, 하루 1회, 최종 코드·가중치 재현 검증을 설명하지만 외부자료 조항은 없다.
- 2026-08-04 FAQ에도 외부자료 조항은 확인되지 않았다.

### 판정

| 항목 | 현재 판정 | 이유 |
|---|---|---|
| 공개 논문에서 구조·손실함수 참고 | 가능 | 외부 관측값을 모델 입력으로 쓰지 않음 |
| 공개 Python 라이브러리 | 조건부 가능성 높음 | 재현 코드·버전 공개 필요, 공고에 금지 없음 |
| 공개 외부 관측값 학습 | **승인 전 금지** | 규정이 명시적으로 허용하지 않음 |
| 공개 사전학습 가중치 | **승인 전 금지** | 학습 데이터 범위·평가기간 중첩 여부가 불명확 |
| 평가기간 외부 값 대조 | **영구 제외 권고** | 정답 또는 익명 사례시각 복원 가능 |
| 합성 데이터·물리 시뮬레이션 | 승인 전 문의 | 외부 관측값과 별도 항목으로 확인 필요 |

라이선스가 열린 데이터라는 사실과 대회에서 사용할 수 있다는 사실은 서로 다르다. 두 조건을 모두 통과해야 한다.

## 2. 현재 로컬 기준선과 실제 병목

| 문제 | 현재 로컬 1순위 | 로컬 검증 | 핵심 병목 | 외부 데이터의 역할 |
|---|---|---:|---|---|
| P1 | offline XGBoost | micro F1 0.860371, test-share weighted 0.813316 | FN의 92.7%가 offset/drift 계열, 긴 구간 경계 누락 | 정상 계절·수심 regime 확장만 보조 |
| P2 | Extrapolated Soft Gate v2 | target-proxy RMSE 0.768367℃ | 2024→2025 계절 전이 일반화, 독립 계절 표본 부족 | 동일 정점·동일계절 연직 구조를 직접 추가 |
| P3 | Long Persistence Shrink | OOF RMSE 0.780161m | 18·24h 폭풍 발달·감쇠 사례 부족, 절대시간 비공개 | 과거 폭풍 life-cycle 사전학습 |

세 수치는 공식 hidden 점수가 아니다. 서로 다른 문제의 수치를 합산하거나 공식 기준값과 직접 비교하지 않는다.

## 3. 외부 데이터 후보와 누출·권리 판정

### 3.1 KIOST ScienceWatch

#### I-ORS 2014~2023 10분 CTD

- 페이지: https://sciwatch.kiost.ac.kr/handle/2020.kiost/46422
- DOI: 10.22808/DATA-2024-6
- 변수: 수온, 염분, 수심, QC
- 표시 라이선스: CC BY 4.0
- 활용 후보: P1 정상 기준선·합성 이상 배경, P2 구조 사전학습의 보조 도메인
- 한계: P2 배경이 명시하듯 I-ORS와 S-ORS는 서로 다른 수괴에 속하므로 S-ORS 대체 자료가 아니다.

#### I-ORS 2004~2023 1시간 해양·기상

- 페이지: https://sciwatch.kiost.ac.kr/handle/2020.kiost/46771
- DOI: 10.22808/DATA-2025-2
- 변수: 5·21·38m 수온, 기온·풍향·풍속·기압·습도·강수
- 표시 라이선스: CC BY 4.0
- 활용 후보: P1 계절 정상 prototype, P3 기상 encoder의 같은 정점 사전학습
- 한계: 파고 변수가 없어 P3 목표를 직접 학습하지 못하며 1시간 평균이라 대회 10/20분 해상도와 다르다.

#### S-ORS 2015~2023 10분 CTD

- 페이지: https://sciwatch.kiost.ac.kr/handle/2020.kiost/46423
- DOI: 10.22808/DATA-2024-7
- 변수: 수온, 염분, 수심, QC
- 활용 후보: **P2 최우선**, P1 정상 배경
- 권리 상태: 페이지에 개별 개방 라이선스가 표시되지 않는다. 저장소 일반 문구와 KORS 포털은 권리가 제한된 것으로 보이므로 별도 허락 전 NO-GO다.

#### S-ORS 2018년 4월 방향 스펙트럼

- 페이지: https://sciwatch.kiost.ac.kr/handle/2020.kiost/46806
- DOI: 10.22808/DATA-2025-3
- 활용 후보: P3 파랑 스펙트럼 표현 연구
- 한계: 한 달뿐이고 목표 `hs`의 장기 폭풍 전이를 늘리기에는 좁다. 개별 라이선스 확인도 필요하다.

### 3.2 KORS 포털

- 다운로드: https://kors.kiost.ac.kr/data/accessData
- 메타데이터: https://kors.kiost.ac.kr/data/metaData
- 포털은 파고·파주기·파향, 풍속·기압, CTD 장비가 세 기지에 존재함을 보여준다.
- 그러나 비회원 다운로드는 자료관리자 코드가 필요하고 포털 하단은 `ALL RIGHT RESERVED`다.
- 2024~2026 자료와 실시간 차트는 대회 train/test 또는 P2 가림 구간·P3 익명 평가기간과 겹친다.

판정: **메타데이터만 참고**, 값은 사용하지 않는다. 특히 2025년 9~10월 S-ORS 중간층은 P2 정답 자체이며, 2024~2025 P1 수온은 합성 이상 주입 전 원신호일 가능성이 높다.

### 3.3 KMA 해양기상부이·파고부이

- 공공데이터포털: https://www.data.go.kr/data/15139440/openapi.do
- 보유기간: 해양기상부이 1996년 7월 이후, 파고부이 2009년 이후(지점별 상이)
- 변수: 풍향·풍속·기압·기온·습도·파고·파주기·파향·수온 등
- 라이선스: 공공저작물 출처표시 제1유형

활용 후보: P3의 48시간 context→6개 lead 직접 다중지평 사전학습. 2023-12-31 이전 storm case만 추출하고, KMA 지점 ID를 대회 station ID와 절대 결합하지 않는다.

한계: 부이와 고정 해양과학기지의 센서·해역·파주기 정의가 다를 수 있다. 도메인 분류 AUC와 local-only 대비 증분 검증이 필수다.

### 3.4 ERA5와 WAVEWATCH III

- ERA5: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview
- NOAA WAVEWATCH III: https://polar.ncep.noaa.gov/waves/download.shtml
- NOAA hindcast archive: https://polar.ncep.noaa.gov/waves/hindcasts/index.php.shutdown.29Aug2023

ERA5는 시간별 기상·해양파 변수를 제공하고 CC-BY 라이선스를 표시한다. NOAA는 장기 WAVEWATCH III hindcast와 GRIB 자료 접근을 제공한다.

활용 후보: P3의 폭풍 regime 사전학습, `wind→wave energy growth/decay` 보조목표, 지역 부이 관측의 결측 보완.

제한:

- 2025-07-04~2026-06-30 값은 사용하지 않는다.
- 익명 case의 context와 reanalysis를 대조해 절대시각을 찾지 않는다.
- 2023-12-31 이전 또는 운영진이 명시한 cutoff 이전 자료만 독립 pretraining corpus로 사용한다.
- coarse grid model은 해양과학기지 국지 파랑과 편향이 있으므로 원값을 test prediction에 직접 더하지 않는다.

### 3.5 Argo·Copernicus Marine

- Argo GDAC: https://argo.ucsd.edu/data/data-from-gdacs/
- Copernicus Global Ocean Physics Reanalysis: https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description

활용 후보: P2의 depth-query profile encoder에 다양한 수온·염분 곡률과 혼합층 형태를 사전학습.

한계: Argo는 대양 심층 profile 중심이고 얕은 황해 고정계류의 4~50m, 10분 내부파·조석 구조와 도메인이 크게 다르다. Copernicus는 8km·일평균·50개 표준층이라 P2의 10분 전이를 직접 복원하지 못한다. S-ORS historical이 허용되지 않을 때만 저우선 challenger다.

## 4. 대안 구조: 추가 미세튜닝이 아닌 문제 재정식화

### P1 — corruption inversion + duration-aware posterior

핵심 사실은 정상 `good_data`에 다섯 이상을 합성 주입했다는 점이다. 다음 구조는 raw anomaly classifier보다 과제 생성과정을 직접 모사한다.

1. 정상 시계열 모델이 `temp_clean`의 양방향 posterior를 추정한다.
2. spike/noise/flatline/offset/drift 변환별 likelihood를 계산한다.
3. 5종 duration prior와 중첩 가능한 semi-Markov decoder가 row posterior를 구한다.
4. XGBoost와 plateau rule은 관측 likelihood expert로 사용하고 최종 label만 decoder가 결정한다.

외부 정상자료는 1번 clean model에만 사용하며 anomaly label로 취급하지 않는다. 기대효과는 offset/drift 내부가 장기 rolling baseline에 흡수되는 문제를 완화하는 것이다. P1은 이미 정상 데이터가 많으므로 외부자료보다 이 재정식화의 효과를 먼저 검증한다.

### P2 — depth-query neural field + exact mask curriculum

고정 3개 target output을 직접 회귀하지 않고 `T(z,t)`를 연속 수심 함수로 모델링한다.

- 입력: 공개층 수온·염분·수심, 계절·반일주조 phase, 성층/혼합 state
- query: 목표 수심 `z`
- 출력: 수온 평균과 불확실도, 수심 1·2차 도함수
- 학습: 과거 전층에서 임의의 연속 내부층 3개와 2개월 block을 함께 가리는 curriculum
- 물리 loss: 공개 layer 1·5 endpoint 일치, 과도한 곡률 억제, 혼합 state에서 수직 균질성, 성층 state에서 비단조 hard constraint 대신 soft curvature prior

2015~2023 S-ORS가 허용되면 연도별 9~10월을 leave-one-year-out으로 검증한다. 이것이 P2에서 가장 큰 기대효과를 가진다.

### P3 — storm-life-cycle pretraining + energy residual head

P3는 단순한 더 큰 Transformer보다 폭풍 발달·감쇠 사례 수가 병목이다.

1. KMA/ERA5/WW3의 2023년 이전 자료에서 `hs≥1.5m`, 48시간 context, 24시간 target 사례를 대량 생성한다.
2. station-agnostic encoder를 `Δ(hs²)`와 6개 lead quantile을 함께 예측하도록 사전학습한다.
3. 풍향·파향 정렬, 풍속², 기압 하강, gust, tp를 사용해 wind input과 swell persistence를 분리한다.
4. 대회 train으로 local sensor calibration head만 fine-tune한다.
5. 현재 persistence-shrink를 residual anchor로 유지해 외부 모델이 확신이 낮을 때 exact fallback한다.

공개 foundation model은 별도 arm이다. MOMENT는 imputation/anomaly까지, Chronos는 zero-shot forecasting을 지원하지만 학습자료 provenance가 대회 평가기간과 완전히 분리됐는지 확인하기 어렵다. 운영진 승인 전 checkpoint는 사용하지 않고, 허용돼도 `frozen zero-shot→local fine-tune` 두 단계와 local-only comparator를 분리한다.

## 5. 채택 게이트

### 공통 법·누출 게이트

1. 운영진 서면 허용 범위 수신
2. 원자료별 라이선스·DOI·버전 확인
3. 허용 cutoff 이후 timestamp 0건
4. P2 2025-09~10 S-ORS, P3 평가기간, P1 2024~2026 원신호 교집합 0건
5. 원자료는 Git·최종 공개 패키지에서 제외, 해시·가공코드·출처만 기록

### 데이터 품질 게이트

- 변수 정의와 단위 일치
- 관측률·QC·센서 deployment mapping 기록
- station/source를 맞히는 domain classifier AUC가 높으면 shared model이 아니라 source-specific pretraining만 허용
- 외부 데이터 제거 ablation을 반드시 함께 실행

### 성능 게이트

| 문제 | 1차 승격 최소조건 |
|---|---|
| P1 | weighted F1 +0.005 이상, bootstrap 90% CI 하한 >0, normal FP/day +10% 미만 |
| P2 | 사전 고정 year/block 전체 RMSE -0.010℃ 이상, 3개 층 모두 비열화 없음, transition block 개선 |
| P3 | case bootstrap RMSE -0.010m 이상, 세 정점·6개 lead 중 최악 하락 제한, 18·24h 개선이 short lead를 훼손하지 않음 |

현재 로컬 검증은 반복 노출되어 완전한 virgin holdout이 아니다. 외부 실험은 최대 두 family만 열고, 공식 제출은 incumbent와 승인된 최고 외부 후보의 사전 정한 비교로 사용한다.

## 6. 실행 우선순위

### Gate 0 — 운영진 회신 전

- 외부 관측값 다운로드·학습 0건 유지
- 세 문제 통합 문의문 발송 준비
- 후보별 metadata/licence manifest만 작성
- 외부자료 없이 P1 corruption inversion의 작은 합성 검증 설계

### Gate 1 — `과거 공개자료 허용` 회신 시

1. P2 S-ORS 2015~2023의 KIOST 권리 허락 확인
2. 허락되면 P2 depth-query neural field 단일 experiment
3. P3 KMA buoy + ERA5/WW3 ≤2023 storm pretraining 단일 experiment
4. P1 external은 위 둘이 성과를 보인 뒤에만 검토

### Gate 2 — `외부자료 불허` 회신 시

- P1 corruption inversion/semi-Markov
- P2 local 2024 exact-mask curriculum + continuous depth model
- P3 local train storm event augmentation + energy residual head

## 7. 중단선

- 운영진 회신이 모호하면 미승인으로 처리한다.
- S-ORS 개별 라이선스가 확인되지 않으면 P2 최우선 외부 경로는 열지 않는다.
- 평가기간이나 hidden target을 복원할 수 있는 값은 운영진이 일반적으로 외부자료를 허용해도 사용하지 않는다.
- 공개 pretrained checkpoint의 학습 데이터 provenance가 불명확하면 구조만 재구현하고 weight는 사용하지 않는다.
- 작은 평균 개선을 위해 세 문제 전체에 복잡한 외부 파이프라인을 확장하지 않는다.

## 8. 1차 출처

- KIMST 공식 공고: https://www.kimst.re.kr/u/news/notice_01/board.do?bno=153421765145711&searchDiv=&searchKeyword=&type=view
- 대회 공식 사이트: https://oceanaidata.org/
- I-ORS 2014~2023 CTD: https://sciwatch.kiost.ac.kr/handle/2020.kiost/46422
- I-ORS 2004~2023 ocean/atmosphere: https://sciwatch.kiost.ac.kr/handle/2020.kiost/46771
- S-ORS 2015~2023 CTD: https://sciwatch.kiost.ac.kr/handle/2020.kiost/46423
- S-ORS wave spectra: https://sciwatch.kiost.ac.kr/handle/2020.kiost/46806
- KMA ocean buoy API: https://www.data.go.kr/data/15139440/openapi.do
- ERA5: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview
- NOAA WAVEWATCH III: https://polar.ncep.noaa.gov/waves/download.shtml
- Argo GDAC: https://argo.ucsd.edu/data/data-from-gdacs/
- Copernicus ocean reanalysis: https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description
- MOMENT: https://proceedings.mlr.press/v235/goswami24a.html
- Chronos: https://www.amazon.science/publications/chronos-learning-the-language-of-time-series

