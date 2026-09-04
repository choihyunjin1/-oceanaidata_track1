# P1 최고점 돌파 연구정찰 — 2026-08-13 KST

상태: 방법론 연구 및 실험 우선순위 제안. 외부 관측값 다운로드·사용 없음. 코드·설정 변경 없음. 대회 제출 없음.

## 1. 결론 먼저

현재 최선은 양방향 offline XGBoost다. 동일한 7일 purge rolling-origin 3개 outer holdout에서 micro F1 0.860371, test 지지율 재가중 F1 0.813316을 기록했다. 이는 공식 test 점수가 아니라 train OOF 로컬 추정치다.

다음 돌파구는 범용 대형 신경망 교체보다 아래 순서가 유력하다.

1. offset·drift 유형별 XGBoost 전문가와 binary 본모델의 inner-OOF 결합
2. 장기 이벤트용 변화점 특징과 구간 proposal → segment classifier
3. 단순 동시 peer mean을 대체하는 계절·성층 gate가 있는 동적 층간 관계 잔차
4. 위 세 결과가 생긴 뒤에만 저비율 경계형 합성, repair SSL, 조건부 ensemble 검증

근거는 명확하다.

- XGBoost의 flatline recall은 0.9997, noise recall은 0.9355다. 이 둘에 모델 용량을 더 쓰는 기대효용은 작다.
- offset recall은 0.6492, drift recall은 0.6462다. 특히 I-ORS layer 1 offset 0.0809, S-ORS layer 2 drift 0.1246, G-ORS layer 1 drift 0.0415다.
- 48시간 이상 true event가 XGBoost FN 3,109행 중 2,393행, 약 77.0%를 차지한다. 장기 이벤트 row recall은 0.6315다.
- 여섯 모델이 모두 놓친 1,354개 양성행 중 offset 605, drift 571, noise+drift 84, drift+offset 45행이다. 모델 다수결만으로는 이 영역을 복원할 수 없다.
- 기존 전역 합성 증강은 weighted F1을 0.768804에서 0.553548로 떨어뜨렸다. recall 증가는 있었지만 precision이 0.4815까지 붕괴했다. 합성은 저비율·유형 제한·hard-negative 보호가 필수다.
- full sequence 실험에서 TCN은 offset 0.7858, drift 0.6966 recall로 XGBoost보다 높았지만 FP 5,789행 때문에 micro F1 0.7676에 그쳤다. deep score는 대체 모델이 아니라 조건부 전문가·구간 특징으로 쓸 가치가 더 크다.

따라서 바로 실행할 1차 묶음은 실험 E1, E2, E3, E4다. 예상 이득 수치는 아래에 모두 “계획용 사전 추정”으로 표시했으며 측정값이나 보장이 아니다.

## 2. 증거 수준과 사용 경계

이 보고서는 문장을 다음처럼 구분한다.

- **로컬 사실**: 현재 저장된 OOF·manifest·공식 배포 파일을 재계산하거나 직접 읽어 확인한 값
- **문헌 사실**: 논문 원문, 공식 proceedings, DOI 또는 저자 공식 저장소가 명시한 내용
- **우리 추론**: 문헌의 방법을 P1의 실패군에 적용해 세운 가설. 대회 성능이 보장되지 않는다.

외부 데이터 정책은 고정한다.

- 외부 해양 관측값, 공개 실시간값, 2023년 이전 관측값을 포함한 어떤 외부 값도 이 정찰에서 내려받거나 열지 않았다.
- 조사한 것은 논문 본문·초록·공식 코드 설명 등 방법론뿐이다.
- 이후 실험도 대회 배포 train/test와 fold 내부에서 만든 합성값만 사용한다. 외부 관측값 사용은 운영진 답변과 사용자 승인 전까지 금지한다.
- 논문 pretrained checkpoint도 학습 데이터 출처가 불명확하거나 외부 시계열 값을 내포할 수 있으므로 기본적으로 금지한다. 구조와 loss만 로컬에서 재구현한다.

## 3. 현재 기준선과 공식성 구분

### 3.1 로컬 모델 현황

| 후보 | 모드 | Micro F1 | Test-share weighted F1 | 판단 |
|---|---|---:|---:|---|
| XGBoost | offline | **0.860371** | **0.813316** | 현재 최선 |
| LightGBM | offline | 0.816737 | 0.768804 | 대체됨 |
| CatBoost | offline | 0.806831 | 0.757848 | 다양성 후보 |
| Patch Transformer + fold-local SSL | offline | 0.799755 | 0.759598 | 대체 실패, 보조 score 후보 |
| TCN + fold-local SSL | offline | 0.767582 | 0.740445 | 대체 실패, recall 전문가 후보 |
| LightGBM | causal | 0.757248 | 0.703250 | 운영 비교용 ablation |
| LightGBM + 전역 합성 | offline | 0.609332 | 0.553548 | 탈락 |

위 값은 동일한 purged rolling-origin outer OOF 진단이다. 반복적인 연구 진단으로 outer 라벨을 이미 여러 번 보았으므로, 새 아이디어 선택에는 outer 결과를 다시 사용하지 않는다. 모든 설정·임계값·결합 가중치는 과거 방향 inner split에서만 선택하고, outer 결과는 사전 동결된 실험의 진단값으로만 기록한다.

로컬 사실의 재현 출처는 다음과 같다.

- XGBoost metrics: artifacts/runs/20260813T153038+0900_cv_378a4e89/metrics.json, SHA-256 de7d92153df58b10177ac6c79a0733164c9c4630fa2c317495ddd8a1eb487e36
- XGBoost OOF: 같은 run의 oof.parquet, SHA-256 d1b9439db6d0d906fa080bd01f1eb8fc21d051c3d056a274e2b02e43c1e55f4a
- 실패 재정찰: artifacts/failure_recon_20260813/research_only_failure_summary.json, SHA-256 bd3acdfbc4542ffc8fed8d6cf3ec807e314b339a2bdcafa4b1680da148f8d44e
- full sequence 결과: artifacts/sequence_full_20260813/sequence_experiment.json, SHA-256 d9462dffc6db59afb5676f823ec1a9b9b35ce2a83a537fefff82b4a09e217f50

### 3.2 주최 제공 scorer와 자체 validator의 차이

- **로컬 사실**: 배포 score.py는 주최 측이 제공한 로컬 scorer이며 SHA-256은 b5a4646717377c4894aa6a8d02a1f894ced395c757e8708f84a56a3bf7d75bca다.
- score.py는 station, year, layer, time key로 정답과 제출을 one-to-one merge한 뒤 2TP / (2TP + FP + FN)을 계산한다. 따라서 키 집합이 같다면 행 순서 변경 자체는 허용한다.
- hidden answer.csv가 배포되지 않았으므로 test 공식 F1은 로컬에서 계산할 수 없다.
- 자체 validator는 sample/test와 동일한 행·키·순서, UTF-8, label, canonical anomaly_type, 파일 SHA 등 재현·구조 조건을 더 엄격하게 검사한다. 이는 점수 인증기가 아니다.
- XGBoost 0.860370838은 공식 scorer와 같은 이진 row-level F1 식을 train rolling-origin OOF에 적용한 로컬 추정치다. 공식 점수나 hidden test 성능이 아니다.
- 공식 기준값 0.548255도 hidden 분할에서 얻은 값이므로 로컬 OOF 0.860371과 직접 비교하지 않는다.

## 4. 실패 구조를 이상 유형에 직접 매핑

아래 수치는 outer 라벨을 사용한 **연구 전용 사후 진단**이다. 가설 생성에는 쓰되 threshold, model selection, promotion 또는 제출 결정에는 쓰지 않는다.

| 유형 | 현재 강점/실패 | 직접 맞는 방법 | 유지해야 할 안전장치 |
|---|---|---|---|
| spike | XGB recall 0.8197; 59개 singleton event 중 11개 miss. TCN 0.5902, Transformer 0.2295로 더 약함 | 양쪽 차분, 3점 복귀도, local MAD, 별도 singleton 전문가 | 어떤 후처리도 singleton 제거 금지. deep smoothing으로 대체 금지 |
| noise | XGB recall 0.9355. 대체로 해결됐지만 G-ORS L1 0.6970, I-ORS L2 0.6429 | multi-scale diff MAD/분산 비, repair residual의 derivative 성분 | 정상 내부파를 잡지 않도록 계절·peer gate와 hard negative 필요 |
| flatline | XGB recall 0.9997; plateau length ≥6 hard rule이 거의 완전 | exact/epsilon plateau 전체 길이와 backfill | 규칙을 frozen override로 유지. 학습 모델이 덮어쓰지 않게 함 |
| offset | XGB recall 0.6492; I-L1 0.0809, S-L6 0.2402. 장기 내부가 정상 baseline에 흡수됨 | pre/post level-shift, change-point boundary, peer/common-mode 제거, segment classification | centered rolling이 이벤트를 기준선으로 학습하는 self-normalization 방지 |
| drift | XGB recall 0.6462; G-L1 0.0415, S-L2 0.1246. 이벤트 초반 decile recall 0.262 | robust slope bank, piecewise-linear CP, CUSUM, peer-detrended slope, onset/offset head | 문헌의 “distribution drift”와 주입된 선형 sensor drift를 혼동하지 않음 |
| 중첩 | composite event row recall 0.6755 대 single 0.8341. noise+drift와 drift+offset unanimous miss가 큼 | multi-label type experts, event proposal의 union, segment-level composite features | anomaly_type 정확 일치가 아니라 중복 제거 membership 사용 |

### 4.1 그룹·길이별 핵심 실패

| 진단 | 값 | 해석 |
|---|---:|---|
| I-ORS layer 1 F1 | 0.4328 | offset·drift와 FP가 모두 집중된 최우선 그룹 |
| S-ORS layer 2 F1 | 0.5064 | drift recall 0.1246, 구간형 복원이 필요 |
| I-ORS layer 5 F1 | 0.6955 | FP는 0이나 recall 0.5332; 보수적 장기 전문가 후보 |
| 48시간 이상 event row recall | 0.6315 | 6,494 양성행 중 2,393 FN |
| 48시간 이상 event hit rate | 14/17 | 시작은 찾더라도 전체 구간을 덜 채우는 문제가 큼 |
| 24~48시간 event row recall | 0.9404 | 중간 길이는 이미 강함 |
| XGB FP run | 88개, 1,093행 | median 3행이나 24~48시간 FP run 2개가 388행을 차지 |

**우리 추론**: 현재 병목은 “이벤트가 존재하는지”만이 아니라 “발견한 장기 이벤트의 경계를 얼마나 온전히 채우는지”다. 그래서 row classifier의 threshold만 낮추는 것보다 변화점으로 후보 구간을 만들고 구간 전체를 분류하는 편이 공식 row F1에 더 직접적이다.

## 5. 2023~2026 1차 출처 연구 결과

### 5.1 평가와 모델 선택: 복잡도가 승리를 보장하지 않는다

- **문헌 사실**: Liu와 Paparrizos의 TSB-AD는 40개 데이터셋의 1,070개 시계열과 40개 detector를 통합 평가했고, 일부 단순 통계 방법이 복잡한 모델보다 강하다는 결과와 metric·dataset bias를 보고했다. NeurIPS 2024, DOI 10.52202/079017-3437. 원문: https://papers.nips.cc/paper_files/paper/2024/hash/c3f3c690b7a99fba16d0efd35cb83b2c-Abstract-Datasets_and_Benchmarks_Track.html
- **문헌 사실**: mTSBench는 344개 labeled MTS, 19개 데이터셋, 24개 detector를 비교했고 단일 detector가 모든 데이터셋을 지배하지 않으며 model selection도 아직 최적과 큰 차이가 있다고 보고했다. TMLR, 2026-02. 원문: https://openreview.net/forum?id=8LfB8HD1WU
- **문헌 사실**: Barrish와 van Vuuren은 부정확한 라벨, 비현실적 anomaly density, 실무 행동과 맞지 않는 metric이 TSAD 연구 결론을 왜곡할 수 있다고 정리했다. TMLR, 2026-05-24. 원문: https://openreview.net/forum?id=RyMLAr5tFU
- **문헌 사실**: AutoTSAD는 서로 다른 강점의 detector·parameterization을 선택·ensemble하며, 대표 regime 추출과 정제 뒤 여러 합성 anomaly를 주입한다. PVLDB 17(11), 2024, DOI 10.14778/3681954.3681978. 원문: https://www.vldb.org/pvldb/vol17/p2987-schmidl.pdf

**우리 추론**: 이미 XGBoost가 deep보다 강한 P1에서는 새 architecture 하나로 전면 교체하기보다 type/segment/peer라는 서로 다른 view를 만들고 동일 inner protocol에서 증분만 측정해야 한다.

### 5.2 multivariate·peer-aware: 정적 평균보다 관계 변화가 신호다

- **문헌 사실**: SARAD는 단순 reconstruction error 밖의 pairwise inter-feature association을 학습하고, association이 시간에 따라 바뀌는 것을 subseries division으로 다룬다. NeurIPS 2024, DOI 10.52202/079017-1533. 원문: https://papers.nips.cc/paper_files/paper/2024/hash/56ad264ac7448239145606cf4106042f-Abstract-Conference.html
- **문헌 사실**: CAROTS는 multivariate 관계를 보존하는 augmentation과 깨뜨리는 augmentation을 구분해 one-class contrastive representation을 학습한다. ICML 2025, PMLR 267. 원문: https://proceedings.mlr.press/v267/kim25aa.html
- **문헌 사실**: SPAGD는 reconstruction residual에 따라 inter-variable graph를 동적으로 조정하고 self-perturbed auxiliary samples로 classifier를 학습한다. NeurIPS 2025 spotlight, 공식 poster 2025-12-03. 원문: https://nips.cc/virtual/2025/poster/116664
- **문헌 사실**: M²AD는 여러 sensor·system의 이질성을 고려해 residual을 global anomaly score로 모으고 Gaussian-mixture/Gamma calibration을 사용한다. AISTATS 2025, PMLR 258, 2025-05-03~05. 원문: https://proceedings.mlr.press/v258/alnegheimish25a.html

**우리 추론**: P1의 layer는 고정 수심이 아니고 성층·내부파로 정상 상관이 사라질 수 있다. 따라서 모든 층의 순간 평균을 쓰는 현재 방식에 계절, depth regime, 최근 정상 상관, peer spread gate를 추가해야 한다. G-ORS에는 peer가 없으므로 단변량 fallback을 유지한다.

### 5.3 offset·drift: 변화점과 배경 변화 분리가 직접적이다

- **문헌 사실**: changeforest는 classifier likelihood ratio로 비모수 multivariate multiple change point를 찾는다. JMLR 24(216), 2023. 원문: https://www.jmlr.org/papers/v24/22-0512.html
- **문헌 사실**: Wu 등은 유사한 분포의 segment를 묶고 MDL 기반 greedy search로 change point 수를 알지 못해도 multivariate change point를 찾는 방법을 제시했다. AISTATS 2024, PMLR 238. 원문: https://proceedings.mlr.press/v238/wu24g.html
- **문헌 사실**: RECURVE는 representation trajectory curvature를 사용해 급격한 변화와 점진적 변화를 모두 다루는 boundary detector다. NeurIPS 2024, DOI 10.52202/079017-0194. 원문: https://papers.nips.cc/paper_files/paper/2024/hash/0b7f639ef28a9035a71f7e0c04c1d681-Abstract-Conference.html
- **문헌 사실**: FOCuS는 많은 window·change size를 효율적으로 포괄하는 online CUSUM 계열 방법을 제시한다. JMLR 24(81), 2023. 원문: https://www.jmlr.org/papers/v24/21-1230.html
- **문헌 사실**: D3R은 비정상적인 환경 분포 변화가 정상점의 false alarm을 만들 수 있어 dynamic decomposition 뒤 reconstruction하는 방식을 제안한다. NeurIPS 2023, DOI 10.52202/075280-0473. 원문: https://papers.nips.cc/paper_files/paper/2023/hash/22f5d8e689d2a011cd8ead552ed59052-Abstract-Conference.html

주의: D3R의 drift는 배경 distribution drift다. 대회가 주입한 시간에 따라 선형 누적되는 sensor drift와 동일 개념이 아니다. P1에서는 D3R을 “정상 계절·성층 변화 제거” 아이디어로만 사용한다.

**우리 추론**: 변화점 score는 offset/drift의 시작·끝에는 강하지만 이벤트 내부 모든 행을 직접 표시하지 못한다. 그래서 change point는 detector가 아니라 candidate boundary generator로 쓰고, 두 boundary 사이를 segment classifier가 채워야 row F1에 맞는다.

### 5.4 masked reconstruction·repair·prototype

- **문헌 사실**: SimMTM은 여러 masked view가 보존한 상보적 temporal variation을 모아 reconstruction하는 masked pretraining을 제안했다. NeurIPS 2023. 원문: https://papers.nips.cc/paper_files/paper/2023/hash/5f9bfdfe3685e4ccdbc0e7fb29cccf2a-Abstract-Conference.html
- **문헌 사실**: PatchTST는 시계열 patch token으로 local semantic을 보존하고 같은 look-back에서 attention memory를 줄여 더 긴 문맥을 볼 수 있으며 masked pretraining을 실험했다. ICLR 2023. 원문: https://openreview.net/forum?id=Jbdc0vTOcol
- **문헌 사실**: MEMTO는 reconstruction over-generalization을 줄이기 위해 normal memory와 input·latent 양쪽 deviation score를 사용한다. NeurIPS 2023. 원문: https://papers.nips.cc/paper_files/paper/2023/hash/b4c898eb1fb556b8d871fbe9ead92256-Abstract-Conference.html
- **문헌 사실**: H-PAD는 point prototype만으로 장기 interval anomaly를 놓치는 문제를 겨냥해 여러 크기의 patch prototype과 period prototype을 결합한다. ICLR 2025. 원문: https://openreview.net/forum?id=8TBGdH3t6a
- **문헌 사실**: IGAD는 reconstruction model이 anomaly도 잘 복원하는 over-generalization을 줄이기 위한 idempotent generation module을 제안하며 추가 trainable parameter 없이 결합할 수 있다고 설명한다. NeurIPS 2025. 원문: https://papers.nips.cc/paper_files/paper/2025/hash/ebba26f4dbcfe7318e4a263c54e9cf42-Abstract-Conference.html
- **문헌 사실**: JuRe는 corruption을 복구하도록 학습한 작은 depthwise-separable residual block과 amplitude·difference·trend·correlation 구조 score를 제안한다. 2026-04-19 arXiv preprint이며 동료심사 전이다. 원문: https://arxiv.org/abs/2604.17388, 저자 코드: https://github.com/iis-esslingen/JuRe

**우리 추론**: P1 full SSL+deep 실험이 XGBoost를 이기지 못했으므로 SSL representation 전체를 다시 학습하는 것보다 repair discrepancy, prototype distance 같은 소수의 orthogonal score를 XGBoost/segment classifier에 넣는 저비용 경로가 우선이다.

### 5.5 weak·synthetic anomaly와 class imbalance

- **문헌 사실**: DADA는 normal decoder와 synthetic-anomaly decoder를 구분하고 mask reconstruction과 adaptive bottleneck을 결합한다. ICLR 2025. 원문: https://proceedings.iclr.cc/paper_files/paper/2025/hash/ca7998666c2e53cc1e882b7268414d8a-Abstract-Conference.html
- **문헌 사실**: CAROTS는 관계 보존 perturbation을 정상 positive, 관계 파괴 perturbation을 synthetic negative로 분리한다. ICML 2025. 원문: https://proceedings.mlr.press/v267/kim25aa.html
- **문헌 사실**: SPAGD는 random perturbation 대신 normal reconstruction 과정의 self-perturbation을 auxiliary anomaly signal로 사용한다. NeurIPS 2025. 원문: https://nips.cc/virtual/2025/poster/116664
- **문헌 사실**: AutoTSAD는 하나의 전역 주입이 아니라 여러 representative regime을 정제한 다음 다양한 anomaly configuration을 생성해 algorithm selection에 사용한다. PVLDB 2024, DOI 10.14778/3681954.3681978.

**우리 추론**: 기존 P1 합성 실패의 원인은 “합성 자체가 무효”라기보다 높은 주입량, 실제 정상 경계와 다른 쉬운 synthetic, 모든 유형을 한 binary loss에 동일하게 넣은 점일 가능성이 크다. 그러나 이 추론은 inner split의 저비율 ablation으로만 검증해야 한다.

### 5.6 event proposal·경계·구간 분류

- **문헌 사실**: RECURVE는 점진적·급격한 class boundary를 함께 찾는 시계열 boundary detector다. NeurIPS 2024, DOI 10.52202/079017-0194.
- **문헌 사실**: changeforest와 Wu 등의 방법은 row마다 독립 label을 내는 대신 distribution이 달라지는 segment boundary를 찾는다. 각각 JMLR 2023, AISTATS 2024.
- **문헌 사실**: boundary onset/offset/presence와 proposal을 함께 최적화하는 방식은 sound event detection에서도 제안됐지만, 2026-01-07 arXiv preprint이고 도메인이 다르다. 원문: https://arxiv.org/abs/2601.04178

**우리 추론**: cross-domain preprint는 직접 근거가 아니라 설계 힌트일 뿐이다. P1의 event proposal은 change-point 1차 출처와 공식 duration 범위를 기반으로 구현하고, onset/offset head는 선택적 ablation으로만 둔다.

### 5.7 calibration과 ensemble

- **문헌 사실**: AutoTSAD는 detector score를 순위화하고 상위의 서로 다른 scoring을 aggregate한다. PVLDB 2024.
- **문헌 사실**: M²AD는 sensor/system별 residual 이질성을 global score로 모으기 위해 mixture와 Gamma calibration을 사용한다. AISTATS 2025.
- **문헌 사실**: Gibbs와 Candès는 arbitrary distribution shift 아래 online conformal procedure를 제안했다. JMLR 25(162), 2024. 원문: https://www.jmlr.org/papers/v25/22-1218.html
- **문헌 사실**: split conformal이 비교환 과정에도 penalty를 포함한 이론적 조건 아래 적용될 수 있다는 결과가 있다. JMLR 25(225), 2024. 원문: https://www.jmlr.org/papers/v25/23-1553.html

**우리 추론**: conformal coverage는 공식 F1 최적화를 보장하지 않는다. P1에서는 alarm-rate 진단이나 regime score normalization에만 쓰고, 최종 임계값은 inner OOF의 공식 F1로 선택한다. 현재 six-model union F1이 0.5608인 점을 보면 무조건 OR/평균은 금지하고, 소수 view의 비음수 결합과 조건부 gate만 검증해야 한다.

## 6. 우선순위 실험 10개

### 공통 검증 계약

모든 실험에 아래를 고정한다.

- 외부 관측값·외부 pretrained weight 0개
- 10분 gap segment 경계를 rolling, change point, proposal, augmentation, SSL window가 절대 넘지 않음
- normalization, prototype, graph, synthesis parameter, calibration은 fold train에서만 적합
- hyperparameter와 threshold는 각 outer train 내부의 past-only inner split에서만 선택
- plateau ≥6 full-run hard override와 spike singleton 보존은 frozen
- 주지표는 test-share weighted F1, 보조로 micro F1, precision/recall, type recall, event recall, 정상 1일당 FP, 최악 station-layer를 기록
- 예상 이득은 현재 XGBoost 대비 weighted F1 절대 변화에 대한 **계획용 사전 추정**이다. 측정값이 아니다.
- 1차 screen은 inner split만 사용한다. 동일 아이디어의 자유 탐색은 최대 12개 설정으로 제한한다.
- outer holdout은 이미 연구 진단에 노출됐으므로 새 실험의 완전히 독립적인 증명으로 부르지 않는다. inner-only 선택을 재현하고 outer는 고정 진단으로 한 번 실행한다.

### E1. 유형별 tree 전문가 + binary 합성기 — 우선순위 1

- **가설**: spike/noise/flatline/offset/drift membership별 XGBoost가 서로 다른 loss surface와 threshold를 학습하면 offset·drift의 희박한 신호가 binary 다수 유형에 묻히는 현상을 줄인다.
- **설계**: binary XGB는 그대로 두고 다섯 one-vs-rest 전문가를 fold train에서 학습한다. 결합기는 inner OOF의 여섯 logit과 plateau/spike rule만 입력받는 비음수 logistic 또는 얕은 monotonic tree로 제한한다. 유형 string은 예측 목적이 아니라 binary score 분해에만 쓴다.
- **직접 목표**: offset/drift, composite event, I-L1, S-L2, G-L1.
- **누출 안전성**: type membership, class weight, threshold, 결합기 모두 inner train/validation에서만 산정. outer anomaly_type는 진단 외 사용 금지.
- **예상 이득**: +0.008~+0.025.
- **비용**: CPU 2~4시간, GPU 불필요. 메모리 약 8~16GB 추가.
- **중단 기준**: 3개 inner split 중 2개에서 +0.003 미만, 전체 precision 0.01 이상 하락, 또는 offset/drift recall 증가가 FP/day 10% 초과 증가를 동반하면 중단.

### E2. leave-center-out 변화점 특징 bank — 우선순위 2

- **가설**: centered rolling median은 긴 offset/drift 내부를 새 정상으로 흡수한다. 현재점 주변의 guard band를 비우고 좌·우 배경을 따로 비교하면 시작·끝 신호를 보존한다.
- **설계**: 6/12/24/48/72시간·7일 scale에서 좌우 median 차, MAD 비, robust linear slope 차, two-sample classifier-AUC 또는 energy-distance proxy, signed CUSUM, peer-detrended 버전을 만든다. offline 방향만 주력으로 사용한다.
- **직접 목표**: offset onset/offset, drift 초반 decile, 48시간 이상 event.
- **누출 안전성**: feature는 같은 segment의 관측값만 사용한다. window·guard 크기는 inner grid에서만 고른다. label은 feature 계산에 사용하지 않는다.
- **예상 이득**: +0.006~+0.018.
- **비용**: CPU 2~5시간. 정확 classifier-AUC가 느리면 robust summary proxy로 1시간 내 screen.
- **중단 기준**: cache 생성이 기존 feature 시간의 3배를 넘거나, E1 없는 단독 inner gain이 +0.003 미만이고 E1 결합도 +0.005 미만이면 중단.

### E3. 변화점 proposal → 구간 classifier — 우선순위 3

- **가설**: 장기 event는 한 번 찾은 뒤 전체 구간을 채워야 row F1이 오른다. row probability hysteresis만으로는 48시간 이상 event의 중간 저점에서 끊긴다.
- **설계**: E2 score peak, XGB high-score island, slope reversal을 candidate boundary로 만든다. 공식 duration 범위 안에서 top-K interval만 열거한다. interval마다 시작 전/내부/종료 후 median·slope·variance, peer association, XGB score quantile, duration, boundary return을 집계해 offset/drift/noise/normal segment classifier를 학습한다. accept된 interval을 row label로 펼친다.
- **직접 목표**: 48시간 이상 FN 2,393행, composite row recall 0.6755, I-L1/S-L2.
- **누출 안전성**: true boundary를 proposal 입력으로 쓰지 않는다. fold train true event는 classifier label에만 사용한다. outer candidate/threshold는 inner에서 동결한다.
- **예상 이득**: +0.006~+0.022.
- **비용**: CPU 3~8시간. candidate top-K를 segment-day당 20 이하로 제한.
- **중단 기준**: proposal recall이 inner true offset/drift event의 90% 미만이거나 후보 대비 positive precision이 5% 미만, 전체 FP row가 10% 이상 증가하면 중단.

### E4. 계절·성층 gate 동적 peer association — 우선순위 4

- **가설**: 순간 leave-one-out 평균은 층별 수온약층과 한 층의 정상 급변을 충분히 설명하지 못한다. 정상 fold-train에서 학습한 pairwise 관계와 최근 상관 gate가 sensor-only offset/drift를 더 잘 분리한다.
- **설계**: station, depth regime, 월/계절별로 정상행만 사용해 pairwise robust slope/intercept, residual MAD, 최근 24h/72h/7d correlation, vertical spread percentile을 적합한다. 성층 spread가 높거나 correlation이 낮으면 peer score를 수축하고, 여러 peer 중 robust median을 쓴다. G-ORS는 long temporal baseline fallback만 사용한다.
- **직접 목표**: I-L1, S-L2, S-L6 offset, peer가 있는 composite event.
- **누출 안전성**: 관계 모수는 fold train의 label=0에서만 적합. validation의 정상/이상 라벨로 gate를 재적합하지 않음. depth missing fallback 별도.
- **예상 이득**: +0.005~+0.018.
- **비용**: CPU 2~6시간, GPU 불필요.
- **중단 기준**: 성층기 정상 FP/day가 10% 이상 증가, I/S 중 한 station weighted F1이 0.01 이상 하락, 또는 peer 없는 G-ORS 성능이 0.005 이상 하락하면 중단.

### E5. 저비율 boundary-hard 합성 + 정상 hard negatives — 우선순위 5

- **가설**: 4% 전역 합성은 지나치게 쉬운 양성과 많은 false alarm을 만들었다. fold train에서 관측된 낮은 amplitude·긴 duration의 offset/drift만 0.25/0.5/1.0% 주입하고, 현재 모델이 헷갈리는 정상 내부파를 hard negative로 함께 주면 precision을 보호할 수 있다.
- **설계**: 정상 segment에만 주입한다. amplitude/duration/overlap 분포는 해당 fold train의 labeled event에서 robust quantile로 추정한다. synthetic loss weight는 real loss의 최대 0.25, synthetic sample은 type expert에만 우선 투입한다. XGB high-score label=0 block을 동일 batch weight로 보강한다.
- **직접 목표**: offset/drift low-score FN, noise+drift, drift+offset.
- **누출 안전성**: 주입 위치와 분포는 fold train만 사용. validation에 합성하지 않음. random seed와 event manifest 저장.
- **예상 이득**: +0.003~+0.012.
- **비용**: CPU 3~8시간. 기존 feature 재계산 비용이 주 비용.
- **중단 기준**: any inner split precision 0.005 이상 하락, FP/day 5% 이상 증가, 또는 0.25% 설정부터 weighted F1 비개선이면 더 높은 비율을 시도하지 않음.

### E6. 작은 repair network의 구조 discrepancy를 tree feature로 — 우선순위 6

- **가설**: 큰 deep detector 대신 정상 manifold로 복구하는 작은 network의 amplitude/difference/trend/peer-correlation discrepancy가 tree와 다른 장기 신호를 제공한다.
- **설계**: JuRe에서 영감을 받은 1개 depthwise-separable residual block을 fold-train 정상 window에만 학습한다. 6h/24h/72h block corruption을 복구하고 raw reconstruction MSE가 아니라 level, derivative, robust trend, peer relation discrepancy 8~16개를 산출해 E1/E3에 추가한다. 외부 checkpoint는 사용하지 않는다.
- **직접 목표**: subtle offset/drift와 noise, deep가 회수하지만 XGB가 놓친 FN.
- **누출 안전성**: normalizer와 network는 fold train label=0만 사용. mask가 gap을 넘지 않음. outer reconstruction loss로 epoch 선택 금지.
- **예상 이득**: +0.003~+0.012.
- **비용**: RTX 5090 기준 fold당 약 1~3시간, 총 3~9시간. bf16 사용.
- **중단 기준**: 첫 fold에서 XGB와 score correlation 0.98 초과이면서 inner gain +0.002 미만, 정상 tail FP가 10% 증가, 또는 6시간 내 첫 fold 완료 불가면 중단.

### E7. multi-scale normal prototype distance — 우선순위 7

- **가설**: 6h/24h/7d normal patch와 계절 prototype의 거리는 장기 offset/drift가 local rolling에 흡수된 뒤에도 정상 regime에서 벗어났음을 보여준다.
- **설계**: fold-train 정상 patch embedding을 station/depth regime별 k-medoids 또는 작은 memory bank로 요약한다. point prototype과 6h/24h patch prototype, 7d/계절 prototype 거리를 각각 feature로 제공한다. H-PAD/MEMTO 전체 architecture가 아니라 distance view만 최소 구현한다.
- **직접 목표**: ≥48h offset/drift, I-L1, no-peer G-L1.
- **누출 안전성**: prototype과 K 선택은 inner train만 사용. validation/test를 cluster에 포함하지 않음. 입력 absolute year shortcut은 제외.
- **예상 이득**: +0.002~+0.010.
- **비용**: CPU/GPU 4~12시간, prototype 수에 따라 4~12GB VRAM.
- **중단 기준**: E6보다 inner gain이 낮고 계산비가 2배 이상, unseen station stress가 0.01 이상 하락, 또는 prototype assignment가 계절만 복제하고 anomaly separation을 못하면 중단.

### E8. onset/offset/type multi-head TCN을 proposal 내부에서만 사용 — 우선순위 8

- **가설**: 기존 TCN의 높은 offset/drift recall과 낮은 precision을 proposal gate로 제한하면 경계 복원에만 장점을 쓸 수 있다.
- **설계**: 기존 TCN에 binary, 5-type, onset, offset, relative-position 또는 duration-bin head를 둔다. event crop과 normal high-score crop을 균형 sampling한다. TCN이 단독 label을 내지 않고 E3 후보의 boundary refinement와 interval feature만 제공한다.
- **직접 목표**: drift 초반 0.262 recall, long-event boundary, composite event.
- **누출 안전성**: onset/offset target은 fold train event에서만 생성. configuration 선택은 inner validation only. outer architecture 순위로 설정을 고르지 않음.
- **예상 이득**: +0.002~+0.010.
- **비용**: RTX 5090 8~20시간. 기존 checkpoint를 초기화에 쓰려면 fold provenance가 동일한 것만 허용.
- **중단 기준**: proposal-gated precision이 XGB보다 0.01 이상 낮음, inner weighted gain +0.004 미만, 또는 12-setting screen에서 8개 연속 기준 미달이면 조기 중단.

### E9. partial-pooling regime calibration — 우선순위 9

- **가설**: 하나의 global threshold가 I-L1/S-L2의 score scale 차이를 놓친다. station-layer별 완전 독립 threshold가 아니라 global logit에 shrinkage된 group offset을 더하면 과적합을 줄이며 recall을 보정할 수 있다.
- **설계**: inner OOF에서 global temperature/Platt + station + layer/depth-regime random-like offset을 L2 shrinkage로 적합한다. 양성 support가 작은 그룹은 global로 수축한다. threshold는 공식 F1 finite grid로 선택하고 4% positive rate를 강제하지 않는다.
- **직접 목표**: I-L1, S-L2, G no-peer score scale, model/segment score 통합.
- **누출 안전성**: calibration은 inner OOF만 사용. outer/test positive rate 또는 group label을 사용하지 않음.
- **예상 이득**: +0.002~+0.008.
- **비용**: CPU 0.5~2시간.
- **중단 기준**: group별 최소 2개 positive event를 만족하지 못하거나, worst-station F1 0.01 이상 하락, bootstrap 90% CI 하한이 0 이하이면 탈락.

### E10. 최대 3개 view의 비음수 조건부 ensemble — 우선순위 10

- **가설**: XGB, E3 segment, E6/E8 sequence score가 서로 다른 오류를 낼 때만 작은 convex/stacked ensemble이 이득을 준다.
- **설계**: XGB를 anchor weight ≥0.5로 고정하고 후보는 최대 2개 추가한다. global mean보다 type/proposal gate를 우선한다. score rank/robust-z normalization과 비음수 weight를 inner OOF에서만 고른다. plateau와 spike rule은 ensemble 뒤 override한다.
- **직접 목표**: XGB-only FN 중 orthogonal model이 회수하는 offset/drift, precision 유지.
- **누출 안전성**: 현재 outer disagreement와 oracle은 아이디어 가능성 진단일 뿐 weight 선택에 사용하지 않는다. 모든 weight와 gate는 새 inner OOF로 다시 적합한다.
- **예상 이득**: +0.002~+0.008.
- **비용**: 기반 모델 score가 있으면 CPU 0.5~2시간.
- **중단 기준**: pairwise probability correlation 0.97 초과이면서 inner gain +0.003 미만, 3개 중 2개 inner split에서 비개선, 또는 simple XGB 대비 weighted +0.005 미만이면 제출 후보로 승격하지 않음.

## 7. 실행 순서와 계산 예산

| 단계 | 실험 | 최대 벽시계 예산 | 통과 시 다음 단계 |
|---|---|---:|---|
| A | E1 type experts | 4h CPU | E2와 결합 |
| B | E2 CP features | 5h CPU | E3 proposal |
| C | E3 segment classifier | 8h CPU | E4 peer와 2×2 ablation |
| D | E4 dynamic peer | 6h CPU | A~D 중 inner top 2 동결 |
| E | E5 targeted synthesis | 8h CPU | +0.005 이상일 때만 유지 |
| F | E6 repair scorer | 9h GPU | E7보다 먼저 비교 |
| G | E7 prototypes | 12h CPU/GPU | E6보다 낫거나 orthogonal할 때만 유지 |
| H | E8 boundary TCN | 20h GPU | proposal gate에서만 평가 |
| I | E9 calibration | 2h CPU | 최종 후보마다 적용/비적용 비교 |
| J | E10 ensemble | 2h CPU | 승격 기준 충족 시 candidate 생성 |

전체를 순차로 모두 돌리지 않는다. A~D에서 weighted +0.01 이상 후보가 나오면 먼저 재현·bootstrap을 완료한다. E~H는 A~D가 해결하지 못한 offset/drift 또는 worst group을 명확히 개선할 때만 연다.

## 8. 승격·중단·제출 판단

새 후보는 다음을 모두 충족해야 한다.

- 동일 inner protocol에서 현재 XGB 대비 weighted F1 +0.005 이상
- 3개 inner split 중 2개 이상 개선, 나머지 하락 0.002 이하
- precision 하락 0.01 이하, 정상 FP/day 상대 증가 10% 미만
- offset 또는 drift recall 중 하나 +0.05 이상, 다른 하나 비열화 0.02 이하
- station별 하락 0.01 이하, I-L1/S-L2 중 적어도 하나 개선
- paired event/day-block bootstrap 90% CI 하한 >0
- frozen 설정으로 outer 진단, full train 재학습, test 재추론, strict validator, 저장 모델 재현을 통과

단, outer가 이미 여러 연구 진단에 노출됐다는 사실을 보고서에 계속 표시한다. test submission은 로컬 결과를 공식 성능으로 포장하지 않고, exact CSV 경로·SHA-256·positive rate·validator 결과를 사용자에게 제시한 뒤 그 파일에 대한 명시적 승인이 있을 때만 진행한다. 하루 1회 기회를 자동으로 쓰지 않는다.

## 9. 권장하지 않는 경로

- flatline을 다시 neural model로 학습해 hard override를 약화하는 것
- positive rate를 4%에 맞추는 threshold 조정
- 모든 deep probability를 XGB와 단순 평균·OR하는 것
- 현재 outer oracle 또는 오류행을 직접 학습/stacking에 넣는 것
- 4% 전역 synthetic injection을 그대로 반복하는 것
- MOMENT·DADA 등 외부 pretrained weight를 provenance 확인 없이 사용하는 것
- G-ORS에 존재하지 않는 peer/depth 값을 impute해 강제로 다변량 model에 맞추는 것
- background distribution drift 문헌을 주입된 linear sensor drift와 동일시하는 것
- outer 결과를 보고 무제한 feature·threshold를 반복 조정하는 것

## 10. 1차 출처 등록부

| 발행일/상태 | 주제 | 1차 출처 | DOI/공식 URL |
|---|---|---|---|
| 2023, NeurIPS | masked SSL | SimMTM | https://papers.nips.cc/paper_files/paper/2023/hash/5f9bfdfe3685e4ccdbc0e7fb29cccf2a-Abstract-Conference.html |
| 2023, ICLR | patch·masked pretrain | PatchTST | https://openreview.net/forum?id=Jbdc0vTOcol |
| 2023, NeurIPS | memory reconstruction | MEMTO | https://papers.nips.cc/paper_files/paper/2023/hash/b4c898eb1fb556b8d871fbe9ead92256-Abstract-Conference.html |
| 2023, JMLR 24(216) | classifier change point | changeforest | https://www.jmlr.org/papers/v24/22-0512.html |
| 2023, JMLR 24(81) | online CUSUM | FOCuS | https://www.jmlr.org/papers/v24/21-1230.html |
| 2023, NeurIPS | 배경 drift decomposition | D3R | DOI 10.52202/075280-0473 |
| 2024, NeurIPS | 신뢰 가능한 TSAD benchmark | TSB-AD | DOI 10.52202/079017-3437 |
| 2024, NeurIPS | spatial association | SARAD | DOI 10.52202/079017-1533 |
| 2024, NeurIPS | gradual/abrupt boundary | RECURVE | DOI 10.52202/079017-0194 |
| 2024, AISTATS | multivariate change point | Wu et al. | https://proceedings.mlr.press/v238/wu24g.html |
| 2024, PVLDB 17(11) | synthetic regime·ensemble | AutoTSAD | DOI 10.14778/3681954.3681978 |
| 2024, JMLR 25(162) | shift-adaptive conformal | Gibbs & Candès | https://www.jmlr.org/papers/v25/22-1218.html |
| 2024, JMLR 25(225) | non-exchangeable conformal | Oliveira et al. | https://www.jmlr.org/papers/v25/23-1553.html |
| 2025-05-03~05, AISTATS | multi-system score calibration | M²AD | https://proceedings.mlr.press/v258/alnegheimish25a.html |
| 2025, ICLR | hybrid patch/period prototype | H-PAD | https://openreview.net/forum?id=8TBGdH3t6a |
| 2025, ICLR | adaptive bottleneck·dual decoder | DADA | https://proceedings.iclr.cc/paper_files/paper/2025/hash/ca7998666c2e53cc1e882b7268414d8a-Abstract-Conference.html |
| 2025-07-13~19, ICML | causal relation augmentation | CAROTS | https://proceedings.mlr.press/v267/kim25aa.html |
| 2025-12-03, NeurIPS spotlight poster | self-perturbation·dynamic graph | SPAGD | https://nips.cc/virtual/2025/poster/116664 |
| 2025, NeurIPS | idempotent reconstruction | IGAD | https://papers.nips.cc/paper_files/paper/2025/hash/ebba26f4dbcfe7318e4a263c54e9cf42-Abstract-Conference.html |
| 2026-02, TMLR | multivariate benchmark·selection | mTSBench | https://openreview.net/forum?id=8LfB8HD1WU |
| 2026-04-19, arXiv preprint | minimal repair network | JuRe | https://arxiv.org/abs/2604.17388 |
| 2026-05-24, TMLR | 실무 정렬 평가 | Barrish & van Vuuren | https://openreview.net/forum?id=RyMLAr5tFU |

## 11. 최종 연구 판단

가장 높은 확률의 개선은 “XGBoost를 버리는 것”이 아니라 XGBoost가 못 보는 장기 구조를 별도 view로 만드는 것이다. P1에서 이미 검증된 강점은 spike·noise·flatline이고, 남은 점수는 offset·drift의 장기 interior, weak station-layer, composite event에 집중되어 있다.

따라서 연구 자원은 E1 유형 전문가, E2 변화점 bank, E3 segment classifier, E4 동적 peer에 먼저 배분한다. 합성은 저비율과 hard-negative 조건을 만족할 때만, deep/SSL은 전체 대체가 아니라 repair·boundary score로만, ensemble은 orthogonality가 실제 inner OOF에서 확인될 때만 사용한다. 이 순서가 현재 증거에 가장 직접적으로 맞고 계산 낭비와 누출 위험도 가장 작다.
