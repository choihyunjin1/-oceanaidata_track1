# P1 G-ORS depth invariance one-shot 독립 감사

- 감사일: 2026-08-13 (Asia/Seoul)
- 실행 ID: `20260813T214518+0900_gors_depth_invariance_one_shot_9d915ffe`
- 판정: **NO-GO — 후보 B를 기각하고 이 실험 계열을 종료한다.**
- 유지 모델: 기존 frozen natural-depth XGBoost incumbent

## 1. 감사 범위와 비교 정의

저장된 OOF 산출물, 원본 train의 키·라벨, test의 정점-층 비중, 실행 감사 파일, 사전등록 영수증, append-only exposure ledger를 서로 독립적으로 대조했다. 모델 학습·추론·튜닝은 다시 실행하지 않았으며, 공개된 outer 결과를 이용한 후속 파라미터 선택도 하지 않았다.

- `S` (secondary context): 기존 frozen natural-depth incumbent
- `A` (deployment-matched baseline): 자연 depth로 fold-train 후 validation의 G-ORS depth만 결측 배포 상태로 변환
- `B` (invariant candidate): G-ORS depth를 fold-train과 validation 양쪽에서 대칭적으로 결측 처리

따라서 주 비교 `B - A`는 동일한 배포 평가 조건에서 train-time 대칭 masking 하나만 바꾼 비교다. seed, fold, XGBoost 700 iterations, fold별 후처리는 고정됐고 inner 재선택은 없었다.

## 2. 행·키·고정 예측 무결성

| 점검 | 독립 확인 결과 |
|---|---:|
| 저장 OOF 행 수 | 421,032 |
| OOF 키 유일성 | PASS |
| frozen reference와 키·fold·행 순서 일치 | PASS |
| 원본 train 라벨과 키 기준 1:1 일치 | PASS |
| frozen secondary 확률 최대 절대오차 | 0.0 |
| frozen secondary 이진 예측 불일치 | 0 |
| Q2 / Q3 / Q4 행 수 | 133,170 / 176,738 / 111,124 |

## 3. 독립 재계산 성능

| Arm | Micro F1 | Test-share weighted F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| S: natural incumbent | 0.8603708380 | 0.8133155526 | 12,946 | 1,093 | 3,109 |
| A: deployment-matched | 0.8572095038 | 0.8043980283 | 12,862 | 1,092 | 3,193 |
| B: symmetric invariant | 0.8541156215 | 0.8070860200 | 12,898 | 1,249 | 3,157 |

- 주 비교 weighted F1 `B - A`: **+0.0026879917**
- 보조 안전성 비교 weighted F1 `B - S`: **-0.0062295325**
- G-ORS F1: A `0.7633410673`, B `0.7553998832`, delta **-0.0079411840**
- Non-G weighted F1: A `0.8119578437`, B `0.8164435441`, delta **+0.0044857004**

A의 weighted F1은 사전 고정값 `0.8043980282796417`을 부동소수점 오차 약 `1e-16` 이내로 정확히 재현했다.

## 4. Fold 및 그룹 안전성

| Fold | A weighted F1 | B weighted F1 | B-A | G 양성 수 |
|---|---:|---:|---:|---:|
| Q2 | 0.7472530349 | 0.7505515407 | +0.0032985058 | 71 |
| Q3 | 0.8588809518 | 0.8626692908 | +0.0037883390 | 995 |
| Q4 | 0.8832899361 | 0.8842750267 | +0.0009850905 | 0 |

세 fold 모두 weighted F1은 비열화였지만, 최악의 non-G 정점-층 하락은 `I-ORS layer 2`의 **-0.0271222986**으로 허용 한계 `-0.01`을 넘었다. 다음으로 큰 하락은 `S-ORS layer 2`의 `-0.0269436052`였다.

## 5. KST block bootstrap과 정상 FP/day

프로젝트 함수를 그대로 재호출하지 않고, Asia/Seoul 기준으로 positive event와 정상 정점-층 일 블록을 독립 구성해 2,000회 paired bootstrap(`seed=20260813`)을 재계산했다.

- 블록 수: 3,089 = positive event 141 + normal station-layer day 2,948
- weighted F1 delta 평균: `-0.0030312189`
- 중앙값: `-0.0031644175`
- 90% CI: **[-0.0092102509, 0.0035531322]**
- `P(B > A)`: `0.206`
- 저장 metrics와의 수치 차이: `0.0`

정상 404,977행에 대한 독립 FP/day 계산:

- A: 1,092 FP / 2,948일 = `0.3704206242`
- B: 1,249 FP / 2,948일 = `0.4236770692`
- 상대 증가: **+14.377289%**
- 저장 metrics와의 수치 차이: `0.0`

## 6. 사전등록 promotion gate

| Gate | 관측값 | 기준 | 판정 |
|---|---:|---:|---|
| Weighted F1 개선 | +0.0026879917 | >= +0.005 | FAIL |
| Bootstrap 90% CI 하한 | -0.0092102509 | > 0 | FAIL |
| G-ORS F1 개선 | -0.0079411840 | >= +0.02 | FAIL |
| Non-G weighted 안전성 | +0.0044857004 | >= -0.001 | PASS |
| 최악 non-G 그룹 하락 | -0.0271222986 | >= -0.01 | FAIL |
| 비열화 fold 수 | 3 / 3 | >= 2 | PASS |
| 정상 FP/day 상대 증가 | +14.377289% | < 10% | FAIL |
| 보조 incumbent 안전성 `B-S` | -0.0062295325 | >= -0.001 | FAIL |

Primary gate는 7개 중 2개만 통과했고, 보조 incumbent 안전성 veto도 실패했다. `promotion_eligible=false`와 독립 계산 결과가 일치한다.

## 7. 해시·원장·실행 폐쇄

| 산출물 | SHA256 | 확인 |
|---|---|---|
| OOF | `695c16969d1f5b7613bd99b5407619a7230f63c1cbd3107719344285ebf57d86` | manifest 일치 |
| metrics | `e76837aec03ecc23999acbe903c75894bfefe1c98d5ee5ca6a77e887e8625a61` | manifest 일치 |
| execution audit | `d3775088f31b0a0b1f29a0cd3e2d553f1fd3f19467d30b73a780902355564a21` | manifest 일치 |
| prereg receipt | `a5e48989eefbe4425b24e2f0f627701e473c07531b6302591166402340c28b29` | manifest 일치 |
| canonical prereg | `7837680a3d89083d1f87e1a04c4db763ea62ab8a71d95d794a4a4abfc11c074e` | receipt·manifest 일치 |

config, deployment stress, frozen metrics/OOF/selection, prereg, 원본 train/test의 해시와 바이트 수도 모두 manifest와 일치했다. Exposure ledger 이벤트는 `preregistered -> outer_evaluated -> closed`, outer exposure count는 정확히 1회이며 run ID와 prereg SHA가 일관됐다. 입력 시점 ledger prefix 해시·길이도 manifest와 일치하여 append-only lifecycle이 확인됐다.

## 8. 최종 결정

G-ORS train-time symmetric depth masking은 전체 weighted F1을 A보다 소폭 높였지만, 사전등록 최소 개선폭에 못 미쳤고 G-ORS 자체 성능, bootstrap 신뢰구간, 최악 그룹, FP/day, incumbent 안전성을 동시에 위반했다. 따라서 후보 B를 승격하거나 제출하지 않는다.

이 outer 결과는 이미 1회 공개됐으므로 동일 계열의 결과 기반 튜닝·재실행에는 사용하지 않는다. 이번 감사 과정에서 업로드·커밋·push는 수행하지 않았다.
