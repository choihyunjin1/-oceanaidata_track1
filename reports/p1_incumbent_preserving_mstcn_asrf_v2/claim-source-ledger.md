# P1 고용량 MS-TCN++/ASRF v2 — 주장·근거 원장

작성일: 2026-08-27  
상태: 사전등록 연구 근거를 보존하고, 같은 날 종료된 봉인 실행의 실제 결과를 사후 부기로 닫았다. 최종 판정은 `NO_GO_CONFIRMATORY`이며 제출은 생성·업로드하지 않았다.

## 해석 원칙

- 아래 논문과 공식 구현은 시계열 분할·경계 예측·다단계 시간 합성곱의 구조적 근거다.
- 원 연구의 대상은 주로 비디오 동작 분할이다. 해양 관측 품질관리의 희소 이진 행 F1 또는 대회 점수 `+3`을 직접 보증하지 않는다.
- `직접 근거`와 `본 실험의 전이 추론`을 분리한다. 전이 추론은 Q2 선택과 Q3·Q4 고정 확인으로 반증 가능해야 한다.
- 아래의 논문·공식 구현은 구조 선택의 1차 출처다. 이번 로컬 historical 결과나 대회 공식 점수의 근거로 확대 해석하지 않는다.
- 각 절의 `실제 종료 판정`은 사전등록 후 얻은 로컬 증거다. 원문 주장이나 사전등록 가설을 사후 수정하지 않고, 통과·반증·미해결을 구분한다.

## 종료 결과 고정

- 실행 범위: Q2 유한 grid 선택 후, 선택 recipe를 고정한 Q3·Q4 fresh refit 및 양쪽 blind seal 이후 단 한 번의 확인 평가.
- Q2 선택 전용 결과: Router anchor F1 `0.7922755741`에서 후보 F1 `0.8904328232`로 `+0.0981572491`; 선택 recipe는 width `512`, epoch `125`, threshold `0.9`, 3-seed raw 평균이었다. 이 수치는 유한 grid 최대 선택을 거친 낙관적 selection evidence이며 promotion evidence가 아니다.
- 확인 결과: Q3 ΔF1 `+0.0132993769`, Q4 ΔF1 `-0.0314844165`, pooled anchor F1 `0.9029170235`, pooled 후보 F1 `0.8977769149`, pooled ΔF1 `-0.0051401087`.
- paired 21일 circular moving-block bootstrap의 90% CI는 `[-0.0281000264, 0.0140666998]`였다. 개선 station은 `1/3`, 추가 행 precision은 `0.3811420983`이었다.
- research gate와 high-impact official-probe gate가 모두 실패했다. 따라서 이 recipe는 승격·공식 probe·제출 대상으로 사용하지 않는다.
- `terminal_result.json`에는 `submission_created=false`, `upload_performed=false`, `official_three_point_gain_claimed=false`가 기록되었다. 공식 P1 test/sample/submission 경로는 실행과 독립 QA에서 접근하지 않았다.

## C1. 다단계 시간 합성곱과 평활화

- 주장: dilated 1D TCN을 여러 단계로 쌓고 이전 단계 예측을 정제하면 긴 시퀀스 문맥을 사용하면서 과분할을 줄일 수 있다.
- 직접 근거: Abu Farha & Gall, *MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation*, CVPR 2019. [논문](https://openaccess.thecvf.com/content_CVPR_2019/html/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.html)
- 본 실험의 전이 추론: 장기 QC 사건 내부의 단발성 예측 흔들림과 국소 증상만으로 놓치는 사건을 시간 문맥으로 복원할 가능성이 있다.
- 한계: 원 연구는 다중 클래스 비디오와 segmental F1을 다루므로 P1의 희소 binary row-F1 개선폭으로 수송할 수 없다.
- 실제 종료 판정: Q2에서는 강한 선택 이득이 관측됐지만, 고정 확인의 pooled ΔF1이 `-0.0051401087`로 반전됐다. 따라서 구조적 가능성은 문헌 근거로 남지만 P1에 안정적으로 전이된다는 실험 가설은 이번 recipe에서 반증됐다.

## C2. MS-TCN++의 dual-dilated generator와 국소 refiner

- 주장: prediction generator에서 두 dilation 방향을 결합하고 후속 refinement stage를 분리하는 구조는 장·단기 문맥 결합을 위한 근거가 있다.
- 직접 근거: Li et al., *MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation*, TPAMI 2020. [논문](https://arxiv.org/abs/2006.09220), [공식 구현](https://github.com/sj-li/MS-TCN2)
- 본 실험의 전이 추론: 2,048행 창 전체를 덮는 generator와 local refinement를 결합하되, 현재 Router 양성은 OR로 보존한다.
- 한계: 논문의 50Salads 실험에서는 다섯 번째 stage에서 성능이 저하되고, DDL을 refinement stage까지 확대하면 과적합으로 정확도가 떨어지며, refinement stage를 3개보다 늘려도 추가 이득이 없었다. 따라서 width 512나 300 epoch가 자동으로 우월하다고 간주하지 않는다.
- 실제 종료 판정: width `512`·epoch `125`가 Q2에서 선택됐으나 Q3와 Q4의 효과 방향이 갈렸다. dual-dilated/refinement 구조의 존재만으로 분기·station 이동에 견고한 개선이 생긴다는 주장은 지지되지 않았다.

## C3. 본 실험은 공식 재현이 아니라 고용량 외삽

- 주장: 공식 MS-TCN++ 설정과 비교하면 본 실험의 width 256/512 및 300 epoch는 큰 용량 외삽이다.
- 직접 근거: 공식 고정 구현의 `train.sh`는 100 epoch, prediction generator 11층, refiner 10층×3을 사용하고 기본 `num_f_maps`는 64다. [고정 `train.sh`](https://raw.githubusercontent.com/sj-li/MS-TCN2/f423a9e65f4ccb1cd7322eb9f94946a19e787993/train.sh), [고정 `main.py`](https://raw.githubusercontent.com/sj-li/MS-TCN2/f423a9e65f4ccb1cd7322eb9f94946a19e787993/main.py), [고정 `model.py`](https://raw.githubusercontent.com/sj-li/MS-TCN2/f423a9e65f4ccb1cd7322eb9f94946a19e787993/model.py)
- 본 실험의 전이 추론: 300 epoch는 마지막 모델을 무조건 채택하기 위한 값이 아니라 1–300 epoch 수렴·과적합 곡선을 관찰하는 사전 고정 상한이다. width 256과 512도 Q2에서 동등한 전체 budget으로 비교한다.
- 한계: crop, padding, batch 64/128, 165개 입력 특징과 희소 이진 손실은 공식 실험과 다르다.
- 실제 종료 판정: Q2의 6개 fit이 모두 `300/300` epoch를 완료했고 nonfinite 합은 0이었다. 선택 epoch는 상한이 아닌 `125`였으므로 `300 epoch 수렴`을 주장하지 않는다. 고용량 탐색은 계획대로 완료됐지만 확인 실패로 우월성 주장은 기각한다.

## C4. class-agnostic boundary supervision

- 주장: 프레임별 분류와 별도의 경계 회귀를 결합하면 과분할을 줄이고 구간 경계를 정제할 수 있다.
- 직접 근거: Ishikawa et al., *Alleviating Over-Segmentation Errors by Detecting Action Boundaries*, WACV 2021. [논문](https://openaccess.thecvf.com/content/WACV2021/html/Ishikawa_Alleviating_Over-Segmentation_Errors_by_Detecting_Action_Boundaries_WACV_2021_paper.html), [공식 구현](https://github.com/yiskw713/asrf)
- 본 실험의 전이 추론: 시작·종료 head는 장기 offset/drift/noise 사건의 구간화를 보조한다. 최종 후보는 `현재 Router OR 디코딩된 장기 사건`이다.
- 한계: P1에서는 경계 행이 사건 내부 행보다 훨씬 적다. class balance, 추가 행 precision, 사건 길이별 recall을 별도로 확인해야 한다.
- 실제 종료 판정: pooled 후보는 Router FN을 `287`행 줄였지만 FP를 `466`행 늘렸고, 추가 `753`행의 precision은 `0.3811420983`에 그쳤다. 경계 head의 독립 기여를 분리한 ablation이 아니므로 일부 recall 증가를 ASRF 효과로 귀속하지 않는다.

## C5. 수렴곡선에 단일 F1 이상의 진단이 필요

- 주장: ASRF 공식 학습은 epoch별 loss 외에도 분류, segment, boundary 지표와 checkpoint를 기록한다.
- 직접 근거: ASRF 공식 고정 `train.py`. [고정 구현](https://raw.githubusercontent.com/yiskw713/asrf/9623f1e8d9a1171333a4eeb65d190997b6c44a95/train.py)
- 본 실험의 전이 추론: epoch·step·LR·손실 구성요소·gradient·시간·VRAM과 checkpoint별 proposal/Router-union 지표를 보존한다.
- 한계: 원 구현의 best validation loss 선택이 P1 F1 checkpoint 선택을 정당화하지는 않는다.
- 실제 종료 판정: Q2 6개 history는 각 300 epoch, Q3·Q4 6개 confirmatory history는 각 125 epoch로 연속 기록됐고 모두 nonfinite 합 0이었다. 학습 안정성은 확인됐지만 Q4와 pooled 일반화 실패를 상쇄하지 않는다.

## C6. 시간 순서를 보존한 검증

- 주장: 비정상 시계열에서는 시간 순서를 보존한 out-of-sample 평가가 무작위 분할보다 현실적인 오류 추정에 유리하다.
- 직접 근거: Cerqueira, Torgo & Mozetič, *Evaluating time series forecasting models*, Machine Learning 2020. [논문](https://arxiv.org/abs/1905.11744)
- 본 실험의 전이 추론: Q2는 width·epoch·threshold 선택 전용으로만 쓰고, 선택 사양을 고정한 뒤 Q3와 Q4를 새로 학습해 한 번만 확인한다. Q3·Q4는 전역 달력 분기가 아니라 각 `(station, year, layer)` 시계열에서 엄격히 순서가 보존되고 exact key가 겹치지 않는 frozen fold로 해석한다.
- 한계: 원 연구는 forecasting을 다룬다. 본 실험의 21일 gap, 전역 달력 envelope의 19시간 50분 중첩, 두 fold를 같은 KST 날짜 cross-section으로 묶는 21일 bootstrap block 길이를 직접 보증하지 않는다.
- 실제 종료 판정: Q2/Q3/Q4 split receipt 모두 `cross_split_window_count=0`, holdout preprocessing/train/truth 선사용 0, runtime input 165를 만족했다. Q3·Q4 blind receipt와 semantic replay가 모두 PASS한 뒤에만 metric을 열었다. 확인 실패는 이 검증 설계가 낙관적 Q2 선택을 실제로 걸러낸 사례다.

## C7. 시드와 선택 변동성

- 주장: 초기화, 데이터 순서와 hyperparameter 선택 변동을 무시하면 작은 benchmark 차이를 과신할 수 있다.
- 직접 근거: Bouthillier et al., *Accounting for Variance in Machine Learning Benchmarks*, MLSys 2021. [논문](https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html), [상세판](https://arxiv.org/abs/2103.03098)
- 본 실험의 전이 추론: 두 width 모두 3개 고정 seed를 완주하고 seed 평균 예측으로만 선택한다. best seed 단독 보고를 금지한다.
- 한계: 3개 seed는 계산 예산상 최소 안정성 점검이지 통계적 충분성의 보장이 아니다. 시간 block bootstrap도 Q2 다중선택과 seed 불확실성을 모두 포함하지 않는다.
- 실제 종료 판정: 등록된 모든 Q2·Q3·Q4 seed fit이 완주했고 개별 best seed가 아닌 3-seed 평균만 사용했다. 그럼에도 Q3/Q4 방향 불일치와 음의 pooled 효과가 남아, seed 평균은 분산 과신 완화 수단일 뿐 도메인 안정성의 보장이 아님을 확인했다.

## C8. incumbent-preserving OR의 수학적 성질

- 주장: 기존 Router 양성을 보존하는 OR 후보는 기존 positive 제거를 0으로 만들지만 F1 개선을 보장하지 않는다.
- 근거 유형: 본 실험의 수학적 유도. 기존 `TP, FP, FN`에서 `a`개 FN을 복구하고 `b`개 FP를 추가하면, F1 개선에는 추가 양성 precision이 기존 F1의 절반보다 커야 한다.
- 본 실험의 적용: 추가 행 수 자체가 아니라 Router FN 회복률, 추가 precision, FP ratio를 함께 gate한다.
- 한계: 이 조건은 aggregate binary F1의 대수적 조건이다. station·사건 길이·공식 도메인 이동을 설명하지 않는다.
- 실제 종료 판정: pooled `anchor_positive_removed_rows=0`으로 집합 invariant는 지켜졌다. 그러나 추가 행 precision `0.3811420983`과 FP 증가 때문에 F1은 하락했다. 즉 수학적 보존 성질은 확인됐고 성능 보장 부재도 동시에 실증됐다.

## 중단 기준

핵심 구조, 검증, 분산과 수렴 주장이 모두 1차 출처 또는 명시적 전이 추론으로 채워졌고, 봉인된 Q2→Q3/Q4 실행도 종료됐다. 추가 문헌 반복 검색은 이번 recipe의 음의 확인 결과를 바꾸지 못하므로 중단한다. 다음 연구는 동일 결과의 재튜닝·재제출이 아니라 Q4/S-ORS·I-ORS의 FP 이동을 사전등록된 새 가설로 다뤄야 한다.

## 로컬 종료 증거와 무결성 원장

| 증거 | 역할 | SHA-256 |
|---|---|---|
| execution seal | 사전 고정 코드·설정·one-shot 범위 | `2d42ce76966876f33daf0bd3e8e62051876f95f92e866588713bcfb84886bb25` |
| attempt lock | 단일 launcher 실행 provenance | `092da264171250e03e49a29ccf9f9440965738b24db9420ed002c38014e61204` |
| selected recipe | Q2에서 고정된 Q3/Q4 recipe | `171618200c69dc8e5039e5404bdeb4e7cb6369f15ab8ff43af1be7492837515a` |
| Q2 blind grid NPZ | 882개 selection cell의 blind 배열 | `867d4b25d968ce4231179181e05eee95cda04d76689c03c1514b907e21dd1f02` |
| Q3 blind NPZ | 첫 confirmatory 후보 봉인 | `afc1f10d0ee6bb3ab1896f8a579a074ec386c73639f08d778355894b7d7347a9` |
| Q4 blind NPZ | 둘째 confirmatory 후보 봉인 | `f04f3d4cb22e75a8ec69ebed1656b618231f9a04552dcb5332670a2594dc948a` |
| confirmatory semantic replays | Q3·Q4 decoder/anchor-union 재현 PASS | `bcede9e5c51f59c26213f956f7db86ebd610053e27395d49391ab59faee9687a` |
| confirmatory metrics | 최종 로컬 지표·gate | `964cae7d7dbb9f413244462eb9258e883e14f6073ad5549fadf482cb9cd03bd4` |
| terminal result | `NO_GO_CONFIRMATORY`, no submission/upload | `7640cc0e29f364a26cd8199a7e9a55acdf329699cd5923679d8f0d513c4af2b1` |

독립 read-only QA는 예상 산출물 `40/40`, 누락·예상 외 파일 0, 최종 assertion 137개 PASS를 확인했다. 판정은 `PASS_CONTROLLED`(`P0=0`, `P1=0`, `P2=2`)이다. 두 P2는 관측된 변조가 아니라 사전 명시된 비적대적 Python 실행 경계 밖의 잔여 위험이다. 따라서 산출물 무결성은 통과했지만 과학적 성능 판정은 `NO_GO`로 유지한다.

## 출처 등록부

모든 웹 출처의 최종 접근일은 2026-08-27이다.

| ID | 제목 | 저자·출판처·일자 | URL·접근 메모 |
|---|---|---|---|
| S1 | *MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation* | Yazan Abu Farha, Jürgen Gall; IEEE/CVF CVPR, 2019-06 | [CVF 공식 공개본](https://openaccess.thecvf.com/content_CVPR_2019/html/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.html) |
| S2 | *MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation* | Shijie Li 외; IEEE TPAMI, 2020; arXiv 제출 2020-06-16 | [arXiv 원문](https://arxiv.org/abs/2006.09220) |
| S3 | MS-TCN2 공식 PyTorch 구현 | sj-li; GitHub; 고정 commit `f423a9e65f4ccb1cd7322eb9f94946a19e787993` | [저장소](https://github.com/sj-li/MS-TCN2), [train.sh](https://raw.githubusercontent.com/sj-li/MS-TCN2/f423a9e65f4ccb1cd7322eb9f94946a19e787993/train.sh), [main.py](https://raw.githubusercontent.com/sj-li/MS-TCN2/f423a9e65f4ccb1cd7322eb9f94946a19e787993/main.py), [model.py](https://raw.githubusercontent.com/sj-li/MS-TCN2/f423a9e65f4ccb1cd7322eb9f94946a19e787993/model.py) |
| S4 | *Alleviating Over-Segmentation Errors by Detecting Action Boundaries* | Yuchi Ishikawa 외; IEEE/CVF WACV, 2021-01 | [CVF 공식 공개본](https://openaccess.thecvf.com/content/WACV2021/html/Ishikawa_Alleviating_Over-Segmentation_Errors_by_Detecting_Action_Boundaries_WACV_2021_paper.html) |
| S5 | ASRF 공식 구현 `train.py` | yiskw713; GitHub; 고정 commit `9623f1e8d9a1171333a4eeb65d190997b6c44a95` | [고정 원문](https://raw.githubusercontent.com/yiskw713/asrf/9623f1e8d9a1171333a4eeb65d190997b6c44a95/train.py) |
| S6 | *Evaluating time series forecasting models: An empirical study on performance estimation methods* | Vítor Cerqueira, Luís Torgo, Igor Mozetič; *Machine Learning* 109, 2020; arXiv 2019-05-28 | [arXiv 원문](https://arxiv.org/abs/1905.11744) |
| S7 | *Accounting for Variance in Machine Learning Benchmarks* | Xavier Bouthillier 외; MLSys 2021 | [MLSys 공식 초록](https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html), [arXiv 상세판](https://arxiv.org/abs/2103.03098) |
| S8 | `OFFICIAL_RESULTS_20260826.json` | 분당독고다이 참가자 UI 관측 영수증; 2026-08-26 | 로컬 검증 파일. UI가 플랫폼 제출 ID를 제공하지 않아 고정 순서·시각·점수·SHA-256을 durable key로 사용한다. 공개 웹 출처가 아니며 공식 test/sample/submission 원본은 접근하지 않았다. |
