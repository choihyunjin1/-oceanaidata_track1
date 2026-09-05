# Historical snapshot — NOT ACTIVE INSTRUCTIONS

Preserved before the 2026-09-05 guidance cleanup. Current entry: [repository README](../../../README.md).
Historical relative links below were originally relative to the repository root. Old permissions, candidate eligibility and commands are not current authorization.

# 반드시 먼저 읽을 메모 — Ocean AI Data P2

> 상태: 2026-09-01 KST 규정 갱신
> P2 데이터 열람, 특징 설계, 검증, 학습, 제출 준비 전에 `00_ORGANIZER_DATA_POLICY.md`, 이 문서와 원본 `README.md`를 처음부터 끝까지 다시 읽는다.

## 1. 문제의 정확한 목표

- 문제명: 소청초 기지 중간층 수온 연직 구조 복원
- 판정이 아니라 **연속값 복원** 문제다.
- 목표: S-ORS 2025년 layer 2·3·4의 가림 구간 수온을 10분 간격으로 복원한다.
- 공칭 수심: layer 2 = 7.04 m, layer 3 = 9.44 m, layer 4 = 14.74 m.
- 가림 구간: 2025-09-01 00:00부터 2025-10-31 23:50까지 KST.
- 가림 층은 `temp`와 `psal`이 함께 제외되어 있다. 가림 층의 염분을 입력 특징으로 사용하면 안 된다.
- 제공 공개층: layer 1(4.19 m), 5(19.59 m), 6(30.68 m), 7(39.45 m), 8(49.35 m).
- 채점 대상은 전체 가림 격자 26,352행이 아니라 `test_index.csv`의 **26,061개 키**다.
- 채점 지표는 세 층 전체 행에 대한 RMSE(℃)다. 낮을수록 좋다.
- 공식 수심 선형보간 기준 RMSE는 1.290264℃다. 동일 로컬 가상 가림 결과와 직접 비교해 후보를 승격한다.

## 2. 제출 계약

열 이름과 순서를 정확히 지킨다.

```text
station,layer,time,temp
```

- 행 수 26,061.
- `(station, layer, time)`은 `test_index.csv`와 정확히 같은 키·순서·유일성을 가져야 한다.
- `temp`는 결측과 무한값이 없는 유한한 숫자이며 -5~45℃ 범위여야 한다.
- CSV는 UTF-8, index 없음으로 저장한다.
- 제출 양성률 같은 분류 개념은 P2에 적용하지 않는다.

## 3. 문제문이 강조한 물리적 특징

- 여름 표층 가열로 성층이 강해지고, 가을 표층 냉각·바람 혼합으로 연직 균질화가 진행된다.
- 태풍은 성층 붕괴를 며칠 단위로 급격히 진행시킬 수 있다.
- 반일주조 약 12.42시간 성분과 내부파에 따른 층간 위상차가 존재할 수 있다.
- 2024년 같은 계절의 실제 연직 전이 자료가 직접적인 학습 근거다.
- 수온–염분 관계는 수괴와 혼합 상태를 나타내지만, target layer 2·3·4의 가림 구간 염분은 사용할 수 없다.
- 공개 layer 1과 5 사이 약 15 m 구간은 수심 선형보간만으로 수온약층 곡률을 표현할 수 없다.
- 모델의 핵심 목표는 공개층 선형보간값 자체가 아니라 그 위에 남는 **프로파일 곡률 잔차**를 복원하는 것이다.

## 4. 로컬 원본에서 확인한 데이터 계약

- 원본 위치는 `P2_DATA_DIR` 환경변수로만 주입한다.
- `observations.csv`: 789,408행, 키 중복 0.
- `test_index.csv`, `sample_submission.csv`, `baseline_interp.csv`: 각각 26,061행이며 키와 순서가 일치한다.
- 층별 채점 수: layer 2 = 8,713 / layer 3 = 8,712 / layer 4 = 8,636.
- 전체 가림 격자 layer 2·3·4의 26,352행은 temp·psal이 모두 결측이다.
- 채점 시각마다 유효한 공개 수온층이 최소 2개 존재한다.
- `observations.csv` 끝에는 2026-01-01 00:00~08:50 KST의 전 변수 결측 padding 432행이 있다. 학습 자료나 미래 문맥으로 사용하지 않는다.
- 2024년 layer 7은 약 49 m지만 2025년 layer 7은 약 39 m이고 layer 8이 약 49 m로 추가된다. 연도 간 layer 번호를 동일 수심 센서로 간주하지 말고 `nominal_depth`와 실제 `depth`를 사용한다.
- 제공 baseline은 대부분의 행에서 목표 수심을 감싸는 가장 가까운 유효 공개층 두 개의 수심 선형보간으로 재현된다.

## 5. 누출 방지와 검증

- 2025년 가림 구간의 실제 KORS/S-ORS 원자료, 실시간 자료, 공개 mirror를 검색·다운로드·대조하지 않는다. 이는 정답을 직접 복원하는 누출이다.
- 2026-09-01 최신 운영진 공지에 따라 배포 데이터 밖의 관측·재분석·예보 자료는 공개 여부와 관계없이 사용할 수 없다. 과거 FAQ 허용 판단은 감사 증거일 뿐 현재 권한이 아니다.
- ERA5·CMEMS·위성·조석·태풍·NASA POWER 및 비배포 KIOST 자료를 특징, 학습, 검증, 선택, 보정 또는 후처리에 사용하지 않는다. 외부 prediction을 이어받은 후보도 비적격이다.
- 실제 관측·기상·해양 자료로 사전학습된 가중치는 금지한다. 합성-only 모델은 `00_ORGANIZER_DATA_POLICY.md`의 네 조건을 모두 입증할 때만 예외다.
- validation에서는 layer 2·3·4의 temp와 psal을 **동시에 모두 가린 상태**로 특징을 계산한다. 한 target layer를 복원할 때 다른 target layer의 값을 입력하면 안 된다.
- 공개층의 동시각 관측, 2024 동일 계절, 2025 가림 전후 관측은 문제에서 제공한 범위이므로 사용할 수 있다.
- 무작위 행 분할을 주 검증으로 쓰지 않는다. 최소한 다음 연속 블록을 독립적으로 보고한다.
  - 2024-09-01~10-31: 같은 계절 전이
  - 2025-07-01~08-31: 가림 직전 강한 성층
  - 2025-11-01~12-31: 가림 직후 혼합 상태
- 모델·창·blend를 선택하는 inner 검증과 최종 성능 보고용 holdout을 구분한다.
- RMSE 전체값뿐 아니라 layer별 RMSE, 시간대별·주별 RMSE, 공개층 결측 패턴별 RMSE, 최대 절대오차를 기록한다.
- 2024 동일 시각 정답을 그대로 복사하는 방법은 계절 전이 시점의 연도 차이로 실패할 수 있으므로 단독 모델로 채택하지 않는다.

## 6. 초기 정찰에서 확인한 기준

- 가상 가림 수심 선형보간 RMSE:
  - 2024년 9~10월: 0.9716℃
  - 2025년 7~8월: 1.6161℃
  - 2025년 11~12월: 0.8686℃
- 고정 파라미터 LightGBM 곡률 잔차 모델의 첫 smoke RMSE:
  - 2024년 9~10월: 0.5424℃
  - 2025년 7~8월: 1.1466℃
  - 2025년 11~12월: 0.7538℃
- 이는 공식 점수가 아니며 초기 구조 확인값이다. 동일 검증 코드를 재현하고 과적합 방지 게이트를 통과하기 전 제출 후보로 승격하지 않는다.

## 7. 2026-08-16 방법 정찰 결정

- PCHIP 수직 곡선보간, 전역 rank-3 EOF, layer별 Ridge, 120개 공개층 동역학, 20개 lean M2 동역학을 동일 연속 블록에서 비교했다.
- 전역 EOF와 PCHIP은 계절·deployment shift에서 불안정하여 중단한다. 성능이 나빴다는 이유로 rank나 spline 설정을 반복 탐색하지 않는다.
- lean M2는 공개층 수온의 6시간 및 12.42시간 전후 변화만 사용하며 정확히 20개 특징을 추가한다. target layer temp·psal은 읽지 않는다.
- lean M2 단독은 8개 유효 계절 블록 중 7개를 개선했지만 2024년 9~10월에는 악화했다.
- 가중치 탐색 없이 V0와 lean M2를 50:50으로 평균한 연구 후보는 8개 블록 모두 개선했다. 전체 paired KST-day bootstrap ΔRMSE는 -0.0769℃, 90% CI는 [-0.0846, -0.0694]였다.
- 이 결과는 hidden 점수가 아니며 `P2_RESEARCH_BLEND50.csv`는 **연구 후보**다. 사용자 승인 전 제출 동결본으로 이름을 바꾸거나 업로드하지 않는다.
- 후속 사전등록 실험에서 공개층 수온의 고정 7일 local M2 amplitude·phase 특징 20개를 lean arm에 추가했다. aggregate RMSE는 1.1234→1.0787℃로 개선됐고 paired KST-day bootstrap 90% CI는 [-0.0551, -0.0347]℃였다.
- 그러나 8개 블록 중 5개만 개선됐으며 2025년 3~4월은 +0.0469℃ 악화해 사전등록된 최대 블록 회귀 +0.02℃를 넘었다. 따라서 이 후보는 `REJECT_AND_CLOSE_FAMILY`이며 제출 CSV를 만들지 않았다.
- M2 amplitude·phase 계열의 window, 최소관측수, blend weight를 사후 재탐색하지 않는다. 이 단계 당시 최선 로컬 후보는 `P2_RESEARCH_BLEND50.csv`였다.
- 이어서 기존 lean-M2 특징을 고정한 채 shared 대 layerwise 구조를 비교하고, 개발 구간에서 이긴 shared 구조만 Optuna 40 trials로 탐색했다. 최대 boosting round는 5,000, early stopping은 200 rounds였다.
- 개발 score-month RMSE는 1.6402→1.5834℃로 개선됐지만 블록별 best iteration은 91·269·1,249·2,038로 불안정했다. median 759 round로 동결한 guard RMSE는 0.7939→0.8026℃로 악화했고 paired KST-day 90% CI는 [+0.0048, +0.0127]℃였다.
- 이 튜닝 세대는 `REJECT_AND_CLOSE_GENERATION`이다. guard 결과를 보고 trial 수, 탐색 범위, epoch 집계법을 바꾸지 않는다. 이 단계 당시 최선 후보는 `P2_RESEARCH_BLEND50.csv`였다.
- 공개 layer 1과 5의 동시각 수온차 절대값만으로 lean-M2 arm을 혼합·성층 전문가로 나누는 단일 구조 가설도 사전등록 후 1회 검증했다. q40·q60은 각 fold의 학습행에서만 계산했고 중앙 20%를 두 전문가가 공유했으며, 최종 weight는 V0 0.5 + 상태조건 lean 0.5로 고정했다.
- 상태조건 후보는 166,268행 aggregate RMSE를 1.1234→1.0997℃로 낮췄고 90% paired KST-day bootstrap CI는 [-0.0300, -0.0173]℃였다. 8개 블록 중 6개와 세 층 모두 개선했지만, hidden 구간 직전 2025년 7~8월이 +0.0079℃ 악화해 사전등록 veto +0.005℃를 넘었다.
- 따라서 `p2_state_conditional_lean_v1`은 `REJECT_AND_CLOSE_FAMILY`다. q40/q60, overlap, blend weight, 상태신호를 사후 재탐색하지 않는다. 이 단계에서는 제출 후보를 만들지 않았으며 당시 최선은 `submissions/p2/P2_RESEARCH_BLEND50.csv`(SHA256 `4de5027a1ac99fbb58a63da17d96ee3ce1c60204a8d284845da2f33466a977b7`)였다.
- 이후 사용자는 공식 채점 기준인 26,061행 통합 RMSE의 직접 최적화를 우선하고, 개별 계절 블록의 소폭 악화를 자동 탈락 사유로 쓰지 않도록 결정했다. 개별 블록·bootstrap은 순위 해석과 위험 진단으로 유지하되 공식 RMSE proxy 최소화가 주 선택 기준이다.
- 점수 최적화 v1은 hidden 구간과 직접 연결되는 2024년 9~10월, 2025년 7~8월, 2025년 11~12월의 69,850행 pooled RMSE로 8개 phase/state 층별 router를 전수 비교했다. layer 2·3은 M2 phase, layer 4는 상태조건 arm을 쓰는 구조가 RMSE 0.7889℃로 phase 0.8064℃, state 0.7982℃보다 낮았다.
- 관련 3블록 leave-one-block-out router RMSE도 0.7961℃로 두 단일 arm보다 낮았고, phase 대비 paired KST-day ΔRMSE는 -0.0175℃, 90% CI [-0.0230, -0.0122]℃였다. 전체 8블록 평균은 phase가 더 낮으므로 이는 hidden Sep~Oct 전이에 맞춘 target-proxy 선택이며 공식 점수는 아니다.
- 제출 가능한 세 후보를 로컬 동결했다. 1순위 `submissions/p2/P2_SCORE_LAYER_ROUTER.csv` SHA256 `069b782588ccad2a1c74d68586769268b104d686f9dc443f8a8ba136afb192b5`, 전역 RMSE challenger `P2_SCORE_PHASE400.csv` SHA256 `dfa35ecbd11c3fd84cc984c84ceb37826a6d87b35bbf5418ee8f460dac90fba6`, 상태 challenger `P2_SCORE_STATE400.csv` SHA256 `a7fc79442f6f14fcb8575375534e5a4a1813733cf7f92e770880d60b5fdbbf10`이다. 모두 26,061행·제출 스키마·저장 모델 재추론을 통과했으며 업로드하지 않았다.
- 400 rounds가 충분히 수렴했는지 확인하기 위해 특징·LightGBM 파라미터·층별 router·학습행을 그대로 두고 최대 5,000 rounds까지 학습했다. 50·100·150·200·300·400·600·800·1,200·1,600·2,400·3,200·4,000·5,000 checkpoint를 동일 69,850행에서 비교했다.
- router RMSE는 400 rounds의 `0.7888895064`가 최저였고 5,000 rounds는 `0.8665403270`으로 `+0.0776508206` 악화했다. 기존 frozen 400-round OOF와 최대 절대오차 0으로 일치했으므로 모델 변경이 아닌 순수 boosting-horizon 비교다.
- 따라서 현재 learning rate 0.04에서는 **400 rounds를 수렴 최적 checkpoint로 유지**한다. 5,000 rounds 모델은 과적합 진단용이며 제출 우선순위에서 제외한다.
- 제출 1순위는 기존과 byte-identical한 `submissions/p2/P2_SCORE_ROUTER_ROUND400.csv` SHA256 `069b782588ccad2a1c74d68586769268b104d686f9dc443f8a8ba136afb192b5`다. 최대학습 진단 파일 `P2_SCORE_ROUTER_5000.csv` SHA256은 `8284e630ada9eee678a6bdb9b47466b1d36282b13cbe59fec14606d43963fcac`이며 업로드하지 않는다.
- 2026-08-16 구조·대형 모델 문헌 정찰 결과, 다음 1순위는 공개층 수직 set encoder + 61일 양방향 multi-scale TCN + DeepONet식 target-depth query를 결합하고 선형보간 잔차를 공동 출력하는 3–8M parameter 모델이다. 단순히 범용 Transformer의 크기만 키우지 않는다.
- 61일 exact-cadence 창 중 공개층 관측률과 목표 정답 가용률이 각각 95% 이상인 창은 endpoint 19,595개, 6시간 stride 약 545개지만 목표 3층이 모든 시각에서 완전한 61일 창은 0개다. 따라서 deep loss는 관측된 목표 정답에만 적용하고 세 목표층을 같은 중앙 구간에서 함께 가리는 structured mask를 사용해야 한다.
- 첫 deep screen은 AdamW learning rate `{1e-4, 3e-4, 1e-3}` × weight decay `{1e-4, 1e-3}`, 최대 300 epoch, patience 30으로 제한한다. 마지막 epoch가 아니라 최저 validation RMSE checkpoint를 복원하고, 선택된 구조 하나만 3개 seed로 재학습한다.
- 우선순위는 custom depth-query BiTCN → ImputeFormer block-missing benchmark → SSSD-S4/CSDI posterior-mean 상한선이다. MOMENT/UniTS pretrained weight는 실제 관측 사전학습 또는 provenance 불명으로 금지하며, 합성-only 네 조건을 별도로 모두 입증하지 않는 한 사용하지 않는다. 문헌 benchmark 개선율은 P2 기대효과가 아니며 모든 승격은 현재 router의 동일 target-proxy RMSE `0.7888895064`와 비교한다.
- 8개 deep 구조 비교와 finalist 재학습 후, tree·Depth-query BiTCN·3-seed LSTI·3-seed TimeMixer++·local patch proxy의 layer별 convex stack을 만들었다. 동일 69,850행 fitted OOF RMSE는 `0.7458139094`, leave-one-block-out weight RMSE는 `0.7756600313`이었다. 제출 형식 후보 `submissions/p2/P2_DEEP_STACK_V1.csv`는 26,061행과 저장 가중치 재현을 통과했으며 SHA256은 `ea5cedbd08817da4da00274e1078689f09a1d9c65d2a464f5f5f5ba9ffcc82e8`이다. 업로드하지 않았다.
- 이어서 public-only phase 81개 특징과 400 boosting iteration을 고정해 LightGBM GBDT·ExtraTrees·DART, XGBoost hist, CatBoost pooled·layerwise를 비교했다. 단독 최강은 ExtraTrees `0.8160964885`였지만 deep stack과의 LOBO pair를 개선한 계열은 layerwise CatBoost뿐이었다: `0.7756600313→0.7745773144`, Δ `-0.0010827169`℃. 2,000회 KST-day bootstrap 90% CI `[-0.0051327138,+0.0032154573]`이 0을 포함하므로 deep 제출 후보는 유지하고, 다음 파라미터 최적화 대상만 layerwise CatBoost로 좁힌다.
- CatBoost fitted pair 가중치는 layer 2=`0.2808396511`, layer 3=`0.2792596524`, layer 4=`0`이다. 연구 challenger `submissions/p2/P2_DEEP_GBM_RESEARCH_V1.csv`는 제출 스키마를 통과했지만 현재 제출 1순위로 승격하지 않으며 업로드하지 않는다.
- 고정 구조검사의 LOBO 순위 상위 3개인 CatBoost layerwise·CatBoost pooled·LightGBM DART를 동시에 최적화했다. 가족당 총 36 trials를 outer 3개 폴드마다 완전히 독립적인 12 trials로 나눴고, 각 outer 라벨은 자기 파라미터 선택에 사용하지 않았다. CatBoost 최대 3,000 rounds·patience 150, DART 200~3,000 rounds를 비교했다.
- 수렴 checkpoint는 layerwise CatBoost layer 2/3/4=`94/132/16`, pooled CatBoost=`158`, DART=`1,600` rounds였다. 모두 최대 예산 안에서 선택됐으므로 미수렴이 아니라 계절 간 전이 실패로 해석한다.
- 튜닝 outer RMSE는 layerwise CatBoost `0.9006249609`, pooled CatBoost `0.9102760433`, DART `0.9273218809`로 각 고정 구조검사 `0.8335485999`, `0.8434187090`, `0.8297197084`보다 악화했다.
- deep pair LOBO RMSE도 각각 `0.7789892471`, `0.7801178738`, `0.7962604830`으로 frozen deep LOBO `0.7756600313`보다 모두 나빴다. DART fitted pair `0.7437219871`은 좋아 보이지만 LOBO 악화 `+0.0206004517`과 90% CI `[+0.0083892063,+0.0333321716]` 때문에 결합 가중치 과적합으로 기각한다.
- 독립 검산은 세 OOF 각각 69,850개 유일 키·truth·RMSE와 세 standalone 및 연구 pair CSV 각각 26,061행을 재현했다. 따라서 상위 3개 추가 trial 확대를 중단하고, 제출 1순위는 `submissions/p2/P2_DEEP_STACK_V1.csv` SHA256 `ea5cedbd08817da4da00274e1078689f09a1d9c65d2a464f5f5f5ba9ffcc82e8`로 유지한다. 어떤 파일도 업로드하지 않았다.
- 계절 전이 병목을 공개층만으로 재진단했다. 숨은 2025년 9~10월의 `|T1-T5|` 분포는 2024년 같은 계절과의 정규화 Wasserstein 거리가 `0.1575`로 2025년 7~8월의 `2.3188`보다 훨씬 가깝지만, 2024년 같은 계절 삼분위 기준으로 저성층 `17.5%`·전이 `44.7%`·강성층 `37.8%`가 한 구간 안에 공존했다. 따라서 월/계절 hard split은 사용하지 않는다.
- 숨은 8,717개 시각 중 `T1-T5`가 유효한 비율은 `70.5%`뿐이다. 다음 단일 구조 가설은 기존 deep 구성 모델을 재학습하지 않고, `|T1-T5|`, 공개층 온도·염분 범위, 24시간 변화, M2 성분과 결측 mask로 layer별 convex weight를 연속 조정하는 missing-aware soft gate다. target layer temp·psal은 gate에 사용하지 않으며, 기존 deep LOBO `0.7756600313℃`를 같은 69,850행에서 넘어야 한다. 이 진단은 아직 새 모델 성능이나 hidden 점수를 증명하지 않는다.
- 위 public-state soft gate를 2026-08-16에 사전 고정된 정규화 7개 값과 3-block nested LOBO로 한 번 풀테스트했다. outer별 선택값은 `0.001 / 10 / 10`, 최종 중앙값은 `10`이었으며 target layer temp·psal과 외부 관측값은 사용하지 않았다.
- 같은 69,850행에서 frozen deep LOBO `0.7756600313℃` 대비 soft gate는 `0.7877324420℃`로 `+0.0120724106℃` 악화했다. 2,000회 paired KST-day bootstrap 90% CI도 `[+0.0059431046,+0.0189742931]`이고 개선 확률은 `0`이었다. 2024년 9~10월은 `+0.0660527166℃`, 2025년 7~8월은 `-0.0069084331℃`, 2025년 11~12월은 `+0.0008375083℃`였다.
- 층별로 layer 2만 `-0.0012057514℃` 개선했고 layer 3·4는 각각 `+0.0041745221℃`, `+0.0233761680℃` 악화했다. baseline OOF 재현 오차 `0`, 기존 deep test 제출 재현 오차 `3.55e-15`, 저장 gate roundtrip 오차 `0`이므로 구현 재현 실패가 아니라 계절 간 gate 일반화 실패로 판정한다.
- 결론은 `REJECT_KEEP_DEEP_STACK`이다. 연구 파일 `submissions/p2/P2_PUBLIC_STATE_SOFT_GATE_V1.csv`는 제출 형식을 통과했지만 업로드 후보가 아니며, 제출 1순위는 계속 `submissions/p2/P2_DEEP_STACK_V1.csv` SHA256 `ea5cedbd08817da4da00274e1078689f09a1d9c65d2a464f5f5ba9ffcc82e8`이다. 정규화 grid·상태 특징·gate 용량을 이 결과에 맞춰 사후 재탐색하지 않는다.
- 사후 실패 진단은 공개층 상태 조건화 가설 전체가 아니라 **현재 forced prior-anchored linear gate**가 실패했음을 확인했다. 원래 grid에는 기존 deep stack을 정확히 유지하는 no-op arm이 없었고, 3개 outer 중 2개가 최대 정규화 `10`을 골라 사실상 움직임 억제를 요구했다.
- 전체 초과 SSE의 핵심은 2024년 9~10월 layer 4였다. 특히 강성층 셀에서 RMSE가 `1.0646→1.2381℃`로 악화해 전체 초과 SSE의 약 `86.9%`를 만들었다. 오차를 `e`, gate 조정을 `a`라 할 때 `ΔMSE=2e·a+a²`이며, 이 셀은 정렬항도 양수라 조정 방향 자체가 틀렸다. 반대로 2025년 7~8월은 정렬항이 음수여서 gate가 개선했다.
- 물리상태 support가 블록과 심하게 교락됐다. 2024년 같은 계절은 low·transition·high가 고르게 존재했지만 2025년 7~8월은 거의 전부 high, 2025년 11~12월은 `|T1-T5|`가 전부 missing이었다. 세 블록 모두에서 관측된 layer-state 셀 3개 중 같은 최적 contributor가 유지된 셀은 `0`개였다.
- 9개 outer-block×layer 셀 중 8개에서 simplex prior가 최소 한 contributor를 `10^-6` 미만으로 만들었다. `log(prior)`에 고정된 gate는 이 전문가를 사실상 되살리지 못했는데, floored 전문가가 행별 최저오차 모델인 비율이 일부 셀에서 `40~76%`였다. 따라서 전문가 조건화 능력을 온전히 시험하지 못했다.
- 다음 허용 가설은 `safe residual gate` 하나다. 정확한 no-op, 모든 contributor의 양의 weight floor, `deep prediction + bounded correction` 구조, KST-day 균등 가중, 공통 state support 밖 자동 no-op을 동시에 **사전 고정된 안전 구조**로 구현한다. 현 outer 결과를 보고 특징·정규화 grid를 다시 고르는 방식은 금지한다.
- `safe residual gate`는 3-block nested LOBO에서 frozen deep RMSE `0.7756600313℃`를 `0.7756562816℃`로 불과 `0.0000037497℃` 낮췄다. 2,000회 paired KST-day bootstrap 90% CI는 `[-0.0000097680,+0.0000025142]℃`로 0을 포함하므로 검증된 개선으로 보지 않는다.
- 안전장치는 2024년 9~10월과 2025년 7~8월을 exact no-op으로 지켜 기존 catastrophic regression은 막았지만, outer OOF에서는 8.06% 행만 보정한 반면 full-train calibrator는 test 행의 66.77%(CSV 실변경 68.40%)를 바꿨다. incumbent 대비 test 보정 최대치는 `0.208832℃`로, 검증 이득에 비해 deployment 개입 범위가 지나치게 크다.
- 따라서 연구 파일 `submissions/p2/P2_SAFE_RESIDUAL_GATE_V1.csv` SHA256 `4404eda36ae0f33a238212b5435f15da71f2df7f5994d2532ca25ad9d5e1df40`는 **제출 금지**다. 이 gate 계열을 사후 임계값 조정으로 구제하지 않으며, 제출 1순위는 계속 `submissions/p2/P2_DEEP_STACK_V1.csv` SHA256 `ea5cedbd08817da4da00274e1078689f09a1d9c65d2a464f5f5f5ba9ffcc82e8`이다.
- 이어서 동결 Deep Stack의 세 목표층을 동시각 공개 layer 1·5 수온 사이로 clip하고 endpoint 방향에 맞게 unit-weight isotonic projection하는 단일 물리 구조를 검증했다. endpoint가 하나라도 결측이거나 목표 세 층 예측이 모두 없으면 exact no-op이며, target layer 값·hidden 정답·외부 관측값은 변환에 사용하지 않았다.
- 69,850행 LOBO RMSE는 `0.7756600313→0.7744179316℃`로 `-0.0012420998℃` 개선됐다. 2,000회 paired KST-day 90% CI는 `[-0.0017287565,-0.0007588605]℃`, 개선 확률은 `1.0`이었다. 2024년 같은 계절은 `-0.0055785101℃`, 2025년 7~8월은 `-0.0000358932℃`, endpoint가 전부 없는 2025년 11~12월은 exact no-op이었고 세 목표층이 모두 개선됐다.
- hidden test에서는 70.44%가 endpoint-eligible이고 31.55% 행이 실제 변경된다. 같은 계절 OOF의 active share 46.16%보다 낮으며 저장 CSV 재현 최대오차는 `3.55e-15`다. 다만 order/clip/both 정찰 뒤 선택한 adaptive research라서 공식 개선을 증명하지는 않는다.
- 현재 로컬 제출 1순위는 `submissions/p2/P2_PHYSICAL_PROFILE_PROJECTION_V1.csv` SHA256 `fd803053475f1060f620861c0a79b7d303df7561ef4e75a70792286d58dc2ca6`로 승격한다. `P2_DEEP_STACK_V1.csv`는 exact no-op fallback·비교 기준으로 함께 동결한다. 두 파일 모두 26,061행 제출 validator를 통과했고 업로드하지 않았다. 보고서는 `reports/generated/p2_physical_profile_projection_2026-08-16/report.html`에 있다.
- 문제에서 제공한 가림 구간 밖 layer 2·3·4 수온을 입력하는 장기 structured-mask BiTCN도 별도 구현했다. 세 목표층을 7일·30일·61일 동안 동시에 가리고, 90일 hourly window와 14일 양쪽 문맥을 사용했다. 개발 블록에서 learning rate `{1e-4,3e-4,1e-3}`와 최대 120 epoch를 비교해 `3e-4`, 35 epoch를 동결했다.
- 이 모델의 standalone hourly RMSE는 2024년 9~10월 `0.6206`, 2025년 7~8월 `1.1232`, 2025년 11~12월 `0.7257`이었고, 동결 Deep 보정 방향은 전체에서 역상관이었다. LOBO와 전체 OOF 보정 가중치가 모두 exact no-op `0`을 골라 `p2_structured_mask_imputer_v1`은 기각한다. 이는 목표층 경계 관측이 무용하다는 결론이 아니라, 현재 61일 직접 외삽 TCN이 공개층 Deep 예측보다 안정적이지 않다는 결과다.
- 안전장치로 거의 no-op이 됐던 public-state soft gate의 **raw cross-fitted expert**를 다시 분해했다. layer 2·4만 raw expert로 교체하고 layer 3은 물리투영 Deep을 유지한 뒤, 공개 layer 1·5 envelope/order로 투영한다. 이후 base에서 routed profile 방향으로 고정 배율 `2.0`만큼 이동하고 같은 물리투영을 다시 적용한다. 배율은 노출된 OOF의 `0~4`, 간격 `0.25` 탐색에서 골랐으므로 adaptive research이며 fresh holdout 또는 공식 점수를 주장하지 않는다.
- 이 `p2_extrapolated_soft_gate_v1`의 69,850행 로컬 RMSE는 `0.7744179316→0.7693416925℃`, Δ `-0.0050762391℃`였다. 2,000회 paired KST-day bootstrap 90% CI는 `[-0.0085891802,-0.0015931192]℃`, 개선 확률은 `0.9945`였다. 2024년 같은 계절 `-0.0010947088℃`, 2025년 7~8월 `-0.0096893786℃`, 2025년 11~12월 `+0.0012951421℃`다. 공식 통합 RMSE 직접 최적화 우선 원칙에 따라 이 작은 후반기 회귀는 진단으로 남기고 전체 후보는 승격한다.
- 현재 로컬 제출 1순위는 `submissions/p2/P2_EXTRAPOLATED_SOFT_GATE_V1.csv`, SHA256 `1149c435e32c2f1c558be1c03857e7d2afd551dc9123f7313b1497cce950d9c3`이다. 26,061행·키 순서·유한 temp·범위 검증을 통과했다. 이전 `P2_PHYSICAL_PROFILE_PROJECTION_V1.csv`는 2순위 안정형 fallback, `P2_DEEP_STACK_V1.csv`는 3순위 원모델 fallback으로 동결한다. 어떤 파일도 업로드하지 않았다.
- v1 이후 층별 보정 방향을 2차원으로 검사했다. routed-base 보정의 block별 least-squares 최적 배율은 layer 2에서 `9.78 / 17.02 / 7.74`로 세 블록 모두 같은 양의 방향이었고, layer 4는 후반 혼합 블록에서만 반대였다. layer 2는 같은 계절 최적값을 반올림한 `10`, layer 4는 v1의 pooled 최적 `2`, layer 3은 `0`으로 고정한 v2를 만들었다.
- `p2_extrapolated_soft_gate_v2`의 로컬 RMSE는 `0.7744179316→0.7683674566℃`, Δ `-0.0060504749℃`다. paired KST-day bootstrap 90% CI `[-0.0099752579,-0.0023072329]℃`, 개선 확률 `0.998`이다. 같은 계절 `-0.0012507915℃`, 2025년 7~8월 `-0.0114692737℃`, 2025년 11~12월 `+0.0012574209℃`이며 layer 2·4는 각각 `-0.0080311231℃`, `-0.0100126477℃` 개선됐다. 이는 노출 OOF 기반 adaptive v2이며 공식 성능이 아니다.
- 현재 로컬 제출 1순위를 `submissions/p2/P2_EXTRAPOLATED_SOFT_GATE_V2.csv`, SHA256 `1c959f818737850fd7fa9c6609ba3ae49dc9a470a269f7313119d840df1736bf`로 갱신한다. v1은 실험 계보로만 보존하고 실제 fallback 순위는 physical projection, 원 Deep Stack 순이다. 보고서는 `reports/generated/p2_extrapolated_soft_gate_2026-08-16/report.html`이며 어떤 파일도 업로드하지 않았다.

## 8. 원본·Git·제출 금지선

- `C:\Users\cedis\Downloads\p2`와 그 안의 ZIP/CSV/README/score.py는 읽기 전용이다.
- 원본, 캐시, 모델, OOF, 제출 CSV를 Git에 넣지 않는다.
- 코드에 개인 절대경로를 하드코딩하지 않는다. `P2_DATA_DIR` 또는 `--data-dir`를 사용한다.
- 사용자의 정확한 파일 승인 없이는 플랫폼 업로드를 실행하지 않는다.
- 제출 전 배포 `score.py`의 입력 계약과 별도 로컬 validator를 모두 통과하고 SHA256을 제시한다.
