# P1 관측 정점·GitHub 참가자 역정찰·P2 우선순위 보고서

작성일: 2026-08-27 KST  
팀: 분당독고다이  
상태: `RESEARCH_COMPLETE_NO_UPLOAD`  

## 결론

P1의 MS-TCN e150은 이 모델 계열의 최대값으로 입증된 것이 아니다. 확인된 사실은 **시험한 체크포인트 중 e150이 최고였고 Public에서도 기존 최고를 갱신했다**는 것뿐이다. 그러나 학습 곡선이 비단조이고 fold별 부호가 갈렸기 때문에, epoch만 더 늘리는 방법의 기대값은 낮다. 다음 큰 개선은 학습시간보다 오류 집합·station/segment proposal·calibration 구조에서 찾아야 한다.

GitHub에서는 2026 대회명, 문제의 고유 폴더명, 정확한 문제명, 관측소 조합과 알려진 팀명을 검색했지만 현재 대회 참가자의 재사용 가능한 공개 저장소를 찾지 못했다. 검색 결과에 나온 `JOISS` 저장소들은 과거의 다른 대회였다. 이는 참가 코드가 없다는 증명이 아니라 **공개·색인된 코드가 아직 발견되지 않았다는 결과**다.

다른 문제의 우선순위는 P2다. 현재 관측 리더보드 기준 P2 headroom은 약 `+2.001464점`, P3는 약 `+0.958814점`이다. 다만 이번에 복구한 P2 checkpoint-0.85 표면은 흥미로운 로컬 신호가 있어도 원 사전등록이 candidate/test prediction과 upload를 명시적으로 금지한다. 생성된 파생 CSV는 감사용으로 격리했고 제출 자격을 부여하지 않았다. 공식 업로드는 수행하지 않았다.

## P1: 최고점인가

### 공식 관측

| 후보 | Public F1 | 문제 점수 | 해석 |
|---|---:|---:|---|
| 이전 Router 최고 | 0.817873 | 28.492736 | 비교 기준 |
| MS-TCN e150 전 정점 결합 | **0.833248** | **28.901363** | 새 공식 최고 |
| MS-TCN e150 G·S 결합 | 0.822488 | 28.615402 | 개선이나 full 미달 |
| GI 무제거 | 0.817968 | 28.495264 | 사실상 이전 최고와 동률 |

e150 full의 개선량은 F1 `+0.015375`, 문제 점수 `+0.408627`이다. 실제 공식 개선이므로 모델 계열이 무의미하다는 결론은 틀리다.

### 왜 최대값은 아닌가

- e145 대비 e150의 pooled local delta가 `-0.000485 → +0.003887`로 다시 상승했다. 마지막 지점이 하강 중이라는 증거가 아니다.
- 반면 분기별 delta는 Q3 `+0.017209`, Q4 `-0.015441`이고 90% 구간 `[-0.01315, +0.02114]`가 0을 가로지른다.
- 따라서 e150은 `observed best checkpoint`이지 수렴 최적점이나 global optimum이 아니다.
- 더 긴 학습만으로 `+3점`을 기대하는 것은 근거가 약하다. 현재 신호는 epoch 수보다 시간구간·정점에 따른 오류 이질성이 크다는 쪽을 지지한다.

## GitHub 참가자 역정찰

### 검색 범위

- 정확한 대회명: `2026년도 해양과학 AIx빅데이터 경진대회`
- 고유 데이터 폴더: `P1_qc_anomaly`, `P2_profile_restore`, `P3_wave_forecast`
- 정확한 문제명: `종합해양과학기지 관측 수온 자동 품질관리`
- 고유 컬럼/관측소 조합: `S-ORS I-ORS G-ORS`, `C0001 lead_h hs_pred`
- 알려진 팀명과 `oceanaidata`
- GitHub 웹 검색, 검색엔진의 `site:github.com`, 인증된 GitHub 코드 검색을 교차 사용

### 결과와 한계

현재 대회와 일치하는 공개 경쟁자 저장소는 발견되지 않았다. 검색엔진이 반환한 [제2회 JOISS 저장소](https://github.com/snghyun331/contest-JOISS)와 [2021 JOISS 저장소](https://github.com/worldpapa/joiss)는 문제·연도·데이터가 다른 과거 대회이므로 제외했다. 우리 저장소 외에 대회 고유 키워드를 포함한 저장소도 확인하지 못했다.

이 결론에는 세 한계가 있다.

1. 비공개 저장소, 아직 push되지 않은 코드, 검색 색인 지연은 관찰할 수 없다.
2. 참가자가 문제명을 쓰지 않고 일반 이름으로 공개하면 exact-key 검색에 잡히지 않는다.
3. GitHub 코드 검색 API의 마지막 질의는 rate limit에 도달해 반복하지 않았다. 앞선 exact-key 질의들은 결과가 없었다.

따라서 GitHub 역정찰은 계속할 가치가 있지만, 현재 즉시 가져올 수 있는 경쟁자 구조는 없다. 하루 한 번 정도 새 저장소만 확인하고 주 연구축으로 삼지는 않는 것이 적절하다.

## 유사 공개 구조에서 얻은 후보

- [DepthDif](https://github.com/simon-donike/DepthDif)는 희소 해양 수온 관측을 dense depth field로 복원하는 conditional diffusion 구조다. surface channel, sparse support, 좌표·날짜 context를 조건으로 쓴다는 점이 P2의 장기 구조 후보와 맞는다.
- [DIRECT](https://github.com/G-Rovscek/DIRECT)는 sparse satellite observation에서 dense SST를 복원하는 diffusion 계열 공개 구현이다. P2와 공간·관측 형태가 달라 직접 이식보다 masked reconstruction objective의 참고 가치가 크다.
- [GRIN](https://github.com/Graph-Machine-Learning-Group/grin)은 multivariate time-series imputation을 graph neural network로 구성한다. P2의 층간·변수간 관계를 명시적으로 표현할 수 있으나, 단일 관측소의 수직층 문제에 적용할 graph 정의가 먼저 필요하다.
- [ImputeFormer](https://doi.org/10.1145/3637528.3671751)는 low-rankness와 temporal modeling을 결합한 결측 시계열 복원 구조다. 장기 결측에서 단순 TCN의 receptive-field 한계를 검증할 차기 구조 후보로 적절하다.

이 구현들은 참가자 코드가 아니며, 외부 데이터·사전학습 가중치의 대회 허용성과 동일한 것도 아니다. 구조 아이디어만 취하고 대회 원본 train으로 새로 학습해야 한다.

## P2 체크포인트 재검토

checkpoint-0.85의 연구용 OOF 결과는 r3 final 대비 aggregate RMSE `-0.047112℃`였다. 고정 α=1에서 세 outer fold 모두 개선했지만, full fraction의 checkpoint는 r3보다 `+0.010618℃` 악화했다. 0.85 선택은 이미 노출된 outer 결과를 사후 확인한 표면이므로 fresh confirmation이 아니다.

더 중요한 것은 원 설정의 권한 경계다.

- `candidate_or_test_prediction_allowed: false`
- `upload_allowed: false`
- `research_only: true`
- `exact_official_incumbent_comparison: false`

이 경계를 재확인한 뒤 파생 bundle을 `QUARANTINED_PROTOCOL_VIOLATION_NOT_SUBMISSION_ELIGIBLE`로 바꾸고 builder를 fail-closed로 수정했다. 기술 QA는 행 `26,061`, Layer 4 `8,636`, 키·순서·hash·물리 범위·Layer 2/3 보존을 모두 통과했지만, 이는 제출 자격을 회복하지 않는다.

| 격리 파일 | SHA-256 | 역할 |
|---|---|---|
| L4 50% blend | `a4482f37cbeb45c306a496ad149f68cc53435dcaf74206691d8b2f3cb3cf6473` | 감사 전용 |
| L4 full | `5ef474790ebb126b86a6be0ac7265f3846f9d594e117bca93f72c87944a3005b` | 감사 전용 |
| Axis U + L4 full | `0f0b7d14643bed9f678805ebb878cf8f408056a62bed8f405e2b42c4e72fdcd3` | 감사 전용 |

공식 업로드는 없었고 숨은 target 값도 읽지 않았다.

## 다음 연구 결정

1. P1은 e150 주변 epoch를 더 촘촘히 돌리는 것을 주축으로 삼지 않는다. 같은 backbone의 proposal/calibration을 바꾸거나 오류가 다른 새 구조를 만든다.
2. 다음 full-scale 계산은 P2에 배정한다. checkpoint-0.85를 그대로 쓰지 않고, **새 사전등록·새 seed·fresh temporal window·전체 train 최종 refit**을 갖춘 Layer-4 구조를 처음부터 실행한다.
3. P2 새 구조는 incumbent 보존 blend를 포함하되, 학습 전에 α grid와 official probe 역할을 고정한다. exposed checkpoint 결과는 설계 근거일 뿐 선택 점수로 재사용하지 않는다.
4. P3는 ERA5 고정 실험이 끝나기 전 구조를 바꾸지 않는다. 현재 입증된 headroom과 다운로드 비용을 고려하면 P2가 우선이다.
5. 총 `+3점`은 현재 증거상 한 모델의 단순 추가 학습보다 P1 구조 개선 + P2 full-scale 개선 + P3 제한적 보완의 합으로 접근해야 한다.

## 주장–근거 원장

| 주장 | 근거 | 위치/URL | 신뢰도·주의 |
|---|---|---|---|
| P1 e150 공식 최고 및 세 후보 점수 | Round F 공식 기록 | `C:/Users/cedis/Downloads/해양 해커톤 제출용/20260827_round_F_mstcn_e150_P1x3/OFFICIAL_RESULTS.md` | 높음; 인증 UI 수기 보존 |
| P2/P3 현재 공식 최고 | Round D 공식 결과 | `C:/Users/cedis/Downloads/해양 해커톤 제출용/20260826_round_D_preregistered_P1x3_P2x3_P3x3/OFFICIAL_RESULTS_20260826.json` | 높음; 당시 스냅샷 |
| P2/P3 관측 headroom | 공식 리더보드 정찰 보고서 | `reports/official_probe_value_deep_research_20260827_v1/report-source.md`; [공식 리더보드](https://oceanaidata.org/app/leaderboard) | 시점 의존 |
| local–official sign/magnitude 불일치 | calibration 원장 | `reports/next_day_breakthrough_deep_research_20260827_v1/local_official_calibration.json` | family 상관·소표본 |
| checkpoint-0.85와 full 비교 | checkpoint metrics | `artifacts/p2_joint_hydrographic_multitask_layer4_checkpoint_v1/metrics.json` | exposed research surface |
| checkpoint 연구 경계 | frozen config | `configs/experiments/p2_joint_hydrographic_multitask_layer4_checkpoint_v1.json` | 명시적 fail-closed 경계 |
| 격리 bundle 기술 무결성 | 독립 QA | `artifacts/p2_checkpoint85_layer4_deployment_v1/INDEPENDENT_QA.json` | 제출 자격과 별개 |
| 공개 참가 repo 미발견 | exact-key GitHub/web 검색 | 본 보고서 검색 범위; 2026-08-27 접근 | 부재의 증명 아님 |
| 해양 희소 복원 diffusion | DepthDif 공식 저장소 | https://github.com/simon-donike/DepthDif | 유사 구조, 대회 참가 코드 아님 |
| sparse SST diffusion | DIRECT 공식 저장소 | https://github.com/G-Rovscek/DIRECT | 관측 형태 차이 |
| graph imputation | GRIN 공식 저장소 | https://github.com/Graph-Machine-Learning-Group/grin | graph 설계 필요 |
| low-rank temporal imputation | ImputeFormer 논문 | https://doi.org/10.1145/3637528.3671751 | 구현·대회 적합성 별도 검증 필요 |

## 중단 기준

대회 고유 GitHub 검색은 exact-key 결과가 없고 마지막 API rate limit까지 확인했으므로 이번 사이클에서 중단했다. 유사 구조는 세 계열로 수렴했고, 현재 의사결정은 P2의 합법적인 fresh full-scale 실험을 새로 설계하는 것이다. 추가 일반 문헌 검색은 이 우선순위를 바꿀 가능성이 낮다.
