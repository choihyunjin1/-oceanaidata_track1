# P1 고용량 MS-TCN++/ASRF v2 — 근거 간극 행렬

작성일: 2026-08-27  
용도: 사전등록 연구 주장과 종료 후 로컬 증거를 분리한다. 원래의 결정·중단 기준을 보존하고 실제 결과와 최종 판정을 오른쪽 두 열에 닫았다.

| 주장/결정 | 최상위 근거 | 사전 신뢰 | 반증·충돌 가능성 | 사전등록된 다음 증거·중단 조건 | 실제 종료 증거 | 최종 판정 |
|---|---|---|---|---|---|---|
| 다단계 dilated TCN과 smoothing은 긴 문맥 및 과분할 감소에 적합하다 | MS-TCN CVPR 2019, MS-TCN++ TPAMI 2020 및 공식 구현 | 구조 수준 높음 | 비디오 다중 클래스에서 해양 희소 이진 QC로의 도메인 이동 | Q2에서 추가 precision·FN recovery gate를 통과하지 못하면 즉시 중단 | Q2 선택 ΔF1 `+0.0981572491`이었으나 고정 확인은 Q3 `+0.0132993769`, Q4 `-0.0314844165`, pooled `-0.0051401087` | 구조 근거는 유지; P1의 안정적 개선 가설은 이번 recipe에서 반증 |
| 경계 head가 장기 사건의 시작·종료를 보조할 수 있다 | ASRF WACV 2021 및 공식 구현 | 구조 수준 중간~높음 | P1 경계 양성의 희소성, 행 F1과 segment F1의 불일치 | Q2 boundary 진단과 Router-union F1가 함께 개선되지 않으면 중단 | pooled FN `287`행 회복과 동시에 FP `466`행 증가; 추가 행 precision `0.3811420983`. 독립 boundary ablation 없음 | 성능 기여 미입증; 복합 모델 결과를 ASRF head에 귀속 금지 |
| width 256/512, 300 epoch가 탐색할 가치가 있다 | 공식 MS-TCN++는 width 64·100 epoch; 본 설정은 계산 측정과 사용자 목표에 따른 외삽 | 낮음~중간 | 용량 증가·장기학습이 과적합 또는 포화될 수 있음 | 1–300 checkpoint 곡선, 3 seed 평균, Q3/Q4 재현성으로만 채택; epoch 300 선택 시 우측 검열 명시 | Q2 6개 fit 모두 300/300, nonfinite 0; width `512`, epoch `125` 선택. Q3/Q4 확인은 실패 | 탐색 가치 검증 완료; 고용량 우월성·수렴 주장은 기각, 후보 `NO_GO` |
| 시간 순서 보존 검증이 무작위 분할보다 적절하다 | Cerqueira et al., Machine Learning 2020 | 원칙 수준 높음 | 본 문제의 series-local fold와 21일 gap/block 길이는 직접 검증되지 않음 | 16개 공통 series 전부 Q2<Q3<Q4, exact-key overlap 0을 실행 전 fail-closed 검증 | 3개 split 모두 `cross_split_window_count=0`, holdout fit/train/truth 선사용 0; Q3·Q4 blind seal과 semantic replay 모두 PASS 후 metric 공개 | 실행 무결성·검증 절차는 PASS; 이 절차가 낙관적 Q2 후보를 탈락시킴 |
| 3 seed 평균은 단일 seed보다 과신을 줄인다 | Bouthillier et al., MLSys 2021 | 방향 수준 높음 | 3 seed는 분산의 충분한 추정이 아님 | best-seed 선택 금지, 6개 Q2 fit 모두 성공해야만 후보 평가 | Q2 6개와 confirmatory 6개 history 완주, 모두 nonfinite 0; 선택·확인은 3-seed raw 평균만 사용 | 절차 가설은 충족; 분기·station 안정성 보장은 여전히 없음 |
| 공식 +3점에는 F1 0.930749 이상이 필요하다 | 2026-08-26 공식 3개 P1 결과에서 관측된 선형 점수 기울기 | 해당 채점식 구간에서 높음 | 채점식이 다른 구간에서 비선형·변경될 가능성 | 로컬 gate는 endpoint와 delta를 동시에 사용; 실제 공식 +3은 별도 승인된 공식 결과로만 확인 | pooled local F1 `0.8977769149`, high-impact gate FAIL. 공식 probe·submission 생성·업로드 0 | 이번 후보는 +3 주장 불가·공식 제출 불가; 공식 임계 자체는 미검증 상태 유지 |
| 로컬 개선이 공식 개선 방향을 예고할 수 있다 | 과거 Router: local +0.00222991, official +0.024163 | 방향만 낮음~중간 | 관측 쌍이 극소수이고 효과 크기 비율 10.84배로 불안정 | 로컬 크기를 공식 점수로 환산하지 않음; 로컬은 후보 배제/승격 증거만 제공 | 이번 후보는 확인 pooled ΔF1 음수여서 로컬 단계에서 배제됐고 공식 결과 쌍을 추가하지 않았다 | transport calibration 간극은 해소되지 않음; 음의 로컬 결과를 공식 점수로 환산 금지 |
| incumbent-preserving OR는 기존 Router 양성을 보호한다 | 집합 연산 및 실행 invariant | 수학적으로 높음 | 추가 FP가 많으면 F1은 하락 가능 | removed-positive=0을 필수 invariant로, 추가 precision·FP ratio를 gate로 사용 | pooled `anchor_positive_removed_rows=0`; 후보 F1은 `0.8977769149`로 anchor `0.9029170235`보다 낮음 | 양성 보존 invariant 확인; F1 보장 부재도 실증 |

## 종료·무결성 영수증

- 최종 상태: `NO_GO_CONFIRMATORY`; research gate FAIL, high-impact official-probe gate FAIL.
- 90% bootstrap CI: `[-0.0281000264, 0.0140666998]`; 개선 station `1/3`.
- terminal SHA-256: `7640cc0e29f364a26cd8199a7e9a55acdf329699cd5923679d8f0d513c4af2b1`.
- confirmatory metrics SHA-256: `964cae7d7dbb9f413244462eb9258e883e14f6073ad5549fadf482cb9cd03bd4`.
- Q2/Q3/Q4 blind NPZ SHA-256: `867d4b25d968ce4231179181e05eee95cda04d76689c03c1514b907e21dd1f02`, `afc1f10d0ee6bb3ab1896f8a579a074ec386c73639f08d778355894b7d7347a9`, `f04f3d4cb22e75a8ec69ebed1656b618231f9a04552dcb5332670a2594dc948a`.
- 독립 read-only QA: exact inventory `40/40`, assertion 137개 PASS, `P0=0`, `P1=0`, `P2=2`의 `PASS_CONTROLLED`. P2는 실행 경계 밖의 사전 명시 잔여 위험이며 관측된 변조가 아니다.
- 공식 test/sample/submission은 접근하지 않았고, `submission_created=false`, `upload_performed=false`, 공식 `+3` 주장도 생성하지 않았다.

## 연구 중단 판단

구조 선택에 관한 핵심 주장은 1차 논문 또는 공식 구현으로 채워졌고, 봉인된 Q2 선택과 Q3/Q4 확인도 끝났다. 이번 recipe의 P1 전이 성능은 `NO_GO`로 닫혔다. 같은 산출물의 결과 기반 재튜닝·재제출은 하지 않으며, 다음 연구가 있다면 Q4 및 S-ORS/I-ORS FP 이동을 설명하는 별도 사전등록 가설과 새로운 confirmatory 분할이 필요하다. 문헌은 구조의 가능성을 지지했지만 로컬 확인 실패를 뒤집는 증거가 아니며, 로컬 결과도 공식 대회 `+3`의 증거가 아니다.
