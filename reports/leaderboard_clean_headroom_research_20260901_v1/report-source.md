# 과거 최고점·clean headroom 재검산 및 구조 전환 연구

## 결론

과거 제출과 현재 로컬 후보를 다시 대조해도 **누락된 더 높은 공식 최고점은 없다.** P1의 규정 적합 공식 최고는 F1 `0.833548` / `28.909341점`, P2는 RMSE `0.424019℃` / `28.012945점`, P3는 외부자료 계보를 제외하면 RMSE `0.583892m` / `24.066168점`이다. P3 화면에 남아 있는 `24.203599점`은 KMA 외부자료 계보이므로 재현 가능한 clean 최고로 취급할 수 없다.

문제별 최고와의 차이는 P1 `3.201112점`, P2 `0.661957점`, clean P3 `0.717875점`이다. 저장된 공식 영수증들로 선형 환산을 재적합하면 필요한 원지표 변화는 대략 P1 `+0.120442 F1`, P2 `-0.052765℃ RMSE`, P3 `-0.045231m RMSE`다. 이는 주최 측 공식식을 단정한 값이 아니라 **계획용 경험 환산**이다.

따라서 P1은 기존 add-only 행·미세 특징·guard 완화로는 격차를 닫을 수 없고, P3도 잔차 기술자 0.001~0.005m 개선 축으로는 부족하다. 이번 사이클에서 P3의 새로운 fractional-change 목표를 올바른 clean 분할로 실제 3회 학습했지만 RMSE가 `0.779105→0.780006m`로 `+0.000901m` 악화했다. 이 가설은 종료한다. 다음 고가치 축은 명시적으로 합성 전용인 TabPFN-2.6 계열을 P1 station×layer별 분류와 P3 lead별 회귀로 검증하는 것이다. 현재 패키지와 로컬 가중치가 없고 라이선스 수락도 사용자 행위가 필요하므로 자동 다운로드나 실행은 하지 않았다.

## 과거 포함 최고점 확인

| 문제 | 우리 공식 최고 | 문제 최고 | 점수 차이 | 계획용 원지표 차이 | 판정 |
|---|---:|---:|---:|---:|---|
| P1 | F1 `0.833548`, `28.909341` | `32.110453` | `3.201112` | leader F1 약 `0.953990`, 차이 `+0.120442` | 과거 최고 맞음; 더 높은 미제출 promotable 후보 0 |
| P2 | RMSE `0.424019℃`, `28.012945` | `28.674902` | `0.661957` | leader RMSE 약 `0.371254℃`, 차이 `-0.052765℃` | v52가 과거 포함 새 최고 |
| P3 clean | RMSE `0.583892m`, `24.066168` | `24.784043` | `0.717875` | leader RMSE 약 `0.538661m`, 차이 `-0.045231m` | 과거 clean 최고 맞음 |
| P3 화면 표시 | RMSE `0.575233m`, `24.203599` | `24.784043` | `0.580444` | 규정 부적합 KMA 계보 | 사용·재현·승격 금지 |

P1 공식 장부에는 2026-08-25 이후의 주요 제출 19건이 SHA와 함께 남아 있고, 플랫폼 전체 이력 확인에서는 22건을 대조했다. 별도 v54 감사는 470개 P1 메타 문서, 155개 experiment ID, 149개 result를 훑었다. 내부 PASS 또는 양의 결과가 있던 9개 계보 중 실제 gate를 통과한 3개는 모두 이미 제출된 SHA와 일치했고, 미제출·비중복 승격 후보는 0개였다.

P2는 v52가 직전 최고 v23보다 공식 RMSE `-0.000957℃`, 점수 `+0.012006`을 기록해 현재 최고다. v52 이후 새 P2 terminal 후보는 발견되지 않았다.

P3의 규정 적합 계보는 외부자료 첫 사용 시점 이전으로 절단했다. 그 구간의 공식 RMSE는 `0.607071→0.599072→0.583892m`로 개선됐고, 마지막 후보가 clean 최고다. 외부 KMA/ERA5/Chronos와 그 후손은 점수가 더 높아도 현재 규정 아래 비교 참고일 뿐 제출 자산이 아니다.

## P1 격차가 큰 이유와 해결 방향

현재 챔피언에 공식적으로 식별된 양의 성분은 이미 모두 포함돼 있다. router 대비 원시 e150, I-ORS 80행, G-ORS 15행, GI 2행은 제거 probe에서 방향이 재확인됐다. S-ORS 238행은 표시상 순효과가 0이지만, 그 부분집합을 고를 수 있는 독립 신호와 fresh holdout이 없다. HistGBDT 4행 추가도 공식 동률이었다.

핵심 병목은 모델 용량보다 **사건 유형과 배포 운송**이다. 내부 Q2-Q4에서는 pooled F1이 약 0.860 수준이어도 공식은 0.833548에 머물렀고, offset/drift와 I-ORS L1, S-ORS L2가 반복 취약했다. 2025Q2/Q3/Q4는 이미 모두 가설에 노출됐으며, 2024는 단일 관측소라 새 다중 관측소 운송 검증이 불가능하다. guard를 무작정 풀면 과거 v28처럼 162,615개 기존 음성행을 전부 양성으로 만드는 붕괴도 있었다. 따라서 같은 feature head에 threshold를 다시 맞추는 것은 해법이 아니다.

다음 모델은 **station×layer별 TabPFN-2.6 classifier**로 고정한다. Q3+Q4의 16개 station-layer cell은 셀당 `7,334~23,279`행이고 165개 past-only 특징이므로 TabPFN-2.6의 공식 권장 범위인 100,000행·2,000특징 안에 들어간다. 각 outer prefix에서 해당 셀만 in-context training set으로 사용하고, 버전·체크포인트·n_estimators·0.5 threshold·anchor-union decoder를 사전 고정한다. 기존 149개 결과와 의미가 다른 모델 계열이며, 사용자 라이선스 수락과 정확한 synthetic-only checkpoint SHA 확보 후에만 실행한다.

동시에 라이선스와 무관한 후순위 축으로는 행 분류가 아니라 **다변량 변화점/사건 구간 제안**을 유지한다. changeforest와 multivariate change-point 연구는 정상구간과 사건구간의 분포 변화를 직접 찾는 근거를 제공하지만, 기존 P1 long-event 실험의 exact target contract는 single-class로 닫혔다. 후속은 그 artifact를 재실행하지 않고, train prefix의 실제 연속 양성 run을 segment label로 정의하는 별도 계약이어야 한다.

## P3 격차와 이번 실행 결과

P3 전체 최고도 33.333점에 가깝지 않다는 사실은 “모두 실패”라기보다 점수 변환이 native RMSE 차이를 압축해 보이게 한다. 문제 최고의 경험 환산 RMSE는 약 `0.538661m`, clean 우리 최고는 `0.583892m`다. 필요한 감소율은 약 `7.75%`다. 0.001m 개선은 약 0.016점 수준이라, 현재 0.718점 clean gap을 닫으려면 한 단계 큰 backbone 변화가 필요하다.

이번 `p3_clean_fractional_change_residual_20260901_c1r1`은 이전 c1의 0-fit split 오류만 고쳤다. 181개 독립 case, 1,086행, 78시간 간격, 72시간 footprint 비중첩을 기존 validation-key SHA와 정확히 맞춘 뒤 scratch CatBoost 3회를 실행했다.

| 지표 | clean fallback | fractional candidate | 변화 |
|---|---:|---:|---:|
| pooled RMSE | `0.779104840m` | `0.780006025m` | `+0.000901185m` |
| 개선 fold | - | `1/3` | 최소 `2/3` 실패 |
| I-ORS RMSE 변화 | - | `+0.002254575m` | 악화 |
| lead 18/24 변화 | - | `+0.001916689m` | 악화 |

결론은 `COMPLETE_NO_GO_CLEAN_FRACTIONAL_CHANGE`다. 3/3 fit, 191.47초, OOF 1,086행, 중복 0, 유한값 PASS, 공식/test/sample/hidden/CSV/upload 접근 0을 독립 재계산했다. 같은 target·weight·split을 재실행하지 않는다.

P3의 다음 backbone은 **lead별 TabPFN-2.6 regressor**다. 24,360개 anchor와 591개 clean feature를 lead별로 유지하면 권장 범위 안에 들어가며, target은 기존과 동일한 `target-current` residual로 둔다. 6개 lead 모델의 direct prediction을 clean fallback과 고정 0.25로 혼합하고, corrected 3-fold 181-case 표면에서 한 번만 검증한다. P3의 이미 실행된 component-loss soft router와 lead-continuous ridge는 반복하지 않는다.

## TabPFN 규정·실행 preflight

공식 저장소는 TabPFN-2.6 기본 모델이 합성 데이터만으로 학습됐다고 명시하며, 2.6의 권장 범위를 100,000행·2,000특징으로 안내한다. 모델 파일은 첫 사용 시 내려받지만 오프라인에서는 명시적 `model_path`로 로드할 수 있다. 2.5/2.6 가중치는 비상업 라이선스 수락이 필요하다. 현재 기본 버전은 3으로 바뀌었으므로 `TabPFNClassifier()` 같은 자동 기본값은 금지하고 `ModelVersion.V2_6` 및 정확한 로컬 체크포인트를 고정해야 한다.

현재 환경은 Python `3.12.10`, PyTorch `2.13.0+cu130`, RTX 5090 `31.84GB`로 계산 자원은 충분하다. 하지만 `tabpfn` 패키지 미설치, `%APPDATA%\\tabpfn` 캐시 부재, 수락된 라이선스/토큰/체크포인트 SHA 부재다. 라이선스 수락은 사용자 행위이므로 자동으로 대신할 수 없다. 다음 실행 조건은 아래 네 가지다.

1. 사용자가 Prior Labs에서 TabPFN-2.6 라이선스를 직접 수락한다.
2. classifier/regressor의 `v2.6_default` 체크포인트를 로컬 고정 디렉터리에 내려받는다.
3. 두 파일의 SHA-256과 모델 카드의 synthetic-only 문구를 compliance manifest와 README에 기록한다.
4. 네트워크를 끈 상태에서 explicit `model_path`, telemetry off, 6시간 wall cap으로 smoke→P1/P3 fixed historical confirmation 순서로 실행한다.

근거: [TabPFN 공식 저장소](https://github.com/PriorLabs/TabPFN), [공식 가중치 접근 문서](https://docs.priorlabs.ai/how-to-access-gated-models), [changeforest JMLR](https://www.jmlr.org/papers/v24/22-0512.html), [multivariate change-point AISTATS 2024](https://proceedings.mlr.press/v238/wu24g.html).

## 최종 운영 판단

- P1: `KEEP_0.833548`; 과거 후보 재제출 금지. TabPFN-2.6 classifier checkpoint가 준비되면 구조 전환 1순위.
- P2: `KEEP_0.424019`; 과거 포함 현재 최고. 점수 격차는 남지만 이번 질문에서 재탐색보다 P1/P3가 우선.
- P3: `KEEP_CLEAN_0.583892`; fractional target 종료. TabPFN-2.6 lead-wise regressor가 다음 구조 전환 1순위.
- 공식 제출: 이번 사이클 0. 새 local gate PASS 후보가 없으므로 제한된 제출권을 쓰지 않는다.
