# P3 결과: 결측 증강은 채택하지 않고, clean 기준과 작은 6h 후보를 보존

기상 묶음 결측 증강은 같은 LightGBM control보다 RMSE **0.00888860m 악화**했다. 더 강한 clean corrected 기준 0.77910484m도 이기지 못했다. 따라서 모델/증강 비율을 바꾸어 재탐색하지 않는다. 사전고정 무학습 후보 세 개 중 6h-only TabPFN25만 −0.00076509m였지만, 이는 **이미 본 검증면의 작은 사후 탐색 개선**이다. 독립적으로 확인된 돌파로 표시하지 않는다. 후속 별도 source-only 재학습 경로에서는 기준선/6h 후보 로컬 CSV 준비까지 완료했으며, 이는 [후속 배포 보고](../p3_score_repair_deploy_20260905_v1/report-source.md)에서 별도로 구분한다.

## 동일 표면 내부 실측

| 후보 | 전체 1,086행 RMSE(m) | 강한 clean 기준 대비 ΔRMSE(m) |
|---|---:|---:|
| Clean corrected single/multi/router/persistence-shrink | 0.77910484 | 0 |
| 새 LightGBM control, 2-seed 평균 | 0.80655461 | +0.02744977 |
| 기상 blockmask, 2-seed 평균 | 0.81544321 | +0.03633837 |
| Control25 + clean75 | 0.78252326 | +0.00341842 |
| Blockmask25 + clean75 | 0.78420964 | +0.00510480 |
| TabPFN 단독(기존 저장 OOF) | 0.83352241 | +0.05441757 |
| TabPFN25 전체 리드(기존 고정 blend) | 0.78324918 | +0.00414434 |
| TabPFN25 **6h만**, 나머지는 clean | 0.77833975 | −0.00076509 |

single25/multi25 사전고정 후보도 각각 0.78103090/0.78074236m로 악화했다. raw/new-model preclip 예측은 ignored OOF에 보존했으며 이번 두 LightGBM arm은 clip 전후 metric이 같았다. 과거 component OOF는 이미 clip된 값이어서 raw preclip 값을 복원했다고 주장하지 않는다.

6h-only 후보의 181 episode paired bootstrap 10,000회에서 ΔRMSE 95% 구간은 **[−0.00261383, +0.00116266]m**다. 이 구간은 반복 노출 검증의 선택 편향을 제거하지 않는다. 전체 효과를 공식 점수 크기로만 바꾸면 약 **+0.0121점**이지만, 실제 Public 점수 예측이 아니다. 2025 H1 RMSE는 0.82427567→0.82506383m로 악화했고 G/I 정점도 소폭 악화했다. 이 위험은 유지한다.

## 설계·검증·원인 단서

- 배포 train 2개 source hash, train feature/anchor cache hash, clean recipe/OOF/key hash를 대조하고 같은 키의 truth를 재계산 비교했다. 공식 test/sample/hidden/CSV/upload는 0이다.
- 3 expanding folds, 181 cases/181 station-episode, 1,086 lead rows, station-global 78h 간격을 재사용했다. 첫 fold 사례 수는49, 겨울79, 2025 H1 53이다. 새 독립 표본을 얻은 실험은 아니다.
- `src/p3_wave/models.py`의 compact residual LightGBM defaults를 재사용했다. 591개 중 weather-derived 330열을 함께 가렸다. original/masked 각각 기존 case weight의0.5, 원래 미관측 사례 중량1.0을 사용했다.
- 합성 289행 context에서 원시 weather를 먼저 가려 재계산한 **1,275개 모든 요약값**이 feature-level 가림과 일치했다. hs/target/6lead 일관성·중량보존·OOF join·pooled RMSE tests 7 PASS, Ruff PASS.
- 기상 미관측54cases/324rows에서 control→blockmask RMSE 0.82782538→0.82957723; 관측127cases/762rows에서 0.79733843→0.80935869m였다. 이번 구성은 두 지원군 모두 개선하지 못했다. 기상 결측 문제 전체가 해결 불가능하다는 일반화는 하지 않는다.
- 실행은 12/12 tree fits, 526.713초(8.78분), CPU2threads/GPU0, 최대 RSS2.949GiB. model native save/reload parity 12/12 PASS. 모델 조건·seed·표면을 실행 도중 변경하지 않았다.
- root가 `qa_oof.npz`의 1,086 unique rows와 동일 truth/reference 및6h의181변경행을 독립 검산하여 PASS를 회신했다. 추가 별도 계산에서 모든12모델 SHA 일치, RMSE·기상군·bootstrap을 재검산했다.

## 다음 행동과 준비도

새 LGBM/증강 가지는 종료한다. 별도 `p3_score_repair_deploy_20260905_v1`에서 clean baseline과 작은6h 후보를 **배포 source→새 학습 OOF→전체 모델 저장→별도 프로세스 추론**으로 재생성한다. 기존답안CSV/과거OOF는 해당 실행의 필수입력이 아니다. 이전 .77910484와 새 GPU/2thread 재학습 OOF의 numerical drift를 분리한다. 작은 개선보다 drift가 크면 개선 확정이라고 쓰지 않는다.

후속 완료: source-only 9-backbone/3-router fit, 새프로세스 reload 오차0, 총1,678.413초를 확인했다. 새 baseline OOF의 key/truth/prediction은 과거 기준선과 exact 일치했다. 별도 root 승인 후 공식 익명 context/index로 로컬 후보2개를 만들었고 hidden/sample/과거제출 입력/업로드/commit/push는0이다. 이 screen 자체의 공식 접근0 영수증은 변경하지 않는다. 현재 머신의 6시간 실측이 최종ZIP/독립폴더/운영진 하드웨어 검증을 뜻하지 않는다.

근거: [고정 설계](preregistration.md), [0-fit 출처·집계](zero-fit-audit.json), [실행 결과](result.json), [재계산 QA](independent-qa.json), [배포 의존 정찰](deployment-inventory.md). 상세 행/예측/모델은 Git 제외 artifact에만 존재한다.
