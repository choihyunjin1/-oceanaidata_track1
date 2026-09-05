# P2 결과 — 연속 결측 증강 3-seed가 내부 전체 후보 1위

**배포 관측만 쓰는 raw v23 DeepSets에 T5/S5 연속 결측 증강을 추가한 3-seed 평균이 내부 RMSE 0.920346→0.859250℃, −0.061096℃(약6.64%)로 개선됐다.** 3개 fold 모두 개선했고 전체 baseline/arm/고정half 비교에서도 가장 좋았다. 새 모델의 공식 점수는 아직 미확인이다. 기존 bin17 답안 또는 공식 점수 역산 계수로 만든 과거0.424019℃ 제출의 재현이나 공식 최고점 갱신을 주장하지 않는다.

24 DeepSets fits, 15 seasonal OAS covariance fits, calibration0, fullfit0으로 연구 실행을 **435.454초**에 완료했다. root의 독립 수치QA 후 별도 deploy ID에서 scratch fullfit3회와 fresh-process 추론을 수행해 26,061행 후보를 준비했다. [배포 영수증](../p2_score_repair_deploy_20260905_v1/predict-result.json), [재현·제출 안내](../p2_score_repair_deploy_20260905_v1/README.md).

## 동일 평가 비교

평가 69,850행, `sqrt(total SSE/N)`. 단순 fold RMSE 평균이 아니다. 원본 raw per-seed 출력은 ignored `artifacts/p2_score_repair_20260905_v1/raw_oof.npz`에 보존했다.

| 후보 | 내부 RMSE ℃ | 의미 |
|---|---:|---|
| v23 + blockmask 3-seed | **0.859250** | 선택 후보, raw 절대예측 평균 |
| raw v23 control 3-seed | 0.920346 | 같은 학습·평가 조건 control |
| raw v23 screen 1-seed | 0.949779 | screen 기준 |
| raw v52 screen 1-seed | 0.953375 | v23보다 약간 악화 |
| chosen/OAS 사전고정50:50 | 0.964908 | 상보적오차 상관0.5403이나 평균은 악화 |
| v23 + actualdepth 1-seed | 1.039733 | 별도 단일변경 악화, 추가seed 미실행 |
| nominal interpolation | 1.240144 | endpoint-clamp, 학습/추론 동일 |
| 다시 학습한 seasonal OAS | 1.415268 | 공식 역산 보정 없는 순수 커널 |
| T1, 결측이면 nominal | 1.681287 | 무학습 기준 |

| 복원 fold | 행 수 | v23 3-seed | blockmask 3-seed | ΔRMSE |
|---|---:|---:|---:|---:|
| 2024-09~10 | 26,273 | 0.492355 | 0.488284 | −0.004071 |
| 2025-07~08 | 26,693 | 1.369964 | 1.286209 | −0.083755 |
| 2025-11~12 | 16,884 | 0.399828 | 0.260733 | −0.139095 |

세 seed 개별 비교도 개선했지만 seed RMSE 표준편차는 control0.04138, blockmask0.04741로 줄지 않았다. 7일 calendar-block을 fold 안에서 resample한 탐색적95% Δ구간은 [−0.122147,−0.009610]℃다. 이미 반복 사용한 historical 자료에서 후보를 고른 결과이므로 이 구간을 fresh holdout 또는 공식 향상 확률로 해석하지 않는다.

## 데이터·누출·규칙

- 단일 입력 `P2_DATA_DIR/observations.csv`; SHA `cc5d6fd9ea398a613e485b34fd530c7dc064fa040fa675127a12318856ab178a`. 관측값, 행별 정답, 예측은 tracked 보고서에 기록하지 않았다.
- 학습 범위 2024-05-01~2026-01-01 KST. 기준 실제 학습 가능 target166,268행. 세 fold는61/62/61일이며 주변±7일을 제외한다. 양쪽 자료가 있을 때 모두 쓰는 복원 평가. 마지막 fold 뒤에는 배포 자료가 없어 과거 관측만 사용되는 한계가 있다.
- DeepSets 특징은 동시각 공개층 프로파일/시간주기/목표층nominal뿐. 목표temp/psal은 특징 구축에서 제외하며 합성 poisoning invariance 검사 통과. OAS 평가창은 목표temp/psal 모두NaN 처리하고 fitting에서도 평가창±7일 제외. 관측 기반 특징 의존은 동시각0시간이다.
- source에 QC열 없음. finite target 및 공개 수온층2개 이상을 proxy로 썼으며 hidden QC와 같은 모집단이라고 주장하지 않는다. 미래공식값, 원관측 정답, 외부기상·재분석, bin17/과거답안, 공식 역산계수 모두 미사용.
- T5/S5 blockmask는 label독립 3/7/14일 calendar block 약30%. 해당공개값을 가린 뒤 baseline/scale/tokens재계산. 원본을 유지하고 복제행 각각0.5 가중치를 배분해 원본별 총 가중치 보존. 결과를 보고 평가행 삭제하지 않았다.
- `actualdepth`는 nominal을 대체하지 않고 유효actual/50와presence 두context만 추가한 독립 arm이다. 유효 수심99.9543%를 확인했으나 해당모델의 결과는 악화했다. blockmask와 결합한 재튜닝은 하지 않았다.

## 결측·수심 위험 진단

평가 중 실제 수심 무효34행, 유효69,816행에서 blockmask가 각각0.695922/0.859322℃로 control0.929506/0.920341℃보다 낮았다. 공개 수온층2/3/4/5개 층화에서도 모두 낮았으나2개지원 표본은34행뿐이다. [추가 층화QA](independent-qa.json).

T5 부재17,064행에서0.404226→0.265130℃, 존재52,786행에서1.033458→0.976862℃. 관측기간 T5 최대 연속 결측11,255개의10분 간격(약78.16일)이 있어 이번3/7/14일 증강이 모든 배포 결측 길이를 포괄했다고 말할 수 없다. 다음 실험에서 이를 결과기반무제한조정하는 대신 고정모델의 공식성능·오차전이부터 확인한다.

## 독립 검증과 남은 판단

`validate-data`와 `analyze-data-quality` 스킬의 분모·모집단·키결합 검증 절차를 적용했다. 지정된 재사용 자산은 Python runner/tests와 aggregate JSON으로 남겼다. 별도 notebook/HTML은 이번 산출물 범위에 추가하지 않았다.

- 모든14개 예측배열의 합산SSE/RMSE 독립재계산, 키중복0, 원본관측target69,850행 직접키대조 exact일치, manifest/의존코드/artifact hash 일치.
- 24개의서로다른fit, 모두60epochs, checkpoint재로드 후inference exact일치. 연구runner/config는 실행후변경0.
- focused synthetic pytest14PASS, deploy synthetic6PASS, RuffPASS. [19-check QA](independent-qa.json).
- 가장 중요한 한계: 공식과 비슷한9~10월 seasonal fold개선은0.0041℃로 작고, 전체개선은 다른계절에 더크다. 내부0.0611℃를 공식에 그대로 대입할 수 없다. 최고점달성은 아직 미확인이다.
- 현재판단: 규정·코드·내부비교 기준의 **제출 후보 준비 가능**. 업로드0, commit/push0. 구체후속은 root가 공식 제출기회와 전체P1/P3진행을 대조해 선택한다.

## 재현 코드와 증거

[실행 계약](contract.md) · [runner](../../scripts/run_p2_score_repair_20260905_v1.py) · [config](../../configs/experiments/p2_score_repair_20260905_v1.json) · [tests](../../tests/test_p2_score_repair_20260905_v1.py) · [전체 결과](result.json) · [source contract](data-contract.json) · [manifest](manifest.json).

연구원본runner SHA `6eb6c0b5e0f64fcc1a987fd9d6461b6ab4f79fe7a749c941cb44e10c01aef549`, config SHA `3d0c0d370455bb84e969107d8854a37a60e8385877cad6bc580117b9bc384a54`.
