# P1 champion bracket-B replacement — local candidate and ZIP replay PASS

새 B107 교체는 retrospective paired 내부 pooled F1을 **0.904916741 → 0.906357812 (+0.001441071)**로 개선했다. 새 학습 17회와 독립 QA를 완료했고, 169,011행 답안 및 SOURCE_ONLY/SAVED_MODELS 패키지를 생성했다. 실제 SAVED_MODELS ZIP을 새 폴더에 풀어 노트북으로 재생한 답안 SHA가 일치했다. 아직 새로운 답안의 공식 점수는 없으며 업로드하지 않았다.

2026-09-07 11:31:01 KST에 CPU4 실행을 시작했다. Launcher 8380 / 실제 학습 worker 41780. 새 17 fits 전체 wall time 1,559.062초. 기존 P1 원형·bracket·P3 실행과 source/weights/locks는 변경하지 않았다.

## 실측 내부 결과

| 범위 | B80 control F1 | B107 candidate F1 | 차이 |
| --- | ---: | ---: | ---: |
| Q3 176,738행 | 0.910931841 | 0.912630971 | +0.001699130 |
| Q4 111,124행 | 0.895950920 | 0.897049822 | +0.001098902 |
| Pooled 287,862행 | 0.904916741 | 0.906357812 | +0.001441071 |

Fold-stratified 7일 block bootstrap 2,000회 CI90 [-0.000686527, +0.003771364], descriptive P(개선)=0.8625. 최악 block ΔF1=-0.016763714, 최악 정점·층 S-ORS/4 ΔF1=-0.012626717. 이 위험은 평균 개선과 별도로 보고하며 자동 탈락 기준으로 쓰지 않는다. 미학습 정점·층은 Q3·Q4 모두 0행이다. CI는 0을 포함하고 평가 자료는 이미 노출됐으므로 확정적 일반화 개선이나 공식 점수의 기대값으로 치환하지 않는다.

독립 QA는 전체 키·truth·source/model hash·개별/합성 모델 확률·최종 composition 재생·confusion/F1을 재계산해 PASS했다(새 fit0, 112.860초). 원 runner source-seal에 union-contract-v2.json 핀이 빠진 사실을 숨기지 않는다. QA에서 해당 JSON이 git HEAD의 내용과 같은지 별도 확인하고 SHA256 `2937bb38986c8256f0b2cc6a529e21af5bf3b92279e50a00aaf2ea08ce6a2aed`를 기록했다. 증거: `result.json`, `independent-qa.json`, `cpu-finish.json`.

## 사전등록된 변경과 비용

- 단일 변경: **원형 B80에 동일한 bracket27 특징을 추가한 B107**. 원형 feature implementation 자체를 사용하며 예전 clean80의 depth dictionary/plateau cap 변경은 가져오지 않는다.
- B의 3 seeds 20260813/20260829/20260847, 700 trees, 학습률·가중치·decoder 0.2/0.1·minrun12 고정. O, MS, 셀 정책, GI의 조합은 비교 양쪽에서 같다.
- Q3/Q4마다 동일 O1 + B80 control3 + B107 candidate3 = 14 fits. Full train B1073 = 3 fits. 합계 **17 new CPU fits**, GPU 학습0. 학습 스레드4는 동시 P3 CPU4와 8코어 호스트를 공유하는 이유가 있으며 시간 강제 종료는 없다.
- 예전 bracket 단독 후보의 F1 0.785944는 clean80·1-seed·threshold0.1 선택 결과이다. 새 원형 B3 교체의 성능이나 점수로 승계하지 않는다. 동일 recipe 모델이 아니므로 그 모델을 재사용하지 않는다.

## 내부 비교의 정확한 의미

- 고정 MS proposal의 원 Q3/Q4 소유 전체 287,862개 키와 fold를 유지한다. MS source hash와 기존 independent QA를 검증한다. 잘리는 119개 달력 경계 양성행을 새 달력 fold로 옮기지 않는다.
- 양 arm에 같은 새 O 모델을 사용한다. 구 O OOF는 7일 purge이고 MS 계보는 21일이므로 그 O 확률을 같은 split이라고 재사용하지 않는다. 새 O 2 fits는 이 비교 혼입을 없애기 위한 비용이다.
- 학습 cutoff는 각 MS receipt의 2025-06-09/09-09 23:50 KST를 사용하고 partition-local feature 계산, 337시간 의존 구간보다 긴 gap을 검사한다. 예측 구간은 원 MS 키 전체이며 미지원 정점·층도 지우지 않고 별도 집계한다.
- 이 데이터는 이미 노출된 **retrospective paired comparison**이다. 새 independent holdout, 이전 28.909점 모델의 정확한 OOF 재생, 공식 점수 상승 입증이 아니다. 평균 pooled F1 증가를 우선 보고하고 worst-block/정점·층 감소는 별도 위험이다.
- Full B 학습은 원본 CSV 행 순서를 유지하고 bracket 계산만 정점·층·시간 정렬→원래 행 순서 복원한다. O/MS full models는 기존 source-retrained exact hash를 재사용한다.

## 검증과 다음 단계

합성 계약4 PASS와 Ruff PASS: 원형80열 불변, bracket27 행순서 정합, 실제 작은 LightGBM fit, 원형 GI anomaly-type 조합 parity, 고정seed/threshold/fit 회계, portable adapter에서 B 외 원형 추론 AST 불변. 합성 결과는 실제 모델 성능 증거가 아니다.

`scripts/p1_champion_bracketb_20260907_finish_cpu.py`가 학습 worker 종료를 기다린 뒤 정상 terminal에만 독립 CPU QA(전체 키·sklearn confusion/F1·모델 probability replay·block bootstrap·support) → 새로운 portable package → tree inference를 수행한다. 이 supervisor의 새 fit은0이다. 완료 경로는 `cpu-finish.json`이다. 기술 오류이면 결과를 보존하고 자동 재학습하지 않는다.

P2 GPU owner가 모든 작업 종료 후 12:11에 명시적으로 반환했다. 12:11:32 KST에 별도 launcher 12272로 release를 시작하여 MS 공식 추론 → combine → 답안 QA → ZIP 포장 → 실제 추출 notebook 재생을 완료했다. Release wall time 113.828초, receipt의 replay+분할/reassembly 검증 구간 59.000초. 모든 schema/key/order/unique/binary/byte-exact 검사 PASS. Source-only 전체7fit를 새로 수행했다고 하지 않고 기존 O/MS cold + 새 B cold + 새 조합 replay의 증거를 분리한다. Jupyter/validate-data skill에 따라 notebook schema 검사와 실제 실행 여부, 과거 점수와 신규 답안 hash를 구분한다.

## 실제 후보 파일과 재현 한계

후보 폴더: `C:/Users/cedis/Documents/OceanFinalDay_20260907/P1_champion_bracketB_v1/`.

| 파일 | bytes | SHA256 |
| --- | ---: | --- |
| ANSWER/P1_champion_bracketB.csv | 6,929,481 | 7538082ade3fbb89887a87e49e0ad1e2313bb31ff424902818e7bc1713774c79 |
| P1_bracketB_SOURCE_ONLY.zip | 141,115 | e071d739a3370be3378e802647f5584267ba9428e023355c83a5560419ba26c9 |
| P1_bracketB_SAVED_MODELS.zip | 589,994,795 | dff64e9ca62c540fa55861cb4ea4eb30af77ef24ab31caecaa7c49f6246c53b9 |

답안 169,011행, 양성 6,372행. SAVED_MODEL_PARTS 15분할 재조립 SHA 검증 PASS; 포털의 첨부 개수·용량 허용을 확인했다는 뜻은 아니다. literal README.md와 TRAIN.ipynb/SAVED_PREDICT.ipynb를 포함한다. 실행된 노트북은 release root의 executed_SAVED_PREDICT.ipynb이고 ZIP 원본 노트북은 변경하지 않았다.

기존 O/MS 모델의 cold 증거 + 이번 B3 fresh 학습 + 새 ZIP의 실제 saved replay가 성립한다. **새 조합 전체를 빈 모델 폴더에서 7-fit 재학습해 답안 SHA가 같다는 검사는 미수행**이다. 과거 6-cell 선택 알고리즘도 미복구 상태이며 리터럴을 숨기거나 새로 복구했다고 주장하지 않는다. 일반 모델에 대한 공식 6시간 제한 적용 범위·마감 시각은 미확인이다.

원형 P1 `57844ef2`와 기존 패키지는 fallback으로 보존한다. 새로운 후보에 공식 점수는 아직 없다. 업로드·최종 지정·삭제·commit·push0. 최종 machine receipt는 `release-result.json`이다.
