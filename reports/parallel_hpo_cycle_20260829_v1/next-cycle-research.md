# 2026-08-29 HPO 후속 연구 메모

상태: **현재 one-shot 종료 전에는 새 모델 실행 금지**

기준 커밋: `8e0dc9ab22a713596425375985a6fe04f878e325`

공식 hidden/test/sample/submission 값 접근·CSV 생성·업로드: `0`

## 결론

현재 자원 배분은 P1의 32-point Sobol full-fidelity 학습을 끝까지 완료하는 데 둔다.
P2와 P3는 다음 후보를 설계했지만, 현재 사이클의 terminal QA·커밋·푸시 전에는 실행하지 않는다.

1. **P1**: 현재 Sobol HPO 결과를 먼저 본다. 실패할 때만 동일 MS-TCN topology에서
   `station x layer x time-regime` 그룹 강건 손실을 새 독립 실험으로 검토한다.
2. **P2**: `seasonal nonparanormal copula-OAS delta`의 **0-fit support audit**만 다음
   후보로 남긴다. 다만 public query support를 확인하려면 현재 금지한 공식 공개 입력
   접근 경계를 건드릴 수 있으므로 별도 계약·승인 전에는 실행하지 않는다.
3. **P3**: v1은 74 successful fits 뒤 75번째 `Ordered x Depthwise` 조합에서 terminal
   technical failure다. 재실행하지 않고, 새 ID의 strict-repair v2만 다음 후보로 남긴다.

## P1 조건부 후속축

현재 MS-TCN++/ASRF 계보의 topology, 165개 past-only features, anchor-union decoder는
고정한다. Sobol HPO가 Q2 pre-confirm gate를 넘지 못하거나 Q3/Q4에서 transport하지
못할 때만 학습 목적함수의 환경 강건성 한 축을 검토한다.

- 환경 후보: `station x layer x quarter`, 단 희소 group은 사전 최소 표본수로 병합한다.
- 비교: 기존 ERM 대 고정된 ERM/worst-group convex combination 한 축.
- 필수 안전장치: 강한 weight decay 또는 early stopping, anchor-positive 제거 0.
- 중단: Q3/Q4 방향 불일치, 한 station에 변경행 80% 이상 집중, pooled delta F1 `<=0`.

Group DRO는 사전 정의된 group의 최악 손실을 낮추지만, 과매개변수 신경망에서는
순진한 적용이 실패하며 더 강한 정규화나 early stopping이 중요하다는 1차 근거가 있다.
이 근거는 P1 성능을 보장하지 않으며, HPO 종료 후 별도 one-shot 가설로만 사용한다.

- Sagawa et al., ICLR 2020: https://arxiv.org/abs/1911.08731
- 공식 구현: https://github.com/kohpangwei/group_DRO

## P2 다음 후보: 0-fit copula support audit

기존 seasonal OAS의 선형 조건부 구조를 유지하고, train-only 계절 셀의 T/S 주변분포만
mid-rank empirical CDF와 inverse-normal transform으로 바꾸는 후보다. 저장소 감사에서
P2의 copula/nonparanormal/rank-Gaussian 구현은 발견되지 않아 구조 독립성은 높다.

먼저 모델 fit 없이 다음을 감사한다.

- nearest-prefix fallback cell `0`
- complete historical train timestamps `>=1000`
- train T/S coordinate별 unique values `>=200`
- query outside train min/max 비율 pooled `<=5%`, cell-coordinate worst `<=10%`
- non-finite transform·key mismatch `0`

하나라도 실패하면 `NO_GO_CLOSE_COPULA_AXIS`로 닫는다. 통과하더라도 세 historical
61-day outer window의 prediction을 truth 전에 seal/hash한 단 한 번의 평가만 허용한다.
기존 source/query discriminator AUC가 모든 fold에서 1.0이었던 반증 때문에 성능 및
official transport confidence는 낮다.

- Liu, Lafferty & Wasserman, JMLR 2009: https://jmlr.org/papers/v10/liu09a.html
- Zhao & Udell, KDD 2020: https://www.kdd.org/kdd2020/accepted-papers/view/missing-value-imputation-for-mixed-data-via-gaussian-copula.html
- gcimpute, JSS 2024: https://www.jstatsoft.org/article/view/v108i04

## P3 다음 후보: valid-combination CatBoost HPO v2

새 실험 ID 후보는 `p3_catboost_valid_hpo_20260829_v2`다. v1의 grid·lock·artifact를
변경하거나 재실행하지 않는다. v2의 구조 allowlist는 다음 세 개뿐이다.

- `Plain + SymmetricTree`
- `Plain + Depthwise`
- `Ordered + SymmetricTree`

기존 depth 3개, bootstrap 2개, regularization profile 2개를 곱해 36 challengers를
만든다. 구조군 층화 successive halving은 `36 -> 9 -> 3 -> 1`로 고정한다.

| 단계 | 후보+control | folds | fits |
|---|---:|---:|---:|
| 300 trees | 36+1 | 2 | 74 |
| 900 trees | 9+1 | 4 | 40 |
| 2500 trees | 3+1 | 6 | 24 |

Selection 최대 138 fits, confirmation 3, gate-pass full refit 1로 historical fit 상한은
142다. 모든 gate, 591 features, residual target, router/shrink/KMA alpha 0.4는 v1과
동일하게 유지한다.

historical attempt lock 전에 CatBoost `1.2.10`을 고정하고, deterministic
`128 x 591` synthetic float32에서 36 challengers와 control을 각각 one-tree fit한다.
finite prediction·shape·실제 parameter를 검증한 smoke receipt와 code/grid/test hash가
모두 맞을 때만 lock을 생성한다. 이 smoke가 실패하면 historical fit `0`으로 폐기한다.

- CatBoost 1.2.10 compatibility source: https://raw.githubusercontent.com/catboost/catboost/v1.2.10/catboost/private/libs/options/catboost_options.cpp
- CatBoost parameter documentation: https://catboost.ai/docs/en/references/training-parameters/common
- Prokhorenkova et al., NeurIPS 2018: https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html
- Jamieson & Talwalkar, AISTATS 2016: https://proceedings.mlr.press/v51/jamieson16.html

## 실행 순서

1. P1 현재 one-shot terminal 대기.
2. P1/P2/P3 현재 사이클 독립 QA, 통합 보고서, commit, non-force push.
3. P1 결과에 따라 conditional Group-DRO를 열거나 닫는다.
4. P3 v2 static config와 synthetic one-tree smoke를 먼저 구현·검증한다.
5. P2는 공식 공개 입력 접근이 필요한 support audit 계약을 별도 검토한 뒤 실행 여부를
   결정한다.

이 메모는 다음 사이클의 가설 공간을 봉인하기 위한 중간 연구 기록이며, 현재 실행 중인
P1의 grid·gate·checkpoint 규칙을 변경하지 않는다.
