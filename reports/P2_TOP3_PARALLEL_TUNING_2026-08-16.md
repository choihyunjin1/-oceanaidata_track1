# P2 상위 3개 GBM 병렬 최적화 결과

## 결론

세 계열 모두 최대 탐색 예산 안에서 수렴 checkpoint를 찾았지만, 동일 outer 검증과 deep LOBO 결합에서 기존 후보를 넘지 못했다. 이번 세대를 `REJECT`하고 제출 1순위는 `P2_DEEP_STACK_V1.csv`로 유지한다.

## 공정 비교 계약

- 대상: CatBoost layerwise, CatBoost pooled, LightGBM DART.
- 특징: 기존 public-only phase 81개 특징 고정.
- 예산: 가족당 총 36 trials, outer 폴드별 독립 12 trials × 3.
- 선택: 각 outer train 내부의 별도 inner 기간만 사용. 해당 outer 라벨은 자기 파라미터 선택에 사용하지 않음.
- CatBoost: 최대 3,000 rounds, early stopping patience 150.
- DART: 표준 early stopping이 지원되지 않아 200·400·600·800·1,200·1,600·2,400·3,000 rounds를 inner 탐색.
- 실행: 3 workers 병렬, worker당 CPU threads 2, Optuna `n_jobs=1`.

## 결과

| 순위 | 계열 | 고정 구조 RMSE | 튜닝 outer RMSE | 변화 | deep pair LOBO | deep 대비 LOBO 변화 | 수렴 rounds |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | CatBoost layerwise | 0.833549 | 0.900625 | +0.067076 | 0.778989 | +0.003329 | L2 94 / L3 132 / L4 16 |
| 2 | CatBoost pooled | 0.843419 | 0.910276 | +0.066857 | 0.780118 | +0.004458 | 158 |
| 3 | LightGBM DART | 0.829720 | 0.927322 | +0.097602 | 0.796260 | +0.020600 | 1,600 |

Frozen deep LOBO RMSE는 `0.7756600313`이다. 세 LOBO 변화는 모두 양수이므로 개선이 아니다.

특히 DART fitted pair RMSE `0.7437219871`은 frozen deep fitted RMSE `0.7458139094`보다 낮지만, LOBO 변화의 90% KST-day bootstrap CI가 `[+0.0083892063,+0.0333321716]`로 전부 악화 방향이다. 이는 결합 가중치 과적합으로 판정한다.

## 선택 파라미터

### CatBoost layerwise

```text
bootstrap_type=MVS
learning_rate=0.0952176995303601
depth=10
l2_leaf_reg=4.7480304513323635
random_strength=1.1308491248486294
rsm=0.9687935001696453
leaf_estimation_iterations=2
subsample=0.8920313838194773
```

### CatBoost pooled

```text
bootstrap_type=MVS
learning_rate=0.05872961273284538
depth=8
l2_leaf_reg=0.5622636573176658
random_strength=0.26986739599783544
rsm=0.8747898871048947
leaf_estimation_iterations=1
subsample=0.7627895169354457
```

### LightGBM DART

```text
n_estimators=1600
learning_rate=0.02015547988009311
num_leaves=43
max_depth=12
min_child_samples=464
feature_fraction=0.5641720450686958
bagging_fraction=0.8177593827741176
bagging_freq=1
reg_alpha=0.0009702787528501448
reg_lambda=20.23008339095893
drop_rate=0.11950804312573086
skip_drop=0.3112737282783449
max_drop=31
```

## 독립 검산

- 세 OOF 각각 69,850행, `(time, layer, block)` 유일성 및 truth 정렬 통과.
- outer RMSE·층별 RMSE·deep fitted/LOBO pair를 원 OOF에서 독립 재계산해 일치.
- 세 standalone CSV와 연구 pair CSV 모두 26,061행, 정확한 키·순서·유한 온도·범위 검사를 통과.
- 연구 pair의 fitted CatBoost 가중치는 세 층 모두 0이어서 새 제출 후보가 아니다.
- 외부 관측값·hidden 정답·플랫폼 업로드 사용 0건.

## 파일과 SHA256

- `artifacts/p2_top3_parallel_tuning_v1/result.json`: `ff7626b8e85f34e87b4acd2218f60d19ee9a3bc1af9fbb984cbbdc79d52a6eda`
- `artifacts/p2_top3_parallel_tuning_v1/independent_validation.json`: `c98f17907ac87b5fbe69b6ecb42d9a6e3b3d7441a79cc488c2abec6f12c33eba`
- `submissions/p2/P2_TUNED_CATBOOST_LAYERWISE_V1.csv`: `249393cb0e6f4ef82761d414ecc9ec8c287e26a4bd0948c0100f7f870e08fa91`
- `submissions/p2/P2_TUNED_CATBOOST_POOLED_V1.csv`: `7f3875e825043fee596363bfdaeaf6ee47290d9bd568f37c02f387c28d3056ab`
- `submissions/p2/P2_TUNED_LGBM_DART_V1.csv`: `59c04fe872ba75a12b90e0955261632e5376a369cf463728607657d74b48015b`

위 세 tuned CSV는 진단 산출물이며 제출 1순위가 아니다. 사용자 승인 없이 업로드하지 않는다.
