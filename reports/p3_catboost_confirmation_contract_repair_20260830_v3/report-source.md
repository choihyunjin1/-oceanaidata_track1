# P3 CatBoost confirmation contract repair v3

## 결론

**기술 복구는 성공했지만 frozen `challenger_21`은 `CONFIRMATION_GATE_FAIL_HPO_CLOSED`로 종료한다.** v2 selection의 ΔRMSE `-0.0228625m`는 182-case confirmation에서 `+0.0079741m` 악화로 역전됐다. 세 fold, 세 station, 여섯 lead가 모두 비개선이고 paired case bootstrap CI90도 `[+0.0015389, +0.0138540]m`로 0보다 완전히 높다. 파라미터·iteration·router·KMA 조합을 재튜닝하거나 selection search를 재실행하지 않는다.

## 기술 복구

- 새 ID: `p3_catboost_confirmation_contract_repair_20260830_v3`
- v2 selection artifact SHA256: `cc9451f3fe6c1138daba947e902b000e0ca371d861d0abc823b47f7a39fc3fb4`
- frozen 후보: `challenger_21`, iteration 138
- 원인: canonical router projection이 의도적으로 제외한 `current_hs`와 `single_prediction`을 v2 confirmation 코드가 그 projection에서 다시 선택했다.
- 수정: frozen router 원본에서 두 컬럼만 supplemental projection으로 읽고, pair key one-to-one·column order·dtype·finite·six-lead·row count를 검사한 뒤 canonical router와 결합했다.
- 0-fit preflight: 182 cases, 1,092 rows, schema SHA256 `8bde87da5d7d7baa7116bb9e3d81bab1a1049ed0406291dc04647ade0a04ca0c`, `READY_GUARDED`.
- focused pytest 7/7 PASS, 관련 Ruff PASS.

## 고정 실행

| fold | train cases | validation cases | iterations | fit seconds |
|---|---:|---:|---:|---:|
| 2024_h2_storm | 7,912 | 49 | 138 | 89.667 |
| winter_transition | 11,754 | 80 | 138 | 100.892 |
| 2025_h1 | 20,899 | 53 | 138 | 124.076 |

- confirmation fits: 3
- control fits: 0; frozen incumbent single/router/KMA prediction 재사용
- selection search fits: 0
- full refit fits: 0
- terminal runtime: 317.812s
- blind prediction SHA256: `51a5b2f53a4853fd3a4a15a1dddf001038377be156b4b9062d9ad3d2ae67cbf0`

## 확인 결과

| metric | control | challenger | ΔRMSE (challenger-control) |
|---|---:|---:|---:|
| pooled | 0.7807818 | 0.7887559 | +0.0079741 |

### fold ΔRMSE

- `2024_h2_storm`: +0.0010436m
- `winter_transition`: +0.0157026m
- `2025_h1`: +0.0021320m

### station ΔRMSE

- `G-ORS`: +0.0081934m
- `I-ORS`: +0.0170522m
- `S-ORS`: +0.0006302m

### lead ΔRMSE

- 3h: +0.0102673m
- 6h: +0.0135513m
- 9h: +0.0094078m
- 12h: +0.0089207m
- 18h: +0.0058656m
- 24h: +0.0023224m

### 불확실성과 gate

- paired whole-case bootstrap 90% CI: `[+0.0015389, +0.0138540]m`
- pooled, bootstrap, nonworse fold/station, 18h, 24h, short-lead, worst station×lead의 8개 promotion check가 모두 실패했다.
- selection의 국소 개선이 confirmation 일반화로 운송되지 않았다는 증거다.

## 독립 QA

`scripts/qa_p3_catboost_confirmation_contract_repair_20260830_v3.py`가 blind prediction을 frozen historical anchors와 다시 결합해 metric, 5,000-replicate bootstrap, gate를 독립 재계산했다.

- result SHA256: `1c2a6df2034f20dab054392ad67dd0039240729412dddf6380918c52957563ca`
- seal SHA256: `f88120d88a3ce80453bc0bb00515e0569d785580092c2042aa544ab31913b667`
- attempt lock SHA256: `f457f378ba94a4a014e159f747afdc2d2b11bac8ec1318404eca0b6781608a67`
- metric/bootstrap/gate exact match: PASS
- truth in blind prediction: 0 columns
- official test/sample/submission/hidden rows read: 0
- CSV written: 0
- upload attempted: false

## 다음 판단

P3의 이 exact 후보와 selection search는 닫는다. 결과를 보고 depth·iteration·KMA alpha를 바꾸는 후속은 허용하지 않는다. 사전등록 우선순위는 이제 P2 nonparanormal Gaussian-copula conditional-mean pilot로 이동한다. P3 전체 CatBoost 계열이나 다른 새로운 정보축까지 불가능하다고 확대 해석하지 않는다.
