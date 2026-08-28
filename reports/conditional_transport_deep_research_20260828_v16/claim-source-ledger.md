# Claim–source ledger — v16

확인일: 2026-08-28

| ID | 주장 | 1차 출처 | 검증 메모 | 신뢰도 | 이번 실험의 한계 |
|---|---|---|---|---|---|
| S1 | multitask GP는 불규칙·상관 시계열의 공동 잠재 상태 표현에 사용 가능 | Futoma et al., ICML 2017, https://proceedings.mlr.press/v70/futoma17a.html | PMLR abstract·방법 확인 | 높음 | P1은 fit 전 calibration support 실패 |
| S2 | GP로 다차원 change surface를 확장할 수 있음 | Herlands et al., AISTATS 2018, https://proceedings.mlr.press/v84/herlands18a.html | PMLR 원문 메타데이터 확인 | 높음 | 대회 센서 이상 탐지에 대한 직접 보장은 아님 |
| S3 | conformal inference는 교환가능성 등 명시적 조건 아래 유한표본 coverage를 제공 | Chernozhukov et al., COLT 2018, https://proceedings.mlr.press/v75/chernozhukov18a.html | PMLR abstract 확인 | 높음 | P1 시간 block의 의존성과 희소 cell은 조건을 약화 |
| S4 | reject/abstain은 covariate shift에서 위험을 줄이는 별도 의사결정 문제 | Li et al., AISTATS 2024, https://proceedings.mlr.press/v238/li24g.html | PMLR abstract·정리 범위 확인 | 높음 | P2의 label reference 자체가 식별 불가여서 veto 학습 전 중단 |
| S5 | spectral wave model은 propagation뿐 아니라 source term과 방향 스펙트럼을 다룸 | ECMWF IFS CY48R1 Part VII, 2023, https://www.ecmwf.int/en/elibrary/81373-ifs-documentation-cy48r1-part-vii-ecmwf-wave-model | ECMWF 공식 문서 확인 | 높음 | 대회 feature는 full spectrum이 아님 |
| S6 | mean wave direction은 spectrum-weighted 평균이며 파고는 스펙트럼 적분량과 연계 | ECMWF Forecast User Guide, 2026, https://confluence.ecmwf.int/spaces/FUG/pages/673550584/Section%2B2A.3.1%2BWave%2Bmeasures%2Band%2Bdefinitions | 공식 정의 확인 | 높음 | 평균 방향만으로 spread·partition 복원 불가 |
| S7 | 방향 스펙트럼은 mean direction 외 width·skewness·kurtosis 정보를 가짐 | Kuik et al., JPO 1988, https://journals.ametsoc.org/abstract/journals/phoc/18/7/1520-0485_1988_018_1020_amftra_2_0_co_2.xml | 저널 abstract 확인 | 높음 | P3 proxy는 이 독립 파라미터들을 관측하지 못함 |

## 내부 실험 주장 provenance

| 문제 | 정본 artifact | 독립 확인 |
|---|---|---|
| P1 | `artifacts/p1_async_latent_state_gp_subset_scan_anchor_union_20260828_v1/result.json` | manifest/result/sealed NPZ SHA 일치, truth 0, deletion 0 |
| P2 | `artifacts/p2_alpha50_supervised_rank1_trainonly_regime_veto_20260828_v1/result.json` | commitment 0, outer truth 0, prediction file 0, retry 0 |
| P3 | `artifacts/p3_era5_wave_directional_energy_memory_20260828_v1/result.json` | source model·seal hash 일치, shadow truth 0, source Δ 양수 |
| 공통 | 세 artifact directory | CSV 0, 공식 row 0, upload 0 |
