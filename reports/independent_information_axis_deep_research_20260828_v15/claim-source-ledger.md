# Claim–source ledger

확인일: 2026-08-28

| ID | 주장 | 출처·저자·연도 | URL | 접근·검증 메모 | 신뢰도 | 반증·한계 |
|---|---|---|---|---|---|---|
| S1 | low-rank background와 sparse corruption은 식별 조건에서 분리 가능 | Candès, Li, Ma, Wright, 2009/2011 | https://arxiv.org/abs/0912.3599 | arXiv abstract와 방법 주장 확인 | 높음 | P1 joint-time panel이 없어 적용 전제 불충족 |
| S2 | PCP는 작은 entrywise noise에서도 안정화 가능 | Zhou et al., 2010 | https://arxiv.org/abs/1001.2363 | arXiv abstract 확인 | 높음 | 실제 센서 잡음·rank/sparsity 조건은 별도 |
| S3 | sparse 항의 temporal variation 벌점은 지속 이상에 맞음 | Sofuoglu, Aviyente, 2020 | https://arxiv.org/abs/2010.12633 | temporally persistent anomaly 목적 확인 | 중상 | 원 논문 domain은 해양 센서가 아님 |
| S4 | 인접 수심·다변량 검사는 해양 T/S QC의 합법적 정보축 | NOAA/IOOS QARTOD TS Manual v2.1, 2020 | https://cdn.ioos.noaa.gov/media/2020/03/QARTOD_TS_Manual_Update2_200324_final.pdf | NOAA-hosted manual 확인 | 높음 | 지역 threshold와 실제 동시 관측 필요 |
| S5 | supervised functional component는 predictor-response 연관을 직접 사용 | Zhang, Sun, Kong, JCGS 33(1), 2024 | https://www.tandfonline.com/doi/abs/10.1080/10618600.2023.2250411 | 저널 abstract·DOI·출판일 확인 | 높음 | 구현은 논문의 전체 sparse optimization이 아니라 고정 rank-1 PLS proxy |
| S6 | multivariate response coefficient에 reduced-rank 제약을 둘 수 있음 | Izenman, JMVA 5(2), 1975 | https://www.sciencedirect.com/science/article/pii/0047259X75900421 | 저널 abstract·DOI 확인 | 높음 | 선형·대표본 이론과 현 시계열 shift 차이 |
| S7 | covariate support mismatch는 target risk를 악화시킬 수 있음 | Pathak, Ma, Wainwright, ICML 2022 | https://proceedings.mlr.press/v162/pathak22a.html | PMLR 원문 abstract 확인 | 높음 | support veto가 P2 계절 회귀를 실제 방지한다는 직접 증거는 없음 |
| S8 | 파랑 진화는 wind-wave interaction/source term을 포함 | ECMWF IFS CY48R1 Part VII, 2023 | https://www.ecmwf.int/en/elibrary/81373-ifs-documentation-cy48r1-part-vii-ecmwf-wave-model | ECMWF 공식 문서 목차·DOI 확인 | 높음 | 대회 모델과 동일한 dynamical model이 아님 |
| S9 | WAVEWATCH III는 third-generation wind-wave model framework | NOAA/NCEP WAVEWATCH III documentation | https://polar.ncep.noaa.gov/waves/wavewatch/wavewatch.shtml | NOAA 공식 문서·manual 목록 확인 | 높음 | local 48h proxy가 full source-term model을 대체하지 못함 |
| S10 | inverse wave age는 wind speed, direction alignment, phase speed를 결합 | Li et al., JGR Oceans, 2024 | https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023JC020686 | AGU/Wiley 본문 정의 확인 | 높음 | 대회 tp는 spectral peak period가 아니므로 hard 0.84 threshold 금지 |

## 내부 실험 주장 provenance

| 주장 | 정본 artifact | 독립 QA |
|---|---|---|
| P1 added 0·window 0 | `artifacts/p1_temporally_fused_rpca_offset_drift_anchor_union_20260828_v1/result.json` | prediction exact-anchor equality 확인 |
| P2 ΔRMSE `-0.004799°C`, Nov-Dec 회귀 | `artifacts/p2_alpha50_supervised_rank1_functional_residual_20260828_v1/result.json` | NPZ SHA·truth-late·proxy disclosure 확인 |
| P3 shadow `-0.011197m`, 17 cases | `artifacts/p3_past_only_wind_wave_memory_regime_increment_20260828_v1/result.json` | shadow SHA·outer 미개방 확인 |
| 공식 접근/CSV/업로드 0 | 세 result와 artifact inventory | `independent-qa.json` PASS |
