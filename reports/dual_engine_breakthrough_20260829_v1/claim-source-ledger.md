# Claim-source ledger

기준 커밋: `a042127046e57e78a48300251f50343fe073a857`

| ID | 주장 | 근거 | 반증·주의 | 상태 |
|---|---|---|---|---|
| P1-C1 | 겹친 창의 시작 위치는 현재 학습되지 않은 nuisance다. | `src/p1_qc/ms_tcn_asrf.py`, `configs/experiments/p1_incumbent_preserving_mstcn_asrf_v2.json`; Mean Teacher 2017; Kayhan 2020 | center-weighted overlap-add가 이미 효과를 줄였을 수 있음 | preflight 필요 |
| P1-C2 | TENT-BN은 현재 backbone에 바로 적용할 수 있다. | Gemini 제안 | `GroupNorm`, `LayerNorm`만 사용하고 BatchNorm이 없음 | 기각 |
| P1-C3 | CRC로 F1 threshold risk를 단조 제어할 수 있다. | Gemini 제안 | F1은 threshold에 대해 일반적으로 단조가 아님 | 기각 |
| P2-C1 | spatial neutral-surface transport가 가능하다. | Gemini 제안 | 단일 S-ORS이며 공간 이웃/망이 없음 | 기각 |
| P2-C2 | two-sided frozen-anchor residual은 저주파 결측 bias를 식별할 수 있다. | fixed-interval smoothing 직관 | boundary-registered prior가 크게 실패했고 target-flank 정보가 부분 중복 | 마지막 falsification |
| P2-C3 | S2/N2/K1/O1은 M2와 독립적인 이득을 줄 수 있다. | 황해 tidal constituent 문헌 | 내부조석 위상은 계절·성층에 따라 변함; 기존 M2 계열 실패 | low-priority sentinel |
| P3-C1 | 과거 wind-wave lag는 새 정보축이다. | Gemini 제안 | `event_phase.py`, `wind_wave_memory.py`, `causal_forcing_sequence.py`, frozen 286 features에 이미 존재 | 기각 |
| P3-C2 | 예측 미래 국지 바람은 코드상 미시험 정보축이다. | repo-wide exact-name audit 0건; anchor table은 미래 hs만 생성 | 자체 미래 바람 skill이 없을 수 있고 원격 swell은 복구 불가 | oracle 필요 |
| P3-C3 | context의 wave energy 적분을 보존해야 한다. | Gemini 제안 | 열린·소산·이류계에서 보존 법칙이 아님; WW3 source terms와 충돌 | 기각 |

## 1차 출처

- Tarvainen & Valpola, 2017, [Mean Teacher](https://papers.nips.cc/paper_files/paper/2017/hash/68053af2923e00204c3ca7c6a3150cf7-Abstract.html)
- Kayhan & van Gemert, 2020, [On Translation Invariance in CNNs](https://openaccess.thecvf.com/content_CVPR_2020/html/Kayhan_On_Translation_Invariance_in_CNNs_Convolutional_Layers_Can_Exploit_Absolute_CVPR_2020_paper.html)
- He et al., 2023, [RAINCOAT](https://proceedings.mlr.press/v202/he23b.html)
- Rauch, Tung & Striebel, 1965, [Maximum likelihood estimates of linear dynamic systems](https://doi.org/10.2514/3.3166)
- Tolman et al., 2014, [NOAA WAVEWATCH III v4.18 manual](https://polar.ncep.noaa.gov/waves/wavewatch/manual.v4.18.pdf)
- Sun et al., 2022, [Wave forecast correction with wind and wave predictors](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022GL100916)

## 저장소 반증

- `reports/p1_value_preflight_robust_screen_20260829_v1/report-source.md`
- `reports/p1_mstcn_lower_bound_veto_20260829_v2/report-source.md`
- `artifacts/p2_boundary_registered_prior_20260827_v1/result.json`
- `reports/independent_information_axis_deep_research_20260828_v15/report-source.md`
- `reports/p3_kma_alpha_surface_sweep_20260829_v1/summary.md`
- `reports/p3_target_shift_retroaudit_20260828_v11/report-source.md`
