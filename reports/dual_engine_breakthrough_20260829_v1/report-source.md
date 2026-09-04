# P1·P2·P3 이중 연구 엔진 돌파구 감사

작성일: 2026-08-29 KST

기준 커밋: `a042127046e57e78a48300251f50343fe073a857`

범위: Gemini 3.7 Flash Extended Deep Research 3건 + GPT Deep Research 3건 + 저장소 독립 대조

공식 test/sample/submission 값 접근: 0건

CSV 생성·업로드: 0건

## 결론

이번 병렬 연구에서 **즉시 공식 제출할 후보는 없다**. 다만 기존 실패축과 데이터 계약을 엄격히 대조한 결과, 값싼 local-only 반증을 할 가치가 남는 축은 다음 세 개다.

1. **P1 — window-phase inconsistency audit**: 동일 행이 서로 다른 겹침 창 위치에서 얼마나 다르게 예측되는지 먼저 측정한다. 실제 구조적 공백이고 기존 calibration/veto/decoder 재튜닝과 다르다.
2. **P3 — perfect-future-wind oracle**: 미래 바람을 완벽히 안다고 가정해도 frozen KMA 잔차를 충분히 줄일 수 있는지 상한부터 잰다. 통과할 때만 train-only 미래 바람 예측기로 내려온다.
3. **P2 — two-sided boundary residual bridge**: 결측 양쪽에서 보이는 frozen-anchor 편향만 고정 smoothstep으로 잇는다. 그러나 과거 boundary-registered prior와 정보원이 일부 겹치고 그 실험이 크게 실패했으므로 3순위의 마지막 falsification이다.

Gemini 보고서는 아이디어 발산에는 유용했지만 현재 저장소와 문제 계약을 충분히 읽지 못했다. 따라서 제안 수를 성과로 세지 않고, 코드상 실제 미시험 여부와 과거 반증을 통과한 것만 후보로 남겼다.

## 엔진별 신뢰도 감사

| 문제 | Gemini 제안 | 저장소 대조 | 판정 |
|---|---|---|---|
| P1 | IRMv1, BatchNorm TENT, CRC threshold | 현 backbone은 GroupNorm/LayerNorm이고 BatchNorm이 없다. F1은 threshold에 대해 단조 risk가 아니다. environment robustness는 이미 근접 축이 실패했다. | TENT·CRC 기각, IRM은 저우선 근접축 |
| P2 | TEOS-10 isopycnal, vertical modes, neutral-surface spatial transport | 단일 S-ORS, 8개 층 중 3개 결측이며 공간 이웃/좌표망이 없다. TEOS/profile 계열은 이미 다수 실행됐다. | 세 축 모두 계약 불일치 또는 중복 |
| P3 | wind-wave relaxation lag, lead MOS, energy-conserving smoothing | 0/1/3/6/9/12h lag와 wind-wave memory, direct multilead, spectral/energy/residual 계열이 이미 구현·실행됐다. \(\int H_s^2dt\) 보존은 열린 파랑계의 물리 법칙도 아니다. | 세 축 모두 중복 또는 물리 오류 |

Gemini가 목표 GitHub를 안정적으로 읽지 못하고 무관한 저장소를 참조한 흔적도 확인됐다. 그러므로 Gemini 결과는 독립 근거가 아니라 **가설 생성기**로만 사용한다.

## P1 — window phase가 1순위인 이유

e150 generator의 이론적 수용영역은 3,969행인데 입력창은 2,048행이며 convolution은 zero padding을 사용한다. 추론은 stride 512의 겹친 창을 center-weighted overlap-add하므로 같은 행이 최대 네 개의 창 위상에서 평가되지만, 위상 간 일관성은 학습하지 않는다. 반면 BCE·soft Dice·boundary·type loss는 이미 존재한다. 따라서 새 F1 loss가 아니라 **창 시작 위치라는 인공 nuisance**가 실제 병목인지 먼저 재는 것이 맞다.

동일 표본의 합법적 perturbation 간 consistency를 학습 신호로 쓰는 근거는 [Mean Teacher](https://papers.nips.cc/paper_files/paper/2017/hash/68053af2923e00204c3ca7c6a3150cf7-Abstract.html)에 있고, zero padding이 CNN에 절대 위치 단서를 줄 수 있다는 반증은 [Kayhan & van Gemert, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Kayhan_On_Translation_Invariance_in_CNNs_Convolutional_Layers_Can_Exploit_Absolute_CVPR_2020_paper.html)과 직접 연결된다. 다만 e150의 center weighting이 이미 문제를 충분히 약화했을 수 있으므로 학습보다 audit가 선행해야 한다.

### P1 가장 싼 preflight

- Q2 e150을 기본 tiling과 `+256행` tiling으로 각각 blind seal한다.
- 같은 행의 probability를 정렬해 `q99(|p0-p256|)`, proposal XOR, 고정 평균의 anchor-union F1을 계산한다.
- 다음 중 하나라도 참이면 학습하지 않고 종료한다.
  - `q99(|p0-p256|) < 0.05`
  - proposal XOR `< 50행`
  - 고정 평균의 Q2 anchor-union `Delta F1 < +0.001`

통과 시에만 exact e150 warm-start 5 epoch, paired view symmetric JS 한 축만 1회 실행한다. Q3·Q4 모두 개선, pooled `>= +0.001`, anchor 제거 0을 요구한다. 이 결과도 공식 제출 승인이 아니라 3-seed 재현 단계로만 승격한다.

## P2 — boundary bridge는 왜 3순위인가

가설은 결측 내부의 고주파를 맞히는 것이 아니라 frozen anchor가 놓친 61일 규모의 layer별 저주파 bias만 결측 전후 72시간의 잔차 median으로 식별하는 것이다.

`e_L = median(y - y_anchor)`와 `e_R`을 결측 양쪽에서 구하고, 고정 cubic smoothstep으로만 연결한 뒤 기존 projector를 정확히 한 번 적용한다. 양쪽 관측을 사용하는 식별 직관은 fixed-interval smoothing과 맞지만, 이 식의 성능을 보장하지는 않는다. [Rauch–Tung–Striebel smoother](https://doi.org/10.2514/3.3166), [forward/backward fixed-interval assimilation](https://doi.org/10.1175/MWR-D-10-05025.1)

중요한 반증이 있다. `p2_boundary_registered_prior_20260827_v1`은 이전 해 prior를 양쪽 7일 boundary bias로 등록했으나 두 블록 모두 candidate가 크게 악화했고 oracle alpha도 음수였다. 새 bridge는 previous-year prior·GBM·alpha 탐색을 제거한 단순한 frozen-anchor residual이라 완전히 같지는 않지만, **target flank 정보원은 부분 중복**이다. 따라서 이 실험을 실행한다면 새 모델 탐색이 아니라 family를 마지막으로 닫는 cheap falsification이어야 한다.

### P2 가장 싼 preflight

- 기존 세 61-day block, frozen alpha50+threeway-crossfit-veto anchor만 사용한다.
- 양쪽 72h anchor prediction을 truth 보기 전에 seal하고, 그 뒤 관측 residual median만 계산한다.
- window/taper/cap/blend grid는 두지 않는다.
- pooled `Delta RMSE <= -0.0020 C`, Sep-Oct `<= -0.0015 C`, 최소 2/3 block 개선, block 악화 `<= +0.0005 C`, layer 악화 `<= +0.0010 C`, paired day bootstrap CI90 upper `< 0`을 모두 요구한다.
- 하나라도 실패하면 boundary-flank family를 닫는다.

M2 이외 `S2/N2/K1/O1` fixed-frequency partial-coherence 검사는 비용이 작지만, 기존 M2의 계절 위상 불안정 반증 때문에 실행 우선순위가 더 낮은 sentinel로만 남긴다.

## P3 — predicted future forcing이 유일하게 남은 정보축인 이유

기존 저장소에는 과거 48h의 lag/window/slope/FFT/wave-age/wind-input proxy와 0–12h wind-wave lag가 이미 있다. wind-wave memory exact 실행은 17 cases에서 G/I가 좋아졌지만 S-ORS가 `+0.005840m` 악화해 shadow gate에서 종료됐다. KMA uniform alpha는 약 0.425 부근에서 포화했고, 14개 adaptive weight 구조도 모두 악화했다. 따라서 과거 lag를 더 추가하거나 alpha를 미세 조정하는 것은 돌파구가 아니다.

반면 `future_wspd`, `target_wspd`, `predicted_future_wind` 계열은 코드 전체에서 0건이었다. 파랑은 공간 전파, wind input, nonlinear interaction, dissipation의 결과라는 [NOAA WAVEWATCH III 문서](https://polar.ncep.noaa.gov/waves/wavewatch/manual.v4.18.pdf)의 구조와도 맞는다. 다만 한 지점의 과거만으로 원격 swell이나 미래 바람을 완전히 복구할 수 없으므로, 모델부터 만들지 말고 정보 상한을 먼저 잰다.

### P3 가장 싼 preflight

- KMA가 존재하는 frozen historical OOF 교집합 179 cases, 18h·24h만 활성화한다.
- Control: frozen KMA + 현재/과거 저차원 wave state.
- Treatment: control + historical train의 **실제 미래 u/v 경로**.
- 동일 fold, 78h embargo, control-only에서 고른 ridge 규제를 양 arm에 공유한다.
- 3/6/9/12h는 exact no-op이다.
- six-lead pooled `Delta RMSE <= -0.006m`, CI90 upper `< 0`, 최소 2/3 fold와 2/3 station 개선, 18h·24h 각각 non-degrade, 최악 station×lead `<= +0.003m`을 모두 요구한다.

이 oracle이 실패하면 noisy deployable forcing은 더 좋아질 수 없으므로 미래 바람 예측기와 MOS를 모두 중단한다. 통과할 때만 48h context에서 +3…24h의 `Delta u/v`를 예측하는 pooled multi-output ridge를 만들고, persistence 대비 18/24h wind-vector MSE skill 5% 이상을 요구한다.

## 최종 우선순위와 실행 경계

| 순위 | 문제 | 지금 실행할 것 | 예상 비용 | 통과 후 다음 단계 | 지금 공식 제출 |
|---:|---|---|---|---|---|
| 1 | P1 | alternate-tiling disagreement audit | CPU 수분 | 5-epoch paired-view one-shot | 금지 |
| 2 | P3 | perfect-future-wind oracle | CPU 2–5분, RAM <2GB | train-only wind forecast + frozen-KMA MOS | 금지 |
| 3 | P2 | fixed boundary residual bridge | CPU 10–30분 | family close 또는 별도 fresh replication | 금지 |

P1의 audit와 P3의 oracle은 모델 학습 전에 병렬 실행할 수 있다. P2는 과거 boundary prior 실패를 감안해 두 결과보다 뒤에 둔다. 모든 축은 one-shot, 결과 기반 grid 0, 공식 파일 접근 0을 유지한다.

## 한계

- historical folds가 많은 연구에 반복 노출돼 adaptive selection bias가 남아 있다.
- Gemini 보고서는 실제 저장소 접근이 불완전해 제안 자체를 근거로 쓸 수 없다.
- P1 window-phase의 실제 disagreement, P3 future-wind oracle의 상한, P2 단순 residual bridge는 아직 계산하지 않았다.
- 로컬 metric과 공식 점수의 수송 배율은 문제·후보별로 안정적이지 않다. 따라서 작은 local 개선을 자동 폐기하지 않되, 방향 일관성·station/fold 분산·CI를 함께 요구한다.

## 연구 중단 기준

각 문제에서 (1) 권위 있는 1차 근거, (2) 현재 코드와의 연결점, (3) 기존 실패축의 직접 반증, (4) 가장 싼 kill gate가 모두 확보됐다. 추가 문헌 검색은 이미 닫힌 모델 계열의 이름만 바꿀 가능성이 높으므로, 다음 정보는 실제 세 preflight 실행에서 얻어야 한다.
