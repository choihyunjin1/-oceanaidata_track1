# P2 구조·대형 모델 정찰 — 2026-08-16

## 실행 결과 업데이트

정찰 후보 8개 계열을 동일한 69,850행 target-proxy OOF에서 실제 비교했다. 모든 모델은 공개 layer 1·5·6·7·8과 공개층 기반 선형보간, 시간·조석 특징만 입력으로 사용했고, 목표 layer 2·3·4의 temp·psal은 입력에서 제외했다. 외부 관측값과 pretrained weight는 사용하지 않았다.

- 단독 최강: 3-seed `LSTI-style`, RMSE `0.764545°C` (1-seed tournament `0.771751°C`)
- 단일 계열 blend 최강: `TimeMixer++-style 50% + frozen tree 50%`, RMSE `0.756885°C`
- 최종 layer별 convex stack: RMSE `0.745814°C`
- 기존 400-round router: RMSE `0.788890°C`
- 최종 개선: `-0.043076°C`
- leave-one-block-out stack: `0.775660°C`
- KST-day paired bootstrap 90% CI: `[-0.055540, -0.030873]°C`
- CSDI-style·SSSD-SSM-style은 최적 tree blend에서 deep weight가 `0`으로 선택되어 기각

최종 후보 `submissions/p2/P2_DEEP_STACK_V1.csv`는 26,061행·키 순서·유한 범위를 통과했고 저장 가중치에서 SHA256 `ea5cedbd08817da4da00274e1078689f09a1d9c65d2a464f5f5f5ba9ffcc82e8`로 완전 재현됐다. 아직 업로드하지 않았다. 아래 내용은 실행 전 설계 근거이므로, 후보 우선순위보다 이 실행 결과가 최신 결론이다.

## 결론

P2의 다음 1순위는 파라미터 수만 키운 범용 Transformer가 아니다. 현재 LightGBM의 강한 출발점인 수심 선형보간 잔차를 유지하면서, 다음 세 구조를 결합한 **Depth-query Bidirectional Multi-scale Residual Network**가 가장 적합하다.

1. 공개 layer 1·5·6·7·8의 동시각 수온·염분·실제 수심을 읽는 수직 프로파일 encoder
2. 61일 blackout 전후와 12.42시간 조석을 함께 읽는 양방향 multi-scale TCN/TimeMixer++ 계열 temporal encoder
3. 목표 수심 7.04·9.44·14.74 m를 연속 좌표로 질의하는 DeepONet식 depth decoder

모델은 선형보간 온도를 직접 대체하지 않고 세 목표층의 `정답 - 수심 선형보간` 잔차를 출력한다. 이 방식은 현재 검증된 baseline과 M2 이득을 보존하면서, 층별 독립 tree가 표현하기 어려운 수온약층 곡률과 시간 연속성을 공동 학습한다.

이 문서는 문헌·공식 구현을 P2에 매핑한 연구 설계다. 아래 논문의 공개 benchmark 개선율은 해당 데이터셋의 결과이며 P2 예상 개선율이 아니다. P2에서의 우열은 동일 blocked validation RMSE로 새로 측정해야 한다.

## P2에 가장 직접적인 문헌 근거

- [Long Short-Term Imputer, TMLR 2025](https://openreview.net/forum?id=9NVJ0ZgEfT)는 짧은 결측과 긴 연속 결측을 별도 imputer로 처리하고, forward/backward 예측의 consistency와 meta-weighting으로 결합한다. 논문은 5개 데이터셋에서 기존 deep model 대비 평균 오차 57.4% 감소를 보고했다. P2의 61일 연속 blackout과 가장 직접적으로 맞지만, 그대로 가져오기보다 목표층 경계 문맥 branch의 설계 근거로 쓰는 편이 낫다.
- [DeepONet, Nature Machine Intelligence 2021](https://doi.org/10.1038/s42256-021-00302-5)은 관측된 입력 함수를 branch net으로, 출력 위치를 trunk net으로 표현해 연속 operator를 학습한다. P2에서는 공개 수심 프로파일을 branch 입력, 목표 수심을 trunk query로 두면 layer 번호가 아니라 실제 깊이에 대해 복원할 수 있다.
- [TimeMixer++, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2b187165e28fdfdc0ffb34d1bfff2b0c-Abstract-Conference.html)는 시간·주파수의 여러 해상도에서 계절·추세 패턴을 분해하고 혼합한다. 공식 imputation 실험은 길이 1,024에서 12.5–50% 무작위 masking이므로 8,784-step 61일 blackout에 대한 직접 증거는 아니지만, M2·일주기·계절 전이를 한 backbone에서 표현하는 근거가 된다.
- [ModernTCN, ICLR 2024](https://openreview.net/forum?id=vpJMJerXHU)과 [공식 구현](https://github.com/luodhhh/ModernTCN)은 순수 convolution 구조로 forecasting·imputation 등 5개 과제에서 성능과 효율의 균형을 보고한다. P2처럼 단일 기지·소수 채널·긴 문맥에서는 full attention보다 안전한 첫 temporal backbone이다.
- [ImputeFormer, KDD 2024](https://doi.org/10.1145/3637528.3671751)과 [공식 구현](https://github.com/tongnie/ImputeFormer)은 low-rank prior와 Transformer 표현력을 결합하고 block-missing 실험 설정을 제공한다. P2의 층을 graph node로 놓을 수 있으나, 로컬 rank-3 EOF가 실패했고 강한 수온약층 곡률이 있어 과도한 smoothing 위험을 별도로 봐야 한다.
- [SSSD-S4, TMLR](https://openreview.net/forum?id=hHiIbk7ApW)과 [공식 구현](https://github.com/AI4HealthUOL/SSSD)은 diffusion과 structured state-space model을 결합하며 blackout missing 시나리오를 직접 다룬다. [CSDI, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html)도 관측값 조건부 diffusion을 명시적으로 imputation에 학습한다. 둘 다 높은 상한선 후보지만, 공식 지표가 확률 예측을 포함하고 sampling이 필요하므로 P2의 단일 RMSE에서는 posterior sample 평균이 실제로 tree보다 낮은지 확인해야 한다.
- [MOMENT, ICML 2024](https://proceedings.mlr.press/v235/goswami24a.html)과 [UniTS, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fe248e22b241ae5a9adf11493c8c12bc-Abstract-Conference.html)는 범용 사전학습 time-series model이 imputation을 지원함을 보였다. 그러나 P2에는 불규칙한 수심 좌표와 해양 수직 구조가 핵심이고, 외부 pretrained weight 허용 여부도 확인되지 않았다. 따라서 첫 실험이 아니라 후순위 challenger다.
- [물리 guidance를 결합한 해양 T/S 복원 연구, Remote Sensing 2025](https://www.mdpi.com/2072-4292/17/17/2954)는 제한된 해양 subsurface 자료에서 Transformer가 CNN보다 약할 수 있고 물리 guidance를 넣은 CNN이 개선됨을 보고했다. 이는 P2에서 모델 크기보다 수심 좌표·baseline residual·조석 구조를 먼저 넣어야 한다는 직접적인 도메인 근거다.

## 후보 우선순위

| 순위 | 후보 | P2 구조 적합성 | 계산량 | 현재 판단 |
|---:|---|---|---|---|
| 1 | Depth-query BiTCN residual hybrid | 매우 높음 | 중~높음 | 첫 구현·튜닝 대상 |
| 2 | ImputeFormer + depth graph + residual head | 높음 | 높음 | 공식 block-missing benchmark로 비교 |
| 3 | SSSD-S4 또는 CSDI posterior mean | 중~높음 | 매우 높음 | 최대 성능 상한·앙상블 다양성 후보 |
| 4 | TimeMixer++ 단독 | 중간 | 높음 | temporal backbone ablation |
| 5 | MOMENT/UniTS fine-tuning | 중간 이하 | 높음 | 외부 weight 허용 확인 뒤 challenger |
| 6 | FNO 단독 | 낮음 | 높음 | sparse irregular-depth P2에는 부적합 |

## 1순위 구조의 구체 설계

### 입력과 출력

- 매 시각 공개 layer 1·5·6·7·8의 `temp`, `psal`, 실제 `depth`, 값·결측 mask
- 선형보간 세 목표층 온도와 공개층 상하 contrast
- 연중·일주기와 M2 12.42시간 harmonic, 기존 lean-M2 특징
- 가림층 값은 관측된 학습 구간에서 정답으로만 쓰며, 복원 입력에서는 세 층을 동시에 가린다.
- 출력은 layer 2·3·4 각각의 선형보간 대비 잔차다.

### 네트워크

1. **Vertical set encoder**: 각 공개층을 `(temp, psal, depth, mask)` token으로 만들고 2–3개 attention/set block으로 동시각 수직 상태를 인코딩한다.
2. **Bidirectional temporal backbone**: 1시간 patch와 6시간 patch를 병렬 사용하고, dilated depthwise TCN 또는 ModernTCN block으로 61일 전체 receptive field를 확보한다. 목표층 blackout 전후 관측은 별도 forward/backward context로 넣고 consistency loss를 둔다.
3. **Depth-query decoder**: 목표 nominal depth와 실제 공개 depth를 Fourier/MLP encoding한 trunk vector를 temporal branch vector와 결합해 세 깊이의 잔차를 출력한다.
4. **Residual skip**: 최종 온도는 `baseline_interp + predicted_residual`이다. 초기 head를 0에 가깝게 두어 학습 시작점이 기존 선형보간보다 급격히 나빠지지 않게 한다.

### 첫 실험의 고정 규모

- 128 hidden channels, 8 temporal blocks, kernel 7, dilation 1–128, 4 attention heads
- 약 3–8M parameters를 목표로 하며 RTX 5090 32 GB에서 bf16 mixed precision 사용
- AdamW, learning rate 후보 `{1e-4, 3e-4, 1e-3}`, weight decay `{1e-4, 1e-3}`
- 최대 300 epochs, patience 30, gradient clipping 1.0
- batch는 61일 sequence 기준 VRAM smoke로 결정하고 gradient accumulation으로 effective batch 16 이상 유지
- seed 3개는 구조·학습률을 고른 뒤에만 재학습한다.

최대 epoch는 학습을 강제로 끝까지 쓰는 값이 아니라 상한이다. train loss, target-proxy RMSE, 2024 same-season RMSE를 매 epoch 저장하고 best checkpoint를 복원한다. LightGBM 5,000-round 결과처럼 후반 과적합이 나타날 수 있으므로 마지막 epoch 모델을 자동 채택하지 않는다.

## 결측 masking과 검증

- P2 observations에는 61일 동안 목표 3개 층과 공개층 조건이 모두 완전한 연속 창이 없다. 반면 exact 10분 cadence, 공개층 95% 이상, 목표 정답 95% 이상 조건의 61일 창은 19,595개 endpoint, 6시간 stride 약 545개다.
- 따라서 training loss는 관측된 목표 정답에만 적용하고, 3개 목표층을 같은 중앙 구간에서 함께 가리는 structured mask를 사용한다.
- mask 길이는 `{6시간, 1일, 7일, 30일, 61일}`을 혼합하되 61일과 30일의 비중을 높여 실제 blackout을 반영한다.
- validation block의 목표층은 입력에서 전부 제거하며, validation labels는 epoch 선택용 RMSE에만 쓴다.
- 최종 비교는 현재 400-round router의 target-proxy RMSE `0.7888895064°C`와 같은 69,850행에서 한다. standalone과 `tree/deep convex blend`를 모두 비교하되 blend weight는 validation OOF에서만 정한다.

## 기대효과를 정직하게 해석하는 기준

- **모델이 무거워서 생기는 이득**: 더 긴 시간 문맥, 비선형 수직 상호작용, joint three-layer output을 동시에 표현할 capacity가 증가한다.
- **구조가 좋아서 생기는 이득**: 선형보간 residual, 실제 depth query, 동시각 공개 프로파일, M2 multi-scale, 양방향 blackout context가 P2 생성 구조와 맞는다.
- P2에서는 두 번째 효과가 먼저다. 외부 benchmark의 개선율을 P2 RMSE 예상치로 옮길 수 없으며, 단순 parameter 증가는 2개 연도라는 작은 독립 계절 표본에서 오히려 과적합할 수 있다.
- 첫 실험의 성공 기준은 standalone 또는 inner-selected blend가 `0.7888895064°C`보다 낮고, 세 target-relevant block 및 세 층의 오차 구조를 함께 설명하는 것이다. 수치가 낮지 않으면 더 큰 diffusion/foundation model로 바로 확장하지 않는다.

## 실행 순서

1. Depth-query BiTCN residual hybrid 6개 optimizer 조합을 최대 300 epoch로 screen한다.
2. 최적 구조 1개를 3 seeds로 재학습하고 current router와 OOF convex blend를 계산한다.
3. 개선이 확인되면 같은 masking/data contract로 ImputeFormer를 한 번 비교한다.
4. tree와 deterministic deep model의 오류 상관이 충분히 다를 때만 SSSD-S4/CSDI posterior mean을 마지막 상한선 실험으로 실행한다.
5. pretrained MOMENT/UniTS는 운영진이 외부 weight 사용을 허용한 경우에만 별도 challenger로 둔다.
