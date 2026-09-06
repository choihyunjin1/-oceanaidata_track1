# P2: 같은 CUDA C3의 학습량·정규화만 비교

결론: 학습 전 계획이다. 아직 개선을 주장하지 않는다. 기존 적격 CUDA C3 및 과거 실패/모델/attempt lock을 보존하고, 새 28 historical fits만 승인된 별도 ID에서 실행한다. 공식 입력·CSV·업로드·외부자료·hidden truth는 0이다.

## 고정 비교와 분모

| recipe | epochs | AdamW weight decay | 나머지 |
|---|---:|---:|---|
| C60 control | 60 | 0.0001 | 현재 적격 C3 그대로 |
| L120 | 120 | 0.0001 | C60과 동일 |
| D60 | 60 | 0.001 | C60과 동일 |

현재 portable C3의 11-context/8-token-feature DeepSet, normalized SmoothL1(beta=1), domain-balanced weights, input-gradient penalty 0.01, 학습률 0.001, batch 4096, 고정 3/7/14-day blockmask augmentation을 그대로 쓴다. augmentation은 원본별 총 가중치를 보존한다. 각 recipe와 control은 GPU0/CPU2/DataLoader0 동일 환경이다. 기존 CPU 대 CUDA 결과를 같은 비교군으로 간주하지 않는다.

원칙은 organizer policy → P2 must-read contract → 배포 README → 현재 v5 평가 계약이다. frozen v5 JSON의 과거 execution.fit_budget=0 표시는 원본을 바꾸지 않고 이번 별도 승인/새 config/receipt에서 28-fit 실행으로 확장한다. v5 8개 구간, two-sided 7-day purge, 원래 날짜를 유지한다. B3 2024-09-01~2024-11-01 KST natural absolute Celsius pooled RMSE가 primary다. 전체 8-fold pooled 및 각 fold/layer/자연 T5 결측/각 구간 마지막 17일 T5+S5 결측은 보조·위험 진단이다. SSE/n으로 계산하며 fold RMSE 평균을 쓰지 않는다.

모든 목표층 2/3/4의 temp와 psal을 특징 생성 전에 함께 숨긴다. 목표 truth는 별도 배열만 유지한다. target actual-depth 부재는 nominal-depth metadata로 원래 C3 동작을 유지하며 평가 행을 삭제하지 않는다. 추가 outage 후 지원되지 않는 행이 있으면 그 scenario 전체 SUPPORT_BLOCKED로 기록한다. B4/B8의 마지막 17일에 대상 행이 없으면 NOT_ESTIMABLE_NO_ROWS이며 날짜 이동 또는 다른 구간 대체를 하지 않는다.

## 선택·추가 seed와 주장 제한

첫 seed 20260901로 3 recipes × 8 folds = 24 fits. L120/D60 중 B3 natural RMSE 최소를 선택하고, exact tie는 all8 pooled RMSE, 선언 순서로 푼다. 이후 C60과 선택 challenger에만 B3 seed 20260902/20260903를 추가하여 4 fits, 총 28 fits다. seed 추가는 첫-seed 결과에 따라 임의로 취소/확장하지 않는다(90분 cap 예외).

선택에 이미 B3의 첫 seed를 사용하므로 나머지 두 seed는 새 독립 holdout이 아니라 seed 민감도 확인이다. 첫 seed, 추가 2-seed 평균, 전체 3-seed 평균을 별도로 기록한다. 전체 8 folds × 3 seeds를 실행했다고 주장하지 않는다. 후보 유지 조건은 B3 3-seed 평균 RMSE의 엄격한 개선뿐이다. CI90 또는 경험적 개선 비율 0.8, outage 무악화 같은 사후 hard gate는 추가하지 않는다. 모든 악화는 따로 공개한다. 공식 점수 상승을 보장하거나 Public 점수로 계수·정규화·epoch를 맞추지 않는다.

## 중복 감사 및 계보

- [기존 objective 2×2](../p2_objective_alignment_20260905_v2/report-source.md): 같은 normalizedHuber/absolute-C-MSE × domain/uniform 실험이 완료됐다. MSE를 다시 넣으려던 초기 제안은 중복이므로 제외했다. 당시 C 첫-seed primary 0.465330203, MSE/domain 0.516945976이었다. 이는 새 v5 8-fold 결과가 아니다.
- [기존 crossfit copula](../p2_crossfit_copula_forward_numeric_20260906_v2/report-source.md): 실패한 동일 residual family는 반복하지 않는다.
- [적격 CUDA C3 cold regeneration](../portable_cleanroom_20260906_v1/P2/report-source.md): 3 fresh full fits와 새 PID replay를 통과한 현재 baseline. 기존 답안 SHA 46d194a1ef40a1deaebd084916644d9359433d2e6ce7d5c0b53d9f515bbec071은 provenance reference일 뿐 이번 학습/평가에 읽지 않는다.
- numerical core: scripts/portable_20260906/P2/02_code/core.py SHA b18a8279a542f38f5ed550307772824ed8f597e088ad2ba7070ebfeff9020c84. 과거 답안·OOF·model을 import/load하지 않는 순수 학습/특징 함수다.
- 기존 portable run.py:306–312는 각 seed baseline + scale × normalized 후 3-seed 평균을 temp에 기록하며 PAVA/envelope projection이 없다. 이번 내부 평가도 같은 무후처리 표면이다.

과거 코드/설정의 epochs·weight_decay 지문을 조회한 범위에서는 현재 C3 blockmask/11-feature recipe의 120 epoch 또는 wd0.001 비교를 찾지 못했다. 무관한 CatBoost learning curve나 다른 구조의 정규화 실험을 동일 실험이라고 세지 않는다.

## 예산·안전·재현

기존 같은 objective 9 fits 216.703초(평균 24.078초/60epoch)로부터 이번 최대 38개의 60-epoch 환산량은 약 915초, 1.5배 여유와 준비/재현 300초를 합쳐 약 28분이다. 하드 training cap 5,400초, 실제 전체 준비/평가/QA 시간을 별도 기록한다. 최초 예정 B1/C60/seed20260901 fit은 자원 pilot이면서 결과에 재사용한다. 첫 fit runtime으로 예측 cap 초과면 성능을 보기 전에 BUDGET_PROJECTION_TERMINAL로 종료하고 자동 재시작·GPU변경을 하지 않는다.

각 fit의 새 state_dict, recipe, seed, 원래/증강 가중치 합계, 학습 키/label/array SHA, 전체 validation prediction SHA를 보존한다. resume이 아니라 exactly-once scratch 학습이다. terminal 이후 새 PID가 28개 모델 전부를 다시 로드하여 모든 natural/outage validation 예측의 exact equality를 검증한다. 128-row probe가 아니다. Python audit hook은 공식 CSV/이전 binary artifact/네트워크를 차단하지만 OS 차원의 완전한 sandbox라고 주장하지 않는다.

실행 순서: focused pytest + Ruff → seal → source-only prepare → root GPU 인계 → execute --gpu-authorized → 별도 PID replay → qa.py. root는 13:11 KST GPU0 인계를 명시 승인했다. 새로운 full-training/공식 답안 제작은 이번 28-fit 범위 밖이며 별도 판단이다.
