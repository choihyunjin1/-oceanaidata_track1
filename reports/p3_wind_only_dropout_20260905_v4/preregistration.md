# P3 v4: 바람만 결측 증강 — 고정된 단일 변경

## 변경과 기존 실패의 차이

Sep5 v1의 전체 기상 blockmask는 압력·기온·습도까지 버렸고, matched LightGBM control 0.80655461m보다 0.81544321m로 악화했다. 이번 가설은 **바람 관측의 부재를 훈련하되, 기압·기온·습도에 남아 있는 폭풍 정보를 보존하면 그 악화를 줄일 수 있다**이다. 개선은 미보장이고, wind-only 결측이 자연적으로 늘 유효한 상태라는 주장도 아니다.

변경은 원시 `wspd/gust/wdir`에 해당하는 모든 요약/방향 sin·cos/gust excess/wind-wave alignment/wind input proxy/변화량을 함께 가리는 것 하나다. hs/tp/hmax/wvdir와 caph/airt/relh는 보존한다. synthetic 289행 context의 원시 바람 세 열을 가려 1,275개 요약을 재계산한 값과 feature-level 가림이 exact 일치함을 학습 전에 확인했다.

풍속·풍향 중 관측이 남아 있는 훈련 행은 원본과 바람 결측 복제본으로 각각 기존 중량의 0.5를 부여한다. 이미 바람이 전부 없는 행은 복제하지 않고 기존 중량1을 유지한다. 모든 6lead에서 target과 원본 행별 총 중량을 보존한다. 이번 주장의 대상은 전체 기상 결측 증강과 구별되는 **완성 학습정책**이며, 더 나은 모델 계열이나 독립 시험지의 발견이 아니다.

## 비교·검증·자원 계약

- 원본 관측: 배포 `train_wave.csv`, `train_atmos.csv`만. 현재 clean source-only 재생성 cache/OOF/key를 SHA로 봉인해 재사용한다. 기존 답안 CSV, 외부 관측, Public-inverse 계수, 공식 입력은 사용하지 않는다.
- 정점별 전체 기간 first-eligible/78h, 동일 episode 재사용 제외, expanding train3fold. 동일49/79/53=181사례, 1,086행. 모든6lead의 target/key 일치 및 원본 train-derived target 대조.
- paired LightGBM control와 wind-only: 같은591features, seed20260905, 700trees, CPU2threads. 3fold×2arm=6historical fits. 새router0. clean comparator의 기존 strictly earlier-fold-only router OOF는 exact 재사용하며 현재 후보의 성능으로 재조정하지 않는다.
- `no_op`, standalone control, control50+clean50, standalone wind-only, wind-only50+clean50만 사전고정. 가장 작은 pooled SSE/RMSE, 동점은 기재 순서. 성능을 본 뒤 비율·특징·seed·학습량을 변경하지 않는다. half/no-op 이외 추가 조정/재시도 없음.
- fold/station/lead, wind observed/missing, nonwind observed/missing, 과거3h hs_min<1.5 onset proxy, case별 RMSE 변화 요약을 보고한다. onset은 정확한 첫 상승시점의 재구성이 아니며 fitting/selection에 쓰지 않는다. 안정성 악화는 별도 위험 표이고 자동 탈락 gate가 아니다.
- 30분/8GiB cap, GPU0. 기존12fit526.713초를 근거로 약4~7분 추정. 비용 초과·기술 실패 시 lock/산출물을 보존하고 자동 재시작하지 않는다.
- 반복 노출한 historical surface의 탐색이다. bootstrap은 기술통계이며 선택편향 제거/새 독립 확인이 아니다.

## 재현과 제출 경계

native model 저장/별도 프로세스의 모든 validation 예측 exact replay 및 독립 SSE 산식 QA를 한다. 이것은 **빈 `03_model`에서 전체 학습 후 `05_answer` 생성** 검사와 다르다. 이번6fit은 historical model만 만들며 full-training/공식 추론 adapter는 실행하지 않는다. 기존 모델 삭제·최종 제출본 교체·CSV·업로드·Git 작업은0.

현재 clean 기준선의 source→full-training→inference 근거는 [별도 canonical 배포 README](../p3_score_repair_deploy_20260905_v1/README.md)와 [실측 보고서](../p3_score_repair_deploy_20260905_v1/report-source.md)다. 해당 runner의 alpha10은 Ridge regularization이고 long-lead shrink0.2는 사전고정 recipe다. 기존 refined Public-alpha 파일을 호출하지 않는다. 그 runner는 TabPFN6h 후보도 함께 만드는 경로이며, 이번 v4가 새로운 빈 폴더 최종 번들 재생 검사를 수행했다는 뜻이 아니다.

실행 config: [p3_wind_only_dropout_20260905_v4.json](../../configs/experiments/p3_wind_only_dropout_20260905_v4.json). 정확한 봉인 source/cache hashes는 실행 전 [preflight.json](preflight.json)의 단일 근거를 따른다.
