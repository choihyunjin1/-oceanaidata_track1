# P3 배포 데이터 전용 incumbent 복원

## 결론

2026-09-01 최신 운영진 규정에 따라 KMA·ERA5·Chronos 및 그 prediction을 물려받은 P3 계보를 비적격으로 분리했다. 외부자료가 공식 제출에 처음 쓰이기 전의 최고 규정 준수 기록은 `P3_REFINED_PUBLIC_OPTIMUM_20260827`이다.

| 항목 | 확정값 |
|---|---:|
| 공식 제출 시각 | 2026-08-27 23:36:30 KST |
| Public RMSE | 0.583892 m |
| 공식 점수 | 24.066168 |
| CSV SHA-256 | `ea65370a5c9291868769ad9e54a54707035dc93a01ffa4772d9fd26342f357aa` |
| CSV bytes | 39,822 |
| 판정 | `ACTIVE_CLEAN_INCUMBENT` |

보관된 CSV는 2026-09-01에 prediction 값을 출력하거나 hidden truth를 열지 않고 byte hash만 다시 계산했다. 재계산 SHA-256은 당시 manifest와 일치했다. 제출 CSV는 Git에 복사하지 않고 기존 로컬 보관본을 유지한다.

## clean 계보

후보는 배포 데이터로 학습한 로컬 CatBoost/router와 persistence 계보의 두 prediction `O`, `A`만 사용한다.

- `O`: `output/2026-08-20/ready/P3_submission.csv`, SHA-256 `d89e69b940c90ea1fbecf1e882bee69136255fffb12601d2fc853d032900e5b7`
- `A`: `artifacts/p3_corrected_fixed_long_shrink_v4/candidate/submission.csv`, SHA-256 `607f7cd4ed2c126d5aa4eb6d8130a651ac465a0c88b4e74c112d585c3421d708`
- builder: `scripts/build_p3_refined_public_optimum_20260827.py`, 2026-09-01 재감사 SHA-256 `505f8e4f1bd6ae63e873c3b706887fcdd34ec0ff512abe426367c9398bc14301`
- 공식점수로 고정한 `alpha=-10.21743189862218`을 12/18/24h에만 적용하고 3/6/9h는 `O`와 byte-level prediction identity를 유지한다.

O/A 원류 manifest는 대회 배포 `train_wave.csv`, `train_atmos.csv`, `test_context.parquet`, `test_index.csv`, `sample_submission.csv`, `baseline_persistence.csv`와 배포 데이터에서 scratch 학습한 모델만 기록한다. ERA5, KMA, Chronos, NASA POWER 또는 외부 pretrained weight의 실사용 참조는 0건이다. `I-ORS` 문자열은 배포 데이터의 정점 범주이며 외부 I-ORS archive를 뜻하지 않는다.

## 시간순 반증

- 2026-08-25~27 clean 공식 후보는 RMSE 0.607071, 0.599072, 0.583892 순으로 개선됐다.
- 같은 8월 27일 `alpha=-12` 후보는 RMSE 0.584611로 더 낮은 성적이었다.
- refined optimum은 기존 24.066167을 24.066168로 갱신했다.
- 최초 외부자료 공식 제출은 2026-08-28 13:57:12 KST의 ERA5 후보이며, 그 영수증도 직전 champion을 RMSE 0.583892·24.066168점으로 기록한다.
- KMA 공식 제출은 그보다 늦은 2026-08-28 23:44 KST부터다. 8월 21일 KMA artifact는 `withdrawn_do_not_submit`, 공식 업로드 0이었다.
- 이후 KMA 후보의 더 높은 점수는 최신 규정상 비준수이므로 incumbent 비교에서 제외한다.

## 운영 결정

1. 이 후보를 P3의 active clean incumbent로 사용한다.
2. 새 P3 실험은 이 clean 계보에서만 시작한다.
3. KMA·ERA5·Chronos와 KMA prediction을 참조한 v21 이후 계보는 연구 증거로만 보존하고 학습·선택·제출·재현에 사용하지 않는다.
4. incumbent 교체 전에는 배포 데이터 allowlist, scratch initialization, external/pretrained reference 0, hash 및 runtime을 독립 QA한다.
5. 과거 외부자료 공식 제출의 철회·자진소명 방식은 운영진에 별도 문의한다.
