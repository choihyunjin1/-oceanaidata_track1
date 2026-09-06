# P3 hmax-only full cold candidate

결론: **새 빈 모델 폴더에서 원자료→12 backbone+5 router 학습→새 PID QA→공식 양식 1,200행 답안→다른 PID exact replay를 1회 완주했다.** 총 1,611.919초(26.87분)로 6시간 상한 이내다. 기존 cache/OOF/model/answer 입력·sample 값·hidden·이 에이전트의 upload는 0이다. 공식 입력은 학습 QA PASS 뒤 승인된 추론 단계에서만 읽었다.

보존 완료본: [`artifacts/p3_forward_candidate_cold_20260906_v1/completed/P3`](../../artifacts/p3_forward_candidate_cold_20260906_v1/completed/P3). 답안 SHA `d45605922ae8ce699c07405d8361765290c39bd38ddd0123f4e2e9ccd01808c5`. [완결 영수증](completion.json)과 [빈 모델 재학습용 ZIP 검증](cold-archive-qa.json)을 참조한다. 공식 점수·업로드 여부는 root의 별도 공식 영수증에서만 판단한다.

새 cold 내부 unweighted pooled RMSE는 **0.68246927556 m / 103,602행**이다. 새 PID가 SSE/N 산식·키·예측과 전체 모델 in-sample probe를 재계산해 일치를 확인했다. 이것은 기존 hmax 실험의 controlled Δ/CI와 구분되는 재생성 수치이며 공식 예상 점수로 역산하지 않는다.

| 완료 단계 | PID | prepare부터 누적 초 |
|---|---:|---:|
| 새 원자료 특징/지원 QA | 33104 | 331.502 |
| 12 backbone + 5 router 학습 | 39980 | 1504.031 |
| 전체 모델/OOF 새 프로세스 QA | 39396 | 1555.832 |
| 로컬 1,200행 답안 | 9712 | 1597.696 |
| 답안 새 프로세스 byte-exact replay | 31216 | 1611.919 |

위 시간은 작업자 간격을 포함한 누적 시간이다. completed 폴더의 63개 파일/178,371,470 bytes를 원 실행 폴더와 모두 SHA 대조해 그대로 보존했다. 16개 합성 tests/Ruff PASS이며, cold ZIP은 새 폴더 실제 추출/동일 manifest 16파일/독립 `-I` 진입점까지 점검했다. **두 번째 전체 재학습이나 새 OS·오프라인 환경 인증은 수행하지 않았다.** 장기 saved-model-only 재생은 별도 `P3_forward_saved_v1` 담당 경로이며 이 완료 폴더의 소모된 clock/lock을 재시작하면 안 된다.

## 실행 계약

- 새 OS 임시 폴더에서 배포 `train_wave.csv`와 `train_atmos.csv`만으로 591개 특징/24,360개 anchor를 새로 생성한다. 기존 cache/OOF/model/answer 입력은 0이다.
- 단일 hmax 제거 후보: categorical single lead 유지, 64개 hmax 파생 base 특징 제거로 527개, router의 hmax_current 제거. numeric 변경과 혼합하지 않는다.
- v5의 정확한 5개 chronological fold/78h purge/episode exclusion을 유지한다. historical single/multi 10 fit + prior OOF router 4 fit, full single/multi 2 fit + 전체 새 OOF router 1 fit, 총 **12 backbone + 5 router**로 상한 고정한다.
- CPU 2 threads, multi는 독점 GPU device 0. prepare 시작부터 QA·추론·재현·작업자 대기까지 21,600초이며 watchdog으로 강제 제한한다. 실패 시 같은 lock을 재시작하지 않는다.
- 모델/학습/OOF 해시와 별도 PID historical 및 full probe replay PASS 전에는 공식 입력을 읽지 않는다. PASS 후 승인된 local 1,200행 답안 생성과 다른 PID의 byte-exact replay만 수행하고 upload는 root 소유다.
- full probe는 in-sample 결정론 진단이며 성능 평가가 아니다. 새 GPU 학습 답안에 과거 SHA의 공식 점수를 이전하지 않는다. 전체 cold 실행은 1회이며 서로 다른 OS/새 오프라인 환경 인증은 별도 미검증이다.

## 선택 근거와 상수 계보

Root가 기존 hmax historical 결과/독립 QA를 확인한 후 이번 완성 후보로 hmax를 선택했다. 기존 결과 SHA `d575a1214d31af5220679d48dd90ea67fe54babdca00d2956f3724f29b74d1aa`, QA SHA `26bfc5076e9a05c80dd170bac5830569e1f537d678dd30716a3a98fdb6505c10`이다. 이것은 새 독립 가설 탐색이 아니며, 기존 수치는 이번 새 cold 재생성의 성능으로 표시하지 않는다.

Long-lead shrink 0.2는 이번 계약에서 고정한 기존 local-adaptive 상수다. 최초 2026-08-17 train OOF의 제한된 0.15/0.20/0.25 진단 후 선택되었으며 virgin holdout 선택·물리상수·이번 학습 산출물 또는 Public 점수 역산 계수가 아니다. 출처 메타데이터만 `06_docs/shrink-provenance.json`에 바이트 그대로 동봉한다. [기존 출처 기록](../portable_cleanroom_20260906_v1/P3/report-source.md)의 0.2 출처 절을 참조한다. 과거 OOF 원시 값은 열지 않았다.

## 사전검토

- P3 담당 및 P2 독립 리뷰에서 source→fold→full→QA→infer의 닫힌 계보를 검토했다.
- P2가 single historical replay의 lead-major reshape 오류를 발견하여 학습 전에 원 `fit_one`과 동일한 anchor_id/lead_h 정렬을 적용했다. 서로 다른 3 anchor×6 lead 합성검사로 재발을 막았다. 성능값을 보고 수정한 것이 아니며 실제 fit은 0이었다.
- 정확 recipe SHA, 복사 source SHA, 고정 router/shrink/ensemble 계수, 12/5 실제 모델 수, 비어 있는 모델 폴더, 591/527 및 native categorical 타입, 48h 경계/78h·episode 배제, 공식 입력 승인 gate, 원자료 시작/끝/QA 후 SHA 재검사를 포함한다.
- `preflight-tests-final-corrected.xml`: 16/16 PASS. 앞선 합성 fixture 실패 기록도 보존한다. 원 helper/source/기존 실험 파일은 변경하지 않았다.

실행 코드: `scripts/portable_20260906/P3_forward_candidate_cold_v1/`. 실제 봉인·PID·단계별 시간·결과 SHA는 각 새 실행 영수증에 기록하고 완료 후 이 문서를 갱신한다.

## 실행 진행 영수증

새 독립 실행 폴더는 `C:/Users/cedis/AppData/Local/Temp/p3_forward_candidate_cold_20260906_v1/P3`이다. manifest SHA `84e47e0954c3c3fad207230855fece56163d48fe96de31c1bc9e35fb9710773c`, runner SHA `474e988bf8051a72f2fe0afa8cf520b57151aa681d5d96497a6723beff3db6de`를 실행 전에 봉인했다.

Prepare PID 33104, 시작 2026-09-06 07:39:13 KST. 24,360 anchor/591 features와 45개 raw-context 검사를 통과했다. train/validation 수는 각각 `[7057,7912,10665,15974,22808]` / `[1051,2628,5388,6708,1492]`이다. `06_docs/prepare.json` SHA `8f2b8541728bba387d6cf2f2a04bf551fca4174bcf48e75287efe1964385d2c1`; 새 생성 features SHA `2cbfbc7e521c395a17d9595982fa5773fb0ad8e9c5ca5484f392769e9dff0c3f`는 이전 source-only cache의 기록된 해시와 같지만 기존 cache를 읽거나 복사한 결과가 아니다. 새 anchor 파일에는 raw episode 계보를 추가했다.

Training PID 39980에서 명시적 `--variant hmax`로 시작하여 정상 완료했다. 전체 실행 동안 설정·상수·후처리 변경 및 추가 학습은 없었다. 새 재학습 ZIP을 사용할 때도 반드시 `train --training-approved --gpu-approved --variant hmax`를 명시한다; numeric은 이번 실행/후보가 아니다.

재현 영수증은 보존본 `06_docs/prepare.json`, `training-result.json`, `training-qa.json`, `answer-qa.json`, `answer-replay-qa.json`에 있다. 학습 QA SHA `a6bdc91b1d4f386522619565d66122848ebddfb9ab8cbc952d5b2be7c7274788`, 답안 replay QA SHA `82a645eeb1e02b0e028df8b51ec742f06ef60eb44be935abb4f952940b868a08`이다.
