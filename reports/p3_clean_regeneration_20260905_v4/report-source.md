# P3 clean 기준 모델: 빈 폴더 실제 학습부터 로컬 답안까지 PASS

## 결론

**기준 모델 재생성 검사 통과 여부: PASS.** 새 `03_model`과 `05_answer`가 비어 있음을 확인한 뒤 배포 train CSV 2개에서 특징을 다시 계산하고 CatBoost 8회와 router 3회를 실제 학습했다. 저장한 새 모델을 별도 프로세스에서 재생한 뒤 승인된 공개 입력으로 P3 답안 1,200행을 생성했다. 전체 독립 QA **133/133 PASS**, 실패0이다. 업로드는0이다.

새 답안 SHA-256은 `6bfa23d25f944df4711c11d1fce82978a96df08b58fdc57f666ac792a7da96b7`로 이전 clean 기준 답안의 대조용 SHA와 실제로 같았다. 이전 모델·OOF·캐시·답안 파일을 읽어서 복원한 것이 아니다. 동일 파일이므로 이번 재생성에서 주장할 새 점수 개선은0이며, **신규 최고점 후보가 아닌 학습 포함 재현성 증거**다. 새 공식 채점은 수행하지 않았다.

아래 세 주장은 구분하며, 이번 실행에서는 각각 통과했다.

| 검증 주장 | 실제 결과 | 근거 |
|---|---|---|
| 배포 데이터 → 빈 모델 폴더 실제 학습 → 로컬 답안 | PASS | `prepare.json`, `training-result.json`, `answer-qa.json`, `independent-qa.json` |
| 새 저장 모델의 별도 프로세스 예측/CSV 재생 | 예측 최대오차0, CSV bytes exact | `fresh-process-replay.json`, `answer-replay-qa.json` |
| 과거 clean 출력과의 일치 | 내부 RMSE drift0 및 답안 SHA exact | `regenerated-oof.json`, `answer-qa.json` |

이는 같은 머신·환경에서의 결과다. 다른 GPU/환경에서 과거와 byte-level 학습 결과가 항상 같다고 보장하지 않으며 운영진 환경의 최종 검증 PASS를 뜻하지 않는다.

## 학습 계약과 계보

고정 recipe는 `configs/experiments/p3_corrected_repeated_forward_catboost_v2.json`이다. single/multi CatBoost 각각 historical 3fold=6fit, full single/multi=2fit, 과거 완료 fold만 사용하는 router=2fit, 새 전체 OOF router=1fit을 학습했다. 최종 저장 파일은 historical6과 full single/multi/router3으로9개다. 중간 prequential router는 성능 검증을 위한 fit으로 최종 모델 파일 수와 구분한다.

배포 train 두 파일에서 24,360개 anchor와591개 특징을 새로 생성했다. full single은146,160 lead rows, full multi는24,360 cases를 사용했다. 기존 준비 캐시를 가져오지 않았고, 새 실행이 직접 만든 `04_validation`만 다음 단계에서 사용했다.

`alpha=10`은 Ridge router의 사전 고정 정규화 계수다. 장기12/18/24h persistence shrink0.2 역시 현재 결과 이전에 고정한 v2 recipe다. 과거 refined-alpha·axis contract·공식 점수의 역산값은 입력하지 않았다. 과거 RMSE/답안 SHA는 출력 대조용 metadata로만 사용했으며, 학습/예측을 그 값에 맞추는 조정·재시도는0이다. TabPFN 모델/가중치도 이번 재생성에는 포함하지 않았다.

## 내부 평가 및 독립 산술

새 source-derived target과 OOF를181 cases×6 leads=1,086행에서 다시 대조했다. pooled SSE `659.2067259186299 m²`, `sqrt(SSE/1086)` RMSE **0.7791048399763751m**이다. 기존 clean metadata와 drift0이다. 정점별78h 간격, storm episode 분리, 완전한 six-lead key, router의 과거 target-ready 시각을 독립 확인했다. 메타 학습 case 수는 각 fold에서0/49/128이며 현재 fold label을 사용하지 않았다.

고정 historical gate는 PASS다. persistence RMSE `0.8634973748601807m` 대비 ΔRMSE `-0.0843925348838056m`, 3fold 모두 개선이다. 이것은 재생성된 기존 기준선의 내부 근거이며 새로운 prospective 검증이나 공식 점수 향상 증거가 아니다.

| 단계 | 검사 / 결과 |
|---|---|
| 실행 전 focused pytest | 11 PASS; native synthetic CPU fit2 포함, historical/official0 |
| Ruff | PASS |
| 공식 입력 전 독립 training QA | 123/123 PASS, 실패0 |
| 저장 모델 fresh replay | training PID33032 → PID32884;181 cases/1,086예측; 최대오차0 |
| 로컬 답안 생성 | PID27320;200 cases/1,200행; schema/key/order/duplicate/finite QA PASS |
| 답안 fresh replay | PID38576; 생성 PID와 분리; CSV bytes exact |
| 전체 독립 QA | 133/133 PASS, 실패0 |

초기 replay CLI 호출 한 번은 `P3_DATA_DIR` 환경변수 미설정으로 main의 입력 경로 해석에서 종료됐다. 데이터 접근·학습·replay receipt 생성 전이었다. 환경변수를 지정한 실제 replay가 위 PID32884의 단일 완료 실행이다. 학습 lock을 재사용하거나 학습을 재시작하지 않았다.

## 실측 시간과 자원

| 단계 | 초 |
|---|---:|
| 배포 train source 재처리/특징 생성 | 369.8508895000 |
| historical6 + full2 + router3 학습 | 1,279.8840523000 |
| 새 저장 모델 local replay | 0.3545261000 |
| 공개 입력 → 첫 로컬 답안 | 3.9170636000 |
| 위 핵심 단계 합 | **1,654.0065315000 (약27.57분)** |
| 별도 답안 exact replay 추가 | 3.8615324000 |

`answer-qa.json`의 `source_train_replay_inference_seconds`는 **1,654.0021373000초**다. 독립적으로 단계 receipt의 표시된 시간을 합하면1,654.0065315000초로 약0.0044초 차이가 난다. replay timer를 기록하는 시점 차이이며, 원 receipt를 변경하지 않았다. 6시간 이내라는 결론에는 영향을 주지 않는다. 이 합계는 단계별 측정 시간으로 GPU 예약 대기, CLI 시작/종료, 별도 pytest/QA 시간까지 포함한 연속 wall-clock 완료 시간은 아니다.

모델 CPU2threads, training process는 logical CPU2개(affinity mask3)로 제한했다. 고정 historical helper의 일부 native predict가 thread_count를 생략하기 때문에 프로세스 전체 상한을 둔 것이다. MultiRMSE GPU 학습 설정은 고정 recipe 그대로다. full single248.456766초, full multi32.849190초였다. 학습 worker/launcher 종료와 GPU 해제를 확인한 후 다른 문제에 GPU를 넘겼다. 현재 머신의6시간 제한 여유를 실측했으며 운영진 하드웨어에서는 아직 검증하지 않았다.

## 로컬 사용 및 제출 범위

답안 파일은 `artifacts/p3_clean_regeneration_20260905_v4/05_answer/submission.csv`다. **P3 / OCN-03 파고 예측 리더보드의 CSV 입력에 사용하는 파일**이며 열은 `case_id,station,lead_h,hs_pred`다. 모델 파일을 CSV 입력란에 올리지 않는다. 현재 답안은 새 점수 개선 파일이 아니라 이미 확인한 clean 출력의 재생성본이므로 중복 제출을 권하지 않는다. 이번 에이전트의 upload는0이다.

원본 데이터·실행 코드·모델·검증 자료·답안은 `01_data`/`02_code`/`03_model`/`04_validation`/`05_answer`로 나뉜다. 상세 실행 명령과 경로는 [README.md](README.md)에 있다. 이미 소비된 prepare/training/inference lock을 지우거나 같은 출력 폴더에서 다시 학습하지 않는다. 이후 새 재현 실행은 별도 새 checkout/출력 계약을 준비해야 한다.

공식 입력은 training QA PASS 이후 승인된 공개 predictor/key만 읽었다. 첫 추론 및 fresh answer replay 각각 공개 context57,800행/index1,200행을 읽었으며 sample/hidden/이전 답안 읽기/외부 관측/사전학습 가중치/업로드는0이다. 공개 입력을 학습이나 선택에 재사용하지 않았다.

`02_code`는 SHA 대조용으로 `src/p3_wave` 전체를 담은 넓은 연구 snapshot이다. 미실행 ERA5/Chronos/Public계보 코드 텍스트가 포함되어 있으나 이 실행의 해당 모델·외부 데이터 로딩은0이다. 현재 snapshot을 최종 portable 제출 ZIP으로 표시하지 않는다. 최종 패키지는 실행 의존성만 선별해야 하며, 독립 경로 이동·환경 설치·OS 차원의 네트워크 차단·운영진 하드웨어6시간 검증은 아직 미완료다. 기존 최종 제출 폴더나 모델을 덮어쓰지 않았다.

## 보존한 근거 SHA-256

| 파일 | SHA-256 |
|---|---|
| runner | `9933a4559216897643d2f03185db29bb1080eed769ee103dbe7b66e868b8442b` |
| config | `6bb67fe780579c2c538ffea7970ca8266e5204e957fca910def395015ba0e586` |
| training-result.json | `8cefe4823f6cfc13bca8889e1135fa888e65bbc91466e2a0e3d97b9ced778540` |
| regenerated-oof.json | `e8d6b3a9ee1ff2c195256ddb9c77b517eefd250d3c22c5c3bb4ddf6519ede636` |
| fresh-process-replay.json | `99dc2f5c74261cee85f57406ae079c1fbb694076ae33183fc5e62671fb3040d1` |
| training-independent-qa.json | `4017d462fa03d4fed76e80035ae59d829a6c0c7549ca800f406f079d20684e13` |
| independent-qa.json | `5850cc130731153f69698a30864380d4be657a3e7e5b1d3f9cdc04cd074795ed` |
| answer-qa.json | `15388843722bd78350ec38fbda1c1a8c99b7a92c558d83a8f7fcdcc1b2239d77` |
| answer-replay-qa.json | `7726470a13aae383737280e4ccfa1e0c669d1f218dda8aa0aa61d6649593fbb1` |

SHA/경로 가드와 독립 산술은 실제 실행 근거이며, 허용 데이터 사용에 대한 OS-level forensic 보증이나 최종 심사 완료를 대신하지 않는다. 코드·config·원 QA 및 학습 산출물은 검증 결과를 맞추기 위해 수정하지 않았다.
