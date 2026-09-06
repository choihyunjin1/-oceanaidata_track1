# P2 결론 — CPU 기준선과 profile-copula 후보의 로컬 제출·재현 패키지 준비 완료

배포 관측만으로 **빈 `03_model`에서 C3 3개와 covariance/CDF 1개를 새로 학습**했고, 학습 검증→26,061행 답안 생성→별도 PID 전체 답안 재현→ZIP 새 추출 후 재현까지 완료했다. 독립 출력 QA **73/73 PASS**다. 원 저장소 밖의 별도 임시 폴더에서 실제 실행한 뒤 동일 hash로 로컬 artifact에 보존했다. 기존 모델·답안·실패 attempt는 삭제하거나 덮어쓰지 않았다.

이 패키지는 내부 평균 개선이 확인된 **in-sample strength1.0** 후보와 **동일 실행 CPU C3 control**만 담는다. 실패한 crossfit 후보의 full/inner 학습은 하지 않았다. 기존 CUDA fallback SHA `46d194…c071`과 이번 CPU control은 다른 파일이므로 옛 공식 RMSE/점수를 옮겨 붙이면 안 된다. 담당 에이전트의 업로드·Git 변경은 0이다. 후속 root의 [공식 비교 보고서](../p2_copula_official_submission_20260906_v1/report-source.md)와 [영수증](../p2_copula_official_submission_20260906_v1/receipt.json)에 두 정확한 SHA의 채점이 연결되었다. **동일 CPU control 대비 후보 개선은 확인했으나 기존 CUDA fallback을 넘지 못했으므로 최종 기준선은 유지한다.**

## 정확한 답안과 업로드 파일

로컬 기준 디렉터리: `artifacts/p2_crossfit_copula_materialization_20260906_v3/`.

| 용도 | 정확한 파일 | SHA256 |
|---|---|---|
| 같은 실행 CPU 기준선 | `05_answer/P2_C3_control.csv` | `ae2df14842ea3dc2d0997662ab98e8cd3b644a75742361ad624018b25a00a479` |
| 내부 평균 개선 후보 | `05_answer/P2_insample_full.csv` | `e0b4005b961180a46198ad3947cfb79c5d49bb043c1ef8d2b7190d5a958c498a` |
| 소스·모델·답안 재현 ZIP | `06_docs/P2_CPU_COPULA_REPRODUCIBILITY.zip` | `e2245ce0a04661964e0bda9848acaf9b5d38fd2115f449740b09b4045adb8609` |

각 CSV는 `station,layer,time,temp`, **26,061개 unique 공식 키, sample 순서, finite ℃**다. 두 답안의 예측은 26,061행 모두 다르며 두 예측 사이 RMS 차이는 0.089494318℃다. 이는 **정답 대비 RMSE나 예상 개선폭이 아니다**. 참값을 열지 않았으므로 이 값을 공식 성능처럼 쓰지 않는다.

OCN-02 `https://oceanaidata.org/app/problems/6`에서 root가 최신 마감/남은 횟수/권한/QA를 확인하고 정확한 파일을 선택한다. 비교용 control을 후보로 착각하지 않는다. 권장 식별 제목은 `P2 CPU C3 clean control 20260906` 및 `P2 C3 profile copula 20260906`; 한 줄 요약은 각각 “배포 관측만으로 새로 학습한 CPU C3 세 seed 평균 기준선”, “동일 C3에 배포 학습 잔차로만 적합한 strength1.0 profile copula 추가”다. 최종 모델 잠금은 답안 채점과 별개다.

## 실제 학습·검증 시간과 프로세스

| 단계 | PID | 측정 초 | 확인한 범위 |
|---|---:|---:|---|
| 빈 폴더 전체 학습 | 38640 | 167.125 | C3 full3 + covariance/CDF1, 이전 모델/답안 읽기 0 |
| 학습 산출물 독립 검증 | 35268 | 17.813 | 166,268개 학습 키 전체 모델 replay, 잔차/CDF/covariance 재계산, 17 checks |
| 공식 공개 키 inference | 32504 | 9.281 | control/후보 각26,061행, target hidden 값 0 |
| 별도 프로세스 전체 CSV replay | 12912 | 9.438 | 두 CSV byte exact |
| ZIP 새 추출 저장 모델 replay | 36144 | 9.344 | 두 CSV byte exact, 추가 fit 0 |

학습+첫 QA/inference/replay 합계 **203.657초**, ZIP replay까지 **213.001초**다. 환경 설치, 개발·8구간 연구 시간은 별도다. CPU2, GPU0, seed20260901/02/03,60epoch, blockmask/domain-balanced normalized-Huber+gradient penalty, correction strength1.0을 고정했다. 최대 학습 계획3,600초 안에서 완료했다.

**기준 모델 재생성 검사: PASS.** 저장 모델 reload만 한 것이 아니라 빈 모델 폴더에서 새로 학습하고 해당 산출물로 답안을 생성했다. 단, **새 전체 학습은 1회**이며 여러 번의 저장 모델 replay를 독립 전체 재학습 2회라고 부르지 않는다. full 학습 행의 replay/잔차 검증은 재현·무결성 검사이지 새 holdout 성능이 아니다.

실행 장소는 원 repo 밖 `C:\Users\cedis\AppData\Local\Temp\ocean_p2_copula_materialization_20260906_v3`; ZIP 추출은 다른 `...\Temp\ocean_p2_copula_zip_replay_20260906_v3`다. `python -I -B`와 package-local imports를 사용했다. 동일 설치 환경을 사용했으므로 새 venv/offline wheel 설치나 OS 수준 물리적 네트워크 차단까지 검증한 것은 아니다. Python network calls는 거부했다.

## 선택 근거와 남은 위험

근거는 [완료된 paired8fold 연구](../p2_crossfit_copula_forward_numeric_20260906_v2/report-source.md)와 해당 result SHA `cd9b6c35285b9cc096b1e1d3fa62bab429388276dbcb6ce492ddd1a287a16266`다. 전체 결과·1,117-check QA·full-key replay가 PASS한 뒤, strict B3 자연입력 평균 개선 arm만 별도 builder가 선택했다.

- 가을 주평가: C3 **0.517316→0.499286℃**, Δ−0.018030184℃. 기술적 paired7day CI90 [−0.029492503, −0.004649777]℃.
- 전체 pooled: 1.233020→1.227144℃. 추가 마지막17일 결측: 1.625741→1.623824℃이지만 CI90은 0을 포함한다.
- 자연 T5 결측은 **0.345084→0.352721℃ 악화**, B4/B7/B8 자연 구간과 일부 추가 결측 구간도 악화한다. 빈 겨울 outage는 NOT_ESTIMABLE이다. 이를 본 뒤 강도/특징/조건부 분기를 바꾸지 않았다.
- 2개 고정 inner 월 crossfit은 B3 1.893761℃로 악화해 materialization 대상에서 제외했다. 미사용 inner 날짜는 source-only 규칙대로 config에 남아 있지만 **inner fit은 0**이다.
- in-sample 보정이 outer-training 잔차를 사용하는 한계와 반복 노출된 역사적 검증의 한계를 유지한다. 공식 점수 상승 보장, 새 독립 확인, bootstrap 개선비율을 공식 확률로 해석하는 주장을 하지 않는다.

이번 full은 기존 OOF/옛 답안의 residual을 재사용하지 않는다. 배포 관측의 training labels와 이번 새 C3의 train predictions로 covariance/CDF를 적합했다. 공개 score 계수, bin17, 외부자료, pretrained weight, hidden truth 계보는 없다. CPU control과 CUDA fallback의 차이는 별도 공식 비교가 필요하며 공식 값은 정확한 SHA에만 귀속된다.

## 독립 검증·원본 불변·이식성

[independent-qa.json](independent-qa.json)의 73개 검사는 원 repo 밖 실행본→보존본 전체 파일 hash, code/config/모델 seal, 3+1 fit, seed/epoch/CPU/시간, empty model, 서로 다른 5개 PID, 학습→QA 링크, 두 답안의 독립 schema/key/order/finite/SHA, ZIP파일 CRC/복사 hash와 새 추출 replay를 확인한다. 모델을 다시 fit하거나 모델 클래스를 import해 평가한 QA가 아니라 receipt/hash/공식 키/생성된 CSV의 별도 검산이다.

공식 공개 입력은 학습 QA 후 처음 열었고 전후 SHA가 같다. sample은 키 열만 로드했다. 다음 SHA는 원본 무결성용이며 답안 값을 기준선으로 사용한 것이 아니다.

- observations: `cc5d6fd9ea398a613e485b34fd530c7dc064fa040fa675127a12318856ab178a`
- test_index: `2397ac4a240e92cd1f75194a92a3adbffb7f9c681739c6f24cac481ab8c26ff0`
- sample_submission: `3804e83d703e2642351c1191cc8cd3c1d0f485795d7f843a096cb6855360362b`
- full config: `65c54b8452956c0448810b5c655d233ff8f0b254926eb3abbceccd47bc22bd90`

6개 합성 계약 검사 PASS/Ruff PASS: 수치 코드의 원 CPU recipe parity, inner purge, native3회 모델 저장·자체 hash와 과거 모델/공식 입력 읽기 거부, 목표 temp/psal 동시 마스크, 빈 복사본, ZIP 제외/추출. 학습 이후 모델/설정/답안은 변경하지 않았다. 추가 archive 문서 포함 및 독립 출력 QA는 새 host 보조 코드이며 고정 runtime을 바꾸지 않는다.

ZIP은 **4,697,502 bytes/19 files**다. `01_data`는 데이터 참조 전용, `02_code`는 자기완결 코드, `03_model`은 이번 학습 모델, `04_logs`는 JSON 증거, `05_answer`는 두 답안, `06_docs`는 manifest와 실제 실행 설명이다. 원본 데이터·attempt locks·학습용 calibration/replay NPZ·캐시·로그 텍스트를 넣지 않았다. covariance/CDF NPZ는 모델 자체이므로 `03_model`에 포함한다. 원본 데이터 재배포 및 Git staging은 없다.

배포 데이터를 별도로 준비하고 `P2_DATA_DIR`를 지정한다. 완료 패키지에서 새 학습은 아래처럼 코드/config만 독립 복사한 후 시행한다. 기존 모델을 지우는 방식은 쓰지 않는다.

```powershell
python 02_code/create_empty_copy.py NEW_DISJOINT_DIRECTORY
# 새 디렉터리로 이동하고 P2_DATA_DIR 지정
python -I -B 02_code/run.py RUN_TRAINING
python -I -B 02_code/run.py VERIFY_TRAINING
python -I -B 02_code/run.py RUN_INFERENCE
python -I -B 02_code/run.py REPLAY
```

정확한 의존성은 패키지 requirements/config에 동봉돼 있다. 새 학습은 제외된 calibration/replay 배열도 다시 생성한다. 이미 학습된 ZIP을 새로 푼 경우 `REPLAY --replay-receipt zip-extraction-replay.json`으로 검증하며 이 검사를 전체 재학습으로 혼동하지 않는다.

## 기록 위치 / 완료·미완료

- 완료: [원본 입력 hash 봉인](official-input-hash-before.json), [학습 전 합성 검사](preflight-validation.json), [패키지 manifest](package-manifest.json), [ZIP 추출 replay](zip-extraction-replay.json), [독립 최종 QA](independent-qa.json).
- 원 학습·학습 QA·inference·replay JSON은 artifact `04_logs/`와 ZIP에 보존한다. 원본 Temp도 그대로 남았다.
- 완료: 40 재사용 +48 연구 신규 fit과 별도 full4fit. 신규 GPU 사용 0, crossfit full 학습 0.
- 미실행: zt_real/time-context 변경, 추가 hyperparameter/조건부 router 탐색, 독립 전체 재학습 두 번째 실행, 새 offline 환경 설치, 이 담당자의 포털 업로드·최종 잠금·Git commit/push.
- 공식 채점은 root의 [별도 영수증](../p2_copula_official_submission_20260906_v1/receipt.json)에 완료 기록이 있다. 기존 CUDA fallback을 유지하며, 반환 점수에 맞춘 계수 역산·재튜닝·추가 fit은 하지 않는다. 로컬 QA PASS와 공식 채점은 별도 증거다.
