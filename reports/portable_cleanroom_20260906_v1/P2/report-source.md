# P2 portable cleanroom — 실제 학습·전체 재현 PASS

결론: **원래 연구 저장소 밖의 새 빈 모델 폴더에서 C3 3개를 scratch 학습하고, 별도 PID 추론 및 세 번째 PID 전체 재현으로 기존 적격 답안 SHA를 정확히 재현했다.** 준비된 모델·원래 답안을 복사해서 추론한 검사가 아니다. 원본과 기존 코드·모델·lock은 모두 보존했고 업로드/Git 작업은 하지 않았다.

## 완성본과 파일 선택

- 독립 패키지: `artifacts/portable_cleanroom_20260906_v1/P2/`.
- **OCN-02/P2 답안**: `05_answer/submission_p2_clean_C3.csv`.
- 답안 SHA: `46d194a1ef40a1deaebd084916644d9359433d2e6ce7d5c0b53d9f515bbec071`.
- 모델/코드 재현 ZIP: `06_docs/P2_PORTABLE_REPRODUCIBILITY.zip`, **594,670bytes / 27members**.
- ZIP SHA: `4ff1fe0909974592410a526e149a6f827c2857aca010c92bccb9e90d7fb9503a`.
- `replay_p2_clean_C3.csv`는 동일 답안 검산본이지 새 후보가 아니다. ZIP/.pt는 리더보드 답안 CSV 입력란에 올리지 않는다.

현재 파일은 이미 공식 제출했던 적격 C3와 동일 SHA이므로 **새 점수 개선 결과가 아니다**. 기존 SHA에 연결된 역사적 공식 RMSE는 0.455143℃ / 27.622418점이지만 이번 작업에서는 공식 채점을 추가하지 않았다. 중복 업로드로 기회를 소모하지 않는다.

## 실측 근거

| 단계 | 결과 | 시간 / PID |
|---|---|---|
| 원본→빈03_model scratch | 3 seeds × 60 epochs, 3 full fits | 68.391초 /26960 |
| 저장 모델→전체 답안 | 26,061 unique keys, schema/order/finite PASS | 11.812초 /33028 |
| 새 프로세스 전체 replay | 26,061행 값·CSV SHA exact | 12.047초 /31864 |
| 독립 QA | 71/71 PASS | [independent-qa.json](independent-qa.json) |
| 합성/코드 검사 | 최초 6 tests PASS, builder 보강 후 관련 1test 재검 PASS, Ruff PASS | [검사 기록](validation-receipt.json) |
| ZIP CRC 및 멤버 SHA | 27개 전부 검증 PASS, source CSV 0 | [archive-receipt.json](archive-receipt.json) |

학습+추론+재현 순수 단계 시간은 총 **92.250초**다. Python startup, 준비·문서화·QA 시간은 별도다. 현재 장비에서 6시간보다 충분히 짧았다는 실측이며 모든 장비의 실행 시간을 보장하는 표현은 아니다.

실제 실행 위치는 OS 임시 폴더 `ocean_p2_portable_20260906_v1`로, `C:\Users\cedis\PycharmProjects\PythonProject` 바깥이다. working directory도 OS Temp였으며 `python -I -B <임시패키지>/02_code/run.py`로 실행했다. core/entry module origin은 패키지 내부 `02_code/core.py`, `02_code/run.py`로 기록됐다. 완성된 최초25개 파일을 artifacts로 복사하고 각 SHA를 비교한 결과 불일치0; 그 뒤 빈 재생성 빌더와 안내문 2개를 추가하여27개를 ZIP에 넣었다. 원래 temp 완료본도 삭제하지 않았다.

## 휴대성과 규정

독립 구조는 `01_data` 안내 / `02_code` / `03_model` / `04_logs` / `05_answer` / `06_docs`다. 실행 코드에는 개인 절대 경로와 원 repo import가 없다. `P2_DATA_DIR`만 실제 배포 파일 위치를 제공한다. 추가 공개 자료·사전학습 가중치·옛 답안/anchor·Public 역산 계수·과거 OAS/alpha 경로를 포함하지 않는다. 순수 함수만 추출한 출처는 패키지 `06_docs/SOURCE_PROVENANCE.json`에 원 파일/함수/라인/SHA로 남겼다.

학습은 운영진 observations.csv만 사용하고 source hash를 전후 검증한다. target 2/3/4층 temp와 psal을 특징 생성 전에 가리고 labels를 별도로 보관한다. 166,268개 유효 원본행, 공개 T5/S5 block augmentation51,354행, 원본별 총가중질량166,268이 기존C3와 동일하다. 모델구조·seed·epoch·손실·가중치·보정은 변경하지 않았다. 패키지 기본CPU2이며 이번에는 동시 작업 예산과 기존 v6 fit의 내부 설정을 유지하기 위해 명시적 `--cpu-threads 1`로 실행했다.

추론과 replay는 각각 index/sample의 station/layer/time 키만26,061행씩 읽었다. sample값/hidden/oldanswer/oldmodel 접근0. 독립QA도 공식키만 각각26,061행을 읽어 생성된 **새 답안**의 키·순서·finite·전체재현을 검산했다. 이는 공식 정답 열람이 아니다. 원본 배포 데이터는 ZIP에 포함하지 않았다.

`build_package.py --output <새폴더>`는 완료 패키지에서도 code/config/docs만 복사하여 빈03_model/04_logs/05_answer를 만든다. 기존 모델·lock을 삭제할 필요가 없다. 빌더 자체는 패키지root에 동봉하고, 실행 명령은 `06_docs/RETRAIN_FROM_EMPTY.md`에 설명했다. Git용 원본은 `scripts/portable_20260906/P2/`; 모델·CSV·ZIP은 Git에 올리지 않는다.

## 완료하지 않은 것과 한계

- 새 venv/다른 PC/운영진 하드웨어에서의 재현은 미검증. 현재 Python3.12.10, NumPy2.3.5, pandas3.0.1, PyTorch2.13.0+cu130, threadpoolctl3.6.0을 사용했다. 별도 wheel bundle은 만들지 않았다.
- Python socket audit hook으로 네트워크 연결을 거부했지만 OS 방화벽/물리적 네트워크 단절 검증은 아니다. 필요한 패키지는 이미 설치된 환경을 사용했다.
- CUDA가 필요하며 버전·하드웨어가 다르면 byte identity가 달라질 수 있다. CPU fallback으로 다른 모델을 조용히 만들지 않는다.
- 모델 최종 지정/잠금, 포털 첨부/업로드, GitHub push는 이번 담당 범위 밖이므로 실행0이다. 최종 모델 제출 UI의 현재 첨부/크기 제한은 별도 확인해야 한다.
- 직전 cross-fit copula 기술 종료는 미완료 상태 그대로다. [수치 검사 정정 설계](copula-numeric-amendment-design.md)만 작성했으며, 새 runner/학습을 실행하지 않았다. 제안된 재사용36신경망+4copula, 남은48신규fits는 후속승인 범위의 설계이지 완료 실적이 아니다.

검증 스킬의 기준에 따라 **scratch lineage, 수치·스키마 QA, 새 PID replay, 역사적 공식 점수, 최종 포털 제출**을 서로 다른 주장으로 구분했다. 본 결과는 로컬 portable 패키지 재현 PASS이며 점수 개선 또는 최종 제출 완료를 의미하지 않는다.
