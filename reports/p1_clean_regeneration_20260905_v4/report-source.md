# P1 clean 기준선 원본부터 재생성 — inner 정책 불일치로 중단

결론: **전체 재생성 미완료 / INNER_POLICY_REGENERATION_MISMATCH**.
새 초기화한 Q4 inner 2 fits 뒤 선택 정책이 달라져 fail-closed로 중단했다.
full 2 fits·공식 추론·CSV 생성은 모두 0이다. 저장 모델 재로딩 PASS와 구분한다.
기존 모델·답안·동결 코드·config·lock을 수정하거나 삭제하지 않는다.

## 실제 결과

canonical [train-result.json](train-result.json): 2026-09-05 23:12:32 KST 시작,
PID 19884, 총 126.047초. 467,282개 학습행과 123,372개 inner 검증행,
80개 특징을 사용해 O/B 두 모델을 새로 학습했다.

| 항목 | 기존 Q4 earlier-inner | 이번 CPU2 새 학습 |
|---|---:|---:|
| O threshold | 0.20 | 0.15 |
| O inner F1 | 0.891901795529498 | 0.8913595166163142 |
| B threshold | 0.20 | 0.20 |
| B inner F1 | 0.8931250768663141 | 0.8931250768663141 |
| 재보정 union threshold | 0.30 | 0.30 |
| union inner F1 | 0.8946535135793448 | 0.8926976463488232 |
| selected policy | balanced_union | balanced |

같은 Python 모델 스택 버전과 동일 소스/config SHA를 확인했다. 명시적인 자원 변경은
CPU4→CPU2이며, XGB 학습 결과가 자원 설정에 민감했을 가능성은 있으나 현재 원인으로 확정하지 않는다.
학습 순서 등 구현 차이도 원인 구분 전에는 배제할 수 없다. 임계값을 강제하거나 실패 lock을
삭제하지 않았으며 자동 재학습하지 않았다. 이 결과는 새 가설의 성능 탈락이 아니라
**기준선 전체 재생성 검증에서 발견한 불일치**다.

다음 조치가 승인된다면 별도 ID에서 원래 CPU4 자원을 그대로 쓴 exact inner 재현부터 확인한다.
현재 신규 모델 2개는 실패 영수증과 함께 보존하고, 공식 답안은 만들지 않았다.

새 `artifacts/p1_clean_regeneration_20260905_v4/03_model`이 빈 상태임을 검사한다.
배포 train.csv로 Q4 earlier-inner O/XGB와 B/LGBM 2개를 새 초기화하여 학습하고,
같은 4개 정책 및 사전등록 threshold grid에서 기존 balanced_union/O0.2/B0.3을
재도출하는지 확인한다. 불일치 시 임계값을 강제하지 않고 중단한다.
동일하면 배포 train 전체로 O/B 2개를 새 학습한다. 기존 checkpoint·답안 입력은 0이다.
CPU 경쟁 완화를 위해 원래 4 threads를 **2 threads**로만 바꾸며 GPU 0, 총 4 fits, 1시간 상한이다.

별도 프로세스에서 신규 full 모델만 로드하고 test 관측 7열/sample 키 4열로 169,011행 답안을
재생성한다. sample 예측값·hidden·외부자료·Public 역산은 사용하지 않는다.
또 다른 프로세스에서 답안 byte-exact replay를 검사한다. 기존 CSV는 읽지 않고 이전 공식
receipt의 SHA 문자열 `064ef022faf2a3e8bc7c70633210847aa494060858374aa43f28f4eced84ec43`와 비교한다.

## 실행 입구

저장소 루트에서 `P1_DATA_DIR`을 원본 배포 P1 폴더로 지정한다.

```powershell
.venv-p1\Scripts\python.exe scripts\verify_p1_clean_regeneration_20260905_v4.py --self-test
.venv-p1\Scripts\python.exe scripts\verify_p1_clean_regeneration_20260905_v4.py --train
.venv-p1\Scripts\python.exe scripts\verify_p1_clean_regeneration_20260905_v4.py --infer
.venv-p1\Scripts\python.exe scripts\verify_p1_clean_regeneration_20260905_v4.py --verify
```

`RUN_TRAINING.cmd`와 `RUN_INFERENCE.cmd`는 위 명령의 별도 입구다.
아티팩트는 `02_code`, `03_model`, `05_answer`로 분리한다. `02_code`는 코드/작은 계보 메타데이터
스냅샷이고 원본 데이터는 복사하지 않는다. 이번 실행은 현재 환경에서 새 학습하는 검사이며,
복사한 코드 디렉터리 독립 실행·네트워크 차단 새 머신·최종 ZIP 검증과는 구분한다.
업로드·최종 모델 잠금·Git 변경은 하지 않는다.
