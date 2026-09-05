# P1 clean 기준선 CPU4 재생성 — 새 학습·답안 생성 PASS, 기존 답안 exact 복원 FAIL

결론: **빈 03_model에서 실제 학습하여 새 답안을 만드는 경로와 별도 PID 답안 replay는 PASS**다.
그러나 **기존 공식 답안 exact 복원은 FAIL**이다. 새 파일에 과거 최고점·공식 점수를 붙이지 않는다.
이전 [v4 실패](../p1_clean_regeneration_20260905_v4/report-source.md)는 보존한다.
이 시도는 성능 튜닝이 아니라 사용자 요청에 따른 빈 모델 폴더부터의 재생성 검증이다.

## 실제 실행·독립 QA

[train-result.json](train-result.json), [inference-qa.json](inference-qa.json),
[replay-qa.json](replay-qa.json), [independent-qa.json](independent-qa.json)이 canonical 근거다.

| 단계 | 실제 결과 |
|---|---|
| 빈 03_model → inner B/O 새 학습 | 2 fits, 467,282 training / 123,372 inner rows |
| Q4 정책 재도출 | balanced_union/O0.2/B0.3, 기존 inner F1 모두 정확 일치 |
| 배포 train 전체 → full O/B 새 학습 | 2 fits, 776,706행, 80열; 총 training 178.422초 |
| 별도 PID 새 답안 생성 | training PID38520 → inference PID2300; 12.656초 |
| 또 다른 PID 새 답안 replay | PID29280, byte-exact PASS; 12.265초 |
| 총 학습+추론+재생 | 203.343초, CPU4/GPU0 |
| 기존 checkpoint·답안 내용 입력 / hidden / upload | 각각 0 |

새 답안: `artifacts/p1_clean_regeneration_20260905_v5/05_answer/P1_submission.csv`,
169,011행, 6,929,481bytes, SHA `5971e145f1ac38b8ee3e34cfd302973ba7a64b8873db11c354d3331221fdb28a`.
새 이상행 6,395개 대 기존 receipt 6,505개이므로 **최소 110개 label이 다르다**.
같은 schema/order/bytes이고 양성수가 달라 단순 직렬화 차이가 아니다. 기존 답안 값을 열지 않았으므로
정확한 changed-row 수는 미측정이다. QA 21개 체크는 전부 통과하되 canonical 동일성은 별도로 FAIL이다.

full B 모델은 SHA `95f2f6a4d9f643ecd26a261c76480acf2a6649d6236f613ef157cc4f2b247b0c`로
기존과 byte-exact 같다. full O는 새 `567ea9c80e7e51c8cefbcd889ef46516268188ff22000b44e61c61209db1df3b`
대 기존 metadata `2c2eb3e3539140cc0dc1aeb6ef8514ab4b6b097e8b6dd2a047d7ac83bc088e5f`로 다르다.
O 학습 단계의 재현 차이가 남았지만 원인은 확정하지 않았다.

읽기 전용 대조상 canonical full runner는 새 프로세스에서 O→B 순서, encoder 1회 적합/공유이다.
v5는 같은 프로세스에서 inner B→O 후 full O→B를 실행하며 모델마다 encoder를 새 적합한다.
CPU4·seed·패키지 버전·feature code·배포 train hash는 동일하다. B 전체 package가 정확 일치하지만
프로세스 상태·스레드 환경 등 모든 원인을 단독 분리한 실험은 아니다. 추가 fit/튜닝은 하지 않았다.

현재 환경에서 소스+train부터 새 모델과 답안을 생성하는 것은 입증했다. **독립 머신·차단망·복제 코드
실행·최종 패키징은 미검증**이다. 원래 공식 파일 exact 재현이나 과거 최고점 복원을 완료했다고 말하지 않는다.

## 사전 고정 계약과 실행 입구

- 새 `03_model`이 비어 있어야 시작한다. 기존 모델·답안 내용은 입력으로 사용하지 않는다.
- 원래 CPU4/seed20260813/동일 패키지/동일 raw 및 열순서/동일 Q4 earlier-inner·21일 purge를 사용한다.
- inner의 O/B 상대 학습 순서를 canonical처럼 B→O로 맞추고 encoder도 모델별로 새 적합한다. 원래 screen의 비비교 flank fit은 총2fit 계약에 따라 재실행하지 않는다. 이 차이가 완전히 무영향이라고 사전 단정하지 않는다.
- Q4 inner 2fit에서 기존 balanced_union/O0.2/B0.3이 다시 선택되어야 full O→B 2fit으로 이어간다. 불일치면 즉시 종료하고 임계값을 강제하지 않는다.
- 총 최대4fit, CPU4/GPU0, 1시간 상한, 1회 잠금. 조건 통과 후 별도 PID 추론 및 재생 검증만 허용한다. 추가 튜닝·학습 반복·업로드·Git0.
- 기존 공식 답안은 열지 않는다. 새 답안의 SHA를 이전 receipt의 `064ef022faf2a3e8bc7c70633210847aa494060858374aa43f28f4eced84ec43` 문자열과 사후 대조한다.

```powershell
.venv-p1\Scripts\python.exe scripts\verify_p1_clean_regeneration_20260905_v5.py --self-test
.venv-p1\Scripts\python.exe scripts\verify_p1_clean_regeneration_20260905_v5.py --train
.venv-p1\Scripts\python.exe scripts\verify_p1_clean_regeneration_20260905_v5.py --infer
.venv-p1\Scripts\python.exe scripts\verify_p1_clean_regeneration_20260905_v5.py --verify
```

`P1_DATA_DIR`은 원본 P1 배포 폴더. `RUN_TRAINING.cmd`, `RUN_INFERENCE.cmd`는 새 artifact 아래 있다.
새 학습 재생성, 저장모델 별도 프로세스 재생, 독립 머신/차단망/복제 코드 실행/최종 패키징은 별개의 판정이다.
이번 검증이 완료되어도 뒤의 독립환경·패키징 항목은 미완료라고 명시한다.
