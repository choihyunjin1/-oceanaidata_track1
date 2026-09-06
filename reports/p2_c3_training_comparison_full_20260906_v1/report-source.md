# P2 L120 3-seed — 로컬 제출 후보 준비 완료

결론: **빈 모델 폴더에서 L120 세 모델을 새로 학습하고, 독립 PID 내부 재생·공식 26,061행 답안·전체 CSV 새 PID 재생까지 모두 통과했다.** 공식 업로드·채점은 아직 하지 않았다. 기존 최고점 적격 C3는 보존하며, 이 후보가 그 공식 점수를 넘었다고 주장하지 않는다.

## 제출 담당자에게

- 문제: OCN-02 / P2 profile restoration.
- 올릴 파일: [submission_p2_L120_3seed.csv](../../artifacts/p2_c3_training_comparison_full_20260906_v1/05_answer/submission_p2_L120_3seed.csv).
- SHA-256: fee6118bb4a4d1d421094aa0174d634cf73e919804ea1b62ab0bab4f1384ce4d.
- 추천 제목: P2 C3 L120 3seed 20260906.
- 한줄 설명: 배포 관측만으로 학습한 C3 120epoch 3-seed 평균, 가을 내부 평균 개선·추가 결측 악화 위험을 분리 기록한 후보.
- replay_p2_L120_3seed.csv는 exact 재현 확인용이며 별도 후보로 또 제출하지 않는다. 포털의 기회·중복·기한 확인과 업로드는 root 담당이다.

## 학습과 내부 근거

[28-fit historical 결과](../p2_c3_training_comparison_20260906_v1/report-source.md) 및821-check QA/root 독립 산술 QA 뒤 승인된 동일 L120만 학습했다. B3 3-seed 자연 RMSE 0.488284326→0.483505057℃(Δ−0.004779270), CI90 [−0.020369156,+0.010871728], 경험적 개선비율0.667이다. 가을 outage 전체0.538760112→0.563601402℃, outage 대상7322행0.445059242→0.561965765℃, 자연T5 결측180행0.535881156→0.765854952℃는 악화했다.

추가 seed는 같은 노출 B3에서 seed 민감도를 확인한 것이지 새 독립 holdout이 아니다. 현재 후보는 작은 주평가 평균 개선을 보존한 정보가치 후보다. 공식 예상 점수로 역산하지 않으며, 공식 점수는 exact CSV가 채점된 후에만 연결한다.

새 full recipe는 120epoch/AdamW wd0.0001/lr0.001/batch4096/gradient0.01, seeds20260901/02/03, 기존11-context/8-token-feature C3 및 고정 domain weights/blockmask다. 제공 observations.csv의 같은166268개 지원 가능한 학습 target만 사용했다. 목표층2/3/4 temp+psal은 특징 생성 전에 함께 숨기며 학습 truth만 별도 배열로 쓴다. validation 모델·OOF·과거답안·외부자료·Public역산 계수는 학습 입력0이다.

## 실제 완료 단계

| 단계 | PID | 실측 | 결과 |
|---|---:|---:|---|
| 빈03_model→3 fresh full fits | 33204 | 121.672초 | SCRATCH_TRAINING_PASS |
| 새PID 전체학습166268행×3모델 normalized예측 exact replay | 33420 | 14.688초 | 19/19 QA PASS |
| 새PID 공식키 inference | 27652 | 11.281초 | 26061행 schema/order/finite PASS |
| 또 다른PID 전체CSV replay | 41928 | 11.203초 | exact SHA PASS |
| 최종 source/model/QA/CSV 확인 | 37700 | 별도 포함 | 16/16 QA PASS |
| full-cycle 전체 | parent8464 | 169.368초 | COMPLETE_LOCAL_CANDIDATE_READY |

2026-09-06 13:39:52~13:42:40 KST, GPU0 단독/CPU2/DataLoader0. 단계 전체30분cap 이내다. historical28 + full3 = **신규 연구·배포 학습31fits**이며 원래 full 모델을 다시 로드해 답안만 만든 것이 아니다. 별도 안전 호환성 검사의 tiny CPU synthetic는 historical suite2fits/full suite4fits(README UTF-8 테스트 정정 후 재실행 포함)로 명확히 분리한다. 실제 historical/full 학습 실패·재시작0이다.

각 full 모델은4865 parameters,120epoch이고 seed별35.828/34.938/35.094초였다. Full-training prediction replay는 모델 저장/로드 일치 검증이지 새로운 성능 검증 점수가 아니다. 일반화 판단은 연결된 historical 동일키 비교에서만 나온다.

## 산출물·계보·규정 경계

루트: [artifacts/p2_c3_training_comparison_full_20260906_v1](../../artifacts/p2_c3_training_comparison_full_20260906_v1/README.md).

- 01_data: 배포 source 환경변수 안내만, 원본 ZIP/CSV 복사0.
- 02_code: 로컬 core.py/base.py/run.py. 원 연구 repo import 없이 자기 폴더에서 로드한다.
- 03_model: 새 model_seed20260901.pt, model_seed20260902.pt, model_seed20260903.pt 및 manifest.
- 04_logs: 학습/재생/추론/최종QA receipt와 consumed lock.
- 05_answer: 후보와 exact replay CSV.
- 06_docs: immutable historical result/QA/replay/root QA 사본.

수치 core SHA b18a8279a542f38f5ed550307772824ed8f597e088ad2ba7070ebfeff9020c84는 기존 적격 C3 그대로다. 기존 portable base는 새 파일에서 후보 CSV 파일명과 명시적 LF 직렬화만 변경했다. 평균은 각 seed baseline+scale×normalized 후 mean3이며 projection/calibration/라우팅을 추가하지 않았다.

학습 단계 공식 index/sample 접근0. 내부 QA 통과 뒤 inference와 replay는 각각 test_index/sample의 **키3열만26061행** 파싱했고 sample temp 값 파싱0이다. 최종QA에서 두 파일의 **전체 바이트 SHA**를 계산한 것과 모델 입력에 sample 값을 사용하는 것을 구분한다. 원본 observations SHA 불변, old-model/old-answer/external-data/hidden 접근0, upload0, Git0이다.

최종 CSV는 station,layer,time,temp/UTF-8/LF/12 significant digits이다. serialization 최대오차4.999734e−11℃, 독립 전체CSV 재생 byte SHA exact다. sample순서와 일치하고 공식index키집합이 정확히 같다. 구형 답안46d194...c071과 다른 SHA이며, 구형값을 읽어 행별 변화량을 계산하지 않았다.

| 근거 | SHA-256 |
|---|---|
| PACKAGE_MANIFEST | ef251e369253b99bc270929b48582bca6a99ea0024eb2a06648607359cc60773 |
| training-result/MODEL_MANIFEST | 50f37074cd438d500a77513c8131f18333cbf5ab4f1e9fa0b5f9e5293c26a8d6 |
| training-independent-qa | 95552a4b680ce38395576f5a63406062d5e6451556c2a71e70ac6aafc14b22b8 |
| inference-result | e0366211b8b89af40a07e2c1148548c50b163d81ac1b53cef6a6445a61219725 |
| replay-result | 26698a5988d8ffc2870c06ac2638aa02720b0f2236d2d5ba418eeec4c6e9472b |
| independent-qa | 39b83a72bd67fda08f8170003ee54311f8462a61bce746415d67ab586a1f4ee7 |
| terminal_result | ab20d8259866e728d4ead5f11a115e5ebfd37a2f1485ee9b6db0746a9c53df06 |
| official index byte SHA | 2397ac4a240e92cd1f75194a92a3adbffb7f9c681739c6f24cac481ab8c26ff0 |
| official sample byte SHA | 3804e83d703e2642351c1191cc8cd3c1d0f485795d7f843a096cb6855360362b |

Full synthetic5/Ruff PASS. 첫 미봉인 synthetic 실행에서 README를 Windows 기본cp949로 읽는 테스트만 실패했고 UTF-8 명시 후 통과했으며, 실제 full 학습에는 영향을 주지 않았다. 원 봉인 historical코드/새full package/이전 모델/답안/lock은 그대로 보존한다.

## 완료와 남은 범위

로컬 후보 준비·학습·내부검증·공식키 추론·전체CSV 재생 완료. 포털 제출/채점, 최종모델 잠금, Git commit/push, ZIP 배포본, 새 가상환경 또는 원repo 밖 cold run은 수행하지 않았다. 현재 폴더를 원repo 밖에서 검증했다고 표현하지 않는다. Python 파일 allowlist/네트워크 차단은 OS sandbox가 아니다.

추가 재학습·공식 점수 기반 조정 없이 exact 후보를 root에게 인계한다. 기존 적격 C3는 fallback으로 남긴다.
