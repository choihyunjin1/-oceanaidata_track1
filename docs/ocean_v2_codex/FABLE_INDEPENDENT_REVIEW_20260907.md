# Fable 독립 검토 — 2026-09-07 05:55 KST (읽기 전용)

범위: `UPLOAD_SET_2.md`와 연결 영수증 대조, P2 투영의 입력 가용성, 공식 제출 요구사항 대응표, 남은 일 분류. 새 학습·재튜닝·업로드·삭제·기존 파일 수정 없음. hidden 정답 접근 없음. 후보 변경 없음.
주의: 05:20~05:50 사이 제 검증 계산(pandas·sklearn)이 CPU를 사용했다. 같은 시간대 P3 cold_1의 single fit이 느려졌을 수 있다(§4-1). 이 검토 이후 무거운 계산은 하지 않는다.

## 1. UPLOAD_SET_2.md ↔ 실제 영수증·파일 대조

### 1-1. P2_L120_s3_proj (모순 없음, 과장 없음)

| 항목 | UPLOAD_SET_2 주장 | 독립 확인(직접 계산·파일 읽기) | 판정 |
|---|---|---|---|
| 답안 경로·행 수·bytes | `...\P2_L120_s3_proj_v2\ANSWER\submission_p2_L120_s3_proj.csv`, 26,061행, 1,190,759 bytes | 존재, 26,061행, 1,190,759 bytes | 일치 |
| SHA256 | `9c5fec38…5118` | 직접 계산 동일; ANSWER_QA.json(PID 1124)·REPLAY_QA.json(PID 33956)·saved_verify receipt(expected)·result.json 모두 동일 | 일치 |
| 키·순서·범위 | schema/key/order/finite PASS | sample_submission과 키·순서 완전 동일, base(fee6118b)와도 동일, finite, 12.43~29.15℃ | 일치 |
| 투영 대상/변경 행 | 25,890 / 10,781 | test_index에서 3층 완전 시각 행 25,890(8,630 시각×3), base 대비 변경 10,781 | 일치 |
| 후처리 성질 | 불완전 프로필 no-op, envelope 이탈 없음 | 불완전 프로필(87 시각, 171행) 변경 0; 완전 시각에서 envelope 이탈 0(원 base는 10,148행 이탈); 단조 위반 0/8,630 시각 | 일치 |
| OOF 수치 | pooled 1.251772→1.236732, B3 0.483505→0.463529, B1 +0.000121, B2 −0.008293, B8 →0.206510 | OOF_QA.json과 동일. 제 독립 재구현(sklearn isotonic)은 pooled 1.239202 / B3 0.463964로 근소 차이(완전 프로필 판정·동률 방향 처리 차이 추정, 방향·크기 동일). Codex 값을 정본으로 인정 | 일치(정본 = Codex) |
| 패키지 ZIP | SOURCE_ONLY 72,130 B `cb2fbf6c…`, SAVED_MODELS 137,582 B `e806d1f4…` | 두 SHA·bytes 직접 계산 일치. 내용: 코드·노트북(TRAIN/PREDICT/PROJECT/SAVED_PREDICT)·config·QA JSON·(SAVED만) 모델 3개. **원자료·답안 CSV·lock 미동봉** 확인 | 일치 |
| 재생성 증거 표현 | "기존 L120 cold 3-fit 계승 + 후처리 독립 재생 + 새 SAVED ZIP 추출 노트북 PASS; 새 결합본 full cold 아님" | result.json `new_combined_full_cold: false`, `new_fits: 0`; CURRENT_README도 동일 문구 | **과장 없음**(요청 조건 정확히 표기) |
| v1 포장 실패 | fingerprint 변경으로 추출 실패, v2에서 격리 | packaging-amendment.md 확인; v2는 `02_code/projection/` 하위 격리, 원 코드·모델 불변 | 일치 |

미검증(문서도 미검증으로 표기): 공식 점수, 새 OS·오프라인 환경, L120 코어의 CUDA 의존(CURRENT_README "L120 core requires CUDA").

### 1-2. P3_numeric_cpudet_s3 (진행 중, 후보 아님 — 문서 표기 정확)

- EXECUTION_LOCK: PID 25544, 시작 2026-09-06T20:14:20Z(=05:14 KST), replications 2, budget 82. config: `cpu_threads 4`, `maximum_seconds_per_cold 14400`, 36 backbone + 5 router/cold.
- cold_1/progress.json(05:49): backbone 5/36, 2,104초. 로그상 fit 소요: #1 407초, #2 790초, #3 70초, #4 764초, #5 72초 → single ≈ 400~790초, multi ≈ 70초 패턴.
- 코드 `p3_cpudet_final_day_20260907_v1.py`: CatBoost `E.Deadline` 콜백 + `tree_count_ != iterations → TimeoutError("no partial promotion")`; 일부 단계 `threadpool_limits(limits=2)`. 문서의 "CPU 4 threads"와 코드 `thread_count=4`는 일치.
- **시간 위험(§4-1)**: 18 single × (400~790초) + 18 multi × 70초 + router 5 = 약 8,500~15,500초. 상한 14,400초에 근접하거나 초과할 수 있다. 초과 시 cold_1이 TimeoutError로 종료되고 후보는 생성되지 않는다(stop_rule: 재시도 없음).

### 1-3. P1 (원형 57844ef2 보존) — 표기 정확
UPLOAD_SET_2·FINAL_RELEASE·FORM.md 모두 "같은 SHA에 연결된 기존 공식 기록 28.909341, 새 채점 아님"으로 일관. 09-07 서버 중복 응답 영수증 존재. 범위 룰 미적용 확인.

## 2. P2 투영의 입력 가용성 검토

| 규칙 | 배포 코드 `02_code/projection/p2_final_day_projection_20260907_v1.py` | OOF 검증 코드 `scripts/verify_final_day_evidence_20260907_v1.py` | 동일 여부 |
|---|---|---|---|
| endpoint 상단 | 같은 UTC 시각의 공개 layer 1 temp | wide[1] | 동일 |
| endpoint 하단 | layer 5→6→7→8 첫 유효값(`bfill(axis=1)[5]`) | `wide[[5,6,7,8]].bfill(axis=1)[5]` | 동일 |
| 적용 조건 | 시각별 목표층 {2,3,4} 3행 완전 ∧ 두 endpoint 유한 | `complete = len==3 and set=={2,3,4}`, endpoint 결측은 skip | 동일 |
| 연산 순서 | clip → 정확한 3점 PAVA(등가중, 방향 `temp_1 <= temp_5`) | clip → pava(increasing=top<=bottom) | 동일(동률 시 증가 방향도 동일) |
| 결측·불완전 | 원값 유지(exact no-op) | 원값 유지 | 동일 |
| 입력 출처 | `observations.csv` 공개층만(`layer in (1,5,6,7,8)`), 목표층 값 미사용, 시각 조인 `many_to_one` | 동일 파일·동일 공개층 | 동일 |

- **hidden에서 가려지는 입력을 쓰지 않는다.** 투영은 공개층(1,5,6,7,8)의 동시각 값만 쓰며, 이 값들은 hidden 구간에도 배포된다(hidden 시각 T1 99.2%, T6/T7/T8 ≈99% 유효, T5 70.1%). 목표층 2/3/4의 관측값은 배포·OOF 어디서도 후처리 입력이 아니다(코드 검사, 배포 CSV 검증: 불완전 프로필 무변경).
- **배포 규칙 커버리지**: 26,061행 중 25,890행 적격(3층 완전). T1 결측 6행 포함 시각은 코드상 skip(제 검증에서 `np.fmin`이 NaN을 무시해 이 6행의 envelope 판정이 부정확했으나, 코드 자체는 유한성 검사로 skip함).
- **T5 결측 regime(test 29.3%, 7,642행)**: 배포 후보에서 이 행들 중 5,882행이 변경됨(endpoint = T6). 이 regime의 OOF 근거는 (a) B8(T5 전결측·혼합) 0.2177→0.2065 개선, (b) 제가 `outage_L120`(T5 마스크 학습 예측)에 T5 숨긴 투영을 적용한 추가 집계: 전체 마스크 행 1.6172→1.5927, B3 마스크 행(2024-10, n=7,284) 0.5611→0.0803. **주의**: B3 마스크 구간의 큰 개선은 2024년 10월 수주가 30 m까지 혼합(T1≈T6)되어 envelope이 좁아진 결과다. test 10/14~10/31은 **강성층(T1−T6≈7.8℃)** 이라 envelope이 넓어 이 크기의 이득은 기대할 수 없다. 이 regime에서 투영의 효과는 "물리적으로 불가능한 값 제거 + 단조성"에 한정되며, 규모는 미측정.
- 결론: 규칙 동일·누출 없음. 성능 기대는 자연 OOF 기준(−0.015 pooled, B3 −0.020)이 상한이고, test의 성층·T5 결측 조합에서는 더 작을 수 있다.

## 3. 공식 최종 제출 요구사항 ↔ 현재 패키지 대응표

| 요구사항 | 근거 | 대응 파일 | 확인 |
|---|---|---|---|
| 모델(코드+학습 가중치)을 09-07까지 제출; 제출 즉시 잠금 | 08-12 공지 §2 | `OceanFinalRelease_20260907/{P1,P2,P3}/P?_SOURCE_ONLY.zip`, `P?_SAVED_MODELS.zip`(P1은 `SAVED_MODEL_PARTS/part001~015.zip` + `REASSEMBLY_MANIFEST.json` + `reassemble.py`), P2 신규 후보는 `OceanFinalDay_20260907/P2_L120_s3_proj_v2/*.zip` | 파일 존재·SHA 확인됨. **마감 시각 미확인** |
| 제출 모델이 업로드한 답안을 재현 | 08-12 공지 §3 | P1 `57844ef2`(빈 폴더 7fit 6,323초 exact + 09-07 서버 동일 판정), P2 `fee6118b`(3fit 173초 exact) / 신규 `9c5fec38`(기존 cold + 후처리 재생), P3 `ff42a6a0`(저장 모델 exact; 새 cold는 660행 상이) | P1·P2 확인, **P3 cold 불일치 미해결** |
| 배포 데이터 외 입력 0, 사전학습 0 | 08-31·09-01 공지 | 각 패키지 README/CURRENT_README·manifest, 원자료 미동봉·`P?_DATA_DIR` 지정 | ZIP 내용 검사로 확인(P2 신규 포함) |
| 리더보드 역산 상수 0, 상수 리터럴·산출물 제거 후 재생성 검증 | 09-02 공지 | P3 α/axis 제거, P2 anchor 제거, P1 패치 제거(일반 룰로 대체). P1 셀 정책은 "과거 로컬 OOF 고정 설정"(FINAL_RELEASE 명시) | 확인. **P1 셀 정책 상수의 출처 서술이 README에 있는지 재확인 필요** |
| 인터넷 차단 환경 재현, 6시간 이내 | 08-31·09-01 공지 | 실측 P1 6,323초, P2 173초, P3 1,497초(cold) / P3 cpudet 예상 ≤14,400초 | 같은 PC·기존 venv 기준 확인. **새 OS·새 venv·오프라인·다른 GPU 미검증**(P1 MS-TCN bf16, P2 L120 CUDA 필요) |
| 환경 정보(버전·GPU·실행 순서) | 08-12 공지(재현 불가 시 미인정) | P1 `06_docs/environment.json`, P2 `requirements.txt`, P3 `02_code/requirements.txt`, START_HERE.md, 각 README | 존재 확인 |
| 답안 CSV 형식 | README·score.py | P1 169,011행 5열, P2 26,061행 4열, P3 1,200행 4열 | validator PASS 확인 |
| 첨부 형식·크기·개수 | 09-05 UI 관측(runbook: 다중 파일, 파일당 ≤50 MB) | P1 parts 각 40,000,120 B(15개, 마지막 30 MB), P1_SAVED_MODELS.zip 590 MB는 직접 첨부 불가, 나머지 ZIP <5 MB | 크기 계산 확인. **오늘 포털 제한 미재확인** |
| 폼 값(제목·요약·저장소 URL·비고) | 09-05 UI 관측 | 각 `FORM.md`(제목·요약·SHA·저장소 URL) | 존재. 비고 문구는 P3 cold 차이·GPU 의존을 포함하도록 갱신 권장 |
| 답안 업로드 → 최종 모델 제출 순서 | 08-12 공지 §2 | FINAL_DAY_PLAN §3·§5 | 절차 문서화됨, 실행은 사용자 |

## 4. 남은 일 (중요도 순, 최대 5)

1. **[제출을 막는 결함] P3 cpudet cold_1의 시간 상한 초과 위험.** 05:49 기준 5/36 fit·2,104초. single fit이 400~790초(CPU 경합 시 후자)라 36 fit + router가 8,500~15,500초로 상한 14,400초에 걸린다. 초과 시 TimeoutError로 후보 소멸(재시도 없음), 두 번째 cold(≈같은 시간)도 15:00 전 착수 불가. 조치: (a) 지금부터 P3 학습과 경합하는 CPU 작업(제 검증 포함)을 멈춘다. (b) cold_1 결과를 기다리되, 실패 시 **fallback = 채점본 ff42a6a0 + SAVED_MODELS(exact) + SOURCE_ONLY(660행 차이 명시)** 로 최종 제출한다. 실행 중 설정 변경·재시작은 하지 않는다(사전등록 stop_rule).
2. **[제출을 막는 결함] 공식 마감 시각·첨부 제한 미확인.** 모든 시간표(15:00 후보 확정, 라운드 2)는 마감 시각에 종속. 사용자가 로그인 화면(문제 페이지·공지)에서 마감 시각·파일당 크기·파일 수 제한을 지금 확인해야 한다. 마감이 이르면 라운드 2·P3 cold_2를 포기하고 라운드 1 결과로 최종 지정.
3. **[검증 위험] GPU 의존 재현성.** P2 L120 "requires CUDA", P1 MS-TCN bf16 GPU 학습. 운영진 차단망 환경의 GPU·드라이버가 다르면 재학습 답안이 달라질 수 있다. 조치(선택, 학습 없음): P2 SAVED_MODELS의 CPU 추론이 같은 답안(fee6118b/9c5fec38)을 내는지 1회 확인해 README에 "CPU 추론 exact" 여부를 기록. P1은 README에 GPU 조건·허용오차 미검증을 명시(이미 FINAL_RELEASE에 있음).
4. **[성능 위험] P2 투영의 test regime 이득은 OOF 기대치보다 작을 수 있음.** 근거는 §2(B3 −0.020은 자연 OOF; test 29% 행은 성층+T5 결측). 라운드 1 공개 점수가 0.418892보다 좋으면 채택, 나쁘면 fee6118b 유지(FINAL_DAY_PLAN 규칙). B1 +0.000121 악화는 탈락 사유가 아님.
5. **[선택적 개선] README 보강.** P1 셀 정책 상수의 출처(과거 로컬 OOF 고정 설정, 선택 프로그램 미복구)와 P3 cold 660행 차이·허용오차 미확인, P2 후처리 근거(자연 OOF pooled −0.015, B3 −0.020, B1 +0.0001)를 각 README/비고에 명시. 09-02 공지 1항(상수 리터럴) 대비.

## 5. 이번 검토에서 새로 계산한 값(집계만, 근거 재현용)
- P2 배포 후보: 변경 10,781행 중 T5 결측 행 5,882, T5 존재 행 4,899; 평균 |변화| 0.0457℃, 최대 1.031℃; base의 envelope 이탈 10,148행 → 후보 0행.
- P2 outage OOF 투영(T5 숨김, sklearn isotonic): 마스크 행 전체 1.617227→1.592738; B3 마스크 행 0.561131→0.080264(n=7,284, 혼합 수주 특성); B8 0.217715→0.206510.
- 제 자연 OOF 재구현: pooled 1.239202, B3 0.463964(Codex 정본 1.236732 / 0.463529와 근소 차이, 원인 미추적·비중요).
