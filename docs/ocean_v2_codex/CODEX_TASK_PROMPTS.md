# CODEX_TASK_PROMPTS — 순서대로 붙여넣는 태스크 프롬프트

각 프롬프트는 자체 완결이다. 새 Codex 세션마다 첫 줄의 "읽을 문서"를 반드시 포함시킨다. 완료 보고는 `00_MASTER_BRIEF.md §6` 형식.

---
## T0 — 공통 모듈
```
저장소 C:\Users\cedis\PycharmProjects\PythonProject 에서 작업한다. 먼저 docs/ocean_v2_codex/00_MASTER_BRIEF.md 와 docs/ocean_v2_codex/COMMON_SPEC.md 를 전부 읽어라.
목표: src/ocean_v2/__init__.py, src/ocean_v2/common/{paths,hashing,determinism,runtime,submission,audit_constants,stats,report}.py 와 tests/ocean_v2/test_common.py 를 COMMON_SPEC대로 구현하고 pyproject.toml에 ocean_v2 패키지를 추가한다(기존 항목 유지).
제약: 브리프 §1 절대 규칙(배포 데이터만, LB 상수 금지, CPU 결정론, 기존 모듈 수정 금지, 커밋 금지). Python은 .venv-p1\Scripts\python.exe.
완료 조건: `.venv-p1\Scripts\python.exe -m pytest tests\ocean_v2\test_common.py -q` 통과; `python -m ocean_v2.common.audit_constants --help` 동작; validate_p1/p2/p3가 각 sample_submission.csv 자체를 통과시키고 열 순서를 바꾼 사본을 거부하는지 확인(데이터 경로는 환경변수 P1_DATA_DIR/P2_DATA_DIR/P3_DATA_DIR).
보고: 파일 목록, 테스트 결과, 남은 이슈.
```

## T1 — P3 골격·특징·CV·안전 기준선 B1
```
저장소 C:\Users\cedis\PycharmProjects\PythonProject. 읽을 문서: docs/ocean_v2_codex/00_MASTER_BRIEF.md, COMMON_SPEC.md(이미 구현됨), P3_SPEC.md 전체, 참고로 reports/claude_recon_20260905/P3_recon.md 와 design_panel/P3_design_*.md.
목표: src/ocean_v2/p3/ 를 P3_SPEC §2~§6대로 구현하고 configs/ocean_v2/p3_base.json 을 만든 뒤, P3_DATA_DIR=C:\Users\cedis\Downloads\p3\데이터셋_P3\P3_wave_forecast 로 `python -m ocean_v2.p3 all --config configs\ocean_v2\p3_base.json --out artifacts\ocean_v2\p3\base > artifacts\ocean_v2\p3\base\run.log 2>&1` 을 실행해 안전 기준선 B1(P3_v2_base)의 CV 리포트·모델·fitted_params·후보 CSV(submissions/claude_v2/p3/P3_v2_base/)를 만든다.
필수: (1) src/p3_wave 원본은 수정 금지, 필요한 함수는 복사; (2) 특징 등가성 테스트(train 슬라이스 vs test 형식 bit-동일; summarize_context 대조)와 purge/episode 무겹침 assert; (3) CV 블록 6분기, 평가면 S_dense/S_onset/S_storm/S_greedy, episode 부트스트랩; (4) 바람-드롭아웃 증강 w_aug ∈ {0.3,0.5,1.0} CV 선택 → fitted_params; (5) 학습은 CPU 결정론(LightGBM deterministic, seed/thread 고정), persistence shrink·라우터·α 없음; (6) predict를 2회 실행해 CSV SHA 동일 확인; (7) validator 통과; (8) 총 소요시간 기록.
보고: cv_report.md 핵심 표(블록별·리드별·정점별, persistence 대비, 기존 계보 재평가 결과), fitted_params 요약, 후보 CSV 경로·SHA, 소요시간, 결정론 확인.
```

## T2 — P2 골격·특징·CV·안전 기준선 R0
```
저장소 C:\Users\cedis\PycharmProjects\PythonProject. 읽을 문서: docs/ocean_v2_codex/00_MASTER_BRIEF.md, COMMON_SPEC.md, P2_SPEC.md 전체, 참고 reports/claude_recon_20260905/P2_recon.md 와 design_panel/P2_design_*.md.
목표: src/ocean_v2/p2/ 를 P2_SPEC §2~§9대로 구현하고 configs/ocean_v2/p2_safe.json 을 만든 뒤, P2_DATA_DIR=C:\Users\cedis\Downloads\p2\데이터셋_P2\P2_profile_restore 로 `python -m ocean_v2.p2 all --config configs\ocean_v2\p2_safe.json --out artifacts\ocean_v2\p2\safe > ...\run.log 2>&1` 을 실행해 R0(P2_v2_safe) CV 리포트·모델·derived_constants·후보 CSV(submissions/claude_v2/p2/P2_v2_safe/)를 만든다.
필수: (1) hidden 26,352행 NaN assert와 목표층 값 셔플 불변 테스트; (2) 조직 baseline 26,061행 정확 재현(최대오차 0) 테스트; (3) 수심 슬롯·실제수심 목표·연속블록 T5/psal 마스킹 증강(고정 RNG)·T5-부재 전문가·envelope+PAVA; (4) CV 8블록 natural/testmatched + 아웃티지 아날로그 M1/M2 + Composite + 비교군(nominal 보간·실제수심 보간·T1 복사); (5) LightGBM 결정론(threads 4, 5 seed), CSV %.5f, predict 2회 SHA 동일; (6) validator 통과, 소요시간 기록.
보고: 블록별/층별/T5 유무별 RMSE 표와 비교군, Composite, derived_constants 요약, 후보 CSV 경로·SHA, 소요시간.
```

## T3 — P1 골격·특징 캐시·CV·안전 기준선 C0
```
저장소 C:\Users\cedis\PycharmProjects\PythonProject. 읽을 문서: docs/ocean_v2_codex/00_MASTER_BRIEF.md, COMMON_SPEC.md, P1_SPEC.md 전체, 참고 reports/claude_recon_20260905/P1_recon.md 와 design_panel/P1_design_*.md.
목표: src/ocean_v2/p1/ 의 data/structure/features(A 기저 76열 + G 밀도만 우선)/cv/models/decode(1~3단계)/calibrate/train/predict/report 를 구현하고 configs/ocean_v2/p1_safe.json 을 만든 뒤, P1_DATA_DIR=C:\Users\cedis\Downloads\데이터셋_P1\P1_qc_anomaly 로 `python -m ocean_v2.p1 all --config configs\ocean_v2\p1_safe.json --out artifacts\ocean_v2\p1\safe > ...\run.log 2>&1` 을 실행해 C0(P1_v2_safe) CV 리포트·모델·fitted_params·후보 CSV(submissions/claude_v2/p1/P1_v2_safe/)를 만든다.
필수: (1) src/p1_qc 원본 수정 금지(build_features 등은 import 또는 복사; depth 계열 4열 제외); (2) 반기 4블록 CV, 양측 purge 21일, 양성 run 시작 블록 귀속, 인코더·분위 상수 fold train 전용; (3) derive_structural_constants(flat_min_run, rate 사전정보 등) → fitted_params; (4) 하드 flatline 룰(자연 FP≈0 검증), edge0 룰, 벡터화 히스테리시스(기존 루프 구현과 동일성 테스트), OOF 후처리 격자 + 중첩-정직 추정; (5) 지표 f1_season/f1_pooled/f1_worst/유형별 recall/계열별 예측률/스트레스 표면; (6) CPU 결정론(LightGBM 3 seed + XGBoost 2 seed), predict 2회 SHA 동일; (7) validator 통과(169,011행), 소요시간 기록.
보고: 블록별 F1 표(H1 두 블록 풀링 1차), 유형별 recall, 계열별 예측률, 후처리 선택값과 중첩 추정, 후보 CSV 경로·SHA, 소요시간.
```

## T4 — P3 사다리 (B2 CatBoost 멤버 → B3 물리 특징 → B4 seed5/HPO → 조건부 calib)
```
저장소 ... 읽을 문서: 00_MASTER_BRIEF.md, P3_SPEC.md §5~§7. 전제: T1 완료(artifacts/ocean_v2/p3/base 존재).
목표: configs/ocean_v2/p3_ens.json, p3_phys.json, p3_seed5_hpo.json 을 만들어 순서대로 실행하고, 각 단계에서 사전 등록 게이트(ΔS_dense<0 ∧ P(개선)≥0.90 ∧ 6블록 중 ≥4 개선 ∧ 정점 악화 ≤+0.010 ∧ 18/24h 악화 ≤+0.005 ∧ ΔS_greedy ≤+0.005)를 코드로 판정한다. 통과한 단계만 누적. 총 후보 평가 ≤12회. calibrate.py의 a_L을 fitted_params에 기록하되 적용 후보(P3_v2_calib)는 CI90이 1을 배제하고 0.7≤a_L≤1.3일 때만 생성.
보고: 사다리 표(후보, S_dense, S_greedy, CI90, 게이트 결과), 최종 지정 후보와 CSV·SHA, 소요시간. 리더보드 점수로 어떤 값도 바꾸지 말 것.
```

## T5 — P2 사다리 (R1 아웃티지 → R3 스택 → R4 HPO → L3 MLD → L4 아날로그 → R6 다양성 → R7 평활)
```
저장소 ... 읽을 문서: 00_MASTER_BRIEF.md, P2_SPEC.md §8~§10. 전제: T2 완료.
목표: 사다리 단계별 설정 JSON을 만들고 순서대로 실행, 게이트(Composite 개선 ∧ B3·B8·M1 악화 ≤+0.01 ∧ 어떤 블록도 ≤+0.02 악화; |Δ|<0.003이면 단순한 쪽) 판정, 통과분 누적. DeepSets 멤버는 CPU deterministic(set_num_threads 고정) 5 seed, 시간 초과 시 2 seed.
보고: 사다리 표(Composite, B3/B8/M1, 층별), derived_constants(스택 가중·s_L), 최종 후보 CSV·SHA, 소요시간.
```

## T6 — P1 사다리 (C1 시간창 → C2 psal → C3 브래킷/노이즈 → C4 구간 완성 → C5 Stage-2 → C5+aug → C6)
```
저장소 ... 읽을 문서: 00_MASTER_BRIEF.md, P1_SPEC.md §3~§8. 전제: T3 완료.
목표: features.py에 B/C/D/E/H 블록, decode.py에 유형별 구간 완성(offset 점프쌍·drift 역투영·noise 확장), context.py+Stage-2, gap 증강 학습을 구현하고 단계별 설정으로 순서대로 실행. 게이트(Δf1_season>0 ∧ Δworst≥−0.005 ∧ CI90 하한>−0.002 ∧ Δstress≥0 ∧ 정점 회귀 ≤0.02) 판정, 통과분 누적(비누적 실패 단계는 제외). 합성 주입 테스트(offset/drift를 합성 계열에 주입해 브래킷 특징·완성 디코더가 정확 경계 복원)를 tests/ocean_v2/test_p1.py에 추가. 모든 임계값은 fitted_params에서만.
보고: 사다리 표(f1_season, f1_pooled, worst, stress, 유형별 recall, 계열별 예측률), 최종 후보 CSV·SHA, 중첩-정직 추정, 소요시간.
```

## T7 — 최종 패키지 v2 + 클린룸
```
저장소 ... 읽을 문서: 00_MASTER_BRIEF.md, PACKAGING_SPEC.md 전체. 전제: T1~T6 완료, 문제별 최종 후보 id 확정(configs/ocean_v2/final_package_v2.json에 기록).
목표: scripts/ocean_v2/build_final_package_v2.py 와 scripts/ocean_v2/cleanroom_verify.ps1 을 구현해 artifacts/official_final_submission_v2_20260907/{P1,P2,P3,upload,MASTER_MANIFEST.json} 을 생성하고, 새 임시 폴더에서 클린룸 재현(학습→추론→SHA 대조, 소요시간 기록)을 실제로 실행한다. 기존 artifacts/official_final_submission_20260905 와 scripts/build_official_final_submission_20260905.py 는 수정하지 않는다. 업로드 파일 전부 ≤50,000,000 bytes, audit_constants 통과, pytest 통과, README 필수 항목 확인. 네트워크 업로드는 절대 하지 않는다.
보고: 폴더 트리, 업로드 파일 목록(bytes·SHA), 클린룸 결과(SHA 일치 여부·소요시간), README 요약.
```

## T8 — 사용자 전달물
```
저장소 ... 읽을 문서: PACKAGING_SPEC.md §4, reports/claude_recon_20260905/00_SUMMARY.md §7. 전제: T7 완료.
목표: docs/ocean_v2_codex/USER_HANDOFF.md 작성 — 문제별 답안 CSV 절대경로·행 수·SHA, 최종 모델 업로드 파일 집합(파일명·bytes·SHA), FORM 값(제목/한 줄 요약/저장소 URL https://github.com/choihyunjin1/-oceanaidata_track1 /비고), 기대 Public 범위(sanity 용도 명시), 클린룸 결과, 삭제 권고 제출 목록, 클릭 순서(답안 업로드 완료 확인 → 삭제 → 문제별 모델 최종 제출). 
```
