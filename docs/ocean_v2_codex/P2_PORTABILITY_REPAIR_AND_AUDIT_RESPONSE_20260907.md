# Fable 감사 정정·최소 수정 결과 — 2026-09-07 15:46 KST

## 결론: P2 차단 결함 수정·새 cold/saved exact, 재검토 대기

사용자 요청으로 새 v2 패키지에서 역사적 기반 답안 SHA 차이에 의한 불필요한 중단을 수정했다. **최종 답안은 기존 채점 SHA 794268f1과 그대로 같다.** 옛 ZIP·모델·채점 CSV·Fable 감사 원문은 보존했다. 포털을 조작하지 않았고 최종 확인은 계속 보류한다. 실행 중 P3는 15:46 PID26996 생존·25/36 backbone이며 변경하지 않았다.

### 실제 변경과 검증

| 항목 | 변경 전 | 변경 후 / 실제 증거 |
|---|---|---|
| P2 평활 진입 | `assert sha(source) == BASE`로 역사적 예측과 달라지면 무조건 중단 | 현재 실행 independent-qa PASS·모든 checks true·현재 answer SHA 일치 요구. 역사적 SHA는 expected/actual/equality로 분리 기록하고 계속 |
| 수학/학습 | L120 3-seed, smooth7 → clip/PAVA | 동일. 새로운 탐색·Public 재적합 없음. 재현 검사에서만 새 3fit |
| 새 cold 노트북 | v1 증거만 존재 | v2 TRAIN 179.766초 + PREDICT 39.172초 + SMOOTH_PROJECT 14.672초 = **233.610초**, 3fit. 별도 PID 후처리 infer/replay 일치 |
| 새 saved 노트북 | v1 증거만 존재 | v2 SAVED_PREDICT **53.360초**, 0fit, infer PID41852 / replay PID42160 exact |
| 최종 답안 | 794268f15a0a7ac18ecd4dc99757083e159c49d639d2a8a72df3f835414cc481 | 같은 SHA, 26061행, 1253654bytes. 기존 공식 성적 귀속 유지, 새 업로드 0 |
| 테스트 | 과거 SHA 불일치 흐름 미검증 | 합성 mismatch 진행·QA 후 변경/실패/빈 QA/누락 거부·평활 불변 + 기존 smooth tests **12 PASS**, 새 스크립트 Ruff PASS |
| manifest | v1 패키지 보존 | core PACKAGE_MANIFEST는 core 파일 불변이라 동일. 새 addon SMOOTH_SOURCE_MANIFEST·최종 FINAL_MANIFEST 재생성, 최종 ZIP CRC/멤버 SHA 검사 |
| 상위 문서 | P1 FORM에서 옛 P2/P3 상위 문서 첨부 지시 | P1 FORM 교정, 옛 START_HERE/RELEASE_MANIFEST는 이력 표시. 새로운 세 문제 선택 문서·manifest 사용 |
| P3 문구 | 선택 전 optional 후보 표현 | 외부 FORM/README에 Public 우선·제출 보류·completion_1+saved 증거·fresh cold 미완 명시. ZIP은 불변 |

새 결과·원본 SHA·노트북 receipts: [candidate-ready](../../reports/p2_smooth7_portability_repair_20260907_v2/candidate-ready.json), [최종 ZIP QA](../../reports/p2_smooth7_portability_repair_20260907_v2/p2-final-ready.json), [외부 문서 변경 전후 SHA](../../reports/p2_smooth7_portability_repair_20260907_v2/outer-document-change-sha.json), [선택 manifest/원 ZIP 불변](../../reports/p2_smooth7_portability_repair_20260907_v2/final-selection-manifest.json), [사전 수정 계약](../../reports/p2_smooth7_portability_repair_20260907_v2/preregistered-repair.md).

## 새 파일 선택 — v1과 혼동 금지

- 새 P2 최종 ZIP: `C:/Users/cedis/Documents/OceanFinalSelected_20260907/P2_v2/P2_FINAL_REPRODUCTION_794268f1_v2.zip`
- **485957 bytes / SHA256 36b4b0e4a0de61277133a85fefd62a6629aa1ac1c109fc12652b1d1cb745408f**.
- 같은 폴더 `FORM.md`와 ZIP 최상위 README 사용. v1 aab30bbe ZIP은 보존용이며 재개 뒤 폼의 파일을 교체해야 한다. 이번 작업은 포털 교체를 하지 않았다.
- 세 문제 상위 안내: `C:/Users/cedis/Documents/OceanFinalSelected_20260907/FINAL_SELECTION_20260907.md` 및 `FINAL_SELECTION_MANIFEST.json`.
- P1은 원형 57844ef2, P2는 같은 답안 794268f1의 새 코드 패키지, P3는 70761aff. P2 두 역할 폴더는 합치지 않고 P3 source/model 확장은 합친다. P1 15분할 첨부 허용 여부는 미확인이다.

## 감사에 대한 정정

1. **P2 해시 게이트 결함은 인정하고 수정했다.** 단순히 검사를 없애지 않고 현재 실행 무결성 검사를 추가했다. 새 portable v2가 실제 cold/saved 진입점이며, 옛 projection-only main의 historical BASE gate는 실행하지 않는 PROJECT 이력 경로에 남는다. 빌더에서 기존 채점 답안과 exact를 요구하는 것은 우리 PC의 패키지 검증이며 검증 PC용 예측 실행 게이트가 아니다.
2. **GitHub 누락 지적은 사실과 다르다.** 기존 커밋 6ecf287에 `scripts/p2_smooth7_portable_20260907.py`, `p2_final_day_projection_20260907_v1.py`, `p2_final_day_materialize_20260907_v1.py`가 모두 있다. P3도 `p3_numeric_cpudet_noshrink_20260907_v1_portable.py`, `p3_cpudet_unbounded_20260907_v2.py` 및 builder가 있다. ZIP 내 이름과 저장소 생성 원본 이름은 다를 수 있다. 새 v2 코드 공개 여부는 실제 Git 상태로 별도 확인해야 하며 기존 커밋이 v2를 담는다고 주장하지 않는다.
3. **저장 모델 경로는 필수 학습 재현의 대체 증거가 아니다.** P3 문구를 이 점까지 보완했다. fresh cold 및 새 모델 no-shrink 답안 비교 전 완료를 주장하지 않는다.
4. **전역/fold-local 경계 차이:** Fable 원 계산이 cross-fold global이었다는 설명을 받았다. 80행 차이는 그 설명과 기존 독립 진단에 부합한다. fold-local은 fold별로 서로 다른 학습 모델의 예측을 섞지 않는 비교이고, 배포 평활은 단일 배포 모델의 시계열에 적용한다. 이것이 Private 성능 보장은 아니다.
5. **P2 접수 상태:** Fable의 15:30 읽기 미접수 관찰은 Fable 제공 증거다. 이번 Codex 수정에서는 포털을 다시 열거나 클릭하지 않았다. 재개 시 최신 접수 상태를 먼저 확인해야 한다.

## 여전히 미확인

타 GPU/드라이버·CPU·fresh venv·OS 차단망 재현은 이번 검사 범위 밖이다. CUDA 요구는 유지된다. 같은 실행의 별도 PID exact replay 검사는 유지하므로 실행 간 비결정성이 있으면 여전히 실패할 수 있다. 과거 SHA 차이 차단을 제거했다고 임의 환경의 재현성이 증명되는 것은 아니다.

P1 여섯 셀 선택 프로그램 미복구, P3 fresh cold 미완, 일반 모델 6시간 적용, 정확한 모델 마감 시각, 첨부 개수, 재현 대상 답안과 허용오차는 남은 검토 항목이다. 감사 문서는 운영진 적격성 인증이 아니다.

다음: [Fable 재검토 프롬프트](FABLE_FINAL_SELECTION_RECHECK_PROMPT_20260907.md). 재검토 PASS 이후에도 사용자 명시적 재개 지시 전 최종 확인 금지.
