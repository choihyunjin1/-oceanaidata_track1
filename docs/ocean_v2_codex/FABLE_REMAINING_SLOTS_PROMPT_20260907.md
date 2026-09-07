# Fable 독립 연구 요청 — 남은 5회 활용 계획

아래 내용을 Fable에게 전달한다. 2026-09-07 12:30 KST 기준 스냅샷이며 실행 시각·진행 상태·잔여 횟수는 다시 확인해야 한다.

```text
저장소 C:\Users\cedis\PycharmProjects\PythonProject.
목표는 규정에 맞고 재생성 가능한 답안의 실제 점수 향상이다. 무조건 새 구조를 찾지 말고, 현재 자산을 활용해 남은 리더보드 기회를 어떻게 쓸지 독립 연구·반론 검토 후 실행 가능한 계획을 작성하라. 이번 요청은 연구·읽기 전용 집계·새 계획 문서 작성이다. 모델 학습, 설정 수정, 답안 생성, 업로드, 삭제, 최종 모델 지정, commit/push는 하지 않는다.

1. 먼저 읽을 정본
- AGENTS.md, 00_ORGANIZER_DATA_POLICY.md.
- README.md와 docs/ocean_v2_codex/UPLOAD_SET_2.md 최신 절.
- reports/final_day_candidate_official_submissions_20260907_v1/receipt.json 및 report-source.md: 실제 공식 채점과 답안 SHA의 정본.
- reports/p1_champion_bracketb_20260907_v1/{result.json,independent-qa.json,report-source.md,release-result.json}.
- reports/p2_l120_s10_proj_20260907_v1/{result.json,independent-qa.json,packaging-amendment.json,report-source.md}, reports/p2_l120_s3_proj_20260907_v1/ 및 reports/final_day_evidence_check_20260907_v1/result.json.
- reports/p3_numeric_cpudet_noshrink_20260907_v1/{result.json,independent-qa.json,internal-evaluation.json,report-source.md}, configs/experiments/p3_numeric_cpudet_noshrink_20260907_v1.json.
- reports/p3_numeric_cpudet_unbounded_20260907_v2/{launch.md,completion-answer-independent-qa.json}.
- docs/ocean_v2_codex/FINAL_DAY_PLAN_20260907.md 및 FABLE_FINAL_DAY_VERIFICATION_FEEDBACK_20260907.md, FABLE_INDEPENDENT_REVIEW_2_20260907.md. 정정 전 수치·명령을 최신 상태로 되살리지 않는다.
필요한 문제의 계약과 데이터 README만 추가로 읽고, 과거 전체 세션을 다시 훑지 않는다.

2. 현재 확정 사실
09-07 12:17~12:19에 아래 네 답안을 실제 업로드했고 채점완료를 확인했다.
P1 bracketB 7538082a: F1 0.827921 / 28.7598점. 원형 57844ef2: F1 0.833548 / 28.909341점보다 낮다.
P2 L120 s3 projection 9c5fec38: RMSE 0.405920℃ / 28.240037점. 기존 fee6118b 0.418892℃ / 28.077280점보다 개선됐다.
P2 L120 s10 projection 672c1df1: RMSE 0.418378℃ / 28.083729점. s3 projection을 넘지 못했다.
P3 CPU s3 completion 2015b387: RMSE 0.609836m / 23.654388점. ff42a6a0 0.604351m / 23.741446점보다 낮다.
새 미채점 P3 no-shrink 70761aff: 1,200행, 긴 12/18/24h의 고정 shrink 0.2만 제거, 새 fit 0. 현재 CPU OOF RMSE 0.681321563→0.677450331m, 951 station×episode / bootstrap4000, P(개선)=0.914, CI90[-0.008524918,+0.000733607]m. 저파고 hs0<1.7은 +0.009921904m 악화. 별도 PID와 실제 ZIP 추출 노트북 답안 SHA 일치. 전체 fresh cold 증명은 아직 대기다. 과거 GPU OOF 수치와 혼동하지 않는다.
남은 횟수는 업로드 직후 P1 2/P2 1/P3 2. 정확한 공식 최종 마감 시각·첨부 제한·일반 모델 6시간 적용 범위는 미확인이다. 15:00는 과거 내부 목표이지 공식 마감이 아니다.

3. 실행 중인 작업 보호
C:\Users\cedis\Documents\OceanFinalDay_20260907\P3_numeric_cpudet_unbounded_v2 의 fresh_cold_2는 CPU4, 빈 폴더부터 backbone36+router5=41fit로 진행 중이다. completion_1은 검증된 과거29backbone 재사용+새7backbone+5router 완료다. 둘을 'fresh cold 2회'로 표현하지 않는다.
프로세스/진행 metadata만 읽고 중단·재시작·thread·seed·feature·shrink·모델·소스·lock·artifact는 바꾸지 않는다. no-shrink 후보는 이미 준비됐으므로 재생성하거나 중복 작업을 제안하지 말고 남은 검증과 제출 우선순위만 판단한다. 새 자원 상한을 자의로 만들지 않되 동시 작업 비용과 경합을 계산한다.

4. 문제별 연구 질문
P1: 같은 원형에 bracket B만 바꾼 내부 개선(F1 +0.001441, P=.8625)이 공식에서는 왜 악화했는가? 지원 셀·계절·run/행 가중·O/MS와 B의 조합·이미 노출된 검증의 선택 편향을 구분하라. 원형 대비 독립적인 이득을 낼 기존 적격 자산이 실제 있는지 확인하고, 남은 2회에 후보 0~2개를 우선순위로 제안하라. 셀별 O/B 조합의 역사적 선택 알고리즘 미복구를 숨기거나 사후 코드를 원본이라고 쓰지 말라. 철회된 범위 룰을 'TP20 보장'으로 복원하지 말라.
P2: s10이 내부 pooled는 1.236732066→1.216631946로 개선했지만 B3는 .463529468→.471278625로 악화하고 공식에서도 s3 projection보다 낮았다는 사실을 함께 설명하라. 8블록 중 개선은 2개이며 B6 기여가 컸다. 마지막 1회가 기존 독립 미채점 후보, 검증된 다른 결합, 또는 예비 슬롯 중 어디에 가치가 큰지 비교하라. 공식 점수에 맞춘 seed 부분집합/혼합 가중 탐색은 금지한다. 내부 데이터로 사전 고정 가능한 선택 절차와 비용이 없는 제안은 실행안에서 제외하라.
P3: 준비된 w=0 후보가 다음 1회를 쓸 만한지 평가하고 마지막 1회의 조건부 분기를 제시하라. 저파고/일부 fold 악화를 자동 탈락으로 삼지 말되 평균 개선, 불확실성, ff42의 whole-cold 불일치와 새 CPU 코어의 재현 검증 상태를 분리하라. 점수 향상과 재현 보장의 효용을 섞어 CPU 후보가 더 좋다고 주장하지 말라. 미리 정한 no-shrink 외의 계수 변경이 필요하면 train-only 선택 근거와 새 독립 검증 비용부터 제시한다.

5. 제약과 판단 방식
배포 데이터만 사용. 외부 관측·KIOST 비배포 원자료·hidden truth는 내부 테스트에도 접근 금지. 인터넷 연구가 필요하면 방법론/공식 문서의 1차 출처만 이용하고 새 데이터나 가중치를 가져오지 않는다. 공식 반환 점수로 계수·임계값·Public 소속·정답을 역산하지 않는다. 독립 후보의 공식 점수 비교는 가능하다.
학습용 자연 급변/고파고를 일괄 제거하지 않는다. 일부 블록 악화만으로 자동 탈락시키지 않는다. 내부 RMSE/F1의 개선을 공식 점수의 확정 개선·상한·하한으로 바꾸지 않는다. 점수 환산을 적을 때는 공인된 변환식 출처와 입력 가정을 명시하고, 없으면 예상 점수 미산출로 둔다.
같은 SHA 재제출은 하지 않는다. 다섯 슬롯을 억지로 채우지는 않되, 제출 가능한 독립 후보를 근거 없이 보류하지도 않는다. 먼저 제출할 준비 후보와 시간이 남으면 만들 후보를 분리한다. 긴 연구 때문에 준비된 후보의 제출 시점을 놓치지 않도록 빠른 1차 권고를 먼저 전달한다.

6. 산출물
docs/ocean_v2_codex/FABLE_REMAINING_SLOTS_REVIEW_20260907.md 를 새로 작성한다. 기존 문서/실험 산출물은 수정하지 않는다.
A. 결론 먼저: 문제별 다음 행동 1줄과 즉시 제출 검토 후보.
B. 표: 문제/남은 슬롯/후보/변경 한 가지/내부 지표와 CI/P개선/악화 위험/중복·규정·재현 상태/새 fit·예상시간의 근거/우선순위/진행·보류 조건.
C. 확정 사실·가능한 설명·미검증 가설을 별도로 분류하고 근거 파일과 JSON키/코드 위치를 적는다. 핵심 집계만 읽기 전용으로 재계산하고 원시 행을 출력하지 않는다.
D. 남은 시간이 30분/90분/3시간일 때의 대체 계획. 실제 마감은 미확인으로 두며 가정 시나리오를 사실로 쓰지 않는다.
E. Codex가 바로 실행할 짧은 프롬프트: 현재 cold 불변, 신규 실험만 별도 ID, 사전등록→내부평가→답안QA→독립replay→패키지→제출준비 순서. 이미 끝난 fit/QA를 이유 없이 반복하지 않는다.
F. 준비된 no-shrink 제출을 지연시킬 필요가 있는지 명시하고, 추가 확인이 필요하면 딱 어떤 파일/검사인지 적는다. 원격 업로드·최종 지정은 이 연구 요청의 권한이 아니다.
```
