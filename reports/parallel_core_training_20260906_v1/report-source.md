# 현재 코어 추가 학습 결과 — 2026-09-06

## 결론과 현재 상태

추가 학습과 독립 검산을 완료했다. 신규 실제 데이터 학습은 총76회(P1 42, P2 31, P3 3)다. P1은 트리 단독의 작은 개선이 실제 고정 MS 결합 평균 개선으로 이어지지 않아 기존 후보를 유지한다. P2 L120과 P3 numeric-lead는 새 full 학습·공식 양식 CSV·새 프로세스 전체 CSV 재현 검사를 마쳤다. 두 후보의 내부 평균 이득에는 불확실성과 구간 악화가 있으므로 공식 점수 상승을 보장하지 않는다. 기존 최고점 답안은 보존했다. 신규 업로드·commit·push·최종 모델 잠금은0이다.

| 문제 | 이번 변경 | 상태 | 신규 실제 학습 | 내부 주지표 | 제출본 |
|---|---|---|---|---|---|
| P1 | 현 O1/B3 트리 학습량·정규화 4정책 | 완료·기존 결합 후보 유지 | 42fit/1641.191초 | 트리 전체 F1 +0.001631434; 고정 MS 결합 −0.000302283 | 새 답안 생성 안 함 |
| P2 | C3 60/120 epochs 및 weight decay 비교 | LOCAL_CANDIDATE_READY_NOT_UPLOADED | 28fit/757.673초 + full3fit/121.672초 | 가을3seed 0.488284326→0.483505057℃ | 26,061행·QA PASS |
| P3 | lead 수치형 전환; hmax 유지 | LOCAL_CANDIDATE_READY_NOT_UPLOADED | full2 + router1 =3, 254.056초 | 0.684038593→0.683538174m, Δ−0.000500418m | 1,200행·QA PASS |

## P3 검증된 범위

- 과거 동일 numeric 실행의 historical 15 backbone+8 router fit 및 103,602행 OOF를 hash로 재사용했다. 이번에 historical 학습을 반복하지 않았다.
- root는 paired OOF의 SSE와 pooled RMSE를 독립 재계산하고 일치 확인했다. 표본을 삭제하지 않았다.
- 역사 비교 CI90은 [−0.001879861, +0.000809684]m, bootstrap 개선 비율0.752다. 이는 회고적 내부 비교이며 공식 성능의 확률이 아니다. Q2_2025는 +0.004406613m 악화했다. 평균 개선 때문에 후보를 유지하되 이 위험을 숨기지 않는다.
- 신규 full single700 CPU2, multi1200 GPU0, OOF router를 학습했다. 공식 입력은 내부 QA 이후에만 추론에서 읽었다. 외부·hidden·sample 값 접근 및 업로드는0이다.
- synthetic28/Ruff PASS, 학습/수치형 QA24 PASS, root 별도 검산28 PASS. 저장 모델의 새로운 프로세스 예측 및 전체 답안 SHA가 일치했다. 이는 scratch 학습을 두 번 반복한 검증은 아니다. 새 최종 portable 패키지 whole-cold 검증은 아직 수행하지 않았다.
- 후보: `artifacts/p3_numeric_candidate_20260906_v1/05_answer/submission.csv`
- SHA256: `ff42a6a08c76f0d58ed2f3a9ea31a08819ada5fa6e9007af942fe0b891937960`
- 공식 점수: 미측정. 내부 RMSE 변화로 공식 점수를 역산하지 않는다.

근거: `reports/p3_numeric_lead_forward_gpu_20260906_v2/result.json`, 신규 `artifacts/p3_numeric_candidate_20260906_v1/04_logs/`의 학습·QA·답안·replay receipt, 본 폴더 `p3-root-independent-qa.json`.

## 남은 작업과 브라우저 차단

1. 2026-09-06 오후의 기존 Chrome 제출관리 탭 연결을 한 번 재확인했으나 `Debugger unattached`였다. 탭 목록 조회 성공은 실제 페이지 연결이나 제출 성공을 뜻하지 않는다. 보안상 차단됐던 재시작을 우회하지 않았다.
2. Chrome 연결 복구 후 현 제출 기회와 동일 SHA 중복 여부를 확인하고 승인된 후보를 제출할 수 있다. 기존 P1 b2f17 제출 감시와 중복 업로드하지 않는다. 이 보고서 시점의 신규 공식 점수는 미측정이다.
3. 후보 재현 검사는 저장 모델의 새 프로세스 재생까지다. 완전히 독립된 오프라인 패키지에서 scratch 학습을 두 번 반복했다는 주장은 하지 않는다.
4. 이번 작업은 commit/push 및 최종 모델 잠금을 하지 않았다.

## P1 — 학습률 절반·학습량 두 배의 작은 이득

세 fold 모두 earlier-inner에서 O_slow(lr0.02/1400trees)가 선택됐다. Outer 결과로 recipe를 고르지 않았다. 원래 O(lr0.04/700trees) 및 B3를 같은CPU2로 다시 학습한 비교다. 실제42fits/1641.191초, 새PID28416의 저장모델/선택 재현36checks PASS(223.797초), root OOF count/F1/모델SHA 검산 PASS.

| 구간 | control F1 | candidate F1 | Δ F1 |
|---|---:|---:|---:|
| 전체Q2/Q3/Q4,421032행 | 0.841633764 | 0.843265197 | +0.001631434 |
| Q2 | 0.759196056 | 0.764561271 | +0.005365215 |
| Q3 | 0.869649123 | 0.870456337 | +0.000807214 |
| Q4 | 0.908802344 | 0.908141703 | −0.000660641 |
| 기존 보조범위Q3/Q4,287862행 | 0.886019090 | 0.886034323 | +0.000015233 |

전체 TP12683→12665, FP1401→1318, FN3372→3390. CI90 [−0.003067240,+0.006128381], bootstrap 개선비율0.7405. 평균 이득은 작고 불확실하다. Q2 미지원셀26062행은 악화됐고 일부정점·층의 recall 손실도 집중된다. 위험을 자동 veto로 쓰지 않지만 전체평균 개선만으로 모든 구간·공식점수 개선을 주장하지 않는다.

기존 고정MS proposal을 두 CPU2 tree에 동일하게 결합하는287862행 exact-key/원fold-ownership 진단도 완료했다. control+MS F1 0.905472150→candidate+MS 0.905169867, Δ−0.000302283이다. CI90 [−0.005915730,+0.004809858], bootstrap 개선비율0.4945. Q3는 −0.00101995, Q4는 +0.00057746이다. **한 Q 악화가 아니라 실제 제출 구성의 pooled 평균이 개선되지 않아** 추가 full13fit을 실행하지 않고 기존 b2f17/fallback을 유지한다. 트리 단독은 연구 후보로 보존한다. 이 보조 비교는 주평가421032행을 대체하지 않는다. 기존CPU4 champion 수치는 자원이 다른 참고비교일 뿐 동일조건 tuning 이득으로 계산하지 않는다.

저장 모델/선택 재생36/36, 트리 독립QA158/158, MS 비교181검사 PASS. Root가 원래 키·fold 귀속·라벨·OR 결합을 재계산했다. [root P1 검산](p1-root-independent-qa.json), [root MS 결합 검산](p1-ms-bridge-root-independent-qa.json).

## P2 — 주평가 평균 개선, 전체/결측 위험 있음

동일 CUDA/CPU2의 60epoch 대120epoch 비교다. 현재 C3는 각 seed의 baseline+scale×normalized 예측을 평균하며 envelope/PAVA를 쓰지 않는다. 새 비교도 같은 표면이다.

| 평가 범위 | C60 | L120 | Δ RMSE℃ |
|---|---:|---:|---:|
| 전체8fold 첫seed,166268행 | 1.270852306 | 1.281373210 | +0.010520904 |
| 가을B3 첫seed,26273행 | 0.465330203 | 0.481479504 | +0.016149301 |
| 가을B3 추가2seed 앙상블 | 0.533473508 | 0.525777767 | −0.007695741 |
| 가을B3 전체3seed 앙상블(주평가) | 0.488284326 | 0.483505057 | −0.004779270 |
| 가을B3 전체3seed·T5 결측 스트레스 전체fold | 0.538760112 | 0.563601402 | +0.024841290 |
| 가을B3 전체3seed·실제 결측 지정구간만 | 약0.445059 | 약0.561966 | 약+0.116907 |

최초 seed로 L120/D60 중 선택한 뒤 사전등록된 추가2seed를 수행했다. 추가seed는 독립 기간 검증이 아니고, 전체8fold3seed 실험도 아니다. 최초seed의 가을/전체 결과는 악화였으며 3seed 가을 평균에서만 개선이 확인됐다. 따라서 보편적으로 더 좋은 모델이나 공식점수 상승을 보장한다고 표현하지 않는다. 사용자의 평균 개선 우선 원칙에 따라 가을 주평가 개선 후보를 유지하고, 기존 최고점 fallback을 보존한 채 검증용 CSV를 제작한다.

실제28fits/757.673초. 새PID에서28개 전체 모델·natural/outage 예측 exact replay(25.156초), 독립QA821 PASS. Root는 별도로 모델SHA와 same-fold training keys/truth/arrays가 모든recipe/seed에서 동일함을 확인하고 주요 SSE/RMSE를 재계산했다. [root P2 검산](p2-root-independent-qa.json).

별도 `p2_c3_training_comparison_full_20260906_v1`의 L120 full3seed(추가3fit), GPU0/CPU2,120epoch/wd1e-4도 완료했다. 학습121.672초, 재생·추론 포함 후속 전체169.368초다. 원28fit 실험을 변경하거나 재시작하지 않았다. 빈 모델 폴더에서 신규 학습 후 새 프로세스 모델 재생19검사, 최종16검사 및 synthetic5/Ruff PASS다. 내부 QA 이후에만 공식26061행을 추론했으며 새PID 전체 CSV SHA가 일치한다. Root는 별도로 schema/key/order/finite/중복·모델SHA·fit 수를 검산했다. [root P2 답안 검산](p2-answer-root-independent-qa.json).

## 새 제출 후보와 보존 범위

| 문제 | CSV 경로(저장소 기준) | 행 수 | SHA256 |
|---|---|---:|---|
| P2 | `artifacts/p2_c3_training_comparison_full_20260906_v1/05_answer/submission_p2_L120_3seed.csv` | 26,061 | `fee6118bb4a4d1d421094aa0174d634cf73e919804ea1b62ab0bab4f1384ce4d` |
| P3 | `artifacts/p3_numeric_candidate_20260906_v1/05_answer/submission.csv` | 1,200 | `ff42a6a08c76f0d58ed2f3a9ea31a08819ada5fa6e9007af942fe0b891937960` |

제출 제목 제안은 `P2 C3 L120 3seed 20260906`, `P3 numeric lead 20260906`이다. P2는 학습량만 늘린 가을 주평가 개선/결측 악화 후보, P3는 hmax를 유지한 리드 수치형 전환의 작은 평균 개선 후보다. P1 기존 b2f17과 세 문제 fallback의 경로·SHA는 [보존 목록](preserved-candidates.json)에 있다. 소스 데이터·기존 모델·답안은 덮어쓰지 않았다. 공식 입력은 P2/P3의 내부 검산 이후 허용된 추론 및 키 검증에만 사용했고, hidden truth·외부 자료·공식 점수 역산은 사용하지 않았다.

P3의 기존 고정 장기리드 persistence 0.2는 2026-08-17 내부 OOF 진단에서 선택된 계보이며 현재 학습에서 새로 적합한 값이나 공식 점수 역산값으로 표현하지 않는다. 상세 출처와 재현 범위는 문제별 보고서에 남겼다. 검산 PASS와 통계적 개선 확정, 공식 점수 개선, 최종 오프라인 패키지 완성은 서로 다른 상태다.
