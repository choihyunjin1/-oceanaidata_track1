# Fable에게: 슬롯 설계 중지, 실제 결과와 최종 패키지 독립 감사

먼저 **[FINAL_SELECTION_REVIEW_HANDOFF_20260907.md](FINAL_SELECTION_REVIEW_HANDOFF_20260907.md)**를 전부 읽으세요. 이 문서가 이전 14:45 진행 표보다 최신인 정본입니다. GitHub에는 작은 코드·집계·문서만 있으며 모델/답안/ZIP/원자료는 로컬 전용입니다. 원격 검토만 가능하면 로컬 검증은 미수행이라고 명시하세요.

## 사용자 최종 제출 보류 — 이 절이 아래 실행 지시보다 우선

사용자가 Fable 검토를 위해 제출 버튼 직전에 멈추도록 지시했습니다. P2는 이 지시 전에 첫 `모델 최종 제출` 버튼을 눌러 최종 확인창이 열린 상태까지 진행했으나, 확인창의 OK는 누르지 않았고 최종 접수 receipt도 확인하지 못했습니다. 마지막 읽기 재연결은 `Debugger unattached`로 실패했으므로 현재 서버 접수 상태를 새로 확인했다고 주장하지 않습니다. P1/P3 최종 제출 버튼은 누르지 않았습니다. 검토 후 사용자가 재개를 지시하기 전까지 모든 최종 확인·업로드·최종 지정은 보류합니다. P3의 실행 중 fresh_cold_2 학습은 변경하거나 중단하지 않습니다.

최신 채점 완료: P1 B5 `cbeb7426` F1 0.831622 / 28.858163점, O_slow `c38ace7a` F1 0.826087 / 28.711047점. 두 후보 모두 기존 원형보다 낮으며 P1/P2/P3 답안 잔여 횟수는 모두 0입니다. 아래 P1 제출 진행 중 표기는 과거 상태입니다.

검토 대상 예정 선택은 P1 `57844ef2` (28.909341), P2 `794268f1` (28.373869), P3 `70761aff` (23.881592)이며, P3 fresh cold 증명은 아직 대기입니다. 최종 제출 확정으로 해석하지 마세요.

P2 준비 파일: `C:/Users/cedis/Documents/OceanFinalSelected_20260907/P2/P2_FINAL_REPRODUCTION_794268f1.zip` (484699 bytes, SHA256 `aab30bbe5e244098c1ef379b09077ccd4e98bd6a6cd0ccf778109a7f5282054d`). `reports/final_selected_submission_20260907/p2-final-ready.json` 및 ZIP 최상위 README/FINAL_MANIFEST를 우선 대조하세요. ZIP·모델·답안은 수정하지 말고 필요한 정정만 감사 문서에 기록하세요.

저장소 `C:/Users/cedis/PycharmProjects/PythonProject`. 먼저 `ALL_REMAINING_SLOTS_TO_FINAL_20260907.md`의 **가장 최신 실행 갱신**을 읽고, 다음을 읽기 전용으로 검토해 주세요. 원본 FABLE_ALL_SLOTS_EXECUTION_DESIGN 문서는 작성 경합으로 P3 잔여 횟수가 오래된 상태입니다. 새 fit, 모델/답안/계수 변경, 업로드, 삭제, 최종 지정, commit/push는 하지 마세요. 실행은 Codex가 담당합니다.

## 확정 사실 — 14:45 이후

- P3 남은 두 슬롯은 이미 no-shrink `70761aff`와 mean/router 제거 `c9fa5366`로 채점했습니다. 각각 Public RMSE **0.595521 / 0.600933 m**, 점수 **23.881592 / 23.795692**. 현재 P3 잔여0입니다. no-shrink는 재가중 OOF 악화 예상과 반대로 Public에서 개선됐습니다. 이것은 재가중이 항상 틀렸다는 증거도, Private 개선의 증거도 아닙니다. 조건부 shrink 신규 두 제출은 오늘 실행할 슬롯이 없습니다.
- 제안한 P2 smooth7 → clip/PAVA `794268f1`를 독립 계산하고 실제 제출했습니다. Public **0.395254℃ / 28.373869점**, 기존 s3 projection보다 −0.010666℃ / +0.133832점. P2 잔여0. 신규 독립 폴더에서 빈 모델 TRAIN→PREDICT→SMOOTH_PROJECT 3-fit 노트북 전체 실행과 별도 SAVED_MODELS 노트북이 같은 최종 SHA입니다.
- P1 B5 / O_slow는 동일 80-feature 원형·같은 MS/cells/GI를 유지한 새 v2 9-fit 학습 완료. 같은 MS-owned Q3/Q4 287862행 독립 bit replay PASS. Q2의 같은 MS evidence가 없어 3fold로 보고하지 않습니다. O_slow ΔF1 +0.001187239, CI90[-0.001344091,+0.003868911], P=.7325; B5 −0.000720695, CI90[-0.002267828,+0.000177531], P=.204. 정보 가치 제출 두 건은 Codex가 QA 후 실행 중입니다. 최신 실제 결과는 문서와 receipt를 확인하세요.
- P3 fresh_cold_2 PID26996 CPU4는 불변이며 14:50경21/36 backbone. 완료 전 whole cold SHA PASS를 주장하지 않습니다.

## 요청하는 감사

1. 위 공식 receipt와 원래 설계의 가설/검증 수치/최종 후보를 구분해 표로 정리하세요. Public만 선택의 유일한 근거로 쓰지 말라는 공지를 유지하세요. Public에 맞춰 w/seed/창을 다시 추정하지 마세요.
2. P2 `reports/p2_l120_s3_smooth7_projection_20260907_v1/{result,independent-qa,candidate-ready,official-receipt}.json`과 portable SOURCE_ONLY/CURRENT_README를 대조하세요. 우리 fold-local B3 .4418539229, pooled1.2047150555입니다. cross-fold global 진단은 B3 .4418588866, pooled1.2046779880, 차이80행으로 Fable 값과 가까우나, 코드 미제공 상태라 원인을 단정하지 않았습니다. 자신의 원 코드에서 fold 경계를 확인해 설명만 남겨 주세요.
3. P1 두 새 후보의 훈련/재생성 출처·공식 점수를 기존 `57844ef2` 원형과 비교하세요. 여섯 셀은 역사적 train OOF 선택 상수이며 선택 프로그램 미복구라는 한계를 숨기지 마세요. 규정 적격성을 운영진 대신 확정하지 마세요.
4. 최종 패키지(학습 포함 SOURCE_ONLY, 저장 모델, 정답, requirements, README/FORM, SHA/분할/재조립, 재현 receipt)의 **서로 다른 버전이 섞이지 않는지** 감사하세요. 기존 고정 README의 옛 답안 지시가 남았으면 새 최상위 START_HERE에서 명확히 우선순위를 안내할 수 있도록 지적하세요. ZIP을 직접 변경하지 마세요.
5. 정확한 모델 마감 시각과 첨부 개수 상한은 여전히 미확인입니다. 08-07 공지의9/7과 답안 quota의자정리셋을 모델마감시각으로 둔갑시키지 마세요. 파일당50MB는 UI 실측입니다. 비적격 과거 제출 삭제는 별도 사용자 결정으로 남깁니다.

산출물: `FABLE_FINAL_SELECTION_AUDIT_20260907.md`. 결론 먼저, 치명적 버전 혼합/재현 결함을 우선하고 `[근거 파일 | 사실 | 가설 | 미확인 | 필요한 수정]`으로 기록해 주세요. Codex의 실행 중 프로세스·model·CSV·manifest·source를 변경하지 마세요. 현재 목표는 남은 슬롯 소모 후 최종 패키지와 **공식 모델 제출 접수까지** 완료하는 것이며 단순 보고로 끝내지 않습니다.
