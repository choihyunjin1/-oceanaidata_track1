# 최종 제출 재개 절차 — 실행 보류 (2026-09-07)

> **최신 상태 — 2026-09-07 18:40 KST:** Fable 재검토 PASS. P3 전체 신규 41fit 완료(22182.109초), 새 모델의 기반 `2015b387…` 및 최종 no-shrink `70761aff…` **exact**, 별도 PID·13개 QA PASS. 추가 후처리 대조 14.016초. 전체 fresh cold는 1회이며 일반 모델 6시간 적용 범위는 미확인입니다. 기존 ZIP·모델·채점 답안 불변. **사용자 최종 제출 재개 지시 전 업로드·확인·삭제 및 Git commit/push 보류**. [직접 검증 receipt](C:/Users/cedis/PycharmProjects/PythonProject/reports/p3_fresh_noshrink_confirmation_20260907_v1/result.json). 아래 이전 시각 기록보다 이 안내가 우선합니다.

Fable 재검토 PASS는 사용자 재개 승인이 아니다. 이 문서는 실행 계획이며 포털 확인·첨부·삭제·확인창 OK를 실행하지 않았다. 사용자의 명시적 재개 지시 전에는 아래 어느 포털 단계도 실행하지 않는다. Git add/commit/push도 별도 승인 전 금지한다. 실행 중 P3 PID 26996의 설정·모델·프로세스를 변경하거나 중단하지 않는다.

## 0. 승인과 파일 확인

재개 지시가 어떤 문제의 업로드·최종 확인까지 포함하는지 확인한다. 단순 감사 PASS나 문서 정리는 승인이 아니다. 파일 이름뿐 아니라 bytes/SHA를 대조한다. 문서와 파일 해시가 다르면 중단하고 보고한다. 마감 시각·첨부 개수 상한·일반 모델 6시간 규칙 적용 여부는 미확인이다. 추정값으로 채우지 않는다.

현재 선택 문서는 C:/Users/cedis/Documents/OceanFinalSelected_20260907/FINAL_SELECTION_20260907.md 및 FINAL_SELECTION_MANIFEST.json이다. 옛 C:/Users/cedis/Documents/OceanFinalRelease_20260907/START_HERE.md와 RELEASE_MANIFEST.json은 어떤 문제에도 첨부하지 않는다.

## a. P2 서버 접수 여부부터 읽기

https://oceanaidata.org/app/problems/6 및 https://oceanaidata.org/app/submissions를 읽는다. 문제명이 OCN-02인지, 문제 페이지의 "모델 최종 제출하기" 버튼과 제출관리의 모델 제출 항목·접수 시각·표시 상태를 함께 확인한다. 버튼 존재만으로 미접수를 확정하지 않는다. 일반 답안 채점 항목과 모델 최종 제출 항목을 구분한다.

중단 조건: 이미 P2 모델 접수 항목이 있으면 중복 제출하지 않고 보고한다. 로그인 불가, 화면 간 모순, 모델 항목 구분 불가, 서버 상태 불명확이면 중단한다. 마지막 Fable 관찰은 15:30 전후이며 현재 서버 상태의 증거가 아니다.

## b. P2 모달 및 v2 첨부 확인

서버 미접수를 확인하고 승인 범위 안에서만 "모델 최종 제출하기" 모달을 연다. 현재 첨부 이름·크기를 읽는다. v1 SHA aab30bbe5e244098c1ef379b09077ccd4e98bd6a6cd0ccf778109a7f5282054d가 남아 있으면 승인 후 모달의 초안 첨부만 제거하고 아래 v2로 교체한다. 과거 서버 제출 기록을 삭제하는 단계가 아니다.

- 첨부: C:/Users/cedis/Documents/OceanFinalSelected_20260907/P2_v2/P2_FINAL_REPRODUCTION_794268f1_v2.zip
- bytes: 485957
- SHA256: 36b4b0e4a0de61277133a85fefd62a6629aa1ac1c109fc12652b1d1cb745408f
- 제목·한 줄 요약·설명: 같은 폴더 FORM.md를 그대로 사용한다. 외부 FORM의 v1 README 시각 정정도 포함한다.

확인할 화면: OCN-02 문제명, v2 파일명/485957 B에 대응하는 크기 표시, 제목 "P2 L120 3-seed smooth7 endpoint projection", 첨부 업로드 완료 상태. 최종 확인창의 실제 문구를 읽고 기록한다. OK는 사용자의 명시적 최종 확인 지시 후에만 누른다.

중단 조건: v1 잔존, 중복 첨부, 파일·제목 불일치, 업로드 중/오류, 승인 없는 확인창, 예기치 않은 제출/덮어쓰기 경고. 승인 없이 OK 또는 확인에 해당하는 키를 누르지 않는다.

## c. P2 접수 확인 후 P1

P2 확인 이후 제출관리에서 모델 항목의 문제명·시각·파일명·표시 상태를 확인한다. 성공 알림만으로 완료 처리하지 않는다. 접수 확인 불가 시 P1로 넘어가지 않고 보고한다.

P1은 C:/Users/cedis/Documents/OceanFinalRelease_20260907/P1/FORM.md를 기준으로 SOURCE_ONLY ZIP, SAVED_MODEL_PARTS의 15 part ZIP, 재조립 manifest/tool, 해당 FORM, 새 FINAL_SELECTION 문서를 준비한다. 큰 SAVED_MODELS.zip 하나와 분할본을 동시에 첨부하지 않는다. 파일별 해시·part 개수·재조립 대상은 기존 manifest와 대조한다. P1 답안 연결 SHA는 57844ef235f987059d17bddb035cc0a98b92bb0d53263cf878c68b31bbc53687이다.

확인할 화면: OCN-01 문제명, FORM과 같은 제목·요약, 15개 part 및 나머지 필수 첨부의 완료 표시. 중단 조건: 첨부 개수/크기 제한, 누락 part, 재조립 자료 미첨부, 중복 모델 접수, 승인 없는 확인창. 제한에 걸리면 SOURCE_ONLY만 + 재학습 안내 대안을 사용자에게 보고하고 선택을 기다린다. 임의 축약·재포장·최종 확인하지 않는다.

## d. P3

C:/Users/cedis/Documents/OceanFinalDay_20260907/P3_numeric_cpudet_noshrink_v2/의 SOURCE_ONLY.zip + SAVED_MODELS.zip + 외부 FORM.md/README.md를 사용한다. SHA256은 각각 c3aed055873bed601397073be4610fb06fa085021122f3656d65c1939faa6e78 및 f0451b9cfe7a3042b9848679d9cfd654e7800e958d6a6e3374bd56332ecfdc7b이다. 답안 연결 SHA는 70761affca4d3fc6f1d24ae53467e5b185b23465926ebb4b851f0300872cddbd이다.

fresh_cold_2 완료 전에는 외부 문서의 "fresh cold 미완" 문구를 유지한다. 저장 모델 replay를 전체 학습 재현 증명으로 바꾸어 쓰지 않는다. 완료 후에는 실제 receipt와 두 답안 해시 대조 결과만 외부 문서에 추가하며 ZIP은 불변이다.

확인할 화면: OCN-03 문제명, FORM 제목·요약, 두 ZIP과 외부 문서의 완료 표시. 중단 조건: 잘못된 계보·구버전 ZIP, 파일 해시 불일치, 접수 상태 모호, 첨부 오류, 사용자 최종 확인 지시 없음. 실행 중 학습을 제출 준비 때문에 중단하지 않는다.

## e. 문제별 접수 receipt

각 문제의 사용자 승인된 최종 확인 직후 /app/submissions에서 모델 항목을 읽고 receipt를 저장한다: 관찰 시각(KST), 문제명/ID, 모델 제출 시각, 첨부 파일명(화면에서 보이는 범위), 화면 표시 상태 원문, 확인 가능한 접수 ID, 해당 로컬 파일 bytes/SHA, 화면 캡처 경로. 화면에 없는 필드는 미확인으로 둔다. 답안 점수 receipt와 모델 접수 receipt를 별개로 보존한다.

네트워크 오류/응답 불명확 시 재클릭하지 말고 제출관리부터 다시 읽는다. 접수됐으면 중단하고 보고한다. 삭제·재제출은 별도 명시 지시 없이는 하지 않는다. 세 문제 실제 접수 확인이 끝나기 전에는 "최종 제출 완료"라고 보고하지 않는다.
