# Fable 검토 2 반영 — 문서·QA 감사

## 결론: 정정 완료, 적격성은 미확인 항목 별도

2026-09-07 11:12 KST. 검증된 ZIP은 바꾸지 않고 **상위 START_HERE·FORM·RELEASE_MANIFEST와 저장소 안내만 정정**했다. 변경 파일 11개, 직접 SHA 대조한 ZIP 9개 불변. 모델 코드·가중치·답안·평가 수치·새 fit·업로드·삭제·최종 지정·commit·push 모두 0. 기존 P3 학습은 계속 실행 중이며 새 학습을 추가하거나 설정을 변경하지 않았다.

`validate-data` 절차로 실측 영수증, 역사적 출처 설명, Fable의 독립 계산 보고, 미확인 규정을 구분했다. 운영진 원문을 이 작업에서 새로 열람한 것은 아니다. 원문 검색 결과는 사용자 제공 [Fable 검토 2](../../docs/ocean_v2_codex/FABLE_INDEPENDENT_REVIEW_2_20260907.md)에 귀속한다.

## 1. 시간 표현 감사

- 일반 모델의 6시간을 공식 규정 준수 사실로 단정하지 않는다. 사전학습 예외 조건 3은 공식 공지에 있으나 일반 모델 적용 범위와 문제지 Ⅳ-2는 미확인이다.
- 내부 정책·README·PACKAGING_SPEC의 목표는 유지하고, 일반 모델로 확장한 목표가 공식 인용이 아님을 추가했다. 사전학습 예외의 공식 조건 자체를 내부 목표로 격하하지 않았다.
- P1 whole-cold 6,323.356초 / saved35.656초, P2 cold173.010초 / saved30.891초, P3 기존 cold1,497.613초 / saved6.026초는 기존 RELEASE_MANIFEST에서 확인했다.
- P2 후처리4.860초는 신규후보 ANSWER_QA.json, saved43.703초는 result.json의 saved_notebook에서 확인했다. 별개 실행이므로 새 결합 whole-cold 시간이라고 합산하지 않았다.
- Windows/Python3.12.10/Ryzen7800X3D/RTX5090, P1 O/B CPU8·MS CPU2+CUDA0, P2 L120CPU2+CUDA0·투영 별도cap없음, P3 legacy coldCPU2+CUDA0·cpudetCPU4를 명시했다. CPU모델명은 CIM으로 현재 확인했으며 실제 fit별 설정은 보존 recipe/receipt에 따른다.
- RELEASE_MANIFEST의 기존 `cold_6h:true` 및 모든 기존 값은 그대로 두고 `documentation_amendment_20260907`만 추가했다. 해당 boolean은 우리 PC에서의21600초 비교이지 규정 인증이 아님을 명시했다. 기존 필드 전체 JSON 구조 동일 검사 PASS.

## 2. P1 셀 정책 출처 감사

| 위치 | 직접 확인한 add/remove | 조치 |
|---|---|---|
| 로컬 P1/SOURCE_ONLY/02_code/composition.py:21-22 | add G-ORS/1,I-ORS/2; remove S-ORS/1,5,6,I-ORS/4 | 코드 불변 |
| scripts/package_preregistered_submission_20260826.py:56-62 | 동일 | 코드 불변 |
| scripts/build_p1_current_router_oof_anchor_v1.py:52-57 및84-89 | 동일 (현재 줄 번호는 검토 문서의84-85보다 펼쳐짐) | 코드 불변 |

라운드 C 원 보고서에서 F1 0.86467009→0.86690000, CI90 [+0.00091805,+0.00370742]와 선택에 사용한 OOF라는 한계를 확인했다. P1 FORM에 라운드 C 보고서·anchor manifest·tree_recipe parameter_provenance·claim-source-ledger를 연결했다. 기록상 local train OOF로 정한 셀이고 Public 재선택이 아니지만 **선택 알고리즘 자체는 미복구**다. 과거 보고서 출처 확인과 선택 알고리즘 독립 재실행을 구별하며, 사후 알고리즘을 새로 만들지 않았다.

## 3. P2 기록 정합

Fable 검토2 §3은 Codex의 유한 endpoint 조건 적용 후166,268행 최대절대차0.0을 보고한다. 이전 차이는 T1 결측 완전프로필34개/102행에 fmin/fmax가 NaN을 무시한 것. 배포 규칙은 무변경이다. UPLOAD_SET_2에 출처와 이번 작업에서 재계산하지 않았음을 명시했다. result.json SHA 불변 확인. OOF 이득은 test 이득의 상한·하한이 아니다.

## 4. P3 실행·완료 판단

11:11 KST CPU_TRAINING33/36, PID41712. supervisor39904 생존, fatal stderr 없음(local Jupyter TCP warning만). 4스레드·seed·모델·shrink(0.2)·split과 실행 코드는 변경하지 않았다. launch에 Fable의 약17,300초+router/QA 추정을 **추정**으로 추가하고, 후속 실제30fit 비용14,712.832초 및 full single120.687초와 함께 기록했다. 완료 후 fresh_cold_2 receipt의 wall time으로 추정을 대체해야 한다. 준수 미증명.

| 최종 보고 판단 항목 | 현재 | 완료 후 필요 |
|---|---|---|
| completion_1 ↔ fresh_cold_2 답안 SHA | 미완료 | exact 비교, 두 번fresh cold라고 하지 않음 |
| 실제 wall time | 독립 fresh cold 미완료 | receipt 실측, 공식 규정 충족과 분리 |
| 사용자 확인 공식 마감 시각 | 미확인 | 사용자 확인값, 가정으로 대체하지 않음 |
| 보존 fallback | ff42a6a0, RMSE0.604351m /23.741446점 | 후보와 비교하되 자동 최종 지정 없음 |

## 5. 변경 전/후 SHA·재QA

새 fit/추론 replay는 수행하지 않았다. ZIP·수치 코드 변경이 없으므로 기존 exact-replay 증거를 유지했다. 이번 QA는 해시 비교·JSON 기존값 동일 검사·문구 및 출처 대조다. 코드 변경이 없어 pytest/Ruff도 반복하지 않았다. 전체 SHA와 불변 파일까지의 상세 목록은 [audit.json](audit.json).

| 변경 파일 | 변경 전 SHA256 | 변경 후 SHA256 | 재QA 실행 여부·결과 | 미수행 사유 |
|---|---|---|---|---|
| 00_ORGANIZER_DATA_POLICY.md | e2b706559cffda650518b8ea01d564b8680b533046b9195b2ef3fa7e7dfe1819 | 9c2d06659a434f465a9f7afd8c1b5bdaf4b6fda120db458382e7373d30095044 | 문서/JSON/해시 QA PASS; replay 미실행 | ZIP/수치 처리 불변 |
| README.md | a60aca8215e76056eb86bd8928fdee19e6f132643dd8244d5331c7307cd99f96 | 20058e54c58b4e3225158faf5c16a8b2c89330f87b700f386935b6410434c4b6 | 문서/JSON/해시 QA PASS; replay 미실행 | ZIP/수치 처리 불변 |
| docs/ocean_v2_codex/PACKAGING_SPEC.md | a2f31c922b4bca4223e040c58e6021b10ea8daa791da93b2e70016d26518b90d | 1cf34f2655acf1880dba6ba858bcd7e3da0cce8b47263971065406d4fafc4723 | 문서/JSON/해시 QA PASS; replay 미실행 | ZIP/수치 처리 불변 |
| docs/FINAL_RELEASE_20260907.md | f035dc377f7bacb0878631024c509481db1dd1bf60a3ab5298b0750ed1d724d3 | ea12ce76f8f2fea787c5815d250c214aebfd4096f18fa3aee9e8c71dc0bf21d5 | 문서/JSON/해시 QA PASS; replay 미실행 | ZIP/수치 처리 불변 |
| docs/ocean_v2_codex/UPLOAD_SET_2.md | bba7713478f127266547d09b20f9ea6e4e4a194875906ee71e460675517f370d | 723da7bc6e9244ecde86abc13ee981e9eccca6505fa464a4e30acdd06a40c565 | 문서/JSON/해시 QA PASS; replay 미실행 | ZIP/수치 처리 불변 |
| reports/p3_numeric_cpudet_unbounded_20260907_v2/launch.md | 31d936c0c7fb4ed3e1943c60a60d35bde54638da9cff6077ee86d37bdcbfefda | 7afffa84d16e581b30a170dcda4694b5a01e2f764c96bc1efca0699ec2b02232 | 문서/JSON/해시 QA PASS; replay 미실행 | ZIP/수치 처리 불변 |
| LOCAL_RELEASE/START_HERE.md | c4d9948e09f250aa022471792b5f2d5242c716091016c89d4cb85769f9c73133 | 26eedf295437880d82e6cfe8811284cc34af75b7c6dd87f5d8984149daa4448d | 문서/JSON/해시 QA PASS; replay 미실행 | ZIP/수치 처리 불변 |
| LOCAL_RELEASE/RELEASE_MANIFEST.json | cedaa970af4e02c791a4378d821204ff6cc0a64c53609d14e90adfae911e08be | b47d03d230260f5bf256a7f1bffafbbae07f858275c053626ecb2a378025f193 | 문서/JSON/해시 QA PASS; replay 미실행 | ZIP/수치 처리 불변 |
| LOCAL_RELEASE/P1/FORM.md | 63ebd2b5228305b30ddf2c3c292f183ebf7b282a0e105b6d9c7c1788c0ebb8c8 | a9236294c9344da155721b714b874eb76362382744621ff9fc6bb2e56f90138a | 문서/JSON/해시 QA PASS; replay 미실행 | ZIP/수치 처리 불변 |
| LOCAL_RELEASE/P2/FORM.md | 693d36a6e80c4f4cb7962a0ae8103dde2c685816013a83e73ee7acab41b323aa | 6323d782dbfdc08603be89add02f69f7550e8a5c2414ae83a66f035dac2dcc07 | 문서/JSON/해시 QA PASS; replay 미실행 | ZIP/수치 처리 불변 |
| LOCAL_RELEASE/P3/FORM.md | 26e59b1ffc107e7ca89f9f3c05a6d5813b4e2908b99c912a77f5632a08988ee3 | e755150d00016caa88aa459989fb2215348b15f3a63686520abd96c6f330e2fb | 문서/JSON/해시 QA PASS; replay 미실행 | ZIP/수치 처리 불변 |

P1/P2의 SOURCE_ONLY/README는 SHA 불변이다. CURRENT_README는 해당 unpacked 경로에 없었고, P3/SOURCE_ONLY 폴더 자체도 없어 P3_SOURCE_ONLY.zip의 README와 frozen/run 메타데이터를 메모리에서 읽기만 했다. ZIP 내부 README/contract/source-manifest 및 ZIP SHA를 변경하지 않는 경로를 택했으므로 추출 후 수치 replay 재QA는 필요하지 않았다. 상위 정정문이 보존 ZIP 설명의 적용 범위를 보완한다.

## 6. 미확인·미수행

공식 모델 제출 마감 시각, 문제지Ⅳ-2(2·3항 및 일반6h 적용), 모델 첨부 크기·개수(P1 15분할 전제), 하루3회 리셋 기준 모두 사용자 확인 대기다. 공고문 PDF 다운로드·문의 작성·포털 작업 없음. 새로운 P1/P2 학습은 이번 문서·QA 범위의 새 fit0 조건에 따라 시작하지 않았다.

