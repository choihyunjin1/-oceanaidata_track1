# P1 INFO_ONLY 공식 비교 후보 — 전체 CSV 재생 PASS

## 결론

**새 학습 없이 고정 P1-A 완성정책의 제출용 CSV가 준비됐다.** 2026-09-05 22:01:19 KST 최초 생성과22:02:07 별도PID 재생 결과가 byte-exact다. 내부 개선 PASS로 바꾼 것이 아니며, 공식 업로드·채점은root가수행하고아직이보고서에서완료를주장하지않는다.

- 답안: `artifacts/p1_depth_information_submission_20260905_v3/05_answer/P1_submission.csv`
- SHA-256: `b41c44f339a2e2fb81943da910892f53e28e08b4e13bf39050da6a5298728363`
- 169,011행, 6,929,481bytes, 정확한열 `station,year,layer,time,label`.
- 제목: `P1 year-safe depth complete policy INFO_ONLY 20260905`
- 설명: 배포 train-only balanced/0.2와 year-safe 수심·final-inner선택을그대로고정한정보확인용완성정책이다. 내부개발평가 A절차는ΔF1−0.002213이었으며 수심단독의인과검증은아니다.

## 선택 근거와 해석 경계

배포 README는train2024–25와test2026을명시한다. [기존control보고서](../p1_clean_control_fulltrain_20260905_v1/report-source.md)는공식169,011행의 nominal-depth가연도lookup때문에모두missing임을기록했다. 이번고정 year-safe계약은관측수심이있는행에서그정보를보존하는완성정책의수송성을확인한다. 이번공식입력집계상rawdepth결측16,368행과nominal-depth결측이정확히같고, 관측depth가있는데nominal-depth만결측인행은0이다. G-ORS의실제depth결측은배포계약상정상이며임의복원하지않는다.

그러나기존공식control은Q4-inner B_union(O0.2/B0.3), 이번후보는final60일inner의balanced0.2다. 공식점수변화를수심만의효과로분해할수없다. [A내부결과](../p1_depth_contract_repair_20260905_v2/report-source.md)의pooledF1 0.848961444/기준선0.851174240/Δ−0.002212796은fold별earlier-inner선택절차의개발평가이며, **최종balanced0.2 자체의새독립holdout성적이아니다**. Q4악화와반복노출검증이라는한계도유지한다. 예상공식점수는미산정이다.

## 확인한 것

- A runner/config/result/독립QA/recipe SHA, dependency·recipe·full2model SHA와학습당시runtimepackageversion일치.
- exact balanced/threshold0.2/추가decoderOFF/current-observation-round2m 유지. 기존학습runner/config/result/checkpoint/lock변경0, 새학습·calibration0.
- 새 adapter synthetic12검사PASS/RuffPASS. 최초테스트1건의Windows기본인코딩문제는테스트UTF-8읽기만고쳐해결했다. 모델·데이터실행실패가아니다.
- 실제CSV의UTF-8/schema/keyset/순서/고유성/정수0·1/유한값검사PASS. 양성6,138행은집계진단이며prevalence를맞추지않았다.
- 기존9월5일공식receipt의P1 SHA와다름. 현재브라우저의전체접수·할당량·마감확인은root의업로드전단계이다.
- PID32232의최초추론17.984초, 별도PID34112의전체CSV재생17.985초. 전체bytes와SHA동일. [추론영수증](inference-qa.json), [재생영수증](replay-qa.json), [사전seal](preflight.json).
- test고유행169,011이며두프로세스누적로드338,022행; sample은각프로세스key4열169,011행만읽었다. samplelabel/hidden/external/옛CSV값0. 접근0은명시적usecols와호출경로의범위이며OS전체감시를뜻하지않는다.

## 학습에서 추론까지의 시간·재현

원배포train776,706행에서 A의12historical+2final-inner+2fullfits까지765.219초가이미완료됐다. 이번추론을더한실측합은783.203초(약13분3초)다. full2fit/save/probe자체는77.516초다. 새raw-to-model학습을반복하지않았으며6시간전체재현가능성을뒷받침하는실측시간증거이지, **인터넷차단clean machine의처음부터재학습검증PASS는아니다**.

이번artifact의02_code는실제로로드된프로젝트및venvPython모듈1,067개의연구용snapshot/hashclosure다. 최종최소배포ZIP이나독립환경설치패키지가아니다. 이번--verify는현재repo/동일환경의별도프로세스이며복제02_code에서의실행으로과장하지않는다. 공식최종모델잠금/패키지갱신/Git/업로드는본lane에서0이다.

Data Analytics validate-data 기준은기술적QA·내부성능·공식비교·재학습재현의서로다른주장을분리하는데적용했다.

## 공식 비교 후 결론 — 2026-09-05 22:10 KST

Root가 위 SHA `b41c44f339a2e2fb81943da910892f53e28e08b4e13bf39050da6a5298728363` 파일을 정확히 한 번 업로드하고, 문제 카드의 채점 완료를 확인했다. 공식 F1은 **0.767370 / 27.150461점**이다. 직전 clean control의 F1 0.790733 / 27.771400점보다 **F1 −0.023363 / −0.620939점** 낮았다. 확인 당시 오늘 P1 잔여 기회는 1/3이었다.

따라서 이번 INFO_ONLY **완성정책의 공식 비교는 개선 실패**다. 내부 성능 실패와 별개로 공식 비교 근거를 확보했지만, 수심 계약과 최종 inner 선택이 함께 달라졌으므로 수심 변경 하나가 점수 하락의 원인이라고 단정하지 않는다. 현재 비교에서는 직전 clean control을 유지하며, 이 점수로 계수·임계값을 역산하거나 추가 튜닝·학습·업로드를 하지 않는다.

공식 접수·채점의 canonical 기록은 root가 작성하는 [공식 영수증](../conditional_validation_and_information_submission_20260905_v3/official-receipt.json)에 연결한다. 본 추가 기록은 root의 직접 확인 메시지를 반영한 것이며, 이 lane에서 브라우저를 다시 조작하거나 기존 봉인 파일을 변경하지 않았다. 위의 “업로드 미완료” 문장은 후보 준비 시점의 역사적 상태다.
