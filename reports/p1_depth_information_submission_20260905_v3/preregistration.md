# P1 INFO_ONLY 완성정책 공식 비교 — 추론 전 계약

사용자의 오늘 공식 비교 승인 범위에서 root가 P1-A를 1회 정보확인용 후보로 선정했다. 기존 실험의 내부 NO_GO를 개선 PASS로 바꾸는 행동이 아니다. 학습·threshold 탐색은 0이며 frozen A balanced/0.2/추가 decoder OFF를 그대로 사용한다.

- 질문: 배포 train2024–25와 test2026 간 연도키에 무관한 수심 계약 및 최종60일 inner 선택을 갖춘 **완성정책**이 현재 clean control과 실제 공식 모집단에서 어떻게 다른가?
- 수심 단독 인과실험이 아니다. 기존 공식 control은 Q4-inner B_union(O0.2/B0.3), 새 후보는 final-inner balanced0.2다. 두 변경을 공식 점수 하나로 분해하지 않는다.
- 내부 pooled F1 0.848961444, 기준선0.851174240, delta−0.002212796. Q4−0.009086; descriptive90% paired block CI는0을 포함한다. 개선 확률·공식 예상점수는 미산정이다.
- frozen source run은 `p1_depth_contract_repair_20260905_v2`. 12 historical+2final-inner+2full 학습 완료/독립24검사 PASS. 기존config/runner/result/model/lock을 수정하지 않는다.
- 사전검증: full2모델·runner/config/result/QA/recipe 및 source dependency hashes, training runtime package version 일치. 새 inference closure를복제·봉인한다.
- 읽기허용: `P1_DATA_DIR/test.csv`의7raw열 및 `sample_submission.csv`의4key열만. sample label/hidden/external/옛CSV값0. G-ORS rawdepth결측은 계약상정상이고 출력의binary/finite와구분한다.
- 생성은 별도v3경로169011행 CSV1개. sample key/order/one-to-one/finite/binary/UTF-8/hash, 기존공식receipt메타의P1 SHA중복을검사한다. 별도프로세스에서전체CSV byte-exact replay한다. 동일SHA면중복제출금지.
- 원시행/공식입력값은로그·보고서에출력하지않고 집계/hash만기록한다. 브라우저할당량/마감/접수·채점확인은root담당이다. 최종모델잠금·Git업로드는본adapter범위아니다.
- 결과를본후threshold/계수/정점별정책을추가변경하지않는다. 공식비교결과는해당CSV SHA에만귀속한다.

실행: `P1_DATA_DIR`를배포P1디렉터리로지정하고 새runner의`--preflight`, `--predict`, 별도프로세스`--verify`를순서대로사용한다. 기존exactly-once학습runner는다시실행하지않는다.
