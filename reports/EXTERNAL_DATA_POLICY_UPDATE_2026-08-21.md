# 외부 공개 데이터 정책 갱신

작성일: 2026-08-21 KST  
상태: 외부 공개 데이터 사용 허용 확인 / 출처·권리·누출 게이트 유지 / 제출·업로드 0건

## 결론

대회 공식 공개 FAQ API의 9번 답변은 외부 공개 데이터 활용을 허용하고 출처 명시를 의무화한다. 참가자 전용 FAQ의 대학부 답변도 외부 데이터 사용이 가능하다고 명시한다. 따라서 기존의 `운영진 별도 서면 승인 전 외부값 0건` 정책은 폐기한다.

외부 데이터를 쓰는 모든 실험에는 다음 조건을 적용한다.

1. 공개 접근 가능한 출처와 사용 가능한 라이선스
2. URL 또는 DOI, 버전, 회수일, 파일 SHA-256, 가공 이력
3. fold 시점에서 재현 가능한 데이터만 사용하는 시간 누출 감사
4. 외부자료를 제거한 동일 split comparator
5. 숨은 정답 또는 원정답을 직접 복원하는 자료의 차단

공식 허용과 제3자 저작권은 별개다. KIMST 공고의 재현성 및 제3자 권리 책임도 그대로 적용한다.

## 공식 근거

- 대회 공식 FAQ API: https://oceanaidata.org/api/faqs
  - FAQ id 9: `외부 공개 데이터 활용도 허용되며, 이 경우 출처를 반드시 명시해야 합니다.`
- 참가자 전용 공지·FAQ: https://oceanaidata.org/app/notices
  - 대학부 외부 데이터 질문: 사용 가능, 평가는 주최측 보유 데이터로 진행
- KIMST 공고 제2026000111호: https://www.kimst.re.kr/u/news/notice_01/board.do?bno=153421765145711&searchDiv=&searchKeyword=&type=view
  - 코드·데이터 재현성 검증, 실행 불가 시 실격 가능성, 제3자 권리 책임

공식 FAQ 응답의 로컬 증거 스냅샷은 `configs/external_data/official_faq_external_data_2026-08-21.json`, 허용 출처 whitelist receipt는 `configs/external_data/official_faq_permission.json`에 보존한다.

## 누출 분류

### 사용 가능 후보

- 라이선스가 확인된 2023년 이전 관측·재분석을 이용한 사전학습
- ERA5, CMEMS, 위성 SST, 조석·태풍처럼 목표 센서의 숨은 정답과 독립적인 공개 공변량
- 외부 자료로 학습한 표현을 대회 train에서 다시 보정하는 모델

### 추가 검토 후보

- 공개 사전학습 checkpoint: 라이선스뿐 아니라 학습 corpus와 평가기간 중첩 provenance가 필요
- 자료 페이지에 개별 라이선스가 없는 S-ORS·G-ORS 자료: 권리자 허가가 필요
- 관측을 동화한 재분석의 목표 변수: 동화 source와 exact-target 복원 가능성을 문서화해야 함

### 사용 금지

- P1 2026 test의 합성 주입 전 clean 수온을 직접 제공하거나 exact-match할 수 있는 KORS 원자료
- P2 2025-09-01~10-31 S-ORS layer 2·3·4 실제 수온·염분 또는 mirror
- P3 익명 case를 실제 시각에 대응시키거나 실제 +3~24시간 파고를 가져오는 자료

이는 외부 데이터 자체를 금지하는 규정이 아니라, hidden answer를 외부에서 회수하지 않기 위한 재현성·평가 무결성 경계다.

## 첫 실행 whitelist

초기 실행은 권리와 기간이 명확한 네 출처만 연다.

| source_id | 자료 | 기간 상한 | 우선 문제 |
|---|---|---:|---|
| `i_ors_ctd_2014_2023` | I-ORS 10분 CTD, CC BY 4.0 | 2023-12-31 | P1/P2 fallback |
| `i_ors_ocean_atmos_2004_2023` | I-ORS 해양·기상, CC BY 4.0 | 2023-12-31 | P1/P3 support |
| `kma_ocean_buoy_pre2024` | KMA 해양기상부이, KOGL Type 1 | 2023-12-31 | P3 pretraining |
| `era5_pre2024` | ERA5 시간별 재분석, CC BY | 2023-12-31 | P3 pretraining |
| `nasa_power_kors_meteorology` | NASA POWER 시간별 기상 재분석 | P1 2026-06-30 / P2 2025-12-31 | P2 mixing covariate, P1 natural-variability covariate |

S-ORS historical CTD는 성능 기대가 가장 높지만 개별 공개 라이선스가 확인되지 않아 whitelist에 넣지 않는다.

## 제출 운영 정정

2026-08-20 갱신된 공식 인터페이스는 예측 답안을 문제별 하루 3회 허용한다. 2026-08-12 참가자 전용 수정 공지는 실제 모델을 코드와 학습 가중치로 9월 7일까지 제출하도록 하며, 모델 제출 즉시 해당 문제의 추가 예측 답안 업로드가 잠긴다고 명시한다. 따라서 모델 제출은 마지막으로 미룬다.

답안 채점 시작 문구는 `8월 25일부터`와 `20일 제출분은 25일 함께 채점`이 함께 존재하므로, 업로드 가능 여부와 채점 시점은 구분해 기록한다.

## 다음 실행

1. P1·P2·P3별 외부 출처 metadata/API 및 라이선스 precheck
2. 다운로드 크기와 변수 support가 통과한 최우선 한 family만 격리 다운로드
3. 외부자료 label-free 품질·domain-shift 검사
4. 기존 frozen 모델과 동일한 로컬 split에서 단일 ablation
5. 승격 gate 실패 시 해당 source/model family 종료
