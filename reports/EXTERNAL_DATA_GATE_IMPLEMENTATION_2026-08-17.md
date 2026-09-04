# 3개 문제 공통 외부데이터 Gate 0 구현 결과

작성일: 2026-08-17 KST  
상태: **구현 완료 / 외부값 사용 차단 / 다운로드 0건 / 학습 0회 / 제출 0회**

## 결론

현재 대회 공지에서 외부 데이터·외부 사전학습 가중치의 허용 범위가 명시적으로 확인되지 않았다. 따라서 외부값을 먼저 받지 않고, 주최 측 서면 승인과 저작권 허가가 모두 확인된 자료만 별도 격리 영역에서 사용할 수 있도록 fail-closed 게이트를 구현했다.

승인 전에는 P1·P2·P3의 어떤 학습 진입점에도 외부 파일을 연결하지 않는다. 특히 다음 값은 승인 여부와 관계없이 사용 후보에서 제외한다.

- P1: 2024~2026 KORS 원자료·실시간값
- P2: 2025-09-01~2025-10-31 S-ORS layer 2·3·4 정답 또는 이를 복원할 수 있는 동기간 자료
- P3: 2025-07-04~2026-06-30 평가기간 자료, 익명 case의 절대시각을 대응시키는 자료

## 구현된 차단 순서

`src/ocean_external/policy.py`는 다음 순서를 강제한다.

1. 주최 측 승인 receipt 파일의 존재·상태·SHA256 확인
2. 승인문에 문제·출처·사용 목적이 각각 명시됐는지 확인
3. 출처 카탈로그의 라이선스와 기간 확인
4. S-ORS처럼 공개 라이선스가 없는 경우 별도 권리자 허가 증빙·SHA256 확인
5. 승인 cutoff와 출처 cutoff 중 더 이른 시각을 적용
6. manifest의 관측 종료시각이 cutoff를 넘지 않는지 확인
7. 위 검사가 모두 끝난 뒤에만 후보 파일을 열어 SHA256 확인

즉 승인·권리·기간 검사가 실패하면 후보 데이터 파일의 존재 여부조차 조회하지 않는다. 이 모듈에는 다운로드 기능과 네트워크 호출이 없다.

## 등록된 출처 메타데이터

카탈로그: `configs/external_data/catalog.toml`  
SHA256: `78e5970a09b584b8ad12e77d9a06693fb35ff73fc3c438d907f2d0641b367cb0`

- 총 9개 메타데이터 항목
- 개방 권리 확인 4개: I-ORS 2종, KMA 해양부이, ERA5
- 추가 권리·provenance 검토 필요 5개: S-ORS, NOAA WW3, Argo, MOMENT, Chronos
- 실제 값 접근 표시: 0개

카탈로그 등록은 사용 허가를 뜻하지 않는다. 모든 항목에 대하여 주최 측 서면 승인 receipt가 먼저 필요하다.

## 사전등록된 첫 실험

### P2

`configs/experiments/p2_external_depth_query_v1.json`

- 상태: `blocked_pending_organizer_approval`
- 주 가설: pre-2024 연직 프로파일을 활용한 continuous-depth query 사전학습
- S-ORS 2015~2023은 주최 승인과 별도로 KIOST 권리자 허가 필요
- I-ORS는 수괴 차이 때문에 fallback 후보로만 유지
- 로컬 RMSE 0.010 ℃ 이상 개선과 세 target layer 비열화를 동시에 요구

### P3

`configs/experiments/p3_external_storm_pretrain_v1.json`

- 상태: `blocked_pending_organizer_approval`
- 주 가설: pre-2024 KMA 부이·ERA5 기반 storm-life-cycle 사전학습
- 48시간 context, 6개 lead, `hs >= 1.5 m`, 같은 정점 78시간 분리를 외부 case builder에도 동일 적용
- 평가기간 외부값과 test 절대시각 대응을 영구 금지
- 로컬 RMSE 0.010 m 이상 개선, 정점·lead별 악화 제한, event-block bootstrap 통과를 요구

두 파일 모두 `execution_authorized=false`이며 다운로드·학습·평가·제출이 전부 false로 고정됐다. 승인 전 실행용 학습 진입점은 만들지 않았다.

## 검증 결과

다음 실패 조건을 자동시험으로 고정했다.

- 승인 receipt 부재
- 주최 승인 증빙 SHA 불일치
- 승인되지 않은 출처·문제·목적
- S-ORS 권리자 허가 부재
- 승인 cutoff 초과
- 권리/provenance 검토 중인 출처
- 후보 파일 SHA 불일치
- P2/P3 사전등록의 실행 봉인 해제

카탈로그 감사 명령:

```powershell
$env:PYTHONPATH='src'
.\.venv-p1\Scripts\python.exe scripts\validate_external_preflight.py --catalog-only
```

승인 후 한 출처를 격리 검증하는 형식:

```powershell
$env:PYTHONPATH='src'
.\.venv-p1\Scripts\python.exe scripts\validate_external_preflight.py `
  --catalog configs\external_data\catalog.toml `
  --approval approvals\organizer_external_data.json `
  --manifest external_data\quarantine\SOURCE\manifest.json `
  --problem P3 `
  --purpose pretraining `
  --source-id kma_ocean_buoy_pre2024
```

`approvals/`와 `external_data/`는 Git ignore 대상이다. 원자료와 승인 원문은 저장소에 넣지 않고, 공유 가능한 보고서에는 논리명과 SHA256만 기록한다.

## 다음 게이트

다음 실행 조건은 주최 측의 명시적 서면 답변이다. 문의 초안은 `reports/EXTERNAL_DATA_APPROVAL_ALL_PROBLEMS_DRAFT_2026-08-17.md`에 준비돼 있으나 자동 발송하지 않았다. 발송 전 팀명·대표자명·접수번호·회신 이메일을 채우고 사용자가 정확한 수신처와 본문을 승인해야 한다.

승인이 오면 허용된 출처만 소량 metadata/sample 검증 → 격리 manifest 생성 → preflight 통과 → domain-shift 감사 순으로 진행한다. 승인 거절 또는 모호한 답변이면 외부값 실험은 종료하고 내부 데이터 기반 대체 구조로 복귀한다.
