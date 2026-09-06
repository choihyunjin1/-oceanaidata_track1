# 2026-09-06 연구·재현 체크포인트

> 09-07 갱신: 현재 선택본은 P1 원형 28.909341 / P2 L120 28.077280 / P3 numeric 23.741446이다. [최종 패키지 안내](FINAL_RELEASE_20260907.md)와 [실제 검증 보고서](../reports/final_release_20260907_v1/report-source.md)가 우선한다. 아래 순서별 기록은 보존된 연구 이력이다.

## 게시 이후 사용자 승인 실행 — 06:39 이후

**P1 bracket 공식 채점 완료: F1 0.785944 / 27.644124점, 기준 대비 +0.008195 / +0.217805점.** 정확한 SHA9031과 [영수증](../reports/p1_bracket_official_submission_20260906_v1/receipt.json)을 연결했다. 최종 모델 잠금은 하지 않았다.

[실행 결과와 공백](../reports/remaining_work_completion_20260906_v1/report-source.md)을 최신 근거로 읽는다. P1 portable cold4fit/753.030초·동일9031 답안·독립47QA 완료. P1 범위/셀 정책은 [독립190QA 비교](../reports/p1_trainfit_postpolicy_20260906_v1/report-source.md)에서 주평가 동률/악화로 비승격이다.

P2 새 수치정정48fit 완료·독립1117QA 후 별도full4fit/167.125초·26,061행 생성/replay/ZIP QA를 완료했다. [07:28 공식대조](../reports/p2_copula_official_submission_20260906_v1/receipt.json) CPU C3 .489080→copula .475174℃로 개선했지만 보존CUDA .455143℃는 못넘었다. 기존P2기준유지.

P3 numeric/hmax 비교는 각각15/10 backbone fit와 독립149/134QA·전체replay를 완료했다. hmax 단독의 새 패키지도 원자료부터12backbone+5router→새PID QA→1200답안→새PIDexact replay를1611.919초에 완료했다. 저장ZIP추출·새PID6.088초exact/28QA도 통과했다. [08:08 공식 채점](../reports/p3_hmax_official_submission_20260906_v1/receipt.json)은 .608184m으로 기존 .607183m을 못넘었다. 기존P3기준유지. **이번 확정 순서 전체는 완료**, 모든 옛 설계/HPO를 실행한 것은 아니다. 과거 기술실패는 보존하며 후속작업은 새 Git 게시를 하지 않았다.

## 아래는 직전 Git 게시 시점의 보존 기록

결론: 세 문제 모두 코드로 다시 학습해 공식 채점된 기준 답안을 재현한 로컬 패키지가 있다. 추가로 P1 bracket 후보의 4-fit 학습·내부 검증·답안 QA를 완료했지만 **새 후보는 공식 미채점**이다. 이번 작업은 연구 자산의 Git 기록·공유이며 대회 업로드나 최종 모델 지정이 아니다.

## 가장 먼저 읽을 문서

- [세 문제 기준 패키지: 어떤 파일을 어디에서 실행하는가](ocean_v2_codex/PORTABLE_PACKAGE_HANDOFF_20260906.md)
- [새 P1 후보: 파일·SHA·학습/추론·실행 제한](ocean_v2_codex/P1_BRACKET_CANDIDATE_HANDOFF_20260906.md)
- [기준 모델 재생성 검사](ocean_v2_codex/BASELINE_REGEN_CHECK.md), [실제 cleanroom 결과와 한계](ocean_v2_codex/CLEANROOM_RESULT.md)
- [평가 계약 변경 이력](ocean_v2_codex/CONTRACT.md), [통합 실측 기록](../reports/portable_cleanroom_20260906_v1/report-source.md)

최초 제안, 사전등록, 실행 중 보고서, 최종 receipt는 서로 다른 시점의 기록이다. 오래된 `NOT_RUN`이나 승인 문구를 현재 상태로 복사하지 않는다. 각 실험의 terminal 결과와 위 최신 안내를 함께 확인한다.

## 현재 파일과 성적의 귀속

| 문제/후보 | 답안 SHA256 | 행 수 | 확인된 성적·상태 |
|---|---|---:|---|
| P1 재생성 기준 | `5971e145f1ac38b8ee3e34cfd302973ba7a64b8873db11c354d3331221fdb28a` | 169,011 | 공식 F1 0.777749 / 27.426319점 |
| P1 새 bracket | `9031c84ea72dfa4294406dd995525e89e8975a76983e9f9d7a7b2ba74dbad93a` | 169,011 | 로컬 QA 완료, 공식 미채점 |
| P2 C3 기준 | `46d194a1ef40a1deaebd084916644d9359433d2e6ce7d5c0b53d9f515bbec071` | 26,061 | 공식 RMSE 0.455143℃ / 27.622418점 |
| P3 clean 기준 | `6bfa23d25f944df4711c11d1fce82978a96df08b58fdc57f666ac792a7da96b7` | 1,200 | 공식 RMSE 0.607183m / 23.696500점 |

근거: [P1 공식 영수증](../reports/p1_regenerated_baseline_official_submission_20260906_v1/receipt.json), [P2·P3 공식 영수증](../reports/official_score_repair_submissions_20260905_v1/receipt.json). 다른 SHA의 과거 최고점을 승계하지 않는다. 잔여 제출 횟수·마감은 과거 영수증에서 추정하지 않고 제출 직전 포털에서 확인한다.

## 무엇을 얻었고 무엇이 아직 안 되었나

### P1 — 구간 모양 특징과 현재 코어 최적화

- 양측 역사 평가: 새 15 fit + 기존 3 fit 재사용. 주평가 F1 0.737031619 → 0.757196860, Δ +0.020165240, CI90 [+0.005014770, +0.037606590]. 최악 fold Δ −0.021024011, 최악 정점·층 Δ −0.084545228을 별도 기록했다. [실험 결과](../reports/p1_tuning_twosided_20260906_v1/report-source.md)
- 미등록 연도 수심 특징의 0-fit 스트레스: 전체 27개 확률 replay와 67/67 QA 통과. 주평가 Δ +0.013476241이지만 최악 셀 Δ −0.096710815로 위험이 남는다. 중단된 v1을 덮지 않고 v2로 기록했다. [스트레스 결과](../reports/p1_tuning_depth_stress_20260906_v2/report-source.md)
- 새 후보는 O80 유지, B에 bracket 27개 특징만 추가한 B107이다. CPU 4 threads, GPU 0, 고정 700 trees로 inner 2 + full 2 = 4 fit. final-inner 재선택 결과 **B107 단독 / threshold 0.1**이다. 임계값을 리더보드나 바깥 평가에서 가져오지 않았다.
- final-inner F1 0.894653514 → 0.900204155 (Δ +0.005550642), TP +75 / FP +41 / FN −75. 이는 **선택에 사용한 구간**이며 독립 확인 성능이나 공식 예상 점수가 아니다. 공식 입력에서 바뀐 1,417행이 모두 정답 수정이라는 뜻도 아니다.
- 학습 254.907초, 전체 workflow 589.317초. 합성 테스트 17 PASS, Ruff PASS, 별도 학습 QA 49/49, 답안 QA 26/26, 새 PID replay exact. [최종 후보 결과](../reports/p1_bracket_candidate_20260906_v1/report-source.md)
- 물리 범위 규칙이나 셀별 O/B 정책은 이 후보에 추가하지 않았다. 학습에서 FP 0이라는 관찰만으로 공식 FP 0을 보장하지 않는다.

### P2 — 재현 기준은 확보, copula는 기술 실패와 성능 실패를 분리

- 기준 C3: 빈 모델에서 3 seed × 60 epochs 학습, 학습→추론 80.203초, 새 PID replay SHA 일치, QA 71 PASS. 이는 새 PC·새 환경 설치나 OS 방화벽 격리 인증을 뜻하지 않는다.
- T5 결측 전문가: 첫 seed 이득이 3 seed에서 재현되지 않았다. pooled RMSE Δ +0.001251499℃, outage Δ +0.030801703℃로 악화했다. [음성 결과](../reports/ocean_v2/p2_t5_expert_negative_result.md)를 읽고 동일 재시도를 반복하지 않는다.
- crossfit copula: 88 계획 중 40 fit 실행 후 비-outage 예측의 3.55e−15℃ 차이를 exact assertion이 거부했다. 상태는 `TERMINAL_TECHNICAL_FAILURE / NOT_ESTIMATED_INCOMPLETE`이며 후보 성능 NO_GO가 아니다. 소요 1,809.485초, QA 321 PASS. B4/B8 outage 평가행 부재는 측정 불가이지 오차 0이 아니다. [실패 보고서](../reports/p2_crossfit_copula_forward_20260906_v1/report-source.md)
- [수치 정정 설계](../reports/portable_cleanroom_20260906_v1/P2/copula-numeric-amendment-design.md)는 설계 상태다. 기존 봉인·실패 기록을 수정하거나 남은 48 fit을 이번 Git 작업에서 실행하지 않는다.

### P3 — 재현 패키지는 확보, numeric lead는 아직 성능 미평가

- pristine run_b: 빈 모델에서 8 backbone + 3 router를 중단 없이 학습·추론, 1,256.163초, 42/42 QA. 별도 recovery는 성공 모델을 재사용한 경로이지 두 번째 완전 cold start가 아니다. 두 경로의 예측·답안은 exact, CBM 파일 bytes는 다르다.
- 최신 ZIP은 `archives_v3`의 cold-start 코드와 saved-model replay 용도를 구분한다. 저장 모델 ZIP을 실제 새 폴더에 풀고 0 fit / 3.668초로 답안 SHA를 재현했다. [패키지 안내](ocean_v2_codex/PORTABLE_PACKAGE_HANDOFF_20260906.md)
- numeric lead: 기준 2 fit 후 전체 예상 349.059분이 90분 예산을 초과해 성능 공개 전 중단. 후보 fit 0, `NOT_EVALUATED_RESOURCE_STOP`이다. 같은 CPU 계획을 반복하거나 구조의 성능 실패로 분류하지 않는다. [중단 기록](../reports/p3_numeric_lead_forward_20260906_v1/report-source.md)
- hmax 제거는 분포 관찰만으로 안전한 개선이라고 단정하지 않는다. 별도 완료된 [wind-only dropout 연구](../reports/p3_wind_only_dropout_20260905_v4/report-source.md)와 현재 기준 패키지를 혼동하지 않는다.

## 패키징 경계와 다음 순서

1. 새 P1 후보의 공식 비교는 별도 업로드 작업에서 frozen SHA 그대로 수행한다. 이 체크포인트는 업로드를 실행하지 않는다.
2. 새 P1 runner는 원 학습 시작부터 3,600초 제한이 있다. 나중에 완료 폴더에서 추론만 재호출하는 영구 패키지로 오인하지 않는다. 현재는 안내대로 **새 빈 폴더에 전체 workflow를 재생성**하며, 기존 clock/lock을 우회하지 않는다. 장기 saved-model 추론 어댑터가 필요하면 별도 코드·검증으로 만든다. 기준 portable ZIP은 fallback으로 보존한다.
3. P2는 0-fit 수치 정정 검증 설계를 먼저 검토하고, P3는 자원 예산을 확인한 뒤 다음 실험을 정한다. Git 게시 때문에 학습을 재시작하지 않는다.

## Git 게시 범위

포함: Sep 5–6 실험 코드·설정·테스트, portable **코드/템플릿**, 재생성·실패·공식 접수의 작은 집계 JSON, 연구 보고서와 안내. 기존 미푸시 평가 계약 commit도 이력 그대로 보존한다. 원 코드·봉인된 runner는 게시를 위한 정리 과정에서 바꾸지 않는다.

제외·로컬 보존: 배포 데이터, 외부/공식 원본, hidden truth, 모델·checkpoint, 답안 CSV, 예측 배열, 캐시, lock, 로그, ZIP, credentials. 저장 출력이 있는 `reports/ocean_forward_support_20260906_v1/review.ipynb`와 실행 원본 `reports/p1_bracket_candidate_20260906_v1/preflight-tests.xml`도 게시 대상에서 제외한다. 이들을 참조하는 과거 보고서는 로컬 증거의 위치를 기록한 것이다. GitHub clone만으로 로컬 모델·답안·ZIP이 내려오지는 않는다.

이번 게시 검증의 실제 범위와 결과는 [게시 검사 영수증](../reports/research_checkpoint_publish_20260906_v1/validation.json)에 남긴다. 기존 실험 QA 재사용과 새 합성 테스트/정적 검사를 구분하며 전체 연구를 다시 학습했다고 하지 않는다.

봉인된 JSON은 줄바꿈까지 해시의 일부다. `.gitattributes`의 기존 9월 5일 보존 규칙을 9월 6일 JSON·portable JSON 계약으로 확장해 Git의 자동 줄바꿈 변환을 막았다. 원본 증거를 재포맷하지 않았으며, 기존 CRLF·문서 끝 공백/빈 줄 경고는 검사 영수증에 명시한다.
