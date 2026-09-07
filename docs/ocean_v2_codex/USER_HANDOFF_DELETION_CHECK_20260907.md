# 남은 슬롯·삭제 대상 읽기 전용 대조 — 2026-09-07

## 결론

P3 no-shrink의 **test-hs0 재가중 악화는 독립 재현**됐다. 제출 우선순위는 낮지만 자동 탈락·영구 보류가 아니다. 공식 결과는 아직 없고, 정보 가치 목적의 제출 가능성은 남는다. 이번 작업은 업로드·삭제·최종 지정·새 fit·commit·push 모두 0이다.

Fable Tier 목록 39건(외부자료 P3 13, LB 계수 P3 6, anchor/LB 계보 P2 20)은 제출관리 **화면 시각·지표·점수와 모두 대조**했다. 그러나 화면은 파일 SHA를 공개하지 않는다. 아래 SHA는 로컬 receipt/manifest 값이지 서버 파일을 내려받아 검증한 값이 아니다. 특히 P3 08-29의 3건은 업로드 SHA 미대조, P2 같은 시각 2건은 파일-점수 연결이 추정이다. 이 표는 자동 삭제 명령이 아니다.

**최종 재현 대상이 최고 Public/최근/선택형 중 무엇인지는 미확인. 채점 완료 답안의 삭제 성공 가능성·복구 가능성·순위 재계산 효과도 미확인이다.** 삭제는 사용자만 하며, 삭제로 규정 위반 이력이 해소되거나 적격성이 인증된다고 보장하지 않는다.

## 1. 공식 원문 확인

2026-09-07 이번 작업에서 로그인된 Chrome의 [공지사항](https://oceanaidata.org/app/notices) 모달과 [제출관리](https://oceanaidata.org/app/submissions)를 읽기 전용 확인했다. 공지별 고유 URL은 모달에 노출되지 않았다. 원문 PDF 다운로드·문의 작성·삭제 버튼 클릭은 하지 않았다.

| 원문 | 확인한 내용 | 확인되지 않은 내용 |
|---|---|---|
| 08-12 16:25 `[제출 안내] 데이터분석 대학부 결과물 제출·채점 방식 안내 (수정)` | 코드·가중치 제출일 09-07. 모델 제출 후 답안 업로드 잠김. 업로드 답안 재현 및 정상 학습 검증 | 여러 답안 중 정확히 어떤 파일이 재현 대상인지; 마감 시각 |
| 08-31 13:58 `[안내] 데이터분석 대학부 외부 데이터 사용 관련 안내` | 배포 외 관측·재분석·예보 사용 금지, KIOST 원자료는 정답 접근과 같음, 오프라인 재현 | 과거 비적격 업로드를 삭제했을 때 처리 방식 |
| 09-02 10:41 `[안내] 리더보드 반환 점수의 사용에 관하여` | 반환 점수 역산으로 계수·임계값 등 적합 금지. 정상적인 후보 비교와 구분. **최종 순위는 Private 종합점수** | Public 최고 파일이 자동 최종 대상이라는 규칙 |
| 08-04 `[FAQ] 자주 묻는 질문` | 세 문제 점수 합산, 배포 데이터/오프라인 재현 원칙 | 최고 Public/최근/선택형 중 재현 대상 선택 방식 |
| 제출관리 화면 | ‘채점 전까지 수정·삭제’ 안내. 채점 완료 카드에 수정 disabled, 삭제 버튼은 표시 | 버튼을 누르지 않았으므로 채점 완료 삭제 가능 여부는 미확인 |

따라서 Fable 정정의 **‘삭제 비용은 표시 총점 하락뿐’도 확인된 사실로 채택하지 않는다.** 80.890824는 P1 28.909341 + P2 28.240037 + P3 23.741446의 Public 산술합이며, 삭제 후 표시 보장·적격성 인증·Private 최종 점수가 아니다. 과거 정찰의 `clean`은 일부 항목에서 ‘외부자료 없음’만 뜻하므로, LB 역산 계보까지 적격하다는 뜻으로 재사용하지 않는다.

## 2. P3 재가중 독립 재계산

입력: CPU completion_1 OOF SHA `bede4edff8ce229d66c6209bce4227f85d63b1c065113a39100b12a0ea40ec16`, CPU/GPU 각각 103,602행. 배포 test_context는 `step_minute==0`의 case_id/hs만 200사례 읽었고 hidden truth는 읽지 않았다. 구간 사례 수는 80/40/33/27/20, OOF의 hs0<1.5 행은 양쪽 모두 0. 원시 행은 출력하지 않았다.

| 항목 | CPU 독립 결과 | Fable 대조 |
|---|---:|---|
| 기준 SSE | 48091.9523451907 | 일치 |
| no-shrink SSE | 47546.9931632454 | 일치 |
| 재가중 기준 RMSE | 0.641526145551452 m | 6자리 일치 |
| 재가중 후보 RMSE | 0.6465586730157941 m | 6자리 일치 |
| 후보−기준 | +0.005032527464342018 m | 6자리 일치, 악화 |
| 문자열 정렬 cluster CI90 | [+0.002354344519521928, +0.007736358008010084] m | 6자리 일치 |
| 개선 resample | 1/2000 = 0.0005 | 소수 3자리 표기 0.001 |
| GPU 재가중 Δ | +0.00543614313198959 m | 같은 악화 방향 |
| GPU numeric-tuple cluster CI90 / P | [+0.0027609871409741416,+0.008230562927926404] / 0/2000 | 0/2000은 개선 불가능 확률 0을 뜻하지 않음 |

첫 독립 계산은 `(station, numeric episode_id)` 정렬로 CI [+0.0023655532084391807,+0.00778958383392973], P=0.001을 얻었다. Fable 원 코드의 `station + '_' + str(episode_id)` 문자열 정렬을 그대로 재현하면 위 CI가 된다. **행 오차·가중치 불일치가 아니라, 같은 seed에서 cluster 순서가 달라 유한 부트스트랩 추출이 달라진 것**이다. 두 계산을 모두 보존했다. 해석에 유리한 CI를 선택한 것이 아니다.

증거:

- `reports/p3_numeric_cpudet_noshrink_20260907_v1/test-mix-check.json`: row-wise와 독립 bin-mean 집계 식 일치, 입력 SHA, CPU/GPU 결과. 원 `result.json` 불변.
- 같은 폴더 `test-mix-bootstrap-order.json`: 문자열 순서 재현, 별도 PID 15692, 0.157초.
- `scripts/verify_p3_test_mix_20260907.py`, `scripts/verify_p3_test_mix_cluster_order_20260907.py`: 두 thread 이하, 학습/답안 쓰기 없음. Ruff PASS (`E402`는 thread 환경변수를 수치 라이브러리 import보다 먼저 설정하기 위해 제외).
- `tests/test_p3_test_mix_check_20260907.py`: 집계 식·SSE/정렬 설명·삭제표 39개 고유 화면 키를 검사하는 focused pytest **3 PASS**. 첫 테스트에서 공지 표 3행까지 잘못 세던 문서 필터를 수정한 뒤 통과했다. 모델/평가 수치 수정 없음.

이것은 **사후적 예측 근거이지 공식 결과가 아니다**. hs0 구간 내 조건부 오차가 test로 전이한다는 가정이 있고, 정점·계절·발달 단계 결합 분포는 보정하지 않았다. Bootstrap은 관측한 200사례 비율과 가중치를 고정했으므로 그 추정 불확실성까지 포함하지 않는다. 이 진단으로 계수·예측을 변경하지 않았다.

## 3. 삭제 검토표 — 화면 시각은 모두 KST 2026년

표의 ‘권고’는 계보 위험에 대한 사용자 검토 권고다. 원격 파일 SHA 확인 및 운영진 판정 완료를 의미하지 않는다. `영수증`은 후보와 실제 점수를 연결한 로컬 기록, `manifest+이력`은 별도 문서 연결, `추정`은 확정 업로드 매핑이 아니다. 점수는 이번 화면 표시 그대로이며 역산하지 않았다.

### Tier 1 — P3 외부자료 계보 13건

| 화면 시각 | RMSE / 점수 | 우리 receipt SHA256 | 계보 / Tier | 삭제 권고·대조 근거 |
|---|---|---|---|---|
| 08-28 13:56 | 0.585738 / 24.036866 | `3967333b790c06495dff619b2a8191b9bec18aa56dff1453ee31f77882ce8a50` | ERA5 / 1 | 권고; E28 영수증. 과거 13:57은 기록 시각 차이 |
| 08-28 23:45 | 0.577671 / 24.164901 | `05470f9c5186498f8ce36b8c265348c6064c4e94b1992b836a60cf16f6aecf20` | KMA / 1 | 권고; D28 manifest + official_results |
| 08-28 23:45 | 0.575262 / 24.203126 | `51f16e8920e1694343f5e0d9ec90c91a19b16afed8d8cb9b02ef310d144e7638` | KMA / 1 | 권고; D28 manifest + official_results |
| 08-29 22:20 | 0.577577 / 24.166383 | **미대조** | KMA lead-split 추정 / 1 | 계보상 권고, 업로드 SHA 확인 전 보류; A29 official_evidence에 점수만 있음 |
| 08-29 22:20 | 0.576264 / 24.187236 | **미대조** | KMA lead-split 추정 / 1 | 계보상 권고, 업로드 SHA 확인 전 보류; A29 official_evidence에 점수만 있음 |
| 08-29 22:51 | 0.581747 / 24.100203 | **미대조**; 후보 manifest는 `5f09ca77e139a33b0e0e51c1dff13d5d9c9fb9a297a750c16db4243fe0dd2d86` | KMA geometry 추정 / 1 | 계보상 권고, 파일-점수 연결 미확정; A29는 READY_NOT_UPLOADED 상태 |
| 08-30 21:19 | 0.575233 / 24.203599 | `144f5e1740a338df881b5076b8d0a8764630c5836982a6ff4326e93c2e24219e` | KMA / 1 | 권고; K30 영수증. 실제 화면은 21:19 |
| 08-30 22:56 | 0.579102 / 24.142185 | `a5a16ba207ed1cccf16383e1de7b932417666917b0eb2b9c54a00fdb7ab67351` | KMA leave-S / 1 | 권고; I30P3 영수증 |
| 08-30 22:56 | 0.578951 / 24.144591 | `868d18d7a2d62d49b6d97712e686db7a55bbb16e38cea2659584a0c397275f4f` | KMA leave-I / 1 | 권고; I30P3 영수증 |
| 08-31 05:07 | 0.576589 / 24.182070 | `cce2904c18855e4ae2d884cf43dc4c61b1366ca300e2d210558d362f264ad0e8` | KMA / 1 | 권고; L31P3 영수증 |
| 08-31 18:14 | 0.590956 / 23.954041 | `1bb1a90c149e566497f95fcb9d1bb1aa3895f4fef341afc6b30d6fe6710ca65d` | KMA router / 1 | 권고; R31 영수증 |
| 08-31 21:23 | 0.589840 / 23.971758 | `b1b72f905e36df994f82ef8dc5c425328c5e0b0b4d16ec2a0dbfd8497c48d0c4` | KMA / 1 | 권고; T31 영수증 |
| 09-01 07:52 | 0.576320 / 24.186338 | `4ca8c0208f6ff2e0ab232b85f4194b4dbf4c33c0a86c5bac27209282d5a4f942` | KMA / 1 | 권고; V42 영수증 |

### Tier 2 — P3 LB 적합 계보 6건

| 화면 시각 | RMSE / 점수 | 우리 receipt SHA256 | 계보 / Tier | 삭제 권고·대조 근거 |
|---|---|---|---|---|
| 08-26 22:01 | 0.599072 / 23.825229 | `57a90beb3f81de65fbf67426811eeaf49427951fa277997adb89c75ef259af56` | LB 축 reverse / 2 | 권고; D26 영수증 |
| 08-26 22:01 | 0.606681 / 23.704466 | `c5ac003e5c0827f6d5f3ec0ac396e230fb3e4266f6668c592095c06ed1e94da1` | LB 축 lead12 / 2 | 권고; D26 영수증 |
| 08-26 22:02 | 0.599382 / 23.820314 | `91ead7470f53aa7e09000bdc667a975b3836f16f201b86b1060ba6ea893212ee` | LB 축 lead18/24 / 2 | 권고; D26 영수증 |
| 08-27 22:19 | 0.583892 / 24.066167 | `ad983de66d450520c261e7c9d5a13a9403a7a67a2a5878492b689448e80b73ab` | LB 이차 alpha / 2 | 권고; FH + G27 |
| 08-27 23:33 | 0.584611 / 24.054757 | `fb379cc73884e2c788898984d33328e5025e6c68b82e8dec42736281a55d6816` | LB alpha bracket / 2 | 권고; G27 manifest + HO 제출 이력 |
| 08-27 23:36 | 0.583892 / 24.066168 | `ea65370a5c9291868769ad9e54a54707035dc93a01ffa4772d9fd26342f357aa` | LB alpha 재적합 / 2 | 권고; RF27 manifest + 공식 결과 |

같은 0.583892 두 건은 **점수 마지막 자리와 시각이 다르다**. 서로 바꾸어 연결하면 안 된다.

### Tier 3 — P2 anchor/LB 적합 계보 20건

| 화면 시각 | RMSE / 점수 | 우리 receipt SHA256 | 계보 / Tier | 삭제 권고·대조 근거 |
|---|---|---|---|---|
| 08-26 22:00 | 0.537238 / 26.592326 | `9cc951801cf6b6cdacc2c826126d9c2f72ef34fc67e46c6a21261c7a1ba845ff` | LB global alpha / 3 | 권고; D26 |
| 08-26 22:01 | 0.541917 / 26.533611 | `5507317f45bf06969d7da6c2ebd750bc5805564d1e3955920eab036724fc1ccc` | LB layer2 / 3 | 권고; D26 |
| 08-26 22:01 | 0.536536 / 26.601139 | `98890354fe792c905b44f9467c0651506c7696abd7091f1380e1825669865cff` | LB layer4 / 3 | 권고; D26 |
| 08-27 22:18 | 0.535727 / 26.611283 | `13181dff0e749a1ea6dac7327b4ea34b8a7efd57a2f57170ba0d206f919cf592` | LB 층별 U / 3 | 권고; FH + G27 |
| 08-27 23:22 | 0.507628 / 26.963865 | `65b754c83c520609e7bd7979b35f86366903cfebf07d3e411f8f97ab3b59abd5` | U anchor + OAS10 / 3 | 권고; OAS10 제출정보 + HO |
| 08-27 23:29 | 0.483661 / 27.264587 | `f46dec7944fe4565307b0242fdab5772a684027f1a42a62404d6e01ba13e0ef7` | U anchor + OAS20 / 3 | 권고; OAS20 제출정보 + HO |
| 08-28 03:56 | 0.445147 / 27.747847 | `6e28ddb8d78c0969e5104d7efbe28e1762f51e80d759fceb86cdef52baa29b96` | U anchor + OAS40 / 3 | 권고; O40 영수증 |
| 08-28 19:03 | 0.431252 / 27.922187 | `bd550127cfbab9bcd2df75ad7d3fb65dafdf62568fca628851b0e0ae1dc241d5` | U anchor + OAS50 / 3 | 권고; O50 영수증 |
| 08-28 23:45 | 0.430250 / 27.934759 | `665485e1b47705cf03ae3537ffe31ebc47c309408428d9c3847fd973ffca94f1` | alpha50 anchor / 3 | 권고; D28 |
| 08-29 22:20 | 0.432244 / 27.909741 | `7d06cf01c67541ec77d4e6af7afefca710774bda50f33155573c382d440672d9` | anchor rank1 x2 / 3 | 권고; C30 recent_official_history |
| 08-29 22:51 | 0.430253 / 27.934720 | `dbcef773a9bbff62c66b2b6a0bca1b9279eb664209ff1560c2dc02e83855b019` | anchor layer shrink / 3 | 조건부 권고; A29/C30 파일-점수 **추정** |
| 08-29 22:51 | 0.430209 / 27.935277 | `bf15d705b3dbaaac4265c608b158e2a9b544fb31957bd12d8d2ca41c25faab0f` | anchor + LB vertex / 3 | 조건부 권고; A29/C30 파일-점수 **추정** |
| 08-30 22:05 | 0.442259 / 27.784078 | `f498c6e1d7e22d11d5571b971454f0e375247fc2ff5ae3387bfcb4186460c4a3` | U/alpha50 anchor + copula / 3 | 권고; C30 영수증 + base v1 config. ‘copula 자체 clean’과 이 답안의 anchor 의존은 별개 |
| 08-30 22:55 | 0.430194 / 27.935464 | `99c6925cec605905c80f2924c5655b3dd83ed712c9f27853c58d6d9e0f74e2e2` | bin17 anchor / 3 | 권고; I30P2 |
| 08-30 22:56 | 0.431267 / 27.922001 | `0d213e97b9435862bbc892ac358afdc99b0a8834740915720f96bd420761d557` | bin18 anchor / 3 | 권고; I30P2 |
| 08-31 05:34 | 0.430800 / 27.927863 | `6cfafc36dd6fe9c87455ffc7f9ed33d6217200cd4b321b7c5b912ffd85b11ab3` | bin17 anchor / 3 | 권고; L31P2 |
| 08-31 18:14 | 0.431532 / 27.918675 | `642265be7eec4505ab9f99c97b7efeda71a3b629d3352e2c3f3927b0142efd06` | anchor + HGB / 3 | 권고; R31 |
| 08-31 18:14 | 0.438464 / 27.831700 | `098b5bb25637b026c13e8eec7afc04fbf78cc6cc25f3da4c177323b15f8978dc` | anchor + residual / 3 | 권고; R31 |
| 09-01 07:52 | 0.424976 / 28.000939 | `a6c62a8abf2ec70cad0b251e2006fb3e9aa2536d35be484fb6f0a4a0bbb34384` | anchor + V23 / 3 | 권고; V23 영수증 |
| 09-01 19:12 | 0.424019 / 28.012945 | `331b1635bb036e773ff73487075e803b1308223e905c28b0d1494ea88b4d94c9` | anchor + V52 / 3 | 권고; V52 영수증 |

## 4. 근거 위치

저장소 기준 상대경로이며 Downloads는 `C:/Users/cedis/Downloads/해양 해커톤 제출용/` 아래다. CSV 원시 값·외부 관측·hidden truth를 열지 않고 아래 메타데이터와 화면만 대조했다.

- D26: Downloads `20260826_round_D_preregistered_P1x3_P2x3_P3x3/OFFICIAL_RESULTS_20260826.json`
- G27: Downloads `20260827_round_G_P2x3_P3x3_PUBLIC_QUADRATIC_READY/SET_MANIFEST.json`
- FH: `reports/finite_horizon_submission_decision_20260827_v1/report-source.md`
- HO: `reports/HACKATHON_HANDOFF_2026-08-28.md`
- RF27: Downloads `20260827_P3_REFINED_PUBLIC_OPTIMUM_READY/MANIFEST.json`
- OAS10/OAS20: Downloads `20260827_P2_SEASONAL_OAS_TS10_PROJECTED_READY/제출정보.txt`, `20260827_P2_SEASONAL_OAS_TS20_PROJECTED_READY/제출정보.txt`
- D28: Downloads `20260828_DEADLINE_INFORMATION_PROBES_READY/SET_MANIFEST.json`
- A29: Downloads `20260829_ADAPTIVE_FINAL_PROBES_READY/SET_MANIFEST.json`
- E28: `reports/approved_parallel_execution_20260828_v9/p3_official_submission_receipt_20260828.json`
- O40: `reports/p2_submit_p1_p3_deep_research_20260828_v1/official_score_receipt.json`
- O50: `reports/p2_oas_alpha50_deployment_20260828_v13/official_score_receipt.json`
- C30: `reports/p2_gaussian_copula_v2_exact_frozen_submission_pack_20260830_v3/result.json`, `official-submission-receipt.json`; `configs/experiments/p2_gaussian_copula_v2_exact_frozen_submission_pack_20260830_v1.json` explicitly pins U `13181dff` and alpha50 `bd550127`.
- I30P2/P3: `reports/official_information_probe_cycle_20260830_v1/p2-official-result.json`, `p3-official-result.json`
- K30: `reports/p3_kma_uniform_0425_official_submission_20260830_v1/official-submission-receipt.json`
- L31P2/P3: `reports/submission_ladders_internal_validation_20260831_v1/p2_3_official_submission_receipt.json`, `p3_2_official_submission_receipt.json`
- R31: `reports/parallel_internal_pass_registry_20260831_v1/official-submission-results-20260831.json`
- T31: `reports/parallel_public_transport_repair_cycle_20260831_v1/official-submission-receipt.json`
- V23/V52/V42: `reports/p2_v23_official_submission_20260901_v1/official-submission-receipt.json`, `reports/p2_v52_official_submission_20260901_v1/official-submission-receipt.json`, `reports/p3_v42_official_submission_20260901_v1/official-submission-receipt.json`
- 목록 출처: `docs/ocean_v2_codex/FINAL_DAY_PLAN_20260907.md` §4, `reports/claude_recon_20260905/SUBMISSIONS_AND_SCORING_recon.md`. 독립적인 전체 계보 코드 감사는 이번 범위가 아니다.

## 5. 슬롯 결정 자료 — 지정·업로드 아님

| 조건 | 이번 확인 | 권고 |
|---|---|---|
| 재가중 재계산이 불일치하면 | 점추정과 Fable 방식 CI까지 일치. 정렬 차이 해결 | 이 조건으로 제출 우선순위를 올릴 근거 없음 |
| 사용자가 P3 CPU 계보를 최종 선택하면 | 현재 `2015b387` Public 0.609836/23.654388점. `70761aff` no-shrink 공식 결과 미측정 | 두 CPU 후보 비교 목적 제출 가능. 재가중 근거는 shrink 유지 쪽. 아직 최종 지정하지 않음 |
| 시간·슬롯 여유가 있고 대안보다 정보 가치가 크면 | 남은 슬롯은 직전 receipt 기준 P1 2/P2 1/P3 2. 마감 시각 미확인 | 사용자의 정보 가치 선택으로 no-shrink 제출 가능. 이번 지시는 업로드 0이므로 실행하지 않음 |
| 일부 구간만 악화하면 | raw pooled는 개선, test-mix 가중 pooled는 악화. 서로 다른 모집단 가정 | worst-block 단독 자동 탈락 금지. Private 성능 미확인, 최고점 돌파 불가능이라고 말하지 않음 |

P1 원형 `57844ef2`(28.909341), P2 `9c5fec38`(28.240037), P3 `ff42a6a0`(23.741446)는 기존 비교 자산으로 보존한다. 신규 최종 지정은 하지 않았다. P2 09-07 s3-proj는 위 Tier3 과거 anchor 목록과 다른 파일이므로 혼동하여 삭제하지 않는다.

## 6. 진행 상태 및 미수행

2026-09-07 **13:44 KST**: fresh_cold_2 PID26996 정상 생존, supervisor PID39904 생존, CPU4 불변, backbone **13/36** 완료. stderr는 kernel TCP 암호화 경고이며 확인한 tail에 학습 fatal 오류는 없다. 설정·프로세스·모델·seed·thread 변경 및 재시작 0.

완료 fit 누적 5,518.672초 / 기존 같은 prefix 5,246.938초 = 속도비 1.0518. 기존 나머지 fit 13,483.456초를 비례 환산하면 약 14,181.752초 남음, 13:44 기준 **17:40 전후 학습 완료 추정**, 이어서 추론·QA·패키지 확인. 진행 중 fit의 경과분을 따로 빼지 않은 추정이므로 정밀 마감 보장이 아니며, 리소스 변화에 따라 달라진다. 안전하게 17:30~18:00대 범위로 본다.

fresh-cold 완료 receipt가 없으므로 답안 SHA `2015b38750d357630d5b2e9eee32807d2961ce752e35eb2b454ee16e579dda56` 대조는 **미수행(실행 중)**. 기존 completion_1은 29 fit 재사용+12 fit completion이며 fresh 두 번 완료라고 쓰지 않는다. 공식 6시간 일반 모델 적용 범위·오늘 마감 시각 미확인.

**13:50 KST 추가 확인:** backbone **14/36** 완료, 같은 prefix 비례 계산의 학습 종료 추정 17:31. 긴 multi fit이 끝날 때 ETA가 이동하므로 사용자 안내 범위는 **17:30~18:00 전후 + QA**로 유지한다. 완료 SHA 대조는 여전히 미수행이다.

새 fit 0(기존 학습은 계속 진행), 업로드 0, 삭제 0, 최종 지정 0, commit 0, push 0. 원 `result.json`, 모델·답안·학습 프로세스는 변경하지 않았다. 사용자만 수행하는 삭제는 정확한 항목·SHA를 확인하고 원격 처리 규칙을 확인한 뒤 판단해야 한다.
