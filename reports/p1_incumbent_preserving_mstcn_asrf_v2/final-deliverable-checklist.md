# P1 incumbent-preserving MS-TCN++/ASRF v2 최종 산출물 체크리스트

작성 기준: 2026-08-27 canonical one-shot 실행 종료 후 산출물  
적용 대상: 최종 기술보고서 DOCX 및 정량 부록 XLSX  
실험 ID: `p1_incumbent_preserving_mstcn_asrf_v2`

## 0. 결론과 표현 경계

- [ ] 첫 문장과 표지 상태는 **`NO_GO_CONFIRMATORY`**로 쓴다.
- [ ] 결론은 “Q2에서는 선택 기준을 통과했으나, 독립 Q3·Q4 확인에서 재현되지 않아 공식 제출 후보로 승격하지 않는다”로 고정한다.
- [ ] `research_result=FAIL`, `high_impact_official_probe_result=FAIL`을 둘 다 명시한다.
- [ ] 공식 제출 생성 `false`, 업로드 `false`, 공식 제출 승인 `false`, 공식 +3점 확인 `false`를 명시한다.
- [ ] 로컬 결과를 공식 점수 개선으로 표현하지 않는다. 공식 약 +3점 주장에는 별도 승인된 공식 P1 F1 `>=0.930749`가 필요하다는 문구를 넣는다.
- [ ] Q2 `+0.098157`은 **선택 전용 낙관적 결과**이며 승격 증거가 아님을 모든 표·그림 캡션에서 반복 확인한다.
- [ ] 실패 원인은 계산 오류가 아니라 전이 실패로 정리한다: 전역 threshold 기반 add-only 복구가 Q4와 2/3 station에서 FP를 과다 추가했다.

## 1. DOCX에 반드시 들어갈 식별·실행 표

| 필드 | 고정값 |
|---|---:|
| 팀 | 분당독고다이 |
| 실험 ID | `p1_incumbent_preserving_mstcn_asrf_v2` |
| terminal status | `NO_GO_CONFIRMATORY` |
| 실행 시작 | `2026-08-27T05:47:41.76402+09:00` |
| 실행 종료 | `2026-08-27T10:03:14.896871+09:00` |
| 경과 시간 | `15333.132851 s` (`4:15:33.133`) |
| 장치 | NVIDIA GeForce RTX 5090 |
| Q2 역할 | qualification/finite-grid selection only |
| Q3·Q4 역할 | primary confirmatory |
| 제출/업로드 | 생성 0회 / 업로드 0회 |

주의: 원본 JSON 필드명에 `_utc`가 있으나 저장값은 `+09:00` 오프셋을 포함한다. 보고서에서는 KST ISO 시각으로 표기한다.

## 2. 선택 레시피 표

| 항목 | 선택값 |
|---|---:|
| width / parameter count | `512` / `52,568,587` |
| batch size | `64` |
| epoch | `125` |
| seeds | `20260827, 20260839, 20260863` |
| representation | raw three-seed ensemble mean |
| high threshold / low threshold | `0.9` / `0.45` |
| minimum added segment | `19 rows` |
| maximum added segment | 없음 |
| boundary snap radius | `12 rows` |
| gap closing | `0 rows` |
| candidate rule | current Router OR decoded proposal |
| type integration | `event_probability * (0.75 + 0.25 * max(P(noise), P(offset), P(drift)))` |
| convergence claim | `Q2_SELECTED_BEFORE_MAX_EPOCH_NO_CONVERGENCE_CLAIM` |

- [ ] `anchor_union=true`, `anchor_positive_removed_rows=0`을 구조적 invariant로 적는다.
- [ ] Q3/Q4 결과 기반 재선택·재튜닝·재실행은 없었음을 적는다.
- [ ] 입력은 74개 수치 특징과 finite/missing flag, 범주·valid/gap 채널을 합친 runtime width `165`임을 적는다.

## 3. Q2 선택 전용 정량 표

| 지표 | Anchor | Candidate | 변화/부가값 |
|---|---:|---:|---:|
| TP | 3,795 | 4,742 | +947 |
| FP | 415 | 539 | +124 |
| FN | 1,575 | 628 | -947 |
| Precision | 0.901425 | 0.897936 | -0.003489 |
| Recall | 0.706704 | 0.883054 | +0.176350 |
| F1 | 0.792276 | 0.890433 | **+0.098157** |
| 추가 행 | - | 1,071 | precision 0.884220 |
| 장기 사건 recall gain | - | - | 0.192401 |
| normal FP row ratio | - | - | 1.298795 |
| anchor positive 제거 | - | - | 0 |

Q2 continuation gate:

| 조건 | 기준 | 관측 | 판정 |
|---|---:|---:|---|
| ΔF1 | `>=0.01` | 0.098157 | PASS |
| 추가 행 precision | `>=0.70` | 0.884220 | PASS |
| 장기 사건 recall gain | `>=0.05` | 0.192401 | PASS |
| normal FP ratio | `<=2.0` | 1.298795 | PASS |
| anchor positive 제거 | `=0` | 0 | PASS |

- [ ] finite grid는 `2 widths × 63 epochs × 7 thresholds × 1 representation = 882`개임을 적는다.
- [ ] 블라인드 semantic replay는 `882/882`, `PASS`, truth column read `0`임을 적는다.
- [ ] 선택점은 고립된 낙관적 peak였음을 적는다: epoch 120 ΔF1 `0.071073`, epoch 125 `0.098157`, epoch 130 `0.078542`; 선택점과 최선 인접점 차이 `0.019615`.

## 4. Q3·Q4 확인 성능 표

| Population | Model | TP | FP | FN | Precision | Recall | F1 | ΔF1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Q3 | Anchor | 5,241 | 74 | 1,156 | 0.986077 | 0.819290 | 0.894980 | - |
| Q3 | Candidate | 5,491 | 203 | 906 | 0.964348 | 0.858371 | 0.908279 | **+0.013299** |
| Q4 | Anchor | 3,720 | 129 | 568 | 0.966485 | 0.867537 | 0.914342 | - |
| Q4 | Candidate | 3,757 | 466 | 531 | 0.889652 | 0.876166 | 0.882857 | **-0.031484** |
| Pooled Q3+Q4 | Anchor | 8,961 | 203 | 1,724 | 0.977848 | 0.838652 | 0.902917 | - |
| Pooled Q3+Q4 | Candidate | 9,248 | 669 | 1,437 | 0.932540 | 0.865512 | 0.897777 | **-0.005140** |

추가 행 진단:

| Population | 추가 TP | 추가 FP | 추가 행 | 추가 precision |
|---|---:|---:|---:|---:|
| Q3 | 250 | 129 | 379 | 0.659631 |
| Q4 | 37 | 337 | 374 | 0.098930 |
| Pooled | 287 | 466 | 753 | 0.381142 |

- [ ] add-only union이 F1을 높이기 위한 pooled 최소 추가 precision은 anchor F1/2인 약 `0.451459`임을 적는다.
- [ ] 관측 pooled 추가 precision `0.381142`는 이 경계보다 약 `0.070317` 낮음을 적는다.
- [ ] 핵심 구조적 실패는 Q4에서 recall을 `+0.008629` 높이는 대신 FP를 `+337` 추가해 F1이 `-0.031484` 하락한 것임을 적는다.

## 5. Station별 확인 표

| Station | Anchor TP/FP/FN | Candidate TP/FP/FN | Anchor F1 | Candidate F1 | ΔF1 |
|---|---|---|---:|---:|---:|
| G-ORS | 666 / 2 / 329 | 711 / 5 / 284 | 0.800962 | 0.831093 | **+0.030131** |
| I-ORS | 2,991 / 6 / 853 | 3,064 / 135 / 780 | 0.874434 | 0.870084 | **-0.004350** |
| S-ORS | 5,304 / 195 / 542 | 5,473 / 529 / 373 | 0.935037 | 0.923869 | **-0.011168** |

- [ ] station 개선 수는 `1/3`으로 표시한다.
- [ ] XLSX에는 각 station의 precision과 recall도 별도 열로 보존한다.

## 6. Bootstrap 및 승격 gate 표

Bootstrap 고정값:

| 필드 | 값 |
|---|---:|
| 방법 | paired circular moving-block bootstrap over pooled whole KST-day cross-sections |
| block | 21일 |
| replicates | 10,000 |
| seed | 20260827 |
| pooled unique KST days | 163 |
| Q3/Q4 shared KST day | 1 |
| mean ΔF1 | -0.005624 |
| CI90 lower / upper | -0.028100 / +0.014067 |

Research success gate:

| 조건 | 기준 | 관측 | 판정 |
|---|---:|---:|---|
| pooled ΔF1 | `>=0.015` | -0.005140 | FAIL |
| 두 fold ΔF1 | 둘 다 `>0` | Q3 +0.013299 / Q4 -0.031484 | FAIL |
| CI90 lower | `>0` | -0.028100 | FAIL |
| anchor positive 제거 | `=0` | 0 | PASS |

High-impact official-probe eligibility gate:

| 조건 | 기준 | 관측 | 판정 |
|---|---:|---:|---|
| pooled F1 | `>=0.930749` | 0.897777 | FAIL |
| pooled ΔF1 | `>=0.027832` | -0.005140 | FAIL |
| 각 fold ΔF1 | 각각 `>=0.01` | Q3 +0.013299 / Q4 -0.031484 | FAIL |
| 추가 행 precision | `>=0.75` | 0.381142 | FAIL |
| CI90 lower | `>=0.015` | -0.028100 | FAIL |
| 개선 station | `>=2/3` | 1/3 | FAIL |
| anchor positive 제거 | `=0` | 0 | PASS |

## 7. 수렴·학습 안정성 표

DOCX에는 다음 12행 요약을 넣고, XLSX에는 원본 epoch 이력 2,550행을 별도 sheet로 보존한다.

| Phase | Width | Seed | Epochs | Min loss (epoch) | Final loss | Final/min | Clips | Nonfinite | Wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q2 | 256 | 20260827 | 300 | 0.007098 (294) | 0.007122 | 1.0033 | 242 | 0 | 751.895 |
| Q2 | 256 | 20260839 | 300 | 0.007172 (230) | 0.118689 | 16.5501 | 217 | 0 | 749.962 |
| Q2 | 256 | 20260863 | 300 | 0.007019 (292) | 0.007053 | 1.0049 | 243 | 0 | 744.043 |
| Q2 | 512 | 20260827 | 300 | 0.005623 (299) | 0.005635 | 1.0020 | 338 | 0 | 1850.872 |
| Q2 | 512 | 20260839 | 300 | 0.006723 (107) | 0.007306 | 1.0867 | 399 | 0 | 1851.406 |
| Q2 | 512 | 20260863 | 300 | 0.007128 (186) | 0.179833 | 25.2300 | 372 | 0 | 1848.988 |
| Q3 | 512 | 20260827 | 125 | 0.010008 (68) | 0.058722 | 5.8678 | 459 | 0 | 999.963 |
| Q3 | 512 | 20260839 | 125 | 0.008733 (122) | 0.010068 | 1.1528 | 475 | 0 | 1000.245 |
| Q3 | 512 | 20260863 | 125 | 0.010373 (78) | 0.115199 | 11.1060 | 533 | 0 | 1000.372 |
| Q4 | 512 | 20260827 | 125 | 0.008344 (96) | 0.105123 | 12.5984 | 692 | 0 | 1354.875 |
| Q4 | 512 | 20260839 | 125 | 0.009516 (125) | 0.009516 | 1.0000 | 598 | 0 | 1354.563 |
| Q4 | 512 | 20260863 | 125 | 0.010706 (74) | 0.284149 | 26.5419 | 647 | 0 | 1356.133 |

- [ ] 모든 run의 nonfinite 합 `0`을 적되, 유한성이 수렴을 뜻하지 않는다고 설명한다.
- [ ] Q3 final/min은 `5.87×, 1.15×, 11.11×`, Q4는 `12.60×, 1.00×, 26.54×`로 여러 seed의 tail 불안정성을 강조한다.
- [ ] epoch 125 선택은 Q2 성능 선택 결과이며 “125에서 수렴” 주장을 금지한다.
- [ ] optimizer training loss는 진단 전용이고 Q3/Q4 holdout gate만 승격을 결정한다고 적는다.

## 8. 용량·runtime 표

| Width | Params | Batch | Calibration step s | Calibration peak allocated bytes |
|---:|---:|---:|---:|---:|
| 256 | 13,177,099 | 128 | 0.336046 | 17,644,587,520 |
| 512 | 52,568,587 | 64 | 0.470469 | 18,073,774,080 |

실행 runtime identity:

- [ ] Python `3.12.10`, NumPy `2.3.5`, pandas `3.0.1`, pyarrow `25.0.1`.
- [ ] PyTorch `2.13.0+cu130`, CUDA `13.0`, cuDNN `92000`, RTX 5090.
- [ ] preflight와 lock 직전 runtime identity가 모두 `PASS_EXACT_RUNTIME_IDENTITY`임을 기록한다.
- [ ] confirmatory width512 peak allocated `18,084,398,080 bytes`, peak reserved `20,707,278,848 bytes`를 XLSX에 보존한다.

## 9. 시간 분할·누출 방지·블라인드 QA 표

| 항목 | Q2 | Q3 | Q4 |
|---|---:|---:|---:|
| rows | 133,170 | 176,738 | 111,124 |
| ordered key SHA-256 | `1df6ae...f6239e` | `8e4b22...8baec` | `4a9134...57f6a` |
| same-fold truth read before receipt | 0 | 0 | 0 |
| blind score SHA-256 | `867d4b...dd1f02` | `afc1f1...7347a9` | `f04f3d...dc948a` |

필수 상세값:

- [ ] Q2 score SHA-256 `867d4b25d968ce4231179181e05eee95cda04d76689c03c1514b907e21dd1f02`, bytes `173,949,505`.
- [ ] Q3 score SHA-256 `afc1f10d0ee6bb3ab1896f8a579a074ec386c73639f08d778355894b7d7347a9`, bytes `4,596,488`.
- [ ] Q4 score SHA-256 `f04f3d4cb22e75a8ec69ebed1656b618231f9a04552dcb5332670a2594dc948a`, bytes `2,947,750`.
- [ ] Q3/Q4 모두 `prior_fold_metrics_computed_before_both_confirmatory_seals=false`.
- [ ] Q3·Q4 semantic replay는 둘 다 `PASS`, decoder/anchor union replay `true`, same-fold truth read `0`.
- [ ] Q4 학습에서 historical Q3 labels 사용은 허용됐지만 Q4 same-fold truth는 봉인 상태였다는 역할 구분을 적는다.
- [ ] fold exact-key overlap `0`, series-local chronology violation `0`, checked series pairs `48`, minimum local gap `10 min`.
- [ ] Q3/Q4는 KST 달력일 1개를 공유하므로 전역 달력 분리라고 쓰지 않는다. pooled bootstrap에서 공유일 cross-section을 한 번만 샘플링했다.
- [ ] 80개 cache 특징 전부 분류, model numeric 특징 74개, 제외된 unbounded cache 열 4개(`depth_regime`, `nominal_depth_m`, `plateau_count`, `plateau_full_length`).
- [ ] bounded future 최대 `168 h`, Q2 train-to-holdout gap `504.1667 h`, non-overlap slack `167.1667 h`, holdout covariate fit/train rows `0`.

## 10. 독립 QA와 무결성 필드

- [ ] 실행 전 preflight `PASS`; protected interface reads `0`.
- [ ] 구현 sanity gate `PASS`: epoch 35, event recall@IoU0.70 `0.969697`, median event IoU `0.976282`, normal windows with any prediction `1`, finite loss/gradients `true`.
- [ ] label-free Router anchor reconstruction `PASS_LABEL_FREE_RECONSTRUCTION`: rows `421,032`, positive rows `13,374`, truth reads `0`.
- [ ] 과학·통계 QA: `P0=0 / P1=0 / P2=1`; metrics, 10,000 bootstrap, gates 독립 재계산 일치. P2는 큰 tail-loss 불안정성.
- [ ] 무결성 QA: `PASS_CONTROLLED`, `P0=0 / P1=0 / P2=2`; exact inventory `40/40`, 독립 assertions `137/137 PASS`.
- [ ] 무결성 P2 두 건은 적대적 동일 프로세스 capability 위조와 동시 same-user 경로 교체라는 controlled-host 잔여 위협임을 축약해 적는다.
- [ ] one-shot attempt lock과 execution seal을 무결성 표에 넣는다.
- [ ] `bundle_manifest.json`: `official_test_sample_submission_accessed=false`, `submission_created=false`, `upload_performed=false`, output count excluding manifest `5`.

## 11. DOCX 최소 해시 표

아래 해시는 DOCX 재현성 절에 반드시 넣는다.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| execution seal | 3,018 | `2d42ce76966876f33daf0bd3e8e62051876f95f92e866588713bcfb84886bb25` |
| config | 25,120 | `1f8940d29ea6b047273e4f53445f62230e7d72bf1f0b14abe9fb18476f0345f0` |
| launcher | 5,199 | `ad5c7a043bc3ce652c8a7b35c8a9ada60a16c23481a67c33b4d7acddfb518d05` |
| runner | 162,708 | `78df1bd3b4777560b0134bc69ad45839c8c0093da39d2c1d3d458b3a3ba9b87a` |
| model implementation | 23,220 | `57c135bfe746d06a53e5c3b83517cb96f8262a765febefbf43e1d3a3c344fd7f` |
| data implementation | 30,025 | `cf5dc2dbbb3ecf05c489b661f5427ff225caaf28af5fb44d292e3289c7bb9adf` |
| attempt lock | 2,894 | `092da264171250e03e49a29ccf9f9440965738b24db9420ed002c38014e61204` |
| selected recipe | 1,261 | `171618200c69dc8e5039e5404bdeb4e7cb6369f15ab8ff43af1be7492837515a` |
| confirmatory metrics | 4,109 | `964cae7d7dbb9f413244462eb9258e883e14f6073ad5549fadf482cb9cd03bd4` |
| terminal result | 6,280 | `7640cc0e29f364a26cd8199a7e9a55acdf329699cd5923679d8f0d513c4af2b1` |

## 12. XLSX 전체 해시 원장

`Artifact_Hashes` sheet에는 최소한 다음 canonical source를 path, bytes, SHA-256, role, verification result 열로 넣는다.

### 실행/QA 영수증

| Path | SHA-256 |
|---|---|
| `artifacts/.../preflight.json` | `4972fe4aaa62d0ed7be607f99855c004d2defd47665147bce8c60263b8fca1ad` |
| `artifacts/.../feature_dependency_receipt.json` | `4629a8176145b7efe42630ba5d302de7f4d3f0d67110ac1bfb271634916f1ecd` |
| `artifacts/.../sanity_gate.json` | `f910199ea1a31a8084b1d039d880685ed5002bb00402b5289f49d8ac3a4a8c51` |
| `artifacts/.../q2_qualification_grid_receipt.json` | `50483ac14ed6f7b60d28b937161f51043b3dd39f522909c087a5c00b5ac6df85` |
| `artifacts/.../q2_blind_semantic_replay.json` | `961622c07880abcf2c2f79a091c898db97c86db733b435322663b458a5336137` |
| `artifacts/.../q3_confirmatory_blind_receipt.json` | `f498ef3aa4f340afbb569cd3d0cb5f62ebfde10ffcb6c4ada8eb92b76f52a711` |
| `artifacts/.../q4_confirmatory_blind_receipt.json` | `ed416fcef57a49d25bbaec8ff29f38d17e688636c3464ebe28c6b57e9ceb87ed` |
| `artifacts/.../confirmatory_blind_semantic_replays.json` | `bcede9e5c51f59c26213f956f7db86ebd610053e27395d49391ab59faee9687a` |
| `artifacts/.../q2_selection.json` | `a8f041d282db08f53a9d4b6d96dc03602a1d2534ee503d5ee4f8e6db916c28dc` |

### 학습/refit history

| Run | SHA-256 |
|---|---|
| Q2 w256 s20260827 | `33d5b40b8b8be645596d5abbe7873f26ebaef962ff0b175052e465903214352b` |
| Q2 w256 s20260839 | `eedb838b0edadce826a5c5a743101fb2b0980c52bf68fbb0dd8d9321492fc530` |
| Q2 w256 s20260863 | `cbf27a2cb3549de6cb22d4bbd8995741e4319b96d906adb48dd643ff6c9ccb11` |
| Q2 w512 s20260827 | `fef8f61924896cb178438b762ab9c61c74103070130facd62584d855cebffe67` |
| Q2 w512 s20260839 | `9560671d6b450f77621392162a2b42cb462fd53137d7ef28707eeb08788e7aba` |
| Q2 w512 s20260863 | `b883772602d3b4065c7830c483e90c4e9dff709f92ab3c6cf1dd83d3684b73c4` |
| Q3 w512 s20260827 | `03890df7010f3ccb9819c7f9538c893392dbc2668755f80f6b77bff05fbee284` |
| Q3 w512 s20260839 | `c381289dce2753f2800ae41fee1c32fc76dc9e226c0801712d93805bb1be639e` |
| Q3 w512 s20260863 | `3e9c2c2dc6d812360a1a7e01b52adeea7b29e55e7ae1c8a9ff307124f184318c` |
| Q4 w512 s20260827 | `5b10ce73a3c0cab6af7e60eebce61de88d3e92365a2387f687db4c877e23740c` |
| Q4 w512 s20260839 | `c6625f14b9f3b4ceeed3f36074e0a38e28323a6e52fb8f5f02ca412aea825cfe` |
| Q4 w512 s20260863 | `7995205b51c25737f4aeedb4b7d02ff87384399d3c8e89f983aa4a957469e41a` |

### Confirmatory checkpoints

| Run | SHA-256 |
|---|---|
| Q3 s20260827 | `e5e11a187980a5c5d54b8ce19ea4fed38a37bbe8042e3c6c6c1cbc19cd9db3b9` |
| Q3 s20260839 | `5258f03af6cf9a2ee09b441300ffa245a4956f426039bfaa8939bd39657ec208` |
| Q3 s20260863 | `3c3b1f05388d94d20c62f3cc37a5d253740a638d51904e0cb1b1f7d9cf300b8c` |
| Q4 s20260827 | `bd0537929c217b0463ce6198fdd83d2763e754ab404ec5e201cc5229f4cf9cc8` |
| Q4 s20260839 | `390664a524f688a91060b5b4d7f60f9aa1a982fbda27baeadb474a196d511522` |
| Q4 s20260863 | `c9d7117700b625aeeaf978bd55e3a30863d4fc0d3b0c221dbcb0972fa8eb5342` |

### Postrun bundle

| Path | Bytes | SHA-256 |
|---|---:|---|
| `result_summary.json` | 19,189 | `0e6c50ec2a068b4b3cfcaf41c21b94425c6afb92ab76132beb8eeedb7dc19c09` |
| `result_summary.md` | 5,105 | `724dab33b47698236c8cd2921aae6dae68d7f1bf91d9a6e006d9d59b4356cbab` |
| `bundle_manifest.json` | 3,883 | `11c380c262ee5ec113ee56c2c8982952cb584502687c2c0be01e08beedb8ed21` |
| `figure_01_training_loss_convergence.png` | 834,650 | `8a8786f0c6f722475af7c706e2837f1daff01b8bad90079e4ff5c3d4256f4b8f` |
| `figure_02_q2_qualification_envelope.png` | 194,891 | `db9ffa5b00fadbc04c242e097a05b45a76b3bc7a4cec75841f9f2360a5c043c2` |
| `figure_03_confirmatory_effects_and_gates.png` | 159,813 | `9d9ef418eed7d19e79c7b6f69e51b4c5085f474e7a6541e60d1480d2866fd022` |

## 13. XLSX 필수 sheet 구조

| Sheet | 필수 내용 |
|---|---|
| `README` | 결론, 데이터 역할, 단위, 로컬/공식 표현 경계, 생성 시각 |
| `Executive_Summary` | terminal status, Q2/Q3/Q4/pooled 핵심값, bootstrap, 제출 여부 |
| `Selected_Recipe` | section 2 전체 key/value와 config/recipe hash |
| `Fold_Metrics` | phase/model별 TP, FP, FN, precision, recall, F1, delta, added TP/FP/rows/precision |
| `Station_Metrics` | station/model별 TP, FP, FN, precision, recall, F1, delta |
| `Gate_Checks` | gate family, metric, comparator, threshold, observed, boolean result, evidence path |
| `Bootstrap` | method, block days, replicates, seed, day count, mean, CI90, shared-day handling |
| `Q2_Selection_Only` | 882-cell grid 설명, 선택점·인접점, Q2 gate, 선택 전용 경고 |
| `Training_Summary` | section 7의 12행과 history hash |
| `Training_Epoch_History` | 12개 history의 2,550 epoch 행; phase/width/seed + 원본 17개 history 필드 |
| `Capacity_Runtime` | width별 params/batch/step/VRAM 및 고정 runtime versions |
| `Split_Blind_QA` | fold rows/time bounds/key hash/truth reads/score hash/replay/overlap/gap |
| `Artifact_Hashes` | 모든 canonical source와 최종 DOCX/XLSX 자체 hash |
| `Independent_QA` | 40/40 inventory, 137 assertions, P0/P1/P2, residual controlled threats |

Workbook QA:

- [ ] 숫자는 text가 아닌 numeric cell로 저장하고 표시 자릿수만 서식으로 제어한다.
- [ ] F1, precision, recall, delta는 원본 count에서 재계산하는 formula 열과 canonical 값 열을 함께 둔다.
- [ ] 재계산 차이는 절대값 `<=1e-12`인지 `QA_match`로 검증한다.
- [ ] 표마다 source path와 SHA-256 열을 둔다.
- [ ] freeze panes, autofilter, 단위, 조건부 색상 범례를 적용한다.
- [ ] 숨은 sheet, 깨진 formula, `#REF!/#VALUE!/#DIV/0!/#N/A`가 없어야 한다.

## 14. 그림 및 DOCX 렌더 QA

- [ ] Figure 1: 12개 training/refit loss 이력과 선택 epoch를 표시하고, loss는 진단 전용이라는 캡션을 붙인다.
- [ ] Figure 2: Q2 epoch 120/125/130을 포함한 qualification envelope와 “selection-only” 경고를 넣는다.
- [ ] Figure 3: Q3/Q4/pooled/station ΔF1, CI90, gate 기준을 한 화면에서 구분한다.
- [ ] 세 그림은 postrun bundle의 고정 hash와 일치하는지 확인한다.
- [ ] DOCX 전 페이지를 이미지 렌더링해 잘림, 표 넘침, 고아 제목, 깨진 한글, 흐린 그림이 없는지 눈으로 확인한다.
- [ ] DOCX 수치와 XLSX canonical 값이 정확히 일치하는지 자동 비교한다.
- [ ] 표/그림 캡션에 population, model role, metric direction, source artifact를 명시한다.
- [ ] 최종 생성 후 DOCX와 XLSX 자체의 bytes/SHA-256을 `Artifact_Hashes` sheet와 인수인계 메모에 추가한다.

## 15. 한계와 다음 전략에 반드시 포함할 항목

- [ ] width 256/512와 300 epoch는 공식 MS-TCN++ 기본보다 큰 외삽이다.
- [ ] 3 seed는 최소 안정성 점검이며 seed 불확실성을 충분히 추정하지 않는다.
- [ ] Q3/Q4는 retrospective window이며, 이번 후보에 한해 prediction-before-truth를 지켰다.
- [ ] bootstrap은 시간 표본 변동을 다루지만 Q2 다중선택/HPO 불확실성을 포함하지 않는다.
- [ ] 관측된 local/official 대응은 적어 로컬 효과 크기를 공식 점수로 수송할 수 없다.
- [ ] 다음 전략은 recall 추가보다 **추가 행 precision의 시기·station 안정화**를 우선 목표로 삼아야 한다.
- [ ] 전역 threshold 하나를 재튜닝하는 것은 이번 확인 결과 기반 사후 튜닝이므로 같은 실험의 승격 근거로 사용할 수 없다. 새 사전등록 실험으로 분리한다.

## 16. 최종 출고 전 fail-closed 확인

- [ ] DOCX와 XLSX 모두 terminal status가 `NO_GO_CONFIRMATORY`인가?
- [ ] pooled ΔF1이 `-0.005140`으로 일치하는가?
- [ ] Q3 양수, Q4 음수, station 1/3 개선이 정확히 표현됐는가?
- [ ] CI90이 0을 포함한다고 명시했는가?
- [ ] Q2를 confirmatory 또는 official evidence로 잘못 부르지 않았는가?
- [ ] 공식 +3점, 제출 가능, 수렴 완료라는 문구가 없는가?
- [ ] 보호 경로 read 0, submission/upload 0이 명시됐는가?
- [ ] 필수 canonical hash가 모두 재검증됐는가?
- [ ] DOCX 전체 페이지와 XLSX 전체 sheet를 시각/구조 QA했는가?
- [ ] 최종 DOCX/XLSX hash와 파일 경로를 기록했는가?

