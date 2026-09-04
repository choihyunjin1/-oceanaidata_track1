# Claim-source ledger

## 원본·로컬 근거

| Claim | 근거 | 한계 |
|---|---|---|
| P1은 263 binary contiguous event와 289 typed contiguous event를 갖는다 | `dataset-audit.json`; `scripts/audit_original_training_structure_20260830_v1.py` | train-only 집계이며 official generalization을 보장하지 않는다 |
| P2 공개층 state는 query/test 없이 만들 수 있고, complete T/S 전층 시각은 47,216개다 | `dataset-audit.json`; `reports/p2_copula_support_audit_20260829_v1/train-only-audit.json` | query support는 의도적으로 검사하지 않았다 |
| P2 고정 seasonal Gaussian copula exact recipe는 pooled signal에도 stability gate를 실패했다 | `reports/p2_gaussian_copula_conditional_mean_20260830_v2/report-source.md` | conditional copula family 전체의 실패가 아니다 |
| P3 과거 mechanism audit의 lead/filter 계약이 잘못됐다 | `scripts/audit_dataset_mechanisms_20260828.py`; `dataset-audit.json` | 과거 artifact는 삭제하지 않고 superseded evidence로 보존한다 |
| P3 shared canonical 60분 rising-midwave 독립 cohort는 243개이며, `t-48h Hs finite` 엄격 진단은 242개다 | `dataset-audit.json`; `artifacts/p3_selection_matched_cohort_preflight_20260830_v1/preflight.json` | canonical support와 strict-complete 민감도 진단을 섞지 않는다 |
| P3 N-HiTS-style dense72 구조는 이미 45 fits에서 세 fold 모두 악화했다 | `artifacts/p3_hierarchical_residual_basis_dense72_20260823_r4/learning_curve_evidence.json` | 모든 self-supervised encoder나 selection-matched 구조가 닫힌 것은 아니다 |
| P1 heterogeneous event-utility proposal은 Q4 support와 precision floor를 함께 실패해 0 fit에서 종료됐다 | `reports/original_dataset_breakthrough_research_20260830_v1/independent-qa.json` | P1 실행 receipt 뒤 runner가 판정 불변인 경계·원자성 hardening을 받아 실행 당시 byte hash는 남지 않았다 |
| P2 state-conditioned copula Stage-1은 pooled ΔRMSE -0.003459°C와 음수 CI90을 보였지만 1/3 window만 개선했고 JJA regression cap을 0.000174°C 초과했다 | `reports/original_dataset_breakthrough_research_20260830_v1/stage1-independent-qa.json`; sealed result SHA `493ed58d5e7893f7c9297f26fe992393fdd3498820b562e1f045a77e1e992e2f` | exact recipe의 NO_GO이며, 결과를 본 뒤 같은 사이클에서 계절별 파라미터를 재조정하지 않았다 |
| P2 Stage-1 receipt는 1회 실행을 기록하지만 runner는 fit 전 fail-closed attempt lock을 만들지 않는다 | P2 config/runner와 `stage1-independent-qa.json` provenance disclosure | “one-shot observed”만 주장하며 crash 전 재실행 불가능성이나 실행 runner의 cryptographic binding은 주장하지 않는다 |
| P3 selection-matched masked SSL + shared Huber Stage-1은 candidate 1.125374m, paired incumbent 0.811219m로 모든 window/station/lead에서 악화했다 | `reports/original_dataset_breakthrough_research_20260830_v1/stage1-independent-qa.json`; sealed result SHA `b6373cf4a6b281096dda4dbf10caf8bc9bebd01b01883269e421379ae702f7f5` | 이 exact 조합만 폐쇄하며 다른 sparse posterior/abstention 구조 전체를 폐쇄하지 않는다 |
| P3 attempt lock은 result embedded attempt와 일치하고 실행 config/runner/module/test hash도 현재 파일과 일치한다 | `stage1-independent-qa.json`; lock SHA `a053715f60c53abbf8f30e3991cee5bb7980aaf5a2224c2b246fb82c029c7ac5` | ignored 원 lock/result가 없는 clean clone에서는 QA script를 단독 재실행할 수 없다 |
| 이상점 후보는 두 Stage-1 모두 진단 표식으로만 사용했고 training row, P2 물리 극값, P3 고파고·급상승 삭제는 0이다 | 두 independent QA JSON과 각 config의 outlier/sensor policy | sensor-error ground truth가 없으므로 flag 자체도 정답으로 간주하지 않는다 |

## 1차 문헌·공식 지침

| Claim | 1차 출처 | 이 프로젝트에서의 사용 한계 |
|---|---|---|
| 유한 validation 기준의 반복 선택은 selection bias를 만든다 | Cawley & Talbot, JMLR 2010, https://www.jmlr.org/papers/v11/cawley10a.html | 인과 진단이 아니라 반복된 local 부호 역전을 해석하는 근거다 |
| F1은 비분해 metric이며 add-only proposal의 precision floor를 명시해야 한다 | Lipton, Elkan & Narayanaswamy, 2014, https://arxiv.org/abs/1402.1892 | calibration 정리는 시계열 shift에 그대로 적용되지 않아 실제 gate는 confusion-matrix 대수와 blocked replay를 쓴다 |
| 알려지지 않은 단조 주변분포에서도 rank 기반 Gaussian-copula dependence를 추정할 수 있다 | Liu, Lafferty & Wasserman, JMLR 2009, https://www.jmlr.org/papers/volume10/liu09a/liu09a.pdf | graphical-model 이론이며 P2 예측 개선을 보장하지 않는다 |
| Gaussian-copula regression에 Kendall τ 기반 latent covariance를 사용할 수 있다 | Cai & Zhang, Statistica Sinica 2018, https://www3.stat.sinica.edu.tw/sstest/oldpdf/A28n219.pdf | sparse latent-linear 가정이 P2에 정확히 성립하는지는 preflight 대상이다 |
| copula dependence parameter는 covariate의 함수가 될 수 있다 | Patton, IER 2006, https://public.econ.duke.edu/~ap172/Patton_IER_2006.pdf; Vatter & Nagler, JCGS 2018, https://doi.org/10.1080/10618600.2018.1451338 | 금융/bivariate 사례에서 P2로의 전이는 가설이다 |
| masked time-series representation은 label 없이 temporal structure를 학습할 수 있다 | Nie et al., ICLR 2023, https://openreview.net/forum?id=Jbdc0vTOcol; Dong et al., NeurIPS 2023, https://papers.nips.cc/paper_files/paper/2023/hash/5f9bfdfe3685e4ccdbc0e7fb29cccf2a-Abstract-Conference.html | benchmark 성능은 P3 폭풍 운송을 보장하지 않으며, 기존 supervised sequence 실패와 구분해 검증해야 한다 |
| sparse variational GP는 inducing variables로 posterior 근사를 낮은 비용에 수행한다 | Titsias, AISTATS 2009, https://proceedings.mlr.press/v5/titsias09a.html | 불확실성은 새 미래 forcing 정보를 만들지 못한다 |
| 해양 T/S QC는 range, spike, rate-of-change, flat-line, multivariate/neighbor tests를 구분하고 threshold를 지역별로 정한다 | U.S. IOOS QARTOD T/S Manual v2.1, https://cdn.ioos.noaa.gov/media/2020/03/QARTOD_TS_Manual_Update2_200324_final.pdf | QC flag가 곧 학습행 삭제를 뜻하지 않는다 |
| 파랑 rate-of-change 검사는 극한 폭풍에서 비활성화할 수 있으며 threshold는 운영자가 정한다 | U.S. IOOS QARTOD Waves Manual, https://cdn.ioos.noaa.gov/attachments/2019/02/QARTOD_Waves_Update2Final.pdf | 정상 폭풍 극값을 자동 삭제하지 말아야 한다는 guard 근거로만 쓴다 |
| Huber M-estimation은 오염에 덜 민감한 위치 추정의 고전적 근거다 | Huber, Annals of Mathematical Statistics 1964, https://doi.org/10.1214/aoms/1177703732 | 센서 오류 판별법 자체가 아니며 hard deletion을 정당화하지 않는다 |

## 정정 기록

- 과거 로컬 ledger의 *High-Dimensional Gaussian Copula Regression* 저자 표기는
  Dey & Zipunnikov이 아니라 T. Tony Cai와 Linjun Zhang으로 바로잡는다.
- 문헌에서 제안된 방법과 로컬 데이터에서 확인된 성능을 분리한다. 문헌 적합성만으로
  후보를 승격하지 않는다.
