# P2-B — 고정 물리 프로파일 트리, 주평가 개선 없이 정상 종료

**A 실패 후 계획된 B까지 실제 학습했다. 3 fits / 39.781초 / 독립 QA 30-check PASS. 가을 주평가에서 C가 이겨 기존 3-seed C 유지, 추가 6 fits 및 fulltrain은 실행하지 않았다.** A/B 전체 결론·비용·주평가/보조표·규정은 [통합 실행 보고서](../p2_objective_alignment_20260905_v2/report-source.md)에 있다.

## 고정 구현

- [설정](../../configs/experiments/p2_physical_profile_tree_20260905_v2.json), [runner](../../scripts/run_p2_physical_profile_tree_20260905_v2.py), [실행 전 seal](preregistration-seal.json).
- 단일 LightGBM: L2, 400 rounds, learning rate .04, 31 leaves, depth7, minchild200, row/column fraction .85, alpha .2/lambda1, CPU2threads. 기존 `src/p2_restore/model.py::_estimator` recipe를 가져오고 resource만 제한했다.
- 69개 공개 특징: 물리 nominal-depth slot 4/19/30/39/49m의 절대 T/S/실제 depth/nominal/presence, ±6/12h 공개 수온 차이, 인접 수심 T/S 차이, target nominal depth/layer, 공개 프로파일 baseline/scale/count, 연중·일중·M2 주기. 연도/elapsed index를 추가하지 않았다.
- target 층 temp·psal 제거와 blockmask를 공개 lag/delta 계산보다 먼저 적용했다. ±12h 특징 범위는 ±7일 purge보다 작다. future 공개 관측 사용은 복원 과제의 허용된 full-history 입력이며 target의 future 값을 사용하는 것이 아니다.
- 공개 수온 support 부족행을 시간축에서 제거해 이웃이 당겨지는 오류를 피했다. 지원되지 않는 입력은 NaN으로 남기고 학습/평가 적격 분모를 명시했다. 임의 이상치 삭제 없음.
- 원본/증강 가중치 질량 보존. training target은 공개 baseline 대비 절대 ℃ residual이다. 저장 Booster 재추론 hash와 수치 동일성을 검증했다.

## 첫 seed 실측

| 정책 | 2024 Sep–Oct | 2025 Jul–Aug | 2025 Nov–Dec | pooled |
|---|---:|---:|---:|---:|
| C | 0.465330203 | 1.361585875 | 0.242470408 | 0.896730522 |
| tree | 0.710383091 | 1.205431007 | 0.645335072 | 0.919654290 |
| 사전고정 50:50 평균 | 0.518502426 | 1.193445330 | 0.365079351 | 0.823186375 |

단위는 모두 RMSE ℃. 주평가 행수 26,273, 전체 69,850. 고정 절반 정책은 fitted coefficient가 아니며 세 번째 모델 fit도 아니다. 첫 seed `20260901`의 C/tree를 평균했다. fullfit 0, calibration 0이다.

고정 가을 outage 6,031행에서 C 0.465796410, tree 0.856762381, 절반 0.583048326℃였다. tree는 이 결측 조건에서도 C를 보완하지 못했다. pooled 개선만으로 주평가를 바꾸지 않았다.

## 실행·검증 영수증

- 상태 `NO_PRIMARY_IMPROVEMENT_P2_AB_COMPLETE`, chosen `C_mean`, primary delta0. 기존 C3seed의 주평가 0.488284326 / pooled 0.859249914℃를 그대로 유지했다.
- 신규3fits, 각400rounds, 학습 wall 5.469/5.266/5.907초. 데이터·특징·저장·재추론·스트레스 포함 전체 runner wall39.781초. 추가6fits 미실행. GPU0, CPU2threads 종료/해제.
- runner SHA `76138e8c6766013851ba17e78525522bd59b97111632889bb6a87bb961e407d3`; config SHA `cf6c22ee0311bdeb99645362c92f66ebb11b076ff09c250c038c06534b325ea3`; raw OOF SHA `3593b5674b13b611863415ac1fa0ee9829c5a3c8cc92d87a56bb18692382e8c5`.
- [독립 재계산](independent-recalculation.json) 30checks PASS. [전용 tests](../../tests/test_p2_physical_profile_tree_20260905_v2.py) 6PASS; A/B/기존 합계27pytest PASS, Ruff PASS.
- QA JSON 직렬화의 NumPy bool만 native bool로 바꾸고 독립 QA를 재실행했다. 봉인된 학습 runner/config/result/artifacts를 바꾸거나 fit을 반복하지 않았다.
- 공식 test/sample/baseline/hidden 접근, CSV, upload 모두0. 신규 예상 공식 점수 `미산정`; 현재 C 공식점수27.622418.

이 결과는 특정 고정 recipe의 반복노출 historical development 결과다. 모든 트리 모델이 불가능하다거나 추가 capacity로 무조건 좋아진다는 결론은 내리지 않는다. 승인된 A/B 범위를 마쳤고 새 ad-hoc 분기는 실행하지 않았다.
