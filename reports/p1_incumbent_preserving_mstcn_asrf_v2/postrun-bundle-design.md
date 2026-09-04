# P1 MS-TCN/ASRF v2 post-run bundle 설계

이 문서는 `scripts/build_p1_mstcn_asrf_v2_postrun_bundle.py`가 최종 실행 뒤 생성할 정적 증거 묶음의 계약이다. 도구는 아직 실행하지 않는다.

## Fail-closed 입력 계약

- `terminal_result.json`이 없으면 즉시 종료하고 어떤 보고서 파일도 쓰지 않는다.
- Q3·Q4 결과가 존재하는 완결 상태는 아래 세 개만 허용한다.
  - `GO_HIGH_IMPACT_OFFICIAL_PROBE_ELIGIBLE_NOT_AUTHORIZED`
  - `GO_RESEARCH_ONLY_NOT_OFFICIAL_PROBE_ELIGIBLE`
  - `NO_GO_CONFIRMATORY`
- `confirmatory_metrics.json`과 terminal 내부 복사본, `selected_recipe.json`과 terminal 내부 복사본이 canonical JSON으로 동일해야 한다.
- Q2는 PASS이고 finite-grid 882개 좌표가 완전·고유해야 한다.
- Q2 width 256/512 × seed 3개의 300 epoch history와, 선택 width × seed 3개의 Q3·Q4 선택-epoch history가 모두 연속·유한이어야 한다.
- pooled current-Router 기준 TP/FP/FN `8961/203/1724`, 정확 F1 `17922/19849`, Router 양성 제거 0을 재검증한다.
- 공식 제출 생성, 업로드, 공식 +3점 확인 플래그는 모두 false여야 한다.
- 학습 label, blind prediction array, checkpoint, 공식 test/sample/submission은 읽지 않는다.

## 출력 계약

완결 검증을 통과할 때만 `reports/p1_incumbent_preserving_mstcn_asrf_v2/postrun_bundle/`에 아래 파일을 쓴다.

1. `result_summary.json` — 보고서 builder가 직접 읽을 구조화 결과
2. `result_summary.md` — 결론부터 시작하는 한국어 기술 보고서 삽입용 문안
3. `figures/figure_01_training_loss_convergence.png`
4. `figures/figure_02_q2_qualification_envelope.png`
5. `figures/figure_03_confirmatory_effects_and_gates.png`
6. `bundle_manifest.json` — 입력·출력 SHA-256과 비접근/비제출 선언

그림과 두 summary를 임시 디렉터리에 먼저 완성한 뒤 고정 위치로 옮기고 manifest를 마지막에 쓴다. 따라서 manifest가 최종 완료 표식이다.

## Chart map

| 그림 | 분석 질문 | 형식·충분성 | 지원하는 해석 | 팔레트·비색상 구분 |
|---|---|---|---|---|
| Figure 01 | 두 Q2 용량과 선택 용량의 Q3·Q4 optimizer loss가 epoch에 따라 어떻게 변했고 seed별 이상 징후가 있는가? | 2×2 log-line; Q2 width 256/512 각 300점 × seed 3개, Q3·Q4 각 선택 epoch점 × seed 3개 | 전체 12개 곡선의 학습 안정성과 tail 기울기 진단. holdout 수렴 주장은 하지 않음 | blue/gold 두 root, seed별 실선·점선·점선형 |
| Figure 02 | Q2 finite grid에서 epoch·capacity별 best ΔF1 envelope와 선택점은 어디인가? | line; width별 63 checkpoint, 각 점은 고정 threshold 7개의 최대 ΔF1 | epoch 125의 고립된 낙관적 peak와 후반 성능 하락을 selection-only 맥락으로 표시 | blue 단일 root 음영·선형, 선택점 gold marker |
| Figure 03 | Q3·Q4·pooled 개선이 gate와 불확실성, station별 부호에서 재현되는가? | grouped bar + dot/interval + horizontal bar; 3 fold aggregate와 3 station | 확인 효과, pooled CI90, high-impact 문턱, station 일관성 | blue/gold 두 root, open diamond 문턱, CI error bar, 음수 open fill |

Figure 02는 Q2 epoch 125가 maximum-over-grid에서 얻은 고립된 낙관적 peak이며 `selection-only`라는 점을 subtitle에 명시한다. Figure 03은 Q2를 제외하고 공식 +3점 주장이 아님을 subtitle에 명시한다. 세 그림 모두 흰 배경, 짙은 잉크, 조용한 회색 grid, Malgun Gothic/DejaVu Sans fallback을 사용한다.

## 보고서 결합 지침

최종 DOCX는 `standard_business_brief`와 `memo_masthead` 외관을 유지하고 다음 순서로 배치한다.

1. 결론 및 terminal status
2. 선택 사양과 수렴 해석 + Figure 01
3. Q2 selection-only 결과 + Figure 02
4. Q3·Q4 확인 결과·bootstrap·gate + Figure 03
5. station 재현성, 한계, 다음 단계

핵심 문구 guard는 다음과 같다.

> `OFFICIAL_PROBE_ELIGIBLE`은 강한 로컬 증거를 뜻할 뿐 공식 제출 승인이나 공식 +3점 확인을 뜻하지 않는다. 별도 승인된 공식 평가에서 F1 0.930749 이상을 관측하기 전에는 +3점이 미확정이다.

DOCX 작성·렌더링·시각 QA는 terminal 완료 후 별도 단계에서 수행한다. 이 helper는 DOCX/XLSX를 만들지 않으며 artifact-operation marker도 호출하지 않는다.

렌더 QA와 독립 결과 QA까지 끝난 최종본만 이후 `C:\Users\cedis\Downloads\해양 해커톤 제출용` 아래의 새 버전 P1 연구 폴더로 복사한다. 이 준비 단계에서는 해당 제출용 폴더를 만들거나 갱신하지 않는다.
