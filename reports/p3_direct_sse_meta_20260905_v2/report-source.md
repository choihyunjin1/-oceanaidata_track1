# P3-A 결론: 직접 SSE 결합과 global bias는 개선되지 않았다

정해진 두 후보를 정확히 한 번 평가했으며 **기존 no-op 기준선을 유지**한다. 결과 기반 비율 변경이나 재학습은 하지 않았다. 이 결과는 해당 2자유도/1자유도 정책의 재사용 historical 평가이며 모든 결합 방법이 실패한다는 증거는 아니다.

| 완성 정책 | 내부 pooled RMSE m | 기준선 대비 m | case bootstrap delta 95% 구간 m |
|---|---:|---:|---:|
| no-op | 0.7791048399763751 | 0 | [0, 0] |
| long-simplex | 0.7794674566793022 | +0.0003626167029271 | [-0.0008236502, +0.0015132552] |
| global bias | 0.7794290247053395 | +0.0003241847289644 | [-0.0163082545, +0.0173654679] |

실제 비교는 동일 181 cases × 6 leads = 1,086행이다. 첫 fold49cases는 보정하지 않았고, 나머지79/53cases의 보정은 각각 이전49/128cases만 적합했다. 별도 정점별/lead별/시간별 계수를 추가하지 않았다. 메타 적합4회, backbone/GPU0회, 승자 배포 메타0회, 실행0.241초. 모든 후보의 전체·fold·lead·station·관측 case-peak 상위10% 진단 및 signed-error는 [result.json](result.json)에 남겼다. 개별 관측/정답/예측 행은 보고서에 포함하지 않고 ignored artifacts에만 보존했다.

실패 해석도 구분해야 한다. global bias는 평균 signed error를 −0.1033905m에서 +0.0066216m로 줄였지만, 겨울전환 RMSE는 −0.0230777m 개선하고 2025 상반기에는 +0.0327251m 악화해 전체 RMSE는 오히려 상승했다. **전체 평균 편향이 작아졌다는 사실이 시간 이동의 제곱오차 개선을 보장하지 않는다.** long-simplex는 겨울전환 +0.0013346m 악화, 2025 상반기 −0.0007286m 개선이었다. 이 해석은 결과에 따른 추가 비율 탐색이나 특정 fold 정책 조합의 근거로 사용하지 않는다.

새 공식 예상점수는 **미산정**이다. 내부 개선이 없으므로 A만의 새 공식 제출물을 만들지 않았다. 공식 test/sample/hidden/CSV/upload0, 외부 관측0, Public 점수 역산0이다. 현행 공식 점수는 root의 공식 영수증으로 별도 보고하며 내부 RMSE를 같은 모집단 점수로 치환하지 않는다.

[독립 QA](independent-qa.json)는 별도 read-only 산술/해시 계산13/13 PASS, focused synthetic pytest7 PASS, Ruff PASS다. first-fold no-op, short-lead 보존, 미래 fold truth 변조 불변성, simplex 합1/비음수, reload 동등성을 검증했다. 접근0은 audit hook allowlist와 코드 경로 증거의 범위이며 OS 전체를 감시한 사실로 과장하지 않는다.

다음은 이미 승인된 독립 P3-B 사건가중치 분기다. A의 계수를 섞지 않고, seed-matched baseline과 새로운 폭풍 사건가중치만 대조한다. GPU는 root가 P2 종료 후 명시적으로 해제했다.

## 재현·출처

- [사전등록](preregistration.md), [고정 config](../../configs/experiments/p3_direct_sse_meta_20260905_v2.json), [runner](../../scripts/run_p3_direct_sse_meta_20260905_v2.py), [read-only QA](../../scripts/qa_p3_direct_sse_meta_20260905_v2.py), [tests](../../tests/test_p3_direct_sse_meta_20260905_v2.py).
- 원본 배포 train SHA와 train-derived cache SHA는 result의 manifest에 있다. [최상위 운영진 정책](../../00_ORGANIZER_DATA_POLICY.md)을 적용했다.
- 결과 SHA-256 `8e7e3a16d3d1505eb97f9746f98dd89f216eae32ee1e9601858f932845c9f5e7`.
- historical 개발표면을 이미 반복 사용했으므로 bootstrap은 descriptive이며 fresh holdout, 선택편향 교정, 일반적인 유의성 검정으로 부르지 않는다.

이 문서는 data-analytics validate-data/analyze-data-quality 검증 지침을 적용해 평가 grain, 기준선 대조, temporal leakage, 범위·불확실성을 명시했다. Git 작업은 하지 않았다.
