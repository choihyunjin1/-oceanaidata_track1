# P3 새 모델 → 최종 no-shrink 답안 완전 일치

## 결론: PASS (한계 포함)

fresh_cold_2의 완전 신규 41fit 모델에서 기반 답안 2015b387…와 최종 no-shrink 답안 70761aff…를 모두 정확히 재현했다. 기존 채점본·모델·ZIP은 변경하지 않았다. 현재 선택본 Public RMSE 0.595521m / 23.881592점에 연결된 SHA와 같으며 새 공식 채점 또는 적격성 인증은 아니다. 최종 제출 확인은 사용자 승인 대기다.

## 직접 실행과 근거

- 기반 완료 receipt: OceanFinalDay_20260907/P3_numeric_cpudet_unbounded_v2/fresh_cold_2-receipt.json. 41 신규 fit, 재사용 0, 22182.109초(6시간 9분 42초), 기존 기반 답안과 exact. 이 학습은 이번 대조 전에 완료됐다.
- 제출 SOURCE_ONLY.zip(c3aed055…)에서 코드를 새 검증 폴더로 추출하고 source manifest를 대조했다. 제출 패키지 noshrink.py SHA aa40719365ad430a2b63caeda30183a3f3a1fd386ba285e8a15b6ea7225ebf44를 그대로 사용했다.
- fresh 모델과 학습 receipt·QA·column metadata만 복사했다. 과거 답안/예측/캐시를 추론 입력으로 복사하지 않았다. 기존 답안 값은 읽지 않고 해시만 대조했다.
- adopt-trained 1.922초 → infer 5.750초(PID 37196) → replay 5.781초(PID 29580). 후처리·독립 QA 전체 wall 14.016초. 추가 학습 0.
- 답안: 1200행 / 39869 B / SHA256 70761affca4d3fc6f1d24ae53467e5b185b23465926ebb4b851f0300872cddbd.
- schema/key/order/unique/finite/range, 독립 PID, 기반·최종 SHA, 원본/복사 모델·source pin·ZIP 불변 등 13개 검사 PASS. result.json에 전체 증거.
- 관련 합성 pytest 2 PASS, 새 검증 helper Ruff PASS. 내부 원형 모델을 다시 튜닝하지 않았다.

## 해석 주의

adopt-trained는 역사적 expected_base_answer_sha256을 null로 만든다. 따라서 내부 receipt의 base_exact_reconstruction=false는 비교 미설정이라는 의미이며 실제 불일치가 아니다. 이 대조 helper가 실제 base_sha256을 2015b387…와 별도 직접 비교해 PASS를 확인했다. 내부 원본 receipt 값은 바꾸지 않았다.

완료된 전체 fresh cold는 1회다. 다른 completion_1은 29fit 재사용+12fit 신규이므로 두 번의 전체 cold라고 부르지 않는다. 이번 검사는 완료된 원형 학습 + 패키지의 고정 addon 단계를 연결한 검증이며 전체 최종 RUN_TRAINING 노트북을 새로 한 번 더 실행한 것은 아니다.

우리 Windows PC에서의 exact 결과다. 타 CPU/OS/새 venv/차단망 재현을 입증하지 않았다. 기반 실측만으로도 6시간을 582.109초 초과했으며, 일반 모델에 공식 6시간 규칙이 적용되는지는 미확인이다. 이를 준수 PASS로 표시하지 않는다. 시각별 실행 간 대기시간을 합산 wall time으로 숨기지 않는다.

## 파일과 재개

- 검증 전용 출력: C:/Users/cedis/Documents/OceanFinalDay_20260907/P3_fresh_noshrink_confirmation_20260907_v1/05_answer/submission_p3_numeric_cpudet_noshrink.csv.
- 기존 제출용 P3_numeric_cpudet_noshrink_v2의 ANSWER와 SOURCE_ONLY/SAVED_MODELS ZIP을 그대로 유지한다.
- helper: scripts/verify_p3_fresh_noshrink_20260907.py, 판정 계약: preregistered-check.md, 직접 결과: result.json.
- final 확인·업로드·삭제·Git stage/commit/push 0. 현재 작업은 내부 QA와 외부 안내 갱신에 한정한다.
