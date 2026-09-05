# P1 screen 결론: flank 후보 실패, 재학습 가능한 강한 control 보존

2026-09-05 실행. **새25개 flank 특징의 선택 후보는 F1을0.85117424에서0.83692788로 악화시켰다. 추가seed는 실행하지 않는다.** 단, 동일 평가의 clean inner-selected control은 XGB 단독0.84322658보다+0.00794766 개선되어 별도 재학습·배포 후보로 보존할 가치가 있다. 이것은 과거 공식28.909341점 모델의 복제 또는 공식 개선 확인이 아니다.

| 내부 평가421,032행 | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|
| XGB 단독 |0.84322658|12,586|1,211|3,469|
| event-day B 단독 |0.84661502|12,568|1,067|3,487|
| flank 단독 |0.84088816|12,327|937|3,728|
| inner-selected control |0.85117424|12,794|1,213|3,261|
| inner-selected flank 후보 |0.83692788|12,864|1,822|3,191|

후보는TP70행을 추가 회수했지만FP609행이 늘었다. 단독 arm에서도F1이 낮으므로 OR 제약만 제거하면 해결된다고 볼 수 없다. 다만 Q2 flank 단독은FP를 크게 줄여F1이0.80944였지만 Q3·Q4에서는 악화했다. 이 사실을 보고 Q2만 사후 선택하지 않는다.

fragmentation stress408,789행에서는 control0.84773494→후보0.83125531로 악화했다. 스트레스 행은 label-independent random gaps로 정해졌고, retained 동일 키의 intact 비교도 fold JSON에 별도 기록되어 있다.

## 예측 점수 표시와 실제 점수 구분

- 내부 후보 ΔF1−0.01424636. 이 차이가 그대로 공식에 이전된다는 가정의 표시환산은 약−0.379점이며 실제 공식 예측이 아니다.
- 보존 control의 XGB 대비 내부 ΔF1+0.00794766은 같은 가정에서 약+0.211점 상당이다. 기존 공식 최고와 모집단·모델이 달라28.909341에 바로 더하지 않는다.
- 공식 입력·CSV·upload0. 공식 점수는 미측정이다.

## 표현 대 임계값 진단

독립 재계산한 pooled average precision은 B0.86215289→flank0.85872858로 악화했다.48h 이상 양성6,494행에서 B recall3974/6494=61.1949%→flank3798/6494=58.4848%; probability<0.05행은2362→2567로 증가했다. 임계값만 교정하면 표현이 유리해진다는 근거가 부족하여 flank 가지를 닫는다.

반면 현재 control의FN3,261행 중 일부라도 이미 탐지된 양성사건 안의 누락은1,867행(57.2524%)이다(Q2 646/Q3 1080/Q4 141). 따라서 사전 계획의 **별도 decoder1개** 분기 조건은 충족한다. 이는 해당 행을 정답으로 패치하라는 뜻이 아니며, 다음 decoder의 전이 비용·선택은 각 학습/inner에서만 결정해야 한다.

과거 `p1_typed_duration_semimarkov_v2`는 약한 typed unary control0.581043→0.583476의 작은 개선이 있었지만 spike recall0.88235→0.41176으로 손상됐다. 따라서 기존 typed 기간제약/고정 penalty, 단순 run extension/CAPA를 그대로 재실행하지 않는다. 새 조건부 계약은 강한 O/B binary score·train-only 전이 통계·현재 hard spike 보존을 명시해야 한다.

## 실행·재현 QA

- LGBM12fits, XGB6fits, inner calibration검색15회(각 threshold8개), runtime763.422초, 관측 RSS상한1.3253GiB, CPU4threads/GPU0.
- 합성 pytest12 PASS, Ruff PASS. root 독립 pooled/unique-key QA도421,032행과 수치 일치를 확인했다.
- outer 저장모델9개를 다시 불러와 특징을 재생성한 예측이 모두 기존OOF와 **완전 일치**, 반복예측도 완전 일치했다. 재학습0. 이는 저장모델 추론 재현을 검증한 것이며 별도seed 재학습 일치 검증은 아니다.
- 원래 runner의 control/union `long_probability_*`에는 B 확률이 전달된 진단상 문제가 있다. F1/TPFPFN/선택/학습에는 영향 없고, 이 필드는 해석하지 않는다. 올바른 모델별 확률/AP는 `saved-model-reload-qa.json`을 사용한다. OR에 단일 대표확률을 정의하지 않는다.
- source allowlist와 호출 경로상 배포train.csv만 읽었다. OS 전체 syscall 감사는 수행하지 않았다. 기존 artifact/dirty worktree/공식 경로는 보존, commit/push0.

검증 스킬을 적용해 약한B control만이 아니라 XGB 및inner-selected 결합을 함께 비교하고, pooled 분모와 재현 체크를 분리했다. 반복 노출된Q2/Q3/Q4이므로 fresh confirmation이 아닌 development replay다.

## 산출물

- [결과와해시](result.json), [모델reload QA/정정진단](saved-model-reload-qa.json), [사전계약](preregistration.md), [reload재검증코드](verify_saved_models.py).
- ignored `artifacts/p1_score_repair_20260905_v1/`:18개joblib,6개OOF parquet,qa_oof.npz,contract/progress/terminal/lock/logs.
- qa_oof SHA256 `94bd1e79ff28fe6bdb7bac35232e5ccde74cf6dff65cc14c597ee476f43274f8`.
- result SHA256 `ba10a102c69d1f6c0500d59ec6c2e89b3931da61ce5b6f3ebf2695479daf21a3`.

## clean control의 별도 fulltrain 계약

최종 배포는 Q2/Q3/Q4별로 다른 모델을 적용하는 것이 아니다. 배포train 마지막60일을 final-inner로, 그 이전21일을 purge하여 O/B 두 모델을 prefix 학습하고 같은 고정 임계값grid/모델군에서 선택한다(2 backbonefits). 그 선택·threshold를 고정한 후 배포train 전체로 O/B를 다시 학습한다(2 backbonefits). depth/spike 통계도 train에서만 적합해 저장하고, 별도 프로세스에서 모든 부품을 로드해 추론한다. 단일기준/union 등 사용하지 않은 부품을 제외할 수 있지만 공식 최고CSV를 입력으로 삼지 않는다. final-inner는 개발자료이며 fresh 외부평가라고 주장하지 않는다. 공식 input/materialization은 root 별도 검토 후에만 수행한다.
