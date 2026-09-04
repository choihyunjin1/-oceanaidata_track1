# P2 v28 fixed target-layer PCGrad DeepSets

## 결론

상태: `EXPLORATORY_NO_GO_LAYER_TASK_PCGRAD`. 9 fits와 prediction/result는 기술 오류 전에 봉인됐다. pooled ΔRMSE `-0.050893587 C`, nominal `+0.638589`점, transport `+0.516907`점.

v26a prospective fold×layer gate는 `False`: non-harm `6/9`, max cell `+0.028618268 C`. 따라서 aggregate 개선에도 안전 후보가 아니다.

최초 실행은 base result 저장 뒤 Markdown report 순서에서 KeyError로 종료됐다. 학습이나 metric을 재실행하지 않고 immutable result/prediction을 독립 QA한다. official/query/hidden/CSV/upload=0.
