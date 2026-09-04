# P1 v26 terminal conclusion

## 결론

`P1_1_GCE_ECDF_ACTION_FRACTION_ADDONLY`은 기술적으로 정상 완료했지만 **NO_PASS**다. ECDF fraction은 v24의 action-budget 폭발을 막았으나, inner top-1의 정밀도가 outer로 전이되지 않았다. 공식 입력·hidden truth·제출 CSV·upload는 모두 0이다.

## 실측

- fits: 정확히 2
- runtime: `19.2509604s`
- Q3: 3 additions, ΔF1 `-0.0002291881`
- Q4: 1 addition, ΔF1 `-0.0001123546`
- pooled: 4 additions, TP 0 / FP 4, precision `0`
- pooled ΔF1: `-0.0001819205`
- raw expected points: `-0.0048351056`
- calibrated expected points: `-0.0102187970`
- bootstrap CI90 low: `-0.0003826060`, P(improve) `0`
- anchor removals: 0
- changed fraction: `0.0000138955`

Inner calibration은 두 prefix 모두 최소 action count 1을 선택했고 precision 1이었다. 같은 fraction을 outer로 수송하면 Q3 3행, Q4 1행이 선택됐지만 네 행 모두 false positive였다. 따라서 absolute scale drift는 해결됐어도 rank precision transport가 성립하지 않았다는 결론이다.

## QA

- runner independent QA: PASS
- py_compile: PASS
- Ruff: PASS
- focused pytest: 8/8 PASS
- sealed prediction 독립 metric 재계산: stored ΔF1/raw points/additions와 일치
- result SHA-256: `0a88987c2f9ab1bb2e4870812428ff913797b35c1aa94fd253ead5f8786f8261`
- prediction SHA-256: `dbe55a6451a562362f08f7d7a951925edcf9c4fb83f941bc2ca6e11b6bf226d3`
- lock SHA-256: `5ecf0d6d0aa8435cee9216d2de67410045fb103e07380ce8ea9ea238ccc1c695`

동일 GCE/ECDF fraction 후보의 fraction 확대, 최소-1 override, tie 변경, cap 완화는 결과 기반 재튜닝이므로 금지한다.
