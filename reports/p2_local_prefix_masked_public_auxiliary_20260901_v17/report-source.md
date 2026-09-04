# P2 local prefix masked-public auxiliary v17

## 결론

상태 `EXPLORATORY_NO_GO_MASKED_PUBLIC_AUXILIARY`; pooled ΔRMSE `-0.052290268℃`, canonical nominal `+0.656114`점, fixed-penalty `+0.534432`점. 모든 값은 노출된 historical surface의 탐색 지표이며 official 성능 주장이 아니다.

External depth-query plan은 downloaded/trained/evaluated가 모두 false이며 실행 output이 없다. v17은 external data나 target reconstruction 없이 각 fold prefix 안에서 current public node 하나만 fixed cycle로 마스킹한다. Vincent et al.은 denoising objective 동기만 제공한다.

official/hidden/CSV/upload=0.
