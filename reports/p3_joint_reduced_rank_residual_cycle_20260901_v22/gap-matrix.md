# Gap matrix

| Question | Evidence | Finding | Decision |
|---|---|---|---|
| Does the case-by-six-lead contract exist? | Read-only historical loader; 1,092 rows / 182 complete cases / exact six leads / finite uniform reference | Yes | Data gate PASS |
| Is reduced-rank regression a valid multivariate method? | Izenman 1975; Velu et al. 1986 | Yes | Method-rationale PASS |
| Is joint multi-horizon output meaningful? | Ben Taieb et al. 2010 | Yes | Method-rationale PASS |
| Is the exact implementation already present? | No rank-truncated coefficient implementation was found | No exact duplicate | Exact novelty PASS |
| Is the semantic input-target mechanism new? | NLinear multi-output Ridge and causal spectral multi-output residual kernel already map causal past context to six residual outputs | No | Semantic novelty FAIL |
| Do rank 1/2 and robust target handling add new information? | They alter capacity/influence only, with no new covariate, target contract, or causal receptive field | No | Architecture-scale gate FAIL |
| Should a v22 fit be run? | Semantic gate failed before preregistration | No | `STOP_SEMANTIC_DUPLICATE` |
| What is the next possible axis? | Ordered continuous-path cross-channel interactions are not named in the audited exact families | Path-signature / Neural-CDE style causal encoder | Proposal only; separate duplication audit required |
