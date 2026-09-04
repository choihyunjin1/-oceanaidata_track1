# Claim-source ledger

| Claim | Evidence |
|---|---|
| 공식 transport pair는 6개다 | `calibration.json.observed_pairs` |
| exact family가 tier보다 우선한다 | `select_penalty`, focused pytest |
| unseen low-DOF raw 하한은 0.059586054점이다 | 공식 KMA residual -0.049586054 + 0.01 |
| hard/unknown raw 하한은 0.331905690점이다 | 공식 ExtraTrees residual -0.321905690 + 0.01 |
| 기존 결과는 소급 재판정하지 않는다 | `calibration.json.policy.retroactive_reclassification_forbidden` |
| nested evaluation이 필요하다 | Cawley & Talbot 2010, JMLR 11:2079-2107 |
| covariate-shift 보정은 mechanism에 의존한다 | Tibshirani et al. 2019, NeurIPS |
| domain-shift model selection은 별도 protocol이 필요하다 | Gulrajani & Lopez-Paz 2021, ICLR |
