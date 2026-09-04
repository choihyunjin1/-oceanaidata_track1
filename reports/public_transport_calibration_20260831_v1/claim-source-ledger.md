# Claim-source ledger

| 주장 | 1차 출처 | 적용 |
|---|---|---|
| 유한 검증 기준의 반복 최적화는 모델 선택 편향을 만든다 | Cawley & Talbot, 2010, JMLR, https://www.jmlr.org/beta/papers/v11/cawley10a.html | 후보 생성과 평가 분리 |
| 시계열 평가에는 block 검증이 적합하다 | Bergmeir & Benítez, 2012, Information Sciences, https://doi.org/10.1016/J.INS.2011.12.028 | 시간순 외부 fold |
| 구조적 의존 자료에는 block CV가 필요하다 | Roberts et al., 2017, Ecography, https://www.wsl.ch/lud/biodiversity_events/papers/Roberts_et_al-2017-Ecography.pdf | station/layer/episode blocking |
| 일반 CV 불확실성은 과소평가될 수 있고 nested CV가 보완한다 | Bates, Hastie & Tibshirani, 2024, JASA, https://arxiv.org/abs/2104.00673 | nested uncertainty |
| covariate shift conformal은 밀도비 가정이 필요하다 | Tibshirani et al., 2019, NeurIPS, https://proceedings.neurips.cc/paper_files/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html | conformal 단독 PASS 금지 |
