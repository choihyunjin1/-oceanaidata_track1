# P3 v32 terminal technical failure

## 결론

- 상태: `INVALID_TERMINAL_TECHNICAL_FAILURE`; 동일 ID 재실행 금지.
- full historical BOCPD feature extraction 중 확률공간 Gaussian likelihood가 극단값에서 모두 underflow하여 posterior normalization이 0이 됐다.
- model fit, outer score, RMSE, slice, CI는 0회 산출됐다. 과학적 `NO_GO`가 아니다.
- 공식 test/sample/submission/hidden, CSV, upload 접근은 모두 0이다.

## Immutable evidence

- config SHA-256: `acec8a03f49517a7bdf688c74f7400b7d0441a1d7e1aadb2e2c7de50f1781c98`
- runner SHA-256: `1d73abeed75ac3bb48d78e83ae91247d52f619758071c93ad46c7941cc2c180e`
- consumed lock SHA-256: `b8add492a65183e5e8f87df6d9543867522ddb32da05f8cb2530b6b225437d6c`
- source result existence: `false`
- exception: `ContractError: BOCPD posterior normalization failed`

Fresh ID v32r1 may replace probability-domain normalization with mathematically equivalent log-domain log-sum-exp only. All scientific settings remain unchanged.
