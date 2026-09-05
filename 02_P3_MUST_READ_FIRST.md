# P3 — active task contract

Read with [organizer policy](00_ORGANIZER_DATA_POLICY.md) and the distributed P3 README.
Historical recipes and former gates are [archived](docs/archive/instructions_20260905/02_P3_MUST_READ_FIRST.md); they are not current instructions.

- Task: forecast significant wave height at the supplied leads from the available case history. Primary metric is pooled RMSE in metres from total SSE/row count.
- Source: P3_DATA_DIR; immutable organizer files. Answer schema: case_id,station,lead_h,hs_pred. Preserve all requested keys/order.
- Only the distributed case context is available at inference. Never identify anonymous case timestamps by matching external observations or use future waves/covariates beyond the case context.
- Build chronological historical episodes with all targets available before a model/router is fit. Purge full context/target dependencies and audit same-station episode overlap.
- Fit scalers, component models, routers, clipping/bias parameters and selection using earlier folds only. Leave-one-fold-out with future OOF labels is not chronological meta-validation.
- Count independent cases/episodes as well as rows; six leads from one case are not six independent observations.
- Current work: [plan v2](docs/SCORE_IMPROVEMENT_PLAN_20260905_V2.md), direct-SSE low-dimensional meta-policy then episode-weighted base models.
- KMA/ERA5/Chronos ancestry is ineligible. The refined-public alpha and its 0.583892m result are also not eligible under the score-use rule; preserve them only as audit history.
- Check CatBoost parameter compatibility synthetically before an expensive run. Never change consumed attempt locks or restart a frozen failure automatically.
- Read [handoff](AI_HANDOFF.md) before official inputs, candidate materialization or final packaging.
