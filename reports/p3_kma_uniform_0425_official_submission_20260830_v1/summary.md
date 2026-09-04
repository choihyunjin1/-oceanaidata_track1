# P3 KMA uniform 42.5% official result

Status: `PUBLIC_BEST_ONLY · PRIVATE_UNCONFIRMED`

The official submission-management page records the `P3_KMA_UNIFORM_0425_20260830_V1` submission at `2026-08-30 21:19 KST` (minute precision), with OCN-03 RMSE `0.575233` and score `24.203599`. Against the prior public best RMSE `0.575262` / score `24.203126`, this is `−0.000029m` and `+0.000473` points. The candidate therefore becomes the new Public best, with two of three daily submissions remaining.

The frozen curve predicted RMSE `0.575232`; the displayed official result differs by `+0.000001m`. This confirms only the already-identified uniform KMA same-axis micro-exploit. It does not promote the KMA family to a new structural model result, does not establish Private improvement, and does not justify another nearby-alpha probe.

Candidate CSV SHA-256: `144f5e1740a338df881b5076b8d0a8764630c5836982a6ff4326e93c2e24219e`. The CSV remains under the ignored `submissions/` staging tree. The verification pass reread only the official submission-management receipt; it performed no new submission, training, prediction, hidden-truth read, or `score.py` access.
