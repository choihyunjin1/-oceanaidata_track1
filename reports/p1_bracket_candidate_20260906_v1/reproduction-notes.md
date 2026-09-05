# P1 candidate regeneration and saved-output limits

This completed folder preserves a **local, independently checked candidate**, not an uploaded model or a permanent saved-model replay service. The frozen runtime measures3600 seconds from `train_result.json.started_unix`; re-running infer/verify in this completed folder tomorrow will correctly fail the wall cap. Do not edit that clock, the attempt lock, recipe, policy or source hashes.

To regenerate later, use a new empty folder outside the original repository. Copy only this package's `02_code` snapshot and README, and create empty `01_data`, `03_model`, `04_logs`, `05_answer`, `06_docs`. Copy `06_docs/qa_training_independent.py` to the new06_docs. Do **not** copy existing models, internal prediction probes, answer files, attempt lock, training result or QA PASS receipts into the new run. Install/use the pinned requirements and supply `P1_DATA_DIR` pointing to the unchanged distributed P1 dataset.

Run, in order, from the new package:

```powershell
python -I 02_code/run.py train
python -I 02_code/run.py model-qa
python 06_docs/qa_training_independent.py <new-package-path> --data <P1-dataset-path>
python -I 02_code/run.py infer
python -I 02_code/run.py verify
```

Training performs exactly4 fresh CPU4 fits. Inner calibration/selection is recomputed from the fixed final-inner interval, not loaded from the completed package's answer or forced to a target positive count. Root's independent helper is a byte-exact copy of `scripts/qa_p1_bracket_candidate_20260906_v1.py` and requires no original repository source. It checks newly regenerated training support, model hashes/settings, full internal probabilities, grid-selection arithmetic and receipt links. All stages must finish in the frozen3600-second budget. A failure is a stop, not permission to override checks or reuse an old answer.

Current evidence is one complete4-fit training workflow, model reload replay in another PID, local inference and complete answer replay in another PID. It is **not** two complete independent training workflows or a clean-machine/other-hardware certification. Cross-machine numerical equivalence remains unverified; a changed environment requires honest new QA. A durable inference-only adapter, an official final-model lock, a ZIP submission bundle or a leaderboard upload would be separate work.

Preserved layout:01_data contains no copied observations;02_code has the source snapshot and dependency pins;03_model contains four newly trained models/encoders/train statistics plus the frozen recipe;04_logs contains train-derived replay arrays;05_answer contains the local CSV;06_docs contains linked validation receipts and this guide. Generated caches are excluded from the preservation copy. Neither training observations nor hidden/sample prediction values are bundled.
