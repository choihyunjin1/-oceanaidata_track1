# P3 forward candidate — full cold source package

Status: code/synthetic preparation; actual training and official inference require
the root operator's separate authorization. This is not the old three-window
baseline, not refined-public alpha, and not a saved-model-only proof.

Supply the unchanged organizer P3 directory through `P3_DATA_DIR`. No observations,
old cache, OOF, model, answer, sample predictions, or credentials are bundled.
Python 3.12.10 with pinned requirements and a compatible NVIDIA GPU is required.
The current machine's installed environment is used for code checks; clean OS,
offline wheels, and six-hour actual cold execution are not yet verified.

## Commands — fresh extraction/folder per attempt

```powershell
python -I 02_code/run.py prepare --prepare-approved
# Only after a specific candidate and exclusive GPU are authorized:
python -I 02_code/run.py train --training-approved --gpu-approved --variant numeric
# Or --variant hmax, fixed before the first fit; never combine the two changes.
python -I 02_code/run.py qa
# Only after the training QA and separate official-input authorization:
python -I 02_code/run.py infer --official-approved
python -I 02_code/run.py verify-answer --official-approved
```

Prepare is hypothesis-neutral: raw train → 591 case-local features / 24,360 anchors.
It records the raw high-wave episode definition, all current/+six target alignments,
five UTC forward split populations and raw context checks. It does not read old
cache/OOF/model/answer values or official context. Numeric and hmax share this stage.

Training first seals the explicit variant. Numeric uses numeric single lead and
591 features; hmax retains categorical lead, removes the same 64 hmax-derived base
columns and router hmax_current, using 527 features. No coefficient/threshold search.
The five forward folds create 10 single/multi backbones, then four routers fit only
safe completed prior-fold OOF. Full deployment creates two backbones plus one router
on all of this run's OOF: **12 backbones + 5 routers total**, not baseline plus candidate.
Single CPU2/700 trees, multi exclusive GPU0/1,200 trees, same fixed clean recipe and
fold/full seeds. Final router uses all historical OOF for deployment, never to claim
fresh historical validation. The 0.2 long-lead persistence shrink is fixed for this
contract, not a physical constant or a newly fitted value. Its original selection
was an August 17 local train-OOF diagnostic comparison of 0.15/0.20/0.25; it was not
a virgin preregistered holdout choice and was not derived by Public-score inversion.
The small provenance-only receipt is included at `06_docs/shrink-provenance.json`;
its source history is `reports/portable_cleanroom_20260906_v1/P3/report-source.md`,
section 0.2 provenance. No historical OOF values are copied or read by this package.

The six-hour clock starts at prepare and includes feature construction, training,
QA, inference, replay and operator gaps. A process watchdog terminates an overrun;
there is no restart of consumed locks or extra fitting after mismatch. Preserve
failures and request a separately authorized new attempt if needed.

`qa` reloads every historical model/router and full model in a new PID, reproduces
OOF/full probes, verifies counts/hashes and independently recomputes pooled SSE/N.
Full probes are in-sample determinism diagnostics, not held-out quality evidence.
Official inference is gated by QA, reads only the anonymous 48h context and index,
and writes `05_answer/submission.csv` (case_id,station,lead_h,hs_pred; 1,200 rows).
`verify-answer` regenerates it in another PID and compares exact bytes and input hashes.
GPU full-training equality to a prior answer is not guaranteed; a score belongs only
to the exact scored SHA, not an independently regenerated file with a different SHA.

Layout: 01_data reference metadata; 02_code complete source/contract; 03_model only
this attempt's trained models; 04_logs new training cache/probes; 05_answer new answer;
06_docs QA. Do not redistribute organizer observations or upload internal OOF/probes.
No network, upload, Git operation, or final model lock is implemented. Network denial
is Python-hook level, not OS-isolation certification. Repeated long-term saved-model
inference is a separate adapter/package, not this cold workflow.
