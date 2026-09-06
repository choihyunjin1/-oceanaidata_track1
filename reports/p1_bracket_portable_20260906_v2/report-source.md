# P1 bracket portable v2

## Current conclusion

Saved-model ZIP extraction and independent-process inference **PASS**: 169,011 rows,
7,014 positive labels, SHA-256
`9031c84ea72dfa4294406dd995525e89e8975a76983e9f9d7a7b2ba74dbad93a`.
This is exact reproduction of the frozen candidate answer, not additional training.
Cold-start ZIP was also extracted to a new folder with an empty `03_model` and
completed **4 fresh CPU-2 fits → independent model/metric QA → inference → a second
answer process**. Its answer has the same `9031...d93a` SHA. Cold model files differ
from the CPU-4 originals; answer-level equivalence passed without forcing policy,
thresholds or predictions. No additional experiment, upload or Git action occurred.

## Frozen scope and budget

- New ID: `p1_bracket_portable_20260906_v2`. Existing candidate, source manifest,
  model files, locks, and 3,600-second experiment clock are unchanged.
- Saved inference: new clock-independent adapter; CPU 2 threads, GPU 0, zero fits,
  600-second per-invocation cap. Actual ZIP extraction and new PID 23008 took
  29.234 seconds. Sample prediction/hidden/prior-answer values read: zero.
- Cold resource amendment: CPU **2**, not the source candidate's CPU **4**.
  Exactly final-inner B/O two fits then full O/B two fits; maximum 4 new fits and
  3,600 seconds for the entire cold workflow. Completed in **753.030 seconds**
  including stage gaps; no extra fit was needed or permitted.
- O 80 / B 107 features, 700 trees, original final-inner dates, training/run
  exclusion, policy selection algorithm and decoder remain unchanged. No depth
  fallback, cell policy, label-zero deletion/downweight, or threshold override.
- A new empty `03_model` is used. Prior model and answer files are not cold-training
  inputs. CPU-4 model hash equality is a comparison, not an enforced gate.
- Cold completion, fresh probability replay, own answer replay, and equality to
  the original answer are separate outcomes. An old official score cannot be
  transferred to a different answer hash. No future FP=0 or score gain is claimed.

## Preflight evidence

Focused synthetic/packaging tests: **10 passed**, zero errors/skips;
[JUnit](preflight-tests.xml). Ruff passed for the new adapter/builder/tests and
generated cold runner/QA helper. Tests cover immutable source hashes, explicit ZIP
members/CRC, no raw data/answer/probes/locks, cold empty-model layout, isolated
`python -I` entrypoints, exact ID/thread-only training changes, and tamper rejection
before official reads. The already validated feature/selection implementation is
copied unchanged; these tests do not constitute a new scientific evaluation.

Frozen hashes:

| Item | SHA-256 |
|---|---|
| New contract | `ad47adb264ccc038e20c20834e3a8025d9bb3f324f5ee513f16747884a6eb64c` |
| Saved adapter | `02cb6a0c8a9bc735ac222d6fd33b22a1e45c22ac3fbe7b530a27326c3d09b63d` |
| Cold runner | `bada4188b1893566804c7e7b2c45ab222ca393d75e718e6c2d7e6ae19a324019` |
| Cold source manifest | `f51c3cf901a955051a972d3d1f50c2dc428c2c2f16f5ca8bc2840328355028a9` |
| Cold config | `0b203584d52c7552112abe727dbbf6c41b704dd6e274f60d2a7840a07741ca15` |
| Saved ZIP | `46713ece06a4ba404ee3e4761a460e6407541d533d9139aa90f9f564cea34a5a` |
| Cold ZIP | `9325d8f98be5dba4e3ce612e56446a5f04ac4aa7abc48e2b70911c468ff04cb9` |

Archives and full member ledger:
`artifacts/p1_bracket_portable_20260906_v2/`.
Saved ZIP: 3,153,897 bytes; cold code-only ZIP: 47,487 bytes.
The saved replay was run from an actually extracted OS-temp copy with the original
repository denied for model/code access. Existing installed Python dependencies
were used; a network-disabled clean OS and offline wheel installation are **not**
verified. The cold ZIP must likewise be extracted to a new folder per attempt.

## Completed stage receipts

| Stage | PID | Seconds | Result |
|---|---:|---:|---|
| Saved ZIP replay, zero fits | 23008 | 29.234 | Exact original answer SHA |
| Cold empty-model training | 9712 | 376.750 | 4/4 new fits; official input zero |
| Cold model replay | 40080 | 119.469 | All four model probabilities exact in new PID |
| Independent cold training QA | 39508 | 26.188 | 47/47 checks PASS |
| Cold local inference | 39264 | 31.266 | 169,011 rows, 7,014 positives, original answer SHA |
| Cold answer replay | 38896 | 31.828 | Separate PID, byte-exact answer |

Evidence: [saved replay](saved-replay-qa.json), [cold training](cold-training-result.json),
[model replay](cold-model-replay-qa.json), [47-check QA](cold-independent-training-qa.json),
[inference](cold-inference-qa.json), [answer replay](cold-answer-replay-qa.json),
[preservation QA](preservation-qa.json).

The cold policy was rederived as `balanced`, threshold `0.1`, internal F1
`0.9002041551579201` (TP 3,748 / FP 134 / FN 697 on 123,372 rows). This is the
same selection result as the source candidate, not a new holdout score. Full-model
probe quality was deliberately not measured. Both original-arm model SHA comparisons
against CPU-4 are false; all four cold model hashes are separately recorded. The
resource change is explicit, and exact model-byte reproduction is **not** claimed.

Cold recipe SHA: `cb54ca37d728fb2b7d65e5a9cbce6accb710d12926638f40850355e7d62e1dba`.
Cold training result SHA: `795390a775edb176ab2c36c799e28659cc98639357157a01d6b59603194c6df2`.
The preserved completed package is
`artifacts/p1_bracket_portable_20260906_v2/cold-validation/P1/`:
39 files, 15,111,089 bytes, every copied file hash exact; source observations and
Python caches excluded. Internal probes/attempt lock remain local, not in either ZIP.

## Use and remaining limits

- **Repeated saved inference:** unzip `P1_bracket_saved_v2.zip`, run
  `python -I <P1>/02_code/archive_infer.py --official-approved --output-root <new-folder>`.
  The new adapter has its own 600-second clock and leaves the old clock/lock untouched.
- **Training from scratch:** unzip `P1_bracket_cold_v2.zip` to a new folder and follow
  its README: `run.py train`, `run.py model-qa`, included independent QA, `run.py infer`,
  `run.py verify`. Supply `P1_DATA_DIR`; the ZIP contains no observations/models/answers.
  The completed `cold-validation` folder must not be restarted: its own 3,600-second
  clock and consumed attempt lock are intentional.
- Current installed, pinned Python environment + actual ZIP relocation were tested;
  a clean OS, offline wheel installation, different hardware, and a second complete
  CPU-2 training run were **not** tested. This is one cold four-fit workflow, not two.
- Training reads only distributed train; inference reads distributed test observations
  and four sample keys. Sample prediction/hidden/external-observation reads, uploads,
  Git changes, and final selection locks by this lane: zero.
- Range/cell-policy research is owned separately by root and was not mixed into these
  packages. Official score evidence is separately owned by root and cannot drive changes
  to this frozen policy. Validation skill guidance led to keeping saved replay, fresh
  training, model-byte equality and answer-byte equality as separate claims.
