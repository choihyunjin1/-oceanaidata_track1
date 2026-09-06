# P2 L120 portable cold reproduction — pre-fit record

This is exact recipe reproduction, not candidate selection or a new score search. The historical eight-fold multiseed completion remains a separate study. B3 Sep–Oct is still the primary evaluation; all-eight-fold evidence does not replace it.

- Same core/base/run/config as the approved L120 full-three-seed package. Only two-stage notebook orchestration and original-repository/network-denial bootstrap are added.
- New full fits: exactly three, seeds20260901/02/03,120epochs,wd0.0001,CPU2/GPU0. No old models/answers/OOF files enter training. Expected recipe and full-data support remain unchanged.
- Start with empty03_model and05_answer outside the original repository. Numerical process original-repository reads are denied, except installed interpreter runtime libraries. P2_DATA_DIR is the distributed dataset reference.
- Execute the actual TRAIN notebook then the actual PREDICT notebook, each in a new kernel; the numerical stages each run in separate subprocesses. Replay all166268 source-training predictions per model before official key parsing; then replay all26061 answer rows and schema/order/finite/hash.
- Cap:1800seconds elapsed across TRAIN and PREDICT; expected approximately3minutes based on previous actual169.368-second full cycle. GPU starts only after current28fit completion/replay/QA and root handoff. No portal upload/Git/final-model lock.
- Compare the cold answer to prior L120 SHA `fee6118bb4a4d1d421094aa0174d634cf73e919804ea1b62ab0bab4f1384ce4d` only after inference. Any mismatch is reported, never optimized away. The C60 fallback is preserved.
- Source-only ZIPv2 SHA `ecf76bd4d51ca2085c8ca3e08f6ffbf1f4ea0dcd8ee9055e2f1fdffa7ee60015`,64,078bytes. PACKAGE_MANIFEST SHA `3457f9735b6b337dff7d1b8ee09f6a9df10351fbd6040877f90c72dc6d168476`.

## Pre-fit technical correction and validation

The first source-only ZIP omitted empty directories because its archiver added files only. Actual outside-repository extraction detected missing03_model/04_logs/05_answer before training (0fits). Preserve that first ZIP (SHA `edd89bfaf8438a7c31937683cd0748c05c1d0ad9e675bed7048cc3640cbb83f8`) and the first extraction. The v2 container adds empty-directory entries; all18file bytes and the package source manifest are unchanged. A synthetic archive/extract regression test was added, followed by6/6focused tests and RuffPASS. No failed model fit or consumed training lock was retried.

Notebook copies and their original hashes are included in the package manifest. Root's generic notebook launcher validation is separate from this P2-specific package test. A fresh virtual environment and OS-level network sandbox are not tested; the installed numerical environment and Python process guards are used.
