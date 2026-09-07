# Packaging amendment — no fit / no inference restart

B5's first actual saved notebook completed and generated SHA `cbeb74263733c116bcb9412ec2b78844b49a1a6ef0fee1021b8b1ed706681163`. The inherited generic ZIP builder then rejected `03_model/tree/own_probe.npz`: its allowlist only covered MS-owned training probes. This is a packaging omission, not a model or scientific failure.

The new variant saved notebook intentionally executes separate-process tree-qa, so it needs the new 256-row, 80-feature TRAIN-only probe. The disjoint variant ZIP builder explicitly validates and permits that one exact filename/schema plus the already allowed MS probes. Other tables, predictions, logs, locks and arbitrary NPZ stay excluded. Original generic builder and original packages are unchanged. No official/hidden data is in this tree probe; its creation reads the first 256 distributed training rows only.

Resume packaging reuses the completed first notebook only after its source and answer SHA checks, verifies any existing SOURCE ZIP member-by-member, creates the missing saved ZIP, then executes a genuinely fresh extracted saved notebook. It does not overwrite a used notebook, answer, model or attempt, and does not retrain.
