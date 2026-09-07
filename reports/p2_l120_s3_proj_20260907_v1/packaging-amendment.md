# P2 packaging-only amendment

The first new package (`P2_L120_s3_proj_v1`) generated the candidate and passed independent scalar QA and separate-PID projection replay, but its extracted saved notebook stopped before inference with `new-process training lineage contract`.

Cause verified in unchanged `02_code/base.py:fingerprint`: it hashes every root `02_code/*.py`. Adding two projection scripts there changed the file set relative to the inherited MODEL_MANIFEST. Model bytes and original code were unchanged. This was a real packaging integrity failure, not evidence against projection quality.

Preserve the failed package and extraction. New packaging attempt `P2_L120_s3_proj_v2` moves only the added postprocess scripts into `02_code/projection/`; original root source fingerprint must equal the training manifest before archive creation. No guard is bypassed, no old receipt is edited, no model is retrained, and no inference attempt is restarted in place. The exact candidate SHA must remain `9c5fec385930ab5a997a70c10550f3ada18872a452938bfc3fa78232e17a5118`.

Official score unknown. Uploads/deletions/final designation: zero.
