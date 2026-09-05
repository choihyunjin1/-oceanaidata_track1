# P1 root launch observations and peer-review boundary

These are root's recorded launch observations, not measurements reconstructed by `audit_rebuilds.py`.

| Run | exec working directory | train PID | infer PID | replay PID |
|---|---|---:|---:|---:|
| a | `C:/Users/cedis/AppData/Local/Temp/ocean_p1_portable_20260906_b` | 35172 | 26960 | 39688 |
| b | `C:/Users/cedis/AppData/Local/Temp/ocean_p1_portable_20260906_c` | 28008 | 14536 | 23944 |

For every train/infer/verify process, root issued the following environment assignments in that process's PowerShell launch, then invoked the existing repo `.venv-p1/Scripts/python.exe -u 02_code/run.py <phase>`:

```powershell
$env:P1_DATA_DIR='C:\Users\cedis\PycharmProjects\PythonProject\artifacts\official_final_submission_20260905\P1\01_data\organizer_dataset'
$env:P1_DENY_REPO='C:\Users\cedis\PycharmProjects\PythonProject'
$env:PYTHONPATH=''
```

Stdout/stderr were redirected to the corresponding new package `04_logs/<phase>.log`; all six processes exited 0. Root subsequently copied completed packages, without source data, to the documented artifact run_a/run_b paths. The retained temp folders remain the execution origins.

Peer static review by the P2 agent found no old-answer ingress or runtime original-repo imports in the scoped entrypoints. It correctly noted that `independent-qa-v2.json`'s boundary fields are descriptive metadata, **not independently computed among its 20 checks**. Those fields rely on the recorded root launches and synthetic guard tests; receipt/hash checks alone do not prove operating-system or native-library I/O isolation.

The Python audit hook was installed after importing package and dependency modules. Local code origin is enforced by entrypoint path setup and source extraction, but this is not a fresh-machine/OS-network-isolation certification. No such certification is claimed. Native-library I/O is not comprehensively traced by this audit hook.
