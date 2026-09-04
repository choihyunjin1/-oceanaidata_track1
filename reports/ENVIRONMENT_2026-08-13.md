# P1 execution environment — 2026-08-13 KST

This is the verified local training environment. It contains no source observations,
credentials, predictions, or personal absolute paths.

## Runtime

- OS: Windows 11 Pro x64, build 26200
- CPU: AMD Ryzen 7 7800X3D, 8 logical processors exposed to the process
- RAM: 63.11 GiB
- Python: CPython 3.12.10 x64 (`MSC v.1943`)
- Environment: repository-local `.venv-p1` (ignored by Git)
- Package lock: `requirements-lock.txt`
- Lock SHA-256: `a1aa280b03af38c1920a8a171da1fdb568b8310edf3bfea66fd8ab2f71c470ba`

## GPU smoke result

- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB
- Driver: 610.88
- Compute capability: 12.0
- PyTorch: `2.13.0+cu130`
- PyTorch CUDA runtime: 13.0
- Native `sm_120` present: yes
- Verified operations: CUDA availability, 4096 x 4096 matrix multiplication,
  synchronization, and backward propagation

The PyTorch wheel is pinned by its official URL and SHA-256 in both
`requirements-dl.txt` and `requirements-lock.txt`. A local CUDA Toolkit or `nvcc`
is not required to consume this prebuilt wheel.

## Determinism and resource policy

- Primary tree baseline: deterministic CPU LightGBM with row-wise execution
- CPU model threads: 8; optimizer-level parallel trial scheduling: 1
- Deep learning: fixed seeds, bfloat16 where supported, checkpoint provenance
- GPU models remain experimental because some GPU kernels are not bitwise deterministic
- Every candidate must still reproduce its submission rows from a stored model before use

## Verification commands

~~~powershell
.venv-p1\Scripts\python.exe -m pip check
.venv-p1\Scripts\python.exe scripts\smoke_cuda.py
.venv-p1\Scripts\python.exe -m pytest -q
~~~
