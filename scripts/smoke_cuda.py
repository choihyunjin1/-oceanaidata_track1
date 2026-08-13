"""Fail-fast validation of the pinned RTX 5090 PyTorch environment."""

from __future__ import annotations

import json

import torch


def main() -> None:
    if torch.__version__ != "2.13.0+cu130":
        raise RuntimeError(f"unexpected torch version: {torch.__version__}")
    if torch.version.cuda != "13.0" or not torch.cuda.is_available():
        raise RuntimeError("CUDA 13.0 runtime is not available")
    capability = torch.cuda.get_device_capability(0)
    if capability != (12, 0) or "sm_120" not in torch.cuda.get_arch_list():
        raise RuntimeError(f"RTX 5090 native architecture missing: {capability}")
    value = torch.randn(4096, 4096, device="cuda", dtype=torch.float16, requires_grad=True)
    loss = (value @ value.T).mean()
    loss.backward()
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(0),
                "capability": capability,
                "sm_120": True,
                "matmul_backward": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
