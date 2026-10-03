"""
Check that PyTorch can see a CUDA GPU before training.

    uv run python scripts/check_gpu.py

Prints the torch version, whether CUDA is available, the device name and its compute
capability, and runs a small matrix multiply on the GPU. Exits with code 1 when CUDA
is not usable, so training never silently falls back to the CPU.
"""

import argparse
import sys


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    import torch

    print(f"torch.__version__        {torch.__version__}")
    print(f"torch.version.cuda       {torch.version.cuda}")
    print(f"torch.cuda.is_available  {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("CUDA is not available. Install an NVIDIA driver that supports CUDA 13.0 (580 or newer) "
              "and run `uv sync` so torch comes from the PyTorch cu130 index.")
        return 1
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"device                   {name} (sm_{major}{minor}, {total:.1f} GiB)")
    print(f"arch list                {' '.join(torch.cuda.get_arch_list())}")
    a = torch.randn(2048, 2048, device="cuda")
    b = (a @ a).sum().item()
    print(f"matmul on GPU            ok ({b:.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
