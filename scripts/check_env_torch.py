from __future__ import annotations

import os
import platform
import sys
import traceback


def main() -> int:
    print("=== Environment Info ===")
    print("sys.executable:", sys.executable)
    print("python:", sys.version.replace("\n", " "))
    print("platform:", platform.platform())
    print("CONDA_DEFAULT_ENV:", os.environ.get("CONDA_DEFAULT_ENV", "(not set)"))

    print("\n=== PyTorch Import Check ===")
    try:
        import torch  # noqa: F401

        print("torch imported OK")
        print("torch.__version__:", torch.__version__)
        return 0
    except Exception as e:
        print("FAILED to import torch")
        print("Exception:", repr(e))
        print("\n--- Full traceback ---")
        traceback.print_exc()

        print("\nSuggested fix (conda, macOS CPU):")
        print("  conda activate mri-seg")
        print("  conda install -c pytorch pytorch torchvision")
        print("\nIf still failing, rebuild the environment:")
        print("  conda deactivate")
        print("  conda env remove -n mri-seg")
        print("  conda env create -f environment.yml")
        print("  conda activate mri-seg")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())