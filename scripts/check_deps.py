#!/usr/bin/env python3
"""
Dependency check script for DishSpace / robot training environment.
Run this before training to verify all required packages are available.

Usage:
    python scripts/check_deps.py
    python scripts/check_deps.py --install   # auto-install missing packages
"""

import importlib
import subprocess
import sys
import argparse

# (import_name, pip_package, required_for)
DEPENDENCIES = [
    # ML / Model
    ("torch",           "torch",                    "training"),
    ("transformers",    "transformers>=4.47.0",      "training"),
    ("peft",            "peft>=0.14.0",              "training (LoRA/DoRA adapters)"),
    ("accelerate",      "accelerate>=0.35.0",        "training"),
    ("diffusers",       "diffusers>=0.31.0",         "training"),
    # Vision / 3D
    ("open3d",          "open3d>=0.18.0",            "point cloud processing"),
    ("cv2",             "opencv-python>=4.10.0",     "image processing"),
    ("numpy",           "numpy>=1.26.0",             "data processing"),
    ("PIL",             "Pillow>=11.0.0",            "image processing"),
    ("scipy",           "scipy>=1.14.0",             "data processing"),
    # Simulation
    ("mujoco",          "mujoco>=3.2.0",             "simulation / MuJoCo"),
    # API
    ("fastapi",         "fastapi>=0.115.0",          "API server"),
    ("uvicorn",         "uvicorn[standard]>=0.32.0", "API server"),
    ("pydantic",        "pydantic>=2.10.0",          "API server"),
    # Data
    ("supabase",        "supabase>=2.3.0",           "dataset upload/download"),
    ("httpx",           "httpx>=0.28.0",             "HTTP client"),
    # ROS
    ("roslibpy",        "roslibpy>=1.7.0",           "ROS bridge"),
    # Utils
    ("dotenv",          "python-dotenv>=1.0.0",      "config"),
    ("structlog",       "structlog>=24.4.0",         "logging"),
    ("tenacity",        "tenacity>=9.0.0",           "retry logic"),
    # Modal (GPU extras)
    ("modal",           "modal>=0.62.0",             "modal GPU workers"),
]

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def check_torch_cuda():
    try:
        import torch
        cuda = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda else "N/A"
        version = torch.__version__
        print(f"  {GREEN}torch {version}{RESET}  |  CUDA: {'YES — ' + device_name if cuda else RED + 'NO (CPU only)' + RESET}")
        if not cuda:
            print(f"  {YELLOW}  WARNING: No CUDA GPU detected. Training will be very slow on CPU.{RESET}")
        return cuda
    except ImportError:
        return False


def check_mujoco():
    try:
        import mujoco
        print(f"  {GREEN}mujoco {mujoco.__version__}{RESET}")
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"  {YELLOW}mujoco installed but failed to load: {e}{RESET}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Check DishSpace dependencies")
    parser.add_argument("--install", action="store_true", help="pip install missing packages automatically")
    args = parser.parse_args()

    print(f"\n{BOLD}=== DishSpace Dependency Check ==={RESET}\n")

    ok = []
    missing = []

    for import_name, pip_pkg, reason in DEPENDENCIES:
        if import_name == "torch":
            # handled separately below
            continue
        if import_name == "mujoco":
            found = check_mujoco()
            if found:
                ok.append(import_name)
            else:
                missing.append((import_name, pip_pkg, reason))
            continue

        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "?")
            print(f"  {GREEN}✓ {import_name} {version}{RESET}  ({reason})")
            ok.append(import_name)
        except ImportError:
            print(f"  {RED}✗ {import_name}{RESET}  — MISSING  ({reason})")
            missing.append((import_name, pip_pkg, reason))

    print(f"\n{BOLD}--- PyTorch / CUDA ---{RESET}")
    cuda_ok = check_torch_cuda()
    if "torch" not in [m[0] for m in missing]:
        ok.append("torch")

    print(f"\n{BOLD}--- Summary ---{RESET}")
    print(f"  {GREEN}Installed:{RESET} {len(ok)}")
    print(f"  {RED}Missing:  {len(missing)}{RESET}")

    if missing:
        print(f"\n{BOLD}Missing packages:{RESET}")
        for import_name, pip_pkg, reason in missing:
            print(f"  pip install \"{pip_pkg}\"  # {reason}")

        if args.install:
            print(f"\n{BOLD}Installing missing packages...{RESET}\n")
            for _, pip_pkg, _ in missing:
                print(f"  Installing {pip_pkg} ...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", pip_pkg])
            print(f"\n{GREEN}Done. Re-run this script to confirm.{RESET}")
        else:
            print(f"\nTo install all missing packages at once:")
            pkgs = " ".join(f'"{p}"' for _, p, _ in missing)
            print(f"  pip install {pkgs}")
            print(f"\nOr re-run with --install to do it automatically:")
            print(f"  python scripts/check_deps.py --install")
            print(f"\nFor the full environment (recommended on RunPod):")
            print(f"  pip install -e '.[dev,gpu]'")
    else:
        print(f"\n{GREEN}{BOLD}All dependencies are installed!{RESET}")
        if not cuda_ok:
            print(f"{YELLOW}Note: CUDA GPU not detected — training will use CPU.{RESET}")

    print()
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
