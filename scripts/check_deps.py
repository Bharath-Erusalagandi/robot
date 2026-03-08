#!/usr/bin/env python3
"""Comprehensive dependency preflight for the DishSpace workspace.

This script validates the full dependency surface used by the repo:
- core runtime packages from pyproject.toml
- optional groups: gpu, demo, dev
- critical symbols used by training / serving code
- CUDA availability for training
- MuJoCo import plus headless render smoke test

Usage:
    python scripts/check_deps.py
    python scripts/check_deps.py --profile train
    python scripts/check_deps.py --profile all --install
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import re
import subprocess
import sys
import tomllib
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

PROFILE_ALIASES = {
    "train": {"core", "train"},
    "api": {"core", "api"},
    "demo": {"core", "demo"},
    "dev": {"core", "dev"},
    "gpu": {"core", "gpu"},
    "all": {"core", "train", "api", "demo", "dev", "gpu"},
}


@dataclass(frozen=True)
class Check:
    import_name: str
    pip_spec: str
    purpose: str
    profiles: frozenset[str]
    symbols: tuple[str, ...] = ()
    runtime_check: Callable[[], tuple[bool, str]] | None = None
    required: bool = True
    runtime_required: bool = True


def _check_torch_runtime() -> tuple[bool, str]:
    import torch

    cuda = torch.cuda.is_available()
    version = getattr(torch, "__version__", "?")
    if cuda:
        device_name = torch.cuda.get_device_name(0)
        return True, f"torch {version} | CUDA: yes ({device_name})"
    return True, f"torch {version} | CUDA: no (CPU only)"


def _check_torchvision_runtime() -> tuple[bool, str]:
    try:
        import torch
        import torchvision
        from torchvision.ops import nms

        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0]], dtype=torch.float32)
        scores = torch.tensor([0.9, 0.8], dtype=torch.float32)
        keep = nms(boxes, scores, 0.5)
        return True, f"torchvision {torchvision.__version__} | nms ok ({len(keep)} kept)"
    except Exception as exc:
        cause = exc.__cause__ or exc.__context__
        detail = f"{type(exc).__name__}: {exc}"
        if cause is not None:
            detail += f" | cause: {type(cause).__name__}: {cause}"
        return False, f"torchvision runtime import failed: {detail}"


def _check_transformers_runtime() -> tuple[bool, str]:
    try:
        from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, Trainer, TrainingArguments

        model_class = "AutoModelForCausalLM"
        try:
            from transformers import AutoModelForVision2Seq  # type: ignore

            _ = AutoModelForVision2Seq
            model_class = "AutoModelForVision2Seq"
        except ImportError:
            _ = AutoModel

        _ = AutoProcessor
        _ = Trainer
        _ = TrainingArguments
        _ = AutoModelForCausalLM
        return True, f"transformers symbols ok | preferred model class: {model_class}"
    except Exception as exc:
        cause = exc.__cause__ or exc.__context__
        detail = f"{type(exc).__name__}: {exc}"
        if cause is not None:
            detail += f" | cause: {type(cause).__name__}: {cause}"
        return False, f"transformers runtime import failed: {detail}"


def _check_peft_runtime() -> tuple[bool, str]:
    try:
        from peft import LoraConfig, PeftModel, get_peft_model

        _ = LoraConfig
        _ = PeftModel
        _ = get_peft_model
        return True, "peft symbols ok"
    except Exception as exc:
        cause = exc.__cause__ or exc.__context__
        detail = f"{type(exc).__name__}: {exc}"
        if cause is not None:
            detail += f" | cause: {type(cause).__name__}: {cause}"
        return False, f"peft runtime import failed: {detail}"


def _run_mujoco_backend_probe(backend: str) -> tuple[bool, str]:
    probe = textwrap.dedent(
        '''
        import os
        os.environ["MUJOCO_GL"] = BACKEND
        os.environ["PYOPENGL_PLATFORM"] = BACKEND
        import mujoco

        xml = "<mujoco model=\"smoke\"><worldbody><light pos=\"0 0 2\"/><camera name=\"cam\" pos=\"0 -1 0.5\" xyaxes=\"1 0 0 0 0 1\"/><geom type=\"plane\" size=\"1 1 0.1\" rgba=\"0.8 0.8 0.8 1\"/><body pos=\"0 0 0.05\"><geom type=\"box\" size=\"0.05 0.05 0.05\" rgba=\"0.2 0.4 0.8 1\"/></body></worldbody></mujoco>"

        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, height=64, width=64)
        try:
            renderer.update_scene(data, camera="cam")
            rgb = renderer.render()
            renderer.enable_depth_rendering()
            depth = renderer.render()
            renderer.disable_depth_rendering()
            assert rgb.shape == (64, 64, 3), rgb.shape
            assert depth.shape == (64, 64), depth.shape
        finally:
            renderer.close()
        print(f"ok:{mujoco.__version__}")
        '''
    ).replace("BACKEND", repr(backend))

    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env={**os.environ, "MUJOCO_GL": backend, "PYOPENGL_PLATFORM": backend},
    )
    if result.returncode == 0:
        version = result.stdout.strip().split(":", 1)[-1] if result.stdout.strip() else "?"
        return True, f"mujoco {version} | {backend} render smoke test passed"

    error_text = (result.stderr or result.stdout).strip().splitlines()
    detail = error_text[-1] if error_text else "unknown MuJoCo backend failure"
    return False, detail


def _check_mujoco_runtime() -> tuple[bool, str]:
    import mujoco

    for backend in ("egl", "osmesa"):
        ok, detail = _run_mujoco_backend_probe(backend)
        if ok:
            if backend == "osmesa":
                return True, f"mujoco {mujoco.__version__} | osmesa render smoke test passed (set MUJOCO_GL=osmesa for CPU rendering)"
            return True, detail

    return (
        False,
        "MuJoCo is installed but headless rendering failed for egl and osmesa. "
        "Install system GL libs with: apt-get update && apt-get install -y libegl1 libegl-mesa0 libgl1 libgl1-mesa-dri libgles2 libglfw3 libglfw3-dev libglew2.2 libgbm1 libdrm2 libglvnd0 libglx-mesa0 libosmesa6 libosmesa6-dev libglib2.0-0 libglvnd-dev mesa-utils libxrender1 libxext6 libsm6",
    )


CHECKS: tuple[Check, ...] = (
    Check("torch", "torch>=2.5.0", "training runtime", frozenset({"core", "train", "gpu"}), runtime_check=_check_torch_runtime),
    Check("torchvision", "torchvision>=0.20.0", "vision ops required by transformers image stack", frozenset({"core", "train", "gpu"}), runtime_check=_check_torchvision_runtime),
    Check("transformers", "transformers>=4.47.0", "model loading / trainer", frozenset({"core", "train", "gpu"}), runtime_check=_check_transformers_runtime),
    Check("peft", "peft>=0.14.0", "LoRA / DoRA adapters", frozenset({"core", "train", "gpu"}), runtime_check=_check_peft_runtime),
    Check("accelerate", "accelerate>=0.35.0", "device_map / distributed helpers", frozenset({"core", "train", "gpu"})),
    Check("diffusers", "diffusers>=0.31.0", "modal image / model stack", frozenset({"core", "train", "gpu"})),
    Check("numpy", "numpy>=1.26.0", "numerics", frozenset({"core", "train", "api", "demo", "dev", "gpu"})),
    Check("cv2", "opencv-python>=4.10.0", "image processing", frozenset({"core", "train", "demo", "gpu"})),
    Check("PIL", "Pillow>=11.0.0", "image processing", frozenset({"core", "train", "demo", "gpu"})),
    Check("scipy", "scipy>=1.14.0", "data processing", frozenset({"core", "train", "gpu"})),
    Check("open3d", "open3d>=0.18.0", "point cloud processing", frozenset({"core", "train", "gpu"})),
    Check(
        "mujoco",
        "mujoco>=3.2.0",
        "simulation / synthetic rendering",
        frozenset({"core", "train", "gpu"}),
        runtime_check=_check_mujoco_runtime,
        runtime_required=False,
    ),
    Check("fastapi", "fastapi>=0.115.0", "API server", frozenset({"core", "api"}), symbols=("FastAPI", "HTTPException")),
    Check("uvicorn", "uvicorn[standard]>=0.32.0", "API serving", frozenset({"core", "api"}), symbols=("run",)),
    Check("pydantic", "pydantic>=2.10.0", "schemas", frozenset({"core", "api"}), symbols=("BaseModel", "Field")),
    Check("multipart", "python-multipart>=0.0.18", "FastAPI multipart uploads", frozenset({"core", "api"})),
    Check("supabase", "supabase>=2.3.0", "dataset storage", frozenset({"core", "api"}), symbols=("create_client",)),
    Check("httpx", "httpx>=0.28.0", "HTTP client", frozenset({"core", "api", "demo", "dev"}), symbols=("Client", "AsyncClient")),
    Check("roslibpy", "roslibpy>=1.7.0", "ROS bridge", frozenset({"core", "api"}), symbols=("Ros", "Topic", "Message")),
    Check("dotenv", "python-dotenv>=1.0.0", "config loading", frozenset({"core", "api", "train"}), symbols=("load_dotenv",)),
    Check("structlog", "structlog>=24.4.0", "logging", frozenset({"core", "train", "api", "gpu"})),
    Check("tenacity", "tenacity>=9.0.0", "retry logic", frozenset({"core"})),
    Check("modal", "modal>=0.62.0", "Modal worker", frozenset({"gpu"}), symbols=("App", "Image", "Volume"), required=False),
    Check("streamlit", "streamlit>=1.31.0", "demo UI", frozenset({"demo"}), required=False),
    Check("plotly", "plotly>=5.18.0", "demo charts", frozenset({"demo"}), required=False),
    Check("streamlit_webrtc", "streamlit-webrtc>=0.47.0", "demo webcam streaming", frozenset({"demo"}), required=False),
    Check("pytest", "pytest>=8.0.0", "test runner", frozenset({"dev"}), required=False),
    Check("pytest_asyncio", "pytest-asyncio>=0.23.0", "async test support", frozenset({"dev"}), required=False),
    Check("anyio", "anyio[trio]>=4.0.0", "async test backend", frozenset({"dev"}), required=False),
    Check("pytest_cov", "pytest-cov>=4.1.0", "coverage plugin", frozenset({"dev"}), required=False),
    Check("ruff", "ruff>=0.3.0", "linting", frozenset({"dev"}), required=False),
    Check("mypy", "mypy>=1.8.0", "type checking", frozenset({"dev"}), required=False),
)


def normalize_requirement_name(requirement: str) -> str:
    requirement = requirement.strip()
    if not requirement:
        return requirement
    requirement = re.split(r"[<>=!~; ]", requirement, maxsplit=1)[0]
    requirement = requirement.split("[", 1)[0]
    return requirement.lower().replace("_", "-")


def declared_requirements() -> dict[str, set[str]]:
    with PYPROJECT_PATH.open("rb") as handle:
        data = tomllib.load(handle)

    project = data.get("project", {})
    result: dict[str, set[str]] = {
        "core": {normalize_requirement_name(req) for req in project.get("dependencies", [])},
    }
    for group, reqs in project.get("optional-dependencies", {}).items():
        result[group] = {normalize_requirement_name(req) for req in reqs}
    return result


def check_manifest_coverage() -> list[str]:
    declared = declared_requirements()
    covered = {normalize_requirement_name(check.pip_spec) for check in CHECKS}
    gaps: list[str] = []
    for group, reqs in declared.items():
        for req in sorted(reqs):
            if req not in covered:
                gaps.append(f"{group}:{req}")
    return gaps


def selected_profiles(profile: str) -> set[str]:
    return PROFILE_ALIASES[profile]


def selected_checks(profile: str) -> list[Check]:
    wanted = selected_profiles(profile)
    return [check for check in CHECKS if check.profiles & wanted]


def module_version(import_name: str, pip_spec: str) -> str:
    try:
        return importlib.metadata.version(normalize_requirement_name(pip_spec))
    except importlib.metadata.PackageNotFoundError:
        try:
            module = importlib.import_module(import_name)
            return getattr(module, "__version__", "?")
        except Exception:
            return "?"


def verify_symbols(module: object, symbols: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    for symbol in symbols:
        if not hasattr(module, symbol):
            missing.append(symbol)
    return missing


def install_missing(checks: list[Check]) -> list[tuple[Check, str]]:
    failures: list[tuple[Check, str]] = []
    seen: set[str] = set()

    for check in checks:
        if check.pip_spec in seen:
            continue
        seen.add(check.pip_spec)
        print(f"  Installing {check.pip_spec} ...")
        first = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--ignore-installed", check.pip_spec],
            capture_output=True,
            text=True,
        )
        if first.returncode == 0:
            print(f"  {GREEN}Installed {check.pip_spec}{RESET}")
            continue

        second = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--break-system-packages", check.pip_spec],
            capture_output=True,
            text=True,
        )
        if second.returncode == 0:
            print(f"  {GREEN}Installed {check.pip_spec} (via --break-system-packages){RESET}")
            continue

        stderr = second.stderr.strip() or first.stderr.strip() or "unknown pip error"
        failures.append((check, stderr))
        print(f"  {RED}Failed to install {check.pip_spec}: {stderr}{RESET}")

    return failures


def print_python_summary() -> None:
    version = sys.version.split()[0]
    print(f"{BOLD}Python{RESET}")
    print(f"  Executable: {sys.executable}")
    print(f"  Version:    {version}")
    if sys.version_info[:2] != (3, 11):
        print(f"  {YELLOW}Recommended on RunPod: Python 3.11. Current environment may still work, but 3.11 is the target.{RESET}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Comprehensive DishSpace dependency preflight")
    parser.add_argument("--profile", choices=sorted(PROFILE_ALIASES), default="all", help="Dependency profile to validate")
    parser.add_argument("--install", action="store_true", help="Install missing packages automatically")
    args = parser.parse_args()

    print(f"\n{BOLD}=== DishSpace Dependency Preflight ==={RESET}\n")
    print_python_summary()

    coverage_gaps = check_manifest_coverage()
    if coverage_gaps:
        print(f"{RED}Checker manifest is stale. Unmapped requirements found in pyproject.toml:{RESET}")
        for gap in coverage_gaps:
            print(f"  {gap}")
        print()
        return 2

    checks = selected_checks(args.profile)
    failures: list[tuple[Check, str]] = []
    warnings: list[str] = []

    print(f"{BOLD}Profile{RESET}")
    print(f"  Requested: {args.profile}")
    print(f"  Expanded:  {', '.join(sorted(selected_profiles(args.profile)))}")
    print()

    for check in checks:
        label = normalize_requirement_name(check.pip_spec)
        try:
            module = importlib.import_module(check.import_name)
            missing_symbols = verify_symbols(module, check.symbols)
            if missing_symbols:
                failures.append((check, f"missing symbols: {', '.join(missing_symbols)}"))
                print(f"  {RED}✗ {check.import_name}{RESET}  missing symbols: {', '.join(missing_symbols)}")
                continue

            version = module_version(check.import_name, check.pip_spec)
            print(f"  {GREEN}✓ {check.import_name} {version}{RESET}  ({check.purpose})")

            if check.runtime_check is not None:
                ok, detail = check.runtime_check()
                color = GREEN if ok else (RED if check.runtime_required else YELLOW)
                print(f"    {color}{detail}{RESET}")
                if not ok:
                    if check.runtime_required:
                        failures.append((check, detail))
                    else:
                        warnings.append(f"{check.import_name}: {detail}")
                elif "CUDA: no" in detail:
                    warnings.append(detail)
        except ImportError as exc:
            message = f"missing import: {exc}"
            if check.required:
                failures.append((check, message))
                print(f"  {RED}✗ {check.import_name}{RESET}  {message}")
            else:
                warnings.append(f"{label}: {message}")
                print(f"  {YELLOW}△ {check.import_name}{RESET}  {message}")
        except Exception as exc:
            failures.append((check, f"runtime failure: {exc}"))
            print(f"  {RED}✗ {check.import_name}{RESET}  runtime failure: {exc}")

    print(f"\n{BOLD}Summary{RESET}")
    print(f"  Checks run: {len(checks)}")
    print(f"  {RED}Failures:{RESET} {len(failures)}")
    print(f"  {YELLOW}Warnings:{RESET} {len(warnings)}")

    if warnings:
        print(f"\n{BOLD}Warnings{RESET}")
        for warning in warnings:
            print(f"  {warning}")

    if failures and args.install:
        print(f"\n{BOLD}Installing missing packages...{RESET}")
        install_targets = [check for check, reason in failures if reason.startswith("missing import:")]
        install_failures = install_missing(install_targets)
        if install_failures:
            print(f"\n{RED}Auto-install failures{RESET}")
            for check, error in install_failures:
                print(f"  {check.pip_spec}: {error}")
            print("\nRecommended fallback:")
            print("  python -m venv /workspace/venv && source /workspace/venv/bin/activate")
            print("  pip install -e '.[dev,gpu,demo]'")
            return 1

        print(f"\n{GREEN}Install step finished. Re-run the script to verify the environment.{RESET}")
        return 1

    if failures:
        print(f"\n{BOLD}Failures{RESET}")
        for check, reason in failures:
            if reason.startswith("missing import:"):
                print(f"  pip install \"{check.pip_spec}\"  # {check.purpose}")
            else:
                print(f"  {check.import_name}: runtime fix required  # {check.purpose}")
            print(f"    reason: {reason}")
        print("\nRecommended full install for this repo:")
        print("  pip install -e '.[dev,gpu,demo]'")
        return 1

    print(f"\n{GREEN}{BOLD}Environment passed preflight.{RESET}")
    print("Run training with:")
    print("  bash scripts/runpod_train.sh --samples 5000 --epochs 5 --include-failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
