#!/usr/bin/env python3
"""
Bootstraps a virtual environment for the SISA class-unlearning project.

Creates ./.venv, detects an NVIDIA GPU via nvidia-smi, installs the matching
CUDA build of PyTorch (falling back to CPU-only if no GPU/driver is found),
then installs the remaining requirements from requirements.txt.

Usage:
    python setup_env.py
"""
import os
import platform
import shutil
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(PROJECT_ROOT, ".venv")

# Tried newest-first; the script falls through to the next tag if a build
# isn't published yet, and to CPU-only if none of them install.
CUDA_WHEEL_TAGS = ["cu128", "cu126", "cu124", "cu121", "cu118"]


def run(cmd, **kwargs):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def venv_paths():
    if platform.system() == "Windows":
        return (
            os.path.join(VENV_DIR, "Scripts", "python.exe"),
            os.path.join(VENV_DIR, "Scripts", "pip.exe"),
        )
    return (
        os.path.join(VENV_DIR, "bin", "python"),
        os.path.join(VENV_DIR, "bin", "pip"),
    )


def create_venv():
    if os.path.isdir(VENV_DIR):
        print(f"Virtual environment already exists at {VENV_DIR}, reusing it.")
        return
    print(f"Creating virtual environment at {VENV_DIR} ...")
    run([sys.executable, "-m", "venv", VENV_DIR])


def detect_gpu():
    """Return True if an NVIDIA GPU + driver is usable via nvidia-smi."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False
    try:
        subprocess.run(
            [nvidia_smi],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False


def install_torch(pip_exe, has_gpu):
    if not has_gpu:
        print("No usable NVIDIA GPU detected (nvidia-smi missing or failed) - installing CPU-only PyTorch.")
        run([pip_exe, "install", "torch", "torchvision"])
        return

    print("NVIDIA GPU detected - installing a CUDA-enabled PyTorch build.")
    for tag in CUDA_WHEEL_TAGS:
        index_url = f"https://download.pytorch.org/whl/{tag}"
        print(f"Trying PyTorch build '{tag}' ...")
        try:
            run([pip_exe, "install", "torch", "torchvision", "--index-url", index_url])
            print(f"Installed PyTorch ({tag}).")
            return
        except subprocess.CalledProcessError:
            print(f"   '{tag}' unavailable, trying an older CUDA build...")
            continue

    print("Could not install any CUDA build of PyTorch - falling back to CPU-only.")
    run([pip_exe, "install", "torch", "torchvision"])


def install_requirements(pip_exe):
    req_file = os.path.join(PROJECT_ROOT, "requirements.txt")
    run([pip_exe, "install", "-r", req_file])


def verify_install(python_exe):
    try:
        result = subprocess.run(
            [python_exe, "-c", "import torch; print('PyTorch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"],
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print("Warning: could not verify the PyTorch install:", e.stderr)


def main():
    create_venv()
    python_exe, pip_exe = venv_paths()

    run([python_exe, "-m", "pip", "install", "--upgrade", "pip"])

    has_gpu = detect_gpu()
    install_torch(pip_exe, has_gpu)
    install_requirements(pip_exe)

    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    if platform.system() == "Windows":
        print(r"Activate with:   .venv\Scripts\activate")
    else:
        print("Activate with:   source .venv/bin/activate")

    verify_install(python_exe)


if __name__ == "__main__":
    main()
