"""Job runner that executes ON a Colab VM (sent by ``colab exec -f`` or ``colab run``).

It must stay self-contained -- standard library only at import time -- because it
runs before this project is installed on the VM. The local driver
(``quantum_workflow.colab``) uploads a job directory, then executes this file.

Why the sentinels: ``colab exec`` exits 0 on uncaught exceptions and on sys.exit(N),
so its exit status says nothing about success. This script prints ``__QW_JOB_OK__``
as its very last line only when everything succeeded, and the driver checks for it.
Under ``colab run`` the exit status does propagate, so failures also raise SystemExit(1).

Options come from environment variables (``colab exec`` cannot pass argv) and can be
overridden on the command line (``colab run`` forwards argv):

  QW_JOB_DIR   directory holding config.json and receiving results  [/content/qw-job]
  QW_CONFIG    config path, relative to the job dir (or repo)         [config.json]
  QW_INSTALL   pip requirement for the project, or "none"             [none]
  QW_REPO      optional git URL to clone into the job dir; QW_CONFIG is then relative
               to the clone, which is also installed when QW_INSTALL is "repo"
  QW_REF       branch/tag to clone                                    [main]
  QW_EXTRAS    extras to install with QW_INSTALL=repo                  [qubit]

Never put secrets in these: ``--env`` values are logged in plaintext by the CLI.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import traceback
from pathlib import Path

OK = "__QW_JOB_OK__"
FAILED = "__QW_JOB_FAILED__"
QE_VERSION = "7.6"
MICROMAMBA_URL = "https://micro.mamba.pm/api/micromamba/linux-64/latest"


def _stream(command: list[str], **kwargs) -> int:
    """Run a command and re-print its output from Python.

    Inside a Jupyter kernel, a child's raw stdout may not reach the client; printing
    line by line from Python always does.
    """
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, **kwargs)
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return process.wait()


def pip_install(requirements: list[str]) -> None:
    if shutil.which("uv"):
        # Colab's kernel runs the system Python (what `colab install` targets with --system);
        # inside a virtualenv, install into the interpreter running this script instead.
        in_venv = sys.prefix != sys.base_prefix
        target = ["--python", sys.executable] if in_venv else ["--system"]
        command = ["uv", "pip", "install", *target, *requirements]
    else:
        command = [sys.executable, "-m", "pip", "install", "-q", *requirements]
    if _stream(command) != 0:
        raise RuntimeError(f"Installation failed: {' '.join(requirements)}")


def ensure_qe(root: str | Path = "/content") -> Path:
    """Install conda-forge Quantum ESPRESSO with micromamba (idempotent); return its bin dir.

    Takes several minutes and ~1 GB the first time on a fresh runtime.
    """
    root = Path(root)
    env = root / "qe-env"
    if not (env / "bin" / "pw.x").exists():
        archive = root / "micromamba.tar.bz2"
        if _stream(["curl", "-fsSL", MICROMAMBA_URL, "-o", str(archive)]) != 0:
            raise RuntimeError("Could not download micromamba")
        mamba = root / "micromamba"
        with tarfile.open(archive, "r:bz2") as tar:
            source = tar.extractfile("bin/micromamba")
            if source is None:
                raise RuntimeError("micromamba archive is missing its executable")
            with source, mamba.open("wb") as target:
                shutil.copyfileobj(source, target)
        mamba.chmod(0o755)
        command = [str(mamba), "create", "-y", "-q", "-r", str(root / "mamba-root"),
                   "-p", str(env), "-c", "conda-forge", "--override-channels", f"qe={QE_VERSION}"]
        if _stream(command) != 0:
            raise RuntimeError("Quantum ESPRESSO installation failed")
    return env / "bin"


def ensure_gpu4pyscf() -> None:
    """Install GPU4PySCF for CUDA 12 if a GPU is present (see pyscf.org/user/gpu.html)."""
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("Config requests device='gpu' but this runtime has no NVIDIA GPU")
    try:
        import gpu4pyscf  # noqa: F401
    except ImportError:
        pip_install(["gpu4pyscf-cuda12x", "cutensor-cu12"])


def _options(argv: list[str]) -> argparse.Namespace:
    env = os.environ.get
    parser = argparse.ArgumentParser(description="Run a quantum-workflow job on this machine")
    parser.add_argument("--job-dir", default=env("QW_JOB_DIR", "/content/qw-job"))
    parser.add_argument("--config", default=env("QW_CONFIG", "config.json"))
    parser.add_argument("--install", default=env("QW_INSTALL", "none"))
    parser.add_argument("--repo", default=env("QW_REPO"))
    parser.add_argument("--ref", default=env("QW_REF", "main"))
    parser.add_argument("--extras", default=env("QW_EXTRAS", "qubit"))
    return parser.parse_args(argv)


def _script_argv() -> list[str]:
    # Under `colab exec` this file runs inside a Jupyter kernel whose argv belongs to
    # the kernel launcher; only `colab run` (or plain python) passes real arguments.
    if sys.argv and sys.argv[0].endswith("ipykernel_launcher.py"):
        return []
    return sys.argv[1:]


def _archive(results: Path, target: Path) -> None:
    def skip_scratch(info: tarfile.TarInfo):
        return None if "/scratch" in f"/{info.name}" else info

    with tarfile.open(target, "w:gz") as tar:
        tar.add(results, arcname="results", filter=skip_scratch)


def main(argv: list[str] | None = None) -> Path:
    options = _options(_script_argv() if argv is None else argv)
    job = Path(options.job_dir)
    try:
        job.mkdir(parents=True, exist_ok=True)
        base = job
        if options.repo:
            base = job / "repo"
            if not base.exists():
                command = ["git", "clone", "-q", "--depth", "1", "--branch", options.ref,
                           options.repo, str(base)]
                if _stream(command) != 0:
                    raise RuntimeError(f"git clone of {options.repo}@{options.ref} failed")
        if options.install == "repo":
            if not options.repo:
                raise ValueError("QW_INSTALL=repo needs QW_REPO")
            extras = f"[{options.extras}]" if options.extras else ""
            pip_install([f"{base}{extras}"])
        elif options.install != "none":
            pip_install([options.install])

        config_path = base / options.config
        config = json.loads(config_path.read_text())
        env = dict(os.environ)
        env.setdefault("OMP_NUM_THREADS", "1")
        if config.get("backend") == "qe":
            env["PATH"] = f"{ensure_qe(job.parent)}{os.pathsep}{env['PATH']}"
        if config.get("device") == "gpu":
            ensure_gpu4pyscf()

        results = job / "results"
        if results.exists():
            raise ValueError(f"{results} already exists; use a fresh job directory")
        command = [sys.executable, "-m", "quantum_workflow.cli", str(config_path),
                   "--output", str(results)]
        if _stream(command, env=env) != 0:
            raise RuntimeError("quantum-workflow reported a failure (see output above)")
        tarball = job / "results.tar.gz"
        _archive(results, tarball)
    except Exception as exc:  # report everything; the driver needs the sentinel
        traceback.print_exc()
        print(f"{FAILED} {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(1) from exc
    print(f"{OK} {tarball}", flush=True)
    return tarball


if __name__ == "__main__":
    main()
