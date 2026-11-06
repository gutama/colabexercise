"""Run a config on a Google Colab VM through the ``colab`` CLI (google-colab-cli).

    quantum-workflow-colab examples/lih-pyscf-qubit.json --output results/lih [--gpu T4]

The driver validates the config locally *before* renting anything, then:
``colab new`` -> upload the job (config, pseudopotentials, a wheel of this project)
-> ``colab exec -f colab_remote.py`` -> check for the success sentinel -> download
and safely extract the results -> ``colab stop`` (in a ``finally``, so Ctrl-C and
errors still release the VM unless ``--keep`` is given).

It encodes the CLI's traps (see docs/colab.md): ``colab exec`` exits 0 on failure,
``--timeout`` defaults to 30 s, reusing a session name orphans a billing VM, file
commands resolve paths from ``/`` rather than ``/content``, uploads need an existing
parent directory, and ``--env`` values are logged in plaintext.
"""
from __future__ import annotations

import argparse
import json
import secrets
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

from . import colab_remote, distribution_name
from .config import QEConfig
from .workflow import load_config

GPUS = ("T4", "L4", "G4", "H100", "A100")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ColabError(RuntimeError):
    pass


class Colab:
    """Thin wrapper over the CLI; every call has stdin closed so auth can never block."""

    def __init__(self, binary: str = "colab", dry_run: bool = False, echo=print):
        self.binary = binary
        self.dry_run = dry_run
        self.echo = echo

    def __call__(self, *args: str, input: str | None = None, stream: bool = False,
                 check: bool = True) -> str:
        command = [self.binary, *args]
        self.echo("$ " + shlex.join(command))
        if self.dry_run:
            return ""
        if stream:
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            if input is not None:
                process.stdin.write(input)
                process.stdin.close()
            lines = []
            for line in process.stdout:
                print(line, end="", flush=True)
                lines.append(line)
            code = process.wait()
            output = "".join(lines)
        else:
            completed = subprocess.run(
                command, input=input if input is not None else None,
                stdin=None if input is not None else subprocess.DEVNULL,
                capture_output=True, text=True,
            )
            code, output = completed.returncode, completed.stdout + completed.stderr
        if check and code != 0:
            raise ColabError(f"{' '.join(command)} exited {code}:\n{output[-2000:]}")
        return output


def build_wheel(destination: Path) -> Path:
    """Build a wheel of the local source tree (uncommitted changes included)."""
    if not (PROJECT_ROOT / "pyproject.toml").is_file():
        raise ColabError("Not running from a source checkout; pass --source git+URL@REF")
    if shutil.which("uv"):
        command = ["uv", "build", "--wheel", "--out-dir", str(destination), str(PROJECT_ROOT)]
    else:
        command = [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(destination),
                   str(PROJECT_ROOT)]
    completed = subprocess.run(command, capture_output=True, text=True)
    wheels = sorted(destination.glob("*.whl"))
    if completed.returncode != 0 or not wheels:
        raise ColabError(f"Wheel build failed:\n{completed.stdout}{completed.stderr}")
    return wheels[-1]


def stage_job(config_path: Path, staging: Path, source: str, extras: str) -> tuple[list[Path], str]:
    """Write the job files to ``staging``; return (files to upload, install requirement)."""
    config = load_config(config_path)
    data = json.loads(config_path.read_text())
    files = []
    if isinstance(config, QEConfig):
        for element, pseudo in config.pseudopotentials.items():
            staged = staging / f"{element}.{pseudo.path.name}"
            shutil.copyfile(pseudo.path, staged)
            data["pseudopotentials"][element]["path"] = staged.name
            files.append(staged)
    (staging / "config.json").write_text(json.dumps(data, indent=2) + "\n")
    files.insert(0, staging / "config.json")
    name = f"{distribution_name()}[{extras}]" if extras else distribution_name()
    if source == "local":
        wheel = build_wheel(staging)
        files.append(wheel)
        requirement = f"{name} @ file://{{job_dir}}/{wheel.name}"
    elif source == "preinstalled":
        requirement = "none"
    elif source.startswith("git+"):
        requirement = f"{name} @ {source}"
    else:
        raise ColabError("--source must be 'local', 'preinstalled' or a git+https://...@ref URL")
    return files, requirement


def safe_extract(tarball: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(tarball, "r:gz") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if not (member.isfile() or member.isdir()) or not target.is_relative_to(destination):
                raise ColabError(f"Unsafe entry in results archive: {member.name}")
        if hasattr(tarfile, "data_filter"):
            tar.extractall(destination, filter="data")
        else:  # Python < 3.10.12 / 3.11.4: members were checked above
            tar.extractall(destination)


def submit(config_path: str | Path, output_dir: str | Path, *, gpu: str | None = None,
           high_mem: bool = False, session: str | None = None, source: str = "local",
           extras: str = "qubit", timeout: float = 3600, keep: bool = False,
           colab_bin: str = "colab", dry_run: bool = False) -> Path | None:
    config_path = Path(config_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ColabError(f"Output directory must be new or empty: {output}")
    config = load_config(config_path)  # fail on a bad config before renting a VM
    device = getattr(config, "device", "cpu")
    if gpu is not None and gpu not in GPUS:
        # The CLI silently turns an unknown GPU name into an A100.
        raise ColabError(f"Unknown GPU {gpu!r}; choose from {', '.join(GPUS)}")
    if device == "gpu" and gpu is None:
        raise ColabError("Config sets device='gpu'; pass --gpu (e.g. --gpu T4)")
    if gpu is not None and device != "gpu":
        print(f"warning: renting a {gpu} GPU for a CPU-only config", file=sys.stderr)

    session = session or f"qw-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"
    job_dir = f"/content/{session}"
    colab = Colab(colab_bin, dry_run=dry_run)

    listing = colab("sessions", check=False) if not dry_run else ""
    if not dry_run and ("accounts.google.com" in listing or "Traceback" in listing):
        raise ColabError(
            "The colab CLI is not authenticated. Run `colab sessions` once in a terminal "
            "and complete the Google sign-in, then retry."
        )
    if session in listing.split():
        raise ColabError(f"Session {session!r} already exists; reusing a name orphans its VM")

    with tempfile.TemporaryDirectory(prefix="qw-colab-") as tmp:
        staging = Path(tmp)
        files, requirement = stage_job(config_path, staging, source, extras)
        requirement = requirement.replace("{job_dir}", job_dir)
        shape = (["--gpu", gpu] if gpu else []) + (["--high-mem"] if high_mem else [])
        try:
            # Inside the try: an allocation that times out client-side may still exist.
            colab("new", "-s", session, *shape)
            env = ["--env", f"QW_JOB_DIR={job_dir}"]
            mkdir = ("import os; os.makedirs(os.environ['QW_JOB_DIR'], exist_ok=True); "
                     "print('__QW_DIR_OK__')")
            out = colab("exec", "-s", session, "--timeout", "120", *env, input=mkdir)
            if not dry_run and "__QW_DIR_OK__" not in out:
                raise ColabError(f"Could not create {job_dir} on the VM:\n{out[-2000:]}")
            for path in files:
                colab("upload", "-s", session, str(path), f"{job_dir}/{path.name}")
            env += ["--env", f"QW_INSTALL={requirement}", "--env", "QW_CONFIG=config.json"]
            out = colab("exec", "-s", session, "-f", colab_remote.__file__,
                        "--timeout", str(timeout), *env, stream=True)
            if dry_run:
                return None
            succeeded = any(line.startswith(colab_remote.OK) for line in out.splitlines())
            if not succeeded or colab_remote.FAILED in out:
                raise ColabError("Remote job failed (output above); the VM is being released")
            local_tarball = staging / "results.tar.gz"
            colab("download", "-s", session, f"{job_dir}/results.tar.gz", str(local_tarball))
            extracted = staging / "extracted"
            extracted.mkdir()
            safe_extract(local_tarball, extracted)
            output.mkdir(parents=True, exist_ok=True)
            for item in (extracted / "results").iterdir():
                shutil.move(str(item), output / item.name)
        finally:
            if keep:
                print(f"Session kept: {session}. Release it with: colab stop -s {session}",
                      file=sys.stderr)
            else:
                colab("stop", "-s", session, check=False)
                if not dry_run:
                    print("Released. Confirm nothing is still billing with: colab sessions",
                          file=sys.stderr)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a quantum-workflow config on Google Colab")
    parser.add_argument("config", help="JSON config (validated locally first)")
    parser.add_argument("--output", required=True, help="New or empty local results directory")
    parser.add_argument("--gpu", choices=GPUS, help="Rent a GPU runtime (default: CPU)")
    parser.add_argument("--high-mem", action="store_true", help="High-RAM shape (Colab Pro)")
    parser.add_argument("--session", help="Session name (default: unique qw-<time>-<hex>)")
    parser.add_argument("--source", default="local",
                        help="'local' (wheel of this checkout), 'preinstalled', or git+URL@REF")
    parser.add_argument("--extras", default="qubit", help="Package extras to install remotely")
    parser.add_argument("--timeout", type=float, default=3600,
                        help="Seconds for the remote job (colab's default is only 30)")
    parser.add_argument("--keep", action="store_true", help="Do not stop the VM afterwards")
    parser.add_argument("--colab-bin", default="colab", help="Path to the colab executable")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the colab commands without running them")
    args = parser.parse_args(argv)
    try:
        result = submit(args.config, args.output, gpu=args.gpu, high_mem=args.high_mem,
                        session=args.session, source=args.source, extras=args.extras,
                        timeout=args.timeout, keep=args.keep, colab_bin=args.colab_bin,
                        dry_run=args.dry_run)
    except (ColabError, ValueError, OSError) as exc:
        print(f"Colab job failed: {exc}", file=sys.stderr)
        return 1
    if result is not None:
        print(f"Results: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
