"""Run provenance: package versions, threads, timing; summary writing."""
from __future__ import annotations

import datetime as _dt
import importlib.metadata as md
import json
import os
import platform
import sys
from pathlib import Path

TRACKED_PACKAGES = ("numpy", "scipy", "pyscf", "openfermion", "pydantic", "gpu4pyscf-cuda12x")
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def environment() -> dict:
    versions = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = md.version(package)
        except md.PackageNotFoundError:
            continue
    from . import __version__

    return {
        "quantum_workflow": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "threads": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "utc_time": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }


def write_summary(output: Path, summary: dict) -> dict:
    """Write summary.json; runners call this last so its presence marks success."""
    summary.setdefault("warnings", [])
    summary["environment"] = environment()
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary
