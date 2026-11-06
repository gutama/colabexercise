"""Backend registry: config model -> runner.

A runner takes a validated config and an existing, empty output directory, writes
its files there, and returns the summary dict it wrote as ``summary.json`` (which it
must write last, so its presence marks a completed run).

To add a backend: define its config model in ``quantum_workflow/config.py`` (and add
it to ``Config``), implement ``run_<name>(config, output) -> dict`` in a module here,
and register it below. See docs/TEMPLATE.md.
"""
from ..config import PySCFConfig, QEConfig
from .pyscf_backend import run_pyscf
from .qe_backend import run_qe

RUNNERS = {
    PySCFConfig: run_pyscf,
    QEConfig: run_qe,
}
