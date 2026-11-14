import builtins
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pyscf import fci

from quantum_workflow.colab_runtime import cuda_extra, resolve_device
from quantum_workflow.workflow import load_config, run

ROOT = Path(__file__).resolve().parents[1]


def test_explicit_gpu_cannot_fall_back_to_cpu(monkeypatch):
    monkeypatch.setattr("quantum_workflow.colab_runtime.shutil.which", lambda _: None)
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cpu") == "cpu"
    with pytest.raises(RuntimeError, match="no NVIDIA GPU"):
        resolve_device("gpu")


@pytest.mark.parametrize("version,extra", [("12.5", "gpu-cuda12"), ("13.0", "gpu-cuda13")])
def test_gpu_packages_match_toolkit_version(monkeypatch, version, extra):
    monkeypatch.setattr("quantum_workflow.colab_runtime.shutil.which", lambda _: "/usr/local/cuda/bin/nvcc")
    monkeypatch.setattr("quantum_workflow.colab_runtime.subprocess.run", lambda *a, **kw:
                        SimpleNamespace(stdout=f"Cuda compilation tools, release {version}, V{version}.0"))
    assert cuda_extra() == extra


def test_missing_gpu_dependencies_do_not_produce_cpu_result(tmp_path, monkeypatch):
    original = builtins.__import__

    def missing_gpu(name, *args, **kwargs):
        if name == "gpu4pyscf":
            raise ImportError("GPU4PySCF deliberately unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_gpu)
    config = load_config(ROOT / "examples/h2-pyscf-gpu.json")
    with pytest.raises(RuntimeError, match="GPU requested"):
        run(config, tmp_path / "missing")
    assert not (tmp_path / "missing/summary.json").exists()


@pytest.mark.parametrize("method", ["hf", "dft"])
def test_gpu_scf_and_export_match_cpu_density_fit(tmp_path, method):
    if os.environ.get("COLABEXERCISE_DEVICE") != "gpu":
        pytest.skip("Executed by --device gpu on an actual Colab GPU runtime")
    config = load_config(ROOT / "examples/h2-pyscf-gpu.json")
    config.method = method
    gpu = run(config, tmp_path / "gpu")
    config.device = "cpu"
    cpu = run(config, tmp_path / "cpu")
    assert gpu["scf_device"] == "gpu" and gpu["scf_implementation"].startswith("gpu4pyscf.")
    assert gpu["gpu"]["tensor_engine"] == "cutensor"
    assert gpu["energy_hartree"] == pytest.approx(cpu["energy_hartree"], abs=1e-7)
    with np.load(tmp_path / "gpu/integrals.npz") as ints:
        energy, _ = fci.direct_spin1.kernel(
            ints["h1"], ints["eri"], gpu["num_spatial_orbitals"],
            (gpu["num_alpha"], gpu["num_beta"]), ecore=gpu["nuclear_repulsion_hartree"],
        )
    assert energy == pytest.approx(-1.1372838345, abs=1e-8)
