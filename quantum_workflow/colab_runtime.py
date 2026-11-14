"""Colab setup with stdlib only; CUDA dependencies live in an isolated venv."""
import ctypes
from functools import lru_cache
from importlib import import_module
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import venv


def resolve_device(requested="auto"):
    if requested not in {"auto", "cpu", "gpu"}:
        raise ValueError("Device must be auto, cpu, or gpu")
    if requested == "cpu":
        return "cpu"
    executable = shutil.which("nvidia-smi")
    available = False
    if executable:
        probe = subprocess.run([executable, "--query-gpu=name", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=30)
        available = probe.returncode == 0 and bool(probe.stdout.strip())
    if available:
        return "gpu"
    if requested == "gpu":
        raise RuntimeError("GPU requested but no NVIDIA GPU is available. Select a GPU runtime in Colab.")
    return "cpu"


def cuda_extra():
    """Use the installed toolkit, not nvidia-smi's maximum driver capability."""
    executable = shutil.which("nvcc")
    if not executable:
        raise RuntimeError("GPU setup requires a CUDA 12 or 13 toolkit with nvcc on PATH")
    version = subprocess.run([executable, "--version"], check=True,
                             capture_output=True, text=True, timeout=30).stdout
    match = re.search(r"release\s+(12|13)\.", version)
    if not match:
        raise RuntimeError(f"Supported GPU toolkits are CUDA 12 and 13; nvcc reported: {version}")
    return "gpu-cuda" + match.group(1)


def prepare_environment(project, device, env_dir, *, with_tests=False):
    """Return an interpreter without replacing the active notebook's packages."""
    project, env_dir = Path(project).resolve(), Path(env_dir).resolve()
    extras = ["test"] if with_tests else []
    if device == "gpu":
        resolve_device("gpu")
        extras.append(cuda_extra())
        python = env_dir / "bin/python"
        if not python.is_file():
            # Colab's Debian Python omits ensurepip; let the host pip target
            # this interpreter, including installing pip in the new venv.
            venv.EnvBuilder(with_pip=False).create(env_dir)
    elif device == "cpu":
        python = Path(sys.executable)
    else:
        raise ValueError("Resolve auto to cpu or gpu before installing")
    requirement = str(project) + ("[" + ",".join(extras) + "]" if extras else "")
    command = [sys.executable, "-m", "pip"]
    if device == "gpu":
        command += ["--python", str(python)]
    command += ["install", "pip", "-e", requirement]
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in child.stdout:
        print(line, end="", flush=True)
    if child.wait():
        raise RuntimeError(f"Environment installation failed for {device}; see pip output above")
    return python


@lru_cache(maxsize=2)
def _preload_cutensor(cuda_major):
    # CuPy 14's cuda-pathfinder shortcut loads libcutensor and returns before
    # loading libcutensorMg, although its backend links against both libraries.
    # Keep handles alive and expose the symbols before GPU4PySCF imports it.
    dist = importlib.metadata.distribution(f"cutensor-cu{cuda_major}")
    return tuple(ctypes.CDLL(str(dist.locate_file("cutensor/lib/" + name)),
                             mode=ctypes.RTLD_GLOBAL)
                 for name in ("libcutensor.so.2", "libcutensorMg.so.2"))


def gpu_info():
    """Verify CUDA execution and cuTENSOR loading in the selected interpreter."""
    import cupy as cp
    major = cp.cuda.runtime.runtimeGetVersion() // 1000
    _preload_cutensor(major)
    import gpu4pyscf
    from gpu4pyscf.lib import cutensor

    device = cp.cuda.runtime.getDevice()
    props = cp.cuda.runtime.getDeviceProperties(device)
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode()
    if cutensor.cutensor is None:
        # Surface the loader error hidden by GPU4PySCF's optional-engine import.
        import_module("cupy_backends.cuda.libs.cutensor")
        import_module("cupyx.cutensor")
        raise RuntimeError("cuTENSOR did not load; check the pinned CuPy/cuTENSOR packages")
    vector = cp.arange(8, dtype=cp.float64)
    if float(cp.dot(vector, vector).get()) != 140.0:
        raise RuntimeError("CUDA numerical smoke check failed")
    cp.cuda.Stream.null.synchronize()
    return {
        "name": name, "device_id": device,
        "compute_capability": f"{props['major']}.{props['minor']}",
        "memory_bytes": int(props["totalGlobalMem"]),
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "cuda_driver": cp.cuda.runtime.driverGetVersion(),
        "cupy_version": cp.__version__, "gpu4pyscf_version": gpu4pyscf.__version__,
        "cutensor_version": importlib.metadata.version(f"cutensor-cu{major}"),
        "tensor_engine": "cutensor", "cuda_dot_check": 140.0,
    }


if __name__ == "__main__":
    print(json.dumps(gpu_info(), indent=2))
