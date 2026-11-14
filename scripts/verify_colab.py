"""Bootstrap and verify the CPU or GPU workflow on Colab.

Run with `colab run --timeout 1800 scripts/verify_colab.py` or directly in a VM.
An existing checkout can be supplied with --project-dir (including unpublished
changes uploaded to the VM); otherwise --ref selects a fresh GitHub checkout.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path("/content/colabexercise"))
    parser.add_argument("--ref", default="main", help="Branch/tag to clone if no checkout exists")
    parser.add_argument("--output", type=Path, help="New or empty results directory")
    parser.add_argument("--qe", action="store_true", help="Also install and run the optional QE CPU example")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu", help="PySCF SCF device")
    parser.add_argument("--skip-install", action="store_true", help="Use dependencies already installed")
    args = parser.parse_args()
    if not (3, 10) <= sys.version_info[:2] < (3, 14):
        parser.error("This project requires Python 3.10–3.13")

    project = args.project_dir.resolve()
    if not project.exists():
        subprocess.run([
            "git", "clone", "--depth", "1", "--branch", args.ref,
            "https://github.com/gutama/colabexercise.git", str(project),
        ], check=True)
    if not (project / "pyproject.toml").is_file():
        parser.error(f"No project checkout at {project}")
    sys.path.insert(0, str(project))
    if args.device == "gpu" and not args.skip_install:
        # A fresh interpreter avoids replacing NumPy/CuPy under Colab's kernel.
        from quantum_workflow.colab_runtime import prepare_environment
        python = prepare_environment(project, "gpu", project.parent / "colabexercise-gpu-env", with_tests=True)
        env = os.environ.copy()
        env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
        child = subprocess.Popen([str(python), str(project / "scripts/verify_colab.py"),
                                  *sys.argv[1:], "--skip-install"], env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in child.stdout:
            print(line, end="", flush=True)
        raise SystemExit(child.wait())
    output = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix="colab-results-"))
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"Results directory must be empty: {output}")

    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["COLABEXERCISE_OUTPUT"] = str(output)
    os.environ["COLABEXERCISE_RUN_QE"] = "1" if args.qe else "0"
    os.environ["COLABEXERCISE_DEVICE"] = args.device
    # Jupyter writes its connection files here on restricted/local machines too.
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(output / "jupyter-runtime"))
    if not args.skip_install:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", f"{project}[test]"], check=True)

    try:
        commit = subprocess.check_output(["git", "-C", str(project), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(project), "status", "--porcelain"], text=True).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = "uploaded-source", None
    # Include a content fingerprint when an uploaded snapshot has no .git data.
    source_hash = hashlib.sha256()
    for path in sorted(project.rglob("*")):
        relative = path.relative_to(project)
        if any(part.startswith(".") or part in {"__pycache__", "results"} or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.is_file() and path.suffix in {".py", ".toml", ".json", ".ipynb"}:
            source_hash.update(relative.as_posix().encode() + b"\0" + path.read_bytes())
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "scipy", "pyscf", "pandas", "pydantic", "nbclient", "pytest")}
    print(json.dumps({"python": platform.python_version(), "platform": platform.platform(),
                      "versions": versions, "source_commit": commit, "source_dirty": dirty}, indent=2), flush=True)
    with (output / "pytest.log").open("w") as log:
        result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rA"],
                                cwd=project, stdout=log, stderr=subprocess.STDOUT)
    print((output / "pytest.log").read_text(), flush=True)
    result.check_returncode()

    import nbformat
    from nbclient import NotebookClient

    completed = []
    for filename in ("colab.ipynb", "lihbn_reproduction.ipynb"):
        notebook = nbformat.read(project / "notebooks" / filename, as_version=4)
        nbformat.validate(notebook)
        client = NotebookClient(notebook, timeout=1200, kernel_name="python3",
                                extra_arguments=[
                                    "--IPKernelApp.kernel_class=ipykernel.ipkernel.IPythonKernel",
                                    "--IPKernelApp.extensions=[]",
                                    "--IPKernelApp.extra_extensions=[]",
                                ],
                                resources={"metadata": {"path": str(project)}})
        try:
            client.execute()
        finally:
            nbformat.write(notebook, output / filename)
        completed.append(filename)
        print(f"Executed every cell: {filename}", flush=True)

    # Reconsume the exported tensors with an independent exact-sector FCI solve.
    import numpy as np
    from pyscf import fci

    h2_file = next(output.glob("pyscf-*/summary.json"))
    h2 = json.loads(h2_file.read_text())
    with np.load(h2_file.parent / "integrals.npz") as ints:
        energy, _ = fci.direct_spin1.kernel(
            ints["h1"], ints["eri"], h2["num_spatial_orbitals"],
            (h2["num_alpha"], h2["num_beta"]), ecore=h2["nuclear_repulsion_hartree"],
        )
    cpu_reference = None
    if args.device == "gpu":
        from quantum_workflow.workflow import load_config, run
        reference_config = load_config(project / "examples/h2-pyscf.json")
        reference_config.density_fitting = True
        cpu_reference = run(reference_config, output / "h2-cpu-density-fit")
        if h2["scf_device"] != "gpu" or not h2["scf_implementation"].startswith("gpu4pyscf."):
            raise RuntimeError("GPU verification must use an actual GPU4PySCF SCF solver")
    expected_scf = cpu_reference["energy_hartree"] if cpu_reference else -1.1167593074
    if not np.isclose(h2["energy_hartree"], expected_scf, atol=1e-8, rtol=0) or not np.isclose(energy, -1.1372838345, atol=1e-8, rtol=0):
        raise RuntimeError("H2/STO-3G (0.74 Angstrom) RHF/FCI anchors failed")
    evidence = json.loads(next(output.glob("lihbn-*/evidence.json")).read_text())
    if evidence["evidence_tier"] != "synthetic_demo":
        raise RuntimeError("The unattended verification must run synthetic Li-hBN DEMO only")
    # Isolated GPU environments do not install the google.colab Python package.
    is_colab = bool(os.environ.get("COLAB_RELEASE_TAG"))
    if not is_colab:
        try:
            is_colab = Path("/content").is_dir() and importlib.util.find_spec("google.colab") is not None
        except ModuleNotFoundError:
            is_colab = False
    report = {
        "status": "passed", "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": "google_colab" if is_colab else "local",
        "python": platform.python_version(), "versions": versions,
        "source_commit": commit, "source_dirty": dirty, "source_sha256": source_hash.hexdigest(),
        "notebooks": completed, "pytest_log": "pytest.log",
        "h2_rhf_hartree": h2["energy_hartree"], "h2_fci_hartree": float(energy),
        "scf_device": h2["scf_device"], "gpu": h2["gpu"],
        "cpu_density_fit_reference_hartree": cpu_reference["energy_hartree"] if cpu_reference else None,
        "lihbn_evidence_tier": evidence["evidence_tier"],
        "lihbn_sector": evidence["sector"], "qe_executed": args.qe,
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    # Connection files can contain live kernel credentials; exclude them.
    shutil.rmtree(output / "jupyter-runtime", ignore_errors=True)
    archive = shutil.make_archive(str(output), "zip", output)
    print(json.dumps(report, indent=2), flush=True)
    print(f"Results archive: {archive}", flush=True)
    print("__COLABEXERCISE_OK__", flush=True)


if __name__ == "__main__":
    main()
