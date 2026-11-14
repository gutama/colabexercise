# Colab molecular simulation workflow

Run small-molecule Hartree–Fock or DFT with PySCF, export a molecular
Hamiltonian for downstream VQE/QSCI, or run a separate plane-wave DFT
calculation with Quantum ESPRESSO (QE). Control local runs through the CLI or
an MCP stdio server; use the [Colab notebook](notebooks/colab.ipynb) for
interactive cloud runs.

[Open molecular workflow in Colab](https://colab.research.google.com/github/gutama/colabexercise/blob/main/notebooks/colab.ipynb)
· [Open synthetic Li-hBN demo in Colab](https://colab.research.google.com/github/gutama/colabexercise/blob/main/notebooks/lihbn_reproduction.ipynb)

Select **L4 GPU** (or A100) under **Runtime → Change runtime type**,
then **Runtime → Run all**. The molecular notebook automatically selects
GPU4PySCF on GPU runtimes and CPU on CPU runtimes, installs the package, and
runs H₂/STO-3G. Set `COLABEXERCISE_DEVICE=gpu` to require GPU execution, or
`cpu` to select CPU explicitly. Set `RUN_QE=True` in its optional QE
cell to install QE 7.6 and run the separate plane-wave example. Browser downloads
are opt-in with `DOWNLOAD_RESULTS=True`. The Li-hBN notebook defaults to
synthetic data; original private materials are needed for its REAL mode.

## Run remotely from a terminal or agent

Install the [Google Colab CLI](https://github.com/googlecolab/colab-cli) on
Linux/macOS with Python 3.12 or newer. The runtime project supports Python
3.10–3.13. Complete the initial Google sign-in before sending any code:

```bash
uv tool install google-colab-cli
colab sessions
colab run --timeout 1800 scripts/verify_colab.py --ref main --output /content/verified
```

This allocates a CPU Colab VM, clones the selected published ref, installs
`.[test]`, runs the complete test suite and both notebooks, and checks the
exported H₂ integrals against RHF/FCI reference energies at **0.74 Å**. It prints
`__COLABEXERCISE_OK__` only after success. `colab run` propagates failures and
releases its VM automatically. The report labels Li-hBN output `synthetic_demo`.
Add `--qe` after the script path to include the optional QE CPU calculation.
Use the branch name in place of `main` when reviewing unpublished-to-main code.

For GPU execution, allocate a GPU **and** select GPU in the verification script:

```bash
colab run --gpu L4 --timeout 1800 scripts/verify_colab.py --ref codex/colab-runtime --device gpu --output /content/verified-gpu
```

The GPU setup detects the actual `nvcc` toolkit version and installs the
matching `gpu-cuda12` or `gpu-cuda13` project extra in an isolated environment
at `/content/colabexercise-gpu-env`. Each extra pins GPU4PySCF, CuPy, cuTENSOR
and GPU libxc together; the active notebook's NumPy/CuPy packages are not
replaced. CUDA 12 uses the upstream-recommended CuPy 13.4.1/cuTENSOR 2.2.0
pair; CUDA 13 uses CuPy 14.0.1/cuTENSOR 2.4.1, matching that CuPy wheel's build.
A real CUDA dot product and
cuTENSOR load check run before SCF. Explicit GPU requests fail if GPU setup
is unavailable.

On Colab's current CUDA 13 image, the GPU4PySCF wheel contains kernels for
compute capability 8.0 and newer. Use **L4 or A100**; a T4 (7.5) is rejected
before installation. T4 use requires a CUDA 12 runtime and its matching extra.

GPU verification exercises both HF and PBE DFT against CPU **density-fitting**
references, and independently solves FCI using the exported H₂ integrals.
The summary records the GPU model, CUDA and package versions, SCF solver
class and device, and the CPU integral-export stage. To run another GPU
calculation after setup:

```bash
/content/colabexercise-gpu-env/bin/python -m quantum_workflow.cli examples/h2-pyscf-gpu.json --output /content/my-gpu-results
```

GPU acceleration applies to PySCF SCF/DFT. The QE installation remains a CPU
build, and the synthetic Li-hBN NumPy demo and full molecular-integral export
remain CPU calculations. The small H₂ check validates correctness; it does
not establish a speedup for such a small molecule.

To retrieve executed notebooks, integrals, evidence JSON and logs, keep a
uniquely named runtime until download finishes:

```bash
colab sessions  # Check that the name below is not already in use.
colab run -s chemistry-unique --keep --timeout 1800 scripts/verify_colab.py --ref main --output /content/verified
colab download -s chemistry-unique /content/verified.zip ./verified.zip
colab stop -s chemistry-unique
colab sessions
```

Stop the retained runtime even if execution or download fails. Colab files are
temporary. `verification.json` records versions, source revision/fingerprint,
energies, notebook completion and the synthetic evidence tier; `pytest.log`
records individual tests. For a source snapshot already uploaded to the VM,
pass `--project-dir /content/colabexercise` instead of cloning. The notebooks
also accept `COLABEXERCISE_REF` and `COLABEXERCISE_OUTPUT` environment variables.
CI runs the same CPU verification on a GitHub runner; a real Colab run must be
reported separately.

## What runs where

| Backend | CPU | NVIDIA GPU | TPU | Outputs |
|---|---|---|---|---|
| PySCF | Yes | Optional GPU4PySCF for SCF/DFT; integral transformation remains CPU | No | Full-space molecular integrals and metadata |
| QE | Yes, using `pw.x` | Only with a separately built GPU-enabled QE; not configured here | No | Plane-wave SCF energy, input and log |

This project does **not** make a local MCP server execute jobs in Colab. The
server runs on its own host; transfer configuration and results manually when
using Colab. Colab runtimes are temporary. GPU availability, CUDA versions and
usage limits vary; GPU4PySCF support depends on the system and calculation.
No TPU support is claimed. GPU mode uses density fitting, which can differ
slightly from CPU's default exact SCF. Set `density_fitting=true` for a CPU
calculation using the same approximation.

## Local setup

Use Linux or WSL with Python 3.10–3.13 and a reasonably recent pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
export OMP_NUM_THREADS=1
quantum-workflow examples/h2-pyscf.json --output results/h2
```

Choose a **new or empty** output directory for each run. After an error, inspect
the output directory and choose a different one or remove its contents yourself.
Configs describe Angstrom coordinates, molecular charge and `spin = Nα − Nβ`
(not multiplicity). PySCF uses restricted HF/KS for spin zero and ROHF/ROKS for
open-shell cases. Set `"method": "dft"` and e.g. `"xc": "pbe"` to use DFT.
Use `"device": "gpu"` with the Colab GPU environment configured above, or
install the matching `.[gpu-cuda12]`/`.[gpu-cuda13]` extra in a separate local
environment. Missing or unsupported GPU functionality produces an error.

## Quantum ESPRESSO

Install a working `pw.x` separately; for example, using an existing conda
installation:

```bash
conda create -y -p "$PWD/.qe-env" -c conda-forge --override-channels 'qe=7.6'
export PATH="$PWD/.qe-env/bin:$PATH"
```

Some Ubuntu 24.04 QE 6.7
packages abort in UPF processing; use another build if this occurs. The Colab
notebook installs a separate conda-forge QE environment using micromamba.
Download a vetted, compatible UPF for each element; use pseudopotentials
matching the XC functional and converge the cutoff values. Example:

```bash
mkdir -p pseudopotentials
curl -fL https://pseudopotentials.quantum-espresso.org/upf_files/H.pbe-kjpaw_psl.1.0.0.UPF \
  -o pseudopotentials/H.pbe-kjpaw_psl.1.0.0.UPF
quantum-workflow examples/h2-qe.json --output results/h2-qe
```

Pseudopotential paths are relative to the JSON file, not the current directory.
The QE backend supports spin-zero, even-electron SCF only, with fixed
occupations, a cubic periodic supercell and Gamma sampling. Set atomic
positions inside the cell and test vacuum size and energy/density cutoffs for
convergence; charged periodic systems require particular care. These examples
are starting points, not converged scientific predictions. QE energies using
pseudopotentials and PySCF all-electron energies use different representations
and should not be equated. QE does **not** export VQE-ready Gaussian molecular
integrals here.

## Hamiltonian interchange

PySCF's `summary.json` records the inputs, software version, SCF energy,
nuclear repulsion, orbital and electron counts, and schema version.
QE's summary records its version and pseudopotential SHA-256 hashes.
`integrals.npz` contains real, full-space
spatial-orbital arrays: `h1[p,q]`, `eri[p,q,r,s]`, `mo_coeff`, `mo_energy`, and
`mo_occ`. `h1` is the bare kinetic plus nuclear-attraction operator, not the
Fock or Kohn–Sham matrix. `eri` is in chemists' notation `(pq|rs)`, **not**
antisymmetrized, in Hartree. The Hamiltonian is

```
H = E_nuc + sum_pq h1[p,q] sum_s a†(p,s) a(q,s)
  + 1/2 sum_pqrs eri[p,q,r,s] sum_st a†(p,s) a†(r,t) a(s,t) a(q,s)
```

For DFT, orbitals are Kohn–Sham orbitals, but the exported Hamiltonian is the
physical electronic Hamiltonian, not an XC energy functional. A downstream
solver must add `nuclear_repulsion_hartree`, convert spin-orbital ordering and
two-electron conventions to its own API, and select any active space/frozen
core consistently. No active-space reduction, VQE, QSCI, pseudopotential-to-
integral conversion, or solver-specific adapter is included. Full `eri` scales
as the fourth power of orbital count; start with small basis sets.

```python
import json
import numpy as np
from pyscf import fci

with open("results/h2/summary.json") as f:
    info = json.load(f)
with np.load("results/h2/integrals.npz") as ints:
    energy, _ = fci.direct_spin1.kernel(
        ints["h1"], ints["eri"], info["num_spatial_orbitals"],
        (info["num_alpha"], info["num_beta"]),
        ecore=info["nuclear_repulsion_hartree"],
    )
print("FCI total energy (Hartree):", energy)
```

## MCP / agent use

Install the package locally, then configure a trusted MCP client (e.g. VS Code
or another MCP-capable agent) with stdio transport. For VS Code, add this to
`.vscode/mcp.json`, replacing the absolute executable path:

```json
{
  "servers": {
    "quantum-workflow": {
      "type": "stdio",
      "command": "/absolute/path/to/colabexercise/.venv/bin/quantum-workflow-mcp"
    }
  }
}
```

The `run_calculation` tool takes `config_path` and `output_dir`, both paths on
the MCP server's machine, and returns the summary or an explicit error. Ask
your agent to create/edit a config, run it into a new directory, and inspect
the output. For QE, ensure the MCP server inherits a `PATH` containing the
working `pw.x`; starting VS Code from an activated environment is one option.
This tool can read local files and write results wherever the server
process has permissions; use only with trusted clients and review actions.
Calculations may be expensive and synchronous; set resource limits externally.

## Test

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest
```

The tests exercise the MCP stdio transport and molecular-integral FCI
round-trip. QE process handling is tested with a mock; a separate H2 smoke
run was also performed with the conda-forge QE 7.6 CPU build.
