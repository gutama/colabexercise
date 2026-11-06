# Colab molecular simulation workflow

Run small-molecule Hartree–Fock or DFT with PySCF, export a molecular
Hamiltonian for downstream VQE/QSCI, or run a separate plane-wave DFT
calculation with Quantum ESPRESSO (QE). Control local runs through the CLI or
an MCP stdio server; use the [Colab notebook](notebooks/colab.ipynb) for
interactive cloud runs.

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
slightly from CPU's default exact SCF. GPU mode has not been validated in this
repository on actual GPU hardware.

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
Use `"device": "gpu"` only after installing GPU4PySCF for your CUDA version
following its [installation guide](https://pyscf.org/user/gpu.html); missing or
unsupported GPU functionality produces an error, not a CPU fallback.

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
