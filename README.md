# Colab quantum-chemistry workflow template

A small, tested template for the classical half of quantum-chemistry-on-quantum-computers work:

```
JSON config ──► PySCF HF/DFT ──► (active-space) integrals ──► qubit Hamiltonian ──► your solver
   │              or QE plane-wave DFT     integrals.npz / FCIDUMP     qubit_hamiltonian.json    (ADAPT-VQE
   │                                                                                             reference included)
   └── validated locally ─► run locally (CLI / MCP tool) or on a Google Colab VM (notebook / colab CLI driver)
```

Every stage is checked against an independent number before its output is written, because these conversions
fail silently: a transposed integral index, a reversed qubit register or a mis-sized Bravyi–Kitaev tree still
returns a plausible-looking energy. A run either reproduces its anchors or stops with an error.

**Using this as a template?** Read [docs/TEMPLATE.md](docs/TEMPLATE.md): renaming, adding molecules, backends,
reference methods, solvers and MCP tools. Colab specifics are in [docs/colab.md](docs/colab.md).

## Quick start

Linux/macOS/WSL, Python 3.10–3.13:

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[qubit,test,dev]'
export OMP_NUM_THREADS=1
quantum-workflow examples/lih-bk.json --output results/lih
quantum-workflow-adapt results/lih --max-operators 10
```

`quantum-workflow CONFIG --check` validates a config and prints it with defaults filled in, without computing.
Each run needs a **new or empty** output directory; `summary.json` is written last, so a directory without it
is a failed or interrupted run.

## Run on Colab

[Open molecular workflow in Colab](https://colab.research.google.com/github/gutama/colabexercise/blob/main/notebooks/colab.ipynb)
· [Open synthetic Li-hBN demo in Colab](https://colab.research.google.com/github/gutama/colabexercise/blob/main/notebooks/lihbn_reproduction.ipynb)

Select **L4 GPU** (or A100) under **Runtime → Change runtime type**,
then **Runtime → Run all**. The molecular notebook automatically selects
GPU4PySCF on GPU runtimes and CPU on CPU runtimes, installs the package, and
runs H₂/STO-3G. Set `COLABEXERCISE_DEVICE=gpu` to require GPU execution, or
`cpu` to select CPU explicitly. Set `RUN_QE=True` in its optional QE
cell to install QE 7.6 and run the separate plane-wave example. After the H₂ check,
the notebook builds a qubit Hamiltonian (HeH⁺ by default; `COLABEXERCISE_QUBIT_EXAMPLE`
picks another example or `none`) and runs the ADAPT-VQE reference on it. Browser downloads
are opt-in with `DOWNLOAD_RESULTS=True`. The Li-hBN notebook defaults to
synthetic data; original private materials are needed for its REAL mode.

### Verify the environment from a terminal or agent

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

### Run your own configs on a Colab VM

```bash
quantum-workflow-colab examples/lih-bk.json --output results/lih --dry-run   # show the plan first
quantum-workflow-colab examples/lih-bk.json --output results/lih
quantum-workflow-colab examples/h2o-dft-gpu.json --output results/h2o-gpu --gpu L4
```

The driver validates the config locally before renting anything, ships a wheel of your
checkout (uncommitted changes included), runs the job, checks for a success sentinel
(`colab exec` exits 0 on failure), downloads and safely extracts the results, and always
runs `colab stop`. GPU configs get the same pinned, isolated `gpu-cuda12`/`gpu-cuda13`
environment as the notebook. See [docs/colab.md](docs/colab.md) for authentication, costs,
recovery and the CLI's traps.

## What runs where

| Piece | CPU | NVIDIA GPU | TPU | Notes |
|---|---|---|---|---|
| PySCF SCF (HF/DFT) | Yes | GPU4PySCF, density-fitted, pinned `gpu-cuda12`/`gpu-cuda13` extras; verified on Colab L4 and A100 | No | Integral export is always CPU and exact |
| Active space, FCIDUMP, FCI/MP2/CCSD(T) references | Yes | — | No | Post-HF references need `method: "hf"`, CPU |
| Qubit Hamiltonian (JW / BK) + checks | Yes | — | No | `qubit` extra (OpenFermion) |
| ADAPT-VQE reference | Yes | — | No | Exact statevector, ≤ 20 qubits |
| Quantum ESPRESSO `pw.x` | Yes | Needs a separately built GPU QE | No | Gamma-point SCF in a cubic supercell |
| Li-hBN Wannier/QE audit (`lihbn_repro`) | Yes | — | No | Synthetic demo by default; see `experiments/lihbn/` |

The MCP server does **not** execute jobs in Colab: it runs on its own host. Colab runtimes are
temporary, and GPU availability, CUDA versions and usage limits vary. No TPU support is claimed.
GPU mode uses density fitting, which can differ slightly from CPU's default exact SCF; set
`density_fitting: true` for a CPU calculation using the same approximation.

## Configuration

```json
{
  "backend": "pyscf",
  "molecule": {"atoms": [{"element": "Li", "xyz": [0, 0, 0]}, {"element": "H", "xyz": [0, 0, 1.595]}],
               "charge": 0, "spin": 0},
  "basis": "sto-3g",
  "method": "hf",
  "active_space": {"n_electrons": 2, "n_orbitals": 5},
  "reference_methods": ["fci", "ccsd", "ccsd(t)"],
  "qubit": {"mapping": "bravyi_kitaev"}
}
```

- Coordinates are Angstrom. `spin` is **2S = Nα − Nβ, not the multiplicity** (a doublet is `1`). Spin zero uses
  RHF/RKS, open shells ROHF/ROKS.
- `method`: `"hf"` or `"dft"` (with `xc`, e.g. `"pbe"`, `"b3lyp"`).
- `device: "gpu"` runs the SCF on GPU4PySCF: use the Colab GPU environment above, or install the matching
  `.[gpu-cuda12]`/`.[gpu-cuda13]` extra in a separate local environment. Missing or unsupported GPU
  functionality is an error, never a CPU fallback. `density_fitting: true` uses the same approximation on CPU
  (GPU runs are always density-fitted). Exported integrals are exact either way, so density-fitted runs record
  the fitting error as a warning instead of failing check 2 below.
- `active_space` (optional): the window around the Fermi level, or explicit 0-based `orbitals`. Core orbitals
  must be doubly occupied and the active orbitals must contain the SCF determinant; otherwise the run stops
  rather than silently changing the reference state. Report results as e.g. *CASCI(4e,4o)/STO-3G*.
- `reference_methods`: `"fci"` diagonalizes the exported Hamiltonian (FCI, or CASCI in an active space);
  `"mp2"`, `"ccsd"`, `"ccsd(t)"` are run in the same orbital space (frozen orbitals outside it). The CCSD T1
  diagnostic is recorded; above 0.02 a warning flags multireference character.
- `qubit` (optional): `mapping` is `"jordan_wigner"` or `"bravyi_kitaev"`. Up to `validate_up_to_qubits`
  (default 16) the particle-number-sector ground state is diagonalized exactly and must equal FCI/CASCI.
- `symmetry: true` makes degenerate orbitals (and therefore integrals) reproducible across runs; so does
  `OMP_NUM_THREADS=1`. Energies are invariant either way.

Unknown keys are rejected. The schema is `quantum_workflow/config.py`.

Examples ([examples/](examples/README.md)): `h2-pyscf.json`, `heh-cation-qubit.json` (asymmetric, catches qubit
ordering bugs), `lih-bk.json` (Bravyi–Kitaev, CCSD(T)), `h2o-cas44.json` (active space), `h2o-dft-gpu.json`
(GPU4PySCF), `h2-pyscf-gpu.json` (GPU verification), `h2-qe.json` (Quantum ESPRESSO).

## Output contract

| File | Contents |
|---|---|
| `summary.json` | Inputs, energies (Hartree), reference energies, orbital/electron counts, active space, qubit metrics and checks, warnings, package versions and thread settings, wall time. `schema_version` 2. |
| `integrals.npz` | `h1[p,q]`, `eri[p,q,r,s]`, `core_energy`, `mo_coeff`, `mo_energy`, `mo_occ`, `active_orbitals`, `core_orbitals` |
| `hamiltonian.fcidump` | The same Hamiltonian in FCIDUMP format (`export_fcidump`, default on) |
| `qubit_hamiltonian.json` | Pauli terms, reference bitstring, conventions (when `qubit` is set) |
| `adapt_vqe.json` | Written by `quantum-workflow-adapt` |

**Integrals** are real, over the *exported* spatial orbitals (the active space if one is set), in Hartree.
`h1` is the bare one-electron operator plus, in an active space, the frozen-core potential; it is not the Fock
or Kohn–Sham matrix. `eri` is in chemists' notation `(pq|rs)`, **not** antisymmetrized. The Hamiltonian is

```
H = core_energy + Σ_pq h1[p,q] Σ_σ a†(p,σ) a(q,σ)
    + ½ Σ_pqrs eri[p,q,r,s] Σ_στ a†(p,σ) a†(r,τ) a(s,τ) a(q,σ)
```

`core_energy` is the nuclear repulsion plus (in an active space) the frozen-core energy; it is already in
`core_energy_hartree`, so do not add `nuclear_repulsion_hartree` again. For DFT the orbitals are Kohn–Sham
orbitals but the exported operator is the physical electronic Hamiltonian, not an XC functional.

```python
import json, numpy as np
from pyscf import fci

info = json.load(open("results/lih/summary.json"))
with np.load("results/lih/integrals.npz") as ints:
    energy, _ = fci.direct_spin1.kernel(ints["h1"], ints["eri"], info["num_spatial_orbitals"],
                                        (info["num_alpha"], info["num_beta"]),
                                        ecore=info["core_energy_hartree"])
```

**Qubit Hamiltonian.** Spin-orbitals are interleaved (qubit `2p` = spatial orbital `p` α, `2p+1` = β). Each term
has `ops` (explicit `[qubit, Pauli]` pairs, OpenFermion indexing) and a `qiskit_label`, which is little-endian:
character `label[n_qubits-1-q]` acts on qubit `q`. `reference_bits[q]` is the computational-basis value of
qubit `q` for the reference determinant **under this mapping** (BK stores parities, so it differs from the
occupation vector; LiH: JW `[1,1,1,1,0,…]`, BK `[1,0,1,0,0,…]`). Coefficients are real (Hermiticity is checked).

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

data = json.load(open("results/lih/qubit_hamiltonian.json"))
op = SparsePauliOp.from_list([(t["qiskit_label"], t["coeff"]) for t in data["terms"]])
ref = QuantumCircuit(data["n_qubits"])
for qubit, bit in enumerate(data["reference_bits"]):
    if bit:
        ref.x(qubit)
print(Statevector(ref).expectation_value(op).real)   # == summary["energy_hartree"] for HF
```

The summary's qubit block also reports `n_terms`, `n_qwc_groups` (deterministic greedy qubit-wise-commuting
groups, one measurement setting each), `max_pauli_weight`, the Pauli 1-norm λ, and
`shots_upper_bound_chemical_accuracy` = ⌈(λ/1.6 mHa)²⌉. That last number is a loose upper bound for importance-sampled
single-term measurement; λ is also the quantity that sets qubitization/QPE cost in fault-tolerant estimates.

## What the checks are

Each run stops with an error if any of these fails:

1. SCF converged.
2. The reference determinant's energy from the exported integrals equals the SCF energy (HF without density
   fitting). This catches
   integral, frozen-core and active-space mistakes.
3. The qubit Hamiltonian's expectation in `reference_bits` equals that energy. This catches notation, factor-of-½,
   spin pairing, qubit ordering and BK reference-state mistakes at once.
4. Up to 16 qubits: the ground state *in the right (N, Sz) sector* equals FCI/CASCI. The unrestricted minimum is
   not used: for HeH⁺ it is neutral HeH, 162 mHa below FCI.

The tests (`python -m pytest`, ~30 s) re-derive the golden numbers: H₂ 0.735 Å RHF −1.1169989968 / FCI
−1.1373060358, HeH⁺ RHF −2.8413824898 / FCI −2.8510240300, LiH RHF −7.8620238601 / FCI −7.8824019323,
H₂O CASCI(4e,4o) −74.9704543855, H₂ JW 15 terms/5 QWC groups (BK 3). They also cover the Qiskit ordering
on an asymmetric molecule, frozen-core integrals against PySCF's CASCI, a triplet reference, JW and BK giving
identical ADAPT runs, and the Colab driver against a fake `colab` CLI.

## ADAPT-VQE reference solver

`quantum-workflow-adapt RESULTS_DIR` reads `qubit_hamiltonian.json` and runs ADAPT-VQE with a fermionic
singles+doubles pool, exact statevectors and analytic gradients, then reports the error against FCI/CASCI.
It is a **classical reference** for checking a pipeline before simulator or hardware runs: no shots, no noise,
memory 2ⁿ×16 bytes, capped at 20 qubits. An ADAPT energy equal to E_HF (zero correlation) or below the sector
ground state is a pool/mapping bug, and the latter raises an error.

## Quantum ESPRESSO

The [wide-bandgap screening benchmark](wbg-benchmark/README.md) uses Quantum ESPRESSO to screen
GaN- and 4H-SiC-class semiconductors. It has its own Python package and calibration cases.

Install a working `pw.x` separately, e.g. `conda create -y -p "$PWD/.qe-env" -c conda-forge --override-channels
'qe=7.6'` and `export PATH="$PWD/.qe-env/bin:$PATH"`. Some Ubuntu 24.04 QE 6.7 packages abort in UPF processing.
Download a vetted pseudopotential per element (paths are relative to the JSON file):

```bash
mkdir -p pseudopotentials
curl -fL https://pseudopotentials.quantum-espresso.org/upf_files/H.pbe-kjpaw_psl.1.0.0.UPF \
  -o pseudopotentials/H.pbe-kjpaw_psl.1.0.0.UPF
quantum-workflow examples/h2-qe.json --output results/h2-qe
```

The QE backend supports spin-zero, even-electron SCF with fixed occupations, a cubic periodic supercell and
Gamma sampling; atoms must lie inside the cell. Energies are converted Ry → Ha (÷2). The summary records each
pseudopotential's SHA-256 and declared functional, and warns when that functional does not match `xc` (QE
does not check this). Converge vacuum and `ecutwfc`/`ecutrho` yourself (8–12× for PAW/ultrasoft). QE
pseudopotential energies and PySCF all-electron energies are not comparable, and QE does not export qubit
Hamiltonians here.

## MCP server

`quantum-workflow-mcp` serves `validate_config`, `run_calculation` and `run_adapt_vqe` over stdio. Paths are on
the server's machine. Colab jobs are deliberately not exposed, because they rent billable VMs.

```bash
claude mcp add quantum-workflow -- "$PWD/.venv/bin/quantum-workflow-mcp"     # Claude Code
```

For VS Code, add to `.vscode/mcp.json`:
`{"servers": {"quantum-workflow": {"type": "stdio", "command": "/abs/path/.venv/bin/quantum-workflow-mcp"}}}`.
For QE, start the client from an environment whose `PATH` contains `pw.x`. The tools read local files and write
wherever the server process can; use them only with trusted clients and set resource limits externally.

## Development

```bash
OMP_NUM_THREADS=1 python -m pytest          # full suite (~30 s); -m "not slow" skips the slower runs
ruff check .
```

CI: `tests.yml` runs lint and the suite on Python 3.10 and 3.13; `colab-cpu.yml` runs
`scripts/verify_colab.py` (suite plus both notebooks, headless); `lihbn-repro.yml` covers the Li-hBN package.
Scope and limits: small molecules. GPU SCF is checked by `verify_colab.py --device gpu` on a Colab GPU
runtime. On 2026-11-15 it passed on an NVIDIA L4 and an A100-SXM4-40GB (CUDA 13): the whole suite plus both
notebooks. H100 was refused for this account (no entitlement) and G4 had no capacity (HTTP 503), so both are
untested. `verify_colab.py --qe` installed QE 7.6 with `ensure_qe` on a Colab CPU VM and ran the H₂ example.
The `quantum-workflow-colab` driver ran a B3LYP GPU job (L4) and the QE job (CPU) on Colab the same day; in CI
it is tested against a fake `colab` CLI. These examples are starting points, not converged scientific
predictions.
