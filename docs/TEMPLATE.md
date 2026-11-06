# Using this repository as a template

The project is deliberately small and layered so each piece can be replaced without touching the others:

```
quantum_workflow/
  config.py          pydantic models: the JSON schema (one model per backend)
  backends/          runners, registered in backends/__init__.py: config -> files + summary
    pyscf_backend.py   SCF, orbital selection, references, exports
    qe_backend.py      Quantum ESPRESSO pw.x
  hamiltonian.py     integrals -> determinant/FCI energies -> qubit operators (the conventions live here)
  adapt.py           an example *consumer* of qubit_hamiltonian.json
  workflow.py        public API: parse_config / load_config / run
  provenance.py      versions, threads, timestamps; writes summary.json
  cli.py, mcp_server.py, colab.py (local driver), colab_remote.py (runs on the VM)
examples/            configs; every one is validated by the tests
tests/               anchors, conversion traps, Colab driver against a fake CLI
notebooks/colab.ipynb
```

## 1. Start your project

1. On GitHub, open **Settings → Template repository** on this repo and then use **Use this template**, or clone
   it and run `git remote set-url origin <your repo>`.
2. Rename the package in one go. Check the result with `grep -rn quantum_workflow --exclude-dir=.git`:
   - the directory `quantum_workflow/` and its imports;
   - `pyproject.toml`: `name`, `description`, the `[project.scripts]` entry points and `packages.find`;
   - `quantum_workflow/__init__.py`: the distribution name passed to `importlib.metadata.version`.
     `distribution_name()` finds the rest automatically, so the Colab driver needs no edit.
3. Update `repo_url` in the first notebook cell and the clone URL in `docs/colab.md`.
4. Run `python -m pip install -e '.[qubit,test,dev]'`, `python -m pytest` and `ruff check .`. CI runs the
   same on every push.

## 2. Add a molecule

Add `examples/<name>.json`; the test suite validates every example automatically. If the molecule is a
reference system for your project, add an **anchor test** that pins its SCF and FCI/CASCI energies (see
`test_golden_anchors_with_reference_methods`). Prefer *asymmetric* molecules for any qubit-level test: H₂ is
symmetric enough that a reversed qubit register gives the right energy.

## 3. Add a reference method

In `backends/pyscf_backend.py::_reference_energies`, compute the energy in the same orbital space
(`frozen` lists the orbitals outside the active space), add the method name to `ReferenceMethod` in
`config.py`, and add a validator rule if it needs a particular `method`/`device`. Test it against a number
you trust, such as FCI for two electrons or CASCI for a small active space.

## 4. Add a backend

1. `config.py`: subclass `BackendConfig` with `backend: Literal["yourcode"]`. Override `resolve_paths` if the
   config references files, then add the class to the `Config` union.
2. `backends/yourcode_backend.py`: `def run_yourcode(config, output: Path) -> dict`. Write your files into
   `output`, and finish with `provenance.write_summary(output, summary)` so `summary.json` is written last.
   Raise `RuntimeError`/`ValueError` on any non-convergence; never return a number you have not checked.
3. Register it in `backends/__init__.py::RUNNERS`, and add an example plus a test (mock the external binary as
   `test_qe_input_and_convergence_check` does).
4. If it must run on Colab, teach `colab_remote.main` how to install its dependencies on the VM.

## 5. Add a downstream solver (VQE, QSCI, SQD, QPE costing, ...)

Consume the **files**, not the PySCF objects, as `adapt.py` does: read `qubit_hamiltonian.json` (or
`integrals.npz`/FCIDUMP) and `summary.json`, then write your own report next to them. Rules that keep it honest:

- Prepare the reference state from `reference_bits`, not from "the first N qubits"; that is wrong under BK and
  for open shells.
- Map any extra fermionic operator (an ansatz or excitation pool) with
  `hamiltonian.map_fermion_operator(op, mapping, n_qubits)`. Without `n_qubits`, OpenFermion sizes a BK tree
  from the operator itself.
- Compare to `reference_energies_hartree.fci` or `qubit_hamiltonian.sector_ground_energy_hartree`, never to the
  unrestricted Fock-space minimum. A variational energy below the sector ground state is a bug.
- For Qiskit use `qiskit_label` as is (it is already little-endian), and test ordering on HeH⁺, not H₂.
- Before scaling, record what the result costs: qubits, terms, QWC groups and the Pauli 1-norm are already in
  the summary. Statevector memory is 2ⁿ × 16 bytes, and FCI grows as C(n, Nα)·C(n, Nβ).

## 6. Add an MCP tool

Decorate a function in `mcp_server.py` with `@mcp.tool()`. The docstring is what the agent reads, so state the
side effects and the cost. Extend `test_mcp_stdio_runs_calculation`. Keep billable actions, such as Colab jobs,
out of MCP unless you add an explicit opt-in.

## 7. Conventions you inherit

- Hartree everywhere. QE's Ry is converted in exactly one place.
- Chemists' `(pq|rs)` integrals over exported spatial orbitals, plus `core_energy`.
- Interleaved spin-orbitals: qubit `2p` = α, `2p+1` = β. OpenFermion qubit `q` is Qiskit qubit `q`.
- `spin` = 2S. Configs reject unknown keys and validate assignments.
- Results directories must be new or empty, and `summary.json` is written last.
- `OMP_NUM_THREADS=1` (or `symmetry: true`) for reproducible integrals when orbitals are degenerate.

A result is a number, the method, and the evidence it is converged or validated, e.g. *CASCI(4e,4o)/STO-3G
−74.97045 Ha; qubit sector ground state agrees to 1e-10; SCF conv_tol 1e-9*.
