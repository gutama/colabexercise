# Project notes for coding agents

Molecular electronic structure → Hamiltonian/qubit export, run locally or on Google Colab. See README.md
(user guide), docs/TEMPLATE.md (extension points) and docs/colab.md (Colab CLI).

## Commands

```bash
python -m pip install -e '.[qubit,test,dev]'
OMP_NUM_THREADS=1 python -m pytest            # ~30 s; add -m "not slow" for a quicker pass
ruff check .
quantum-workflow examples/h2o-cas44.json --check     # validate a config without computing
quantum-workflow examples/h2o-cas44.json --output results/x && quantum-workflow-adapt results/x
quantum-workflow-colab CONFIG --output DIR --dry-run # always dry-run before renting a Colab VM
```

## Rules

- **Every number is wrong until it reproduces an anchor.** New conversions, backends or solvers need a test
  against an independent reference (FCI/CASCI, a second mapping, a second package). Test qubit ordering on an
  asymmetric molecule (HeH⁺, LiH), never only H₂.
- Conventions are fixed in `quantum_workflow/hamiltonian.py`: Hartree, chemists' `(pq|rs)`, interleaved
  spin-orbitals, OpenFermion qubit q = Qiskit qubit q (labels little-endian). Don't change them; convert at the edges.
- Map any fermionic operator with `hamiltonian.map_fermion_operator(op, mapping, n_qubits)` (BK register size).
  Prepare reference states from `reference_bits`. Compare against sector ground states, not Fock-space minima.
- `spin` in configs is 2S, not multiplicity. Configs reject unknown keys; keep it that way.
- Runners write `summary.json` last via `provenance.write_summary`. Raise on non-convergence; never return an
  unchecked number. Don't print fabricated or placeholder energies.
- Keep `quantum_workflow/colab_remote.py` stdlib-only at import time (it runs before the package is installed).

## Colab safety

- A Colab VM bills until `colab stop`. The driver stops it in `finally`; after manual use, check `colab sessions`
  for leftovers (`[?]` = orphan).
- `colab exec` exits 0 on failure: check the `__QW_JOB_OK__` sentinel. Pass `--timeout` (the default is 30 s).
- Never put secrets in `colab --env` (logged in plaintext) or print/commit `~/.config/colab-cli/token.json`.
- Don't expose Colab jobs as MCP tools without an explicit opt-in.
