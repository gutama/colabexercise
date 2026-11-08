# Li-hBN reproduction in Colab (prepared 8 November 2026)

> **Evidence boundary.** This experiment is built from the uploaded `mylogbook_python_snippets_2026-10-20_v02.tex` (20 October 2026). The actual private `dipojono/lihbn` repository, `lihbn_hf_k3.out`, Wannier90 `*_hr.dat`, and correlated two-electron integrals were not accessible for this preparation. **No original Li-hBN physical result has been reproduced in this work.** The default notebook uses synthetic demo data.

## Start in Google Colab

[Open the Li-hBN notebook on this branch](https://colab.research.google.com/github/gutama/colabexercise/blob/feat/lihbn-colab-reproduction/notebooks/lihbn_reproduction.ipynb)

The feature branch is published on GitHub; open the Colab link directly.
The notebook defaults to cloning the published branch automatically, with a
manual ZIP fallback. GitHub Actions executes the notebook from an existing
checkout, so CI success does **not** test a fresh Google Colab bootstrap.
A live Colab runtime validation is still outstanding. Alternatively, set `SOURCE_MODE="ZIP"` to upload the offline reproduction package. Run on a **CPU runtime**. Its default mode is `DEMO`, creating two small **synthetic** spin-resolved Wannier90-format files and running deterministic tests. Nothing requires a GPU or launches a 20-day computation. It saves inputs' SHA-256, versions, eigenvectors and numerical checks in a private Colab output directory.

Switch `DATA_MODE='REAL'` only after mounting Google Drive or uploading the original per-spin Wannier90 `*_hr.dat`. Use paths to the specific 97-Wannier/149-R files. Original records are not put into this public repository. Output of H(R=0) and the full Fourier H(k=Gamma) are **different**; the notebook defaults to the note's H(R=0) convention. Use `H_MODE='gamma'` explicitly to make a different comparison.

The original notebook selects physical, 1-based indices:

- spin UP: `[19,33,55,56,57,67,68,69,70,71,72]`
- spin DOWN: `[19,31,55,56,57,67,68,69,70,71,72]`

These form a **spin-blocked** 22-mode *one-particle* Hamiltonian.
The experiment also exports an **interleaved** 22×22 copy related by an
exact permutation of spin-orbital labels. This is NOT a correlated state.
Crucially, the existing `clifford_qc.models.effective_hamiltonian` format
assumes a **shared spin-independent** one-body matrix. **Do not average,
equate, or silently reuse one Li-hBN spin block for the other**; a separate,
independently checked spin-dependent adapter is required before any
material-specific `clifford_qc` calculation. They must not be equated with a 22-qubit **correlated** Li-hBN wavefunction. The original selected active-space electron numbers are not established in the attachment. For real data, leave `N_UP` and `N_DOWN` set to `None` until verified. If given, the code constructs the quadratic-fermion occupied projectors and a bounded fixed-sector cross-check, not a correlated many-body solver.

The notebook also provides optional read-only checks for:

- `lihbn_hf_k3.out` (HF/ACE convergence **not** assumed from inner-SCF completion);
- two `*.eig` files against the spin-resolved `lihbn_nscf_atomic_projection.csv` (144 k-points, 200 bands per spin);
- a standalone, clearly synthetic `clifford_qc` Hubbard-dimer benchmark (opt-in package installation). This tests solver wiring, not the real Li-hBN interacting Hamiltonian.

## What remains for a genuine scientific reproduction

1. Preserve original QE 7.5 input, pseudopotentials, complete `pw.x` output, scratch/checkpoints, hardware/job logs, and completion/energy convergence evidence. A 3x3x1/1x1x1 HF/ACE feasibility run is **not** interchangeable with the original 12x12x1/2x2x1 target or with converged full HF.
2. Provide validated `*_hr.dat` for both spin channels and, to assess printed-HR Fourier discrepancy, direct `H^(W)(k)`, matching `U(k)`/`U_dis(k)`, k-point/gauge metadata, and hopping degeneracies. A parser's Hermiticity check does not resolve that discrepancy.
3. Supply verified active-space occupancy, the same orbital gauge/order, a justified two-electron interaction tensor and double-counting correction, plus independent benchmark energies, before comparing any many-body wavefunctions.
4. Evaluate residual norms, symmetry defects, energy, wavefunction **projectors/overlaps** (aware of degeneracy), 1-RDMs, and memory/time under matched Hamiltonians.

## Local execution and tests

```bash
python -m pip install numpy scipy pandas pytest nbformat nbclient ipykernel
PYTHONPATH=. python -m pytest tests/test_lihbn_repro.py -q
```

The notebook imports `lihbn_repro` from the cloned repo directly without changing the existing PySCF/QE workflow. Everything synthetic is explicitly labeled; no calculation is performed asynchronously or on external servers by this package.

## Files

- `notebooks/lihbn_reproduction.ipynb`: Colab entry point and explicit REAL/DEMO switching.
- `lihbn_repro/wannier.py`: `_hr.dat` validator, R0/Gamma, orbital eigensolver, fixed-sector noninteracting cross-check.
- `lihbn_repro/qe_audit.py`: read-only HF/ACE log checker.
- `lihbn_repro/validation.py`: optional `.eig` ↔ NSCF checks.
- `tests/test_lihbn_repro.py`: seven deterministic numerical safety tests.

### Historical notebook facts (not re-executed here)

- QE NSCF: 144 k-points × 400 eigenvalues and occupations = 57,600 records; Hartree in QE XML, transformed to eV for Wannier files.
- Wannier: 97 per spin, 149 real-space R points = 1,401,941 `(R,m,n)` records per spin.
- HF/ACE on 12 October: nine converged *inner* SCF cycles; about 6 d 22 h elapsed; no `JOB DONE.` observed in the notebook checkpoint.
- October 20 note explicitly leaves printed-HR Fourier reconstruction, interacting tensor, double counting, and completed direct HF benchmark unresolved.

Do not relabel these historical facts as live outputs of this experiment.
