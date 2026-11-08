# Codex handoff — Li-hBN reproduction in `gutama/colabexercise`

## Objective

Review and land the self-contained Colab reproduction bundle as a **new** experiment, without changing the existing `quantum_workflow` PySCF/QE code, `notebooks/colab.ipynb`, or any other experiments.

The source is `mylogbook_python_snippets_2026-10-20_v02.tex`, not live output from the inaccessible `dipojono/lihbn` private repository.

## Files to add

- `notebooks/lihbn_reproduction.ipynb`
- `lihbn_repro/{__init__,wannier,qe_audit,validation}.py`
- `tests/test_lihbn_repro.py`
- `experiments/lihbn/{README.md,.gitignore,CODEX_TASK.md}`

## Tasks

1. Verify `_hr.dat` parsing with synthetic and production-format fixtures, including degeneracy handling, exact `(R,m,n)` coverage, `H(R)=H(-R)^dagger`, and phase conventions. Do **not** equate `H(R=0)` with Fourier `H(k=0)`.
2. Independently check spin-blocked one-body eigensolutions against a fixed-(Nup,Ndown) second-quantized Hamiltonian; require energy residuals and state overlap in a nondegenerate test. Do not assume active-space electron numbers for original Li-hBN.
3. Check `audit_qe_hf` is **read only** and does not confuse inner-SCF convergence with complete HF/ACE. Avoid claiming job status when only historical logs are present.
4. Run `PYTHONPATH=. python -m pytest tests/test_lihbn_repro.py -q` and the default synthetic notebook end-to-end on CPU. The code should not need remote compute for default tests.
5. Review the opt-in Clifford QC smoke cell against current `gutama/clifford_qc` APIs. The toy Hubbard-dimer benchmark is *synthetic* and is not validation of the correlated Li-hBN Hamiltonian.
6. On the PR, explicitly report which outputs are synthetic, which require original private data, the known HF/ACE completion boundary, and whether real Colab execution was performed.

## Expected acceptance criteria

- Notebook opens in Google Colab and runs from an uploaded source ZIP (before repo branch publication) or from GitHub once published.
- Test suite succeeds; no original Li-hBN physical result is fabricated.
- No private `dipojono/lihbn` data, pseudopotentials, production `.save/`, tokens, SSH host details, or large output files are committed.
- Original active-space spin-blocked ordering and spin-dependent Wannier selection are preserved.
- No claim that a spin-independent onsite-Hubbard convenience schema is the real 22-mode Li-hBN Hamiltonian.
- No live HF/ACE calculation auto-started by the notebook.

## Git workflow (now that GitHub App permission is granted)

```bash
git checkout -b feat/lihbn-colab-reproduction
git apply --check /path/to/lihbn_colab_reproduction.patch
git apply /path/to/lihbn_colab_reproduction.patch
PYTHONPATH=. python -m pytest tests/test_lihbn_repro.py -q
git add lihbn_repro notebooks/lihbn_reproduction.ipynb experiments/lihbn tests/test_lihbn_repro.py
git commit -m 'Add Li-hBN Colab eigenvector reproduction and HF audit'
git push -u origin feat/lihbn-colab-reproduction
```

Open a PR to `main`, let Codex review tests/scientific claims, and merge only after review. Once PR is merged, change the source notebook `SOURCE_BRANCH` to `main` and default `SOURCE_MODE` to `GITHUB` if desired.
