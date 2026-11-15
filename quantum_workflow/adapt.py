"""Exact-statevector ADAPT-VQE reference solver for an exported qubit Hamiltonian.

This consumes ``qubit_hamiltonian.json`` (the file contract, not PySCF objects) and
is a *classical reference*: exact 2^n statevectors, analytic gradients, no shots or
noise. Use it to check a pipeline -- mapping, reference state, operator pool --
before spending simulator or hardware time. It is not a quantum advantage claim, and
memory grows as 2^n x 16 bytes.

Behaviour worth knowing (each is asserted in tests/test_adapt.py):
* At a canonical RHF reference every single-excitation gradient vanishes (Brillouin),
  so a singles-only pool -- or doubles mapped on a mis-sized BK register -- stops at
  step 0 with E = E_HF. Treat E_ADAPT == E_HF as a pool or mapping bug.
* JW and BK must give identical energies and operator sequences; the encoding only
  changes Pauli weight.
* Near-degenerate gradients tie below optimizer noise; ties within ``tie_tolerance``
  go to the lowest pool index so runs are reproducible.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

from . import hamiltonian as qh


def excitation_pool(reference_occupation: list[int]):
    """Spin-conserving singles and doubles relative to the reference, as (label, A = T - T+)."""
    of = qh._openfermion()
    occupied = [i for i, f in enumerate(reference_occupation) if f]
    virtual = [a for a, f in enumerate(reference_occupation) if not f]
    pool = []
    for i in occupied:
        for a in virtual:
            if i % 2 == a % 2:
                t = of.FermionOperator(((a, 1), (i, 0)))
                pool.append((f"{a}^ {i}", t - of.hermitian_conjugated(t)))
    for i, j in combinations(occupied, 2):
        for a, b in combinations(virtual, 2):
            if sorted((i % 2, j % 2)) == sorted((a % 2, b % 2)):
                t = of.FermionOperator(((a, 1), (b, 1), (j, 0), (i, 0)))
                pool.append((f"{a}^ {b}^ {j} {i}", t - of.hermitian_conjugated(t)))
    return pool


def basis_state(bits: list[int]) -> np.ndarray:
    """Statevector for computational-basis bits in OpenFermion's (big-endian) ordering."""
    n = len(bits)
    index = sum(int(b) << (n - 1 - q) for q, b in enumerate(bits))
    state = np.zeros(2**n, dtype=complex)
    state[index] = 1.0
    return state


def _energy_and_gradient(thetas, operators, hamiltonian, psi0):
    from scipy.sparse.linalg import expm_multiply

    states = [psi0]
    psi = psi0
    for theta, op in zip(thetas, operators, strict=True):
        psi = expm_multiply(theta * op, psi)
        states.append(psi)
    sigma = hamiltonian @ psi
    energy = float(np.vdot(psi, sigma).real)
    gradient = np.zeros(len(thetas))
    for j in reversed(range(len(thetas))):
        gradient[j] = 2.0 * np.real(np.vdot(sigma, operators[j] @ states[j + 1]))
        sigma = expm_multiply(-thetas[j] * operators[j], sigma)
    return energy, gradient


def adapt_vqe(qubit_data: dict, max_operators: int = 30, gradient_tolerance: float = 1e-4,
              tie_tolerance: float = 1e-7) -> dict:
    of = qh._openfermion()
    from scipy.optimize import minimize
    from scipy.sparse.linalg import expm_multiply

    n = qubit_data["n_qubits"]
    mapping = qubit_data["mapping"]
    if n > 20:
        raise ValueError(f"{n} qubits needs a 2^{n} statevector; this reference stops at 20")
    qop = qh.qubit_hamiltonian_from_dict(qubit_data)
    hamiltonian = of.get_sparse_operator(qop, n_qubits=n).tocsr()
    psi0 = basis_state(qubit_data["reference_bits"])
    reference_energy = float(np.vdot(psi0, hamiltonian @ psi0).real)
    expected = qh.basis_state_energy(qop, qubit_data["reference_bits"])
    if abs(reference_energy - expected) > 1e-9:
        raise RuntimeError("Reference bits and statevector ordering disagree")

    pool = excitation_pool(qubit_data["reference_occupation"])
    matrices = [
        of.get_sparse_operator(qh.map_fermion_operator(op, mapping, n), n_qubits=n).tocsr()
        for _, op in pool
    ]
    chosen: list[int] = []
    thetas = np.zeros(0)
    energy = reference_energy
    history = []
    converged = False
    psi = psi0
    for _ in range(max_operators + 1):
        h_psi = hamiltonian @ psi
        gradients = np.array([2.0 * np.real(np.vdot(h_psi, m @ psi)) for m in matrices])
        largest = float(np.abs(gradients).max()) if len(gradients) else 0.0
        if largest < gradient_tolerance:
            converged = True
            break
        if len(chosen) == max_operators:
            break
        pick = int(np.flatnonzero(np.abs(gradients) >= largest - tie_tolerance)[0])
        chosen.append(pick)
        operators = [matrices[k] for k in chosen]
        result = minimize(
            _energy_and_gradient, np.append(thetas, 0.0), jac=True, method="BFGS",
            args=(operators, hamiltonian, psi0), options={"gtol": 1e-8, "maxiter": 1000},
        )
        thetas = result.x
        energy = float(result.fun)
        psi = psi0
        for theta, op in zip(thetas, operators, strict=True):
            psi = expm_multiply(theta * op, psi)
        history.append({
            "operator": pool[pick][0],
            "pool_index": pick,
            "max_gradient": largest,
            "energy_hartree": energy,
        })
    return {
        "schema_version": 1,
        "method": "ADAPT-VQE (exact statevector, fermionic singles+doubles pool)",
        "mapping": mapping,
        "n_qubits": n,
        "pool_size": len(pool),
        "gradient_tolerance": gradient_tolerance,
        "converged": converged,
        "reference_energy_hartree": reference_energy,
        "energy_hartree": energy,
        "operators": [pool[k][0] for k in chosen],
        "parameters": [float(t) for t in thetas],
        "history": history,
    }


def run_adapt(results_dir: str | Path, output: str | Path | None = None, **options) -> dict:
    """Run ADAPT-VQE on ``results_dir/qubit_hamiltonian.json`` and write a JSON report."""
    results = Path(results_dir)
    source = results / "qubit_hamiltonian.json"
    if not source.is_file():
        raise ValueError(f"{source} not found; run a PySCF config with a 'qubit' section first")
    target = Path(output) if output else results / "adapt_vqe.json"
    if target.exists():
        raise ValueError(f"Refusing to overwrite {target}")
    report = adapt_vqe(json.loads(source.read_text()), **options)
    summary_file = results / "summary.json"
    if summary_file.is_file():
        summary = json.loads(summary_file.read_text())
        exact = summary.get("reference_energies_hartree", {}).get("fci")
        if exact is None:
            exact = (summary.get("qubit_hamiltonian") or {}).get("sector_ground_energy_hartree")
        if exact is not None:
            report["exact_energy_hartree"] = exact
            report["error_hartree"] = report["energy_hartree"] - exact
            error = abs(report["error_hartree"])
            report["chemical_accuracy"] = error < qh.CHEMICAL_ACCURACY_HARTREE
            if report["error_hartree"] < -1e-8:
                raise RuntimeError(
                    "ADAPT energy is below the sector ground state: the reference or pool "
                    "leaks particle number -- this is a bug, not a result"
                )
    target.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", help="Output directory of a run with a qubit Hamiltonian")
    parser.add_argument("--max-operators", type=int, default=30)
    parser.add_argument("--gradient-tolerance", type=float, default=1e-4)
    parser.add_argument("--output", help="Report path (default: RESULTS/adapt_vqe.json)")
    args = parser.parse_args(argv)
    try:
        report = run_adapt(args.results, args.output, max_operators=args.max_operators,
                           gradient_tolerance=args.gradient_tolerance)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"ADAPT-VQE failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
