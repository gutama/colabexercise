"""Molecular Hamiltonian export: active spaces, reference determinants, qubit mappings.

Conventions fixed for every file this project writes
----------------------------------------------------
* Energies are Hartree. No silent unit conversion happens anywhere.
* ``h1[p, q]`` and ``eri[p, q, r, s] = (pq|rs)`` (chemists' notation, not
  antisymmetrized) over the exported spatial orbitals. ``core_energy`` is the
  constant to add: nuclear repulsion, plus the frozen-core energy for an active space.
* Spin-orbitals are interleaved: spatial orbital p -> 2p (alpha), 2p + 1 (beta).
* Qubit q is OpenFermion qubit q. Qiskit labels are little-endian, so qubit q is
  the character ``label[n_qubits - 1 - q]``.

Each of these is easy to write plausibly and wrongly -- a transposed eri index, a
reversed register, a Bravyi-Kitaev tree sized from the operator instead of the
register -- and the result is a believable energy that is off by Hartrees. Every
function here is checked against an independent anchor in tests/test_hamiltonian.py,
on asymmetric molecules (H2 is too symmetric to expose ordering bugs).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MAPPINGS = ("jordan_wigner", "bravyi_kitaev")
CHEMICAL_ACCURACY_HARTREE = 1.6e-3
QUBIT_ORDER_DOC = (
    "interleaved spin-orbitals: qubit 2p = spatial orbital p (alpha), 2p+1 = p (beta)"
)
QISKIT_LABEL_DOC = "little-endian: qiskit_label[n_qubits - 1 - q] acts on qubit q"


@dataclass(frozen=True)
class MolecularHamiltonian:
    """Spatial-orbital Hamiltonian in chemists' notation:

        H = core_energy + sum_pq h1[p,q] sum_s a+(p,s) a(q,s)
            + 1/2 sum_pqrs eri[p,q,r,s] sum_st a+(p,s) a+(r,t) a(s,t) a(q,s)

    The occupations describe the reference determinant per spatial orbital.
    """

    core_energy: float
    h1: np.ndarray
    eri: np.ndarray
    alpha_occupation: tuple[int, ...]
    beta_occupation: tuple[int, ...]

    @property
    def n_orbitals(self) -> int:
        return self.h1.shape[0]

    @property
    def n_qubits(self) -> int:
        return 2 * self.n_orbitals

    @property
    def n_alpha(self) -> int:
        return sum(self.alpha_occupation)

    @property
    def n_beta(self) -> int:
        return sum(self.beta_occupation)

    def spin_orbital_occupation(self) -> list[int]:
        """Reference determinant as interleaved spin-orbital occupations."""
        occ = []
        for a, b in zip(self.alpha_occupation, self.beta_occupation, strict=True):
            occ += [a, b]
        return occ


# --------------------------------------------------------------------------
# Orbital selection and integrals (PySCF)
# --------------------------------------------------------------------------
def select_orbitals(n_electrons: int, mo_occ, n_active_electrons=None, n_active_orbitals=None,
                    orbitals=None) -> tuple[list[int], list[int]]:
    """Return (core, active) MO indices; full space when no active space is given.

    Core orbitals must be doubly occupied in the SCF solution and the active
    orbitals must hold exactly the active electrons, so the SCF determinant lies
    inside the active space. Anything else silently changes the reference state.
    """
    mo_occ = np.asarray(mo_occ)
    nmo = len(mo_occ)
    if n_active_orbitals is None:
        return [], list(range(nmo))
    core_electrons = n_electrons - n_active_electrons
    if core_electrons < 0 or core_electrons % 2:
        raise ValueError(
            f"{n_electrons} electrons cannot be split into {n_active_electrons} active "
            "electrons and doubly occupied core orbitals"
        )
    n_core = core_electrons // 2
    if orbitals is None:
        active = list(range(n_core, n_core + n_active_orbitals))
    else:
        active = sorted(int(i) for i in orbitals)
    if max(active) >= nmo or n_core + n_active_orbitals > nmo:
        raise ValueError(f"Active space does not fit in {nmo} molecular orbitals")
    chosen = set(active)
    core = [i for i in range(nmo) if i not in chosen][:n_core]
    if any(mo_occ[i] < 1.5 for i in core):
        raise ValueError(f"Core orbitals {core} are not all doubly occupied in the SCF solution")
    held = float(mo_occ[active].sum())
    if abs(held - n_active_electrons) > 1e-6:
        raise ValueError(
            f"Active orbitals {active} hold {held:g} electrons in the SCF solution, "
            f"expected {n_active_electrons}; choose orbitals that contain the reference"
        )
    return core, active


def build_hamiltonian(mol, mo_coeff, mo_occ, core: list[int], active: list[int]
                      ) -> MolecularHamiltonian:
    """Exact-integral (frozen-core) Hamiltonian over ``active`` orbitals.

    The core potential is built from exact J/K rather than ``mf.get_jk`` so that a
    density-fitted (e.g. GPU) SCF object cannot mix fitted and exact integrals.
    """
    from pyscf import ao2mo, scf

    mo_coeff = np.asarray(mo_coeff)
    mo_occ = np.asarray(mo_occ)
    if not np.all(np.isclose(mo_occ, np.round(mo_occ))):
        raise ValueError("Fractional occupations are not supported for Hamiltonian export")
    hcore_ao = scf.hf.get_hcore(mol)
    core_energy = float(mol.energy_nuc())
    h1_ao = hcore_ao
    if core:
        c_core = mo_coeff[:, core]
        dm_core = 2.0 * c_core @ c_core.T
        vj, vk = scf.hf.get_jk(mol, dm_core)
        v_core = vj - 0.5 * vk
        core_energy += float(np.einsum("ij,ji->", hcore_ao + 0.5 * v_core, dm_core))
        h1_ao = hcore_ao + v_core
    c_act = mo_coeff[:, active]
    n = len(active)
    h1 = c_act.T @ h1_ao @ c_act
    eri = ao2mo.restore(1, ao2mo.kernel(mol, c_act), n)
    occ = mo_occ[active]
    return MolecularHamiltonian(
        core_energy=core_energy,
        h1=0.5 * (h1 + h1.T),
        eri=np.asarray(eri),
        alpha_occupation=tuple(int(o > 0.5) for o in occ),
        beta_occupation=tuple(int(o > 1.5) for o in occ),
    )


def determinant_energy(ham: MolecularHamiltonian) -> float:
    """Energy of the reference determinant, straight from the spatial integrals."""
    a = np.flatnonzero(ham.alpha_occupation)
    b = np.flatnonzero(ham.beta_occupation)
    coulomb = np.einsum("iijj->ij", ham.eri)
    exchange = np.einsum("ijji->ij", ham.eri)
    energy = ham.core_energy + ham.h1[a, a].sum() + ham.h1[b, b].sum()
    energy += 0.5 * (coulomb[np.ix_(a, a)] - exchange[np.ix_(a, a)]).sum()
    energy += 0.5 * (coulomb[np.ix_(b, b)] - exchange[np.ix_(b, b)]).sum()
    energy += coulomb[np.ix_(a, b)].sum()
    return float(energy)


def exact_energy(ham: MolecularHamiltonian, max_determinants: int = 5_000_000) -> float:
    """Lowest eigenvalue of the exported Hamiltonian at fixed (N_alpha, N_beta).

    In the full orbital space this is FCI; in an active space it is CASCI.
    """
    from pyscf import fci

    n, na, nb = ham.n_orbitals, ham.n_alpha, ham.n_beta
    n_det = math.comb(n, na) * math.comb(n, nb)
    if n_det > max_determinants:
        raise ValueError(
            f"Exact diagonalization needs {n_det:,} determinants (> {max_determinants:,}); "
            "use a smaller active space"
        )
    solver = fci.direct_spin1.FCI()
    solver.conv_tol = 1e-12
    solver.max_cycle = 500
    energy, _ = solver.kernel(ham.h1, ham.eri, n, (na, nb), ecore=ham.core_energy)
    if not getattr(solver, "converged", True):
        raise RuntimeError("Exact diagonalization (FCI/CASCI) did not converge")
    return float(energy)


# --------------------------------------------------------------------------
# Spin-orbital tensors and qubit mappings (OpenFermion)
# --------------------------------------------------------------------------
def spin_orbital_tensors(ham: MolecularHamiltonian):
    """(constant, h1_so, h2_so) in OpenFermion's InteractionOperator convention.

        H = constant + sum h1_so[p,q] a+_p a_q + sum h2_so[p,q,r,s] a+_p a+_q a_r a_s

    Chemist (pq|rs) maps to h2_so[2p+s, 2r+t, 2s+t, 2q+s] = 1/2 (pq|rs): note the
    p,r,s,q index order, the factor 1/2, and spin s on the p/q slots, t on r/s.
    """
    n = ham.n_orbitals
    h1_so = np.zeros((2 * n, 2 * n))
    h2_so = np.zeros((2 * n,) * 4)
    physicist = 0.5 * ham.eri.transpose(0, 2, 3, 1)  # [p, r, s, q] <- (pq|rs)
    for sigma in (0, 1):
        h1_so[sigma::2, sigma::2] = ham.h1
        for tau in (0, 1):
            h2_so[sigma::2, tau::2, tau::2, sigma::2] = physicist
    return ham.core_energy, h1_so, h2_so


def _openfermion():
    try:
        import openfermion
    except ImportError as exc:
        raise RuntimeError(
            "Qubit Hamiltonians need OpenFermion: pip install -e '.[qubit]'"
        ) from exc
    return openfermion


def map_fermion_operator(operator, mapping: str, n_qubits: int):
    """Map a FermionOperator with an explicit register size.

    ``openfermion.bravyi_kitaev(op)`` without ``n_qubits`` sizes the BK tree from
    the highest index in ``op`` -- wrong for any operator (e.g. an excitation)
    that does not touch the last spin-orbital.
    """
    of = _openfermion()
    if mapping == "jordan_wigner":
        return of.jordan_wigner(operator)
    if mapping == "bravyi_kitaev":
        return of.bravyi_kitaev(operator, n_qubits=n_qubits)
    raise ValueError(f"Unknown mapping {mapping!r}; choose from {MAPPINGS}")


def qubit_hamiltonian(ham: MolecularHamiltonian, mapping: str = "jordan_wigner"):
    """OpenFermion QubitOperator for the exported Hamiltonian."""
    of = _openfermion()
    constant, h1_so, h2_so = spin_orbital_tensors(ham)
    interaction = of.InteractionOperator(constant, h1_so, h2_so)
    qop = map_fermion_operator(of.get_fermion_operator(interaction), mapping, ham.n_qubits)
    qop.compress(abs_tol=1e-12)
    return qop


def number_operator_supports(mapping: str, n_qubits: int) -> np.ndarray:
    """Row j marks the qubits of the Z-string that n_j maps to: n_j = (1 - Z_S)/2."""
    of = _openfermion()
    rows = np.zeros((n_qubits, n_qubits), dtype=np.uint8)
    for j in range(n_qubits):
        image = map_fermion_operator(of.FermionOperator(((j, 1), (j, 0))), mapping, n_qubits)
        strings = [term for term in image.terms if term]
        if len(strings) != 1 or any(pauli != "Z" for _, pauli in strings[0]):
            raise RuntimeError(f"n_{j} is not a single Z-string under {mapping}")
        rows[j, [q for q, _ in strings[0]]] = 1
    return rows


def _solve_gf2(matrix: np.ndarray, rhs) -> list[int]:
    a = np.concatenate([matrix.copy() % 2, np.asarray(rhs, dtype=np.uint8)[:, None] % 2], axis=1)
    n = matrix.shape[1]
    row = 0
    pivots = []
    for col in range(n):
        hits = np.flatnonzero(a[row:, col]) + row
        if len(hits) == 0:
            continue
        a[[row, hits[0]]] = a[[hits[0], row]]
        for r in np.flatnonzero(a[:, col]):
            if r != row:
                a[r] ^= a[row]
        pivots.append(col)
        row += 1
    if row != n:
        raise RuntimeError("Number-operator supports are not linearly independent")
    return [int(a[i, -1]) for i in range(n)]


def reference_bits(ham: MolecularHamiltonian, mapping: str) -> list[int]:
    """Computational-basis bits (indexed by qubit) encoding the reference determinant.

    Under Jordan-Wigner this is the occupation vector itself. Under Bravyi-Kitaev
    qubits store partial parities, so the same determinant is a *different*
    bitstring; using the JW bitstring with a BK Hamiltonian is wrong by Hartrees.
    """
    return _solve_gf2(number_operator_supports(mapping, ham.n_qubits),
                      ham.spin_orbital_occupation())


def basis_state_energy(qop, bits) -> float:
    """<b|H|b> for a computational basis state: only Z/identity strings contribute."""
    energy = 0.0
    for term, coeff in qop.terms.items():
        if all(pauli == "Z" for _, pauli in term):
            parity = sum(bits[q] for q, _ in term) % 2
            energy += coeff.real * (-1) ** parity
    return float(energy)


def sector_ground_energy(qop, mapping: str, n_qubits: int, n_alpha: int, n_beta: int) -> float:
    """Ground energy restricted to N = n_alpha + n_beta and 2Sz = n_alpha - n_beta.

    The unrestricted minimum of a qubit Hamiltonian lives anywhere in Fock space and
    need not have your electron count (HeH+ STO-3G: 162 mHa below FCI, neutral HeH).
    """
    of = _openfermion()
    from scipy.sparse.linalg import eigsh

    hamiltonian = of.get_sparse_operator(qop, n_qubits=n_qubits).tocsr()
    number = np.zeros(2**n_qubits)
    two_sz = np.zeros(2**n_qubits)
    for j in range(n_qubits):
        n_j = map_fermion_operator(of.FermionOperator(((j, 1), (j, 0))), mapping, n_qubits)
        diagonal = of.get_sparse_operator(n_j, n_qubits=n_qubits).diagonal().real
        number += diagonal
        two_sz += diagonal if j % 2 == 0 else -diagonal
    sector = np.flatnonzero(
        (np.abs(number - (n_alpha + n_beta)) < 1e-9) & (np.abs(two_sz - (n_alpha - n_beta)) < 1e-9)
    )
    block = hamiltonian[sector][:, sector]
    if len(sector) <= 2048:
        return float(np.linalg.eigvalsh(block.toarray())[0])
    return float(eigsh(block, k=1, which="SA", tol=1e-12)[0][0])


def qwc_groups(qop) -> list[list[tuple]]:
    """Deterministic greedy qubit-wise-commuting grouping (largest |coeff| first).

    Each group can be measured with one basis-rotation layer. Greedy is not optimal;
    it is reproducible, which a shot-count comparison needs.
    """
    terms = sorted((t for t in qop.terms if t), key=lambda t: (-abs(qop.terms[t]), t))
    groups: list[tuple[dict, list]] = []
    for term in terms:
        for basis, members in groups:
            if all(basis.get(q, p) == p for q, p in term):
                basis.update(term)
                members.append(term)
                break
        else:
            groups.append((dict(term), [term]))
    return [members for _, members in groups]


def qubit_metrics(qop) -> dict:
    """Size and cost indicators for measurement (NISQ) and block-encoding (FT) use."""
    coeffs = {t: c for t, c in qop.terms.items() if t}
    one_norm = float(sum(abs(c) for c in coeffs.values()))
    return {
        "n_terms": len(qop.terms),
        "n_qwc_groups": len(qwc_groups(qop)),
        "max_pauli_weight": max((len(t) for t in coeffs), default=0),
        "pauli_one_norm_hartree": one_norm,
        "shots_upper_bound_chemical_accuracy": math.ceil(
            (one_norm / CHEMICAL_ACCURACY_HARTREE) ** 2),
    }


def qiskit_label(term, n_qubits: int) -> str:
    label = ["I"] * n_qubits
    for q, pauli in term:
        label[n_qubits - 1 - q] = pauli
    return "".join(label)


def qubit_hamiltonian_to_dict(qop, ham: MolecularHamiltonian, mapping: str) -> dict:
    n = ham.n_qubits
    terms = []
    for term in sorted(qop.terms, key=lambda t: (len(t), t)):
        coeff = complex(qop.terms[term])
        if abs(coeff.imag) > 1e-10:
            raise RuntimeError(f"Non-Hermitian qubit Hamiltonian: term {term} has coeff {coeff}")
        terms.append({
            "ops": [[q, pauli] for q, pauli in term],
            "coeff": coeff.real,
            "qiskit_label": qiskit_label(term, n),
        })
    return {
        "schema_version": 1,
        "mapping": mapping,
        "n_qubits": n,
        "num_alpha": ham.n_alpha,
        "num_beta": ham.n_beta,
        "qubit_order": QUBIT_ORDER_DOC,
        "qiskit_label_convention": QISKIT_LABEL_DOC,
        "reference_bits": reference_bits(ham, mapping),
        "reference_occupation": ham.spin_orbital_occupation(),
        "energy_unit": "hartree",
        "terms": terms,
    }


def qubit_hamiltonian_from_dict(data: dict):
    of = _openfermion()
    qop = of.QubitOperator()
    for entry in data["terms"]:
        qop += of.QubitOperator(tuple((int(q), p) for q, p in entry["ops"]), entry["coeff"])
    return qop
