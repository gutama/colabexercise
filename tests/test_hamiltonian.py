"""Conversion traps, each checked against an independent anchor (see hamiltonian.py)."""
import json

import numpy as np
import pytest
from conftest import H2, HEH_CATION, LIH, WATER

openfermion = pytest.importorskip("openfermion")

from pyscf import ao2mo, dft, gto, mcscf, scf  # noqa: E402

from quantum_workflow import hamiltonian as qh  # noqa: E402
from quantum_workflow.workflow import parse_config, run  # noqa: E402


def mean_field(molecule, method=scf.RHF, basis="sto-3g"):
    mol = gto.M(atom=[(a["element"], a["xyz"]) for a in molecule["atoms"]], basis=basis,
                charge=molecule["charge"], spin=molecule["spin"], verbose=0)
    return mol, method(mol).run()


def full_space(mol, mf):
    core, active = qh.select_orbitals(mol.nelectron, mf.mo_occ)
    return qh.build_hamiltonian(mol, mf.mo_coeff, mf.mo_occ, core, active)


def test_spin_tensors_match_reference_loop():
    """Vectorized chemist->physicist conversion equals the explicit index loop."""
    mol, mf = mean_field(HEH_CATION)
    ham = full_space(mol, mf)
    _, h1_so, h2_so = qh.spin_orbital_tensors(ham)
    n = ham.n_orbitals
    expected = np.zeros_like(h2_so)
    for p, q, r, s in np.ndindex(n, n, n, n):
        for sigma in (0, 1):
            for tau in (0, 1):
                expected[2 * p + sigma, 2 * r + tau, 2 * s + tau, 2 * q + sigma] = \
                    0.5 * ham.eri[p, q, r, s]
    assert np.array_equal(h2_so, expected)
    assert np.array_equal(h1_so[::2, ::2], ham.h1) and not h1_so[::2, 1::2].any()


@pytest.mark.parametrize("mapping", qh.MAPPINGS)
@pytest.mark.parametrize("molecule, scf_energy, fci_energy", [
    (H2, -1.1169989968, -1.1373060358),
    (HEH_CATION, -2.8413824898, -2.8510240300),
    (LIH, -7.8620238601, -7.8824019323),
])
def test_qubit_hamiltonian_anchors(mapping, molecule, scf_energy, fci_energy):
    mol, mf = mean_field(molecule)
    ham = full_space(mol, mf)
    qop = qh.qubit_hamiltonian(ham, mapping)
    bits = qh.reference_bits(ham, mapping)
    assert qh.basis_state_energy(qop, bits) == pytest.approx(scf_energy, abs=1e-9)
    sector = qh.sector_ground_energy(qop, mapping, ham.n_qubits, ham.n_alpha, ham.n_beta)
    assert sector == pytest.approx(fci_energy, abs=1e-9)
    assert qh.exact_energy(ham) == pytest.approx(fci_energy, abs=1e-9)


def test_h2_term_and_group_counts():
    mol, mf = mean_field(H2)
    ham = full_space(mol, mf)
    jw = qh.qubit_metrics(qh.qubit_hamiltonian(ham, "jordan_wigner"))
    bk = qh.qubit_metrics(qh.qubit_hamiltonian(ham, "bravyi_kitaev"))
    assert (jw["n_terms"], jw["n_qwc_groups"]) == (15, 5)
    assert (bk["n_terms"], bk["n_qwc_groups"]) == (15, 3)
    assert jw["pauli_one_norm_hartree"] == pytest.approx(bk["pauli_one_norm_hartree"])


def test_bravyi_kitaev_reference_is_a_different_bitstring():
    mol, mf = mean_field(LIH)
    ham = full_space(mol, mf)
    assert qh.reference_bits(ham, "jordan_wigner") == [1, 1, 1, 1] + [0] * 8
    assert qh.reference_bits(ham, "bravyi_kitaev") == [1, 0, 1, 0] + [0] * 8
    # The trap: BK Hamiltonian evaluated on the JW bitstring.
    wrong = qh.basis_state_energy(qh.qubit_hamiltonian(ham, "bravyi_kitaev"),
                                  qh.reference_bits(ham, "jordan_wigner"))
    # Not variational in the orbitals, so SCF convergence error enters at first order.
    assert wrong == pytest.approx(-5.7359398889, abs=1e-7)


def test_unrestricted_fock_space_minimum_is_the_wrong_answer():
    mol, mf = mean_field(HEH_CATION)
    ham = full_space(mol, mf)
    qop = qh.qubit_hamiltonian(ham, "jordan_wigner")
    fock_minimum = openfermion.linalg.get_ground_state(openfermion.get_sparse_operator(qop))[0]
    assert fock_minimum == pytest.approx(-3.0134857193, abs=1e-9)  # neutral HeH
    assert qh.sector_ground_energy(qop, "jordan_wigner", 4, 1, 1) == pytest.approx(-2.85102403)


def test_qiskit_labels_are_little_endian_numpy_check():
    """Build the matrix from qiskit_label strings with plain numpy; HeH+ breaks reversal."""
    mol, mf = mean_field(HEH_CATION)
    ham = full_space(mol, mf)
    data = qh.qubit_hamiltonian_to_dict(qh.qubit_hamiltonian(ham, "jordan_wigner"), ham,
                                        "jordan_wigner")
    paulis = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
              "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1])}

    def matrix(label):
        out = np.array([[1.0]])
        for char in label:  # leftmost character = highest qubit (little-endian)
            out = np.kron(out, paulis[char])
        return out

    h = sum(t["coeff"] * matrix(t["qiskit_label"]) for t in data["terms"])
    index = sum(bit << q for q, bit in enumerate(data["reference_bits"]))  # qubit q -> 2^q
    assert h[index, index].real == pytest.approx(-2.8413824898, abs=1e-9)
    reversed_h = sum(t["coeff"] * matrix(t["qiskit_label"][::-1]) for t in data["terms"])
    assert reversed_h[index, index].real == pytest.approx(-0.5713302151, abs=1e-9)


def test_qiskit_roundtrip_reproduces_rhf():
    qiskit = pytest.importorskip("qiskit")
    from qiskit.quantum_info import SparsePauliOp, Statevector

    mol, mf = mean_field(HEH_CATION)
    ham = full_space(mol, mf)
    for mapping in qh.MAPPINGS:
        data = qh.qubit_hamiltonian_to_dict(qh.qubit_hamiltonian(ham, mapping), ham, mapping)
        op = SparsePauliOp.from_list([(t["qiskit_label"], t["coeff"]) for t in data["terms"]])
        circuit = qiskit.QuantumCircuit(data["n_qubits"])
        for qubit, bit in enumerate(data["reference_bits"]):
            if bit:
                circuit.x(qubit)
        energy = Statevector(circuit).expectation_value(op).real
        assert energy == pytest.approx(mf.e_tot, abs=1e-9), mapping


@pytest.mark.parametrize("method", [scf.RHF, dft.RKS])
def test_frozen_core_matches_pyscf_casci(method):
    mol, mf = mean_field(WATER, method)
    core, active = qh.select_orbitals(mol.nelectron, mf.mo_occ, 4, 4)
    ham = qh.build_hamiltonian(mol, mf.mo_coeff, mf.mo_occ, core, active)
    casci = mcscf.CASCI(mf, 4, 4)
    h1, ecore = casci.get_h1eff()
    assert ham.core_energy == pytest.approx(ecore, abs=1e-10)
    assert np.allclose(ham.h1, h1, atol=1e-10)
    assert np.allclose(ham.eri, ao2mo.restore(1, casci.get_h2eff(), 4), atol=1e-10)
    assert qh.exact_energy(ham) == pytest.approx(casci.kernel()[0], abs=1e-8)


@pytest.mark.parametrize("mapping", qh.MAPPINGS)
def test_open_shell_triplet_reference_and_sector(tmp_path, mapping):
    """O atom triplet: N_alpha - N_beta = 2, so 'first N spin-orbitals' would be wrong."""
    config = parse_config({
        "backend": "pyscf", "molecule": {"atoms": [{"element": "O", "xyz": [0, 0, 0]}], "spin": 2},
        "reference_methods": ["fci"], "qubit": {"mapping": mapping},
    })
    summary = run(config, tmp_path / mapping)
    info = summary["qubit_hamiltonian"]
    assert (summary["num_alpha"], summary["num_beta"]) == (5, 3)
    assert info["reference_energy_hartree"] == pytest.approx(summary["energy_hartree"], abs=1e-8)
    assert info["sector_ground_energy_hartree"] == pytest.approx(
        summary["reference_energies_hartree"]["fci"], abs=1e-8)
    occupation = json.loads((tmp_path / mapping / "qubit_hamiltonian.json").read_text())[
        "reference_occupation"]
    assert occupation == [1, 1, 1, 1, 1, 1, 1, 0, 1, 0]


def test_json_roundtrip_and_hermiticity(tmp_path):
    summary = run(parse_config({"backend": "pyscf", "molecule": LIH,
                                "qubit": {"mapping": "bravyi_kitaev"}}), tmp_path / "lih")
    data = json.loads((tmp_path / "lih/qubit_hamiltonian.json").read_text())
    qop = qh.qubit_hamiltonian_from_dict(data)
    assert qop == openfermion.hermitian_conjugated(qop)
    assert qh.basis_state_energy(qop, data["reference_bits"]) == pytest.approx(
        summary["energy_hartree"], abs=1e-9)
    assert data["n_qubits"] == 12 and len(data["terms"]) == summary["qubit_hamiltonian"]["n_terms"]
