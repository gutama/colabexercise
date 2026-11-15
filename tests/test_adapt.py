import json

import numpy as np
import pytest
from conftest import H2, LIH

openfermion = pytest.importorskip("openfermion")

from quantum_workflow import hamiltonian as qh  # noqa: E402
from quantum_workflow.adapt import (  # noqa: E402
    adapt_vqe,
    basis_state,
    excitation_pool,
    run_adapt,
)
from quantum_workflow.workflow import parse_config, run  # noqa: E402


def export(tmp_path, molecule, mapping, name=None):
    out = tmp_path / (name or mapping)
    run(parse_config({"backend": "pyscf", "molecule": molecule, "reference_methods": ["fci"],
                      "qubit": {"mapping": mapping}}), out)
    return out


def test_h2_reaches_fci_with_one_double_excitation(tmp_path):
    report = run_adapt(export(tmp_path, H2, "jordan_wigner"))
    assert report["operators"] == ["2^ 3^ 1 0"]
    assert report["converged"] and report["chemical_accuracy"]
    assert report["energy_hartree"] == pytest.approx(-1.1373060358, abs=1e-9)
    with pytest.raises(ValueError, match="overwrite"):
        run_adapt(tmp_path / "jordan_wigner")


@pytest.mark.slow
def test_lih_jordan_wigner_and_bravyi_kitaev_agree(tmp_path):
    reports = {m: adapt_vqe(json.loads((export(tmp_path, LIH, m) / "qubit_hamiltonian.json")
                                       .read_text()), max_operators=6)
               for m in qh.MAPPINGS}
    jw, bk = reports["jordan_wigner"], reports["bravyi_kitaev"]
    assert jw["operators"] == bk["operators"]
    assert jw["energy_hartree"] == pytest.approx(bk["energy_hartree"], abs=1e-9)
    # Variational, and within chemical accuracy of FCI (-7.8824019323) after 6 operators.
    assert -7.8824019323 < jw["energy_hartree"] < -7.8824019323 + qh.CHEMICAL_ACCURACY_HARTREE


def _gradients(data, pool, mapping, naive_bk=False):
    n = data["n_qubits"]
    hamiltonian = openfermion.get_sparse_operator(qh.qubit_hamiltonian_from_dict(data), n_qubits=n)
    psi = basis_state(data["reference_bits"])
    h_psi = hamiltonian @ psi
    gradients = []
    for _, op in pool:
        mapped = (openfermion.bravyi_kitaev(op) if naive_bk
                  else qh.map_fermion_operator(op, mapping, n))
        matrix = openfermion.get_sparse_operator(mapped, n_qubits=n)
        gradients.append(2 * np.real(np.vdot(h_psi, matrix @ psi)))
    return np.array(gradients)


def test_brillouin_singles_vanish_and_bk_register_size_matters(tmp_path):
    data = json.loads((export(tmp_path, LIH, "bravyi_kitaev") / "qubit_hamiltonian.json")
                      .read_text())
    pool = excitation_pool(data["reference_occupation"])
    singles = [entry for entry in pool if entry[0].count("^") == 1]
    assert np.abs(_gradients(data, singles, "bravyi_kitaev")).max() < 1e-6
    double = [entry for entry in pool if entry[0] == "5^ 10^ 3 2"]
    correct = _gradients(data, double, "bravyi_kitaev")[0]
    naive = _gradients(data, double, "bravyi_kitaev", naive_bk=True)[0]
    assert abs(correct) > 0.05  # a real correlation channel...
    assert abs(naive - correct) > 0.05  # ...lost by sizing the BK tree from the operator
