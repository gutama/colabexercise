"""PySCF backend: HF/DFT, (active-space) integral export, references, qubit Hamiltonian."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pyscf
from pyscf import dft, gto, scf

from .. import hamiltonian as qh
from ..config import PySCFConfig
from ..provenance import write_summary

# The integrals are recomputed from the SCF orbitals; their reference determinant must
# reproduce the SCF energy to well within the SCF convergence threshold.
CONSISTENCY_TOLERANCE = 1e-7


def build_molecule(config: PySCFConfig):
    return gto.M(
        atom=[(atom.element, atom.xyz) for atom in config.molecule.atoms],
        unit="Angstrom",
        charge=config.molecule.charge,
        spin=config.molecule.spin,
        basis=config.basis,
        symmetry=config.symmetry,
        verbose=0,
    )


def mean_field(config: PySCFConfig, mol):
    if config.method == "hf":
        mf = scf.RHF(mol) if mol.spin == 0 else scf.ROHF(mol)
    else:
        mf = dft.RKS(mol) if mol.spin == 0 else dft.ROKS(mol)
        mf.xc = config.xc
    if config.device == "gpu":
        try:
            import gpu4pyscf  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("GPU requested; install a compatible GPU4PySCF/CUDA stack") from exc
        mf = mf.density_fit().to_gpu()
    mf.max_cycle = config.max_cycle
    mf.conv_tol = config.convergence_tolerance
    return mf


def run_pyscf(config: PySCFConfig, output: Path) -> dict:
    mol = build_molecule(config)
    mf = mean_field(config, mol)
    energy = float(mf.kernel())
    if not mf.converged:
        raise RuntimeError(f"PySCF {config.method} did not converge in {config.max_cycle} cycles")
    if config.device == "gpu":
        mf = mf.to_cpu()
    mo_coeff = np.asarray(mf.mo_coeff)
    mo_energy = np.asarray(mf.mo_energy)
    mo_occ = np.asarray(mf.mo_occ)

    space = config.active_space
    core, active = qh.select_orbitals(
        mol.nelectron, mo_occ,
        n_active_electrons=space.n_electrons if space else None,
        n_active_orbitals=space.n_orbitals if space else None,
        orbitals=space.orbitals if space else None,
    )
    ham = qh.build_hamiltonian(mol, mo_coeff, mo_occ, core, active)
    reference_energy = qh.determinant_energy(ham)
    warnings = _warnings(config, mo_energy, active)
    if config.method == "hf" and config.device == "cpu":
        if abs(reference_energy - energy) > CONSISTENCY_TOLERANCE:
            raise RuntimeError(
                f"Exported integrals are inconsistent: reference determinant gives "
                f"{reference_energy:.10f} Ha but SCF gave {energy:.10f} Ha"
            )
    elif config.device == "gpu":
        warnings.append(
            "GPU SCF used density fitting; exported integrals are exact, so the reference "
            "determinant energy differs from the SCF energy by the fitting error"
        )

    files = ["summary.json", "integrals.npz"]
    np.savez_compressed(
        output / "integrals.npz",
        h1=ham.h1,
        eri=ham.eri,
        core_energy=np.float64(ham.core_energy),
        mo_coeff=mo_coeff,
        mo_energy=mo_energy,
        mo_occ=mo_occ,
        active_orbitals=np.asarray(active, dtype=int),
        core_orbitals=np.asarray(core, dtype=int),
    )
    if config.export_fcidump:
        from pyscf.tools import fcidump

        fcidump.from_integrals(
            str(output / "hamiltonian.fcidump"), ham.h1, ham.eri, ham.n_orbitals,
            ham.n_alpha + ham.n_beta, nuc=ham.core_energy, ms=ham.n_alpha - ham.n_beta,
        )
        files.append("hamiltonian.fcidump")

    references = _reference_energies(config, mf, ham, core, active, warnings)
    summary = {
        "schema_version": 2,
        "backend": "pyscf",
        "pyscf_version": pyscf.__version__,
        "config": config.model_dump(mode="json"),
        "energy_hartree": energy,
        "scf_converged": True,
        "nuclear_repulsion_hartree": float(mol.energy_nuc()),
        "core_energy_hartree": ham.core_energy,
        "reference_determinant_energy_hartree": reference_energy,
        "reference_energies_hartree": references,
        "total_spatial_orbitals": int(mo_coeff.shape[1]),
        "num_spatial_orbitals": ham.n_orbitals,
        "num_alpha": ham.n_alpha,
        "num_beta": ham.n_beta,
        "active_space": {"core_orbitals": core, "active_orbitals": active} if space else None,
        "homo_lumo_gap_hartree": _gap(mo_energy, mo_occ),
        "integrals_file": "integrals.npz",
        "integral_convention": (
            "chemist: eri[p,q,r,s] = (pq|rs) over exported spatial orbitals, Hartree; "
            "add core_energy_hartree"
        ),
    }
    if config.qubit is not None:
        summary["qubit_hamiltonian"] = _export_qubit(config, ham, references, output)
        files.append("qubit_hamiltonian.json")
    summary["warnings"] = warnings
    summary["files"] = files
    return write_summary(output, summary)


def _reference_energies(config, mf, ham, core, active, warnings) -> dict:
    references = {}
    frozen = sorted(set(range(np.asarray(mf.mo_coeff).shape[1])) - set(active)) or None
    if "fci" in config.reference_methods:
        references["fci"] = qh.exact_energy(ham)
    if "mp2" in config.reference_methods:
        from pyscf import mp

        references["mp2"] = float(mp.MP2(mf, frozen=frozen).run().e_tot)
    if {"ccsd", "ccsd(t)"} & set(config.reference_methods):
        from pyscf import cc

        mycc = cc.CCSD(mf, frozen=frozen)
        mycc.conv_tol = 1e-10
        mycc.run()
        if not mycc.converged:
            raise RuntimeError("CCSD did not converge")
        if "ccsd" in config.reference_methods:
            references["ccsd"] = float(mycc.e_tot)
        if "ccsd(t)" in config.reference_methods:
            if max(ham.n_alpha, ham.n_beta) >= ham.n_orbitals:
                # PySCF's UCCSD(T) divides by the virtual count; there are no triples anyway.
                raise ValueError("ccsd(t) needs virtual orbitals in both spin channels; "
                                 "use a larger basis or active space")
            references["ccsd(t)"] = float(mycc.e_tot + mycc.ccsd_t())
        if hasattr(mycc, "get_t1_diagnostic"):
            t1 = float(mycc.get_t1_diagnostic())
            references["ccsd_t1_diagnostic"] = t1
            if t1 > 0.02:
                warnings.append(f"CCSD T1 diagnostic {t1:.3f} > 0.02: multireference character")
    return references


def _export_qubit(config, ham, references, output: Path) -> dict:
    options = config.qubit
    qop = qh.qubit_hamiltonian(ham, options.mapping)
    data = qh.qubit_hamiltonian_to_dict(qop, ham, options.mapping)
    expected = qh.determinant_energy(ham)
    on_qubits = qh.basis_state_energy(qop, data["reference_bits"])
    if abs(on_qubits - expected) > 1e-8:
        raise RuntimeError(
            f"Qubit Hamiltonian check failed: reference state gives {on_qubits:.10f} Ha, "
            f"integrals give {expected:.10f} Ha"
        )
    info = {
        "file": "qubit_hamiltonian.json",
        "mapping": options.mapping,
        "n_qubits": ham.n_qubits,
        "reference_bits": data["reference_bits"],
        "reference_energy_hartree": on_qubits,
        **qh.qubit_metrics(qop),
    }
    if ham.n_qubits <= options.validate_up_to_qubits:
        sector = qh.sector_ground_energy(
            qop, options.mapping, ham.n_qubits, ham.n_alpha, ham.n_beta)
        exact = references.get("fci")
        if exact is None:
            exact = qh.exact_energy(ham)
        if abs(sector - exact) > 1e-7:
            raise RuntimeError(
                f"Qubit Hamiltonian check failed: sector ground state {sector:.10f} Ha, "
                f"FCI/CASCI {exact:.10f} Ha"
            )
        info["sector_ground_energy_hartree"] = sector
    else:
        info["sector_ground_energy_hartree"] = None
    (output / "qubit_hamiltonian.json").write_text(json.dumps(data, indent=1) + "\n")
    return info


def _gap(mo_energy, mo_occ):
    occupied = mo_energy[mo_occ > 0]
    virtual = mo_energy[mo_occ == 0]
    if len(occupied) == 0 or len(virtual) == 0:
        return None
    return float(virtual.min() - occupied.max())


def _warnings(config, mo_energy, active) -> list[str]:
    warnings = []
    energies = np.sort(np.asarray(mo_energy)[active])
    degenerate = bool(np.any(np.diff(energies) < 1e-6))
    if degenerate and not config.symmetry and os.environ.get("OMP_NUM_THREADS") != "1":
        warnings.append(
            "Degenerate orbitals: integrals and orbital-dependent quantities can change "
            "between runs under threaded BLAS (energies do not). Set OMP_NUM_THREADS=1 "
            "or symmetry=true for reproducible exports."
        )
    return warnings
