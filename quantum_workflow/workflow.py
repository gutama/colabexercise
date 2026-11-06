import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyscf
from pyscf import ao2mo, dft, gto, scf

from .config import PySCFConfig, QEConfig


def load_config(path: str | Path) -> PySCFConfig | QEConfig:
    config_path = Path(path).resolve()
    data = json.loads(config_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a JSON object")
    if data.get("backend") == "pyscf":
        return PySCFConfig.model_validate(data)
    if data.get("backend") == "qe":
        config = QEConfig.model_validate(data)
        for pseudo in config.pseudopotentials.values():
            if not pseudo.path.is_absolute():
                pseudo.path = config_path.parent / pseudo.path
        return config
    raise ValueError("backend must be 'pyscf' or 'qe'")


def run(config: PySCFConfig | QEConfig, output_dir: str | Path) -> dict:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    if isinstance(config, PySCFConfig):
        return run_pyscf(config, output)
    return run_qe(config, output)


def make_molecule(config: PySCFConfig | QEConfig):
    return gto.M(
        atom=[(atom.element, atom.xyz) for atom in config.molecule.atoms],
        unit="Angstrom",
        charge=config.molecule.charge,
        spin=config.molecule.spin,
        basis=config.basis if isinstance(config, PySCFConfig) else "sto-3g",
        verbose=0,
    )


def write_summary(output: Path, summary: dict) -> dict:
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def run_pyscf(config: PySCFConfig, output: Path) -> dict:
    mol = make_molecule(config)
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
    energy = float(mf.kernel())
    if not mf.converged:
        raise RuntimeError(f"PySCF {config.method} did not converge in {config.max_cycle} cycles")
    if config.device == "gpu":
        mf = mf.to_cpu()
    coeff = np.asarray(mf.mo_coeff)
    h1 = coeff.T @ mf.get_hcore() @ coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, coeff), coeff.shape[1])
    np.savez_compressed(
        output / "integrals.npz",
        h1=h1,
        eri=eri,
        mo_coeff=coeff,
        mo_energy=np.asarray(mf.mo_energy),
        mo_occ=np.asarray(mf.mo_occ),
    )
    return write_summary(output, {
        "schema_version": 1,
        "backend": "pyscf",
        "pyscf_version": pyscf.__version__,
        "config": config.model_dump(mode="json"),
        "energy_hartree": energy,
        "nuclear_repulsion_hartree": float(mol.energy_nuc()),
        "num_spatial_orbitals": int(coeff.shape[1]),
        "num_alpha": int(mol.nelec[0]),
        "num_beta": int(mol.nelec[1]),
        "integrals_file": "integrals.npz",
        "integral_convention": "chemist: eri[p,q,r,s] = (pq|rs), spatial orbitals, Hartree",
    })


def run_qe(config: QEConfig, output: Path) -> dict:
    mol = make_molecule(config)
    if mol.nelectron % 2:
        raise ValueError("QE backend requires an even electron count")
    executable = shutil.which("pw.x")
    if executable is None:
        raise RuntimeError("Quantum ESPRESSO pw.x was not found on PATH")
    for pseudo in config.pseudopotentials.values():
        if not pseudo.path.is_file():
            raise FileNotFoundError(f"Pseudopotential not found: {pseudo.path}")
    (output / "pseudos").mkdir()
    (output / "scratch").mkdir()
    species_lines = []
    pseudo_hashes = {}
    for element, pseudo in config.pseudopotentials.items():
        filename = f"{element}.UPF"
        staged = output / "pseudos" / filename
        shutil.copyfile(pseudo.path, staged)
        pseudo_hashes[element] = hashlib.sha256(staged.read_bytes()).hexdigest()
        species_lines.append(f"{element} {pseudo.mass:.10g} {filename}")
    positions = "\n".join(
        f"{atom.element} {' '.join(f'{x:.12g}' for x in atom.xyz)}"
        for atom in config.molecule.atoms
    )
    cell = config.cell_angstrom
    qe_input = f"""&CONTROL
  calculation = 'scf', prefix = 'molecule', pseudo_dir = './pseudos', outdir = './scratch'
/
&SYSTEM
  ibrav = 0, nat = {len(config.molecule.atoms)}, ntyp = {len(species_lines)},
  ecutwfc = {config.ecutwfc_ry:.12g}, ecutrho = {config.ecutrho_ry:.12g},
  tot_charge = {config.molecule.charge}, occupations = 'fixed', input_dft = '{config.xc}'
/
&ELECTRONS
  conv_thr = {config.convergence_tolerance:.12g}, electron_maxstep = {config.max_steps}
/
ATOMIC_SPECIES
{chr(10).join(species_lines)}
ATOMIC_POSITIONS angstrom
{positions}
K_POINTS gamma
CELL_PARAMETERS angstrom
{cell:.12g} 0 0
0 {cell:.12g} 0
0 0 {cell:.12g}
"""
    (output / "pw.in").write_text(qe_input)
    with (output / "pw.out").open("w") as log:
        completed = subprocess.run(
            [executable, "-in", "pw.in"], cwd=output, stdout=log,
            stderr=subprocess.STDOUT, check=False,
        )
    text = (output / "pw.out").read_text(errors="replace")
    energies = re.findall(r"!\s+total energy\s*=\s*([-+0-9.EeDd]+)\s+Ry", text)
    if (completed.returncode != 0 or "JOB DONE." not in text
            or "convergence has been achieved" not in text or not energies):
        raise RuntimeError(f"Quantum ESPRESSO failed or did not converge; inspect {output / 'pw.out'}")
    energy_ry = float(energies[-1].replace("D", "E").replace("d", "e"))
    version = re.search(r"Program PWSCF\s+v\.(\S+)", text)
    return write_summary(output, {
        "schema_version": 1,
        "backend": "qe",
        "config": config.model_dump(mode="json"),
        "qe_version": version.group(1) if version else "unknown",
        "pseudopotential_sha256": pseudo_hashes,
        "energy_ry": energy_ry,
        "energy_hartree": energy_ry / 2,
        "boundary_conditions": "periodic cubic supercell, gamma point",
        "output_file": "pw.out",
    })
