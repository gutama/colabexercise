"""Quantum ESPRESSO backend: gamma-point SCF of a molecule in a cubic supercell."""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

from pyscf.data import elements

from ..config import QEConfig
from ..provenance import write_summary

_FUNCTIONAL_ALIASES = {
    "PBE": ("PBE", "PBX PBC", "PBXPBC"),
    "PBESOL": ("PBESOL", "PSX PSC"),
    "PZ": ("PZ", "SLA PZ", "LDA"),
    "LDA": ("PZ", "SLA PZ", "LDA"),
}


def run_qe(config: QEConfig, output: Path) -> dict:
    n_electrons = sum(elements.charge(atom.element) for atom in config.molecule.atoms)
    n_electrons -= config.molecule.charge
    if n_electrons % 2:
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
    functionals = {}
    warnings = []
    for element, pseudo in config.pseudopotentials.items():
        filename = f"{element}.UPF"
        staged = output / "pseudos" / filename
        shutil.copyfile(pseudo.path, staged)
        pseudo_hashes[element] = hashlib.sha256(staged.read_bytes()).hexdigest()
        functional = _upf_functional(staged)
        functionals[element] = functional
        if functional and not _functional_matches(config.xc, functional):
            warnings.append(
                f"{element} pseudopotential was generated with '{functional}' but xc is "
                f"'{config.xc}'; QE will not catch this mismatch"
            )
        species_lines.append(f"{element} {pseudo.mass:.10g} {filename}")
    (output / "pw.in").write_text(_input_deck(config, species_lines))
    with (output / "pw.out").open("w") as log:
        completed = subprocess.run(
            [executable, "-in", "pw.in"], cwd=output, stdout=log,
            stderr=subprocess.STDOUT, check=False,
        )
    text = (output / "pw.out").read_text(errors="replace")
    energies = re.findall(r"!\s+total energy\s*=\s*([-+0-9.EeDd]+)\s+Ry", text)
    if (completed.returncode != 0 or "JOB DONE." not in text
            or "convergence has been achieved" not in text or not energies):
        raise RuntimeError(
            f"Quantum ESPRESSO failed or did not converge; inspect {output / 'pw.out'}")
    energy_ry = float(energies[-1].replace("D", "E").replace("d", "e"))
    version = re.search(r"Program PWSCF\s+v\.(\S+)", text)
    return write_summary(output, {
        "schema_version": 2,
        "backend": "qe",
        "config": config.model_dump(mode="json"),
        "qe_version": version.group(1) if version else "unknown",
        "pseudopotential_sha256": pseudo_hashes,
        "pseudopotential_functionals": functionals,
        "energy_ry": energy_ry,
        "energy_hartree": energy_ry / 2,
        "boundary_conditions": "periodic cubic supercell, gamma point",
        "output_file": "pw.out",
        "warnings": warnings,
        "files": ["summary.json", "pw.in", "pw.out", "pseudos/"],
    })


def _input_deck(config: QEConfig, species_lines: list[str]) -> str:
    positions = "\n".join(
        f"{atom.element} {' '.join(f'{x:.12g}' for x in atom.xyz)}"
        for atom in config.molecule.atoms
    )
    cell = config.cell_angstrom
    return f"""&CONTROL
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


def _upf_functional(path: Path) -> str | None:
    """Functional recorded in a UPF header (v2 attribute or v1 text), if present."""
    head = path.read_text(errors="replace")[:20000]
    match = re.search(r'functional\s*=\s*"([^"]+)"', head, re.IGNORECASE)
    if match is None:
        match = re.search(r"^\s*(.+?)\s+Exchange-Correlation functional", head, re.MULTILINE)
    return " ".join(match.group(1).split()) if match else None


def _functional_matches(xc: str, functional: str) -> bool:
    normalized = " ".join(functional.upper().replace("-", " ").split())
    candidates = _FUNCTIONAL_ALIASES.get(xc.upper(), (xc.upper(),))
    return any(alias in normalized for alias in candidates)
