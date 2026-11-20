"""Quantum ESPRESSO input generation and parsers for the stage outputs."""

import itertools
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from ase.io.espresso import write_espresso_in
from ase.units import Bohr, Hartree
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

CM1_TO_THZ = 0.0299792458
PREFIX = "candidate"


def pw_kmesh(structure: Structure, qe, job: str) -> tuple[int, int, int]:
    if job == "hse":
        return qe.hse_kmesh
    return tuple(
        q * math.ceil(max(k, qe.pbe_k_length_angstrom / length) / q)
        for k, q, length in zip(qe.pbe_kmesh, qe.dfpt_qmesh, structure.lattice.abc, strict=True)
    )


def prepare_pw(path: Path, structure: Structure, qe, names: dict, job: str) -> None:
    relax = job == "relax"
    hybrid = job == "hse"
    control = {
        "calculation": "vc-relax" if relax else "scf",
        "prefix": PREFIX,
        "outdir": "./scratch",
        "pseudo_dir": "./pseudo",
        "tstress": True,
        "tprnfor": True,
        "disk_io": "low",
    }
    if relax:
        control["forc_conv_thr"] = qe.force_thr_ry_bohr
    system = {
        "ecutwfc": qe.ecutwfc_ry,
        "ecutrho": qe.ecutwfc_ry * qe.ecutrho_ratio,
        "occupations": "fixed",
        "input_dft": "HSE" if hybrid else qe.xc,
    }
    if not relax:
        system["nbnd"] = qe.nbnd
    if hybrid:
        system.update(
            exx_fraction=0.25,
            screening_parameter=0.106,
            exxdiv_treatment="gygi-baldereschi",
            x_gamma_extrapolation=True,
            **{f"nqx{i}": n for i, n in enumerate(qe.exchange_qmesh, 1)},
        )
    data = {"control": control, "system": system, "electrons": {"conv_thr": qe.conv_thr_ry}}
    if relax:
        data["ions"] = {"ion_dynamics": "bfgs"}
        data["cell"] = {"cell_dynamics": "bfgs", "press": 0.0, "press_conv_thr": qe.press_thr_kbar}
    atoms = AseAtomsAdaptor.get_atoms(structure)
    with path.open("w") as handle:
        write_espresso_in(
            handle,
            atoms,
            input_data=data,
            pseudopotentials=names,
            kpts=pw_kmesh(structure, qe, job),
            crystal_coordinates=True,
        )


def prepare_ph(path: Path, qe) -> None:
    n1, n2, n3 = qe.dfpt_qmesh
    path.write_text(
        f"DFPT mesh and Gamma dielectric response\n"
        f"&inputph\n prefix='{PREFIX}', outdir='./scratch', fildyn='dyn',\n"
        f" fildvscf='dvscf', lqdir=.true., ldisp=.true., nq1={n1}, nq2={n2}, nq3={n3},\n"
        f" epsil=.true., zeu=.true., tr2_ph={qe.tr2_ph:.12e},\n/\n"
    )


def prepare_q2r(path: Path) -> None:
    path.write_text("&input\n fildyn='dyn', flfrc='forces.fc', zasr='crystal',\n/\n")


def prepare_matdyn(path: Path, qe) -> None:
    mesh = qe.sample_qmesh
    points = itertools.product(*(range(n) for n in mesh))
    lines = [
        "&input",
        " flfrc='forces.fc', flfrq='phonons.freq', asr='crystal',",
        " q_in_cryst_coord=.true.,",
        "/",
        str(math.prod(mesh)),
    ]
    lines.extend(" ".join(f"{j / n:.12f}" for j, n in zip(p, mesh, strict=True)) for p in points)
    path.write_text("\n".join(lines) + "\n")


def finite_numbers(value: str) -> list[float]:
    numbers = [float(x.replace("D", "E").replace("d", "e")) for x in value.split()]
    if not all(math.isfinite(x) for x in numbers):
        raise ValueError("Nonfinite numeric data in QE output")
    return numbers


def pw_result(path: Path, relax: bool = False) -> tuple[dict, Structure | None]:
    # QE's data-file-schema.xml stores energy in Hartree and positions/cell in Bohr.
    root = ET.parse(path).getroot()
    for node in root.iter():
        node.tag = node.tag.split("}")[-1]
    out = root.find("output")
    if out is None:
        raise ValueError("QE XML has no output")
    key = "opt_conv" if relax else "scf_conv"
    if out.findtext(f"convergence_info/{key}/convergence_achieved") != "true":
        raise ValueError(f"QE XML has no confirmed {key} convergence")
    (energy,) = finite_numbers(out.findtext("total_energy/etot", ""))
    result = {"total_energy_ev": energy * Hartree}
    if relax:
        coords = out.find("atomic_structure")
        if coords is None:
            raise ValueError("QE XML is missing the final atomic structure")
        cell = [finite_numbers(coords.findtext(f"cell/a{i}", "")) for i in (1, 2, 3)]
        atoms = coords.findall("atomic_positions/atom")
        structure = Structure(
            np.asarray(cell) * Bohr,
            [a.attrib["name"] for a in atoms],
            np.asarray([finite_numbers(a.text or "") for a in atoms]) * Bohr,
            coords_are_cartesian=True,
        )
        return result, structure

    bands = out.find("band_structure")
    if bands is None or any(
        bands.findtext(key) != "false" for key in ("lsda", "noncolin", "spinorbit")
    ):
        raise ValueError("Only nonmagnetic, scalar-relativistic QE band outputs are supported")
    if bands.findtext("occupations_kind") != "fixed":
        raise ValueError("Fixed occupations required for the sampled-gap parser")
    (nelec,) = finite_numbers(bands.findtext("nelec", ""))
    if nelec <= 0 or not (nelec / 2).is_integer():
        raise ValueError("Only fixed, even electron counts are supported")
    filled = int(nelec / 2)
    edges = []
    for item in bands.findall("ks_energies"):
        levels = finite_numbers(item.findtext("eigenvalues", ""))
        occ = finite_numbers(item.findtext("occupations", ""))
        if (
            len(levels) <= filled
            or len(occ) != len(levels)
            or min(occ[:filled]) < 1 - 1e-6
            or max(map(abs, occ[filled:])) > 1e-6
        ):
            raise ValueError("Missing empty bands or inconsistent fixed occupations")
        edges.append((max(levels[:filled]), min(levels[filled:])))
    if not edges or len(edges) != int(bands.findtext("nks", "0")):
        raise ValueError("Incomplete k-point eigenvalue data")
    vbm, cbm = max(x[0] for x in edges), min(x[1] for x in edges)
    result.update(
        sampled_gap_ev=max(0.0, (cbm - vbm) * Hartree),
        vbm_ev=vbm * Hartree,
        cbm_ev=cbm * Hartree,
        direct_sampled_gap_ev=max(0, min(c - v for v, c in edges) * Hartree),
        sampled_kpoints=len(edges),
        band_energy_source="QE XML (Hartree)",
    )
    return result, None


def phonon_result(path: Path, nat: int, mesh: tuple) -> dict:
    """Read matdyn flfrq output (cm^-1). Support both 3- and 4-column q headers."""
    lines = iter(line for line in path.read_text().splitlines() if line.strip())
    header = next(lines, "")
    match = re.search(r"nbnd\s*=\s*(\d+),\s*nks\s*=\s*(\d+)", header)
    if not match or tuple(map(int, match.groups())) != (3 * nat, math.prod(mesh)):
        raise ValueError("matdyn mode or q-point count differs from the plan")
    low, high = math.inf, -math.inf
    for _ in range(math.prod(mesh)):
        if len(finite_numbers(next(lines, ""))) not in (3, 4):
            raise ValueError("Incomplete matdyn q-point record")
        modes = []
        while len(modes) < 3 * nat:
            row = next(lines, "")
            if not row:
                raise ValueError("Incomplete matdyn frequency record")
            modes.extend(finite_numbers(row))
        if len(modes) != 3 * nat:
            raise ValueError("Unexpected matdyn frequency count")
        low, high = min(low, *modes), max(high, *modes)
    if next(lines, None) is not None:
        raise ValueError("Unexpected trailing matdyn data")
    return {
        "min_phonon_thz": low * CM1_TO_THZ,
        "max_phonon_thz": high * CM1_TO_THZ,
        "sample_qmesh": list(mesh),
        "frequency_source": "matdyn flfrq (cm^-1)",
    }


def gamma_response(path: Path, nat: int) -> dict:
    """Read electronic permittivity and Born charges from a text Gamma matrix."""
    data = path.read_text()
    point = re.search(r"q\s*=\s*\(([^)]+)\)", data)
    if point is None or finite_numbers(point[1]) != [0, 0, 0]:
        raise ValueError("The first DFPT dynamical matrix must be at Gamma")
    dielectric = data.partition("Dielectric Tensor:")[2].split("Effective Charges E-U:", 1)
    if len(dielectric) != 2:
        raise ValueError("Gamma response is missing dielectric tensor or Born charges")

    def matrix(block):
        rows = [finite_numbers(line) for line in block.splitlines() if line.strip()]
        if len(rows) != 3 or any(len(row) != 3 for row in rows):
            raise ValueError("Expected a 3 by 3 Gamma response tensor")
        return rows

    eps = matrix(dielectric[0])
    if not np.allclose(eps, np.transpose(eps), atol=1e-6) or np.min(np.linalg.eigvalsh(eps)) <= 0:
        raise ValueError("Electronic dielectric tensor must be symmetric positive definite")
    charges = re.findall(
        r"atom\s*#\s*(\d+)[ \t]*\n((?:[^\n]*\n){3})",
        dielectric[1].split("Effective Charges U-E:", 1)[0],
    )
    if [int(i) for i, _ in charges] != list(range(1, nat + 1)):
        raise ValueError("Incomplete Born effective charges")
    return {
        "relative_electronic_epsilon_cartesian": eps,
        "born_charges_e_cartesian": [matrix(tensor) for _, tensor in charges],
    }
