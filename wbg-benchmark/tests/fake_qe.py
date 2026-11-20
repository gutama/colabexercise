"""Finite, nonphysical fixtures for checking orchestration without a QE executable."""

import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from ase.io import read
from ase.units import Bohr

program = sys.argv[1]
if os.environ.get("WBG_FIXTURE_NONCONVERGENCE") == program:
    print("Convergence has not been achieved")
    print("JOB DONE.")
    raise SystemExit(0)
deck = Path("input.in").read_text()
if program == "pw":
    atoms = read("input.in", format="espresso-in")
    relax = "vc-relax" in deck
    asset = Path("scratch/candidate.save")
    asset.mkdir(parents=True)
    if relax:
        atoms.set_cell(atoms.cell * 1.01, scale_atoms=True)
    root = ET.Element("qes:espresso", {"xmlns:qes": "urn:qe-fixture"})
    out = ET.SubElement(root, "output")
    conv = ET.SubElement(out, "convergence_info")
    for key in ("scf_conv", "opt_conv") if relax else ("scf_conv",):
        ET.SubElement(ET.SubElement(conv, key), "convergence_achieved").text = "true"
    ET.SubElement(ET.SubElement(out, "total_energy"), "etot").text = "-15"
    cell = ET.SubElement(out, "atomic_structure")
    lattice = ET.SubElement(cell, "cell")
    for idx, row in enumerate(atoms.cell, 1):
        ET.SubElement(lattice, f"a{idx}").text = " ".join(str(x / Bohr) for x in row)
    coords = ET.SubElement(cell, "atomic_positions")
    for atom in atoms:
        ET.SubElement(coords, "atom", {"name": atom.symbol}).text = " ".join(
            str(x / Bohr) for x in atom.position
        )
    if not relax:
        bands = ET.SubElement(out, "band_structure")
        entries = {
            "lsda": "false",
            "noncolin": "false",
            "spinorbit": "false",
            "occupations_kind": "fixed",
            "nelec": "2",
            "nks": "2",
        }
        for tag, value in entries.items():
            ET.SubElement(bands, tag).text = value
        gap = 0.14 if re.search(r"input_dft\s*=\s*'HSE'", deck) else 0.08
        for high in (-0.50, -0.49):
            pair = ET.SubElement(bands, "ks_energies")
            ET.SubElement(pair, "eigenvalues").text = f"{high} {-0.49 + gap}"
            ET.SubElement(pair, "occupations").text = "1 0"
    ET.ElementTree(root).write(asset / "data-file-schema.xml")
elif program == "ph":
    assert Path("scratch/candidate.save/data-file-schema.xml").is_file()
    mat = "4 0 0\n0 4 0\n0 0 4\n"
    nat = int(sys.argv[2])
    Path("dyn0").write_text("1 1 1\n1\n0 0 0\n")
    gamma = "q = ( 0 0 0 )\nDielectric Tensor:\n" + mat
    gamma += "Effective Charges E-U: Z_{alpha}{s,beta}\n"
    gamma += "".join(f"atom # {i}\n{mat}" for i in range(1, nat + 1))
    Path("dyn1").write_text(gamma)
elif program == "q2r":
    assert Path("dyn0").exists() and Path("dyn1").exists()
    Path("forces.fc").write_text("fixture IFCs\n")
elif program == "matdyn":
    assert Path("forces.fc").exists()
    nat = int(sys.argv[2])
    nqs = int(deck.split("/\n")[1].splitlines()[0])
    header = f" &plot nbnd={3 * nat}, nks={nqs} /\n"
    values = [os.environ.get("WBG_FIXTURE_MIN_CM1", "-10")] + ["230"] * (3 * nat - 1)
    rows = (" ".join(values[i : i + 6]) for i in range(0, len(values), 6))
    text = "0 0 0 0\n" + "\n".join(rows) + "\n"
    Path("phonons.freq").write_text(header + text * nqs)
else:
    raise RuntimeError(program)
print(f"Program {program} TEST FIXTURE, log and results are synthetic")
print("JOB DONE.")
