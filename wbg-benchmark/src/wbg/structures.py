"""Calibration seeds, ASE/QE interchange, and primitive-cell matching."""

import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

from ase.build import bulk
from pymatgen.analysis.structure_matcher import SpeciesComparator, StructureMatcher
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def load_structure(source: str, base: Path) -> Structure:
    if source == "builtin:gan-wurtzite":
        return AseAtomsAdaptor.get_structure(bulk("GaN", "wurtzite", a=3.189, c=5.185, u=0.377))
    if source == "builtin:sic-4h":
        # Ideal 4H ABAC stacking (four Si-C bilayers). Relax internal coordinates.
        xy = [(0, 0), (1 / 3, 2 / 3), (0, 0), (2 / 3, 1 / 3)]
        sites = [(x, y, layer / 4 + dz) for layer, (x, y) in enumerate(xy) for dz in (0, 3 / 16)]
        return Structure(Lattice.hexagonal(3.073, 10.053), ["Si", "C"] * 4, sites)
    structure = Structure.from_file(base / source)
    if not structure.is_ordered:
        raise ValueError("Supply an ordered structure with fully occupied sites")
    return structure


def canonical(structure: Structure) -> Structure:
    primitive = SpacegroupAnalyzer(structure, symprec=0.05).find_primitive()
    reduced = structure if primitive is None else primitive
    return reduced.get_reduced_structure(reduction_algo="niggli")


def fingerprint(structure: Structure) -> dict:
    symmetry = SpacegroupAnalyzer(structure, symprec=0.05).get_symmetry_dataset()
    if symmetry is None:
        raise ValueError("Could not determine structure symmetry")
    return {
        "formula": structure.composition.reduced_formula,
        "sites": len(structure),
        "space_group": int(symmetry.number),
        "wyckoff_sites": sorted(
            f"{site.specie.symbol}:{letter}"
            for site, letter in zip(structure, symmetry.wyckoffs, strict=True)
        ),
    }


class DuplicateIndex:
    """Group by composition and confirm with a geometric match of reduced primitives."""

    def __init__(self):
        self.groups = defaultdict(list)
        self.matcher = StructureMatcher(
            ltol=0.12,
            stol=0.20,
            angle_tol=3,
            primitive_cell=False,
            scale=False,
            attempt_supercell=False,
            comparator=SpeciesComparator(),
        )

    def add(self, identifier: str, structure: Structure) -> dict:
        primitive = canonical(structure)
        info = fingerprint(primitive)
        group = self.groups[info["formula"]]
        for other_id, other in group:
            if self.matcher.fit(primitive, other):
                return {"id": identifier, "duplicate_of": other_id, **info}
        group.append((identifier, primitive))
        return {"id": identifier, "duplicate_of": None, **info}


def audit_mdb(path: Path, limit: int | None = None) -> dict:
    """Read CIF members directly and keep MDB labels separate from QE evidence."""
    import csv

    index = DuplicateIndex()
    matches, errors = [], []
    with ZipFile(path) as archive:
        rows = list(
            csv.DictReader(archive.read("mdb-materials/materials.csv").decode().splitlines())
        )
        for row in rows[:limit]:
            key = row["id"]
            try:
                if not re.fullmatch(r"[0-9a-f]{8}", key):
                    raise ValueError("Invalid MDB candidate ID")
                cif = archive.read(f"mdb-materials/structures/{key}.cif").decode()
                with warnings.catch_warnings(record=True) as notices:
                    struct = Structure.from_str(cif, fmt="cif")
                if not struct.is_ordered:
                    raise ValueError("Disordered structure")
                matches.append(
                    {
                        **index.add(key, struct),
                        "mdb_provenance": row["provenance"],
                        "parse_warnings": [str(item.message) for item in notices],
                    }
                )
            except (ValueError, KeyError, TypeError) as exc:
                errors.append({"id": key, "error": str(exc)})
    return {
        "archive_rows": len(rows),
        "archive_provenance": dict(Counter(row["provenance"] for row in rows)),
        "examined": len(rows[:limit]),
        "duplicates": sum(item["duplicate_of"] is not None for item in matches),
        "matches": matches,
        "errors": errors,
    }
