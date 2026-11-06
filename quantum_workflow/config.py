from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Atom(StrictModel):
    element: str = Field(pattern=r"^[A-Z][a-z]?$")
    xyz: tuple[float, float, float]


class Molecule(StrictModel):
    atoms: list[Atom] = Field(min_length=1)
    charge: int = 0
    spin: int = Field(default=0, ge=0, description="Number of alpha minus beta electrons")


class PySCFConfig(StrictModel):
    backend: Literal["pyscf"]
    molecule: Molecule
    basis: str = "sto-3g"
    method: Literal["hf", "dft"] = "hf"
    xc: str = "pbe"
    device: Literal["cpu", "gpu"] = "cpu"
    max_cycle: int = Field(default=100, ge=1)
    convergence_tolerance: float = Field(default=1e-9, gt=0)


class Pseudopotential(StrictModel):
    path: Path
    mass: float = Field(gt=0)


class QEConfig(StrictModel):
    backend: Literal["qe"]
    molecule: Molecule
    pseudopotentials: dict[str, Pseudopotential]
    cell_angstrom: float = Field(default=15.0, gt=0)
    ecutwfc_ry: float = Field(default=50.0, gt=0)
    ecutrho_ry: float = Field(default=400.0, gt=0)
    xc: str = Field(default="PBE", pattern=r"^[A-Za-z0-9_-]+$")
    convergence_tolerance: float = Field(default=1e-8, gt=0)
    max_steps: int = Field(default=100, ge=1)

    @model_validator(mode="after")
    def validate_species(self):
        if self.molecule.spin != 0:
            raise ValueError("QE backend currently supports only spin=0")
        species = {atom.element for atom in self.molecule.atoms}
        if set(self.pseudopotentials) != species:
            raise ValueError("Provide exactly one pseudopotential for each element")
        if self.ecutrho_ry < self.ecutwfc_ry:
            raise ValueError("ecutrho_ry must be at least ecutwfc_ry")
        return self
