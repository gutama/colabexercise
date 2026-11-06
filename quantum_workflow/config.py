"""Validated JSON configuration models.

A configuration is a JSON object whose ``backend`` key selects the model. Unknown
keys are rejected and assignments are re-validated, so a typo or an invalid edit
fails loudly instead of being silently ignored.

To add a backend: subclass ``BackendConfig`` with a unique ``backend`` literal,
add it to ``Config`` below, and register a runner in ``quantum_workflow.backends``.
"""
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_assignment=True)


class Atom(StrictModel):
    element: str = Field(pattern=r"^[A-Z][a-z]?$")
    xyz: tuple[float, float, float] = Field(description="Cartesian position in Angstrom")


class Molecule(StrictModel):
    atoms: list[Atom] = Field(min_length=1)
    charge: int = 0
    spin: int = Field(default=0, ge=0, description="2S = N_alpha - N_beta, NOT the multiplicity")


class BackendConfig(StrictModel):
    """Common base: every backend has a molecule and may resolve relative paths."""

    molecule: Molecule

    def resolve_paths(self, base_dir: Path) -> None:
        """Make file references relative to the config file absolute. Default: none."""


class ActiveSpace(StrictModel):
    n_electrons: int = Field(ge=1, description="Active electrons (alpha + beta)")
    n_orbitals: int = Field(ge=1, description="Active spatial orbitals; qubits = 2 x this")
    orbitals: list[int] | None = Field(
        default=None,
        description="Explicit 0-based MO indices. Default: the window around the Fermi level.",
    )

    @model_validator(mode="after")
    def check_counts(self):
        if self.n_electrons > 2 * self.n_orbitals:
            raise ValueError("active_space.n_electrons exceeds 2 * n_orbitals")
        if self.orbitals is not None:
            if len(self.orbitals) != self.n_orbitals:
                raise ValueError("active_space.orbitals must list exactly n_orbitals indices")
            if len(set(self.orbitals)) != len(self.orbitals) or min(self.orbitals) < 0:
                raise ValueError("active_space.orbitals must be distinct non-negative indices")
        return self


class QubitOptions(StrictModel):
    mapping: Literal["jordan_wigner", "bravyi_kitaev"] = "jordan_wigner"
    validate_up_to_qubits: int = Field(
        default=16, ge=0, le=24,
        description="Exact particle-number-sector diagonalization is run up to this size",
    )


ReferenceMethod = Literal["fci", "mp2", "ccsd", "ccsd(t)"]


class PySCFConfig(BackendConfig):
    backend: Literal["pyscf"]
    basis: str = "sto-3g"
    method: Literal["hf", "dft"] = "hf"
    xc: str = "pbe"
    device: Literal["cpu", "gpu"] = "cpu"
    symmetry: bool = Field(
        default=False,
        description="Point-group symmetry; makes degenerate orbitals reproducible across runs",
    )
    max_cycle: int = Field(default=100, ge=1)
    convergence_tolerance: float = Field(default=1e-9, gt=0)
    active_space: ActiveSpace | None = None
    reference_methods: list[ReferenceMethod] = Field(
        default_factory=list,
        description="Classical reference energies. 'fci' diagonalizes the exported Hamiltonian.",
    )
    export_fcidump: bool = True
    qubit: QubitOptions | None = Field(
        default=None, description="Export a qubit Hamiltonian (requires the 'qubit' extra)",
    )

    @model_validator(mode="after")
    def check_options(self):
        methods = self.reference_methods
        if len(set(methods)) != len(methods):
            raise ValueError("reference_methods contains duplicates")
        post_hf = set(methods) - {"fci"}
        if post_hf and self.method != "hf":
            raise ValueError("mp2/ccsd/ccsd(t) references require method='hf'")
        if post_hf and self.device != "cpu":
            raise ValueError("mp2/ccsd/ccsd(t) references are only supported with device='cpu'")
        space = self.active_space
        if space is not None:
            if space.n_electrons < self.molecule.spin:
                raise ValueError("active_space.n_electrons must be at least molecule.spin")
            if (space.n_electrons - self.molecule.spin) % 2:
                raise ValueError(
                    "active_space.n_electrons and molecule.spin must have equal parity")
        return self


class Pseudopotential(StrictModel):
    path: Path
    mass: float = Field(gt=0)


class QEConfig(BackendConfig):
    backend: Literal["qe"]
    pseudopotentials: dict[str, Pseudopotential]
    cell_angstrom: float = Field(default=15.0, gt=0)
    ecutwfc_ry: float = Field(default=50.0, gt=0)
    ecutrho_ry: float = Field(default=400.0, gt=0)
    xc: str = Field(default="PBE", pattern=r"^[A-Za-z0-9_-]+$")
    convergence_tolerance: float = Field(default=1e-8, gt=0)
    max_steps: int = Field(default=100, ge=1)

    @model_validator(mode="after")
    def validate_system(self):
        if self.molecule.spin != 0:
            raise ValueError("QE backend currently supports only spin=0")
        species = {atom.element for atom in self.molecule.atoms}
        if set(self.pseudopotentials) != species:
            raise ValueError("Provide exactly one pseudopotential for each element")
        if self.ecutrho_ry < self.ecutwfc_ry:
            raise ValueError("ecutrho_ry must be at least ecutwfc_ry")
        for atom in self.molecule.atoms:
            if not all(0 <= x < self.cell_angstrom for x in atom.xyz):
                raise ValueError(
                    f"{atom.element} at {atom.xyz} lies outside the cubic cell "
                    f"[0, {self.cell_angstrom}) Angstrom"
                )
        return self

    def resolve_paths(self, base_dir: Path) -> None:
        for pseudo in self.pseudopotentials.values():
            if not pseudo.path.is_absolute():
                pseudo.path = base_dir / pseudo.path


Config = Annotated[Union[PySCFConfig, QEConfig], Field(discriminator="backend")]
CONFIG_ADAPTER = TypeAdapter(Config)
