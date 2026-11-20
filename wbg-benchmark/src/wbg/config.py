"""Case settings. Meshes, cutoffs and thresholds are kept explicit."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Positive = Annotated[float, Field(gt=0)]
Count = Annotated[int, Field(gt=0)]
Mesh = tuple[Count, Count, Count]
Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)


class Pseudo(Model):
    path: str
    source: str = Field(min_length=1)
    sha256: Digest | None = None


class QE(Model):
    ecutwfc_ry: Positive = 80.0
    ecutrho_ratio: Positive = 4.0
    xc: Literal["PBE", "PBEsol"] = "PBE"
    nbnd: Count
    pbe_kmesh: Mesh
    pbe_k_length_angstrom: Positive = 35.0
    hse_kmesh: Mesh
    exchange_qmesh: Mesh
    dfpt_qmesh: Mesh
    sample_qmesh: Mesh
    conv_thr_ry: Positive = 1e-9
    force_thr_ry_bohr: Positive = 1e-4
    press_thr_kbar: Positive = 0.5
    tr2_ph: Positive = 1e-14
    # The argv can include MPI options, for example ["mpirun", "-np", "4", "pw.x"].
    commands: dict[Literal["pw", "ph", "q2r", "matdyn"], list[str]] = Field(
        default_factory=lambda: {key: [key + ".x"] for key in ("pw", "ph", "q2r", "matdyn")}
    )
    threads: Count = 1
    timeout_seconds: Positive = 7200.0

    @model_validator(mode="after")
    def check_meshes(self):
        if self.ecutrho_ratio < 4:
            raise ValueError("The NC charge density cutoff ratio must be at least four")
        if any(k % q for k, q in zip(self.hse_kmesh, self.exchange_qmesh, strict=True)):
            raise ValueError("The HSE k mesh must be divisible by the exchange q mesh")
        if any(s < q or s % q for s, q in zip(self.sample_qmesh, self.dfpt_qmesh, strict=True)):
            raise ValueError("Sample q mesh must be a refinement of the DFPT q mesh")
        if self.xc != "PBE":
            raise ValueError(
                "Stages 1–4 share PBE NC potentials with HSE06. Use PBE in this scaffold"
            )
        if set(self.commands) != {"pw", "ph", "q2r", "matdyn"} or any(
            not cmd or not cmd[0] for cmd in self.commands.values()
        ):
            raise ValueError("Supply nonempty argv for pw, ph, q2r, and matdyn")
        return self


class Conditions(Model):
    temperature_k: Positive = 300.0
    carrier_density_cm3: Positive = 1e16
    direction: Literal["a", "c"]
    regime: Literal["bulk"] = "bulk"


class Case(Model):
    id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{0,39}$")]
    track: Literal["gan", "sic"]
    structure: str
    calibration: bool = False
    pseudos: dict[str, Pseudo]
    qe: QE
    conditions: Conditions
    transport: Literal["amset", "epw"] = "epw"


def load_case(path: Path) -> Case:
    return Case.model_validate_json(path.read_text())
