"""Recorded inputs and auditable, three-state screening gates."""

import math
from pathlib import Path
from typing import Literal

from pydantic import Field

from .config import Case, Conditions, Digest, Model, Positive
from .util import sha256

# Inclusive screening thresholds from the handoff. FOM comparison is strict.
TARGETS = {
    "gan": {"gap_ev": 3.4, "breakdown_mv_cm": 3.3, "electron_cm2_vs": 1000.0},
    "sic": {"gap_ev": 3.26, "breakdown_mv_cm": 2.8, "electron_cm2_vs": 500.0},
}
BASELINES = {"gan": "gan-wurtzite", "sic": "sic-4h"}
EPSILON_VACUUM_F_CM = 8.8541878128e-14
AMSET_SCATTERING = ("ADP", "POP", "IMP", "PIE")
MIN_PHONON_THZ = -1.0


class Source(Model):
    artifact: str
    sha256: Digest
    method: str = Field(min_length=1)


class Scalar(Model):
    value: float
    source: Source


class Transport(Model):
    backend: Literal["amset", "epw"]
    conditions: Conditions
    electron_cm2_vs: Positive
    hole_cm2_vs: Positive | None = None
    solver: Literal["ibte"] | None = None
    # Velocity needs an independent high-field calculation or cited measurement.
    saturation_cm_s: Scalar | None = None
    source: Source


class AmsetTriage(Transport):
    backend: Literal["amset"]
    scattering: frozenset[Literal["ADP", "POP", "IMP", "PIE"]] = Field(min_length=4)
    free_carrier_screening: Literal[True]


class Baseline(Model):
    reference_id: Literal["gan-wurtzite", "sic-4h"]
    relative_static_epsilon: Positive
    breakdown_mv_cm: Positive
    breakdown_method: Literal["directional", "kim_pilania_ramprasad"] = "directional"
    transport: Transport
    source: Source


class Evidence(Model):
    candidate_id: str
    relaxed_structure_sha256: Digest
    hull_mev_atom: Scalar | None = None
    # Mesh Wannier interpolation and edge curvature fits can be supplied separately.
    hse06_gap_ev: Scalar | None = None
    electron_mass_m0: Scalar | None = None
    hole_mass_m0: Scalar | None = None
    relative_static_epsilon: Scalar | None = None
    breakdown_mv_cm: Scalar | None = None
    amset: AmsetTriage | None = None
    transport: Transport | None = None
    baseline: Baseline | None = None


def _sources(evidence: Evidence) -> list[Source]:
    sources = [evidence.baseline.source] if evidence.baseline is not None else []
    for item in (
        evidence.hull_mev_atom,
        evidence.hse06_gap_ev,
        evidence.electron_mass_m0,
        evidence.hole_mass_m0,
        evidence.relative_static_epsilon,
        evidence.breakdown_mv_cm,
        evidence.amset,
        evidence.transport,
        evidence.baseline.transport if evidence.baseline else None,
    ):
        if item is not None:
            sources.append(item.source)
            if isinstance(item, Transport) and item.saturation_cm_s is not None:
                sources.append(item.saturation_cm_s.source)
    return sources


def load_evidence(path: Path, case: Case, relaxed: Path) -> Evidence:
    record = Evidence.model_validate_json(path.read_text())
    if record.candidate_id != case.id or record.relaxed_structure_sha256 != sha256(relaxed):
        raise ValueError("External evidence does not match the candidate and relaxed structure")
    for source in _sources(record):
        root = path.parent.resolve()
        artifact = (root / source.artifact).resolve()
        if not artifact.is_relative_to(root) or not artifact.is_file():
            raise ValueError(f"Missing evidence artifact under {root}: {source.artifact}")
        if sha256(artifact) != source.sha256:
            raise ValueError(f"Evidence checksum mismatch: {source.artifact}")
    if record.hull_mev_atom is not None and record.hull_mev_atom.value < 0:
        raise ValueError("Hull distance must be nonnegative")
    if record.hse06_gap_ev is not None and record.hse06_gap_ev.value < 0:
        raise ValueError("Gap must be nonnegative")
    for metric in (
        record.relative_static_epsilon,
        record.breakdown_mv_cm,
        record.electron_mass_m0,
        record.hole_mass_m0,
    ):
        if metric is not None and metric.value <= 0:
            raise ValueError("Dielectric constant, breakdown field and masses must be positive")
    if any(
        transport is not None and transport.backend != case.transport
        for transport in (record.transport, record.baseline.transport if record.baseline else None)
    ):
        raise ValueError("Transport and baseline must match the case backend")
    for transport in (
        record.amset,
        record.transport,
        record.baseline.transport if record.baseline else None,
    ):
        if transport is None:
            continue
        if transport.conditions != case.conditions:
            raise ValueError("Transport evidence must match case conditions")
        if (transport.backend == "epw") != (transport.solver == "ibte"):
            raise ValueError("EPW evidence and its reference must use solver 'ibte'")
        if transport.saturation_cm_s is not None and transport.saturation_cm_s.value <= 0:
            raise ValueError("Saturation velocity must be positive")
    if record.baseline is not None and record.baseline.reference_id != BASELINES[case.track]:
        raise ValueError("The baseline must match the selected track")
    return record


def gate(value, op, limit, unit, evidence):
    if value is None or limit is None:
        status = "pending"
    elif {"ge": lambda: value >= limit, "le": lambda: value <= limit, "gt": lambda: value > limit}[
        op
    ]():
        status = "pass"
    else:
        status = "fail"
    return {
        "status": status,
        "value": value,
        "comparison": op,
        "threshold": limit,
        "unit": unit,
        "evidence": evidence,
    }


def _status(gates):
    outcomes = {item["status"] for item in gates.values()}
    return "fail" if "fail" in outcomes else "pending" if "pending" in outcomes else "pass"


def baliga(epsilon_r: float, mobility: float, field_mv_cm: float) -> float:
    return EPSILON_VACUUM_F_CM * epsilon_r * mobility * (field_mv_cm * 1e6) ** 3


def johnson(field_mv_cm: float, velocity_cm_s: float) -> float:
    return (field_mv_cm * 1e6 * velocity_cm_s / (2 * math.pi)) ** 2


def intrinsic_breakdown(gap_ev: float, max_phonon_thz: float) -> float:
    """Kim–Pilania–Ramprasad Eq. (3), in MV/cm with ordinary frequency in THz."""
    return 0.24442 * math.exp(0.315 * math.sqrt(gap_ev * max_phonon_thz))


def assess(case: Case, jobs: dict, evidence: Evidence | None) -> dict:
    limits = TARGETS[case.track]

    def external(key):
        field = getattr(evidence, key, None)
        return (field.value, field.source.model_dump()) if field is not None else (None, None)

    def metric(job, key):
        result = jobs.get(job, {})
        return result.get(key), result.get("provenance")

    hull, hull_src = external("hull_mev_atom")
    pbe, pbe_src = metric("pbe", "sampled_gap_ev")
    hybrid, hybrid_src = metric("hse", "sampled_gap_ev")
    interpolated, interpolated_src = external("hse06_gap_ev")
    # A mesh gap is diagnostic. It can overestimate the fundamental gap
    # (notably in 4H-SiC), so require validated off-grid band edges to promote.
    phonon, phonon_src = metric("matdyn", "min_phonon_thz")
    cutoff, cutoff_src = metric("matdyn", "max_phonon_thz")
    breakdown, breakdown_src = external("breakdown_mv_cm")
    breakdown_method = "directional" if breakdown is not None else None
    if hybrid is None:
        interpolated, interpolated_src = None, None
    model_field = None
    if interpolated is None or phonon is None or phonon < MIN_PHONON_THZ:
        breakdown, breakdown_src, breakdown_method = None, None, None
    elif cutoff is not None and cutoff > 0 and interpolated > 0:
        model_field = intrinsic_breakdown(interpolated, cutoff)
        if breakdown is None:
            breakdown = model_field
            breakdown_method = "kim_pilania_ramprasad"
            breakdown_src = {
                "method": "Kim–Pilania–Ramprasad Eq. (3), scalar intrinsic estimate",
                "reference": "https://doi.org/10.1021/acs.chemmater.5b04109",
                "gap_ev": interpolated,
                "gap_source": interpolated_src,
                "max_phonon_thz": cutoff,
                "phonon_source": cutoff_src,
            }
    groups = {
        "1": {
            "hull": gate(hull, "le", 50.0, "meV/atom", hull_src),
            "pbe_mesh_gap": gate(pbe, "ge", 1.8, "eV", pbe_src),
        },
        "2": {
            "hse06_fundamental_gap": gate(
                interpolated, "ge", limits["gap_ev"], "eV", interpolated_src
            )
        },
        "3": {
            "phonon_mesh_min": gate(phonon, "ge", MIN_PHONON_THZ, "THz", phonon_src),
            "breakdown_field": gate(
                breakdown, "ge", limits["breakdown_mv_cm"], "MV/cm", breakdown_src
            ),
        },
    }
    amset = (
        evidence.amset if evidence and all(_status(g) == "pass" for g in groups.values()) else None
    )
    triage = gate(
        amset.electron_cm2_vs if amset else None,
        "ge",
        limits["electron_cm2_vs"],
        "cm^2/(V s)",
        amset.source.model_dump() if amset else None,
    )
    transport = (
        evidence.transport
        if evidence and case.transport == "epw" and triage["status"] == "pass"
        else None
    )
    static, static_src = external("relative_static_epsilon")
    base = evidence.baseline if evidence else None
    matched_field = base is not None and base.breakdown_method == breakdown_method
    mu = transport.electron_cm2_vs if transport else None
    v = transport.saturation_cm_s if transport else None
    base_v = base.transport.saturation_cm_s if base else None
    bfom = (
        baliga(static, mu, breakdown)
        if all(x is not None for x in (static, mu, breakdown))
        else None
    )
    base_bfom = (
        baliga(base.relative_static_epsilon, base.transport.electron_cm2_vs, base.breakdown_mv_cm)
        if matched_field
        else None
    )
    jfom = johnson(breakdown, v.value) if breakdown is not None and v else None
    base_jfom = johnson(base.breakdown_mv_cm, base_v.value) if matched_field and base_v else None
    fom_src = {
        "dielectric": static_src,
        "breakdown": breakdown_src,
        "transport": transport.source.model_dump() if transport else None,
        "baseline": base.model_dump(mode="json") if base else None,
    }
    groups["4"] = {
        "amset_triage_mobility": triage,
        "electron_mobility": gate(
            mu, "ge", limits["electron_cm2_vs"], "cm^2/(V s)", fom_src["transport"]
        ),
        "bfom": gate(bfom, "gt", base_bfom, "W/cm^2", fom_src),
    }
    stages = {key: {"status": _status(gates), "gates": gates} for key, gates in groups.items()}
    return {
        "schema_version": 2,
        "candidate_id": case.id,
        "track": case.track,
        "role": "calibration" if case.calibration else "candidate",
        "conditions": case.conditions.model_dump(),
        "stages_1_to_4": _status(stages),
        "stages": stages,
        "diagnostics": {
            "hse06_mesh_gap_ev": hybrid,
            "hse06_mesh_gap_source": hybrid_src,
            "dfpt_response": jobs.get("dfpt"),
            "max_phonon_thz": cutoff,
            "intrinsic_breakdown_estimate_mv_cm": model_field,
            "breakdown_method": breakdown_method,
            "matched_reference_breakdown_method": matched_field,
            "electron_mass_m0": external("electron_mass_m0")[0],
            "electron_mass_source": external("electron_mass_m0")[1],
            "hole_mass_m0": external("hole_mass_m0")[0],
            "hole_mass_source": external("hole_mass_m0")[1],
            "hole_mobility_cm2_vs": transport.hole_cm2_vs if transport else None,
            "johnson_fom_v2_s2": jfom,
            "reference_johnson_fom_v2_s2": base_jfom,
            "saturation_velocity": v.model_dump() if v else None,
        },
    }
