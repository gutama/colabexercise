"""Resumable single-case execution. The input snapshot is immutable within a run."""

import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from platform import python_version

from pymatgen.core import Structure

from . import qe
from .config import Case, load_case
from .evidence import AMSET_SCATTERING, assess, load_evidence
from .structures import canonical, fingerprint, load_structure
from .util import sha256, write_json


@dataclass(frozen=True)
class Task:
    name: str
    stage: int
    through: int
    program: str
    requires: tuple[str, ...] = ()


TASKS = (
    Task("relax", 1, 1, "pw"),
    Task("pbe", 1, 1, "pw", ("relax",)),
    Task("dfpt", 3, 2, "ph", ("pbe",)),
    Task("q2r", 3, 2, "q2r", ("dfpt",)),
    Task("matdyn", 3, 2, "matdyn", ("q2r",)),
    Task("hse", 2, 2, "pw", ("relax", "matdyn")),
)


def commands(case: Case, task: Task) -> list[str]:
    return [*case.qe.commands[task.program], "-in", "input.in"]


def starter_plan(config: Path) -> tuple[Case, Structure, dict]:
    case = load_case(config)
    seed = load_structure(case.structure, config.parent)
    species = {element.symbol for element in seed.composition.elements}
    if species != set(case.pseudos):
        raise ValueError(f"Expected one pseudopotential for each of {sorted(species)}")
    return (
        case,
        seed,
        {
            "schema_version": 2,
            "case": case.model_dump(mode="json"),
            "seed_fingerprint": fingerprint(canonical(seed)),
            "seed_pbe_kmesh": qe.pw_kmesh(seed, case.qe, "relax"),
            "tasks": [{**asdict(task), "argv": commands(case, task)} for task in TASKS],
            "handoffs": [
                {"stage": 1, "requires": ["pbe"], "deliverable": "compatible phase hull evidence"},
                {
                    "stage": 2,
                    "requires": ["hse"],
                    "deliverable": "fundamental gap and mass evidence",
                },
                {
                    "stage": 3,
                    "requires": ["dfpt", "matdyn"],
                    "deliverable": "static permittivity and optional alternate breakdown evidence",
                },
                {
                    "stage": 4,
                    "tier": 1,
                    "requires": ["hse", "dfpt", "matdyn"],
                    "deliverable": "AMSET triage",
                },
                *(
                    [{"stage": 4, "tier": 2, "requires": ["AMSET pass"], "deliverable": "EPW IBTE"}]
                    if case.transport == "epw"
                    else []
                ),
            ],
        },
    )


def pseudo_names(case: Case) -> dict:
    return {element: element + ".UPF" for element in case.pseudos}


def plan(config: Path, output: Path) -> dict:
    case, seed, data = starter_plan(config)
    output.mkdir(parents=True, exist_ok=False)
    seed.to(filename=output / "seed.cif")
    write_json(output / "plan.json", data)
    preview = output / "preview"
    preview.mkdir()
    for name in ("relax", "pbe", "hse"):
        qe.prepare_pw(preview / f"{name}.in", seed, case.qe, pseudo_names(case), name)
    qe.prepare_ph(preview / "dfpt.in", case.qe)
    qe.prepare_q2r(preview / "q2r.in")
    qe.prepare_matdyn(preview / "matdyn.in", case.qe)
    return data


def _pseudos(case: Case, base: Path, seed: Structure) -> dict:
    result = {}
    total = 0.0
    counts = Counter(site.specie.symbol for site in seed)
    for element, entry in case.pseudos.items():
        path = (base / entry.path).resolve()
        if entry.sha256 is None or not path.is_file() or sha256(path) != entry.sha256:
            raise ValueError(f"{element}: supply a vetted UPF v2 and its matching SHA-256")
        with path.open(errors="replace") as stream:
            prefix = stream.read(128_000)
        match = re.search(r"<PP_HEADER\b([^>]+)>", prefix, re.DOTALL)
        attrs = dict(re.findall(r"""(\w+)\s*=\s*["']([^"']+)["']""", match[1] if match else ""))
        aliases = {"PBE", "SLA PW PBX PBC", "PBX PBC", "PBXPBC"}
        if (
            attrs.get("element", "").strip() != element
            or attrs.get("pseudo_type", "").strip().upper() not in {"NC", "ONCV"}
            or " ".join(attrs.get("functional", "").upper().split()) not in aliases
            or attrs.get("has_so", "F").strip().upper() not in {"F", "FALSE", ".FALSE."}
            or attrs.get("relativistic", "").strip().lower() == "full"
        ):
            raise ValueError(f"{element}: expected a scalar-relativistic PBE NC/ONCV UPF v2")
        (valence,) = qe.finite_numbers(attrs.get("z_valence", ""))
        if valence <= 0:
            raise ValueError(f"{element}: invalid UPF valence electron count")
        total += counts[element] * valence
        result[element] = {"path": str(path), "sha256": entry.sha256, "source": entry.source}
    if not (total / 2).is_integer() or case.qe.nbnd <= total / 2:
        raise ValueError("Require even valence count and at least one empty band")
    return result


def _attempt_path(root: Path, records: dict, name: str) -> Path:
    return root / records[name]["directory"]


def _run_command(argv, directory: Path, case: Case) -> None:
    with (directory / "output.log").open("w") as log:
        with subprocess.Popen(
            argv,
            cwd=directory,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "OMP_NUM_THREADS": str(case.qe.threads)},
        ) as proc:
            try:
                code = proc.wait(timeout=case.qe.timeout_seconds)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                raise
    log = (directory / "output.log").read_text(errors="replace")
    if code or not re.search(r"^\s*JOB DONE\.\s*$", log, re.MULTILINE):
        raise RuntimeError(
            f"QE did not finish successfully (exit {code}); see {directory}/output.log"
        )
    if re.search(r"convergence (?:has )?not (?:been )?achieved", log, re.IGNORECASE):
        raise RuntimeError(f"QE reported nonconvergence. See {directory}/output.log")


def _prepare(task, directory, root, records, case, seed, pseudos):
    path = directory / "input.in"
    if task.program == "pw":
        structure = (
            seed
            if task.name == "relax"
            else Structure.from_file(_attempt_path(root, records, "relax") / "relaxed.json")
        )
        (directory / "pseudo").mkdir()
        for element, name in pseudo_names(case).items():
            target = directory / "pseudo" / name
            shutil.copyfile(pseudos[element]["path"], target)
            if sha256(target) != pseudos[element]["sha256"]:
                raise ValueError(f"{element}: UPF changed while staging")
        (directory / "scratch").mkdir()
        qe.prepare_pw(path, structure, case.qe, pseudo_names(case), task.name)
    elif task.name == "dfpt":
        if records["pbe"]["result"]["sampled_gap_ev"] <= 0:
            raise ValueError("Insulating PBE bands are required for dielectric DFPT")
        shutil.copytree(_attempt_path(root, records, "pbe") / "scratch", directory / "scratch")
        shutil.copytree(_attempt_path(root, records, "pbe") / "pseudo", directory / "pseudo")
        qe.prepare_ph(path, case.qe)
    elif task.name == "q2r":
        for file in _attempt_path(root, records, "dfpt").glob("dyn[0-9]*"):
            shutil.copyfile(file, directory / file.name)
        qe.prepare_q2r(path)
    elif task.name == "matdyn":
        shutil.copyfile(_attempt_path(root, records, "q2r") / "forces.fc", directory / "forces.fc")
        qe.prepare_matdyn(path, case.qe)


def _result(task, directory, case, seed):
    if task.program == "pw":
        xml = directory / "scratch" / (qe.PREFIX + ".save") / "data-file-schema.xml"
        result, structure = qe.pw_result(xml, relax=task.name == "relax")
        primary = xml
        if structure is not None:
            if structure.composition != seed.composition:
                raise ValueError("Relaxed structure composition differs from input")
            structure.to(filename=directory / "relaxed.json", fmt="json")
    elif task.name == "dfpt":
        primary = directory / "dyn0"
        if not any(directory.glob("dyn[1-9]*")):
            raise ValueError("No dynamical matrices produced by ph.x")
        result = {
            "mesh": list(case.qe.dfpt_qmesh),
            "response_directory": "scratch",
            **qe.gamma_response(directory / "dyn1", len(seed)),
        }
    elif task.name == "q2r":
        primary = directory / "forces.fc"
        result = {}
    else:
        primary = directory / "phonons.freq"
        result = qe.phonon_result(primary, len(seed), case.qe.sample_qmesh)
    if not primary.stat().st_size:
        raise ValueError(f"Empty QE artifact: {primary}")
    return result, primary


def _results(root, records):
    results = {}
    for name, record in records.items():
        if record["status"] != "complete":
            continue
        for relative, digest in record["artifacts"].items():
            if sha256(root / relative) != digest:
                raise ValueError(f"Checkpoint artifact changed: {relative}")
        results[name] = record["result"]
    return results


def _evaluate(root, case, records, evidence_path):
    evidence = None
    if evidence_path is not None:
        if records.get("relax", {}).get("status") != "complete":
            raise ValueError("Complete relaxation before attaching evidence")
        evidence = load_evidence(
            evidence_path, case, _attempt_path(root, records, "relax") / "relaxed.json"
        )
    return assess(case, _results(root, records), evidence)


def evaluate(config: Path, root: Path, evidence: Path | None = None) -> dict:
    case, seed, data = starter_plan(config)
    if not (root / "checkpoint.json").exists():
        raise ValueError("Run the case before evaluating its checkpoint")
    state = json.loads((root / "checkpoint.json").read_text())
    if (
        state["plan"] != json.loads(json.dumps(data))
        or Structure.from_file(root / "seed.json").as_dict() != seed.as_dict()
    ):
        raise ValueError("Run configuration differs. Use the original case and structure")
    return _evaluate(root, case, state["tasks"], evidence)


def run(config: Path, root: Path, through: int, evidence: Path | None = None) -> dict:
    case, seed, data = starter_plan(config)
    pseudos = _pseudos(case, config.parent, seed)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _execute(root, through, evidence, case, seed, data, pseudos)


def _execute(root, through, evidence, case, seed, data, pseudos):
    data = json.loads(json.dumps(data))
    checkpoint = root / "checkpoint.json"
    seed_path = root / "seed.json"
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        if (
            state["plan"] != data
            or state["pseudos"] != pseudos
            or Structure.from_file(seed_path).as_dict() != seed.as_dict()
        ):
            raise ValueError("Inputs changed. Start a new work directory")
    else:
        seed.to(filename=seed_path, fmt="json")
        state = {
            "plan": data,
            "pseudos": pseudos,
            "tasks": {},
            "python": python_version(),
            "packages": {name: version(name) for name in ("wbg-benchmark", "ase", "pymatgen")},
        }
        write_json(checkpoint, state)
    records = state["tasks"]
    _results(root, records)
    blocked = None
    stage_one_passed = False
    for task in TASKS:
        if task.through > through:
            break
        if records.get(task.name, {}).get("status") == "complete":
            continue
        if task.stage > 1 and not case.calibration and (not stage_one_passed or task.name == "hse"):
            review = _evaluate(root, case, records, evidence)
            stage_one_passed = review["stages"]["1"]["status"] == "pass"
            if not stage_one_passed:
                blocked = "Stages 2–3 require the Stage 1 hull and gap gates to pass"
                break
            if (
                task.name == "hse"
                and review["stages"]["3"]["gates"]["phonon_mesh_min"]["status"] != "pass"
            ):
                blocked = "HSE06 requires the PBE phonon stability gate to pass"
                break
        if any(records.get(dep, {}).get("status") != "complete" for dep in task.requires):
            raise ValueError(f"{task.name}: prerequisite task did not complete")
        jobs = root / task.name
        jobs.mkdir(exist_ok=True)
        attempt = jobs / f"attempt-{len(list(jobs.glob('attempt-*'))) + 1:03d}"
        attempt.mkdir()
        record = {"status": "running", "directory": str(attempt.relative_to(root))}
        records[task.name] = record
        write_json(checkpoint, state)
        try:
            _prepare(task, attempt, root, records, case, seed, pseudos)
            argv = commands(case, task)
            _run_command(argv, attempt, case)
            result, primary = _result(task, attempt, case, seed)
            files = [primary, attempt / "input.in", attempt / "output.log"]
            if task.name == "relax":
                files.append(attempt / "relaxed.json")
            if task.name == "dfpt":
                files.extend(sorted(attempt.glob("dyn[0-9]*")))
            hashes = {str(path.relative_to(root)): sha256(path) for path in files}
            result["provenance"] = {
                "argv": argv,
                "artifacts": hashes,
                "pseudopotentials": pseudos,
                "program_banner": next(
                    (
                        line.strip()
                        for line in (attempt / "output.log")
                        .read_text(errors="replace")
                        .splitlines()
                        if "Program " in line
                    ),
                    None,
                ),
            }
            record.update(status="complete", result=result, artifacts=hashes)
        except (Exception, KeyboardInterrupt) as exc:
            record.update(status="failed", error=str(exc))
            write_json(checkpoint, state)
            raise
        write_json(checkpoint, state)
    report = _evaluate(root, case, records, evidence)
    if through == 4:
        handoff = _transport_request(root, records, case, report, blocked)
        blocked = handoff["blocked"]
        write_json(root / "transport-request.json", handoff)
    report["blocked"] = blocked
    write_json(root / "evaluation.json", report)
    return report


def _transport_request(root, records, case, report, blocked):
    stages = report["stages"]
    if blocked is None and any(stages[str(i)]["status"] != "pass" for i in (1, 2, 3)):
        blocked = "AMSET requires the Stage 1–3 gates to pass"
    triage = stages["4"]["gates"]["amset_triage_mobility"]
    epw = case.transport == "epw"
    epw_blocked = None
    if epw and triage["status"] != "pass":
        epw_blocked = f"EPW requires an AMSET mobility pass (currently {triage['status']})"
    backend = "epw" if epw and blocked is None and epw_blocked is None else "amset"
    relaxed = _attempt_path(root, records, "relax") / "relaxed.json"
    return {
        "schema_version": 2,
        "backend": backend,
        "candidate_id": case.id,
        "track": case.track,
        "conditions": case.conditions.model_dump(),
        "blocked": blocked or (epw_blocked if triage["status"] == "fail" else None),
        "relaxed_structure": str(relaxed.relative_to(root)),
        "relaxed_structure_sha256": sha256(relaxed),
        "hse_artifacts": records.get("hse", {}).get("result", {}).get("provenance"),
        "dfpt_artifacts": records.get("dfpt", {}).get("result", {}).get("provenance"),
        "hse_save_directory": _relative_output(records, "hse", "scratch/candidate.save"),
        "dfpt_response_directory": _relative_output(records, "dfpt", "scratch"),
        "tiers": {
            "amset": {
                "status": triage["status"],
                "blocked": blocked,
                "settings": {
                    "scattering_type": AMSET_SCATTERING,
                    "free_carrier_screening": True,
                },
                "requires": [
                    "validated QE band and wavefunction bridge",
                    "elastic, deformation potential and piezoelectric tensors",
                    "effective polar optical phonon frequency",
                    "static and electronic dielectric tensors",
                ],
            },
            "epw": {
                "status": (
                    stages["4"]["status"] if backend == "epw" else "blocked" if epw else "disabled"
                ),
                "blocked": (blocked or epw_blocked) if epw else None,
                "settings": {"iterative_bte": True},
                "requires": [
                    "QE NSCF on commensurate coarse grids",
                    "Wannier projections and interpolation check",
                    "collect dvscf/dynamical matrices and phsave",
                    "converged fine k/q meshes and iterative BTE",
                ],
            },
        },
    }


def _relative_output(records: dict, task: str, suffix: str) -> str | None:
    record = records.get(task, {})
    if record.get("status") == "complete":
        return str(Path(record["directory"]) / suffix)
    return None
