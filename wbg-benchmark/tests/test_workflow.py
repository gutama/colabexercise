import json
import sys
from pathlib import Path

import pytest
from ase.io import read

from wbg.config import load_case
from wbg.evidence import Evidence, assess, intrinsic_breakdown
from wbg.structures import DuplicateIndex, load_structure
from wbg.util import sha256
from wbg.workflow import evaluate, plan, run

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(("name", "nat"), [("gan", 4), ("sic", 8)])
def test_calibration_plans_and_supercell_identity(tmp_path, name, nat):
    config = ROOT / f"configs/{name}.json"
    data = plan(config, tmp_path / "plan")
    geometry = data["seed_fingerprint"]
    assert (geometry["sites"], geometry["space_group"]) == (nat, 186)
    assert [job["name"] for job in data["tasks"]] == [
        "relax",
        "pbe",
        "dfpt",
        "q2r",
        "matdyn",
        "hse",
    ]
    for job in ("relax", "pbe", "hse"):
        assert len(read(tmp_path / f"plan/preview/{job}.in", format="espresso-in")) == nat
    assert "epsil=.true." in (tmp_path / "plan/preview/dfpt.in").read_text()
    structure = load_structure(load_case(config).structure, config.parent)
    case = load_case(config)
    for k, q, length in zip(data["seed_pbe_kmesh"], case.qe.dfpt_qmesh, structure.lattice.abc):
        assert k % q == 0 and k * length >= 35
    supercell = structure.copy()
    supercell.make_supercell([2, 2, 3])
    index = DuplicateIndex()
    assert index.add("seed", structure)["duplicate_of"] is None
    assert index.add("replica", supercell)["duplicate_of"] == "seed"


@pytest.fixture(params=["gan", "sic"])
def case_file(tmp_path, request):
    case = json.loads((ROOT / f"configs/{request.param}.json").read_text())
    nat = "4" if request.param == "gan" else "8"
    valence = {"Ga": 13, "N": 5, "Si": 4, "C": 4}
    fake = str(ROOT / "tests/fake_qe.py")
    for key, value in case["pseudos"].items():
        path = tmp_path / f"{key}.UPF"
        path.write_text(
            '<UPF version="2.0.1">\n'
            f'<PP_HEADER element="{key}" pseudo_type="NC" functional="SLA PW PBX PBC" '
            f'z_valence="{valence[key]}" />\n</UPF>\n'
        )
        value.update(path=path.name, source="Synthetic test fixture", sha256=sha256(path))
    case["qe"].update(
        dfpt_qmesh=[1, 1, 1],
        sample_qmesh=[2, 2, 2],
        commands={key: [sys.executable, fake, key, nat] for key in ("pw", "ph", "q2r", "matdyn")},
    )
    path = tmp_path / "test-case.json"
    path.write_text(json.dumps(case))
    return path


def test_checkpoint_resume_provenance_and_transport_handoff(case_file, tmp_path):
    root = tmp_path / "run"
    report = run(case_file, root, 4)
    assert report["stages_1_to_4"] == "pending"
    assert report["stages"]["3"]["gates"]["phonon_mesh_min"]["status"] == "pass"
    assert report["stages"]["2"]["gates"]["hse06_fundamental_gap"]["status"] == "pending"
    assert report["diagnostics"]["hse06_mesh_gap_ev"] == pytest.approx(3.809594, abs=1e-5)
    assert (
        report["diagnostics"]["dfpt_response"]["relative_electronic_epsilon_cartesian"][0][0] == 4
    )
    task_file = root / "pbe/attempt-001/input.in"
    # ASE round-trip checks that stage 1's SCF uses the relaxed cell.
    case = load_case(case_file)
    seed = load_structure(case.structure, case_file.parent)
    assert read(task_file, format="espresso-in").cell.lengths()[0] == pytest.approx(
        seed.lattice.a * 1.01
    )
    handoff = json.loads((root / "transport-request.json").read_text())
    assert handoff["backend"] == "amset" and handoff["blocked"]
    assert handoff["tiers"]["epw"]["status"] == "blocked"
    assert sha256(root / handoff["relaxed_structure"]) == handoff["relaxed_structure_sha256"]
    _check_matching_evidence(case_file, root, handoff)
    assert run(case_file, root, 4) == report
    assert len(list((root / "pbe").glob("attempt-*"))) == 1

    task_file.write_text("modified checkpoint artifact")
    with pytest.raises(ValueError, match="Checkpoint artifact changed"):
        evaluate(case_file, root)


def _check_matching_evidence(case_file, root, handoff):
    fixture = case_file.parent / "result-fixture.txt"
    fixture.write_text("Synthetic evidence for testing score and evidence binding.")
    source = {"artifact": fixture.name, "sha256": sha256(fixture), "method": "Fixture"}

    def scalar(value):
        return {"value": value, "source": source}

    transport = {
        "backend": "epw",
        "solver": "ibte",
        "conditions": handoff["conditions"],
        "electron_cm2_vs": 2000,
        "saturation_cm_s": scalar(2.5e7),
        "source": source,
    }
    evidence = {
        "candidate_id": handoff["candidate_id"],
        "relaxed_structure_sha256": handoff["relaxed_structure_sha256"],
        "hull_mev_atom": scalar(0),
        "hse06_gap_ev": scalar(3.5),
        "breakdown_mv_cm": scalar(4.0),
        "relative_static_epsilon": scalar(10),
        "transport": transport,
        "baseline": {
            "reference_id": handoff["candidate_id"],
            "relative_static_epsilon": 10,
            "breakdown_mv_cm": 3.3,
            "transport": {**transport, "electron_cm2_vs": 1000},
            "source": source,
        },
    }
    path = case_file.parent / "evidence.json"
    path.write_text(json.dumps(evidence))
    report = run(case_file, root, 4, path)
    assert report["stages_1_to_4"] == "pending"
    handoff_path = root / "transport-request.json"
    triage = json.loads(handoff_path.read_text())
    assert triage["backend"] == "amset" and triage["blocked"] is None
    assert triage["tiers"]["epw"]["blocked"].endswith("(currently pending)")

    evidence["amset"] = {
        "backend": "amset",
        "conditions": handoff["conditions"],
        "electron_cm2_vs": 100,
        "scattering": ["ADP", "POP", "IMP", "PIE"],
        "free_carrier_screening": True,
        "source": source,
    }
    path.write_text(json.dumps(evidence))
    report = run(case_file, root, 4, path)
    assert report["stages_1_to_4"] == "fail"
    assert report["stages"]["4"]["gates"]["electron_mobility"]["status"] == "pending"
    assert json.loads(handoff_path.read_text())["backend"] == "amset"

    evidence["amset"]["electron_cm2_vs"] = 1200
    path.write_text(json.dumps(evidence))
    report = run(case_file, root, 4, path)
    assert report["stages_1_to_4"] == "pass"
    assert report["diagnostics"]["johnson_fom_v2_s2"] > 0
    epw = json.loads(handoff_path.read_text())
    assert epw["backend"] == "epw" and epw["blocked"] is None
    assert epw["tiers"]["epw"]["settings"] == {"iterative_bte": True}

    del evidence["transport"]["solver"]
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="solver 'ibte'"):
        evaluate(case_file, root, path)

    fixture.write_text("Modified fixture")
    with pytest.raises(ValueError, match="Evidence checksum mismatch"):
        evaluate(case_file, root, path)


def test_noncalibration_requires_hull_and_checks_missing_potentials(case_file, tmp_path):
    path = tmp_path / "missing"
    with pytest.raises(ValueError, match="matching SHA-256"):
        run(ROOT / "configs/gan.json", path, 1)
    assert not path.exists()
    case = json.loads(case_file.read_text())
    case["calibration"] = False
    case_file.write_text(json.dumps(case))
    report = run(case_file, tmp_path / "noncalibration", 4)
    assert "Stage 1 hull and gap" in report["blocked"]
    assert not (tmp_path / "noncalibration/dfpt").exists()
    assert not (tmp_path / "noncalibration/hse").exists()
    assert report["stages"]["1"]["status"] == "pending"


def test_stage_thresholds_and_fail_priority():
    case = load_case(ROOT / "configs/sic.json")
    data = assess(case, {"pbe": {"sampled_gap_ev": 1.7}}, None)
    assert data["stages_1_to_4"] == "fail"
    assert data["stages"]["2"]["gates"]["hse06_fundamental_gap"]["threshold"] == 3.26
    assert data["stages"]["4"]["gates"]["electron_mobility"]["threshold"] == 500


def test_phonon_stability_blocks_hybrid(case_file, tmp_path, monkeypatch):
    case = json.loads(case_file.read_text())
    case["calibration"] = False
    case_file.write_text(json.dumps(case))
    root = tmp_path / "unstable"
    run(case_file, root, 1)
    relaxed = root / "relax/attempt-001/relaxed.json"
    artifact = tmp_path / "hull.txt"
    artifact.write_text("Synthetic 10 meV/atom hull")
    evidence = tmp_path / "hull.json"
    evidence.write_text(
        json.dumps(
            {
                "candidate_id": case["id"],
                "relaxed_structure_sha256": sha256(relaxed),
                "hull_mev_atom": {
                    "value": 10,
                    "source": {
                        "artifact": artifact.name,
                        "sha256": sha256(artifact),
                        "method": "Synthetic phase hull",
                    },
                },
            }
        )
    )
    monkeypatch.setenv("WBG_FIXTURE_MIN_CM1", "-50")
    report = run(case_file, root, 2, evidence)
    assert report["stages"]["1"]["status"] == "pass"
    assert report["stages"]["3"]["gates"]["phonon_mesh_min"]["status"] == "fail"
    assert "HSE06 requires" in report["blocked"]
    assert not (root / "hse").exists()


def test_intrinsic_field_uses_interpolated_gap_and_sampled_cutoff():
    source = {"artifact": "gap.json", "sha256": "a" * 64, "method": "fixture"}
    case = load_case(ROOT / "configs/gan.json")
    jobs = {
        "hse": {"sampled_gap_ev": 3.9},
        "matdyn": {"min_phonon_thz": 0.0, "max_phonon_thz": 23.65},
    }
    evidence = Evidence.model_validate(
        {
            "candidate_id": case.id,
            "relaxed_structure_sha256": "b" * 64,
            "hse06_gap_ev": {"value": 3.8, "source": source},
        }
    )
    field = assess(case, jobs, evidence)["stages"]["3"]["gates"]["breakdown_field"]
    assert intrinsic_breakdown(0, 23.65) == pytest.approx(0.24442)
    assert field["value"] == pytest.approx(4.843, abs=0.005)
    assert field["evidence"]["gap_ev"] == 3.8
    assert field["evidence"]["max_phonon_thz"] == 23.65


def test_failed_task_is_preserved_and_can_be_retried(case_file, tmp_path, monkeypatch):
    root = tmp_path / "restart"
    monkeypatch.setenv("WBG_FIXTURE_NONCONVERGENCE", "ph")
    with pytest.raises(RuntimeError, match="nonconvergence"):
        run(case_file, root, 3)
    checkpoint = json.loads((root / "checkpoint.json").read_text())
    assert checkpoint["tasks"]["dfpt"]["status"] == "failed"
    assert "q2r" not in checkpoint["tasks"]
    monkeypatch.delenv("WBG_FIXTURE_NONCONVERGENCE")
    run(case_file, root, 3)
    assert (root / "dfpt/attempt-001/output.log").is_file()
    assert (root / "dfpt/attempt-002/dyn1").is_file()
    assert not (root / "pbe/attempt-002").exists()
