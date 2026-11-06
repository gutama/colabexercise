import asyncio
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import H2, LIH, ROOT, WATER
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pyscf import fci
from pyscf.tools import fcidump

from quantum_workflow.cli import main as cli_main
from quantum_workflow.config import PySCFConfig, QEConfig
from quantum_workflow.workflow import load_config, parse_config, run

EXAMPLES = sorted((ROOT / "examples").glob("*.json"))


@pytest.mark.parametrize("path", EXAMPLES, ids=[p.name for p in EXAMPLES])
def test_every_example_validates(path):
    assert load_config(path).backend in {"pyscf", "qe"}


def test_h2_integrals_reproduce_fci_energy(tmp_path):
    summary = run(load_config(ROOT / "examples/h2-pyscf.json"), tmp_path / "h2")
    data = np.load(tmp_path / "h2/integrals.npz")
    assert data["h1"].shape == (2, 2)
    assert data["eri"].shape == (2, 2, 2, 2)
    energy, _ = fci.direct_spin1.kernel(
        data["h1"], data["eri"], 2, (summary["num_alpha"], summary["num_beta"]),
        ecore=summary["core_energy_hartree"],
    )
    assert energy == pytest.approx(-1.1372838345, abs=1e-8)
    assert summary["energy_hartree"] == pytest.approx(-1.1167593074, abs=1e-8)
    assert summary["core_energy_hartree"] == summary["nuclear_repulsion_hartree"]
    assert summary["reference_determinant_energy_hartree"] == pytest.approx(
        summary["energy_hartree"], abs=1e-9)
    assert summary["environment"]["packages"]["pyscf"]
    assert summary["wall_time_seconds"] >= 0


def test_golden_anchors_with_reference_methods(tmp_path):
    """Anchors re-verified on PySCF 2.15 / Python 3.13 (H2 0.735 A, LiH 1.595 A, STO-3G)."""
    h2 = run(parse_config({"backend": "pyscf", "molecule": H2,
                           "reference_methods": ["fci", "ccsd"]}), tmp_path / "h2")
    assert h2["nuclear_repulsion_hartree"] == pytest.approx(0.7199689944, abs=1e-9)
    assert h2["energy_hartree"] == pytest.approx(-1.1169989968, abs=1e-9)
    assert h2["reference_energies_hartree"]["fci"] == pytest.approx(-1.1373060358, abs=1e-9)
    # Two electrons: CCSD is exact up to its convergence threshold.
    assert h2["reference_energies_hartree"]["ccsd"] == pytest.approx(-1.1373060358, abs=1e-7)
    lih = run(parse_config({"backend": "pyscf", "molecule": LIH,
                            "reference_methods": ["fci"]}), tmp_path / "lih")
    assert lih["energy_hartree"] == pytest.approx(-7.8620238601, abs=1e-9)
    assert lih["reference_energies_hartree"]["fci"] == pytest.approx(-7.8824019323, abs=1e-9)


def test_active_space_casci_anchor_and_fcidump_roundtrip(tmp_path):
    config = parse_config({
        "backend": "pyscf", "molecule": WATER, "reference_methods": ["fci", "ccsd"],
        "active_space": {"n_electrons": 4, "n_orbitals": 4},
    })
    summary = run(config, tmp_path / "cas")
    assert summary["active_space"] == {"core_orbitals": [0, 1, 2], "active_orbitals": [3, 4, 5, 6]}
    assert (summary["num_spatial_orbitals"], summary["num_alpha"], summary["num_beta"]) == (4, 2, 2)
    assert summary["reference_energies_hartree"]["fci"] == pytest.approx(-74.9704543855, abs=1e-8)
    # Frozen-core CCSD in the same space sits close to (not exactly at) CASCI.
    assert summary["reference_energies_hartree"]["ccsd"] == pytest.approx(-74.97045, abs=2e-3)
    # The HF determinant inside the active space must reproduce the SCF energy.
    assert summary["reference_determinant_energy_hartree"] == pytest.approx(
        summary["energy_hartree"], abs=1e-8)
    dump = fcidump.read(str(tmp_path / "cas/hamiltonian.fcidump"))
    energy, _ = fci.direct_spin1.kernel(dump["H1"], dump["H2"], dump["NORB"], (2, 2),
                                        ecore=dump["ECORE"])
    assert energy == pytest.approx(-74.9704543855, abs=1e-8)


def test_active_space_rejects_selection_without_reference(tmp_path):
    def space(**active_space):
        return parse_config({"backend": "pyscf", "molecule": WATER, "active_space": active_space})

    # Only virtual orbitals: the SCF determinant is not in the space.
    with pytest.raises(ValueError, match="hold 0 electrons"):
        run(space(n_electrons=2, n_orbitals=2, orbitals=[5, 6]), tmp_path / "virtual")
    # Too many occupied orbitals active: a virtual orbital would be pushed into the core.
    with pytest.raises(ValueError, match="not all doubly occupied"):
        run(space(n_electrons=4, n_orbitals=4, orbitals=[2, 3, 4, 5]), tmp_path / "core")
    with pytest.raises(ValueError, match="does not fit"):
        run(space(n_electrons=4, n_orbitals=4, orbitals=[5, 6, 7, 8]), tmp_path / "range")
    with pytest.raises(ValueError, match="parity"):
        parse_config({"backend": "pyscf", "molecule": WATER,
                      "active_space": {"n_electrons": 3, "n_orbitals": 4}})


def test_open_shell_atom_and_nonconvergence(tmp_path):
    config = PySCFConfig.model_validate({
        "backend": "pyscf", "molecule": {
            "atoms": [{"element": "H", "xyz": [0, 0, 0]}], "spin": 1,
        },
    })
    summary = run(config, tmp_path / "atom")
    assert (summary["num_alpha"], summary["num_beta"]) == (1, 0)
    config = load_config(ROOT / "examples/h2-pyscf.json")
    config.max_cycle = 1
    with pytest.raises(RuntimeError, match="did not converge"):
        run(config, tmp_path / "nonconverged")
    assert not (tmp_path / "nonconverged/summary.json").exists()


def test_triples_need_virtual_orbitals_in_both_spin_channels(tmp_path):
    # O triplet / STO-3G fills all five alpha orbitals; PySCF's UCCSD(T) would divide by zero.
    config = parse_config({"backend": "pyscf", "reference_methods": ["ccsd(t)"], "molecule": {
        "atoms": [{"element": "O", "xyz": [0, 0, 0]}], "spin": 2}})
    with pytest.raises(ValueError, match="virtual orbitals in both spin channels"):
        run(config, tmp_path / "o")


def test_dft_and_output_directory_guard(tmp_path):
    config = load_config(ROOT / "examples/h2-pyscf.json")
    config.method = "dft"
    summary = run(config, tmp_path / "dft")
    assert summary["energy_hartree"] < -1
    with np.load(tmp_path / "dft/integrals.npz") as data:
        energy, _ = fci.direct_spin1.kernel(
            data["h1"], data["eri"], summary["num_spatial_orbitals"],
            (summary["num_alpha"], summary["num_beta"]),
            ecore=summary["core_energy_hartree"],
        )
    # FCI is orbital-invariant: Kohn-Sham orbitals give the same exact energy.
    assert energy == pytest.approx(-1.1372838345, abs=1e-8)
    with pytest.raises(ValueError, match="must be empty"):
        run(config, tmp_path / "dft")


def test_config_validation_errors():
    with pytest.raises(ValueError, match="does not match any of the expected tags"):
        parse_config({"backend": "nwchem", "molecule": H2})
    with pytest.raises(ValueError, match="Extra inputs"):
        parse_config({"backend": "pyscf", "molecule": H2, "basis_set": "sto-3g"})
    with pytest.raises(ValueError, match="require method='hf'"):
        parse_config({"backend": "pyscf", "molecule": H2, "method": "dft",
                      "reference_methods": ["ccsd"]})
    config = parse_config({"backend": "pyscf", "molecule": H2})
    with pytest.raises(ValueError):
        config.max_cycle = 0  # assignments are validated too


def test_qe_input_and_convergence_check(tmp_path, monkeypatch):
    pseudo = tmp_path / "H.upf"
    pseudo.write_text('<UPF version="2.0.1"><PP_HEADER functional="PZ" /></UPF>')
    config = load_config(ROOT / "examples/h2-qe.json")
    config.pseudopotentials["H"].path = pseudo
    backend = "quantum_workflow.backends.qe_backend"
    monkeypatch.setattr(f"{backend}.shutil.which", lambda _: "/usr/bin/pw.x")

    def fake_run(args, cwd, stdout, stderr, check):
        assert args == ["/usr/bin/pw.x", "-in", "pw.in"]
        assert (cwd / "pseudos/H.UPF").read_text() == pseudo.read_text()
        assert "K_POINTS gamma" in (cwd / "pw.in").read_text()
        stdout.write("convergence has been achieved\n!    total energy = -2.0 Ry\nJOB DONE.\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(f"{backend}.subprocess.run", fake_run)
    summary = run(config, tmp_path / "qe")
    assert summary["energy_hartree"] == -1
    assert summary["pseudopotential_functionals"] == {"H": "PZ"}
    assert any("PZ" in warning and "PBE" in warning for warning in summary["warnings"])

    def failed_run(args, cwd, stdout, stderr, check):
        stdout.write("! total energy = -2.0 Ry\nJOB DONE.\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(f"{backend}.subprocess.run", failed_run)
    with pytest.raises(RuntimeError, match="did not converge"):
        run(config, tmp_path / "failed")
    assert not (tmp_path / "failed/summary.json").exists()


def test_config_rejects_bad_species_and_resolves_pseudo_paths():
    config = load_config(ROOT / "examples/h2-qe.json")
    expected = ROOT / "examples/../pseudopotentials/H.pbe-kjpaw_psl.1.0.0.UPF"
    assert config.pseudopotentials["H"].path == expected
    data = json.loads((ROOT / "examples/h2-qe.json").read_text())
    data["pseudopotentials"] = {}
    with pytest.raises(ValueError, match="pseudopotential"):
        QEConfig.model_validate(data)
    data = json.loads((ROOT / "examples/h2-qe.json").read_text())
    data["molecule"]["atoms"][0]["xyz"] = [7.5, 7.5, 16.0]
    with pytest.raises(ValueError, match="outside the cubic cell"):
        QEConfig.model_validate(data)


def test_cli_check_and_exit_codes(tmp_path, capsys):
    assert cli_main([str(ROOT / "examples/h2o-cas44.json"), "--check"]) == 0
    assert json.loads(capsys.readouterr().out)["active_space"]["n_orbitals"] == 4
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"backend": "pyscf", "molecule": H2, "typo": 1}))
    assert cli_main([str(bad), "--check"]) == 2


def test_mcp_stdio_runs_calculation(tmp_path):
    async def check():
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "quantum_workflow.mcp_server"]
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = {tool.name for tool in (await session.list_tools()).tools}
                assert {"validate_config", "run_calculation", "run_adapt_vqe"} <= tools
                checked = await session.call_tool("validate_config", {
                    "config_path": str(ROOT / "examples/heh-cation-qubit.json"),
                })
                assert not checked.isError
                result = await session.call_tool("run_calculation", {
                    "config_path": str(ROOT / "examples/heh-cation-qubit.json"),
                    "output_dir": str(tmp_path / "mcp"),
                })
                assert not result.isError
                assert (tmp_path / "mcp/qubit_hamiltonian.json").is_file()
                adapt = await session.call_tool("run_adapt_vqe", {
                    "results_dir": str(tmp_path / "mcp"),
                })
                assert not adapt.isError
                report = json.loads((tmp_path / "mcp/adapt_vqe.json").read_text())
                assert report["chemical_accuracy"]
                error = await session.call_tool("run_calculation", {
                    "config_path": str(ROOT / "examples/h2-pyscf.json"),
                    "output_dir": str(tmp_path / "mcp"),
                })
                assert error.isError
    asyncio.run(check())
