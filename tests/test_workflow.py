import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pyscf import fci

from quantum_workflow.config import PySCFConfig, QEConfig
from quantum_workflow.workflow import load_config, run


ROOT = Path(__file__).resolve().parents[1]


def test_h2_integrals_reproduce_fci_energy(tmp_path):
    summary = run(load_config(ROOT / "examples/h2-pyscf.json"), tmp_path / "h2")
    data = np.load(tmp_path / "h2/integrals.npz")
    assert data["h1"].shape == (2, 2)
    assert data["eri"].shape == (2, 2, 2, 2)
    energy, _ = fci.direct_spin1.kernel(
        data["h1"], data["eri"], 2, (summary["num_alpha"], summary["num_beta"]),
        ecore=summary["nuclear_repulsion_hartree"],
    )
    assert energy == pytest.approx(-1.1372838345, abs=1e-8)
    assert summary["energy_hartree"] == pytest.approx(-1.1167593074, abs=1e-8)


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


def test_dft_and_output_directory_guard(tmp_path):
    config = load_config(ROOT / "examples/h2-pyscf.json")
    config.method = "dft"
    summary = run(config, tmp_path / "dft")
    assert summary["energy_hartree"] < -1
    with np.load(tmp_path / "dft/integrals.npz") as data:
        energy, _ = fci.direct_spin1.kernel(
            data["h1"], data["eri"], summary["num_spatial_orbitals"],
            (summary["num_alpha"], summary["num_beta"]),
            ecore=summary["nuclear_repulsion_hartree"],
        )
    assert energy == pytest.approx(-1.1372838345, abs=1e-8)
    with pytest.raises(ValueError, match="must be empty"):
        run(config, tmp_path / "dft")


def test_qe_input_and_convergence_check(tmp_path, monkeypatch):
    pseudo = tmp_path / "H.upf"
    pseudo.write_text("test")
    config = load_config(ROOT / "examples/h2-qe.json")
    config.pseudopotentials["H"].path = pseudo
    monkeypatch.setattr("quantum_workflow.workflow.shutil.which", lambda _: "/usr/bin/pw.x")

    def fake_run(args, cwd, stdout, stderr, check):
        assert args == ["/usr/bin/pw.x", "-in", "pw.in"]
        assert (cwd / "pseudos/H.UPF").read_text() == "test"
        assert "K_POINTS gamma" in (cwd / "pw.in").read_text()
        stdout.write("convergence has been achieved\n!    total energy = -2.0 Ry\nJOB DONE.\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("quantum_workflow.workflow.subprocess.run", fake_run)
    assert run(config, tmp_path / "qe")["energy_hartree"] == -1

    def failed_run(args, cwd, stdout, stderr, check):
        stdout.write("! total energy = -2.0 Ry\nJOB DONE.\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("quantum_workflow.workflow.subprocess.run", failed_run)
    with pytest.raises(RuntimeError, match="did not converge"):
        run(config, tmp_path / "failed")
    assert not (tmp_path / "failed/summary.json").exists()


def test_config_rejects_bad_species_and_resolves_pseudo_paths():
    config = load_config(ROOT / "examples/h2-qe.json")
    assert config.pseudopotentials["H"].path == ROOT / "examples/../pseudopotentials/H.pbe-kjpaw_psl.1.0.0.UPF"
    data = json.loads((ROOT / "examples/h2-qe.json").read_text())
    data["pseudopotentials"] = {}
    with pytest.raises(ValueError, match="pseudopotential"):
        QEConfig.model_validate(data)


def test_mcp_stdio_runs_calculation(tmp_path):
    async def check():
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "quantum_workflow.mcp_server"]
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert "run_calculation" in [tool.name for tool in tools.tools]
                result = await session.call_tool("run_calculation", {
                    "config_path": str(ROOT / "examples/h2-pyscf.json"),
                    "output_dir": str(tmp_path / "mcp"),
                })
                assert not result.isError
                assert (tmp_path / "mcp/integrals.npz").is_file()
                error = await session.call_tool("run_calculation", {
                    "config_path": str(ROOT / "examples/h2-pyscf.json"),
                    "output_dir": str(tmp_path / "mcp"),
                })
                assert error.isError
    asyncio.run(check())
