"""Colab driver against a fake `colab` CLI (tests/fake_colab.py); no Google account needed."""
import io
import json
import sys
import tarfile

import pytest
from conftest import H2, ROOT

from quantum_workflow import colab_remote
from quantum_workflow.colab import ColabError, build_wheel, safe_extract, submit


@pytest.fixture
def fake_colab(tmp_path, monkeypatch):
    remote = tmp_path / "vm"
    remote.mkdir()
    shim = tmp_path / "colab"
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{ROOT / "tests/fake_colab.py"}" "$@"\n')
    shim.chmod(0o755)
    monkeypatch.setenv("FAKE_COLAB_ROOT", str(remote))
    return shim, remote


def calls(remote):
    return (remote / "calls.log").read_text().splitlines()


def sessions(remote):
    return json.loads((remote / "sessions.json").read_text())


def test_submit_runs_job_downloads_results_and_releases_vm(fake_colab, tmp_path):
    shim, remote = fake_colab
    out = submit(ROOT / "examples/heh-cation-qubit.json", tmp_path / "results",
                 session="qw-test", source="preinstalled", colab_bin=str(shim))
    summary = json.loads((out / "summary.json").read_text())
    assert summary["reference_energies_hartree"]["fci"] == pytest.approx(-2.85102403, abs=1e-8)
    assert (out / "qubit_hamiltonian.json").is_file()
    log = calls(remote)
    assert log[0] == "sessions" and log[1] == "new -s qw-test"
    exec_call = next(line for line in log if " -f " in line)
    assert "--timeout 3600" in exec_call  # never colab's 30 s default
    assert "upload -s qw-test" in "\n".join(log)
    assert log[-1] == "stop -s qw-test"
    assert sessions(remote) == []


def test_remote_failure_is_detected_although_exec_exits_zero(fake_colab, tmp_path):
    shim, remote = fake_colab
    config = tmp_path / "bad.json"
    config.write_text(json.dumps({"backend": "pyscf", "molecule": H2, "max_cycle": 1}))
    with pytest.raises(ColabError, match="Remote job failed"):
        submit(config, tmp_path / "results", session="qw-fail", source="preinstalled",
               colab_bin=str(shim))
    assert calls(remote)[-1] == "stop -s qw-fail"
    assert sessions(remote) == []
    assert not (tmp_path / "results").exists()


def test_refuses_before_renting_anything(fake_colab, tmp_path, monkeypatch):
    shim, remote = fake_colab
    (remote / "sessions.json").write_text(json.dumps(["qw-taken"]))
    with pytest.raises(ColabError, match="orphans"):
        submit(ROOT / "examples/h2-pyscf.json", tmp_path / "a", session="qw-taken",
               source="preinstalled", colab_bin=str(shim))
    with pytest.raises(ColabError, match="device='gpu'"):
        submit(ROOT / "examples/h2o-dft-gpu.json", tmp_path / "b", colab_bin=str(shim))
    with pytest.raises(ColabError, match="Unknown GPU"):
        submit(ROOT / "examples/h2-pyscf.json", tmp_path / "c", gpu="A10",
               colab_bin=str(shim))
    monkeypatch.setenv("FAKE_COLAB_UNAUTH", "1")
    with pytest.raises(ColabError, match="not authenticated"):
        submit(ROOT / "examples/h2-pyscf.json", tmp_path / "d", source="preinstalled",
               colab_bin=str(shim))
    assert not any(line.startswith("new") for line in calls(remote))


def test_dry_run_prints_commands_only(fake_colab, tmp_path, capsys):
    shim, remote = fake_colab
    assert submit(ROOT / "examples/h2-pyscf.json", tmp_path / "out", session="qw-dry",
                  gpu="T4", source="git+https://github.com/example/repo@main",
                  colab_bin=str(shim), dry_run=True) is None
    printed = capsys.readouterr().out
    assert "new -s qw-dry --gpu T4" in printed
    assert "QW_INSTALL=colab-quantum-chemistry[qubit] @ git+https://github.com/example/repo@main" \
        in printed
    assert not (remote / "calls.log").exists()


def test_remote_runner_archives_results_without_scratch(tmp_path, capsys):
    job = tmp_path / "job"
    job.mkdir()
    (job / "config.json").write_text((ROOT / "examples/h2-pyscf.json").read_text())
    tarball = colab_remote.main(["--job-dir", str(job), "--install", "none"])
    assert capsys.readouterr().out.strip().splitlines()[-1] == f"{colab_remote.OK} {tarball}"
    (job / "results" / "scratch").mkdir()
    (job / "results" / "scratch" / "wfc.dat").write_text("large")
    colab_remote._archive(job / "results", job / "again.tar.gz")
    with tarfile.open(job / "again.tar.gz") as tar:
        names = tar.getnames()
    assert "results/summary.json" in names and not any("scratch" in n for n in names)
    with pytest.raises(SystemExit):
        colab_remote.main(["--job-dir", str(job), "--install", "none"])  # results exist
    assert colab_remote.FAILED in capsys.readouterr().out


def test_safe_extract_rejects_path_traversal(tmp_path):
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("../escaped.txt")
        info.size = 4
        tar.addfile(info, io.BytesIO(b"oops"))
    (tmp_path / "dest").mkdir()
    with pytest.raises(ColabError, match="Unsafe"):
        safe_extract(archive, tmp_path / "dest")
    assert not (tmp_path / "escaped.txt").exists()


@pytest.mark.slow
def test_wheel_contains_remote_runner(tmp_path):
    import zipfile

    wheel = build_wheel(tmp_path)
    names = zipfile.ZipFile(wheel).namelist()
    assert "quantum_workflow/colab_remote.py" in names
    assert "quantum_workflow/backends/pyscf_backend.py" in names
