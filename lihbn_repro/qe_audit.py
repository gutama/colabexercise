#!/usr/bin/env python3
"""Read-only status audit of a Quantum ESPRESSO HF / ACE stdout log.

Does not modify input/output files or interact with running pw.x processes.
Snapshots can be compared on subsequent invocations; filesystem and ps readings
are observations, NOT proofs of numerical progress or job completion.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

NUM = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"
ENERGY = re.compile(r"!\s*total energy\s*=\s*" + NUM + r"\s*Ry", re.I)
ACC = re.compile(r"estimated scf accuracy\s*<\s*" + NUM + r"\s*Ry", re.I)
FOCK = re.compile(r"Fock energy\s*\(\s*ACE\s*\)\s*(?:=|:)\s*" + NUM + r"\s*Ry", re.I)
MAG = re.compile(r"(total|absolute) magnetization\s*=\s*" + NUM, re.I)
INNER = re.compile(r"convergence has been achieved in\s+\d+\s+iterations", re.I)
MARKERS = re.compile(r"ACE|EXX|Fock energy|iteration #|JOB DONE|convergence has been achieved", re.I)


def _number(s: str) -> float:
    return float(s.replace("D", "E").replace("d", "e"))


def _ps() -> list[dict]:
    try:
        r = subprocess.run(["ps", "-C", "pw.x", "-o", "pid=,etime=,pcpu=,rss=,stat="],
                           capture_output=True, text=True, check=True, timeout=5)
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    rows = []
    for s in r.stdout.splitlines():
        fields = s.split()
        if len(fields) < 5:
            continue
        try:
            rows.append({"pid": int(fields[0]), "elapsed": fields[1],
                         "cpu_percent_lifetime": float(fields[2]),
                         "rss_gib": round(int(fields[3]) / 1024**2, 3),
                         "process_state": fields[4]})
        except ValueError:
            pass
    return rows


def _memory() -> dict:
    p = Path("/proc/meminfo")
    if not p.exists():
        return {}
    values = {}
    for line in p.read_text().splitlines():
        key, _, rest = line.partition(":")
        if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            values[key + "_GiB"] = round(int(rest.split()[0]) / 1024**2, 3)
    return values


def audit(log_file: Path, processes: bool = False) -> dict:
    stat = log_file.stat()
    counts = {"inner_scf_convergences": 0, "scf_total_energy_entries": 0,
              "ace_fock_energy_entries": 0}
    result = {"log_file": str(log_file.resolve()), "log_size_bytes": stat.st_size,
              "log_mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
              "observed_at_utc": datetime.now(timezone.utc).isoformat(),
              "log_age_minutes": round((datetime.now(timezone.utc).timestamp() - stat.st_mtime) / 60, 1),
              "job_done": False, "ace_mentioned": False,
              "last_total_energy_ry": None, "last_scf_accuracy_ry": None,
              "last_ace_fock_energy_ry": None,
              "total_magnetization": None, "absolute_magnetization": None}
    recent = deque(maxlen=12)
    with log_file.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            if "JOB DONE." in line:
                result["job_done"] = True
            if "ACE" in line or "ace" in line:
                result["ace_mentioned"] = True
            if INNER.search(line):
                counts["inner_scf_convergences"] += 1
            m = ENERGY.search(line)
            if m:
                counts["scf_total_energy_entries"] += 1
                result["last_total_energy_ry"] = _number(m.group(1))
            m = ACC.search(line)
            if m:
                result["last_scf_accuracy_ry"] = _number(m.group(1))
            m = FOCK.search(line)
            if m:
                counts["ace_fock_energy_entries"] += 1
                result["last_ace_fock_energy_ry"] = _number(m.group(1))
            m = MAG.search(line)
            if m:
                result[m.group(1).lower() + "_magnetization"] = _number(m.group(2))
            if MARKERS.search(line):
                recent.append({"line": i, "text": line.strip()[:220]})
    result["counts"] = counts
    result["recent_markers"] = list(recent)
    if processes:
        result["local_pw_processes"] = _ps()
        result["local_memory"] = _memory()
    return result


def compare(now: dict, old: dict) -> dict:
    if now["log_file"] != old.get("log_file"):
        raise ValueError("Previous snapshot refers to a different log path")
    delta = {"log_byte_growth": now["log_size_bytes"] - old.get("log_size_bytes", 0)}
    for k in now["counts"]:
        delta[k + "_new"] = now["counts"][k] - old.get("counts", {}).get(k, 0)
    delta["completion_marker_appeared"] = now["job_done"] and not old.get("job_done", False)
    return delta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log_file", type=Path, help="Existing QE pw.x stdout file")
    p.add_argument("--processes", action="store_true", help="Include LOCAL host ps and memory (not remote)")
    p.add_argument("--previous", type=Path, help="Previously saved snapshot JSON")
    p.add_argument("--save", type=Path, help="Write a new snapshot JSON outside the QE log")
    args = p.parse_args()
    result = audit(args.log_file, processes=args.processes)
    if args.previous:
        previous = json.loads(args.previous.read_text(encoding="utf-8"))
        result["delta_since_previous"] = compare(result, previous)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.save:
        if args.save.resolve() == args.log_file.resolve():
            raise SystemExit("Refusing to overwrite the QE log")
        args.save.write_text(json.dumps({k: v for k, v in result.items() if k != "delta_since_previous"},
                                        indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if result["job_done"]:
        print("STATUS: QE completion marker found. Check errors and physical convergence separately.")
    else:
        print("STATUS: No QE JOB DONE. Inner SCF convergence is NOT HF/ACE job completion.")
        print("NOTE: A quiet/buffered log does not alone prove a hang; inspect actual process/CPU and output over time.")


if __name__ == "__main__":
    main()
