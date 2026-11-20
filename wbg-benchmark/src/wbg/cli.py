"""CLI for planning, running, scoring, and auditing reference candidates."""

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from . import workflow
from .evidence import Evidence
from .structures import DuplicateIndex, audit_mdb, load_structure
from .util import write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wbg", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("config", type=Path)
        command.add_argument("--workdir", type=Path, required=True)
        if name != "plan":
            command.add_argument("--evidence", type=Path)
        if name == "run":
            command.add_argument(
                "--through",
                type=int,
                choices=(1, 2, 3, 4),
                default=1,
                help="1 PBE triage, 2–3 phonons then HSE06, 4 transport handoff",
            )
    mdb = sub.add_parser("audit-mdb", help="Find duplicates in the MDB candidate archive")
    mdb.add_argument("archive", type=Path)
    mdb.add_argument("--limit", type=int, help="Check only the first N rows")
    mdb.add_argument("--output", type=Path)
    dedupe = sub.add_parser("dedupe", help="Compare candidates by reduced primitive structure")
    dedupe.add_argument("structures", metavar="STRUCTURE", nargs="+", type=Path)
    schema = sub.add_parser("evidence-schema", help="Print the external evidence JSON schema")
    schema.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = workflow.plan(args.config.resolve(), args.workdir.resolve())
        elif args.command == "run":
            result = workflow.run(
                args.config.resolve(), args.workdir.resolve(), args.through, args.evidence
            )
        elif args.command == "evaluate":
            result = workflow.evaluate(args.config.resolve(), args.workdir.resolve(), args.evidence)
        elif args.command == "audit-mdb":
            if args.limit is not None and args.limit < 1:
                raise ValueError("--limit must be positive")
            result = audit_mdb(args.archive, args.limit)
        elif args.command == "dedupe":
            index = DuplicateIndex()
            result = {
                "matches": [
                    index.add(str(path), load_structure(str(path.resolve()), Path.cwd()))
                    for path in args.structures
                ]
            }
        else:
            result = Evidence.model_json_schema()
        if getattr(args, "output", None):
            write_json(args.output, result)
        else:
            print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        ET.ParseError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"wbg: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
