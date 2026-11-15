"""Command line: run a config locally (``quantum-workflow CONFIG --output DIR``)."""
import argparse
import json
import sys

from pydantic import ValidationError

from .workflow import load_config, run


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a molecular simulation from a JSON config")
    parser.add_argument("config", help="Path to JSON configuration")
    parser.add_argument("--output", help="New or empty output directory")
    parser.add_argument("--check", action="store_true",
                        help="Validate the config, print it with defaults filled in, and exit")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.check:
            print(json.dumps(config.model_dump(mode="json"), indent=2))
            return 0
        if not args.output:
            parser.error("--output is required unless --check is given")
        result = run(config, args.output)
    except ValidationError as exc:
        print(f"Invalid configuration:\n{exc}", file=sys.stderr)
        return 2
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Calculation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
