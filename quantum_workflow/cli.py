import argparse
import json
import sys

from .workflow import load_config, run


def main():
    parser = argparse.ArgumentParser(description="Run a molecular simulation")
    parser.add_argument("config", help="Path to JSON configuration")
    parser.add_argument("--output", required=True, help="Empty output directory")
    args = parser.parse_args()
    try:
        result = run(load_config(args.config), args.output)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Calculation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
