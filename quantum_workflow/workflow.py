"""Public API: load/parse a config and run it into an empty output directory."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .backends import RUNNERS
from .config import CONFIG_ADAPTER, PySCFConfig, QEConfig

AnyConfig = PySCFConfig | QEConfig


def parse_config(data: dict, base_dir: str | Path | None = None) -> AnyConfig:
    """Validate an in-memory config; relative file paths resolve against ``base_dir``."""
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a JSON object")
    config = CONFIG_ADAPTER.validate_python(data)
    config.resolve_paths(Path(base_dir or Path.cwd()).resolve())
    return config


def load_config(path: str | Path) -> AnyConfig:
    """Load and validate a JSON config; relative paths resolve against its directory."""
    config_path = Path(path).resolve()
    return parse_config(json.loads(config_path.read_text()), config_path.parent)


def run(config: AnyConfig, output_dir: str | Path) -> dict:
    """Run ``config`` into a new or empty directory and return its summary.

    ``summary.json`` is written last: an output directory without it is a failed or
    interrupted run, whatever else it contains.
    """
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    runner = RUNNERS.get(type(config))
    if runner is None:
        raise ValueError(f"No runner registered for {type(config).__name__}")
    start = time.perf_counter()
    summary = runner(config, output)
    summary["wall_time_seconds"] = round(time.perf_counter() - start, 3)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary
