"""MCP stdio server exposing local calculations to a trusted agent.

Paths are on the machine running this server. The tools read local files and write
results wherever this process may; calculations run synchronously and can be
expensive, so set resource limits externally and use only with trusted clients.
Colab jobs are deliberately *not* exposed here: they rent billable VMs.
"""
from mcp.server.fastmcp import FastMCP

from .adapt import run_adapt
from .workflow import load_config, run

mcp = FastMCP("quantum-workflow")


@mcp.tool()
def validate_config(config_path: str) -> dict:
    """Validate a PySCF/QE JSON config and return it with all defaults filled in.

    Cheap and side-effect free: use it before run_calculation.
    """
    return load_config(config_path).model_dump(mode="json")


@mcp.tool()
def run_calculation(config_path: str, output_dir: str) -> dict:
    """Run a local PySCF or QE calculation from JSON, saving to a new or empty directory.

    Returns the summary (energies, files written, validation results, warnings).
    Paths refer to the machine running this server, not a remote Colab runtime.
    """
    return run(load_config(config_path), output_dir)


@mcp.tool()
def run_adapt_vqe(results_dir: str, max_operators: int = 30,
                  gradient_tolerance: float = 1e-4) -> dict:
    """Exact-statevector ADAPT-VQE on RESULTS_DIR/qubit_hamiltonian.json (a classical reference).

    Writes RESULTS_DIR/adapt_vqe.json and returns it, including the error against
    FCI/CASCI when the run computed one. Limited to 20 qubits.
    """
    return run_adapt(results_dir, max_operators=max_operators,
                     gradient_tolerance=gradient_tolerance)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
