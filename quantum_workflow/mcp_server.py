from mcp.server.fastmcp import FastMCP

from .workflow import load_config, run

mcp = FastMCP("quantum-workflow")


@mcp.tool()
def run_calculation(config_path: str, output_dir: str) -> dict:
    """Run a local PySCF or QE calculation from JSON, saving to an empty directory.

    Paths refer to the machine running this server, not a remote Colab runtime.
    """
    return run(load_config(config_path), output_dir)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
