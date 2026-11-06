import ast
import json

import nbformat
from conftest import ROOT

NOTEBOOK = ROOT / "notebooks/colab.ipynb"


def code_cells():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    return [cell for cell in notebook.cells if cell.cell_type == "code"]


def test_notebook_is_valid_clean_and_parses():
    for cell in code_cells():
        assert not cell.get("outputs") and cell.get("execution_count") is None
        assert cell.source.startswith("# @title"), "every code cell gets a Colab title"
        python = "\n".join(line for line in cell.source.splitlines()
                           if not line.lstrip().startswith(("%", "!")))
        ast.parse(python)


def test_notebook_examples_exist():
    sources = "\n".join(cell.source for cell in code_cells())
    marker = 'example = "'
    options = sources.split(marker, 1)[1].split("# @param", 1)[1].split("\n", 1)[0]
    for name in json.loads(options.strip()):
        assert (ROOT / "examples" / name).is_file(), name
