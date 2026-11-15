import ast
import re

import nbformat
import pytest
from conftest import ROOT

NOTEBOOKS = ["colab.ipynb", "lihbn_reproduction.ipynb"]


def code_cells(name):
    notebook = nbformat.read(ROOT / "notebooks" / name, as_version=4)
    nbformat.validate(notebook)
    return [cell for cell in notebook.cells if cell.cell_type == "code"]


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebooks_are_valid_clean_and_parse(name):
    for cell in code_cells(name):
        assert not cell.get("outputs") and cell.get("execution_count") is None
        python = "\n".join(line for line in cell.source.splitlines()
                           if not line.lstrip().startswith(("%", "!")))
        ast.parse(python)


def test_colab_notebook_references_existing_examples_and_distinct_outputs():
    sources = "\n".join(cell.source for cell in code_cells("colab.ipynb"))
    referenced = set(re.findall(r"examples/([\w.+-]+\.json)", sources))
    referenced |= set(re.findall(r"COLABEXERCISE_QUBIT_EXAMPLE', '([\w+-]+\.json)'", sources))
    assert referenced, "expected the notebook to run example configs"
    for name in referenced:
        assert (ROOT / "examples" / name).is_file(), name
    # scripts/verify_colab.py reads the H2 run from pyscf-*; other runs must not match it.
    prefixes = re.findall(r"prefix='([\w-]+)'", sources)
    assert prefixes.count("pyscf-") == 1 and "qubit-" in prefixes
