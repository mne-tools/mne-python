import os
import nbformat
from pytest_notebook.execution import execute_notebook

PATH = os.path.dirname(os.path.realpath(__file__))


def test_execute_notebook():
    """Test executing a notebook that should fail."""
    notebook = nbformat.read(
        os.path.join(PATH, "test.ipynb"), as_version=4,
    )
    exec_results = execute_notebook(notebook)
    if exec_results.exec_error:
        raise exec_results.exec_error
