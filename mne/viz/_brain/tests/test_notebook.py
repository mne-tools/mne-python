import os
import pytest

from mne.datasets import testing
from mne.utils import requires_version

PATH = os.path.dirname(os.path.realpath(__file__))


@pytest.mark.slowtest
@testing.requires_testing_data
@requires_version('nbformat')
def test_notebook_3d_backend(renderer_notebook):
    """Test executing a notebook that should not fail."""
    import nbformat

    notebook = nbformat.read(
        os.path.join(PATH, "test.ipynb"), as_version=4,
    )
    exec_results = renderer_notebook.execute_notebook(notebook)
    if exec_results.exec_error:
        raise exec_results.exec_error
