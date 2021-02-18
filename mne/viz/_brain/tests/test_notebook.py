import os
import pytest

from mne.datasets import testing
from mne.utils import requires_version

PATH = os.path.dirname(os.path.realpath(__file__))


@pytest.mark.slowtest
@testing.requires_testing_data
@requires_version('nbformat')
@requires_version('nbclient')
@requires_version('ipympl')
def test_notebook_3d_backend(renderer_notebook, brain_gc):
    """Test executing a notebook that should not fail."""
    import nbformat
    from nbclient import NotebookClient

    notebook_filename = os.path.join(PATH, "test.ipynb")
    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)
    client = NotebookClient(nb)
    client.execute()
