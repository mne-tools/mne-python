import os

PATH = os.path.dirname(os.path.realpath(__file__))


def test_notebook_3d_backend(renderer_notebook):
    """Test executing a notebook that should not fail."""
    import nbformat

    notebook = nbformat.read(
        os.path.join(PATH, "test.ipynb"), as_version=4,
    )
    exec_results = renderer_notebook.execute_notebook(notebook)
    if exec_results.exec_error:
        raise exec_results.exec_error
