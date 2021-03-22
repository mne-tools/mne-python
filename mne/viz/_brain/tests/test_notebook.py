# -*- coding: utf-8 -*-
#
# Authors: Guillaume Favelier <guillaume.favelier@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>

# NOTE: Tests in this directory must be self-contained because they are
# executed in a separate IPython kernel.

from mne.utils import requires_version


@requires_version('ipytest')
def test_3d_notebook(nbexec):
    """Test 3D tests using the notebook."""
    module = 'mne.viz._brain.tests.test_brain'
    code = f"""\
import {module} as module
assert module.__file__ is not None
import ipytest
ipytest.config(raise_on_error=True, rewrite_asserts=True, run_in_thread=True)
print(ipytest.run("-k", "screenshot", module=module, return_exit_code=True))
"""
    cell = nbexec(code)
    raise RuntimeError('\n'.join(out['text'] for out in cell.outputs))
