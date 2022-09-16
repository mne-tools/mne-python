# -*- coding: utf-8 -*-
#
# Authors: Alex Rockhill <aprockhill@mailbox.org>

# NOTE: Tests in this directory must be self-contained because they are
# executed in a separate IPython kernel.

import sys
import pytest
from mne.utils import check_version


pytestmark = pytest.mark.skipif(
    sys.platform.startswith('win') or not check_version('ipympl'),
    reason='need ipympl and nbexec does not work on Windows')


def test_widget_abstraction_notebook(nbexec):
    """Test the GUI widgets abstraction in notebook."""
    from mne.viz import set_3d_backend
    from mne.viz.backends.renderer import _get_backend
    from mne.viz.backends.tests.test_abstract import _do_widget_tests
    from IPython import get_ipython

    set_3d_backend('notebook')
    backend = _get_backend()

    ipython = get_ipython()
    ipython.magic('%matplotlib widget')

    _do_widget_tests(backend)
