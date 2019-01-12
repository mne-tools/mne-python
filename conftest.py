# -*- coding: utf-8 -*-
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import pytest
import warnings
import mne


@pytest.fixture(scope='session')
def matplotlib_config(doctest_namespace):
    """Configure matplotlib for viz tests."""
    import matplotlib
    matplotlib.use('agg')  # don't pop up windows
    import matplotlib.pyplot as plt
    assert plt.get_backend() == 'agg'
    # overwrite some params that can horribly slow down tests that
    # users might have changed locally (but should not otherwise affect
    # functionality)
    plt.ioff()
    plt.rcParams['figure.dpi'] = 100
    try:
        with warnings.catch_warnings(record=True):  # traits
            from mayavi import mlab
    except Exception:
        pass
    else:
        mlab.options.backend = 'test'
    doctest_namespace['mne'] = mne
