# -*- coding: utf-8 -*-
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import pytest
import warnings
# For some unknown reason, on Travis-xenial there are segfaults caused on
# the line pytest -> pdb.Pdb.__init__ -> "import readline". Forcing an
# import here seems to prevent them (!?). This suggests a potential problem
# with some other library stepping on memory where it shouldn't. It only
# seems to happen on the Linux runs that install Mayavi. Anectodally,
# @larsoner has had problems a couple of years ago where a mayavi import
# seemed to corrupt SciPy linalg function results (!), likely due to the
# associated VTK import, so this could be another manifestation of that.
import readline  # noqa


@pytest.fixture(scope='session')
def matplotlib_config():
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
        from traits.etsconfig.api import ETSConfig
    except Exception:
        pass
    else:
        ETSConfig.toolkit = 'qt4'
    try:
        with warnings.catch_warnings(record=True):  # traits
            from mayavi import mlab
    except Exception:
        pass
    else:
        mlab.options.backend = 'test'
