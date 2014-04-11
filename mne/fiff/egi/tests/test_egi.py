# Authors: Denis A. Engemann  <denis.engemann@gmail.com>
#          simplified BSD-3 license


import os
import os.path as op
from copy import deepcopy
import warnings

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose)
from nose.tools import assert_true, assert_raises, assert_equal

from mne.fiff.egi import read_raw_egi
from mne.utils import _TempDir
from mne.externals.six.moves import zip

warnings.simplefilter('always')  # enable b/c these tests throw warnings

base_dir = op.join(op.dirname(__file__), 'data')
egi_fname = op.join(base_dir, 'test_egi.raw')


def test_io_egi():
    """Test impoting EGI files"""
    raw = read_raw_egi(egi_fname)

    print(raw)

    print(raw.info)

