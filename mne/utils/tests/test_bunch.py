# -*- coding: utf-8 -*-
# Authors: Clemens Brunner <clemens.brunner@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

import pickle
from mne.utils import BunchConstNamed
from mne.utils._bunch import NamedInt, NamedFloat


def test_pickle():
    """Test if BunchConstNamed object can be pickled."""
    b1 = BunchConstNamed()
    b1.x = 1
    b1.y = 2.12
    assert isinstance(b1.x, int)
    assert isinstance(b1.x, NamedInt)
    assert repr(b1.x) == '1 (x)'
    assert isinstance(b1.y, float)
    assert isinstance(b1.y, NamedFloat)
    assert repr(b1.y) == '2.12 (y)'

    b2 = pickle.loads(pickle.dumps(b1))
    assert b1 == b2
