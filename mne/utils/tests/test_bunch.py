# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import pickle

from mne.utils import BunchConstNamed
from mne.utils._bunch import NamedFloat, NamedInt


def test_pickle():
    """Test if BunchConstNamed object can be pickled."""
    b1 = BunchConstNamed()
    b1.x = 1
    b1.y = 2.12
    assert isinstance(b1.x, int)
    assert isinstance(b1.x, NamedInt)
    assert repr(b1.x) == "1 (x)"
    assert isinstance(b1.y, float)
    assert isinstance(b1.y, NamedFloat)
    assert repr(b1.y) == "2.12 (y)"

    b2 = pickle.loads(pickle.dumps(b1))  # nosec B301
    assert b1 == b2
