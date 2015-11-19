
# Author: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD (3-clause)

import os.path as op
import inspect

from mne.io import read_raw_nicolet
from mne.io.tests.test_raw import _test_raw_object, _test_raw_filter

FILE = inspect.getfile(inspect.currentframe())
base_dir = op.join(op.dirname(op.abspath(FILE)), 'data')
fname = op.join(base_dir, 'test_nicolet_raw.data')


def test_data():
    """Test reading raw nicolet files."""
    raw = _test_raw_object(read_raw_nicolet, fname)
    _test_raw_filter(raw, precision=5)
