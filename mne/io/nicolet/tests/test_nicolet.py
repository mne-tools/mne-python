
# Author: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD (3-clause)

import os.path as op
import inspect

from mne import concatenate_raws
from mne.io import read_raw_nicolet

FILE = inspect.getfile(inspect.currentframe())
parent_dir = op.dirname(op.abspath(FILE))
data_dir = op.join(parent_dir, 'data')


def test_data():
    """Test reading raw nicolet files."""
    fname = op.join(data_dir, 'test_nicolet_raw.data')
    read_raw_nicolet(fname)

    # Make sure concatenation works
    raw = read_raw_nicolet(fname, preload=True)
    concatenate_raws([raw.copy(), raw])
