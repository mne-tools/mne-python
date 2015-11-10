
# Author: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD (3-clause)

import os.path as op

from mne import concatenate_raws
from mne.io import read_raw_nicolet

base_dir = op.join(op.dirname(op.realpath(__file__)), 'data')
fname = op.join(base_dir, 'test_nicolet_raw.data')


def test_data():
    """Test reading raw nicolet files."""
    read_raw_nicolet(fname)

    # Make sure concatenation works
    raw = read_raw_nicolet(fname, preload=True)
    concatenate_raws([raw.copy(), raw])
