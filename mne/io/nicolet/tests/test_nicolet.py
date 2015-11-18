
# Author: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD (3-clause)

import os.path as op
import inspect
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from mne import concatenate_raws
from mne.io import read_raw_nicolet, Raw
from mne.utils import _TempDir

FILE = inspect.getfile(inspect.currentframe())
base_dir = op.join(op.dirname(op.abspath(FILE)), 'data')
fname = op.join(base_dir, 'test_nicolet_raw.data')


def test_data():
    """Test reading raw nicolet files."""
    tempdir = _TempDir()
    data = list()
    for preload in [False, True]:
        raw = read_raw_nicolet(fname, preload=preload)
        picks = [2, 3, 12, 13]
        data.append(raw[picks, 20:30][0])
    assert_array_equal(data[0], data[1])

    # Make sure concatenation works
    raw = read_raw_nicolet(fname, preload=True)
    raw2 = concatenate_raws([raw.copy(), raw])

    # Test saving and reading
    out_fname = op.join(tempdir, 'test_nicolet_raw.fif')
    raw2.save(out_fname, tmax=raw.times[-1])
    raw2 = Raw(out_fname)

    data1, times1 = raw[:10, :]
    data2, times2 = raw2[:10, :]
    assert_array_almost_equal(data1, data2, 9)
    assert_array_almost_equal(times1, times2)

    data_frame = np.zeros((4, 11))
    data = raw._read_segment_file(data_frame, slice(1, 5), None, 0, 10, 20,
                                  None, None)
    assert_array_equal(raw._data[1:5, 10:21], data)
