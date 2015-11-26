
# Author: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD (3-clause)

import os.path as op
import inspect
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
    raw = read_raw_nicolet(fname, preload=False)
    raw_preload = read_raw_nicolet(fname, preload=True)
    picks = [2, 3, 12, 13]
    assert_array_equal(raw[picks, 20:30][0], raw_preload[picks, 20:30][0])

    # Make sure concatenation works
    raw2 = concatenate_raws([raw_preload.copy(), raw_preload])

    # Test saving and reading
    out_fname = op.join(tempdir, 'test_nicolet_raw.fif')
    raw2.save(out_fname, tmax=raw.times[-1])
    raw2 = Raw(out_fname)

    full_data = raw_preload._data
    data1, times1 = raw[:10:3, 10:12]
    data2, times2 = raw2[:10:3, 10:12]
    data3, times3 = raw2[[0, 3, 6, 9], 10:12]
    assert_array_almost_equal(data1, full_data[:10:3, 10:12], 9)
    assert_array_almost_equal(data1, data2, 9)
    assert_array_almost_equal(data1, data3, 9)
    assert_array_almost_equal(times1, times2)
    assert_array_almost_equal(times1, times3)
