# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#         Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import os.path as op

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal
from nose.tools import assert_true

from mne.fiff import read_evoked, write_evoked

fname = op.join(op.dirname(__file__), 'data', 'test-ave.fif')

try:
    import nitime
except ImportError:
    have_nitime = False
else:
    have_nitime = True
nitime_test = np.testing.dec.skipif(not have_nitime, 'nitime not installed')


def test_io_evoked():
    """Test IO for evoked data
    """
    ave = read_evoked(fname)

    ave.crop(tmin=0)

    write_evoked('evoked.fif', ave)
    ave2 = read_evoked('evoked.fif')

    assert_array_almost_equal(ave.data, ave2.data)
    assert_array_almost_equal(ave.times, ave2.times)
    assert_equal(ave.nave, ave2.nave)
    assert_equal(ave.aspect_kind, ave2.aspect_kind)
    assert_equal(ave.last, ave2.last)
    assert_equal(ave.first, ave2.first)


def test_evoked_resample():
    """Test for resampling of evoked data
    """
    # upsample, write it out, read it in
    ave = read_evoked(fname)
    sfreq_normal = ave.info['sfreq']
    ave.resample(2 * sfreq_normal)
    write_evoked('evoked.fif', ave)
    ave_up = read_evoked('evoked.fif')

    # compare it to the original
    ave_normal = read_evoked(fname)

    # and compare the original to the downsampled upsampled version
    ave_new = read_evoked('evoked.fif')
    ave_new.resample(sfreq_normal)

    assert_array_almost_equal(ave_normal.data, ave_new.data, 2)
    assert_array_almost_equal(ave_normal.times, ave_new.times)
    assert_equal(ave_normal.nave, ave_new.nave)
    assert_equal(ave_normal.aspect_kind, ave_new.aspect_kind)
    assert_equal(ave_normal.last, ave_new.last)
    assert_equal(ave_normal.first, ave_new.first)

    # for the above to work, the upsampling just about had to, but
    # we'll add a couple extra checks anyway
    assert_true(len(ave_up.times) == 2 * len(ave_normal.times))
    assert_true(ave_up.data.shape[1] == 2 * ave_normal.data.shape[1])


def test_io_multi_evoked():
    """Test IO for multiple evoked datasets
    """
    aves = read_evoked(fname, [0, 1, 2, 3])
    write_evoked('evoked.fif', aves)
    aves2 = read_evoked('evoked.fif', [0, 1, 2, 3])
    for [ave, ave2] in zip(aves, aves2):
        assert_array_almost_equal(ave.data, ave2.data)
        assert_array_almost_equal(ave.times, ave2.times)
        assert_equal(ave.nave, ave2.nave)
        assert_equal(ave.aspect_kind, ave2.aspect_kind)
        assert_equal(ave.last, ave2.last)
        assert_equal(ave.first, ave2.first)


@nitime_test
def test_evoked_to_nitime():
    """ Test to_nitime """
    aves = read_evoked(fname, [0, 1, 2, 3])
    evoked_ts = aves[0].to_nitime()
    assert_equal(evoked_ts.data, aves[0].data)

    picks2 = [1, 2]
    aves = read_evoked(fname, [0, 1, 2, 3])
    evoked_ts = aves[0].to_nitime(picks=picks2)
    assert_equal(evoked_ts.data, aves[0].data[picks2])
