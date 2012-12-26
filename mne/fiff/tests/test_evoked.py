# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#         Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import os.path as op

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal,\
                          assert_array_equal
from nose.tools import assert_true, assert_raises

from mne.fiff import read_evoked, write_evoked
from mne.utils import _TempDir, requires_pandas, requires_nitime

fname = op.join(op.dirname(__file__), 'data', 'test-ave.fif')
fname_gz = op.join(op.dirname(__file__), 'data', 'test-ave.fif.gz')

tempdir = _TempDir()


def test_io_evoked():
    """Test IO for evoked data (fif + gz) with integer and str args
    """
    ave = read_evoked(fname, 0)

    ave.crop(tmin=0)

    write_evoked(op.join(tempdir, 'evoked.fif'), ave)
    ave2 = read_evoked(op.join(tempdir, 'evoked.fif'))

    assert_array_almost_equal(ave.data, ave2.data)
    assert_array_almost_equal(ave.times, ave2.times)
    assert_equal(ave.nave, ave2.nave)
    assert_equal(ave._aspect_kind, ave2._aspect_kind)
    assert_equal(ave.kind, ave2.kind)
    assert_equal(ave.last, ave2.last)
    assert_equal(ave.first, ave2.first)

    # test compressed i/o
    ave2 = read_evoked(fname_gz, 0)
    ave2.crop(tmin=0)
    assert_array_equal(ave.data, ave2.data)

    # test str access
    setno = 'Left Auditory'
    assert_raises(ValueError, read_evoked, fname, setno, kind='stderr')
    assert_raises(ValueError, read_evoked, fname, setno, kind='standard_error')
    ave3 = read_evoked(fname, setno)
    ave3.crop(tmin=0)
    assert_array_equal(ave.data, ave3.data)


def test_evoked_resample():
    """Test for resampling of evoked data
    """
    # upsample, write it out, read it in
    ave = read_evoked(fname, 0)
    sfreq_normal = ave.info['sfreq']
    ave.resample(2 * sfreq_normal)
    write_evoked(op.join(tempdir, 'evoked.fif'), ave)
    ave_up = read_evoked(op.join(tempdir, 'evoked.fif'), 0)

    # compare it to the original
    ave_normal = read_evoked(fname, 0)

    # and compare the original to the downsampled upsampled version
    ave_new = read_evoked(op.join(tempdir, 'evoked.fif'), 0)
    ave_new.resample(sfreq_normal)

    assert_array_almost_equal(ave_normal.data, ave_new.data, 2)
    assert_array_almost_equal(ave_normal.times, ave_new.times)
    assert_equal(ave_normal.nave, ave_new.nave)
    assert_equal(ave_normal._aspect_kind, ave_new._aspect_kind)
    assert_equal(ave_normal.kind, ave_new.kind)
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
    write_evoked(op.join(tempdir, 'evoked.fif'), aves)
    aves2 = read_evoked(op.join(tempdir, 'evoked.fif'), [0, 1, 2, 3])
    types = ['Left Auditory', 'Right Auditory', 'Left visual', 'Right visual']
    aves3 = read_evoked(op.join(tempdir, 'evoked.fif'), types)
    for aves_new in [aves2, aves3]:
        for [ave, ave_new] in zip(aves, aves_new):
            assert_array_almost_equal(ave.data, ave_new.data)
            assert_array_almost_equal(ave.times, ave_new.times)
            assert_equal(ave.nave, ave_new.nave)
            assert_equal(ave.kind, ave_new.kind)
            assert_equal(ave._aspect_kind, ave_new._aspect_kind)
            assert_equal(ave.last, ave_new.last)
            assert_equal(ave.first, ave_new.first)
    # this should throw an error since there are mulitple datasets
    assert_raises(ValueError, read_evoked, fname)


@requires_nitime
def test_evoked_to_nitime():
    """ Test to_nitime """
    aves = read_evoked(fname, [0, 1, 2, 3])
    evoked_ts = aves[0].to_nitime()
    assert_equal(evoked_ts.data, aves[0].data)

    picks2 = [1, 2]
    aves = read_evoked(fname, [0, 1, 2, 3])
    evoked_ts = aves[0].to_nitime(picks=picks2)
    assert_equal(evoked_ts.data, aves[0].data[picks2])


@requires_pandas
def test_as_data_frame():
    """Test Pandas exporter"""
    ave = read_evoked(fname, [0])[0]
    assert_raises(ValueError, ave.as_data_frame, picks=np.arange(400))
    df = ave.as_data_frame()
    assert_true((df.columns == ave.ch_names).all())
    df = ave.as_data_frame(use_time_index=False)
    assert_true('time' in df.columns)
    assert_array_equal(df.values[:, 1], ave.data[0] * 1e13)
    assert_array_equal(df.values[:, 3], ave.data[2] * 1e15)
