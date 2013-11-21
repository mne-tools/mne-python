# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#         Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import os.path as op
from copy import deepcopy

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal,\
                          assert_array_equal, assert_allclose
from nose.tools import assert_true, assert_raises

from mne.fiff import read_evoked, write_evoked, pick_types
from mne.utils import _TempDir, requires_pandas, requires_nitime

fname = op.join(op.dirname(__file__), 'data', 'test-ave.fif')
fname_gz = op.join(op.dirname(__file__), 'data', 'test-ave.fif.gz')

tempdir = _TempDir()


def test_io_evoked():
    """Test IO for evoked data (fif + gz) with integer and str args
    """
    ave = read_evoked(fname, 0)

    write_evoked(op.join(tempdir, 'evoked.fif'), ave)
    ave2 = read_evoked(op.join(tempdir, 'evoked.fif'))

    # This not being assert_array_equal due to windows rounding
    assert_true(np.allclose(ave.data, ave2.data, atol=1e-16, rtol=1e-3))
    assert_array_almost_equal(ave.times, ave2.times)
    assert_equal(ave.nave, ave2.nave)
    assert_equal(ave._aspect_kind, ave2._aspect_kind)
    assert_equal(ave.kind, ave2.kind)
    assert_equal(ave.last, ave2.last)
    assert_equal(ave.first, ave2.first)

    # test compressed i/o
    ave2 = read_evoked(fname_gz, 0)
    assert_true(np.allclose(ave.data, ave2.data, atol=1e-16, rtol=1e-8))

    # test str access
    setno = 'Left Auditory'
    assert_raises(ValueError, read_evoked, fname, setno, kind='stderr')
    assert_raises(ValueError, read_evoked, fname, setno, kind='standard_error')
    ave3 = read_evoked(fname, setno)
    assert_array_almost_equal(ave.data, ave3.data, 19)


def test_shift_time_evoked():
    """ Test for shifting of time scale
    """
    # Shift backward
    ave = read_evoked(fname, 0)
    ave.shift_time(-0.1, relative=True)
    write_evoked(op.join(tempdir, 'evoked.fif'), ave)

    # Shift forward twice the amount
    ave_bshift = read_evoked(op.join(tempdir, 'evoked.fif'), 0)
    ave_bshift.shift_time(0.2, relative=True)
    write_evoked(op.join(tempdir, 'evoked.fif'), ave_bshift)

    # Shift backward again
    ave_fshift = read_evoked(op.join(tempdir, 'evoked.fif'), 0)
    ave_fshift.shift_time(-0.1, relative=True)
    write_evoked(op.join(tempdir, 'evoked.fif'), ave_fshift)

    ave_normal = read_evoked(fname, 0)
    ave_relative = read_evoked(op.join(tempdir, 'evoked.fif'), 0)

    assert_true(np.allclose(ave_normal.data, ave_relative.data,
                            atol=1e-16, rtol=1e-3))
    assert_array_almost_equal(ave_normal.times, ave_relative.times, 10)

    assert_equal(ave_normal.last, ave_relative.last)
    assert_equal(ave_normal.first, ave_relative.first)

    # Absolute time shift
    ave = read_evoked(fname, 0)
    ave.shift_time(-0.3, relative=False)
    write_evoked(op.join(tempdir, 'evoked.fif'), ave)

    ave_absolute = read_evoked(op.join(tempdir, 'evoked.fif'), 0)

    assert_true(np.allclose(ave_normal.data, ave_absolute.data,
                            atol=1e-16, rtol=1e-3))
    assert_equal(ave_absolute.first, int(-0.3 * ave.info['sfreq']))


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


def test_evoked_detrend():
    """Test for detrending evoked data
    """
    ave = read_evoked(fname, 0)
    ave_normal = read_evoked(fname, 0)
    ave.detrend(0)
    ave_normal.data -= np.mean(ave_normal.data, axis=1)[:, np.newaxis]
    picks = pick_types(ave.info, meg=True, eeg=True, exclude='bads')
    assert_true(np.allclose(ave.data[picks], ave_normal.data[picks],
                            rtol=1e-8, atol=1e-16))


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
    """Test evoked Pandas exporter"""
    ave = read_evoked(fname, [0])[0]
    assert_raises(ValueError, ave.as_data_frame, picks=np.arange(400))
    df = ave.as_data_frame()
    assert_true((df.columns == ave.ch_names).all())
    df = ave.as_data_frame(use_time_index=False)
    assert_true('time' in df.columns)
    assert_array_equal(df.values[:, 1], ave.data[0] * 1e13)
    assert_array_equal(df.values[:, 3], ave.data[2] * 1e15)


def test_evoked_proj():
    """Test SSP proj operations
    """
    for proj in [True, False]:
        ave = read_evoked(fname, setno=0, proj=proj)
        assert_true(all(p['active'] == proj for p in ave.info['projs']))

        # test adding / deleting proj
        if proj:
            assert_raises(ValueError, ave.add_proj, [],
                          {'remove_existing': True})
            assert_raises(ValueError, ave.del_proj, 0)
        else:
            projs = deepcopy(ave.info['projs'])
            n_proj = len(ave.info['projs'])
            ave.del_proj(0)
            assert_true(len(ave.info['projs']) == n_proj - 1)
            ave.add_proj(projs, remove_existing=False)
            assert_true(len(ave.info['projs']) == 2 * n_proj - 1)
            ave.add_proj(projs, remove_existing=True)
            assert_true(len(ave.info['projs']) == n_proj)

    ave = read_evoked(fname, setno=0, proj=False)
    data = ave.data.copy()
    ave.apply_proj()
    assert_allclose(np.dot(ave._projector, data), ave.data)
