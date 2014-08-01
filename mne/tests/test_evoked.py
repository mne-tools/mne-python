# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#         Denis Engemann <denis.engemann@gmail.com>
#         Andrew Dykstra <andrew.r.dykstra@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
from copy import deepcopy
import warnings

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_equal,
                           assert_array_equal, assert_allclose)
from nose.tools import assert_true, assert_raises, assert_not_equal

from mne import (equalize_channels, pick_types, read_evoked, write_evoked,
                 read_evokeds, write_evokeds)
from mne.evoked import _get_peak, EvokedArray
from mne.epochs import EpochsArray

from mne.utils import _TempDir, requires_pandas, requires_nitime

from mne.io.meas_info import create_info
from mne.externals.six.moves import cPickle as pickle

warnings.simplefilter('always')

fname = op.join(op.dirname(__file__), '..', 'io', 'tests', 'data',
                'test-ave.fif')
fname_gz = op.join(op.dirname(__file__), '..', 'io', 'tests', 'data',
                   'test-ave.fif.gz')

tempdir = _TempDir()


def test_hash_evoked():
    """Test evoked hashing
    """
    ave = read_evokeds(fname, 0)
    ave_2 = read_evokeds(fname, 0)
    assert_equal(hash(ave), hash(ave_2))
    # do NOT use assert_equal here, failing output is terrible
    assert_true(pickle.dumps(ave) == pickle.dumps(ave_2))

    ave_2.data[0, 0] -= 1
    assert_not_equal(hash(ave), hash(ave_2))


def test_io_evoked():
    """Test IO for evoked data (fif + gz) with integer and str args
    """
    ave = read_evokeds(fname, 0)

    write_evokeds(op.join(tempdir, 'evoked-ave.fif'), ave)
    ave2 = read_evokeds(op.join(tempdir, 'evoked-ave.fif'))[0]

    # This not being assert_array_equal due to windows rounding
    assert_true(np.allclose(ave.data, ave2.data, atol=1e-16, rtol=1e-3))
    assert_array_almost_equal(ave.times, ave2.times)
    assert_equal(ave.nave, ave2.nave)
    assert_equal(ave._aspect_kind, ave2._aspect_kind)
    assert_equal(ave.kind, ave2.kind)
    assert_equal(ave.last, ave2.last)
    assert_equal(ave.first, ave2.first)

    # test compressed i/o
    ave2 = read_evokeds(fname_gz, 0)
    assert_true(np.allclose(ave.data, ave2.data, atol=1e-16, rtol=1e-8))

    # test str access
    condition = 'Left Auditory'
    assert_raises(ValueError, read_evokeds, fname, condition, kind='stderr')
    assert_raises(ValueError, read_evokeds, fname, condition,
                  kind='standard_error')
    ave3 = read_evokeds(fname, condition)
    assert_array_almost_equal(ave.data, ave3.data, 19)

    # test deprecation warning for read_evoked and write_evoked
    # XXX should be deleted for 0.9 release
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        ave = read_evoked(fname, setno=0)
        assert_true(w[0].category == DeprecationWarning)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        write_evoked(op.join(tempdir, 'evoked-ave.fif'), ave)
        assert_true(w[0].category == DeprecationWarning)

    # test read_evokeds and write_evokeds
    types = ['Left Auditory', 'Right Auditory', 'Left visual', 'Right visual']
    aves1 = read_evokeds(fname)
    aves2 = read_evokeds(fname, [0, 1, 2, 3])
    aves3 = read_evokeds(fname, types)
    write_evokeds(op.join(tempdir, 'evoked-ave.fif'), aves1)
    aves4 = read_evokeds(op.join(tempdir, 'evoked-ave.fif'))
    for aves in [aves2, aves3, aves4]:
        for [av1, av2] in zip(aves1, aves):
            assert_array_almost_equal(av1.data, av2.data)
            assert_array_almost_equal(av1.times, av2.times)
            assert_equal(av1.nave, av2.nave)
            assert_equal(av1.kind, av2.kind)
            assert_equal(av1._aspect_kind, av2._aspect_kind)
            assert_equal(av1.last, av2.last)
            assert_equal(av1.first, av2.first)
            assert_equal(av1.comment, av2.comment)

    # test warnings on bad filenames
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        fname2 = op.join(tempdir, 'test-bad-name.fif')
        write_evokeds(fname2, ave)
        read_evokeds(fname2)
    assert_true(len(w) == 2)


def test_shift_time_evoked():
    """ Test for shifting of time scale
    """
    # Shift backward
    ave = read_evokeds(fname, 0)
    ave.shift_time(-0.1, relative=True)
    write_evokeds(op.join(tempdir, 'evoked-ave.fif'), ave)

    # Shift forward twice the amount
    ave_bshift = read_evokeds(op.join(tempdir, 'evoked-ave.fif'), 0)
    ave_bshift.shift_time(0.2, relative=True)
    write_evokeds(op.join(tempdir, 'evoked-ave.fif'), ave_bshift)

    # Shift backward again
    ave_fshift = read_evokeds(op.join(tempdir, 'evoked-ave.fif'), 0)
    ave_fshift.shift_time(-0.1, relative=True)
    write_evokeds(op.join(tempdir, 'evoked-ave.fif'), ave_fshift)

    ave_normal = read_evokeds(fname, 0)
    ave_relative = read_evokeds(op.join(tempdir, 'evoked-ave.fif'), 0)

    assert_true(np.allclose(ave_normal.data, ave_relative.data,
                            atol=1e-16, rtol=1e-3))
    assert_array_almost_equal(ave_normal.times, ave_relative.times, 10)

    assert_equal(ave_normal.last, ave_relative.last)
    assert_equal(ave_normal.first, ave_relative.first)

    # Absolute time shift
    ave = read_evokeds(fname, 0)
    ave.shift_time(-0.3, relative=False)
    write_evokeds(op.join(tempdir, 'evoked-ave.fif'), ave)

    ave_absolute = read_evokeds(op.join(tempdir, 'evoked-ave.fif'), 0)

    assert_true(np.allclose(ave_normal.data, ave_absolute.data,
                            atol=1e-16, rtol=1e-3))
    assert_equal(ave_absolute.first, int(-0.3 * ave.info['sfreq']))


def test_evoked_resample():
    """Test for resampling of evoked data
    """
    # upsample, write it out, read it in
    ave = read_evokeds(fname, 0)
    sfreq_normal = ave.info['sfreq']
    ave.resample(2 * sfreq_normal)
    write_evokeds(op.join(tempdir, 'evoked-ave.fif'), ave)
    ave_up = read_evokeds(op.join(tempdir, 'evoked-ave.fif'), 0)

    # compare it to the original
    ave_normal = read_evokeds(fname, 0)

    # and compare the original to the downsampled upsampled version
    ave_new = read_evokeds(op.join(tempdir, 'evoked-ave.fif'), 0)
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
    ave = read_evokeds(fname, 0)
    ave_normal = read_evokeds(fname, 0)
    ave.detrend(0)
    ave_normal.data -= np.mean(ave_normal.data, axis=1)[:, np.newaxis]
    picks = pick_types(ave.info, meg=True, eeg=True, exclude='bads')
    assert_true(np.allclose(ave.data[picks], ave_normal.data[picks],
                            rtol=1e-8, atol=1e-16))


@requires_nitime
def test_evoked_to_nitime():
    """ Test to_nitime """
    ave = read_evokeds(fname, 0)
    evoked_ts = ave.to_nitime()
    assert_equal(evoked_ts.data, ave.data)

    picks2 = [1, 2]
    ave = read_evokeds(fname, 0)
    evoked_ts = ave.to_nitime(picks=picks2)
    assert_equal(evoked_ts.data, ave.data[picks2])


@requires_pandas
def test_as_data_frame():
    """Test evoked Pandas exporter"""
    ave = read_evokeds(fname, 0)
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
        ave = read_evokeds(fname, condition=0, proj=proj)
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

    ave = read_evokeds(fname, condition=0, proj=False)
    data = ave.data.copy()
    ave.apply_proj()
    assert_allclose(np.dot(ave._projector, data), ave.data)


def test_get_peak():
    """Test peak getter
    """

    evoked = read_evokeds(fname, condition=0, proj=True)
    assert_raises(ValueError, evoked.get_peak, ch_type='mag', tmin=1)
    assert_raises(ValueError, evoked.get_peak, ch_type='mag', tmax=0.9)
    assert_raises(ValueError, evoked.get_peak, ch_type='mag', tmin=0.02,
                  tmax=0.01)
    assert_raises(ValueError, evoked.get_peak, ch_type='mag', mode='foo')
    assert_raises(RuntimeError, evoked.get_peak, ch_type=None, mode='foo')
    assert_raises(ValueError, evoked.get_peak, ch_type='misc', mode='foo')

    ch_idx, time_idx = evoked.get_peak(ch_type='mag')
    assert_true(ch_idx in evoked.ch_names)
    assert_true(time_idx in evoked.times)

    ch_idx, time_idx = evoked.get_peak(ch_type='mag',
                                       time_as_index=True)
    assert_true(time_idx < len(evoked.times))

    data = np.array([[0., 1.,  2.],
                     [0., -3.,  0]])

    times = np.array([.1, .2, .3])

    ch_idx, time_idx = _get_peak(data, times, mode='abs')
    assert_equal(ch_idx, 1)
    assert_equal(time_idx, 1)

    ch_idx, time_idx = _get_peak(data * -1, times, mode='neg')
    assert_equal(ch_idx, 0)
    assert_equal(time_idx, 2)

    ch_idx, time_idx = _get_peak(data, times, mode='pos')
    assert_equal(ch_idx, 0)
    assert_equal(time_idx, 2)

    assert_raises(ValueError, _get_peak, data + 1e3, times, mode='neg')
    assert_raises(ValueError, _get_peak, data - 1e3, times, mode='pos')


def test_drop_channels_mixin():
    """Test channels-dropping functionality
    """
    evoked = read_evokeds(fname, condition=0, proj=True)
    drop_ch = evoked.ch_names[:3]
    ch_names = evoked.ch_names[3:]

    ch_names_orig = evoked.ch_names
    dummy = evoked.drop_channels(drop_ch, copy=True)
    assert_equal(ch_names, dummy.ch_names)
    assert_equal(ch_names_orig, evoked.ch_names)
    assert_equal(len(ch_names_orig), len(evoked.data))

    evoked.drop_channels(drop_ch)
    assert_equal(ch_names, evoked.ch_names)
    assert_equal(len(ch_names), len(evoked.data))


def test_pick_channels_mixin():
    """Test channel-picking functionality
    """
    evoked = read_evokeds(fname, condition=0, proj=True)
    ch_names = evoked.ch_names[:3]

    ch_names_orig = evoked.ch_names
    dummy = evoked.pick_channels(ch_names, copy=True)
    assert_equal(ch_names, dummy.ch_names)
    assert_equal(ch_names_orig, evoked.ch_names)
    assert_equal(len(ch_names_orig), len(evoked.data))

    evoked.pick_channels(ch_names)
    assert_equal(ch_names, evoked.ch_names)
    assert_equal(len(ch_names), len(evoked.data))


def test_equalize_channels():
    """Test equalization of channels
    """
    evoked1 = read_evokeds(fname, condition=0, proj=True)
    evoked2 = evoked1.copy()
    ch_names = evoked1.ch_names[2:]
    evoked1.drop_channels(evoked1.ch_names[:1])
    evoked2.drop_channels(evoked2.ch_names[1:2])
    my_comparison = [evoked1, evoked2]
    equalize_channels(my_comparison)
    for e in my_comparison:
        assert_equal(ch_names, e.ch_names)


def test_array_epochs():
    """Test creating evoked from array
    """

    # creating
    rng = np.random.RandomState(42)
    data1 = rng.randn(20, 60)
    sfreq = 1e3
    ch_names = ['EEG %03d' % (i + 1) for i in range(20)]
    types = ['eeg'] * 20
    info = create_info(ch_names, sfreq, types)
    evoked1 = EvokedArray(data1, info, tmin=-0.01)

    # save, read, and compare evokeds
    tmp_fname = op.join(tempdir, 'evkdary-ave.fif')
    evoked1.save(tmp_fname)
    evoked2 = read_evokeds(tmp_fname)[0]
    data2 = evoked2.data
    assert_allclose(data1, data2)
    assert_allclose(evoked1.times, evoked2.times)
    assert_equal(evoked1.first, evoked2.first)
    assert_equal(evoked1.last, evoked2.last)
    assert_equal(evoked1.kind, evoked2.kind)
    assert_equal(evoked1.nave, evoked2.nave)

    # now compare with EpochsArray (with single epoch)
    data3 = data1[np.newaxis, :, :]
    events = np.c_[10, 0, 1]
    evoked3 = EpochsArray(data3, info, events=events, tmin=-0.01).average()
    assert_allclose(evoked1.data, evoked3.data)
    assert_allclose(evoked1.times, evoked3.times)
    assert_equal(evoked1.first, evoked3.first)
    assert_equal(evoked1.last, evoked3.last)
    assert_equal(evoked1.kind, evoked3.kind)
    assert_equal(evoked1.nave, evoked3.nave)

    # test match between channels info and data
    ch_names = ['EEG %03d' % (i + 1) for i in range(19)]
    types = ['eeg'] * 19
    info = create_info(ch_names, sfreq, types)
    assert_raises(ValueError, EvokedArray, data1, info, tmin=-0.01)
