# Authors: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD 3 clause

from datetime import datetime
from nose.tools import assert_raises, assert_true
from numpy.testing import (assert_equal, assert_array_equal,
                           assert_array_almost_equal)
import os.path as op

import numpy as np

from mne import create_info
from mne.utils import run_tests_if_main
from mne.io import read_raw_fif, RawArray, concatenate_raws
from mne.annotations import Annotations
from mne.datasets import testing

data_dir = op.join(testing.data_path(download=False), 'MEG', 'sample')
fif_fname = op.join(data_dir, 'sample_audvis_trunc_raw.fif')


@testing.requires_testing_data
def test_annotations():
    """Test annotation class."""
    raw = read_raw_fif(fif_fname)
    onset = np.array(range(10))
    duration = np.ones(10)
    description = np.repeat('test', 10)
    dt = datetime.utcnow()
    meas_date = raw.info['meas_date']
    # Test time shifts.
    for orig_time in [None, dt, meas_date[0], meas_date]:
        annot = Annotations(onset, duration, description, orig_time)

    assert_raises(ValueError, Annotations, onset, duration, description[:9])
    assert_raises(ValueError, Annotations, [onset, 1], duration, description)
    assert_raises(ValueError, Annotations, onset, [duration, 1], description)

    # Test combining annotations with concatenate_raws
    raw2 = raw.copy()
    orig_time = (meas_date[0] + meas_date[1] * 0.000001 +
                 raw2.first_samp / raw2.info['sfreq'])
    annot = Annotations(onset, duration, description, orig_time)
    raw2.annotations = annot
    assert_array_equal(raw2.annotations.onset, onset)
    concatenate_raws([raw, raw2])
    assert_array_almost_equal(onset + 20., raw.annotations.onset, decimal=2)
    assert_array_equal(annot.duration, raw.annotations.duration)
    assert_array_equal(raw.annotations.description, np.repeat('test', 10))

    # Test combining with RawArray and orig_times
    data = np.random.randn(2, 1000) * 10e-12
    sfreq = 100.
    info = create_info(ch_names=['MEG1', 'MEG2'], ch_types=['grad'] * 2,
                       sfreq=sfreq)
    info['meas_date'] = 0
    raws = []
    for i, fs in enumerate([12300, 100, 12]):
        raw = RawArray(data.copy(), info, first_samp=fs)
        ants = Annotations([1., 2.], [.5, .5], 'x', fs / sfreq)
        raw.annotations = ants
        raws.append(raw)
    raw = RawArray(data.copy(), info)
    raw.annotations = Annotations([1.], [.5], 'x', None)
    raws.append(raw)
    raw = concatenate_raws(raws)
    assert_array_equal(raw.annotations.onset, [1., 2., 11., 12., 21., 22.,
                                               31.])
    raw.annotations.delete(2)
    assert_array_equal(raw.annotations.onset, [1., 2., 12., 21., 22., 31.])
    raw.annotations.append(5, 1.5, 'y')
    assert_array_equal(raw.annotations.onset, [1., 2., 12., 21., 22., 31., 5])
    assert_array_equal(raw.annotations.duration, [.5, .5, .5, .5, .5, .5, 1.5])
    assert_array_equal(raw.annotations.description, ['x', 'x', 'x', 'x', 'x',
                                                     'x', 'y'])

    # Test concatenating annotations with and without orig_time.
    raw = read_raw_fif(fif_fname)
    last_time = raw.last_samp / raw.info['sfreq']
    raw2 = raw.copy()
    raw.annotations = Annotations([45.], [3], 'test', raw.info['meas_date'])
    raw2.annotations = Annotations([2.], [3], 'BAD', None)
    raw = concatenate_raws([raw, raw2])

    assert_array_almost_equal(raw.annotations.onset, [45., 2. + last_time],
                              decimal=2)


@testing.requires_testing_data
def test_raw_reject():
    """Test raw data getter with annotation reject."""
    info = create_info(['a', 'b', 'c', 'd', 'e'], 100, ch_types='eeg')
    raw = RawArray(np.ones((5, 15000)), info)
    raw.annotations = Annotations([2, 100, 105, 148], [2, 8, 5, 8], 'BAD')
    data = raw.get_data([0, 1, 3, 4], 100, 11200, 'omit')
    assert_array_equal(data.shape, (4, 9900))

    # with orig_time and complete overlap
    raw = read_raw_fif(fif_fname)
    raw.annotations = Annotations([44, 47, 48], [1, 3, 1], 'BAD',
                                  raw.info['meas_date'])
    data, times = raw.get_data(range(10), 0, 6000, 'omit', True)
    assert_array_equal(data.shape, (10, 4799))
    assert_equal(times[-1], raw.times[5999])
    assert_array_equal(data[:, -100:], raw[:10, 5900:6000][0])

    data, times = raw.get_data(range(10), 0, 6000, 'NaN', True)
    assert_array_equal(data.shape, (10, 6000))
    assert_equal(times[-1], raw.times[5999])
    assert_true(np.isnan(data[:, 313:613]).all())  # 1s -2s
    assert_true(not np.isnan(data[:, 614].any()))
    assert_array_equal(data[:, -100:], raw[:10, 5900:6000][0])
    assert_array_equal(raw.get_data(), raw[:][0])


run_tests_if_main()
