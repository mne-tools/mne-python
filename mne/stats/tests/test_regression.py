# Authors: Teon Brooks <teon.brooks@gmail.com>
#          Denis A. Engemann <denis.engemann@gmail.com>
#          Jona Sassenhagen <jona.sassenhagen@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import warnings

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from scipy.signal import hann

from nose.tools import assert_raises, assert_true, assert_equal

import mne
from mne import read_source_estimate
from mne.datasets import testing
from mne.stats.regression import linear_regression, linear_regression_raw
from mne.io import RawArray

warnings.simplefilter('always')

data_path = testing.data_path(download=False)
stc_fname = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-lh.stc')
raw_fname = data_path + '/MEG/sample/sample_audvis_trunc_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_trunc_raw-eve.fif'


@testing.requires_testing_data
def test_regression():
    """Test Ordinary Least Squares Regression."""
    tmin, tmax = -0.2, 0.5
    event_id = dict(aud_l=1, aud_r=2)

    # Setup for reading the raw data
    raw = mne.io.read_raw_fif(raw_fname)
    events = mne.read_events(event_fname)[:10]
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        baseline=(None, 0))
    picks = np.arange(len(epochs.ch_names))
    evoked = epochs.average(picks=picks)
    design_matrix = epochs.events[:, 1:].astype(np.float64)
    # makes the intercept
    design_matrix[:, 0] = 1
    # creates contrast: aud_l=0, aud_r=1
    design_matrix[:, 1] -= 1
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        lm = linear_regression(epochs, design_matrix, ['intercept', 'aud'])
        assert_true(w[0].category == RuntimeWarning)
        assert_true('non-data' in '%s' % w[0].message)

    for predictor, parameters in lm.items():
        for value in parameters:
            assert_equal(value.data.shape, evoked.data.shape)

    assert_raises(ValueError, linear_regression, [epochs, epochs],
                  design_matrix)

    stc = read_source_estimate(stc_fname).crop(0, 0.02)
    stc_list = [stc, stc, stc]
    stc_gen = (s for s in stc_list)
    with warnings.catch_warnings(record=True):  # divide by zero
        warnings.simplefilter('always')
        lm1 = linear_regression(stc_list, design_matrix[:len(stc_list)])
    lm2 = linear_regression(stc_gen, design_matrix[:len(stc_list)])
    for val in lm2.values():
        # all p values are 0 < p <= 1 to start, but get stored in float32
        # data, so can actually be truncated to 0. Thus the mlog10_p_val
        # actually maintains better precision for tiny p-values.
        assert_true(np.isfinite(val.p_val.data).all())
        assert_true((val.p_val.data <= 1).all())
        assert_true((val.p_val.data >= 0).all())
        # all -log10(p) are non-negative
        assert_true(np.isfinite(val.mlog10_p_val.data).all())
        assert_true((val.mlog10_p_val.data >= 0).all())
        assert_true((val.mlog10_p_val.data >= 0).all())

    for k in lm1:
        for v1, v2 in zip(lm1[k], lm2[k]):
            assert_array_equal(v1.data, v2.data)


@testing.requires_testing_data
def test_continuous_regression_no_overlap():
    """Test regression without overlap correction, on real data."""
    tmin, tmax = -.1, .5

    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    raw.apply_proj()
    events = mne.read_events(event_fname)
    event_id = dict(audio_l=1, audio_r=2)

    raw = raw.pick_channels(raw.ch_names[:2])

    epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                        baseline=None, reject=None)

    revokeds = linear_regression_raw(raw, events, event_id,
                                     tmin=tmin, tmax=tmax,
                                     reject=None)

    # Check that evokeds and revokeds are nearly equivalent
    for cond in event_id.keys():
        assert_allclose(revokeds[cond].data,
                        epochs[cond].average().data, rtol=1e-15)

    # Test events that will lead to "duplicate" errors
    old_latency = events[1, 0]
    events[1, 0] = events[0, 0]
    assert_raises(ValueError, linear_regression_raw,
                  raw, events, event_id, tmin, tmax)

    events[1, 0] = old_latency
    events[:, 0] = range(len(events))
    assert_raises(ValueError, linear_regression_raw, raw,
                  events, event_id, tmin, tmax, decim=2)


def test_continuous_regression_with_overlap():
    """Test regression with overlap correction."""
    signal = np.zeros(100000)
    times = [1000, 2500, 3000, 5000, 5250, 7000, 7250, 8000]
    events = np.zeros((len(times), 3), int)
    events[:, 2] = 1
    events[:, 0] = times
    signal[events[:, 0]] = 1.
    effect = hann(101)
    signal = np.convolve(signal, effect)[:len(signal)]
    raw = RawArray(signal[np.newaxis, :], mne.create_info(1, 100, 'eeg'))

    assert_allclose(effect,
                    linear_regression_raw(raw, events, {1: 1}, tmin=0)[1]
                    .data.flatten())
