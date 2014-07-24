# Authors: Teon Brooks <teon@nyu.edu>
#          Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import warnings

import numpy as np
from numpy.testing import assert_array_equal

from nose.tools import assert_raises, assert_true, assert_equal

import mne
from mne import read_source_estimate
from mne.datasets import sample
from mne.stats.regression import linear_regression

data_path = sample.data_path(download=False)
subjects_dir = op.join(data_path, 'subjects')
stc_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-meg-lh.stc')


@sample.requires_sample_data
def test_regression():
    """Test Ordinary Least Squares Regression
    """
    data_path = sample.data_path()
    raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
    event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
    tmin, tmax = -0.2, 0.5
    event_id = dict(aud_l=1, aud_r=2)

    # Setup for reading the raw data
    raw = mne.io.Raw(raw_fname)
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
        lm = linear_regression(epochs, design_matrix, ['intercept', 'aud'])
        assert_true(w[0].category == UserWarning)
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
        lm1 = linear_regression(stc_list, design_matrix[:len(stc_list)])
    lm2 = linear_regression(stc_gen, design_matrix[:len(stc_list)])

    for k in lm1:
        for v1, v2 in zip(lm1[k], lm2[k]):
            assert_array_equal(v1.data, v2.data)
