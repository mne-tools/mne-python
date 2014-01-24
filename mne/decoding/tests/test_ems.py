# Author: Denis A. Engemann <d.engemann@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
from itertools import combinations

from nose.tools import assert_equal, assert_raises
import numpy as np
from numpy.testing import assert_array_almost_equal

from mne import fiff, Epochs, read_events
from mne.utils import _TempDir, requires_sklearn
from mne.decoding import compute_ems
from scipy.io import loadmat

tempdir = _TempDir()

data_dir = op.join(op.dirname(__file__), '..', '..', 'fiff', 'tests', 'data')
curdir = op.join(op.dirname(__file__))

raw_fname = op.join(data_dir, 'test_raw.fif')
event_name = op.join(data_dir, 'test-eve.fif')

tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)


@requires_sklearn
def test_ems():
    """Test event-matched spatial filters"""
    raw = fiff.Raw(raw_fname, preload=False)
    events = read_events(event_name)
    picks = fiff.pick_types(raw.info, meg=True, stim=False, ecg=False,
                            eog=False, exclude='bads')
    picks = picks[1:13:3]
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True)
    """'bootstrap'
    scipy.io.savemat('test_ems.mat', {'data': np.transpose(epochs_data,
                                                           [1, 2, 0]),
                                      'conds': conditions})
    # matlab
    epochs = load('test_ems.mat')
    [trl, sfl] = ems_ncond(epochs.data, boolean(epochs.conds))
    save('sfl.mat', 'sfl')
    """
    assert_raises(ValueError, compute_ems, epochs, [1, 'hahah'])
    trial_surrogates, spatial_filter = compute_ems(epochs)

    conditions = [np.intp(epochs.events[:, 2] == k) for k in [1, 3]]
    trial_surrogates3, spatial_filter3 = compute_ems(epochs, conditions)
    trial_surrogates4, spatial_filter4 = compute_ems(epochs,
                                                     np.array(conditions))
    surrogates = [trial_surrogates4, trial_surrogates3, trial_surrogates]
    spatial_filters = [spatial_filter, spatial_filter3, spatial_filter4]
    """
    # critical tests gainst matlab
    trial_surrogates2, spatial_filter2 = [loadmat(op.join(curdir, k))[k[:3]]
                                          for k in ['trl.mat', 'sfl.mat']]
    sorrogates.append(trial_surrogates2)
    spatial_filters.append(spatial_filter2)
    """
    candidates = combinations(surrogates, 2)
    candidates2 = combinations(spatial_filters, 2)

    for a, b  in list(candidates2) +  list(candidates):
        assert_equal(a.shape, b.shape)
        assert_array_almost_equal(a, b, 15)
