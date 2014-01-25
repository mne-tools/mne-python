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
    events = np.r_[read_events(event_name), [[38711,     0,     1]]]

    picks = fiff.pick_types(raw.info, meg=True, stim=False, ecg=False,
                            eog=False, exclude='bads')
    picks = picks[1:13:3]
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True)

    # XXX : can you remove this?
    """how to create data for tests against matlab
    scipy.io.savemat('test_ems.mat', {'data': np.transpose(epochs_data,
                                                           [1, 2, 0]),
                                      'conds': conditions})
    # matlab
    epochs = load('test_ems.mat')
    [trl, sfl] = ems_ncond(epochs.data, boolean(epochs.conds))
    save('sfl.mat', 'sfl')
    """

    assert_raises(ValueError, compute_ems, epochs, [1, 3])
    epochs.equalize_event_counts(epochs.event_id, copy=False)

    assert_raises(ValueError, compute_ems, epochs, [1, 'hahah'])
    surrogates, filters, conditions = compute_ems(epochs)
    assert_equal(list(set(conditions)), [1, 3])

    conditions = [np.intp(epochs.events[:, 2] == k) for k in [1, 3]]
    surrogates3, filters3 = compute_ems(epochs, conditions)[:2]
    surrogates4, filters4 = compute_ems(epochs,
                                                     np.array(conditions))[:2]

    surrogates = [surrogates4, surrogates3, surrogates]
    filterss = [filters, filters3, filters4]

    candidates = combinations(surrogates, 2)
    candidates2 = combinations(filterss, 2)

    for a, b in list(candidates2) + list(candidates):
        assert_equal(a.shape, b.shape)
        assert_array_almost_equal(a, b, 15)

    events = read_events(event_name)
    event_id2 = dict(aud_l=1, aud_r=2, vis_l=3)
    epochs = Epochs(raw, events, event_id2, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True)
    epochs.equalize_event_counts(epochs.event_id, copy=False)

    n_expected = sum([len(epochs[k]) for k in ['aud_l', 'vis_l']])

    assert_raises(ValueError, compute_ems, epochs)
    surrogates, filters, conditions = compute_ems(epochs, ['aud_r', 'vis_l'])
    assert_equal(n_expected, len(surrogates))
    assert_equal(n_expected, len(conditions))
    assert_equal(list(set(conditions)), [2, 3])
