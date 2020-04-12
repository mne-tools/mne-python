# Author: Denis A. Engemann <d.engemann@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal
import pytest

from mne import io, Epochs, read_events, pick_types
from mne.utils import requires_sklearn, run_tests_if_main
from mne.decoding import compute_ems, EMS

data_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
curdir = op.join(op.dirname(__file__))

raw_fname = op.join(data_dir, 'test_raw.fif')
event_name = op.join(data_dir, 'test-eve.fif')

tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)


@requires_sklearn
def test_ems():
    """Test event-matched spatial filters."""
    from sklearn.model_selection import StratifiedKFold
    raw = io.read_raw_fif(raw_fname, preload=False)

    # create unequal number of events
    events = read_events(event_name)
    events[-2, 2] = 3
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                       eog=False, exclude='bads')
    picks = picks[1:13:3]
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True)
    pytest.raises(ValueError, compute_ems, epochs, ['aud_l', 'vis_l'])
    epochs.equalize_event_counts(epochs.event_id)

    pytest.raises(KeyError, compute_ems, epochs, ['blah', 'hahah'])
    surrogates, filters, conditions = compute_ems(epochs)
    assert_equal(list(set(conditions)), [1, 3])

    events = read_events(event_name)
    event_id2 = dict(aud_l=1, aud_r=2, vis_l=3)
    epochs = Epochs(raw, events, event_id2, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True)
    epochs.equalize_event_counts(epochs.event_id)

    n_expected = sum([len(epochs[k]) for k in ['aud_l', 'vis_l']])

    pytest.raises(ValueError, compute_ems, epochs)
    surrogates, filters, conditions = compute_ems(epochs, ['aud_r', 'vis_l'])
    assert_equal(n_expected, len(surrogates))
    assert_equal(n_expected, len(conditions))
    assert_equal(list(set(conditions)), [2, 3])

    # test compute_ems cv
    epochs = epochs['aud_r', 'vis_l']
    epochs.equalize_event_counts(epochs.event_id)
    cv = StratifiedKFold(n_splits=3)
    compute_ems(epochs, cv=cv)
    compute_ems(epochs, cv=2)
    pytest.raises(ValueError, compute_ems, epochs, cv='foo')
    pytest.raises(ValueError, compute_ems, epochs, cv=len(epochs) + 1)
    raw.close()

    # EMS transformer, check that identical to compute_ems
    X = epochs.get_data()
    y = epochs.events[:, 2]
    X = X / np.std(X)  # X scaled outside cv in compute_ems
    Xt, coefs = list(), list()
    ems = EMS()
    assert_equal(ems.__repr__(), '<EMS: not fitted.>')
    # manual leave-one-out to avoid sklearn version problem
    for test in range(len(y)):
        train = np.setdiff1d(range(len(y)), np.atleast_1d(test))
        ems.fit(X[train], y[train])
        coefs.append(ems.filters_)
        Xt.append(ems.transform(X[[test]]))
    assert_equal(ems.__repr__(), '<EMS: fitted with 4 filters on 2 classes.>')
    assert_array_almost_equal(filters, np.mean(coefs, axis=0))
    assert_array_almost_equal(surrogates, np.vstack(Xt))


run_tests_if_main()
