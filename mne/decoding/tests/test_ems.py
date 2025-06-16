# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_equal

pytest.importorskip("sklearn")

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.estimator_checks import parametrize_with_checks

from mne import Epochs, io, pick_types, read_events
from mne.decoding import EMS, compute_ems

data_dir = Path(__file__).parents[2] / "io" / "tests" / "data"
raw_fname = data_dir / "test_raw.fif"
event_name = data_dir / "test-eve.fif"
tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)


def test_ems():
    """Test event-matched spatial filters."""
    raw = io.read_raw_fif(raw_fname, preload=False)

    # create unequal number of events
    events = read_events(event_name)
    events[-2, 2] = 3
    picks = pick_types(
        raw.info, meg=True, stim=False, ecg=False, eog=False, exclude="bads"
    )
    picks = picks[1:13:3]
    epochs = Epochs(
        raw, events, event_id, tmin, tmax, picks=picks, baseline=(None, 0), preload=True
    )
    pytest.raises(ValueError, compute_ems, epochs, ["aud_l", "vis_l"])
    epochs.equalize_event_counts(epochs.event_id)

    pytest.raises(KeyError, compute_ems, epochs, ["blah", "hahah"])
    surrogates, filters, conditions = compute_ems(epochs)
    assert_equal(list(set(conditions)), [1, 3])

    events = read_events(event_name)
    event_id2 = dict(aud_l=1, aud_r=2, vis_l=3)
    epochs = Epochs(
        raw,
        events,
        event_id2,
        tmin,
        tmax,
        picks=picks,
        baseline=(None, 0),
        preload=True,
    )
    epochs.equalize_event_counts(epochs.event_id)

    n_expected = sum([len(epochs[k]) for k in ["aud_l", "vis_l"]])

    pytest.raises(ValueError, compute_ems, epochs)
    surrogates, filters, conditions = compute_ems(epochs, ["aud_r", "vis_l"])
    assert_equal(n_expected, len(surrogates))
    assert_equal(n_expected, len(conditions))
    assert_equal(list(set(conditions)), [2, 3])

    # test compute_ems cv
    epochs = epochs["aud_r", "vis_l"]
    epochs.equalize_event_counts(epochs.event_id)
    cv = StratifiedKFold(n_splits=3)
    compute_ems(epochs, cv=cv)
    compute_ems(epochs, cv=2)
    pytest.raises(ValueError, compute_ems, epochs, cv="foo")
    pytest.raises(ValueError, compute_ems, epochs, cv=len(epochs) + 1)
    raw.close()

    # EMS transformer, check that identical to compute_ems
    X = epochs.get_data(copy=False)
    y = epochs.events[:, 2]
    X = X / np.std(X)  # X scaled outside cv in compute_ems
    Xt, coefs = list(), list()
    ems = EMS()
    assert_equal(ems.__repr__(), "<EMS: not fitted.>")
    # manual leave-one-out to avoid sklearn version problem
    for test in range(len(y)):
        train = np.setdiff1d(range(len(y)), np.atleast_1d(test))
        ems.fit(X[train], y[train])
        coefs.append(ems.filters_)
        Xt.append(ems.transform(X[[test]]))
    assert_equal(ems.__repr__(), "<EMS: fitted with 4 filters on 2 classes.>")
    assert_array_almost_equal(filters, np.mean(coefs, axis=0))
    assert_array_almost_equal(surrogates, np.vstack(Xt))


@parametrize_with_checks([EMS()])
def test_sklearn_compliance(estimator, check):
    """Test compliance with sklearn."""
    check(estimator)
