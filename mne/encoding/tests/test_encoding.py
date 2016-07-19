# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Jean-Remi King <jeanremi.king@gmail.com>
#          Chris Holdgraf <choldgraf@gmail.com>
#
# License: BSD (3-clause)
import warnings
import os.path as op

from nose.tools import assert_raises, assert_true
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from mne import io, Epochs, read_events, pick_types
from mne.utils import (requires_sklearn, run_tests_if_main)


data_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_dir, 'test_raw.fif')
event_name = op.join(data_dir, 'test-eve.fif')

np.random.seed(1337)

tmin, tmax = -0.1, 0.5
event_id = dict(aud_l=1, vis_l=3)

warnings.simplefilter('always')

# Loading raw data + epochs
raw = io.read_raw_fif(raw_fname, preload=True)
events = read_events(event_name)
picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                   eog=False, exclude='bads')
picks = picks[0:2]

with warnings.catch_warnings(record=True):
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=None, preload=True)


@requires_sklearn
def test_rerp():
    from mne.encoding import EventRelatedRegressor, clean_inputs
    rerp = EventRelatedRegressor(raw, events, est='cholesky',
                                 event_id=event_id, tmin=tmin, tmax=tmax,
                                 preproc_func_xy=clean_inputs, picks=picks)
    rerp.fit()

    cond = 'aud_l'
    evoked_erp = rerp.to_evoked()[cond]
    evoked_avg = epochs[cond].average()

    assert_array_almost_equal(evoked_erp.data, evoked_avg.data, 12)

    # Make sure events are MNE-style
    raw_nopreload = io.read_raw_fif(raw_fname, preload=False)
    assert_raises(ValueError, EventRelatedRegressor, raw, events[:, 0])
    # Data needs to be preloaded
    assert_raises(ValueError, EventRelatedRegressor, raw_nopreload, events)
    # Data must be fit before we get evoked coefficients
    rerp = EventRelatedRegressor(raw, events, est='cholesky',
                                 event_id=event_id, tmin=tmin, tmax=tmax,
                                 preproc_func_xy=clean_inputs, picks=picks)
    assert_raises(ValueError, rerp.to_evoked)


@requires_sklearn
def test_feature():
    from mne.encoding import (DataIndexer, DataMasker, EventsBinarizer,
                              DataDelayer, clean_inputs)
    from sklearn.linear_model import Ridge

    sfreq = raw.info['sfreq']
    # Delayer must have sfreq if twin given
    assert_raises(ValueError, DataDelayer, time_window=[tmin, tmax],
                  sfreq=None)
    # Must give either twin or delays
    assert_raises(ValueError, DataDelayer, time_window=[tmin, tmax],
                  sfreq=sfreq, delays=[1.])
    assert_raises(ValueError, DataDelayer)
    # EventsBinarizer must have proper events shape
    binarizer = EventsBinarizer(raw.n_times)
    assert_raises(ValueError, binarizer.fit, events)
    # Subsetter works for indexing
    data = np.arange(100)[:, np.newaxis]
    sub = DataIndexer(np.arange(50))
    data_subset = sub.fit_transform(data)
    assert_array_equal(data_subset, data[:50])
    # Subsetter works for decimation
    sub = DataIndexer(decimate=10)
    data_subset = sub.fit_transform(data)
    assert_array_equal(data_subset, data[::10])

    # Subsetter indices must not exceed length of data
    sub = DataIndexer([1, 99999999])
    assert_raises(ValueError, sub.fit, raw._data.T)
    # Cleaning inputs must have same n times
    assert_raises(ValueError, clean_inputs, raw._data.T, raw._data[0, :-1].T)

    # Create data
    X = np.tile(np.arange(100), [10, 1]).T.astype(float)
    y = np.arange(100)
    mod = DataMasker(Ridge())

    # This should remove no datapoints
    mod.fit(X, y)
    assert_true(mod.mask.sum() == X.shape[0])

    # Test that it removes nans
    X[:20, :] = np.nan
    mod.fit(X, y)
    assert_true(mod.mask.sum() == X.shape[0] - 20)
    # Make sure the right indices were removed
    assert_true((np.isnan(X[~mod.mask]).all().all()))
    # Make sure nans won't work with ints
    assert_raises(ValueError, mod.fit, X.astype(int), y)

    # Ensure that other numbers work
    X = np.tile(np.arange(100), [10, 1]).T.astype(float)
    y = np.arange(100)
    mod = DataMasker(Ridge(), mask_val=10)
    mod.fit(X, y)
    assert_true(np.where(~mod.mask)[0][0] == mod.mask_val)


run_tests_if_main()
