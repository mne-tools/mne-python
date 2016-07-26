# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Jean-Remi King <jeanremi.king@gmail.com>
#          Chris Holdgraf <choldgraf@gmail.com>
#
# License: BSD (3-clause)
import warnings
import os.path as op

from nose.tools import assert_raises, assert_true, assert_equal
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
    from mne.encoding import EventRelatedRegressor
    rerp = EventRelatedRegressor(raw, events, est='cholesky',
                                 event_id=event_id, tmin=tmin, tmax=tmax,
                                 picks=picks)
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
                                 picks=picks)
    assert_raises(ValueError, rerp.to_evoked)


@requires_sklearn
def test_feature():
    from mne.encoding import (SampleMasker, EventsBinarizer,
                              FeatureDelayer)
    from sklearn.linear_model import Ridge

    sfreq = raw.info['sfreq']
    # Delayer must have sfreq if twin given
    assert_raises(ValueError, FeatureDelayer, time_window=[tmin, tmax],
                  sfreq=None)
    # Must give either twin or delays
    assert_raises(ValueError, FeatureDelayer, time_window=[tmin, tmax],
                  sfreq=sfreq, delays=[1.])
    assert_raises(ValueError, FeatureDelayer)
    # EventsBinarizer must have proper events shape
    binarizer = EventsBinarizer(raw.n_times)
    assert_raises(ValueError, binarizer.fit, events)
    # Subsetter works for indexing
    data = np.arange(1, 100)[:, np.newaxis]
    masker = SampleMasker(Ridge(), ixs=np.arange(50))
    assert_array_equal(data[masker.ixs], data[:50])

    # Subsetter indices must not exceed length of data
    sub = SampleMasker(Ridge(), ixs=[1, 99999999])
    assert_raises(ValueError, sub.fit, data, data[:, 0])

    # Create data
    X = np.tile(np.arange(100), [10, 1]).T.astype(float)
    y = np.arange(100)
    mod = SampleMasker(Ridge(), mask_val=np.nan)

    # This should remove no datapoints
    mod.fit(X, y)
    assert_true(mod.mask.sum() == X.shape[0])

    # Test that it removes nans
    X[:20, :] = np.nan
    mod.fit(X, y)
    assert_true(mod.mask.sum() == (X.shape[0] - 20))
    # Make sure the right indices were removed
    assert_true((np.isnan(X[~mod.mask]).all().all()))

    # Ensure that other numbers work
    X = np.tile(np.arange(100), [10, 1]).T.astype(float)
    y = np.arange(100)
    mod = SampleMasker(Ridge(), mask_val=10)
    mod.fit(X, y)
    assert_true(np.where(~mod.mask)[0][0] == mod.mask_val)

    # Make sure a callable returns a 1-d output
    mod = SampleMasker(Ridge(), mask_val=lambda a: a < 5)
    assert_raises(ValueError, mod.fit, X, y)
    # Make sure callable works
    mod = SampleMasker(Ridge(), mask_val=lambda a: (a < 5).all(1))
    mod.fit(X, y)
    assert_equal(mod.mask.sum(), 95)


@requires_sklearn
def test_encoding():
    from mne.encoding import (SampleMasker, get_coefs, get_final_est)
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import Ridge
    # Make sure estimator pulling works
    mod = Ridge()
    pipe = make_pipeline(Ridge())
    samp = SampleMasker(pipe)
    est = get_final_est(pipe)
    assert_true(isinstance(est, type(mod)))
    est = get_final_est(samp)
    assert_true(isinstance(est, type(mod)))
    # Est must be fit first
    assert_raises(ValueError, get_coefs, est)
    # Coefs are correctly taken
    est.fit([[1, 2], [3, 4]], [1, 2])
    coefs = get_coefs(get_final_est(samp), 'coef_')
    assert_equal(coefs.shape[-1], 2)
    # Incorrect coefficient name
    assert_raises(ValueError, get_coefs, est, 'foo')


run_tests_if_main()
