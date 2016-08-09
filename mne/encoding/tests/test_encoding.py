# Authors: Chris Holdgraf <choldgraf@gmail.com>
#
# License: BSD (3-clause)
import warnings
import os.path as op

from nose.tools import assert_raises, assert_true, assert_equal
import numpy as np
from numpy.testing import assert_array_equal

from mne import io, read_events, pick_types
from mne.utils import (requires_sklearn, run_tests_if_main)


data_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_dir, 'test_raw.fif')
event_name = op.join(data_dir, 'test-eve.fif')

np.random.seed(1337)

tmin, tmax = -0.1, 0.5
event_id = dict(aud_l=1, vis_l=3)

warnings.simplefilter('always')

# Loading raw data
raw = io.read_raw_fif(raw_fname, preload=True)
picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                   eog=False, exclude='bads')
picks = picks[0:2]


@requires_sklearn
def test_feature():
    from mne.encoding import (SampleMasker, EventsBinarizer,
                              FeatureDelayer, DelaysVectorizer)
    from sklearn.linear_model import Ridge
    from scipy.sparse import spmatrix, csr_matrix
    events = read_events(event_name)
    events[:, 0] -= raw.first_samp

    # --- Feature Delayer and DelayVectorizer---
    # Explicit delays + sfreq
    X = np.random.randn(1000, 2)
    X_sp = np.zeros([10, 2])
    X_sp[1, 0] = 1
    X_sp = csr_matrix(X_sp)
    for idel, isfreq in [[[0, 1, 2], 1], [[0, .1, .2], 10]]:
        delayer = FeatureDelayer(delays=idel, sfreq=isfreq)
        # sfreq must be int/float
        assert_raises(ValueError, FeatureDelayer, sfreq=[1])
        # Delays must be 1D
        assert_raises(ValueError, FeatureDelayer, delays=[idel])
        # Delays must be int/float
        assert_raises(ValueError, FeatureDelayer,
                      delays=np.array(idel, dtype=np.complex))

        for idata in [X, X_sp]:
            X_delayed = delayer.transform(X)
            assert_array_equal(X_delayed[:, 0, 0], X[:, 0])
            assert_array_equal(X_delayed[:-1, 0, 1], X[1:, 0])
            assert_equal(X_delayed.shape[-1], len(idel))

            vectorizer = DelaysVectorizer()
            X_vec = vectorizer.transform(X_delayed, idel)
            X_vec_inv = vectorizer.inverse_transform(X_vec)
            assert_array_equal(X_delayed, X_vec_inv)
            assert_array_equal(vectorizer.delays_, np.array(idel))
            # len(y) must be len(X)
            assert_raises(ValueError, vectorizer.transform, X_delayed,
                          y=idel[:2])
            # y must be 1D
            assert_raises(ValueError, vectorizer.transform, X_delayed,
                          y=[idel])
            # Must be an array
            assert_raises(ValueError, vectorizer.transform, [1, 2, 3],
                          y=idel[:2])
            # Must be 3D
            assert_raises(ValueError, vectorizer.transform, X_delayed[0],
                          y=idel[:2])
            # Inv X must be an array
            assert_raises(ValueError, vectorizer.inverse_transform, [1, 2, 3])

    # --- Events Binarizer ---
    # EventsBinarizer must have proper events shape
    binarizer = EventsBinarizer(raw.n_times)
    assert_raises(ValueError, binarizer.transform, events)

    # Test outputs are correct when working properly
    events = events[events[:, 0] <= raw.n_times, :]
    binarizer = EventsBinarizer(raw.n_times, sparse=True)
    ev_cont = binarizer.transform(events[:, 0])
    assert_true(isinstance(ev_cont, spmatrix))
    assert_equal(ev_cont.shape[1], raw.n_times)
    assert_true(all([1 == ev_cont[0, ii] for ii in events[:, 0]]))
    # n_times must be an int
    assert_raises(ValueError, EventsBinarizer, 5.)
    # sfreq must be a float
    assert_raises(ValueError, EventsBinarizer, 5, sfreq=[1])
    # sparse must be bool
    assert_raises(ValueError, EventsBinarizer, 5, sparse=1)
    # Events must not have ix greater than n_times
    events[-1, 0] = 99999
    assert_raises(ValueError, binarizer.transform, events[:, 0])

    # --- SampleMasker ---
    # Subsetter works for indexing
    data = np.arange(100)[:, np.newaxis]
    masker = SampleMasker(Ridge(), samples_train=data[:50, 0],
                          samples_pred=data[50:, 0])
    assert_array_equal(data[masker.samples_train], data[:50])
    assert_array_equal(data[masker.samples_pred], data[50:])
    # Subsetter indices must not exceed length of data
    sub = SampleMasker(Ridge(), samples_train=[1, 99999999])
    assert_raises(ValueError, sub.fit, data, data[:, 0])
    # Create data
    X = np.tile(np.arange(100.), [10, 1]).T
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
    assert_true(np.all(np.isnan(X[~mod.mask])))

    # Ensure that other numbers work
    X = np.tile(np.arange(100.), [10, 1]).T
    y = np.arange(100)
    mod = SampleMasker(Ridge(), mask_val=10)
    mod.fit(X, y)
    assert_true(np.where(~mod.mask)[0][0] == mod.mask_val)

    # Make sure a callable returns a 1-d output
    def tmp(a):
        return a < 5
    mod = SampleMasker(Ridge(), mask_val=tmp)
    assert_raises(ValueError, mod.fit, X, y)

    # Make sure callable works
    def tmp(a):
        return (a < 5).all(1)
    mod = SampleMasker(Ridge(), mask_val=tmp)
    mod.fit(X, y)
    assert_equal(mod.mask.sum(), 95)

    # Prediction samples
    mod = SampleMasker(Ridge(), mask_val=np.nan, samples_pred=[1, 2, 3])
    mod.fit(X, y)
    y_pred = mod.predict(X)
    assert_equal(y_pred.shape[0], 3)

    # Misc
    assert_raises(ValueError, SampleMasker, Ridge(), mask_val=0,
                  samples_train=[1, 2])
    assert_raises(ValueError, SampleMasker, Ridge(), mask_val=0,
                  mask_condition='foo')


@requires_sklearn
def test_encoding():
    from mne.encoding import SampleMasker
    from mne.encoding.model import _check_estimator, get_coefs, _get_final_est
    from sklearn.pipeline import make_pipeline, Pipeline
    from sklearn.linear_model import Ridge
    # Make sure estimator pulling works
    mod = Ridge()
    pipe = make_pipeline(Ridge())
    samp = SampleMasker(pipe)
    est = _get_final_est(pipe)
    assert_true(isinstance(est, type(mod)))
    est = _get_final_est(samp)
    assert_true(isinstance(est, type(mod)))
    # Est must be fit first
    assert_raises(ValueError, get_coefs, est)
    # Coefs are correctly taken
    est.fit([[1, 2], [3, 4]], [1, 2])
    coefs = get_coefs(samp, 'coef_')
    assert_equal(coefs.shape[-1], 2)
    # Incorrect coefficient name
    assert_raises(ValueError, get_coefs, est, 'foo')

    # Make sure the checks are working
    # None returns Ridge instance
    assert_true(isinstance(_check_estimator(None), Pipeline))
    assert_true(isinstance(_get_final_est(_check_estimator(None)), type(mod)))
    # Correct ridge solver
    assert_equal(_get_final_est(_check_estimator('lsqr')).solver, 'lsqr')
    # Incorrect string type
    assert_raises(ValueError, _check_estimator, 'foo')
    # Strings return an estimator instance
    assert_true(isinstance(_get_final_est(_check_estimator(None)), Ridge))
    # Estimator must have fit/predict methods
    assert_raises(ValueError, _check_estimator, lambda a: a + 1)

run_tests_if_main()
