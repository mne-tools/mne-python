# # Authors: Chris Holdgraf <choldgraf@gmail.com>
# #
# # License: BSD (3-clause)
# import warnings
# import os.path as op

# from nose.tools import assert_raises, assert_true, assert_equal
# import numpy as np
# from numpy.testing import assert_array_equal, assert_array_almost_equal

# from mne import io, read_events, pick_types
# from mne.utils import (requires_sklearn, run_tests_if_main)
# from mne.encoding import (SubsetEstimator, EventsBinarizer,
#                           FeatureDelayer, get_coefs)
# from mne.encoding.model import _check_regressor, _get_final_est
# from scipy.sparse import spmatrix, csr_matrix


# data_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
# raw_fname = op.join(data_dir, 'test_raw.fif')
# event_name = op.join(data_dir, 'test-eve.fif')

# np.random.seed(1337)

# tmin, tmax = -0.1, 0.5
# event_id = dict(aud_l=1, vis_l=3)

# warnings.simplefilter('always')

# # Loading raw data
# raw = io.read_raw_fif(raw_fname, preload=True)
# picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
#                    eog=False, exclude='bads')
# picks = picks[0:2]


# @requires_sklearn
# def test_feature():
#     from sklearn.cross_validation import cross_val_score, cross_val_predict
#     from sklearn.linear_model import Ridge
#     events = read_events(event_name)
#     events[:, 0] -= raw.first_samp

#     # --- Feature Delayer ---
#     # Explicit delays + sfreq
#     X = np.random.randn(1000, 2)
#     X_sp = np.zeros([10, 2])
#     X_sp[1, 0] = 1
#     X_sp = csr_matrix(X_sp)
#     for idel, isfreq in [[[0, 1, 2], 1], [[0, .1, .2], 10]]:
#         delayer = FeatureDelayer(delays=idel, sfreq=isfreq)
#         # sfreq must be int/float
#         assert_raises(ValueError, FeatureDelayer, sfreq=[1])
#         # Delays must be 1D
#         assert_raises(ValueError, FeatureDelayer, delays=[idel])
#         # Delays must be int/float
#         assert_raises(ValueError, FeatureDelayer,
#                       delays=np.array(idel, dtype=np.complex))

#         for idata in [X, X_sp]:
#             X_delayed = delayer.transform(X)
#             assert_array_equal(X_delayed[:, 0, 0], X[:, 0])
#             assert_array_equal(X_delayed[:-1, 0, 1], X[1:, 0])
#             assert_equal(X_delayed.shape[-1], len(idel))

#     # --- Events Binarizer ---
#     # EventsBinarizer must have proper events shape
#     binarizer = EventsBinarizer(raw.n_times)
#     assert_raises(ValueError, binarizer.transform, events)

#     # Test outputs are correct when working properly
#     events = events[events[:, 0] <= raw.n_times, :]
#     binarizer = EventsBinarizer(raw.n_times, sparse=True)
#     ev_cont = binarizer.transform(events[:, 0])
#     assert_true(isinstance(ev_cont, spmatrix))
#     assert_equal(ev_cont.shape[1], raw.n_times)
#     assert_true(all([1 == ev_cont[0, ii] for ii in events[:, 0]]))
#     # n_times must be a number
#     assert_raises(ValueError, EventsBinarizer, 'foo')
#     # sfreq must be a float
#     assert_raises(ValueError, EventsBinarizer, 5, sfreq=[1])
#     # sparse must be bool
#     assert_raises(ValueError, EventsBinarizer, 5, sparse=1)
#     # Events must not have ix greater than n_times
#     events[-1, 0] = 99999
#     assert_raises(ValueError, binarizer.transform, events[:, 0])

#     # --- SubsetEstimator ---
#     # Subsetter works for indexing
#     data = np.arange(20)[:, np.newaxis]
#     masker = SubsetEstimator(Ridge(), samples_train=data[:10, 0],
#                              samples_pred=data[10:, 0])
#     assert_array_equal(data[masker.samples_train], data[:10])
#     assert_array_equal(data[masker.samples_pred], data[10:])
#     # Subsetter indices must not exceed length of data
#     sub = SubsetEstimator(Ridge(), samples_train=[1, 99999999])
#     assert_raises(ValueError, sub.fit, data, data[:, 0])
#     # Create data
#     X = np.tile(np.arange(100.), [10, 1]).T
#     y = np.arange(100)
#     mod = SubsetEstimator(Ridge(), remove_value=np.nan)

#     # This should remove no datapoints, and test that fit_transform works
#     for func in ['fit', 'fit_transform']:
#         getattr(mod, func)(X, y)
#         assert_true(mod.mask_train_.dtype == bool)
#         assert_true(mod.mask_train_.sum() == X.shape[0])

#     # Test that it removes nans
#     X[:20, :] = np.nan
#     mod.fit(X, y)
#     assert_true(mod.mask_train_.sum() == (X.shape[0] - 20))
#     # Make sure the right indices were removed
#     assert_true(np.all(np.isnan(X[~mod.mask_train_])))

#     # Ensure that other numbers work
#     X = np.tile(np.arange(100.), [10, 1]).T
#     y = np.arange(100.)
#     mod = SubsetEstimator(Ridge(alpha=0), remove_value=10)
#     mod.fit(X, y)
#     assert_true(np.where(~mod.mask_train_)[0][0] == mod.remove_value)

#     # Check the prediction masking
#     y_pred = mod.predict(X)
#     assert_array_almost_equal(y_pred, y[mod.mask_predict_])
#     assert_true(mod.mask_predict_.dtype == bool)

#     # In this case, no values should be remove in prediction
#     mod = SubsetEstimator(Ridge(alpha=0), remove_value=10,
#                           remove_condition_pred='none')
#     mod.fit(X, y)
#     y_pred = mod.predict(X)
#     assert_array_almost_equal(y_pred, y)

#     # Check errors
#     assert_raises(ValueError, SubsetEstimator, Ridge(),
#                   remove_condition_pred='foo')

#     # Make sure a callable returns a 1-d output
#     def callable_mask(a):
#         return a < 5
#     mod = SubsetEstimator(Ridge(), remove_value=callable_mask)
#     assert_raises(ValueError, mod.fit, X, y)

#     # Make sure callable works
#     def callable_mask(a):
#         return (a < 5).all(1)
#     mod = SubsetEstimator(Ridge(), remove_value=callable_mask)
#     mod.fit(X, y)
#     assert_equal(mod.mask_train_.sum(), 95)

#     # Prediction samples
#     mod = SubsetEstimator(Ridge(alpha=0.), remove_value=np.nan,
#                           samples_pred=[1, 2, 3])
#     mod.fit(X, y)
#     y_pred = mod.predict(X)
#     assert_equal(y_pred.shape[0], 3)

#     # Transforming and scoring
#     X = np.random.randn(10)[:, np.newaxis]
#     y = X * 3.
#     mod.fit(X, y)
#     X_trans = mod.transform(X)
#     assert_array_equal(X_trans, X[mod.mask_train_])
#     assert_equal(mod.score(X, y), 1.)

#     # Misc
#     assert_raises(ValueError, SubsetEstimator, Ridge(), remove_value=0,
#                   samples_train=[1, 2])
#     assert_raises(ValueError, SubsetEstimator, Ridge(), remove_value=0,
#                   remove_condition='foo')

#     # Make sure that it's compatible with sklearn pipelines etc
#     # This won't work if you provide `ixs` as it messes up the CV
#     mod = SubsetEstimator(Ridge(alpha=0.), remove_value=1)
#     cross_val_score(mod, X, y)
#     cross_val_predict(mod, X, y)


# @requires_sklearn
# def test_encoding():
#     from sklearn.pipeline import make_pipeline
#     from sklearn.linear_model import Ridge
#     # Make sure estimator pulling works
#     mod = Ridge()
#     pipe = make_pipeline(Ridge())
#     samp = SubsetEstimator(pipe)
#     est = _get_final_est(pipe)
#     assert_true(isinstance(est, type(mod)))
#     est = _get_final_est(samp)
#     assert_true(isinstance(est, type(mod)))
#     # Est must be fit first
#     assert_raises(ValueError, get_coefs, est)
#     # Coefs are correctly taken
#     est.fit([[1, 2], [3, 4]], [1, 2])
#     coefs = get_coefs(samp, 'coef_')
#     assert_equal(coefs.shape[-1], 2)
#     # Incorrect coefficient name
#     assert_raises(ValueError, get_coefs, est, 'foo')

#     # Make sure the checks are working
#     mod = _check_regressor(Ridge())
#     assert_true(isinstance(mod, Ridge))
#     # None returns Ridge instance
#     assert_true(isinstance(_check_regressor(None), Ridge))
#     assert_true(isinstance(_get_final_est(_check_regressor(None)), Ridge))
#     # Correct ridge solver
#     assert_equal(_get_final_est(_check_regressor('lsqr')).solver, 'lsqr')
#     assert_true(isinstance(_get_final_est(_check_regressor('lsqr')), Ridge))
#     # Incorrect string type
#     assert_raises(ValueError, _check_regressor, 'foo')
#     # Estimator must have fit/predict methods
#     assert_raises(ValueError, _check_regressor, lambda a: a + 1)

# run_tests_if_main()
