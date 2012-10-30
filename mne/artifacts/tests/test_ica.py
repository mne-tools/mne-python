# Author: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import os.path as op
from nose.tools import assert_true
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy import stats

from mne import fiff, Epochs, read_events, cov
from mne.artifacts import ICA, ica_find_ecg_events, ica_find_eog_events
from mne.artifacts.ica import score_funcs

have_sklearn = True
try:
    import sklearn
except ImportError:
    have_sklearn = False

sklearn_test = np.testing.dec.skipif(not have_sklearn,
                                     'scikit-learn not installed')

raw_fname = op.join(op.dirname(__file__), '..', '..', 'fiff', 'tests', 'data',
                    'test_raw.fif')
event_name = op.join(op.dirname(__file__), '..', '..', 'fiff', 'tests',
                     'data', 'test-eve.fif')
evoked_nf_name = op.join(op.dirname(__file__), '..', '..', 'fiff', 'tests',
                         'data', 'test-nf-ave.fif')

test_cov_name = op.join(op.dirname(__file__), '..', '..', 'fiff', 'tests',
                        'data', 'test-cov.fif')

event_id, tmin, tmax = 1, -0.2, 0.5
raw = fiff.Raw(raw_fname, preload=True)
events = read_events(event_name)
picks = fiff.pick_types(raw.info, meg=True, stim=False,
                        ecg=False, eog=False, exclude=raw.info['bads'])

picks2 = fiff.pick_types(raw.info, meg=True, stim=False,
                        ecg=False, eog=True, exclude=raw.info['bads'])

reject = dict(grad=1000e-12, mag=4e-12, eeg=80e-6, eog=150e-6)
flat = dict(grad=1e-15, mag=1e-15)

test_cov = cov.read_cov(test_cov_name)
epochs = Epochs(raw, events[:4], event_id, tmin, tmax, picks=picks2,
                baseline=(None, 0), preload=True)

start, stop = 0, 500


@sklearn_test
def test_ica():
    """Test ICA on raw and epochs
    """
    # Test ICA raw
    ica = ICA(noise_cov=None, n_components=25, random_state=0)
    ica_cov = ICA(noise_cov=test_cov, n_components=25, random_state=0)

    print ica  # to test repr
    ica.decompose_raw(raw, picks=None, start=0, stop=30)  # test default picks

    ica.decompose_raw(raw, picks=picks, start=start, stop=stop)
    sources = ica.get_sources_raw(raw)
    assert_true(sources.shape[0] == ica.n_components)

    raw2 = ica.pick_sources_raw(raw, exclude=[], copy=True)
    raw2 = ica.pick_sources_raw(raw, exclude=[1, 2], copy=True)
    raw2 = ica.pick_sources_raw(raw, include=[1, 2],
                                exclude=[], copy=True)
    assert_array_almost_equal(raw2[:, :][1], raw[:, :][1])

    ica_cov.decompose_raw(raw, picks=picks)
    print ica  # to test repr

    ica_cov.get_sources_raw(raw)
    assert_true(sources.shape[0] == ica.n_components)

    raw2 = ica_cov.pick_sources_raw(raw, exclude=[], copy=True)
    raw2 = ica_cov.pick_sources_raw(raw, exclude=[1, 2], copy=True)
    raw2 = ica_cov.pick_sources_raw(raw, include=[1, 2],
                                    exclude=[], copy=True)
    assert_array_almost_equal(raw2[:, :][1], raw[:, :][1])

    # Test epochs sources selection using raw fit.
    epochs2 = ica.pick_sources_epochs(epochs, exclude=[], copy=True)
    assert_array_almost_equal(epochs2.get_data(), epochs.get_data())

    # Test score_funcs and find_sources
    sfunc_test = [ica.find_sources_raw(raw, target='EOG 061', score_func=n)
                  for  n, f in score_funcs.items()]

    [assert_true(ica.n_components == len(scores)) for scores in sfunc_test]

    scores = ica.find_sources_raw(raw, score_func=stats.skew)
    assert_true(scores.shape == ica.index.shape)

    ecg_scores = ica.find_sources_raw(raw, target='MEG 1531',
                                      score_func='pearsonr')

    ecg_events = ica_find_ecg_events(raw, sources[np.abs(ecg_scores).argmax()])

    assert_true(ecg_events.ndim == 2)

    eog_scores = ica.find_sources_raw(raw, target='EOG 061',
                                      score_func='pearsonr')

    eog_events = ica_find_eog_events(raw, sources[np.abs(eog_scores).argmax()])

    assert_true(eog_events.ndim == 2)

    # Test ICA epochs

    ica.decompose_epochs(epochs, picks=picks2)

    sources = ica.get_sources_epochs(epochs)
    assert_true(sources.shape[1] == ica.n_components)

    epochs2 = ica.pick_sources_epochs(epochs, exclude=[], copy=True)
    epochs2 = ica.pick_sources_epochs(epochs, exclude=[0], copy=True)
    epochs2 = ica.pick_sources_epochs(epochs, include=[0],
                                      exclude=[], copy=True)
    assert_array_almost_equal(epochs2.get_data(),
                              epochs.get_data())

    ica_cov.decompose_epochs(epochs, picks=picks2)

    sources = ica_cov.get_sources_epochs(epochs)
    assert_true(sources.shape[1] == ica.n_components)

    epochs2 = ica_cov.pick_sources_epochs(epochs, exclude=[], copy=True)
    epochs2 = ica_cov.pick_sources_epochs(epochs, exclude=[0], copy=True)
    epochs2 = ica_cov.pick_sources_epochs(epochs, include=[0],
                                          exclude=[], copy=True)
    assert_array_almost_equal(epochs2._data, epochs._data)

    scores = ica.find_sources_epochs(epochs, score_func=stats.skew)
    assert_true(scores.shape == ica.index.shape)

    sfunc_test = [ica.find_sources_epochs(epochs, target='EOG 061',
                    score_func=n) for n, f in score_funcs.items()]

    [assert_true(ica.n_components == len(scores)) for scores in sfunc_test]

