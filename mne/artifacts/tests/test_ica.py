# Author: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import os.path as op
from nose.tools import assert_true
from numpy.testing import assert_array_almost_equal

from mne import fiff, Epochs, read_events, cov
from mne.artifacts import ICA

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
reject = dict(grad=1000e-12, mag=4e-12, eeg=80e-6, eog=150e-6)
flat = dict(grad=1e-15, mag=1e-15)

test_cov = cov.read_cov(test_cov_name)
epochs = Epochs(raw, events[:4], event_id, tmin, tmax, picks=picks,
                baseline=(None, 0), preload=True)

ica = ICA(noise_cov=None, n_components=25, random_state=0)
ica_cov = ICA(noise_cov=test_cov, n_components=25, random_state=0)

start, stop = 0, 9009


def test_ica_raw():
    """Test ICA on raw"""
    print ica  # to test repr
    ica.decompose_raw(raw, picks=picks)
    ica_cov.decompose_raw(raw, picks=picks)
    print ica  # to test repr

    sources = ica.get_sources_raw(raw, picks=picks)
    assert_true(sources.shape[0] == ica.n_components)

    ica_cov.get_sources_raw(raw, picks=picks)
    assert_true(sources.shape[0] == ica.n_components)

    raw2 = ica.pick_sources_raw(raw, exclude=[], sort_method='kurtosis',
                                copy=True)
    assert_array_almost_equal(raw2._data, raw._data)


def test_pick_sources_epochs_from_raw():
    """Test epochs sources selection using raw fit."""
    epochs2 = ica.pick_sources_epochs(epochs, exclude=[],
                                      sort_method='kurtosis', copy=True)
    assert_array_almost_equal(epochs2._data, epochs._data)


def test_ica_epochs():
    """Test ICA epochs"""
    ica.decompose_epochs(epochs, picks=picks)
    ica_cov.decompose_epochs(epochs, picks=picks)

    sources = ica.get_sources_epochs(epochs, picks=picks)
    assert_true(sources.shape[1] == ica.n_components)

    sources = ica_cov.get_sources_epochs(epochs, picks=picks)
    assert_true(sources.shape[1] == ica.n_components)

    epochs2 = ica.pick_sources_epochs(epochs, exclude=[],
                                      sort_method='kurtosis', copy=True)
    assert_array_almost_equal(epochs2._data, epochs._data)
