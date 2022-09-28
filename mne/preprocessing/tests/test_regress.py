# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause


import os.path as op

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mne.datasets import testing
from mne.io import read_raw_fif
from mne.preprocessing import (regress_artifact, create_eog_epochs,
                               EOGRegression, read_eog_regression)
from mne.utils import requires_version

data_path = testing.data_path(download=False)
raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_raw.fif')


@testing.requires_testing_data
def test_regress_artifact():
    """Test regressing artifact data."""
    raw = read_raw_fif(raw_fname).pick_types(meg=False, eeg=True, eog=True)
    raw.load_data()
    epochs = create_eog_epochs(raw)
    epochs.apply_baseline((None, None))
    orig_data = epochs.get_data('eeg')
    orig_norm = np.linalg.norm(orig_data)
    epochs_clean, betas = regress_artifact(epochs)
    regress_artifact(epochs, betas=betas, copy=False)  # inplace, and w/betas
    assert_allclose(epochs_clean.get_data(), epochs.get_data())
    clean_data = epochs_clean.get_data('eeg')
    clean_norm = np.linalg.norm(clean_data)
    assert orig_norm / 2 > clean_norm > orig_norm / 10
    with pytest.raises(ValueError, match=r'Invalid value.*betas\.shape.*'):
        regress_artifact(epochs, betas=betas[:-1])
    # Regressing channels onto themselves should work
    epochs, betas = regress_artifact(epochs, picks='eog', picks_artifact='eog')
    assert np.ptp(epochs.get_data('eog')) < 1E-15  # constant value
    assert_allclose(betas, 1)


@testing.requires_testing_data
def test_eog_regression():
    """Test regressing artifact data using the EOGRegression class."""
    raw_meg_eeg = read_raw_fif(raw_fname)
    raw = raw_meg_eeg.copy().pick(['eeg', 'eog', 'stim'])

    # Test various errors
    with pytest.raises(RuntimeError, match='Projections need to be applied'):
        model = EOGRegression(proj=False).fit(raw)
    with pytest.raises(RuntimeError, match='requires raw data to be loaded'):
        model = EOGRegression().fit(raw)
    raw.load_data()

    # Test regression on raw data
    model = EOGRegression()
    assert str(model) == '<EOGRegression | not fitted>'
    model.fit(raw)
    assert str(model) == '<EOGRegression | fitted to 1 artifact channel>'
    assert model.coef_.shape == (59, 1)  # 59 EEG channels, 1 EOG channel
    raw_clean = model.apply(raw)
    # Some signal must have been removed
    assert np.ptp(raw_clean.get_data('eeg')) < np.ptp(raw.get_data('eeg'))

    # Test regression on epochs
    epochs = create_eog_epochs(raw)
    model = EOGRegression().fit(epochs)
    epochs = model.apply(epochs)
    # Since these were blinks, they should be mostly gone
    assert np.ptp(epochs.get_data('eeg')) < 1E-4

    # Test regression on evoked
    evoked = epochs.average('all')
    model = EOGRegression().fit(evoked)
    evoked = model.apply(evoked)
    assert model.coef_.shape == (59, 1)
    # Since this was a blink evoked, signal should be mostly gone
    assert np.ptp(evoked.get_data('eeg')) < 1E-4

    # Test regression on evoked and applying to raw, with different ordering of
    # channels. This should not work.
    raw_ = raw.copy().drop_channels(['EEG 001'])
    raw_ = raw_.add_channels([raw.copy().pick(['EEG 001'])])
    model = EOGRegression().fit(evoked)
    with pytest.raises(ValueError, match='data channels are not compatible'):
        model.apply(raw_)

    # Test in-place operation
    raw_ = model.apply(raw, copy=False)
    assert raw_ is raw
    assert raw_._data is raw._data
    raw_ = model.apply(raw, copy=True)
    assert raw_ is not raw
    assert raw_._data is not raw._data

    # Test plotting with one channel type
    fig = model.plot()
    assert len(fig.axes) == 2  # (one topomap and one colorbar)
    assert fig.axes[0].title.get_text() == 'eeg/EOG 061'

    # Test plotting with multiple channel types
    raw_meg_eeg.load_data()
    fig = EOGRegression().fit(raw_meg_eeg).plot()
    assert len(fig.axes) == 6  # (3 topomaps and 3 colorbars)
    assert fig.axes[0].title.get_text() == 'grad/EOG 061'
    assert fig.axes[1].title.get_text() == 'mag/EOG 061'
    assert fig.axes[2].title.get_text() == 'eeg/EOG 061'

    # Test plotting with multiple channel types, multiple regressors)
    m = EOGRegression(picks_artifact=['EEG 001', 'EOG 061']).fit(raw_meg_eeg)
    assert str(m) == '<EOGRegression | fitted to 2 artifact channels>'
    fig = m.plot()
    assert len(fig.axes) == 12  # (6 topomaps and 3 colorbars)
    assert fig.axes[0].title.get_text() == 'grad/EEG 001'
    assert fig.axes[1].title.get_text() == 'mag/EEG 001'
    assert fig.axes[4].title.get_text() == 'mag/EOG 061'
    assert fig.axes[5].title.get_text() == 'eeg/EOG 061'


@requires_version('h5io')
@testing.requires_testing_data
def test_read_eog_regression(tmp_path):
    """Test saving and loading an EOGRegression object."""
    raw = read_raw_fif(raw_fname).pick(['eeg', 'eog'])
    raw.load_data()
    model = EOGRegression().fit(raw)

    model.save(tmp_path / 'weights.h5', overwrite=True)
    model2 = read_eog_regression(tmp_path / 'weights.h5')
    assert_array_equal(model.picks, model2.picks)
    assert_array_equal(model.picks_artifact, model2.picks_artifact)
    assert_array_equal(model.exclude, model2.exclude)
    assert_array_equal(model.coef_, model2.coef_)
    assert model.proj == model2.proj
    assert model.info_.keys() == model2.info_.keys()
