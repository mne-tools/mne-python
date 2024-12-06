# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal, assert_array_less

from mne import pick_types
from mne.datasets import testing
from mne.io import read_raw_fif
from mne.preprocessing import (
    EOGRegression,
    create_eog_epochs,
    read_eog_regression,
    regress_artifact,
)

data_path = testing.data_path(download=False)
raw_fname = data_path / "MEG" / "sample" / "sample_audvis_trunc_raw.fif"


@testing.requires_testing_data
def test_regress_artifact():
    """Test regressing artifact data."""
    raw = read_raw_fif(raw_fname).pick(["eeg", "eog"])
    raw.load_data()
    epochs = create_eog_epochs(raw)
    epochs.apply_baseline((None, None))
    orig_data = epochs.get_data("eeg")
    orig_norm = np.linalg.norm(orig_data)
    epochs_clean, betas = regress_artifact(epochs)
    regress_artifact(epochs, betas=betas, copy=False)  # inplace, and w/betas
    assert_allclose(epochs_clean.get_data(copy=False), epochs.get_data(copy=False))
    clean_data = epochs_clean.get_data("eeg")
    clean_norm = np.linalg.norm(clean_data)
    assert orig_norm / 2 > clean_norm > orig_norm / 10
    with pytest.raises(ValueError, match=r"Invalid value.*betas\.shape.*"):
        regress_artifact(epochs, betas=betas[:-1])
    # Regressing channels onto themselves should work
    epochs, betas = regress_artifact(epochs, picks="eog", picks_artifact="eog")
    assert np.ptp(epochs.get_data("eog")) < 1e-15  # constant value
    assert_allclose(betas, 1)
    # proj should only be required of channels being processed
    raw = read_raw_fif(raw_fname).crop(0, 1).load_data()
    raw.del_proj()
    raw.set_eeg_reference(projection=True)
    model = EOGRegression(proj=False, picks="meg", picks_artifact="eog")
    model.fit(raw)
    model.apply(raw)
    model = EOGRegression(proj=False, picks="eeg", picks_artifact="eog")
    with pytest.raises(RuntimeError, match="Projections need to be applied"):
        model.fit(raw)
    raw.del_proj()
    with pytest.raises(RuntimeError, match="No average reference for the EEG"):
        model.fit(raw)


@testing.requires_testing_data
def test_eog_regression():
    """Test regressing artifact data using the EOGRegression class."""
    raw_meg_eeg = read_raw_fif(raw_fname)
    raw = raw_meg_eeg.copy().pick(["eeg", "eog", "stim"])

    # Test various errors
    with pytest.raises(RuntimeError, match="Projections need to be applied"):
        model = EOGRegression(proj=False).fit(raw)
    with pytest.raises(RuntimeError, match="requires raw data to be loaded"):
        model = EOGRegression().fit(raw)
    raw.load_data()

    # Test regression on raw data
    model = EOGRegression()
    assert str(model) == "<EOGRegression | not fitted>"
    model.fit(raw)
    assert str(model) == "<EOGRegression | fitted to 1 artifact channel>"
    assert model.coef_.shape == (59, 1)  # 59 EEG channels, 1 EOG channel
    raw_clean = model.apply(raw)
    # Some signal must have been removed
    assert np.ptp(raw_clean.get_data("eeg")) < np.ptp(raw.get_data("eeg"))

    # Test regression on epochs
    epochs = create_eog_epochs(raw)
    model = EOGRegression().fit(epochs)
    epochs = model.apply(epochs)
    # Since these were blinks, they should be mostly gone
    assert np.ptp(epochs.get_data("eeg")) < 1e-4

    # Test regression on evoked
    evoked = epochs.average("all")
    model = EOGRegression().fit(evoked)
    evoked = model.apply(evoked)
    assert model.coef_.shape == (59, 1)
    # Since this was a blink evoked, signal should be mostly gone
    assert np.ptp(evoked.get_data("eeg")) < 1e-4

    # Test regression on evoked and applying to raw, with different ordering of
    # channels. This should not work.
    raw_ = raw.copy().drop_channels(["EEG 001"])
    raw_ = raw_.add_channels([raw.copy().pick(["EEG 001"])])
    model = EOGRegression().fit(evoked)
    with pytest.raises(ValueError, match="data channels are not compatible"):
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
    assert fig.axes[0].title.get_text() == "eeg/EOG 061"

    # Test plotting with multiple channel types
    raw_meg_eeg.load_data()
    fig = EOGRegression().fit(raw_meg_eeg).plot()
    assert len(fig.axes) == 6  # (3 topomaps and 3 colorbars)
    assert fig.axes[0].title.get_text() == "grad/EOG 061"
    assert fig.axes[1].title.get_text() == "mag/EOG 061"
    assert fig.axes[2].title.get_text() == "eeg/EOG 061"

    # Test plotting with multiple channel types, multiple regressors)
    m = EOGRegression(picks_artifact=["EEG 001", "EOG 061"]).fit(raw_meg_eeg)
    assert str(m) == "<EOGRegression | fitted to 2 artifact channels>"
    fig = m.plot()
    assert len(fig.axes) == 12  # (6 topomaps and 3 colorbars)
    assert fig.axes[0].title.get_text() == "grad/EEG 001"
    assert fig.axes[1].title.get_text() == "mag/EEG 001"
    assert fig.axes[4].title.get_text() == "mag/EOG 061"
    assert fig.axes[5].title.get_text() == "eeg/EOG 061"


@testing.requires_testing_data
def test_read_eog_regression(tmp_path):
    """Test saving and loading an EOGRegression object."""
    pytest.importorskip("h5io")
    raw = read_raw_fif(raw_fname).pick(["eeg", "eog"])
    raw.load_data()
    model = EOGRegression().fit(raw)

    model.save(tmp_path / "weights.h5", overwrite=True)
    model2 = read_eog_regression(tmp_path / "weights.h5")
    assert_array_equal(model.picks, model2.picks)
    assert_array_equal(model.picks_artifact, model2.picks_artifact)
    assert_array_equal(model.exclude, model2.exclude)
    assert_array_equal(model.coef_, model2.coef_)
    assert model.proj == model2.proj
    assert model.info_.keys() == model2.info_.keys()


@testing.requires_testing_data
def test_regress_artifact_bads():
    """Test that bad channels are handled properly."""
    # Pick the first few EEG channels
    raw = read_raw_fif(raw_fname).del_proj().set_eeg_reference(projection=True)
    picks_all = np.concatenate(
        [
            pick_types(raw.info, meg=True)[:4],
            pick_types(raw.info, eeg=True)[:8],
            pick_types(raw.info, eog=True)[:1],
        ]
    )
    raw.pick(picks_all).load_data()
    assert len(raw.ch_names) == 13  # 4 MEG, 8 EEG, 1 EOG
    del picks_all
    picks = pick_types(raw.info, eeg=True)
    assert_array_equal(picks, np.arange(4, 12))
    norms = np.linalg.norm(raw.get_data(picks), axis=1)
    raw_reg, _ = regress_artifact(raw, picks=picks, picks_artifact="eog")
    assert_allclose(raw_reg.get_data("meg"), raw.get_data("meg"))  # unchanged
    data_reg = raw_reg.get_data()
    norms_reg = np.linalg.norm(data_reg[picks], axis=1)
    suppression = 20 * np.log10(norms / norms_reg)
    assert_array_less(3, suppression)  # at least 3 dB suppression
    # Adding some bad channels shouldn't affect anything when we supply picks
    raw.info["bads"] = raw.ch_names[:2] + raw.ch_names[-2:-1]
    raw_reg, _ = regress_artifact(raw, picks=picks, picks_artifact="eog")
    data_reg_new = raw_reg.get_data()
    assert_allclose(data_reg, data_reg_new)
