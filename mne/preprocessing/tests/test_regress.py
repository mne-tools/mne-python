# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause


import os.path as op

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne.datasets import testing
from mne.io import read_raw_fif
from mne.preprocessing import regress_artifact, create_eog_epochs

data_path = testing.data_path(download=False)
raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_raw.fif')


@testing.requires_testing_data
def test_regress_artifact():
    """Test regressing data."""
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
    with pytest.raises(ValueError, match='cannot be contained in'):
        regress_artifact(epochs, picks='eog', picks_artifact='eog')
