# Authors:  Dirk Gütlin <dirk.guetlin@gmail.com>
#
# License: BSD (3-clause)
import os.path as op

import numpy as np
from scipy.io import loadmat

from mne.preprocessing.asr import ASR
from mne.io import read_raw_eeglab
from mne.datasets import testing
from mne.utils import run_tests_if_main

# set paths
data_path = op.join(testing.data_path(download=False), 'EEGLAB')
eeg_fname = op.join(data_path, 'test_raw.set')
valid_data_path = "./data/matlab_asr_data.mat"


def test_asr():
    """Test whether ASR correlates sufficiently with original version."""
    valid_data = loadmat(valid_data_path)["data"][0][0][0]
    raw = read_raw_eeglab(eeg_fname)

    # calculate clean data using ASR
    asr = ASR(sfreq=raw.info["sfreq"], cutoff=2.5, blocksize=10, win_len=0.5,
              win_overlap=0.66, max_dropout_fraction=0.1,
              min_clean_fraction=0.25, ab=None)
    asr.fit(raw.get_data())
    cleaned = asr.transform(raw.get_data(), lookahead=0.25, stepsize=32,
                            maxdims=0.66)

    # check if the data is highly equal to the MATLAB data
    corrs = [np.corrcoef(i, j)[0, 1] for (i, j) in zip(cleaned, valid_data)]
    assert np.mean(corrs) > 0.94


run_tests_if_main()
