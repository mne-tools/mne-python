# Authors: Robert Luke <mail@robertluke.net>
#          Eric Larson <larson.eric.d@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

import pytest
import numpy as np

from mne import create_info, EvokedArray
from mne.channels import make_standard_montage


@pytest.fixture()
def fnirs_evoked():
    """Create a fnirs evoked."""
    montage = make_standard_montage('biosemi16')
    ch_names = montage.ch_names
    ch_types = ['eeg'] * 16
    info = create_info(ch_names=ch_names, sfreq=20, ch_types=ch_types)
    evoked_data = np.random.randn(16, 30)
    evoked = EvokedArray(evoked_data, info=info, tmin=-0.2, nave=4)
    evoked.set_montage(montage)
    evoked.set_channel_types({'Fp1': 'hbo', 'Fp2': 'hbo', 'F4': 'hbo',
                             'Fz': 'hbo'}, verbose='error')
    return evoked
