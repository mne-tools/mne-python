# Authors: Robert Luke <mail@robertluke.net>
#          Eric Larson <larson.eric.d@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD-3-Clause

import pytest
import numpy as np
import os.path as op

from mne import create_info, EvokedArray, events_from_annotations, Epochs
from mne.channels import make_standard_montage
from mne.datasets.testing import data_path, _pytest_param
from mne.preprocessing.nirs import optical_density, beer_lambert_law
from mne.io import read_raw_nirx


@pytest.fixture()
def fnirs_evoked():
    """Create an fnirs evoked structure."""
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


@pytest.fixture(params=[_pytest_param()])
def fnirs_epochs():
    """Create an fnirs epoch structure."""
    fname = op.join(data_path(download=False),
                    'NIRx', 'nirscout', 'nirx_15_2_recording_w_overlap')
    raw_intensity = read_raw_nirx(fname, preload=False)
    raw_od = optical_density(raw_intensity)
    raw_haemo = beer_lambert_law(raw_od, ppf=6.)
    evts, _ = events_from_annotations(raw_haemo, event_id={'1.0': 1})
    evts_dct = {'A': 1}
    tn, tx = -1, 2
    epochs = Epochs(raw_haemo, evts, event_id=evts_dct, tmin=tn, tmax=tx)
    return epochs
