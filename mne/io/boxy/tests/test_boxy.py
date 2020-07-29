# Authors: Kyle Mathewson, Jonathan Kuziek <kuziekj@ualberta.ca>
#
# License: BSD (3-clause)

import os

import numpy as np
import scipy.io as spio

import mne
from mne.datasets.testing import data_path, requires_testing_data


@requires_testing_data
def test_boxy_load():
    """Test reading BOXY files."""
    # Determine to which decimal place we will compare.
    thresh = 1e-10

    # Load AC, DC, and Phase data.
    boxy_raw_dir = os.path.join(data_path(download=False),
                                'BOXY', 'boxy_short_recording')

    mne_dc = mne.io.read_raw_boxy(boxy_raw_dir, 'DC', verbose=True).load_data()
    mne_ac = mne.io.read_raw_boxy(boxy_raw_dir, 'AC', verbose=True).load_data()
    mne_ph = mne.io.read_raw_boxy(boxy_raw_dir, 'Ph', verbose=True).load_data()

    # Load p_pod data.
    p_pod_dir = os.path.join(data_path(download=False),
                             'BOXY', 'boxy_short_recording',
                             'boxy_p_pod_files', '1anc071a_001.mat')
    ppod_data = spio.loadmat(p_pod_dir)

    ppod_ac = np.transpose(ppod_data['ac'])
    ppod_dc = np.transpose(ppod_data['dc'])
    ppod_ph = np.transpose(ppod_data['ph'])

    # Compare MNE loaded data to p_pod loaded data.
    assert (abs(ppod_ac - mne_ac._data) <= thresh).all()
    assert (abs(ppod_dc - mne_dc._data) <= thresh).all()
    assert (abs(ppod_ph - mne_ph._data) <= thresh).all()
