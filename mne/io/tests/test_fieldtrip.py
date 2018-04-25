# -*- coding: UTF-8 -*-
# Authors: Thomas Hartmann <thomas.hartmann@th-ht.de>
#          Dirk Gütlin <dirk.guetlin@stud.sbg.ac.at>
#
# License: BSD (3-clause)

import mne
import numpy as np
import os.path
from mne.datasets import testing
from mne.utils import requires_h5py


@testing.requires_testing_data
@requires_h5py
def test_whole_process():
    """Test the reader functions for FieldTrip data."""
    test_data_folder = os.path.join(mne.datasets.testing.data_path(),
                                    'fieldtrip')
    all_versions = ['v7', 'v73']
    for version in all_versions:
        f_name_raw = os.path.join(test_data_folder, 'raw_%s.mat' % (version, ))
        f_name_epoched = os.path.join(test_data_folder,
                                      'epoched_%s.mat' % (version,))
        f_name_avg = os.path.join(test_data_folder,
                                  'averaged_%s.mat' % (version,))
        f_name_events = os.path.join(test_data_folder, 'events.eve')

        # load everything
        data_raw = mne.io.read_raw_fieldtrip(f_name_raw, data_name='data')
        data_epoched = mne.io.read_epochs_fieldtrip(f_name_epoched,
                                                    data_name='data_epoched')
        data_avg = mne.io.read_evoked_fieldtrip(f_name_avg,
                                                data_name='data_avg')
        events = mne.read_events(f_name_events)

        mne_epoched = mne.Epochs(data_raw, events, tmin=-0.05, tmax=0.05,
                                 preload=True, baseline=None)
        np.testing.assert_almost_equal(data_epoched.get_data(),
                                       mne_epoched.get_data()[:, :, :-1])

        mne_avg = mne_epoched.average(
            picks=np.arange(0, len(mne_epoched.ch_names)))
        np.testing.assert_almost_equal(data_avg.data, mne_avg.data[:, :-1])
