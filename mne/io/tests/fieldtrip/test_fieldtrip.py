# -*- coding: UTF-8 -*-
# Authors: Thomas Hartmann <thomas.hartmann@th-ht.de>
#          Dirk Gütlin <dirk.guetlin@stud.sbg.ac.at>
#
# License: BSD (3-clause)

import mne
import numpy as np
import os.path
import pytest
from mne.datasets import testing
from .helpers import (check_info_fields, get_data_paths, get_raw_data,
                      get_epoched_data, get_averaged_data, _has_h5py,
                      pandas_not_found_warning_msg)
from mne.utils import requires_h5py, _check_pandas_installed, requires_pandas


all_systems = ['neuromag306']


@testing.requires_testing_data
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_averaged():
    """Test comparing reading an Evoked object and the FieldTrip version."""
    for cur_system in all_systems:
        test_data_folder_ft = get_data_paths(cur_system)
        mne_avg = get_averaged_data(cur_system)

        all_versions = ['v7', 'v73']
        for version in all_versions:
            cur_fname = os.path.join(test_data_folder_ft,
                                     'averaged_%s.mat' % (version,))
            if version == 'v73' and not _has_h5py():
                with pytest.raises(ImportError):
                    mne.io.read_evoked_fieldtrip(cur_fname)
                continue

            avg_ft = mne.io.read_evoked_fieldtrip(cur_fname)
            avg_ft.pick_types(meg=True, eeg=True, ref_meg=True)

            mne_data = mne_avg.data[:, :-1]
            ft_data = avg_ft.data

            np.testing.assert_almost_equal(mne_data, ft_data)
            check_info_fields(mne_avg, avg_ft)


@testing.requires_testing_data
def test_epoched():
    """Test comparing reading an Epochs object and the FieldTrip version."""
    has_pandas = _check_pandas_installed(strict=False) is not False
    for cur_system in all_systems:
        test_data_folder_ft = get_data_paths(cur_system)
        mne_epoched = get_epoched_data(cur_system)

        all_versions = ['v7', 'v73']
        for version in all_versions:
            cur_fname = os.path.join(test_data_folder_ft,
                                     'epoched_%s.mat' % (version,))
            if has_pandas:
                pandas = _check_pandas_installed()
                if version == 'v73' and not _has_h5py():
                    with pytest.raises(ImportError):
                        mne.io.read_epochs_fieldtrip(cur_fname)
                    continue
                epoched_ft = mne.io.read_epochs_fieldtrip(cur_fname)
                assert isinstance(epoched_ft.metadata, pandas.DataFrame)
            else:
                with pytest.warns(RuntimeWarning,
                                  message=pandas_not_found_warning_msg):
                    if version == 'v73' and not _has_h5py():
                        with pytest.raises(ImportError):
                            mne.io.read_epochs_fieldtrip(cur_fname)
                        continue
                    epoched_ft = mne.io.read_epochs_fieldtrip(cur_fname)
                    assert epoched_ft.metadata is None

            mne_data = mne_epoched.get_data()[:, :, :-1]
            ft_data = epoched_ft.get_data()

            np.testing.assert_almost_equal(mne_data, ft_data)
            check_info_fields(mne_epoched, epoched_ft)


@testing.requires_testing_data
def test_raw():
    """Test comparing reading a raw fiff file and the FieldTrip version."""
    # Load the raw fiff file with mne
    for cur_system in all_systems:
        test_data_folder_ft = get_data_paths(cur_system)
        raw_fiff_mne = get_raw_data(cur_system)

        all_versions = ['v7', 'v73']

        for version in all_versions:
            cur_fname = os.path.join(test_data_folder_ft,
                                     'raw_%s.mat' % (version,))

            if version == 'v73' and not _has_h5py():
                with pytest.raises(ImportError):
                    mne.io.read_raw_fieldtrip(cur_fname)
                continue
            raw_fiff_ft = mne.io.read_raw_fieldtrip(cur_fname)

            # Check that the data was loaded correctly
            np.testing.assert_almost_equal(raw_fiff_mne.get_data(),
                                           raw_fiff_ft.get_data())

            # Check info field
            check_info_fields(raw_fiff_mne, raw_fiff_ft)


@testing.requires_testing_data
@requires_h5py
@requires_pandas
def test_whole_process_old():
    """Test the reader functions for FieldTrip data."""
    test_data_folder = os.path.join(mne.datasets.testing.data_path(),
                                    'fieldtrip/old')
    all_versions = ['v7', 'v73']
    for version in all_versions:
        f_name_raw = os.path.join(test_data_folder, 'raw_%s.mat' % (version,))
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
