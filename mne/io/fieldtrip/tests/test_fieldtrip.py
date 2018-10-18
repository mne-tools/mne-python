# -*- coding: UTF-8 -*-
# Authors: Thomas Hartmann <thomas.hartmann@th-ht.de>
#          Dirk Gütlin <dirk.guetlin@stud.sbg.ac.at>
#
# License: BSD (3-clause)

import mne
import os.path
import pytest
import itertools
from mne.datasets import testing
from .helpers import (check_info_fields, get_data_paths, get_raw_data,
                      get_epoched_data, get_averaged_data, _has_h5py,
                      pandas_not_found_warning_msg, get_raw_info, check_data)
from mne.utils import _check_pandas_installed

# missing: KIT: biggest problem here is that the channels do not have the same
# names.
# EGI: no calibration done in FT. so data is VERY different
# CNT epochs and averages are not tested because small differences in
# the raw data create big ones

all_systems_raw = ['neuromag306', 'CTF', 'CNT', 'BTI', 'eximia']
all_systems_epochs = ['neuromag306', 'CTF', 'CNT']
all_versions = ['v7', 'v73']
use_info = [True]
all_test_params_raw = list(itertools.product(all_systems_raw, all_versions,
                                             use_info))
all_test_params_epochs = list(itertools.product(all_systems_epochs,
                                                all_versions,
                                                use_info))


@testing.requires_testing_data
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('cur_system, version, use_info',
                         all_test_params_epochs)
def test_averaged(cur_system, version, use_info):
    """Test comparing reading an Evoked object and the FieldTrip version."""
    test_data_folder_ft = get_data_paths(cur_system)
    mne_avg = get_averaged_data(cur_system)
    if use_info:
        info = get_raw_info(cur_system)
    else:
        info = None

    cur_fname = os.path.join(test_data_folder_ft,
                             'averaged_%s.mat' % (version,))
    if version == 'v73' and not _has_h5py():
        with pytest.raises(ImportError):
            mne.io.read_evoked_fieldtrip(cur_fname, info)
        return

    avg_ft = mne.io.read_evoked_fieldtrip(cur_fname, info)

    mne_data = mne_avg.data[:, :-1]
    ft_data = avg_ft.data

    check_data(mne_data, ft_data, cur_system)
    check_info_fields(mne_avg, avg_ft, use_info)


@testing.requires_testing_data
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('cur_system, version, use_info',
                         all_test_params_epochs)
def test_epoched(cur_system, version, use_info):
    """Test comparing reading an Epochs object and the FieldTrip version."""
    has_pandas = _check_pandas_installed(strict=False) is not False
    test_data_folder_ft = get_data_paths(cur_system)
    mne_epoched = get_epoched_data(cur_system)
    if use_info:
        info = get_raw_info(cur_system)
    else:
        info = None

    cur_fname = os.path.join(test_data_folder_ft,
                             'epoched_%s.mat' % (version,))
    if has_pandas:
        pandas = _check_pandas_installed()
        if version == 'v73' and not _has_h5py():
            with pytest.raises(ImportError):
                mne.io.read_epochs_fieldtrip(cur_fname, info)
            return
        epoched_ft = mne.io.read_epochs_fieldtrip(cur_fname, info)
        assert isinstance(epoched_ft.metadata, pandas.DataFrame)
    else:
        with pytest.warns(RuntimeWarning,
                          message=pandas_not_found_warning_msg):
            if version == 'v73' and not _has_h5py():
                with pytest.raises(ImportError):
                    mne.io.read_epochs_fieldtrip(cur_fname, info)
                return
            epoched_ft = mne.io.read_epochs_fieldtrip(cur_fname, info)
            assert epoched_ft.metadata is None

    mne_data = mne_epoched.get_data()[:, :, :-1]
    ft_data = epoched_ft.get_data()

    check_data(mne_data, ft_data, cur_system)
    check_info_fields(mne_epoched, epoched_ft, use_info)


@testing.requires_testing_data
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('cur_system, version, use_info', all_test_params_raw)
def test_raw(cur_system, version, use_info):
    """Test comparing reading a raw fiff file and the FieldTrip version."""
    # Load the raw fiff file with mne
    test_data_folder_ft = get_data_paths(cur_system)
    raw_fiff_mne = get_raw_data(cur_system, drop_extra_chs=True)
    if use_info:
        info = get_raw_info(cur_system)
    else:
        info = None

    cur_fname = os.path.join(test_data_folder_ft,
                             'raw_%s.mat' % (version,))

    if version == 'v73' and not _has_h5py():
        with pytest.raises(ImportError):
            mne.io.read_raw_fieldtrip(cur_fname, info)
        return
    raw_fiff_ft = mne.io.read_raw_fieldtrip(cur_fname, info)

    # Check that the data was loaded correctly
    check_data(raw_fiff_mne.get_data(),
               raw_fiff_ft.get_data(),
               cur_system)

    # Check info field
    check_info_fields(raw_fiff_mne, raw_fiff_ft, use_info)
