# -*- coding: UTF-8 -*-
# Authors: Thomas Hartmann <thomas.hartmann@th-ht.de>
#          Dirk Gütlin <dirk.guetlin@stud.sbg.ac.at>
#
# License: BSD-3-Clause

from contextlib import nullcontext
import copy
import itertools
import os.path

import pytest
import numpy as np

import mne
from mne.datasets import testing
from mne.io import read_raw_fieldtrip
from mne.io.fieldtrip.utils import NOINFO_WARNING, _create_events
from mne.io.fieldtrip.tests.helpers import (
    check_info_fields, get_data_paths, get_raw_data, get_epochs, get_evoked,
    pandas_not_found_warning_msg, get_raw_info, check_data,
    assert_warning_in_record)
from mne.io.tests.test_raw import _test_raw_reader
from mne.utils import _check_pandas_installed, _record_warnings

# missing: KIT: biggest problem here is that the channels do not have the same
# names.
# EGI: no calibration done in FT. so data is VERY different

all_systems_raw = ['neuromag306', 'CTF', 'CNT', 'BTI', 'eximia']
all_systems_epochs = ['neuromag306', 'CTF', 'CNT']
all_versions = ['v7', 'v73']
use_info = [True, False]
all_test_params_raw = list(itertools.product(all_systems_raw, all_versions,
                                             use_info))
all_test_params_epochs = list(itertools.product(all_systems_epochs,
                                                all_versions,
                                                use_info))
# just for speed we skip some slowest ones -- the coverage should still
# be sufficient
for obj in (all_test_params_epochs, all_test_params_raw):
    for key in [('CTF', 'v73', True), ('neuromag306', 'v73', False)]:
        obj.pop(obj.index(key))
    for ki, key in enumerate(obj):
        if key[1] == 'v73':
            obj[ki] = pytest.param(*obj[ki], marks=pytest.mark.slowtest)

no_info_warning = {'expected_warning': RuntimeWarning,
                   'match': NOINFO_WARNING}

pymatreader = pytest.importorskip('pymatreader')  # module-level
testing_path = mne.datasets.testing.data_path(download=False)


@pytest.mark.slowtest
@testing.requires_testing_data
# Reading the sample CNT data results in a RuntimeWarning because it cannot
# parse the measurement date. We need to ignore that warning.
@pytest.mark.filterwarnings('ignore:.*parse meas date.*:RuntimeWarning')
@pytest.mark.filterwarnings('ignore:.*number of bytes.*:RuntimeWarning')
@pytest.mark.parametrize('cur_system, version, use_info',
                         all_test_params_epochs)
def test_read_evoked(cur_system, version, use_info):
    """Test comparing reading an Evoked object and the FieldTrip version."""
    test_data_folder_ft = get_data_paths(cur_system)
    mne_avg = get_evoked(cur_system)
    if use_info:
        info = get_raw_info(cur_system)
        ctx = nullcontext()
    else:
        info = None
        ctx = pytest.warns(**no_info_warning)

    cur_fname = os.path.join(test_data_folder_ft,
                             'averaged_%s.mat' % (version,))
    with ctx:
        avg_ft = mne.io.read_evoked_fieldtrip(cur_fname, info)

    mne_data = mne_avg.data[:, :-1]
    ft_data = avg_ft.data

    check_data(mne_data, ft_data, cur_system)
    check_info_fields(mne_avg, avg_ft, use_info)


@testing.requires_testing_data
# Reading the sample CNT data results in a RuntimeWarning because it cannot
# parse the measurement date. We need to ignore that warning.
@pytest.mark.filterwarnings('ignore:.*parse meas date.*:RuntimeWarning')
@pytest.mark.filterwarnings('ignore:.*number of bytes.*:RuntimeWarning')
@pytest.mark.parametrize('cur_system, version, use_info',
                         all_test_params_epochs)
# Strange, non-deterministic Pandas errors:
# "ValueError: cannot expose native-only dtype 'g' in non-native
# byte order '<' via buffer interface"
@pytest.mark.skipif(os.getenv('AZURE_CI_WINDOWS', 'false').lower() == 'true',
                    reason='Pandas problem on Azure CI')
def test_read_epochs(cur_system, version, use_info, monkeypatch):
    """Test comparing reading an Epochs object and the FieldTrip version."""
    pandas = _check_pandas_installed(strict=False)
    has_pandas = pandas is not False
    test_data_folder_ft = get_data_paths(cur_system)
    mne_epoched = get_epochs(cur_system)
    if use_info:
        info = get_raw_info(cur_system)
        ctx = nullcontext()
    else:
        info = None
        ctx = pytest.warns(**no_info_warning)

    cur_fname = os.path.join(test_data_folder_ft,
                             'epoched_%s.mat' % (version,))
    if has_pandas:
        with ctx:
            epoched_ft = mne.io.read_epochs_fieldtrip(cur_fname, info)
        assert isinstance(epoched_ft.metadata, pandas.DataFrame)
    else:
        with _record_warnings() as warn_record:
            epoched_ft = mne.io.read_epochs_fieldtrip(cur_fname, info)
            assert epoched_ft.metadata is None
            assert_warning_in_record(pandas_not_found_warning_msg, warn_record)
            if info is None:
                assert_warning_in_record(NOINFO_WARNING, warn_record)

    mne_data = mne_epoched.get_data()[:, :, :-1]
    ft_data = epoched_ft.get_data()

    check_data(mne_data, ft_data, cur_system)
    check_info_fields(mne_epoched, epoched_ft, use_info)
    read_mat = pymatreader.read_mat

    # weird sfreq
    def modify_mat(fname, variable_names=None, ignore_fields=None):
        out = read_mat(fname, variable_names, ignore_fields)
        if 'fsample' in out['data']:
            out['data']['fsample'] = np.repeat(out['data']['fsample'], 2)
        return out

    monkeypatch.setattr(pymatreader, 'read_mat', modify_mat)
    with pytest.warns(RuntimeWarning, match='multiple'):
        mne.io.read_epochs_fieldtrip(cur_fname, info)


@testing.requires_testing_data
# Reading the sample CNT data results in a RuntimeWarning because it cannot
# parse the measurement date. We need to ignore that warning.
@pytest.mark.filterwarnings('ignore:.*parse meas date.*:RuntimeWarning')
@pytest.mark.filterwarnings('ignore:.*number of bytes.*:RuntimeWarning')
@pytest.mark.parametrize('cur_system, version, use_info', all_test_params_raw)
def test_read_raw_fieldtrip(cur_system, version, use_info):
    """Test comparing reading a raw fiff file and the FieldTrip version."""
    # Load the raw fiff file with mne
    test_data_folder_ft = get_data_paths(cur_system)
    raw_fiff_mne = get_raw_data(cur_system, drop_extra_chs=True)
    if use_info:
        info = get_raw_info(cur_system)
        if cur_system in ('BTI', 'eximia'):
            ctx = pytest.warns(RuntimeWarning, match='cannot be found in')
        else:
            ctx = nullcontext()
    else:
        info = None
        ctx = pytest.warns(**no_info_warning)

    cur_fname = os.path.join(test_data_folder_ft,
                             'raw_%s.mat' % (version,))

    with ctx:
        raw_fiff_ft = mne.io.read_raw_fieldtrip(cur_fname, info)

    if cur_system == 'BTI' and not use_info:
        raw_fiff_ft.drop_channels(['MzA', 'MxA', 'MyaA',
                                   'MyA', 'MxaA', 'MzaA'])

    if cur_system == 'eximia' and not use_info:
        raw_fiff_ft.drop_channels(['TRIG2', 'TRIG1', 'GATE'])

    # Check that the data was loaded correctly
    check_data(raw_fiff_mne.get_data(),
               raw_fiff_ft.get_data(),
               cur_system)

    # standard tests
    with _record_warnings():
        _test_raw_reader(
            read_raw_fieldtrip, fname=cur_fname, info=info,
            test_preloading=False,
            test_kwargs=False)  # TODO: This should probably work

    # Check info field
    check_info_fields(raw_fiff_mne, raw_fiff_ft, use_info)


@testing.requires_testing_data
def test_load_epoched_as_raw():
    """Test whether exception is thrown when loading epochs as raw."""
    test_data_folder_ft = get_data_paths('neuromag306')
    info = get_raw_info('neuromag306')
    cur_fname = os.path.join(test_data_folder_ft, 'epoched_v7.mat')

    with pytest.raises(RuntimeError):
        mne.io.read_raw_fieldtrip(cur_fname, info)


@testing.requires_testing_data
def test_invalid_trialinfocolumn():
    """Test for exceptions when using wrong values for trialinfo parameter."""
    test_data_folder_ft = get_data_paths('neuromag306')
    info = get_raw_info('neuromag306')
    cur_fname = os.path.join(test_data_folder_ft, 'epoched_v7.mat')

    with pytest.raises(ValueError):
        mne.io.read_epochs_fieldtrip(cur_fname, info, trialinfo_column=-1)

    with pytest.raises(ValueError):
        mne.io.read_epochs_fieldtrip(cur_fname, info, trialinfo_column=3)


@testing.requires_testing_data
def test_create_events():
    """Test 2dim trialinfo fields."""
    test_data_folder_ft = get_data_paths('neuromag306')
    cur_fname = os.path.join(test_data_folder_ft, 'epoched_v7.mat')
    original_data = pymatreader.read_mat(cur_fname, ['data', ])

    new_data = copy.deepcopy(original_data)
    new_data['trialinfo'] = np.array([[1, 2, 3, 4],
                                      [1, 2, 3, 4],
                                      [1, 2, 3, 4]])

    with pytest.raises(ValueError):
        _create_events(new_data, -1)

    for cur_col in np.arange(4):
        evts = _create_events(new_data, cur_col)
        assert np.all(evts[:, 2] == cur_col + 1)

    with pytest.raises(ValueError):
        _create_events(new_data, 4)


@testing.requires_testing_data
@pytest.mark.parametrize('version', all_versions)
def test_one_channel_elec_bug(version):
    """Test if loading data having only one elec in the elec field works."""
    fname = os.path.join(testing_path, 'fieldtrip',
                         'one_channel_elec_bug_data_%s.mat' % (version, ))

    with pytest.warns(**no_info_warning):
        mne.io.read_raw_fieldtrip(fname, info=None)


@testing.requires_testing_data
# Reading the sample CNT data results in a RuntimeWarning because it cannot
# parse the measurement date. We need to ignore that warning.
@pytest.mark.filterwarnings('ignore:.*parse meas date.*:RuntimeWarning')
@pytest.mark.filterwarnings('ignore:.*number of bytes.*:RuntimeWarning')
@pytest.mark.parametrize('version', all_versions)
@pytest.mark.parametrize('type', ['averaged', 'epoched', 'raw'])
def test_throw_exception_on_cellarray(version, type):
    """Test for a meaningful exception when the data is a cell array."""
    fname = os.path.join(get_data_paths('cellarray'),
                         '%s_%s.mat' % (type, version))

    info = get_raw_info('CNT')

    with pytest.raises(RuntimeError, match='Loading of data in cell arrays '
                                           'is not supported'):
        if type == 'averaged':
            mne.read_evoked_fieldtrip(fname, info)
        elif type == 'epoched':
            mne.read_epochs_fieldtrip(fname, info)
        elif type == 'raw':
            mne.io.read_raw_fieldtrip(fname, info)


@testing.requires_testing_data
def test_with_missing_channels():
    """Test _create_info when channels are missing from info."""
    cur_system = 'neuromag306'
    test_data_folder_ft = get_data_paths(cur_system)
    info = get_raw_info(cur_system)
    del info['chs'][1:20]
    info._update_redundant()

    with pytest.warns(RuntimeWarning):
        mne.io.read_raw_fieldtrip(
            os.path.join(test_data_folder_ft, 'raw_v7.mat'), info)
        mne.read_evoked_fieldtrip(
            os.path.join(test_data_folder_ft, 'averaged_v7.mat'), info)
        mne.read_epochs_fieldtrip(
            os.path.join(test_data_folder_ft, 'epoched_v7.mat'), info)


@testing.requires_testing_data
@pytest.mark.filterwarnings('ignore: Importing FieldTrip data without an info')
@pytest.mark.filterwarnings('ignore: Cannot guess the correct type')
def test_throw_error_on_non_uniform_time_field():
    """Test if an error is thrown when time fields are not uniform."""
    fname = os.path.join(testing_path, 'fieldtrip', 'not_uniform_time.mat')

    with pytest.raises(RuntimeError, match='Loading data with non-uniform '
                                           'times per epoch is not supported'):
        mne.io.read_epochs_fieldtrip(fname, info=None)


@testing.requires_testing_data
@pytest.mark.filterwarnings('ignore: Importing FieldTrip data without an info')
def test_throw_error_when_importing_old_ft_version_data():
    """Test if an error is thrown if the data was saved with an old version."""
    fname = os.path.join(testing_path, 'fieldtrip', 'old_version.mat')

    with pytest.raises(RuntimeError, match='This file was created with '
                                           'an old version of FieldTrip. You '
                                           'can convert the data to the new '
                                           'version by loading it into '
                                           'FieldTrip and applying '
                                           'ft_selectdata with an '
                                           'empty cfg structure on it. '
                                           'Otherwise you can supply '
                                           'the Info field.'):
        mne.io.read_epochs_fieldtrip(fname, info=None)
