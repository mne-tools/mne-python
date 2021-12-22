# -*- coding: UTF-8 -*-
# Authors: Thomas Hartmann <thomas.hartmann@th-ht.de>
#          Dirk Gütlin <dirk.guetlin@stud.sbg.ac.at>
#
# License: BSD-3-Clause
from functools import partial
import os

import numpy as np

import mne
from mne.io.constants import FIFF
from mne.utils import object_diff


info_ignored_fields = ('file_id', 'hpi_results', 'hpi_meas', 'meas_id',
                       'meas_date', 'highpass', 'lowpass', 'subject_info',
                       'hpi_subsystem', 'experimenter', 'description',
                       'proj_id', 'proj_name', 'line_freq', 'gantry_angle',
                       'dev_head_t', 'bads', 'ctf_head_t', 'dev_ctf_t')

ch_ignore_fields = ('logno', 'cal', 'range', 'scanno', 'coil_type', 'kind',
                    'loc', 'coord_frame', 'unit')

info_long_fields = ('hpi_meas', 'projs')

system_to_reader_fn_dict = {'neuromag306': mne.io.read_raw_fif,
                            'CNT': partial(mne.io.read_raw_cnt),
                            'CTF': partial(mne.io.read_raw_ctf,
                                           clean_names=True),
                            'BTI': partial(mne.io.read_raw_bti,
                                           head_shape_fname=None,
                                           rename_channels=False,
                                           sort_by_ch_name=False),
                            'EGI': mne.io.read_raw_egi,
                            'eximia': mne.io.read_raw_eximia}

ignore_channels_dict = {'BTI': ['MUz', 'MLx', 'MLy', 'MUx', 'MUy', 'MLz']}

drop_extra_chans_dict = {'EGI': ['STI 014', 'DIN1', 'DIN3',
                                 'DIN7', 'DIN4', 'DIN5', 'DIN2'],
                         'eximia': ['GateIn', 'Trig1', 'Trig2']}

system_decimal_accuracy_dict = {'CNT': 2}

pandas_not_found_warning_msg = 'The Pandas library is not installed. Not ' \
                               'returning the original trialinfo matrix as ' \
                               'metadata.'


def _has_h5py():
    try:
        import h5py  # noqa
        return True
    except ImportError:
        return False


def _remove_ignored_ch_fields(info):
    if 'chs' in info:
        for cur_ch in info['chs']:
            for cur_field in ch_ignore_fields:
                if cur_field in cur_ch:
                    del cur_ch[cur_field]


def _remove_long_info_fields(info):
    for cur_field in info_long_fields:
        if cur_field in info:
            del info[cur_field]


def _remove_ignored_info_fields(info):
    for cur_field in info_ignored_fields:
        if cur_field in info:
            del info[cur_field]

    _remove_ignored_ch_fields(info)
    _remove_bad_dig_fields(info)


def _remove_bad_dig_fields(info):
    # The reference location appears to be lost, so we cannot add it.
    # Similarly, fiducial locations do not appear to be stored, so we
    # cannot add those, either. Same with HPI coils.
    if info['dig'] is not None:
        with info._unlock():
            info['dig'] = [d for d in info['dig']
                           if d['kind'] == FIFF.FIFFV_POINT_EEG and
                           d['ident'] != 0]  # ref


def get_data_paths(system):
    """Return common paths for all tests."""
    test_data_folder_ft = os.path.join(mne.datasets.testing.data_path(),
                                       'fieldtrip/ft_test_data', system)

    return test_data_folder_ft


def get_cfg_local(system):
    """Return cfg_local field for the system."""
    from mne.externals.pymatreader import read_mat
    cfg_local = read_mat(os.path.join(get_data_paths(system), 'raw_v7.mat'),
                         ['cfg_local'])['cfg_local']

    return cfg_local


def get_raw_info(system):
    """Return the info dict of the raw data."""
    cfg_local = get_cfg_local(system)

    raw_data_file = os.path.join(mne.datasets.testing.data_path(),
                                 cfg_local['file_name'])
    reader_function = system_to_reader_fn_dict[system]

    info = reader_function(raw_data_file, preload=False).info
    with info._unlock():
        info['comps'] = []
    return info


def get_raw_data(system, drop_extra_chs=False):
    """Find, load and process the raw data."""
    cfg_local = get_cfg_local(system)

    raw_data_file = os.path.join(mne.datasets.testing.data_path(),
                                 cfg_local['file_name'])
    reader_function = system_to_reader_fn_dict[system]

    raw_data = reader_function(raw_data_file, preload=True)
    crop = min(cfg_local['crop'], np.max(raw_data.times))
    if system == 'eximia':
        crop -= 0.5 * (1.0 / raw_data.info['sfreq'])
    raw_data.crop(0, crop)
    raw_data.del_proj('all')
    with raw_data.info._unlock():
        raw_data.info['comps'] = []
    raw_data.drop_channels(cfg_local['removed_chan_names'])

    if system in ['EGI']:
        raw_data._data[0:-1, :] = raw_data._data[0:-1, :] * 1e6

    if system in ['CNT']:
        raw_data._data = raw_data._data * 1e6

    if system in ignore_channels_dict:
        raw_data.drop_channels(ignore_channels_dict[system])

    if system in drop_extra_chans_dict and drop_extra_chs:
        raw_data.drop_channels(drop_extra_chans_dict[system])

    return raw_data


def get_epochs(system):
    """Find, load and process the epoched data."""
    cfg_local = get_cfg_local(system)
    raw_data = get_raw_data(system)

    if cfg_local['eventtype'] in raw_data.ch_names:
        stim_channel = cfg_local['eventtype']
    else:
        stim_channel = 'STI 014'

    if system == 'CNT':
        events, event_id = mne.events_from_annotations(raw_data)
        events[:, 0] = events[:, 0] + 1
    else:
        events = mne.find_events(raw_data, stim_channel=stim_channel,
                                 shortest_event=1)

        if isinstance(cfg_local['eventvalue'], np.ndarray):
            event_id = list(cfg_local['eventvalue'].astype('int'))
        else:
            event_id = [int(cfg_local['eventvalue'])]

        event_id = [id for id in event_id if id in events[:, 2]]

    epochs = mne.Epochs(raw_data, events=events,
                        event_id=event_id,
                        tmin=-cfg_local['prestim'],
                        tmax=cfg_local['poststim'], baseline=None)

    return epochs


def get_evoked(system):
    """Find, load and process the avg data."""
    epochs = get_epochs(system)
    return epochs.average(picks=np.arange(len(epochs.ch_names)))


def check_info_fields(expected, actual, has_raw_info, ignore_long=True):
    """
    Check if info fields are equal.

    Some fields are ignored.
    """
    expected = expected.info.copy()
    actual = actual.info.copy()

    if not has_raw_info:
        _remove_ignored_info_fields(expected)
        _remove_ignored_info_fields(actual)

    _remove_long_info_fields(expected)
    _remove_long_info_fields(actual)

    # we annoyingly have two ways of representing this, so just always use
    # an empty list here
    for obj in (expected, actual):
        if obj['dig'] is None:
            with obj._unlock():
                obj['dig'] = []

    d = object_diff(actual, expected, allclose=True)
    assert d == '', d


def check_data(expected, actual, system):
    """Check data for equality."""
    decimal = 7
    if system in system_decimal_accuracy_dict:
        decimal = system_decimal_accuracy_dict[system]

    np.testing.assert_almost_equal(expected, actual, decimal=decimal)


def assert_warning_in_record(warning_message, warn_record):
    """Assert that a warning message is in the records."""
    all_messages = [str(w.message) for w in warn_record]
    assert warning_message in all_messages
