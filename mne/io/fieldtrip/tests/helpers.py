# -*- coding: UTF-8 -*-
# Authors: Thomas Hartmann <thomas.hartmann@th-ht.de>
#          Dirk Gütlin <dirk.guetlin@stud.sbg.ac.at>
#
# License: BSD (3-clause)
from functools import partial
import os
import types

import numpy as np
import pytest

import mne


info_ignored_fields = ('file_id', 'hpi_results', 'hpi_meas', 'meas_id',
                       'meas_date', 'highpass', 'lowpass', 'subject_info',
                       'hpi_subsystem', 'experimenter', 'description',
                       'proj_id', 'proj_name', 'line_freq', 'gantry_angle',
                       'dev_head_t', 'dig', 'bads', 'projs', 'ctf_head_t',
                       'dev_ctf_t')

ch_ignore_fields = ('logno', 'cal', 'range', 'scanno', 'coil_type', 'kind',
                    'loc', 'coord_frame', 'unit')

info_long_fields = ('hpi_meas', )

system_to_reader_fn_dict = {'neuromag306': mne.io.read_raw_fif,
                            'CNT': partial(mne.io.read_raw_cnt,
                                           stim_channel=False,
                                           montage=None),
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
    if any([type_ in raw_data for type_ in ['eeg', 'ecog', 'seeg']]):
        raw_data.set_eeg_reference([])
    else:
        with pytest.raises(ValueError, match='No EEG, ECoG or sEEG channels'):
            raw_data.set_eeg_reference([])
    raw_data.del_proj('all')
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

    if info_long_fields:
        _remove_long_info_fields(expected)
        _remove_long_info_fields(actual)

    assert_deep_almost_equal(expected, actual)


def check_data(expected, actual, system):
    """Check data for equality."""
    decimal = 7
    if system in system_decimal_accuracy_dict:
        decimal = system_decimal_accuracy_dict[system]

    np.testing.assert_almost_equal(expected, actual, decimal=decimal)


def assert_deep_almost_equal(expected, actual, *args, **kwargs):
    """
    Assert that two complex structures have almost equal contents.

    Compares lists, dicts and tuples recursively. Checks numeric values
    using test_case's :py:meth:`unittest.TestCase.assertAlmostEqual` and
    checks all other values with :py:meth:`unittest.TestCase.assertEqual`.
    Accepts additional positional and keyword arguments and pass those
    intact to assertAlmostEqual() (that's how you specify comparison
    precision).

    This code has been adapted from
    https://github.com/larsbutler/oq-engine/blob/master/tests/utils/helpers.py
    """
    is_root = '__trace' not in kwargs
    trace = kwargs.pop('__trace', 'ROOT')

    if isinstance(expected, np.ndarray) and expected.size == 0:
        expected = None

    if isinstance(actual, np.ndarray) and actual.size == 0:
        actual = None

    try:
        if isinstance(expected, (int, float, complex)):
            np.testing.assert_almost_equal(expected, actual, *args, **kwargs)
        elif isinstance(expected, (list, tuple, np.ndarray,
                                   types.GeneratorType)):
            if isinstance(expected, types.GeneratorType):
                expected = list(expected)
                actual = list(actual)

                np.testing.assert_equal(len(expected), len(actual))
            for index in range(len(expected)):
                v1, v2 = expected[index], actual[index]
                assert_deep_almost_equal(v1, v2,
                                         __trace=repr(index), *args, **kwargs)
        elif isinstance(expected, dict):
            np.testing.assert_equal(set(expected), set(actual))
            for key in expected:
                assert_deep_almost_equal(expected[key], actual[key],
                                         __trace=repr(key), *args, **kwargs)
        else:
            np.testing.assert_equal(expected, actual)
    except AssertionError as exc:
        exc.__dict__.setdefault('traces', []).append(trace)
        if is_root:
            trace = ' -> '.join(reversed(exc.traces))
            message = ''
            try:
                message = exc.message
            except AttributeError:
                pass
            exc = AssertionError("%s\nTRACE: %s" % (message, trace))
        raise exc


def assert_warning_in_record(warning_message, warn_record):
    """Assert that a warning message is in the records."""
    all_messages = [str(w.message) for w in warn_record]
    assert warning_message in all_messages
