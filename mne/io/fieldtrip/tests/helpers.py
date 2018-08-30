# -*- coding: UTF-8 -*-
# Authors: Thomas Hartmann <thomas.hartmann@th-ht.de>
#          Dirk Gütlin <dirk.guetlin@stud.sbg.ac.at>
#
# License: BSD (3-clause)
import types
import numpy as np
import copy
import os
from mne.io.constants import FIFF
import mne

from functools import partial

info_ignored_fields = ('file_id', 'hpi_results', 'hpi_meas', 'meas_id',
                       'meas_date', 'highpass', 'lowpass', 'subject_info',
                       'hpi_subsystem', 'experimenter', 'description',
                       'proj_id', 'proj_name', 'line_freq', 'gantry_angle',
                       'dev_head_t', 'dig', 'bads', 'projs', 'ctf_head_t',
                       'dev_ctf_t')

ch_ignore_fields = ('logno', 'cal', 'range', 'scanno')

info_long_fields = ('hpi_meas', )

system_to_reader_fn_dict = {'neuromag306': mne.io.read_raw_fif,
                            'CNT': partial(mne.io.read_raw_cnt, montage=None),
                            'CTF': partial(mne.io.read_raw_ctf,
                                           clean_names=True),
                            'BTI': partial(mne.io.read_raw_bti,
                                           head_shape_fname=None,
                                           rename_channels=False,
                                           sort_by_ch_name=False),
                            'EGI': mne.io.read_raw_egi,
                            'KIT': mne.io.read_raw_kit,
                            'eximia': mne.io.read_raw_eximia}

ignore_channels_dict = {'BTI': ['MUz', 'MLx', 'MLy', 'MUx', 'MUy', 'MLz']}

drop_extra_chans_dict = {'EGI': ['STI 014', 'DIN1', 'DIN3',
                                 'DIN7', 'DIN4', 'DIN5', 'DIN2'],
                         'eximia': ['GateIn', 'Trig1', 'Trig2']}

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


def _remove_tangential_plane_from_ori(info):
    if 'chs' in info:
        for cur_ch in info['chs']:
            cur_ch['loc'][3:9] = 0
            cur_ch['loc'][3:] = np.around(cur_ch['loc'][3:], 2)
            cur_ch['loc'] = np.around(cur_ch['loc'], 3)


def _remove_long_info_fields(info):
    for cur_field in info_long_fields:
        if cur_field in info:
            del info[cur_field]


def _remove_ignored_info_fields(info):
    for cur_field in info_ignored_fields:
        if cur_field in info:
            del info[cur_field]

    _remove_ignored_ch_fields(info)


def _transform_chs_to_head_coords(info):
    if 'dev_ctf_t' in info and info['dev_ctf_t'] is not None:
        trans = info['dev_ctf_t']
    elif 'dev_head_t' in info and info['dev_head_t'] is not None:
        trans = info['dev_head_t']
    else:
        return

    for cur_ch in info['chs']:
        if cur_ch['coord_frame'] == FIFF.FIFFV_COORD_DEVICE:
            cur_trans_orig = mne.io.tag._loc_to_coil_trans(cur_ch['loc'])
            trans_transformed = np.dot(trans['trans'], cur_trans_orig)
            cur_ch['loc'] = mne.io.tag._coil_trans_to_loc(trans_transformed)
            cur_ch['coord_frame'] = FIFF.FIFFV_COORD_HEAD


def get_data_paths(system):
    """Return common paths for all tests."""
    test_data_folder_ft = os.path.join(mne.datasets.testing.data_path(),
                                       'fieldtrip/ft_test_data', system)

    return test_data_folder_ft


def get_cfg_local(system):
    """Return cfg_local field for the system."""
    from mne.externals.pymatreader import read_mat
    cfg_local = read_mat(os.path.join(get_data_paths(system), 'raw_v7.mat'),
                         ('cfg_local',))['cfg_local']

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


def get_raw_data(system, drop_sti_cnt=True, drop_extra_chs=False):
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
    raw_data.set_eeg_reference([])
    raw_data.del_proj('all')
    raw_data.info['comps'] = []
    raw_data.drop_channels(cfg_local['removed_chan_names'])

    if system == 'CNT':
        raw_data._data[0:-1, :] = raw_data._data[0:-1, :] * 1e6

    if system == 'CNT' and drop_sti_cnt:
        raw_data.drop_channels(['STI 014'])

    if system in ignore_channels_dict:
        raw_data.drop_channels(ignore_channels_dict[system])

    if system in drop_extra_chans_dict and drop_extra_chs:
        raw_data.drop_channels(drop_extra_chans_dict[system])

    return raw_data


def get_epoched_data(system):
    """Find, load and process the epoched data."""
    cfg_local = get_cfg_local(system)
    raw_data = get_raw_data(system, drop_sti_cnt=False)

    if cfg_local['eventtype'] in raw_data.ch_names:
        stim_channel = cfg_local['eventtype']
    else:
        stim_channel = 'STI 014'

    events = mne.find_events(raw_data, stim_channel=stim_channel,
                             shortest_event=1)

    if system == 'CNT':
        raw_data.drop_channels(['STI 014'])

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


def get_averaged_data(system):
    """Find, load and process the avg data."""
    data = get_epoched_data(system)

    return data.average(picks=np.arange(len(data.ch_names)))


def check_info_fields(expected, actual, has_raw_info, ignore_long=True):
    """
    Check if info fields are equal.

    Some fields are ignored.
    """
    expected = copy.deepcopy(expected.info)
    actual = copy.deepcopy(actual.info)

    if not has_raw_info:
        _transform_chs_to_head_coords(expected)
        _transform_chs_to_head_coords(actual)

        _remove_ignored_info_fields(expected)
        _remove_ignored_info_fields(actual)

        # Coordinates are now in head reference frame. The orientation rotation
        # matrix is thus redundant in the sense that the third column is the
        # cross product of the first two. The third is the unit vector of the
        # direction perpendicular to the coil. FieldTrip only stores this
        # vector.
        # We recreate the rotation matrix using
        # `mne.transforms.rotation3d_align_z_axis`. However, the first and
        # second
        # column of the rotation matrix are now arbitrary and thus need to be
        # deleted.
        # We are also allowing the orientation to be more inaccurate (up to 2
        # decimal points)
        _remove_tangential_plane_from_ori(expected)
        _remove_tangential_plane_from_ori(actual)

    if info_long_fields:
        _remove_long_info_fields(expected)
        _remove_long_info_fields(actual)

    assert_deep_almost_equal(expected, actual)


def check_data(expected, actual):
    """Check data for equality."""
    np.testing.assert_almost_equal(expected, actual)


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
