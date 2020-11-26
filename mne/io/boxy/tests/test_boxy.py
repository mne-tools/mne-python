# Authors: Kyle Mathewson, Jonathan Kuziek <kuziek@ualberta.ca>
#
# License: BSD (3-clause)

import os.path as op

import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_array_equal,
                           assert_array_less)
import scipy.io as spio

from mne import pick_types
from mne.datasets import testing
from mne.io import read_raw_boxy
from mne.io.tests.test_raw import _test_raw_reader

data_path = testing.data_path(download=False)
boxy_0_40 = op.join(
    data_path, 'BOXY', 'boxy_0_40_recording',
    'boxy_0_40_notriggers_unparsed.txt')
p_pod_0_40 = op.join(
    data_path, 'BOXY', 'boxy_0_40_recording', 'p_pod_10_6_3_loaded_data',
    'p_pod_10_6_3_notriggers_unparsed.mat')
boxy_0_84 = op.join(
    data_path, 'BOXY', 'boxy_0_84_digaux_recording',
    'boxy_0_84_triggers_unparsed.txt')
boxy_0_84_parsed = op.join(
    data_path, 'BOXY', 'boxy_0_84_digaux_recording',
    'boxy_0_84_triggers_parsed.txt')
p_pod_0_84 = op.join(
    data_path, 'BOXY', 'boxy_0_84_digaux_recording',
    'p_pod_10_6_3_loaded_data', 'p_pod_10_6_3_triggers_unparsed.mat')


def _assert_ppod(raw, p_pod_file):
    have_types = raw.get_channel_types(unique=True)
    assert 'fnirs_fd_phase' in raw, have_types
    assert 'fnirs_cw_amplitude' in raw, have_types
    assert 'fnirs_fd_ac_amplitude' in raw, have_types
    ppod_data = spio.loadmat(p_pod_file)

    # Compare MNE loaded data to p_pod loaded data.
    map_ = dict(dc='fnirs_cw_amplitude', ac='fnirs_fd_ac_amplitude',
                ph='fnirs_fd_phase')
    for key, value in map_.items():
        ppod = ppod_data[key].T
        m = np.median(np.abs(ppod))
        assert 1e-1 < m < 1e5, key  # our atol is meaningful
        atol = m * 1e-10
        py = raw.get_data(value)
        if key == 'ph':  # radians
            assert_array_less(-np.pi, py)
            assert_array_less(py, 3 * np.pi)
            py = np.rad2deg(py)
        assert_allclose(py, ppod, atol=atol, err_msg=key)


@testing.requires_testing_data
def test_boxy_load():
    """Test reading BOXY files."""
    raw = read_raw_boxy(boxy_0_40, verbose=True)
    assert raw.info['sfreq'] == 62.5
    _assert_ppod(raw, p_pod_0_40)

    # Grab our different data types.
    mne_ph = raw.copy().pick(picks='fnirs_fd_phase')
    mne_dc = raw.copy().pick(picks='fnirs_cw_amplitude')
    mne_ac = raw.copy().pick(picks='fnirs_fd_ac_amplitude')

    # Check channel names.
    first_chans = ['S1_D1', 'S2_D1', 'S3_D1', 'S4_D1', 'S5_D1',
                   'S6_D1', 'S7_D1', 'S8_D1', 'S9_D1', 'S10_D1']
    last_chans = ['S1_D8', 'S2_D8', 'S3_D8', 'S4_D8', 'S5_D8',
                  'S6_D8', 'S7_D8', 'S8_D8', 'S9_D8', 'S10_D8']

    assert mne_dc.info['ch_names'][:10] == [i_chan + ' ' + 'DC'
                                            for i_chan in first_chans]
    assert mne_ac.info['ch_names'][:10] == [i_chan + ' ' + 'AC'
                                            for i_chan in first_chans]
    assert mne_ph.info['ch_names'][:10] == [i_chan + ' ' + 'Ph'
                                            for i_chan in first_chans]

    assert mne_dc.info['ch_names'][70::] == [i_chan + ' ' + 'DC'
                                             for i_chan in last_chans]
    assert mne_ac.info['ch_names'][70::] == [i_chan + ' ' + 'AC'
                                             for i_chan in last_chans]
    assert mne_ph.info['ch_names'][70::] == [i_chan + ' ' + 'Ph'
                                             for i_chan in last_chans]

    # Since this data set has no 'digaux' for creating trigger annotations,
    # let's make sure our Raw object has no annotations.
    assert len(raw.annotations) == 0


@testing.requires_testing_data
@pytest.mark.parametrize('fname', (boxy_0_84, boxy_0_84_parsed))
def test_boxy_filetypes(fname):
    """Test reading parsed and unparsed BOXY data files."""
    # BOXY data files can be saved in two formats (parsed and unparsed) which
    # mostly determines how the data is organised.
    # For parsed files, each row is a single timepoint and all
    # source/detector combinations are represented as columns.
    # For unparsed files, each row is a source and each group of n rows
    # represents a timepoint. For example, if there are ten sources in the raw
    # data then the first ten rows represent the ten sources at timepoint 1
    # while the next set of ten rows are the ten sources at timepoint 2.
    # Detectors are represented as columns.

    # Since p_pod is designed to only load unparsed files, we will first
    # compare MNE and p_pod loaded data from an unparsed data file. If those
    # files are comparable, then we will compare the MNE loaded data between
    # parsed and unparsed files.
    raw = read_raw_boxy(fname, verbose=True)
    assert raw.info['sfreq'] == 79.4722
    _assert_ppod(raw, p_pod_0_84)

    # Grab our different data types.
    unp_dc = raw.copy().pick('fnirs_cw_amplitude')
    unp_ac = raw.copy().pick('fnirs_fd_ac_amplitude')
    unp_ph = raw.copy().pick('fnirs_fd_phase')

    # Check channel names.
    chans = ['S1_D1', 'S2_D1', 'S3_D1', 'S4_D1',
             'S5_D1', 'S6_D1', 'S7_D1', 'S8_D1']

    assert unp_dc.info['ch_names'] == [i_chan + ' ' + 'DC'
                                       for i_chan in chans]
    assert unp_ac.info['ch_names'] == [i_chan + ' ' + 'AC'
                                       for i_chan in chans]
    assert unp_ph.info['ch_names'] == [i_chan + ' ' + 'Ph'
                                       for i_chan in chans]


@testing.requires_testing_data
@pytest.mark.parametrize('fname', (boxy_0_84, boxy_0_84_parsed))
def test_boxy_digaux(fname):
    """Test reading BOXY files and generating annotations from digaux."""
    srate = 79.4722
    raw = read_raw_boxy(fname, verbose=True)

    # Grab our different data types.
    picks_dc = pick_types(raw.info, fnirs='fnirs_cw_amplitude')
    picks_ac = pick_types(raw.info, fnirs='fnirs_fd_ac_amplitude')
    picks_ph = pick_types(raw.info, fnirs='fnirs_fd_phase')
    assert_array_equal(picks_dc, np.arange(0, 8) * 3 + 0)
    assert_array_equal(picks_ac, np.arange(0, 8) * 3 + 1)
    assert_array_equal(picks_ph, np.arange(0, 8) * 3 + 2)

    # Check that our event order matches what we expect.
    event_list = ['1.0', '2.0', '3.0', '4.0', '5.0']
    assert_array_equal(raw.annotations.description, event_list)

    # Check that our event timings are what we expect.
    event_onset = [i_time * (1.0 / srate) for i_time in
                   [105, 185, 265, 344, 424]]
    assert_allclose(raw.annotations.onset, event_onset, atol=1e-6)

    # Now let's compare parsed and unparsed events to p_pod loaded digaux.
    # Load our p_pod data.
    ppod_data = spio.loadmat(p_pod_0_84)
    ppod_digaux = np.transpose(ppod_data['digaux'])[0]

    # Now let's get our triggers from the p_pod digaux.
    # We only want the first instance of each trigger.
    prev_mrk = 0
    mrk_idx = list()
    duration = list()
    tmp_dur = 0
    for i_num, i_mrk in enumerate(ppod_digaux):
        if i_mrk != 0 and i_mrk != prev_mrk:
            mrk_idx.append(i_num)
        if i_mrk != 0 and i_mrk == prev_mrk:
            tmp_dur += 1
        if i_mrk == 0 and i_mrk != prev_mrk:
            duration.append((tmp_dur + 1) * (1.0 / srate))
            tmp_dur = 0
        prev_mrk = i_mrk
    onset = np.asarray([i_mrk * (1.0 / srate) for i_mrk in mrk_idx])
    description = np.asarray([str(float(i_mrk))for i_mrk in
                              ppod_digaux[mrk_idx]])
    assert_array_equal(raw.annotations.description, description)
    assert_allclose(raw.annotations.onset, onset, atol=1e-6)


@testing.requires_testing_data
@pytest.mark.parametrize('fname', (boxy_0_40, boxy_0_84, boxy_0_84_parsed))
def test_raw_properties(fname):
    """Test raw reader properties."""
    _test_raw_reader(read_raw_boxy, fname=fname, boundary_decimal=1)
