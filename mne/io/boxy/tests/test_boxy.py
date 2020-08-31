# Authors: Kyle Mathewson, Jonathan Kuziek <kuziek@ualberta.ca>
#
# License: BSD (3-clause)

import os

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import scipy.io as spio

import mne
from mne.datasets.testing import data_path, requires_testing_data


@requires_testing_data
def test_boxy_load():
    """Test reading BOXY files."""
    # Determine to which decimal place we will compare.
    thresh = 1e-10

    # Load AC, DC, and Phase data.
    boxy_file = os.path.join(data_path(download=False),
                             'BOXY', 'boxy_0_40_recording',
                             'boxy_0_40_notriggers_unparsed.txt')

    boxy_data = mne.io.read_raw_boxy(boxy_file, verbose=True).load_data()

    # Test sampling rate.
    assert boxy_data.info['sfreq'] == 62.5

    # Grab our different data types.
    chans_dc = np.arange(0, 80) * 3 + 0
    chans_ac = np.arange(0, 80) * 3 + 1
    chans_ph = np.arange(0, 80) * 3 + 2

    mne_dc = boxy_data.copy().pick(chans_dc)
    mne_ac = boxy_data.copy().pick(chans_ac)
    mne_ph = boxy_data.copy().pick(chans_ph)

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

    # Check description.
    assert mne_dc._annotations.description.size == 0
    assert mne_ac._annotations.description.size == 0
    assert mne_ph._annotations.description.size == 0

    # Check duration.
    assert mne_dc._annotations.duration.size == 0
    assert mne_ac._annotations.duration.size == 0
    assert mne_ph._annotations.duration.size == 0

    # Check onset.
    assert mne_dc._annotations.onset.size == 0
    assert mne_ac._annotations.onset.size == 0
    assert mne_ph._annotations.onset.size == 0

    # Load p_pod data.
    p_pod_file = os.path.join(data_path(download=False),
                              'BOXY', 'boxy_0_40_recording',
                              'p_pod_10_6_3_loaded_data',
                              'p_pod_10_6_3_notriggers_unparsed.mat')
    ppod_data = spio.loadmat(p_pod_file)

    ppod_ac = np.transpose(ppod_data['ac'])
    ppod_dc = np.transpose(ppod_data['dc'])
    ppod_ph = np.transpose(ppod_data['ph'])

    # Compare MNE loaded data to p_pod loaded data.
    assert (abs(ppod_ac - mne_ac._data) <= thresh).all()
    assert (abs(ppod_dc - mne_dc._data) <= thresh).all()
    assert (abs(ppod_ph - mne_ph._data) <= thresh).all()


@requires_testing_data
def test_boxy_filetypes():
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

    # Determine to which decimal place we will compare.
    thresh = 1e-10

    # Load AC, DC, and Phase data.
    boxy_file = os.path.join(data_path(download=False),
                             'BOXY', 'boxy_0_84_digaux_recording',
                             'boxy_0_84_triggers_unparsed.txt')

    boxy_data = mne.io.read_raw_boxy(boxy_file, verbose=True).load_data()

    # Test sampling rate.
    assert boxy_data.info['sfreq'] == 79.4722

    # Grab our different data types.
    chans_dc = np.arange(0, 8) * 3 + 0
    chans_ac = np.arange(0, 8) * 3 + 1
    chans_ph = np.arange(0, 8) * 3 + 2

    unp_dc = boxy_data.copy().pick(chans_dc)
    unp_ac = boxy_data.copy().pick(chans_ac)
    unp_ph = boxy_data.copy().pick(chans_ph)

    # Check channel names.
    chans = ['S1_D1', 'S2_D1', 'S3_D1', 'S4_D1',
             'S5_D1', 'S6_D1', 'S7_D1', 'S8_D1']

    assert unp_dc.info['ch_names'] == [i_chan + ' ' + 'DC'
                                       for i_chan in chans]
    assert unp_ac.info['ch_names'] == [i_chan + ' ' + 'AC'
                                       for i_chan in chans]
    assert unp_ph.info['ch_names'] == [i_chan + ' ' + 'Ph'
                                       for i_chan in chans]

    # Load p_pod data.
    p_pod_file = os.path.join(data_path(download=False),
                              'BOXY', 'boxy_0_84_digaux_recording',
                              'p_pod_10_6_3_loaded_data',
                              'p_pod_10_6_3_triggers_unparsed.mat')
    ppod_data = spio.loadmat(p_pod_file)

    ppod_ac = np.transpose(ppod_data['ac'])
    ppod_dc = np.transpose(ppod_data['dc'])
    ppod_ph = np.transpose(ppod_data['ph'])

    # Compare MNE loaded data to p_pod loaded data.
    assert (abs(ppod_ac - unp_ac._data) <= thresh).all()
    assert (abs(ppod_dc - unp_dc._data) <= thresh).all()
    assert (abs(ppod_ph - unp_ph._data) <= thresh).all()

    # Now let's load our parsed data.
    boxy_file = os.path.join(data_path(download=False),
                             'BOXY', 'boxy_0_84_digaux_recording',
                             'boxy_0_84_triggers_unparsed.txt')

    boxy_data = mne.io.read_raw_boxy(boxy_file, verbose=True).load_data()

    # Test sampling rate.
    assert boxy_data.info['sfreq'] == 79.4722

    # Grab our different data types.
    par_dc = boxy_data.copy().pick(chans_dc)
    par_ac = boxy_data.copy().pick(chans_ac)
    par_ph = boxy_data.copy().pick(chans_ph)

    # Check channel names.
    assert par_dc.info['ch_names'] == [i_chan + ' ' + 'DC'
                                       for i_chan in chans]
    assert par_ac.info['ch_names'] == [i_chan + ' ' + 'AC'
                                       for i_chan in chans]
    assert par_ph.info['ch_names'] == [i_chan + ' ' + 'Ph'
                                       for i_chan in chans]

    # Compare parsed and unparsed data.
    assert (abs(unp_dc._data - par_dc._data) == 0).all()
    assert (abs(unp_ac._data - par_ac._data) == 0).all()
    assert (abs(unp_ph._data - par_ph._data) == 0).all()


@requires_testing_data
def test_boxy_digaux():
    """Test reading BOXY files and generating annotations from digaux."""
    # We'll test both parsed and unparsed boxy data files.
    # Set our comparison threshold and sampling rate.
    thresh = 1e-6
    srate = 79.4722

    # Load AC, DC, and Phase data from a parsed file first.
    boxy_file = os.path.join(data_path(download=False),
                             'BOXY', 'boxy_0_84_digaux_recording',
                             'boxy_0_84_triggers_parsed.txt')

    boxy_data = mne.io.read_raw_boxy(boxy_file, verbose=True).load_data()

    # Grab our different data types.
    chans_dc = np.arange(0, 8) * 3 + 0
    chans_ac = np.arange(0, 8) * 3 + 1
    chans_ph = np.arange(0, 8) * 3 + 2

    par_dc = boxy_data.copy().pick(chans_dc)
    par_ac = boxy_data.copy().pick(chans_ac)
    par_ph = boxy_data.copy().pick(chans_ph)

    # Check that our event order matches what we expect.
    event_list = ['1.0', '2.0', '3.0', '4.0', '5.0']
    assert_array_equal(par_dc.annotations.description, event_list)
    assert_array_equal(par_ac.annotations.description, event_list)
    assert_array_equal(par_ph.annotations.description, event_list)

    # Check that our event timings are what we expect.
    event_onset = [i_time * (1.0 / srate) for i_time in
                   [105, 185, 265, 344, 424]]
    assert_allclose(par_dc.annotations.onset, event_onset, atol=thresh)
    assert_allclose(par_ac.annotations.onset, event_onset, atol=thresh)
    assert_allclose(par_ph.annotations.onset, event_onset, atol=thresh)

    # Now we'll load data from an unparsed file.
    boxy_file = os.path.join(data_path(download=False),
                             'BOXY', 'boxy_0_84_digaux_recording',
                             'boxy_0_84_triggers_unparsed.txt')

    boxy_data = mne.io.read_raw_boxy(boxy_file, verbose=True).load_data()

    # Grab our different data types.
    unp_dc = boxy_data.copy().pick(chans_dc)
    unp_ac = boxy_data.copy().pick(chans_ac)
    unp_ph = boxy_data.copy().pick(chans_ph)

    # Check that our event order matches what we expect.
    event_list = ['1.0', '2.0', '3.0', '4.0', '5.0']
    assert_array_equal(unp_dc.annotations.description, event_list)
    assert_array_equal(unp_ac.annotations.description, event_list)
    assert_array_equal(unp_ph.annotations.description, event_list)

    # Check that our event timings are what we expect.
    event_onset = [i_time * (1.0 / srate) for i_time in
                   [105, 185, 265, 344, 424]]
    assert_allclose(unp_dc.annotations.onset, event_onset, atol=thresh)
    assert_allclose(unp_ac.annotations.onset, event_onset, atol=thresh)
    assert_allclose(unp_ph.annotations.onset, event_onset, atol=thresh)

    # Now let's compare parsed and unparsed events to p_pod loaded digaux.
    # Load our p_pod data.
    p_pod_file = os.path.join(data_path(download=False),
                              'BOXY', 'boxy_0_84_digaux_recording',
                              'p_pod_10_6_3_loaded_data',
                              'p_pod_10_6_3_triggers_unparsed.mat')

    ppod_data = spio.loadmat(p_pod_file)
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

    # Check that our event orders match.
    assert_array_equal(par_dc.annotations.description, description)
    assert_array_equal(par_ac.annotations.description, description)
    assert_array_equal(par_ph.annotations.description, description)
    assert_array_equal(unp_dc.annotations.description, description)
    assert_array_equal(unp_ac.annotations.description, description)
    assert_array_equal(unp_ph.annotations.description, description)

    # Check that our event timings match.
    assert_allclose(par_dc.annotations.onset, onset, atol=thresh)
    assert_allclose(par_ac.annotations.onset, onset, atol=thresh)
    assert_allclose(par_ph.annotations.onset, onset, atol=thresh)
    assert_allclose(unp_dc.annotations.onset, onset, atol=thresh)
    assert_allclose(unp_ac.annotations.onset, onset, atol=thresh)
    assert_allclose(unp_ph.annotations.onset, onset, atol=thresh)
