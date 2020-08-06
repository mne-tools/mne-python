# Authors: Kyle Mathewson, Jonathan Kuziek <kuziekj@ualberta.ca>
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


@requires_testing_data
def test_boxy_filetypes():
    """Test reading parsed and unparsed BOXY data files."""
    # BOXY data files can be saved in two types (parsed and unparsed) which
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
    boxy_raw_dir = os.path.join(data_path(download=False),
                                'BOXY', 'boxy_digaux_recording', 'unparsed')

    unp_dc = mne.io.read_raw_boxy(boxy_raw_dir, 'DC', verbose=True).load_data()
    unp_ac = mne.io.read_raw_boxy(boxy_raw_dir, 'AC', verbose=True).load_data()
    unp_ph = mne.io.read_raw_boxy(boxy_raw_dir, 'Ph', verbose=True).load_data()

    # Load p_pod data.
    p_pod_dir = os.path.join(data_path(download=False),
                             'BOXY', 'boxy_digaux_recording', 'p_pod',
                             'p_pod_digaux_unparsed.mat')
    ppod_data = spio.loadmat(p_pod_dir)

    ppod_ac = np.transpose(ppod_data['ac'])
    ppod_dc = np.transpose(ppod_data['dc'])
    ppod_ph = np.transpose(ppod_data['ph'])

    # Compare MNE loaded data to p_pod loaded data.
    assert (abs(ppod_ac - unp_ac._data) <= thresh).all()
    assert (abs(ppod_dc - unp_dc._data) <= thresh).all()
    assert (abs(ppod_ph - unp_ph._data) <= thresh).all()

    # Now let's load our parsed data.
    boxy_raw_dir = os.path.join(data_path(download=False),
                                'BOXY', 'boxy_digaux_recording', 'parsed')

    par_dc = mne.io.read_raw_boxy(boxy_raw_dir, 'DC', verbose=True).load_data()
    par_ac = mne.io.read_raw_boxy(boxy_raw_dir, 'AC', verbose=True).load_data()
    par_ph = mne.io.read_raw_boxy(boxy_raw_dir, 'Ph', verbose=True).load_data()

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
    boxy_raw_dir = os.path.join(data_path(download=False),
                                'BOXY', 'boxy_digaux_recording', 'parsed')

    # The type of data shouldn't matter, but we'll test all three.
    par_dc = mne.io.read_raw_boxy(boxy_raw_dir, 'DC', verbose=True).load_data()
    par_ac = mne.io.read_raw_boxy(boxy_raw_dir, 'AC', verbose=True).load_data()
    par_ph = mne.io.read_raw_boxy(boxy_raw_dir, 'Ph', verbose=True).load_data()

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
    boxy_raw_dir = os.path.join(data_path(download=False),
                                'BOXY', 'boxy_digaux_recording', 'unparsed')

    # The type of data shouldn't matter, but we'll test all three.
    unp_dc = mne.io.read_raw_boxy(boxy_raw_dir, 'DC', verbose=True).load_data()
    unp_ac = mne.io.read_raw_boxy(boxy_raw_dir, 'AC', verbose=True).load_data()
    unp_ph = mne.io.read_raw_boxy(boxy_raw_dir, 'Ph', verbose=True).load_data()

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
    p_pod_dir = os.path.join(data_path(download=False),
                             'BOXY', 'boxy_digaux_recording',
                             'p_pod', 'p_pod_digaux_unparsed.mat')

    ppod_data = spio.loadmat(p_pod_dir)
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
