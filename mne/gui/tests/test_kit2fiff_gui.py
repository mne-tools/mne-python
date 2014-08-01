# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from nose.tools import assert_true, assert_false, assert_equal

import mne
from mne.io.kit.tests import data_dir as kit_data_dir
from mne.io import Raw
from mne.utils import _TempDir, requires_traits

mrk_pre_path = os.path.join(kit_data_dir, 'test_mrk_pre.sqd')
mrk_post_path = os.path.join(kit_data_dir, 'test_mrk_post.sqd')
sqd_path = os.path.join(kit_data_dir, 'test.sqd')
hsp_path = os.path.join(kit_data_dir, 'test_hsp.txt')
fid_path = os.path.join(kit_data_dir, 'test_elp.txt')
fif_path = os.path.join(kit_data_dir, 'test_bin_raw.fif')

tempdir = _TempDir()
tgt_fname = os.path.join(tempdir, 'test-raw.fif')
std_fname = os.path.join(tempdir, 'test_std-raw.fif')


@requires_traits
def test_kit2fiff_model():
    """Test CombineMarkersModel Traits Model"""
    from mne.gui._kit2fiff_gui import Kit2FiffModel

    model = Kit2FiffModel()
    assert_false(model.can_save)
    model.markers.mrk1.file = mrk_pre_path
    model.markers.mrk2.file = mrk_post_path
    model.sqd_file = sqd_path
    model.hsp_file = hsp_path
    assert_false(model.can_save)
    model.fid_file = fid_path

    # export raw
    assert_true(model.can_save)
    raw_out = model.get_raw()
    raw_out.save(tgt_fname)
    raw = Raw(tgt_fname)

    # Compare exported raw with the original binary conversion
    raw_bin = Raw(fif_path)
    trans_bin = raw.info['dev_head_t']['trans']
    assert_equal(raw_bin.info.keys(), raw.info.keys())
    trans_transform = raw_bin.info['dev_head_t']['trans']
    assert_allclose(trans_transform, trans_bin, 0.1)

    # Averaging markers
    model.markers.mrk3.method = "Average"
    trans_avg = model.dev_head_trans
    assert_false(np.all(trans_avg == trans_transform))
    assert_allclose(trans_avg, trans_bin, 0.1)

    # Test exclusion of one marker
    model.markers.mrk3.method = "Transform"
    model.use_mrk = [1, 2, 3, 4]
    assert_false(np.all(model.dev_head_trans == trans_transform))
    assert_false(np.all(model.dev_head_trans == trans_avg))
    assert_false(np.all(model.dev_head_trans == np.eye(4)))

    # test setting stim channels
    model.stim_slope = '+'
    events_bin = mne.find_events(raw_bin, stim_channel='STI 014')

    model.stim_chs = '<'
    raw = model.get_raw()
    events = mne.find_events(raw, stim_channel='STI 014')
    assert_array_equal(events, events_bin)

    events_rev = events_bin.copy()
    events_rev[:, 2] = 1
    model.stim_chs = '>'
    raw = model.get_raw()
    events = mne.find_events(raw, stim_channel='STI 014')
    assert_array_equal(events, events_rev)

    model.stim_chs = 'man'
    model.stim_chs_manual = list(range(167, 159, -1))
    raw = model.get_raw()
    events = mne.find_events(raw, stim_channel='STI 014')
    assert_array_equal(events, events_bin)
