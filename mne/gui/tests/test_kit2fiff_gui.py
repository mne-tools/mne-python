# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from nose import SkipTest
from nose.tools import assert_true, assert_false, assert_equal

import mne
from mne.io.kit.tests import data_dir as kit_data_dir
from mne.io import read_raw_fif
from mne.utils import _TempDir, requires_mayavi, run_tests_if_main

mrk_pre_path = os.path.join(kit_data_dir, 'test_mrk_pre.sqd')
mrk_post_path = os.path.join(kit_data_dir, 'test_mrk_post.sqd')
sqd_path = os.path.join(kit_data_dir, 'test.sqd')
hsp_path = os.path.join(kit_data_dir, 'test_hsp.txt')
fid_path = os.path.join(kit_data_dir, 'test_elp.txt')
fif_path = os.path.join(kit_data_dir, 'test_bin_raw.fif')

warnings.simplefilter('always')


@requires_mayavi
def test_kit2fiff_model():
    """Test Kit2Fiff model."""
    from mne.gui._kit2fiff_gui import Kit2FiffModel
    tempdir = _TempDir()
    tgt_fname = os.path.join(tempdir, 'test-raw.fif')

    model = Kit2FiffModel()
    assert_false(model.can_save)
    assert_equal(model.misc_chs_desc, "No SQD file selected...")
    assert_equal(model.stim_chs_comment, "")
    model.markers.mrk1.file = mrk_pre_path
    model.markers.mrk2.file = mrk_post_path
    model.sqd_file = sqd_path
    assert_equal(model.misc_chs_desc, "160:192")
    model.hsp_file = hsp_path
    assert_false(model.can_save)
    model.fid_file = fid_path
    assert_true(model.can_save)

    # events
    model.stim_slope = '+'
    assert_equal(model.get_event_info(), {1: 2})
    model.stim_slope = '-'
    assert_equal(model.get_event_info(), {254: 2, 255: 2})

    # stim channels
    model.stim_chs = "181:184, 186"
    assert_array_equal(model.stim_chs_array, [181, 182, 183, 186])
    assert_true(model.stim_chs_ok)
    assert_equal(model.get_event_info(), {})
    model.stim_chs = "181:184, bad"
    assert_false(model.stim_chs_ok)
    assert_false(model.can_save)
    model.stim_chs = ""
    assert_true(model.can_save)

    # export raw
    raw_out = model.get_raw()
    raw_out.save(tgt_fname)
    raw = read_raw_fif(tgt_fname)

    # Compare exported raw with the original binary conversion
    raw_bin = read_raw_fif(fif_path)
    trans_bin = raw.info['dev_head_t']['trans']
    want_keys = list(raw_bin.info.keys())
    assert_equal(sorted(want_keys), sorted(list(raw.info.keys())))
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

    model.stim_coding = '<'
    raw = model.get_raw()
    events = mne.find_events(raw, stim_channel='STI 014')
    assert_array_equal(events, events_bin)

    events_rev = events_bin.copy()
    events_rev[:, 2] = 1
    model.stim_coding = '>'
    raw = model.get_raw()
    events = mne.find_events(raw, stim_channel='STI 014')
    assert_array_equal(events, events_rev)

    model.stim_coding = 'channel'
    model.stim_chs = "160:161"
    raw = model.get_raw()
    events = mne.find_events(raw, stim_channel='STI 014')
    assert_array_equal(events, events_bin + [0, 0, 32])

    # test reset
    model.clear_all()
    assert_equal(model.use_mrk, [0, 1, 2, 3, 4])
    assert_equal(model.sqd_file, "")


@requires_mayavi
def test_kit2fiff_gui():
    """Test Kit2Fiff GUI."""
    if os.environ.get('TRAVIS_OS_NAME') == 'linux':
        raise SkipTest("Skipping on Travis for Linux due to GUI error")
    home_dir = _TempDir()
    os.environ['_MNE_GUI_TESTING_MODE'] = 'true'
    os.environ['_MNE_FAKE_HOME_DIR'] = home_dir
    try:
        with warnings.catch_warnings(record=True):  # traits warnings
            warnings.simplefilter('always')
            ui, frame = mne.gui.kit2fiff()
            assert_false(frame.model.can_save)
            assert_equal(frame.model.stim_threshold, 1.)
            frame.model.stim_threshold = 10.
            frame.model.stim_chs = 'save this!'
            # ui.dispose() should close the Traits-UI, but it opens modal
            # dialogs which interrupt the tests. This workaround triggers
            # saving of configurations without closing the window:
            frame.save_config(home_dir)

            # test setting persistence
            ui, frame = mne.gui.kit2fiff()
            assert_equal(frame.model.stim_threshold, 10.)
            assert_equal(frame.model.stim_chs, 'save this!')
    finally:
        del os.environ['_MNE_GUI_TESTING_MODE']
        del os.environ['_MNE_FAKE_HOME_DIR']


run_tests_if_main()
