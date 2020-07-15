# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

import mne
from mne.io.kit.tests import data_dir as kit_data_dir
from mne.io import read_raw_fif
from mne.utils import (requires_mayavi, run_tests_if_main, traits_test,
                       modified_env)

mrk_pre_path = os.path.join(kit_data_dir, 'test_mrk_pre.sqd')
mrk_post_path = os.path.join(kit_data_dir, 'test_mrk_post.sqd')
sqd_path = os.path.join(kit_data_dir, 'test.sqd')
hsp_path = os.path.join(kit_data_dir, 'test_hsp.txt')
fid_path = os.path.join(kit_data_dir, 'test_elp.txt')
fif_path = os.path.join(kit_data_dir, 'test_bin_raw.fif')


@requires_mayavi
@traits_test
def test_kit2fiff_model(tmpdir):
    """Test Kit2Fiff model."""
    from mne.gui._kit2fiff_gui import Kit2FiffModel
    tempdir = str(tmpdir)
    tgt_fname = os.path.join(tempdir, 'test-raw.fif')

    model = Kit2FiffModel()
    assert not model.can_save
    assert model.misc_chs_desc == "No SQD file selected..."
    assert model.stim_chs_comment == ""
    model.markers.mrk1.file = mrk_pre_path
    model.markers.mrk2.file = mrk_post_path
    model.sqd_file = sqd_path
    assert model.misc_chs_desc == "160:192"
    model.hsp_file = hsp_path
    assert not model.can_save
    model.fid_file = fid_path
    assert model.can_save

    # events
    model.stim_slope = '+'
    assert model.get_event_info() == {1: 2}
    model.stim_slope = '-'
    assert model.get_event_info() == {254: 2, 255: 2}

    # stim channels
    model.stim_chs = "181:184, 186"
    assert_array_equal(model.stim_chs_array, [181, 182, 183, 186])
    assert model.stim_chs_ok
    assert model.get_event_info() == {}
    model.stim_chs = "181:184, bad"
    assert not model.stim_chs_ok
    assert not model.can_save
    model.stim_chs = ""
    assert model.can_save

    # export raw
    raw_out = model.get_raw()
    raw_out.save(tgt_fname)
    raw = read_raw_fif(tgt_fname)

    # Compare exported raw with the original binary conversion
    raw_bin = read_raw_fif(fif_path)
    trans_bin = raw.info['dev_head_t']['trans']
    want_keys = list(raw_bin.info.keys())
    assert sorted(want_keys) == sorted(list(raw.info.keys()))
    trans_transform = raw_bin.info['dev_head_t']['trans']
    assert_allclose(trans_transform, trans_bin, 0.1)

    # Averaging markers
    model.markers.mrk3.method = "Average"
    trans_avg = model.dev_head_trans
    assert not np.all(trans_avg == trans_transform)
    assert_allclose(trans_avg, trans_bin, 0.1)

    # Test exclusion of one marker
    model.markers.mrk3.method = "Transform"
    model.use_mrk = [1, 2, 3, 4]
    assert not np.all(model.dev_head_trans == trans_transform)
    assert not np.all(model.dev_head_trans == trans_avg)
    assert not np.all(model.dev_head_trans == np.eye(4))

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
    assert model.use_mrk == [0, 1, 2, 3, 4]
    assert model.sqd_file == ""


@requires_mayavi
@traits_test
def test_kit2fiff_gui(check_gui_ci, tmpdir):
    """Test Kit2Fiff GUI."""
    home_dir = str(tmpdir)
    with modified_env(_MNE_GUI_TESTING_MODE='true',
                      _MNE_FAKE_HOME_DIR=home_dir):
        from pyface.api import GUI
        gui = GUI()
        gui.process_events()

        ui, frame = mne.gui.kit2fiff()
        assert not frame.model.can_save
        assert frame.model.stim_threshold == 1.
        frame.model.stim_threshold = 10.
        frame.model.stim_chs = 'save this!'
        frame.save_config(home_dir)
        ui.dispose()

        gui.process_events()

        # test setting persistence
        ui, frame = mne.gui.kit2fiff()
        assert frame.model.stim_threshold == 10.
        assert frame.model.stim_chs == 'save this!'

        # set and reset marker file
        points = [[-0.084612, 0.021582, -0.056144],
                  [0.080425, 0.021995, -0.061171],
                  [-0.000787, 0.105530, 0.014168],
                  [-0.047943, 0.091835, 0.010240],
                  [0.042976, 0.094380, 0.010807]]
        assert_array_equal(frame.marker_panel.mrk1_obj.points, 0)
        assert_array_equal(frame.marker_panel.mrk3_obj.points, 0)
        frame.model.markers.mrk1.file = mrk_pre_path
        assert_allclose(frame.marker_panel.mrk1_obj.points, points, atol=1e-6)
        assert_allclose(frame.marker_panel.mrk3_obj.points, points, atol=1e-6)
        frame.marker_panel.mrk1_obj.label = True
        frame.marker_panel.mrk1_obj.label = False
        frame.kit2fiff_panel.clear_all = True
        assert_array_equal(frame.marker_panel.mrk1_obj.points, 0)
        assert_array_equal(frame.marker_panel.mrk3_obj.points, 0)
        ui.dispose()

        gui.process_events()


run_tests_if_main()
