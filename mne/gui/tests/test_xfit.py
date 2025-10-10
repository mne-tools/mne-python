# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.


import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

import mne
from mne.channels import read_vectorview_selection
from mne.datasets import testing
from mne.viz import ui_events
from mne.viz.utils import _get_color_list

data_path = testing.data_path(download=False)
subjects_dir = data_path / "subjects"
fname_dip = data_path / "MEG" / "sample" / "sample_audvis_trunc_set1.dip"
fname_evokeds = data_path / "MEG" / "sample" / "sample_audvis_trunc-ave.fif"


def _gui_with_two_dipoles():
    """Create a dipolefit GUI and add two dipoles to it."""
    from mne.gui import dipolefit

    g = dipolefit(fname_evokeds)
    dip = mne.read_dipole(fname_dip)[[0, 1]]
    g.add_dipole(dip, name=["rh1", "rh2"])
    return g


@pytest.mark.slowtest
@testing.requires_testing_data
def test_dipolefit_gui_basic(renderer_interactive_pyvistaqt):
    """Test basic functionality of the dipole fitting GUI."""
    from mne.gui import dipolefit

    # Test basic interface elements.
    g = dipolefit(fname_evokeds)
    assert g._evoked.comment == "Left Auditory"  # MNE-Sample data should be loaded
    assert g._current_time == g._evoked.times[84]  # time of max GFP

    # Test fitting a single dipole.
    assert len(g._dipoles) == len(g.dipoles) == 0
    g._on_fit_dipole()
    assert len(g._dipoles) == len(g.dipoles) == 1
    dip = g.dipoles[0]
    assert dip.name == "Left Auditory"
    assert len(dip.times) == 1
    assert_equal(dip.times, g._current_time)
    assert_allclose(dip.amplitude, 6.152221e-08)
    assert_allclose(dip.pos, [[0.04568744, 0.00753845, 0.06737837]], atol=1e-7)
    assert_allclose(dip.ori, [[0.45720003, -0.72124413, -0.52036049]], atol=1e-7)
    old_dip1_timecourse = g._dipoles[0]["timecourse"]

    # Test fitting a second dipole with a subset of channels at a different time.
    g._on_sensor_data()  # open sensor selection window
    picks = read_vectorview_selection("Left", info=g._evoked.info)
    ui_events.publish(g._fig_sensors, ui_events.ChannelsSelect(picks))
    assert sorted(g._fig_sensors.lasso.selection) == sorted(picks)
    ui_events.publish(g._fig, ui_events.TimeChange(0.1))  # change time
    assert g._current_time == 0.1
    g._on_fit_dipole()
    assert len(g._dipoles) == len(g.dipoles) == 2
    dip2 = g.dipoles[1]
    assert_equal(dip2.times, g._evoked.times[np.searchsorted(g._evoked.times, 0.1) - 1])
    assert_allclose(dip2.amplitude, 4.422736e-08)
    assert_allclose(dip2.pos, [[-0.05893074, -0.00202937, 0.05113064]], atol=1e-7)
    assert_allclose(dip2.ori, [[0.3017588, -0.88550684, -0.35329769]], atol=1e-7)
    # Adding the second dipole should have affected the timecourse of the first.
    new_dip1_timecourse = g._dipoles[0]["timecourse"]
    assert not np.allclose(old_dip1_timecourse, new_dip1_timecourse)

    # Test differences between the two dipoles
    assert list(g._dipoles.keys()) == [0, 1]
    dip1_dict, dip2_dict = g._dipoles.values()
    assert dip1_dict["dip"] is dip
    assert dip2_dict["dip"] is dip2
    assert dip1_dict["num"] == 0
    assert dip2_dict["num"] == 1
    assert_allclose(
        dip1_dict["helmet_pos"], [0.10320071, 0.00946581, 0.07516293], atol=1e-7
    )
    assert_allclose(
        dip2_dict["helmet_pos"], [-0.11462019, -0.00727073, 0.04561434], atol=1e-7
    )
    assert dip1_dict["color"] == _get_color_list()[0]
    assert dip2_dict["color"] == _get_color_list()[1]

    # Test changing dipole model
    assert g._multi_dipole_method == "Multi dipole (MNE)"
    old_timecourses = np.vstack((dip1_dict["timecourse"], dip2_dict["timecourse"]))
    g._on_select_method("Single dipole")
    new_timecourses = np.vstack((dip1_dict["timecourse"], dip2_dict["timecourse"]))
    assert not np.allclose(old_timecourses, new_timecourses)
    g._fig._renderer.close()


@pytest.mark.slowtest
@testing.requires_testing_data
def test_dipolefit_gui_toggle_meshes(renderer_interactive_pyvistaqt):
    """Test toggling the visibility of the meshes the dipole fitting GUI."""
    from mne.gui import dipolefit

    g = dipolefit(fname_evokeds)
    assert list(g._actors.keys()) == ["helmet", "occlusion_surf", "head", "sensors"]
    g.toggle_mesh("helmet", show=True)
    assert g._actors["helmet"].visibility
    g.toggle_mesh("helmet")
    assert not g._actors["helmet"].visibility
    with pytest.raises(ValueError, match="Invalid value for the 'name' parameter"):
        g.toggle_mesh("non existent")
    g._fig._renderer.close()


@pytest.mark.slowtest
@testing.requires_testing_data
def test_dipolefit_gui_dipole_controls(renderer_interactive_pyvistaqt):
    """Test the controls for the dipoles in the dipole fitting GUI."""
    g = _gui_with_two_dipoles()

    dip1, dip2 = g._dipoles.values()
    assert dip1["active"] and dip2["active"]
    old_timecourses = np.vstack((dip1["timecourse"], dip2["timecourse"]))

    # Toggle a dipole off and on.
    g._on_dipole_toggle(False, dip2["num"])
    assert not dip2["active"]
    new_timecourses = np.vstack((dip1["timecourse"], dip2["timecourse"]))
    assert not np.allclose(old_timecourses, new_timecourses, atol=1e-9)
    g._on_dipole_toggle(True, dip2["num"])
    assert dip2["active"]
    new_timecourses = np.vstack((dip1["timecourse"], dip2["timecourse"]))
    assert np.allclose(old_timecourses, new_timecourses, atol=0)

    # Toggle fixed orientation off and on.
    assert dip1["fix_ori"] and dip2["fix_ori"]
    g._on_dipole_toggle_fix_orientation(False, dip1["num"])
    assert not dip1["fix_ori"]
    new_timecourses = np.vstack((dip1["timecourse"], dip2["timecourse"]))
    assert not np.allclose(old_timecourses, new_timecourses, atol=1e-9)
    g._on_dipole_toggle_fix_orientation(True, dip1["num"])
    assert dip1["fix_ori"]
    new_timecourses = np.vstack((dip1["timecourse"], dip2["timecourse"]))
    assert np.allclose(old_timecourses, new_timecourses, atol=0)

    # Change the names.
    g._on_dipole_set_name("dipole1", dip1["num"])
    g._on_dipole_set_name("dipole2", dip2["num"])
    assert dip1["dip"].name == "dipole1"
    assert dip2["dip"].name == "dipole2"
    assert dip1["line_artist"].get_label() == "dipole1"  # legend labels
    assert dip2["line_artist"].get_label() == "dipole2"

    # Remove a dipole.
    g._on_dipole_delete(dip1["num"])
    assert len(g.dipoles) == 1
    assert 1 in g._dipoles  # dipole number should not change
    assert list(g._dipoles.keys())[0] == 1
    assert list(g._dipoles.values())[0]["num"] == 1
    g._on_fit_dipole()
    assert 2 in g._dipoles
    assert list(g._dipoles.keys())[1] == 2
    assert list(g._dipoles.values())[1]["num"] == 2  # new dipole number
    g._fig._renderer.close()


@pytest.mark.slowtest
@testing.requires_testing_data
def test_dipolefit_gui_save_load(tmpdir, renderer_interactive_pyvistaqt):
    """Test saving and loading dipoles in the dipole fitting GUI."""
    g = _gui_with_two_dipoles()
    g.save(tmpdir / "test.dip")
    g.save(tmpdir / "test.bdip")
    dip_from_file = mne.read_dipole(tmpdir / "test.dip")
    g.add_dipole(dip_from_file)
    g.add_dipole(mne.read_dipole(tmpdir / "test.bdip"))
    assert len(g.dipoles) == 6
    assert [d.name for d in g.dipoles] == ["rh1", "rh2", "rh1", "rh2", "dip4", "dip5"]
    assert_allclose(
        np.vstack([d.pos for d in g.dipoles[:2]]), dip_from_file.pos, atol=0
    )
    assert_allclose(
        np.vstack([d.pos for d in g.dipoles[2:4]]), dip_from_file.pos, atol=0
    )
    assert_allclose(
        np.vstack([d.pos for d in g.dipoles[4:]]), dip_from_file.pos, atol=0
    )
    g._fig._renderer.close()
