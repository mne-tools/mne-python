# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import re

import numpy as np
import pytest
from matplotlib.colors import to_hex
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
fname_trans = data_path / "MEG" / "sample" / "sample_audvis_trunc-trans.fif"
fname_cov = data_path / "MEG" / "sample" / "sample_audvis_trunc-cov.fif"
fname_stc = data_path / "MEG" / "sample" / "sample_audvis_trunc-meg"
fname_bem_sol = subjects_dir / "sample" / "bem" / "sample-320-320-320-bem-sol.fif"


@pytest.fixture(scope="module")
def _sample_evoked():
    """Read the evoked data (module scoped, do not modify: use ``sample_evoked``)."""
    return mne.read_evokeds(fname_evokeds, condition=0)


@pytest.fixture
def sample_evoked(_sample_evoked):
    """Get the evoked data used throughout these tests."""
    return _sample_evoked.copy()


@pytest.fixture(scope="module")
def surf_maps_meg(_sample_evoked):
    """Compute the MEG-only field map (as ``dipolefit`` would without a trans)."""
    return mne.make_field_map(_sample_evoked, trans=None, origin="auto", verbose=False)


@pytest.fixture(scope="module")
def surf_maps_eeg_meg(_sample_evoked):
    """Compute both the EEG and MEG field maps (needs a head<->MRI transform)."""
    return mne.make_field_map(
        _sample_evoked,
        trans=fname_trans,
        origin="auto",
        subject="sample",
        subjects_dir=subjects_dir,
        verbose=False,
    )


def _selected_sensors(g):
    names = []
    for actor in g._actors["sensors"]:
        cloud = actor.GetMapper().GetInput()
        # color is hardcoded for now, so changeable
        green = (cloud.point_data["colors"] == [0, 255, 0, 100]).all(axis=1)
        names.extend(cloud.field_data["ch_names"][green])
    return sorted(names)


@pytest.mark.slowtest
@testing.requires_testing_data
def test_dipolefit_gui_basic(
    tmp_path, sample_evoked, surf_maps_meg, renderer_interactive_pyvistaqt
):
    """Test basic functionality of the dipole fitting GUI."""
    from mne.gui import dipolefit
    from mne.gui._dipolefit import _STATUS_IDLE

    # Test basic interface elements.
    evoked = sample_evoked
    data_before = evoked.data.copy()
    g = dipolefit(evoked, baseline=(None, 0), surf_maps=surf_maps_meg)

    assert evoked.comment == "Left Auditory"  # MNE-Sample data should be loaded
    assert_allclose(evoked.data, data_before, atol=0)  # input is not modified
    assert g._evoked.baseline == (evoked.times[0], 0)  # baseline applied to a copy
    assert g._current_time == evoked.times[84]  # time of max GFP

    # The sensors consist of multiple actors, toggling them should affect all of them.
    g.toggle_mesh("sensors", show=False)
    assert not g._actors["sensors"][0].GetVisibility()
    g.toggle_mesh("sensors")  # show=None toggles the current visibility
    assert g._actors["sensors"][0].GetVisibility()

    # The GUI starts out idle, with the splash screen (if any: in testing mode
    # `show=False`, so there was none) closed and forgotten by `_qt_safe_window`.
    assert g._status_label.get_value() == _STATUS_IDLE
    assert not hasattr(g, "_splash")

    # Slow operations are announced in the status bar and make the GUI
    # un-interactable. Nested uses of `_busy` collapse into the outermost one.
    window = g._renderer._window
    assert window.isEnabled()
    cursor = g._renderer._window_get_cursor().shape()
    with g._busy("Busy..."):
        assert g._status_label.get_value() == "Busy..."
        assert not window.isEnabled()
        assert g._renderer._window_get_cursor().shape() != cursor  # busy cursor
        with g._busy("Nested..."):
            assert g._status_label.get_value() == "Busy..."  # the outermost one wins
        assert not window.isEnabled()  # only the outermost one restores the GUI
    assert g._status_label.get_value() == _STATUS_IDLE
    assert window.isEnabled()
    assert g._renderer._window_get_cursor().shape() == cursor

    # An event handler that runs while `_busy` paints the busy state (it processes
    # events once) must see itself as nested, not tear the busy state down.
    orig_process = g._renderer._process_events
    reentered = list()

    def process_and_reenter():
        orig_process()
        if not reentered:
            reentered.append(True)
            with g._busy("Nested during repaint..."):
                pass

    g._renderer._process_events = process_and_reenter
    try:
        with g._busy("Busy..."):
            assert reentered
            assert g._status_label.get_value() == "Busy..."
            assert not window.isEnabled()
    finally:
        g._renderer._process_events = orig_process
    assert g._status_label.get_value() == _STATUS_IDLE
    assert window.isEnabled()
    assert g._renderer._window_get_cursor().shape() == cursor

    # Test fitting a single dipole.
    assert len(g._dipoles) == len(g.dipoles) == 0
    g.fit_dipole()
    assert g._renderer._window_get_cursor().shape() == cursor  # busy cursor restored
    assert g._status_label.get_value() == _STATUS_IDLE
    assert len(g._dipoles) == len(g.dipoles) == 1
    dip = g.dipoles[0]
    assert dip.name == "Left Auditory"
    assert len(dip.times) == 1
    assert_equal(dip.times, g._current_time)
    old_dip1_timecourse = g._dipoles[0]["timecourse"]

    # Check the position of the fitted dipole against the pre-computed dipole in the
    # testing dataset. The pre-computed dipole only needs to give us the general area
    # in which we expect the fitted dipole and does not have to be a perfect match.
    ref_dip = mne.read_dipole(fname_dip)
    ref_dip_t = ref_dip[[np.argmin(np.abs(ref_dip.times - g._current_time))]]
    assert_allclose(dip.pos, ref_dip_t.pos, atol=0.012)  # somewhat near the reference

    # Test fitting a second dipole with a subset of channels at a different time.
    g._on_sensor_data()  # open sensor selection window
    g._on_sensor_data()  # already open, so this is a no-op
    picks = read_vectorview_selection("Left", info=evoked.info)
    ui_events.publish(g._fig_sensors, ui_events.ChannelsSelect(picks))
    assert sorted(g._fig_sensors.lasso.selection) == sorted(picks)
    assert _selected_sensors(g) == sorted(picks)
    ui_events.publish(g._fig, ui_events.TimeChange(0.09))  # change time
    assert g._current_time == 0.09

    # The time (and the goodness-of-fit, once there are dipoles) is labeled on the time
    # line of the traces plot, not in the 3D view.
    assert g._fig._time_label is None
    assert not hasattr(g._fig, "_time_label_actor")
    assert re.fullmatch(r"90 ms · GOF \d+%", g._time_text.get_text())
    assert g._time_text.get_position()[0] == 0.09
    # both move with the time, so they are drawn on top of a cached background
    # rather than triggering a full redraw of the traces plot
    blit_artists = g._renderer._mplcanvas._blit_artists
    assert blit_artists == [g._time_line, g._time_text]

    g.fit_dipole()
    assert len(g._dipoles) == len(g.dipoles) == 2
    dip2 = g.dipoles[1]

    # During tests, matplotlib does not open an actual window so we need to force the
    # close event.
    g._fig_sensors.canvas.callbacks.process("close_event", None)
    assert _selected_sensors(g) == []

    # The selected time of 0.09 is not actually in evoked.times, find the closest value
    # that is (0.08990784...). That should be the time recorded in the dipole object.
    closest_time = evoked.times[np.argmin(np.abs(evoked.times - g._current_time))]
    assert dip2.times[0] == closest_time

    # Check that the general area of the second dipole is now in the left hemisphere.
    ref_dip_t = ref_dip[[np.argmin(np.abs(ref_dip.times - g._current_time))]]
    assert_allclose(dip2.pos, ref_dip_t.pos, atol=0.012)  # somewhat near the reference

    # Adding the second dipole should have affected the timecourse of the first.
    new_dip1_timecourse = g._dipoles[0]["timecourse"]
    assert not np.allclose(old_dip1_timecourse, new_dip1_timecourse, atol=1e-10)

    # Test differences between the two dipoles
    assert list(g._dipoles.keys()) == [0, 1]
    dip1_dict, dip2_dict = g._dipoles.values()
    assert dip1_dict["dip"] is dip
    assert dip2_dict["dip"] is dip2
    assert dip1_dict["num"] == 0
    assert dip2_dict["num"] == 1
    assert dip1_dict["color"] == _get_color_list()[0]
    assert dip2_dict["color"] == _get_color_list()[1]

    # The name field of each dipole is styled with the color of its trace.
    for dip_dict in (dip1_dict, dip2_dict):
        style = dip_dict["widgets"][1].widget.styleSheet()
        assert to_hex(dip_dict["color"]) in style
        assert "color:black;" in style  # both colors are light enough for black text

    # Timecourses are stored in Am, but displayed in nAm, with the goodness-of-fit of
    # the combined model shown on a twin axis.
    for dip_dict in (dip1_dict, dip2_dict):
        assert_allclose(
            dip_dict["line_artist"].get_ydata(), dip_dict["timecourse"] * 1e9, atol=0
        )
    assert g._gof_ax.get_ylim() == (0, 100)
    assert g._gof_line.get_ydata().max() <= 100

    # Fitted dipoles have goodness-of-fit information that should be saved along.
    fname = tmp_path / "fitted.dip"
    g.save(fname)
    assert mne.read_dipole(fname).khi2 is not None

    # Test changing the dipole model through the dropdown widget (like a user would).
    # The status bar should name the model that is being fitted.
    messages = list()
    orig_set_status = g._set_status

    def record_status(message=_STATUS_IDLE):
        messages.append(message)
        orig_set_status(message)

    g._set_status = record_status
    assert g._multi_dipole_method == "Multi dipole (MNE)"
    old_timecourses = np.vstack((dip1_dict["timecourse"], dip2_dict["timecourse"]))
    g._method_combo.set_value("Single dipole")
    assert g._multi_dipole_method == "Single dipole"
    # The refit is deferred to the event loop so that the combo box popup can close
    # and repaint before the slow computation starts.
    assert g._refit_pending
    assert messages == []
    g._renderer._process_events()  # run the deferred refit
    assert not g._refit_pending
    assert "Fitting Single dipole model..." in messages
    new_timecourses = np.vstack((dip1_dict["timecourse"], dip2_dict["timecourse"]))
    assert not np.allclose(old_timecourses, new_timecourses, atol=1e-10)

    # Selecting the method that is already active does not pointlessly refit.
    messages.clear()
    g._on_select_method("Single dipole")
    assert not g._refit_pending
    assert messages == []
    with pytest.raises(ValueError, match="Invalid value for the 'method'"):
        g._on_select_method("foo")

    # Switching back refits (and reproduces) the multi-dipole model.
    g._method_combo.set_value("Multi dipole (MNE)")
    g._renderer._process_events()
    assert "Fitting Multi dipole (MNE) model..." in messages
    roundtrip = np.vstack((dip1_dict["timecourse"], dip2_dict["timecourse"]))
    assert np.allclose(roundtrip, old_timecourses, atol=0)
    g._set_status = orig_set_status

    g.close()


@pytest.mark.slowtest
@testing.requires_testing_data
def test_dipolefit_gui_dipole_controls(
    sample_evoked, surf_maps_meg, renderer_interactive_pyvistaqt
):
    """Test the controls for the dipoles in the dipole fitting GUI."""
    from mne.gui import dipolefit

    evoked = sample_evoked
    g = dipolefit(evoked, surf_maps=surf_maps_meg, show_sensors=False)

    # Test toggling the visibility of the meshes.
    assert list(g._actors.keys()) == ["helmet", "occlusion_surf", "head"]
    g.toggle_mesh("helmet", show=True)
    assert g._actors["helmet"].visibility
    g.toggle_mesh("helmet")
    assert not g._actors["helmet"].visibility
    with pytest.raises(ValueError, match="Invalid value for the 'name' parameter"):
        g.toggle_mesh("non existent")

    # Each mesh also gets an opacity slider, initialized to its current opacity. The
    # head surface is drawn translucent (see `_plot_head_surface`).
    assert_allclose(g._get_mesh_opacity("head"), 0.2, atol=0)
    g._mesh_widgets["head"][1].set_value(0.4)  # [checkbox, opacity slider]
    assert_allclose(g._actors["head"].GetProperty().GetOpacity(), 0.4, atol=1e-4)

    # Camera presets.
    g._set_camera_preset("Top")
    with pytest.raises(ValueError, match="Invalid value for the 'name' parameter"):
        g._set_camera_preset("Sideways")

    # Test toggling dipoles off and on. This is done through the GUI widgets, which are
    # ordered: [active, name, fix orientation, delete].
    dip = mne.read_dipole(fname_dip)[[12, 15]]  # 80ms and 90ms
    g.add_dipole(dip, name=["rh", "lh"])
    dip1, dip2 = g._dipoles.values()
    assert dip1["active"] and dip2["active"]

    # Each trace is marked with a dot at the time the dipole was fitted, and hovering
    # the dipole's row in the GUI emphasizes both.
    for dip_dict in (dip1, dip2):
        assert dip_dict["dot_artist"].get_xdata() == [dip_dict["dip"].times[0]]
        assert_allclose(
            dip_dict["dot_artist"].get_ydata(),
            np.interp(
                dip_dict["dip"].times[0],
                evoked.times,
                dip_dict["line_artist"].get_ydata(),
            ),
            atol=0,
        )
    from qtpy.QtCore import QEvent
    from qtpy.QtWidgets import QApplication

    lw, ms = dip1["line_artist"].get_linewidth(), dip1["dot_artist"].get_markersize()
    # Hover the actual Qt widget (the dipole's name field), so that the enter/leave
    # event filter is exercised as well.
    name_widget = dip1["widgets"][1]._widget
    QApplication.sendEvent(name_widget, QEvent(QEvent.Type.Enter))
    assert dip1["line_artist"].get_linewidth() > lw
    assert dip1["dot_artist"].get_markersize() > ms
    QApplication.sendEvent(name_widget, QEvent(QEvent.Type.Leave))
    assert dip1["line_artist"].get_linewidth() == lw
    assert dip1["dot_artist"].get_markersize() == ms
    g._on_dipole_hover(99, True)  # deleted dipole: no-op rather than an error
    old_timecourses = np.vstack((dip1["timecourse"], dip2["timecourse"]))
    dip2["widgets"][0].set_value(False)
    assert not dip2["active"]
    new_timecourses = np.vstack((dip1["timecourse"], dip2["timecourse"]))
    assert not np.allclose(old_timecourses, new_timecourses, atol=1e-9)

    # With all dipoles disabled, there is nothing to fit and no arrows to update.
    dip1["widgets"][0].set_value(False)
    assert g.dipoles == []
    g.set_time(0.05)
    assert g._current_time == 0.05

    dip1["widgets"][0].set_value(True)
    dip2["widgets"][0].set_value(True)
    assert dip1["active"] and dip2["active"]
    new_timecourses = np.vstack((dip1["timecourse"], dip2["timecourse"]))
    assert np.allclose(old_timecourses, new_timecourses, atol=0)

    # Toggle fixed orientation off and on.
    assert dip1["fix_ori"] and dip2["fix_ori"]
    dip1["widgets"][2].set_value(False)
    assert not dip1["fix_ori"]
    new_timecourses = np.vstack((dip1["timecourse"], dip2["timecourse"]))
    assert not np.allclose(old_timecourses, new_timecourses, atol=1e-9)
    dip1["widgets"][2].set_value(True)
    assert dip1["fix_ori"]
    new_timecourses = np.vstack((dip1["timecourse"], dip2["timecourse"]))
    assert np.allclose(old_timecourses, new_timecourses, atol=0)

    # Change the names of the dipoles.
    dip1["widgets"][1].set_value("dipole1")
    g._on_dipole_set_name("dipole2", dip2["num"])
    assert dip1["dip"].name == "dipole1"
    assert dip2["dip"].name == "dipole2"

    # Remove a dipole (through the "delete" button).
    line, dot = dip1["line_artist"], dip1["dot_artist"]
    dip1["widgets"][3].set_value(None)
    assert line not in g._renderer._mplcanvas.axes.lines
    assert dot not in g._renderer._mplcanvas.axes.lines
    assert len(g.dipoles) == 1
    assert 1 in g._dipoles  # dipole number should not change
    assert list(g._dipoles.keys())[0] == 1
    assert list(g._dipoles.values())[0]["num"] == 1
    g.fit_dipole()
    assert 2 in g._dipoles
    assert list(g._dipoles.keys())[1] == 2
    assert list(g._dipoles.values())[1]["num"] == 2  # new dipole number

    # Fitting the timecourse of a single dipole, with a free orientation.
    g._on_dipole_toggle(False, 2)  # only leave a single dipole active
    g._on_select_method("Single dipole")
    g._renderer._process_events()  # run the deferred refit
    assert dip2["fix_ori"]
    assert_allclose(dip2["orientation"], dip2["dip"].ori.repeat(len(evoked.times), 0))
    g._on_dipole_toggle_fix_orientation(False, dip2["num"])
    assert not dip2["fix_ori"]
    assert dip2["orientation"].shape == (len(evoked.times), 3)
    assert not np.allclose(dip2["orientation"][0], dip2["orientation"][-1], atol=1e-9)

    g.close()


@pytest.mark.slowtest
@testing.requires_testing_data
def test_dipolefit_gui_save_load(
    tmp_path, sample_evoked, renderer_interactive_pyvistaqt
):
    """Test saving and loading dipoles in the dipole fitting GUI."""
    from mne.gui import dipolefit

    # Not passing `surf_maps` means they are computed on the fly.
    g = dipolefit(sample_evoked, show_sensors=False)
    dip = mne.read_dipole(fname_dip)[[12, 15]]  # 80ms and 90ms
    g.add_dipole(dip, name=["rh", "lh"])

    g.save(tmp_path / "test.dip")
    g.save(tmp_path / "test.bdip")
    dip_from_file = mne.read_dipole(tmp_path / "test.dip")
    g.add_dipole(dip_from_file)  # names are taken from the ";" separated dip.name
    g.add_dipole(mne.read_dipole(tmp_path / "test.bdip"))  # bdip stores no names
    assert len(g.dipoles) == 6
    assert [d.name for d in g.dipoles] == ["rh", "lh", "rh", "lh", "dip4", "dip5"]
    for start in [0, 2, 4]:
        assert_allclose(
            np.vstack([d.pos for d in g.dipoles[start : start + 2]]),
            dip_from_file.pos,
            atol=0,
        )

    # A single dipole can be given a name directly. When the name of the `Dipole` object
    # cannot be split into one name per dipole, it is used for all of them.
    assert dip_from_file.name == "rh;lh"  # cannot be split into a single name
    g.add_dipole(dip_from_file[[0]], name="single")
    g.add_dipole(dip_from_file[[1]])
    assert [d.name for d in g.dipoles[6:]] == ["single", "rh;lh"]

    with pytest.raises(ValueError, match="Number of names"):
        g.add_dipole(dip_from_file, name=["too", "many", "names"])

    g.close()


@pytest.mark.slowtest
@testing.requires_testing_data
def test_dipolefit_params(
    tmp_path, sample_evoked, surf_maps_eeg_meg, renderer_interactive_pyvistaqt
):
    """Test setting various parameters in the dipole fitting GUI."""
    from mne.gui import dipolefit

    # Test different type of covariance estimators.
    evoked = sample_evoked

    g = dipolefit(
        evoked, surf_maps=surf_maps_eeg_meg, cov=None, show_sensors=False
    )  # ad-hoc
    assert g._cov["diag"]
    assert_allclose(  # default ad-hoc variation for grads, mags and eeg
        g._cov["data"][[0, 1, 2, 306]], [2.5e-25, 2.5e-25, 4e-28, 4e-14], atol=0
    )

    # cov="baseline" needs baseline-corrected data (this evoked has baseline=None).
    with pytest.raises(ValueError, match='cov="baseline" requires'):
        dipolefit(evoked, surf_maps=surf_maps_eeg_meg, cov="baseline")
    g = dipolefit(
        evoked,
        baseline=(None, 0),
        surf_maps=surf_maps_eeg_meg,
        cov="baseline",
        show_sensors=False,
    )
    assert_allclose(  # compute var on baseline period
        g._cov["data"][[0, 1, 2, 306]],
        [3.5e-24, 3.5e-24, 3.0e-27, 2.3e-12],
        rtol=0.1,
        atol=0,
    )

    # The following tests are rolled into one call to `dipolefit` in order to save time.
    #  - Specify a channel type
    #  - Specify custom covariance.
    #  - Specify BEM model.
    #  - Specify an initial time.
    cov = mne.read_cov(fname_cov)
    bem = mne.make_sphere_model(r0=(0.0, 0.0, 0.04), verbose=False)
    initial_time = 0.0123
    eeg_maps = [m for m in surf_maps_eeg_meg if m["kind"] == "eeg"]
    g = dipolefit(
        evoked,
        ch_type="eeg",
        surf_maps=eeg_maps,
        cov=cov,
        bem=bem,
        initial_time=initial_time,
        show_sensors=False,
    )
    assert set(g._evoked.get_channel_types()) == {"eeg"}
    assert_allclose(g._cov["data"], cov["data"], atol=0)
    assert_equal(g._bem["r0"], bem["r0"])
    assert g._current_time == initial_time

    # Without an MEG helmet, no arrows are drawn on it.
    g.add_dipole(mne.read_dipole(fname_dip)[[12]])
    (dipole,) = g._dipoles.values()
    assert dipole["helmet_coords"] is None
    assert dipole["arrow_mesh"] is None
    assert dipole["helmet_arrow_actor"] is None
    g._on_dipole_delete(dipole["num"])
    assert len(g.dipoles) == 0

    # Without any dipoles, there is nothing to save.
    g.save(tmp_path / "empty.dip")
    assert not (tmp_path / "empty.dip").exists()

    g.close()


@pytest.mark.slowtest
@testing.requires_testing_data
def test_dipolefit_stc(
    sample_evoked, surf_maps_eeg_meg, renderer_interactive_pyvistaqt
):
    """Test showing a SourceEstimate underneath the fieldlines."""
    from mne.gui import dipolefit

    evoked = sample_evoked

    # By default, the STC file has different timestamps from the evoked.
    with pytest.raises(ValueError, match="The time samples of the source estimate"):
        dipolefit(
            evoked, stc=fname_stc, surf_maps=surf_maps_eeg_meg, show_sensors=False
        )

    # Make the evoked timestamps line up with those of the STC.
    stc = mne.read_source_estimate(fname_stc)
    with pytest.warns():
        evoked = evoked.crop(0, 0.245).decimate(3)
    with evoked.info._unlock():
        evoked.info["sfreq"] = 100
    evoked._set_times(stc.times)

    # A source estimate needs a transform to be shown in the correct place.
    with pytest.raises(ValueError, match="`trans` cannot be `None`"):
        dipolefit(
            evoked,
            stc=stc,
            surf_maps=surf_maps_eeg_meg,
            show_sensors=False,
            baseline=(0, 0),
        )

    # Now it should work. Passing `bem` as a path loads the BEM solution file.
    g = dipolefit(
        evoked,
        stc=stc,
        trans=fname_trans,
        subject="sample",
        subjects_dir=subjects_dir,
        surf_maps=surf_maps_eeg_meg,
        bem=fname_bem_sol,
        show_sensors=False,
        baseline=(0, 0),
    )
    assert isinstance(g._stc, mne.SourceEstimate)
    assert not g._bem["is_sphere"]
    assert "solution" in g._bem

    # The cortex is drawn translucent by default
    assert g._actors["brain"][0].GetProperty().GetOpacity() == 0.5

    # The colorbar of the source estimate is registered as a "mesh" that can be toggled,
    # and starts out hidden as it takes up a lot of space.
    assert g._actors["colorbar"] == [
        g._stc_brain._scalar_bar,
        g._stc_brain._scalar_bar_ticks,
    ]
    assert not any(actor.GetVisibility() for actor in g._actors["colorbar"])
    assert len(g._mesh_widgets["colorbar"]) == 1  # checkbox only, no opacity slider
    g._mesh_widgets["colorbar"][0].set_value(True)
    assert all(actor.GetVisibility() for actor in g._actors["colorbar"])
    g.close()


@pytest.mark.slowtest
@testing.requires_testing_data
def test_dipolefit_gui_scraper(
    tmp_path, sample_evoked, surf_maps_meg, renderer_interactive_pyvistaqt
):
    """Test the scraper for the dipole fitting GUI."""
    pytest.importorskip("sphinx_gallery")
    from mne.gui import dipolefit
    from mne.viz.backends._pyvista import _ALL_PLOTTERS

    (tmp_path / "_images").mkdir()
    gallery_conf = dict(builder_name="html", src_dir=tmp_path)
    scraper = mne.gui._GUIScraper()

    # By default a GUI is scraped once and then closed.
    g = dipolefit(sample_evoked, surf_maps=surf_maps_meg, show_sensors=False)
    img = tmp_path / "_images" / "temp.png"
    block_vars = dict(example_globals=dict(gui=g), image_path_iterator=iter([str(img)]))
    assert not getattr(g, "_scraped", False)
    assert scraper(None, block_vars, gallery_conf)
    assert img.is_file()
    assert g._scraped
    assert g._renderer.plotter._closed
    assert scraper._preserved_guis == []
    assert scraper(None, block_vars, gallery_conf) == ""  # only scraped once

    # With ``# sphinx_gallery_preserve_gui = True`` it is scraped for every code block
    # and kept open until close_preserved() is called (from the doc build's
    # reset_modules).
    g = dipolefit(sample_evoked.copy(), surf_maps=surf_maps_meg, show_sensors=False)
    assert g._renderer.plotter._id_name in _ALL_PLOTTERS
    imgs = [tmp_path / "_images" / f"preserved{ii}.png" for ii in range(2)]
    block_vars = dict(
        example_globals=dict(gui=g),
        image_path_iterator=iter([str(img) for img in imgs]),
        file_conf=dict(preserve_gui=True),
    )
    for img in imgs:
        assert scraper(None, block_vars, gallery_conf)
        assert img.is_file()
        assert not g._renderer.plotter._closed
        # deregistered from the PyVista scraper, which would otherwise screenshot the
        # plotter a second time and then close it
        assert g._renderer.plotter._id_name not in _ALL_PLOTTERS
    assert scraper._preserved_guis == [g]
    scraper.close_preserved()
    assert scraper._preserved_guis == []
    assert g._renderer.plotter._closed


@pytest.mark.slowtest
@testing.requires_testing_data
def test_dipolefit_rapid_time_changes(
    sample_evoked, surf_maps_eeg_meg, renderer_interactive_pyvistaqt
):
    """Test that rapid time changes leave all linked views at the same time."""
    from qtpy.QtCore import QEvent, QObject
    from qtpy.QtWidgets import QApplication

    from mne.gui import dipolefit

    # Same stc-aligned configuration as test_dipolefit_stc.
    evoked = sample_evoked
    stc = mne.read_source_estimate(fname_stc)
    evoked = evoked.crop(0, 0.245).decimate(3, verbose="error")
    with evoked.info._unlock():
        evoked.info["sfreq"] = 100
    evoked._set_times(stc.times)
    g = dipolefit(
        evoked,
        stc=stc,
        trans=fname_trans,
        subject="sample",
        subjects_dir=subjects_dir,
        surf_maps=surf_maps_eeg_meg,
        show_sensors=False,
        baseline=(0, 0),
    )

    class _QueuedTimeChange(QEvent):
        def __init__(self, time):
            super().__init__(QEvent.Type.User)
            self.time = time

    class _Publisher(QObject):
        """Publish a TimeChange per Qt event, like rapid moves of the time slider."""

        def customEvent(self, event):
            # Publish on the brain figure so that the brain's handler (which
            # processes pending Qt events, and thereby the next queued
            # publication) runs before the other subscribers do.
            ui_events.publish(g._stc_brain, ui_events.TimeChange(time=event.time))

    # Queue up the publications as pending Qt events, then deliver them all, as
    # happens when the time slider is scrolled faster than the views can redraw.
    times = evoked.times[[10, 11, 12, 13]]
    app = QApplication.instance()
    publisher = _Publisher()
    for time in times:
        app.postEvent(publisher, _QueuedTimeChange(time))
    g._renderer._process_events()

    # Every view must end up at the last published time.
    assert g._stc_brain._current_time == times[-1]  # brain data + time line
    assert g._fig._current_time == times[-1]  # field lines
    assert g._current_time == times[-1]  # dipole arrows
    g.close()
