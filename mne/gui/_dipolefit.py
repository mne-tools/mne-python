# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import pyvista

from .._fiff.pick import pick_types
from ..bem import (
    ConductorModel,
    _ensure_bem_surfaces,
    make_sphere_model,
    read_bem_solution,
)
from ..cov import _ensure_cov, compute_whitener, make_ad_hoc_cov
from ..dipole import Dipole, fit_dipole
from ..evoked import Evoked
from ..forward import convert_forward_solution, make_field_map
from ..forward._make_forward import _ForwardModeler
from ..minimum_norm import apply_inverse, make_inverse_operator
from ..source_estimate import (
    SourceEstimate,
    _BaseSurfaceSourceEstimate,
    read_source_estimate,
)
from ..source_space import setup_volume_source_space
from ..surface import _normal_orth
from ..transforms import _get_trans, _get_transforms_to_coord_frame, apply_trans
from ..utils import (
    _auto_weakref,
    _check_option,
    _validate_type,
    fill_doc,
    logger,
    verbose,
)
from ..viz import EvokedField
from ..viz._3d import _get_3d_option, _plot_head_surface, _plot_sensors_3d
from ..viz.backends._utils import _qt_app_exec, _qt_safe_window, _splash_message
from ..viz.ui_events import ChannelsSelect, TimeChange, link, publish, subscribe
from ..viz.utils import _get_color_list, _is_dark

# Message shown in the status bar when the GUI is not busy doing something else.
_STATUS_IDLE = "Ready"
# Meshes that start out hidden (everything not listed here starts out visible).
_MESH_VISIBLE = dict(colorbar=False)
# Default alpha values for some of the meshes
_MESH_ALPHA = dict(brain=0.5)
# Meshes for which opacity cannot meaningfully be set (2D overlays).
_MESH_NO_OPACITY = ("colorbar",)
# Line width and marker size of the dipole traces, when not/when hovered.
_TRACE_LINEWIDTH, _TRACE_LINEWIDTH_HOVER = 1.5, 2.5
_TRACE_MARKERSIZE, _TRACE_MARKERSIZE_HOVER = 4, 7
# Standard views of the head, in the "mri" coordinate frame used by the 3D display.
_CAMERA_PRESETS = {
    "Left": dict(azimuth=180, elevation=90, roll=90),
    "Right": dict(azimuth=0, elevation=90, roll=270),
    "Front": dict(azimuth=90, elevation=90, roll=0),
    "Back": dict(azimuth=270, elevation=90, roll=180),
    "Top": dict(azimuth=90, elevation=0, roll=0),
}


@fill_doc
class DipoleFitUI:
    """GUI for interactive dipole fitting, inspired by MEGIN's XFit program.

    Parameters
    ----------
    evoked : instance of Evoked | path-like
        Evoked data to show fieldmap of and fit dipoles to.
    %(baseline_evoked)s
    cov : instance of Covariance | "baseline" | None
        Noise covariance matrix. If ``None``, an ad-hoc covariance matrix is used with
        default values for the diagonal elements (see Notes). If ``"baseline"``, the
        diagonal elements is estimated from the baseline period of the evoked data.
    bem : instance of ConductorModel | path-like | None
        Boundary element model to use in forward calculations, or a path to the BEM
        solution file (``"-bem-sol.fif"``) to read it from. If ``None``, a spherical
        model is used.
    initial_time : float | None
        Initial time point to show. If ``None``, the time point of the maximum field
        strength is used.
    trans : instance of Transform | None
        The transformation from head coordinates to MRI coordinates. If ``None``,
        the identity matrix is used and everything will be done in head coordinates.
    stc : instance of SourceEstimate | None
        An optional distributed source estimate to show alongside the fieldmap. The time
        samples need to match those of the evoked data.
    subject : str | None
        The subject name. If ``None``, no MRI data is shown.
    %(subjects_dir)s
    surf_maps : list | None
        The surface mapping information obtained with make_field_map. If ``None``, one
        will be generated based on the given data.
    %(rank)s
    show_density : bool
        Whether to show the density of the fieldmap.
    ch_type : "meg" | "eeg" | None
        Type of channels to use for the dipole fitting. By default (``None``) both MEG
        and EEG channels will be used.
    show_sensors : bool
        Whether to show the sensors in the 3D view.
    %(n_jobs)s
    show : bool
        Show the GUI if True.
    block : bool
        Whether to halt program execution until the figure is closed.
    %(verbose)s

    Attributes
    ----------
    dipoles : list of Dipole
        All currently enabled dipoles in the model.
    """

    @_qt_safe_window(splash="_splash", window="_init_renderer.figure.plotter")
    def __init__(
        self,
        evoked,
        *,
        baseline=None,
        cov=None,
        bem=None,
        initial_time=None,
        trans=None,
        stc=None,
        subject=None,
        subjects_dir=None,
        surf_maps=None,
        rank="info",
        show_density=True,
        ch_type=None,
        show_sensors=True,
        n_jobs=None,
        show=True,
        block=False,
        verbose=None,
    ):
        _validate_type(evoked, Evoked, "evoked")
        if baseline is not None:
            evoked = evoked.copy().apply_baseline(baseline)

        if cov is None:
            logger.info("Using ad-hoc noise covariance.")
            cov = make_ad_hoc_cov(evoked.info)
        elif cov == "baseline":
            if evoked.baseline is None:
                raise ValueError(
                    'cov="baseline" requires baseline-corrected data. Set the '
                    "baseline parameter or baseline-correct the evoked data first."
                )
            logger.info(
                f"Estimating noise covariance from baseline ({evoked.baseline[0]:.3f} "
                f"to {evoked.baseline[1]:.3f} seconds)."
            )
            std = dict()
            for typ in set(evoked.get_channel_types(only_data_chs=True)):
                baseline = evoked.copy().pick(typ).crop(*evoked.baseline)
                std[typ] = baseline.data.std(axis=1).mean()
            cov = make_ad_hoc_cov(evoked.info, std)
        else:
            cov = _ensure_cov(cov)

        _validate_type(bem, ("path-like", ConductorModel, None), "bem")
        if bem is None:
            bem = make_sphere_model("auto", "auto", evoked.info)
        elif not isinstance(bem, ConductorModel):
            # a path means a BEM solution file (cf. _make_forward._setup_bem)
            bem = read_bem_solution(bem)
        bem = _ensure_bem_surfaces(bem, extra_allow=(ConductorModel,))

        if ch_type is not None:
            evoked = evoked.copy().pick(ch_type)

        # Everything below is potentially slow, so bring up the window (still hidden)
        # and its splash screen first, and narrate the progress on it.
        self._busy_depth = 0
        self._busy_cursor = None
        self._refit_pending = False
        self._status_label = None
        self._configure_window(show=show)

        if surf_maps is None:
            self._set_status("Computing field maps...")
            surf_maps = make_field_map(
                evoked,
                trans=trans,
                origin=bem["r0"] if bem["is_sphere"] else "auto",
                subject=subject,
                subjects_dir=subjects_dir,
                n_jobs=n_jobs,
                verbose=verbose,
            )

        if initial_time is None:
            # Set initial time to moment of maximum field power.
            data = evoked.copy().pick(surf_maps[0]["ch_names"]).data
            initial_time = evoked.times[np.argmax(np.mean(data**2, axis=0))]

        if stc is not None:
            _validate_type(stc, ("path-like", _BaseSurfaceSourceEstimate), "stc")
            if not isinstance(stc, _BaseSurfaceSourceEstimate):
                self._set_status("Loading source estimate...")
                stc = read_source_estimate(stc)

            if len(stc.times) != len(evoked.times) or not np.allclose(
                stc.times, evoked.times
            ):
                raise ValueError(
                    "The time samples of the source estimate do not match those of the "
                    "evoked data."
                )
            if trans is None:
                raise ValueError(
                    "`trans` cannot be `None` when showing the fieldlines in "
                    "combination with a source estimate."
                )

        # Get transforms to convert all the various meshes to MRI space.
        head_mri_t = _get_trans(trans, "head", "mri")[0]
        to_cf_t = _get_transforms_to_coord_frame(
            evoked.info, head_mri_t, coord_frame="mri"
        )

        self._set_status("Preparing forward model...")
        self.fwd = _ForwardModeler(
            info=evoked.info,
            trans=trans,
            bem=bem,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        # Initialize all the private attributes.
        self._actors = dict()
        self._mesh_widgets = dict()
        self._bem = bem
        self._ch_type = ch_type
        self._cov = cov
        self._current_time = initial_time
        self._dipoles = dict()
        self._evoked = evoked
        self._helmet_surf = None
        self._surf_maps = surf_maps
        self._fig_sensors = None
        self._multi_dipole_method = "Multi dipole (MNE)"
        self._show_density = show_density
        self._stc = stc
        self._subjects_dir = subjects_dir
        self._subject = subject
        self._time_line = None
        self._time_text = None
        self._gof_ax = None
        self._gof_line = None
        self._head_mri_t = head_mri_t
        self._to_cf_t = to_cf_t
        self._rank = rank
        self._verbose = verbose
        self._n_jobs = n_jobs

        # Configure the GUI. The window stays hidden until it is fully composed:
        # `stc.plot` and `EvokedField` do not show figures they did not create
        # themselves (we pass ours in), so the window only appears at the `show()`
        # at the very end, all at once.
        self._configure_main_display(show_sensors=show_sensors)  # sets self._fig
        self._configure_dock()
        self._set_status()

        # must be done last
        if show:
            # Settle all pending widget layouts, still hidden and synchronously.
            self._renderer._window_settle_layouts()
            # Render the scene into the framebuffer now that the layouts (and hence
            # the 3D view size) are final: the first paint after showing blits
            # whatever the framebuffer holds, and for complex scenes the fresh
            # render in `show()` only completes after that first paint, which would
            # briefly show a stale, mis-framed image otherwise.
            for plotter in self._renderer._all_plotters:
                plotter._render()
            # Hand the splash screen back to the renderer, which closes it once the
            # window has actually appeared on screen (see `_qt_safe_window`).
            self._renderer.figure.splash = self._splash
            self._renderer.show()
        if block and self._renderer._kind != "notebook":
            _qt_app_exec(self._renderer.figure.store["app"])

    @property
    def _renderer(self):
        return self._fig._renderer

    def _configure_window(self, *, show):
        """Create the (still hidden) main window, its splash screen and status bar."""
        from ..viz.backends.renderer import _get_renderer

        splash = "Initializing dipole fitting GUI..." if show else False
        self._init_renderer = renderer = _get_renderer(
            size=(1080, 720),
            bgcolor="white",
            smooth_shading=_get_3d_option("smooth_shading"),
            # The window is only shown at the very end of ``__init__``, when it is
            # fully drawn: a window that pops up empty and then slowly fills itself in
            # looks broken.
            show=False,
            splash=splash,
        )
        # Showing any window closes the splash screen (see `_qt_safe_window`), so keep
        # it to ourselves until the main window is ready to be shown.
        self._splash = getattr(renderer.figure, "splash", None)
        if not hasattr(self._splash, "showMessage"):  # not the Qt backend
            self._splash = None
        renderer.figure.splash = False
        renderer.set_interaction("terrain")
        self._fig3d = renderer.scene()

        # Status bar, narrating what the GUI is doing (see `_set_status`).
        renderer._status_bar_initialize()
        self._status_label = renderer._status_bar_add_label(_STATUS_IDLE, stretch=1)

    def _set_status(self, message=_STATUS_IDLE):
        """Show what the GUI is currently doing, or ``"Ready"`` when it is idle.

        During startup the main window is not up yet, so the message is shown on the
        splash screen as well (the status bar shows it once the window appears).
        """
        if self._status_label is not None:
            self._status_label.set_value(message)
            # Repaint just this widget: unlike processing the event queue, this cannot
            # re-enter any of the event handlers of the GUI.
            self._status_label.update()
        # `_qt_safe_window` deletes `_splash` when `__init__` is done, hence getattr.
        splash = getattr(self, "_splash", None)
        if splash is not None:
            _splash_message(splash, message)

    @contextmanager
    def _busy(self, message):
        """Show ``message`` and block interaction while a slow operation runs.

        Nested uses collapse into the outermost one, so that operations that trigger
        one another (e.g. fitting a dipole refits all timecourses) show a single
        message and restore the cursor only once.
        """
        r = self._renderer
        # Increment the depth *before* processing events below: an event handler that
        # runs during that processing and uses `_busy` itself must see itself as
        # nested, or it would tear the busy state down mid-operation.
        self._busy_depth += 1
        try:
            if self._busy_depth == 1:
                self._busy_cursor = r._window_get_cursor()
                self._set_status(message)
                r._window_set_enabled(False)
                r._window_set_cursor(r._window_new_cursor("WaitCursor"))
                # Paint the busy state before starting the computation. The window is
                # disabled, so no user input can be delivered while we do this.
                r._process_events()
            yield
        finally:
            self._busy_depth -= 1
            if self._busy_depth == 0:
                r._window_set_cursor(self._busy_cursor)
                r._window_set_enabled(True)
                self._set_status()

    @property
    def dipoles(self):
        """A list of all the fitted dipoles that are enabled in the GUI."""
        return [d["dip"] for d in self._dipoles.values() if d["active"]]

    def _configure_main_display(self, show_sensors=True):
        """Configure main 3D display of the GUI."""
        fig_into = self._fig3d

        self._stc_brain = None
        if self._stc is not None:
            self._set_status("Plotting source estimate...")
            kwargs = dict(
                subject=self._subject,
                subjects_dir=self._subjects_dir,
                hemi="both",
                time_viewer=False,
                initial_time=self._current_time,
                time_label=None,  # the traces plot shows the current time
                # The source estimate is only a rough guide for where to put
                # dipoles, so map each surface vertex to its nearest source rather
                # than smoothing: the upsampling is then a gather instead of a
                # sparse matrix product, which is cheaper on every time change.
                smoothing_steps="nearest",
                brain_kwargs=dict(units="m", show=False),
                figure=fig_into,
                # the GUI renders on a white figure, so the Brain (and hence its
                # colorbar) needs to select a black foreground color
                background="white",
            )
            if isinstance(self._stc, SourceEstimate):
                kwargs["surface"] = "white"
            self._stc_brain = self._stc.plot(**kwargs)
            self._actors["brain"] = self._stc_brain._actors["data"]
            # a translucent cortex keeps the dipole arrows inside it visible,
            # set here in addition to "alpha" for Brain (that only controls the
            # alpha of the brain surface, not its overlay)
            self.set_mesh_opacity("brain", _MESH_ALPHA["brain"], update=False)
            colorbar = [
                actor
                for actor in (
                    self._stc_brain._scalar_bar,
                    self._stc_brain._scalar_bar_ticks,
                )
                if actor is not None
            ]
            if len(colorbar) > 0:
                self._actors["colorbar"] = colorbar
            fig_into = self._stc_brain  # plot into the brain instead

            # Rendering the brain mesh in a translucent manner requires a higher setting
            # for the depth peeling to prevent artifacts.
            fig_into._renderer.plotter.enable_depth_peeling(
                number_of_peels=6, occlusion_ratio=1e-7
            )

        self._set_status("Plotting field lines...")
        fig_ef = EvokedField(
            self._evoked,
            self._surf_maps,
            time=self._current_time,
            time_label=None,  # the time is shown on the time line of the traces plot
            interpolation="linear",
            alpha=0,
            contour_line_opacity=0.5,
            show_density=self._show_density,
            foreground="black",
            background="white",
            fig=fig_into,  # can be Figure3D or Brain instance; we own its window
        )
        del fig_into
        fig_ef.separate_canvas = False  # needed to plot the timeline later
        fig_ef.set_contour_line_width(2)
        if self._stc is not None:
            link(self._stc_brain, fig_ef)

        for surf_map in fig_ef._surf_maps:
            if surf_map["map_kind"] == "meg":
                helmet_mesh = surf_map["mesh"]
                helmet_mesh._actor.prop.culling = "back"
                self._actors["helmet"] = helmet_mesh._actor
                # needed later to draw the big arrows on the helmet
                self._helmet_surf = surf_map["surf"]
                # For MEG fieldlines, we want to occlude the ones not facing us,
                # otherwise it's hard to interpret them. Since the "contours" object
                # does not support backface culling, we create an opaque mesh to put in
                # front of the contour lines with frontface culling.
                occl_surf = deepcopy(surf_map["surf"])
                occl_surf["rr"] -= 1e-3 * occl_surf["nn"]
                occl_act, _ = fig_ef._renderer.surface(occl_surf, color="white")
                occl_act.prop.culling = "front"
                occl_act.prop.lighting = False
                self._actors["occlusion_surf"] = occl_act
            elif surf_map["map_kind"] == "eeg":
                head_mesh = surf_map["mesh"]
                head_mesh._actor.prop.culling = "back"
                self._actors["head"] = head_mesh._actor

        show_meg = (self._ch_type is None or self._ch_type == "meg") and any(
            [m["kind"] == "meg" for m in self._surf_maps]
        )
        show_eeg = (self._ch_type is None or self._ch_type == "eeg") and any(
            [m["kind"] == "eeg" for m in self._surf_maps]
        )
        meg_picks = pick_types(self._evoked.info, meg=show_meg, ref_meg=False)
        eeg_picks = pick_types(self._evoked.info, meg=False, eeg=show_eeg)
        picks = np.concatenate((meg_picks, eeg_picks))
        self._ch_names = [self._evoked.ch_names[i] for i in picks]

        for m in self._surf_maps:
            if m["kind"] == "eeg":
                head_surf = m["surf"]
                break
        else:
            self._set_status("Plotting head surface...")
            self._actors["head"], _, head_surf = _plot_head_surface(
                renderer=fig_ef._renderer,
                head="head",
                subject=self._subject,
                subjects_dir=self._subjects_dir,
                bem=self._bem,
                coord_frame="mri",
                to_cf_t=self._to_cf_t,
                alpha=0.2,
            )
            self._actors["head"].prop.culling = "back"

        if show_sensors:
            self._set_status("Plotting sensors...")
            sensors = _plot_sensors_3d(
                renderer=fig_ef._renderer,
                info=self._evoked.info,
                to_cf_t=self._to_cf_t,
                picks=picks,
                meg=["sensors"] if show_meg else False,
                eeg=["original"] if show_eeg else False,
                fnirs=False,
                warn_meg=False,
                head_surf=head_surf,
                units="m",
                sensor_alpha=dict(meg=0.1, eeg=1.0),
                orient_glyphs=False,
                scale_by_distance=False,
                project_points=False,
                surf=None,
                check_inside=None,
                nearest=None,
                sensor_colors=dict(
                    meg=["gray" for _ in meg_picks],
                    eeg=["white" for _ in eeg_picks],
                ),
            )
            self._actors["sensors"] = sum(sensors.values(), [])

        subscribe(fig_ef, "time_change", self._on_time_change)
        subscribe(fig_ef, "channels_select", self._on_channels_select)
        self._fig = fig_ef

        # Adjust camera (needs self._fig, hence after setting it)
        self._set_camera_preset("Left")
        for name, visible in _MESH_VISIBLE.items():
            if not visible and name in self._actors:
                self.toggle_mesh(name, show=False)

    def _configure_dock(self):
        """Configure the left and right dock areas of the GUI."""
        self._set_status("Setting up controls...")
        r = self._renderer

        # Visibility and opacity controls for the various meshes, one row per mesh.
        layout = r._dock_add_group_box("Meshes", collapse=True)
        grid = r._layout_create("grid")
        r._layout_add_widget(layout, grid)
        r._dock_add_label("visible", layout=grid, row=0, col=0)
        r._dock_add_label("opacity", layout=grid, row=0, col=1)

        @_auto_weakref
        def _toggle_mesh(show, name):
            self.toggle_mesh(name, show=bool(show))

        @_auto_weakref
        def _set_mesh_opacity(opacity, name):
            self.set_mesh_opacity(name, opacity)

        row = 0
        for actor_name in self._actors:
            if actor_name == "occlusion_surf":  # implementation detail, not a "mesh"
                continue
            row += 1
            widgets = [
                r._dock_add_check_box(
                    name=actor_name,
                    value=_MESH_VISIBLE.get(actor_name, True),
                    callback=partial(_toggle_mesh, name=actor_name),
                    layout=grid,
                    row=row,
                    col=0,
                )
            ]
            # 2D overlays like the colorbar get a visibility checkbox only.
            if actor_name not in _MESH_NO_OPACITY:
                widgets.append(
                    r._dock_add_slider(
                        name=None,
                        value=self._get_mesh_opacity(actor_name),
                        rng=[0, 1],
                        callback=partial(_set_mesh_opacity, name=actor_name),
                        double=True,
                        layout=grid,
                        row=row,
                        col=1,
                    )
                )
            self._mesh_widgets[actor_name] = widgets

        # Camera presets
        camera_layout = r._dock_add_layout(vertical=False)

        @_auto_weakref
        def _set_camera_preset(name):
            self._set_camera_preset(name)

        for preset in _CAMERA_PRESETS:
            r._dock_add_button(
                name=preset,
                callback=partial(_set_camera_preset, name=preset),
                style="toolbutton",
                tooltip=f"View the {preset.lower()} of the head",
                layout=camera_layout,
            )
        r._layout_add_widget(r._dock_layout, camera_layout)

        # Right dock
        r._dock_initialize(name="Dipole fitting", area="right")
        r._dock_add_button("Sensor data", self._on_sensor_data)
        r._dock_add_button("Fit dipole", self.fit_dipole)
        methods = ["Multi dipole (MNE)", "Single dipole"]

        @_auto_weakref
        def _on_select_method(method):
            self._on_select_method(method)

        self._method_combo = r._dock_add_combo_box(
            "Dipole model",
            value="Multi dipole (MNE)",
            rng=methods,
            callback=_on_select_method,
        )
        self._dipole_box = r._dock_add_group_box(name="Dipoles", collapse=False)

        @_auto_weakref
        def _save(fname):
            return self.save(fname)

        self._save_button = r._dock_add_file_button(
            name="save_dipoles",
            desc="Save dipoles",
            save=True,
            func=_save,
            tooltip="Save the dipoles to disk",
            filter_="Dipole files (*.dip  *.bdip)",
            initial_directory=".",
        )
        self._save_button.set_enabled(False)
        r._dock_add_stretch()

    def toggle_mesh(self, name, show=None):
        """Toggle a mesh on or off.

        Parameters
        ----------
        name : str
            Name of the mesh to toggle.
        show : bool | None
            Whether to show the mesh. If None, the visibility of the mesh is toggled.
        """
        actors = self._get_actors(name)
        if show is None:
            show = not actors[0].GetVisibility()
        for act in actors:
            act.SetVisibility(show)
        self._renderer._update()

    def set_mesh_opacity(self, name, opacity, *, update=True):
        """Set the opacity of a mesh.

        Parameters
        ----------
        name : str
            Name of the mesh.
        opacity : float
            The opacity of the mesh, between 0 (fully transparent) and 1 (opaque).
        update : bool
            If True, update the display immediately.
        """
        # The actors are a mix of PyVista wrappers and plain VTK actors, so stick to
        # the VTK API here (which both understand).
        for act in self._get_actors(name):
            act.GetProperty().SetOpacity(float(opacity))
        if update:
            self._renderer._update()

    def _get_actors(self, name):
        """Get the actors of a mesh as a list."""
        _check_option("name", name, self._actors.keys())
        actors = self._actors[name]
        # self._actors[name] is sometimes a list and sometimes not. Make it
        # always be a list to simplify the code.
        if not isinstance(actors, list):
            actors = [actors]
        return actors

    def _get_mesh_opacity(self, name):
        """Get the current opacity of a mesh."""
        return self._get_actors(name)[0].GetProperty().GetOpacity()

    def _set_camera_preset(self, name):
        """Point the camera at one of the standard views of the head."""
        _check_option("name", name, list(_CAMERA_PRESETS))
        self._renderer.set_camera(
            **_CAMERA_PRESETS[name], distance=0.55, focalpoint=(0, 0, 0.03)
        )
        self._renderer._update()

    def set_time(self, time):
        """Set the time point currently shown in the GUI.

        This is the programmatic equivalent of dragging the time slider, and is also the
        time at which :meth:`fit_dipole` will fit a dipole.

        Parameters
        ----------
        time : float
            The time to show, in seconds. Values outside the time range of the evoked
            data are clipped to the nearest valid time.
        """
        publish(self._fig, TimeChange(time=float(time)))

    def _on_time_change(self, event):
        new_time = np.clip(event.time, self._evoked.times[0], self._evoked.times[-1])
        self._current_time = new_time
        if self._time_line is not None:
            self._time_line.set_xdata([new_time])
            self._update_time_text()
            # only the time line and its label moved, so the traces can be blitted
            # from the cached background instead of being redrawn
            self._renderer._mplcanvas.update_blit_artists()
        self._update_arrows()

    def _update_time_text(self):
        """Label the time line with the current time and goodness-of-fit."""
        if self._time_text is None:
            return
        text = f"{self._current_time * 1e3:.0f} ms"
        if self._gof_line is not None and self._gof_line.get_visible():
            gof = np.interp(
                self._current_time, self._evoked.times, self._gof_line.get_ydata()
            )
            text += f" · GOF {gof:.0f}%"
        self._time_text.set_x(self._current_time)
        self._time_text.set_text(text)

    # TODO: Need to expose a public method for opening the sensor-data window and for
    # programmatically selecting the channels to fit dipoles to.
    def _on_sensor_data(self):
        """Show sensor data and allow sensor selection."""
        if self._fig_sensors is not None:
            return
        fig = self._evoked.plot_topo(select=True)
        fig.canvas.mpl_connect("close_event", self._on_sensor_data_close)
        link(self._fig, fig, recursive=True)
        self._fig_sensors = fig

    def _on_sensor_data_close(self, event):
        """Handle closing of the sensor selection window."""
        publish(self._fig, ChannelsSelect(ch_names=[]))
        self._fig_sensors = None

    def _on_channels_select(self, event):
        """Color selected sensor meshes."""
        selected_channels = set(event.ch_names)
        if "sensors" in self._actors:
            # Possibly multiple sensor types.
            for actor in self._actors["sensors"]:
                cloud = actor.GetMapper().GetInput()
                selected_idx = np.isin(
                    cloud.field_data["ch_names"], list(selected_channels)
                )
                colors = cloud.point_data["colors"]
                colors[selected_idx] = [0, 255, 0, 100]
                colors[~selected_idx] = [0, 0, 0, 10]
                cloud.point_data["colors"] = colors
        self._renderer._update()

    def fit_dipole(self):
        """Fit a single dipole and add it to the model.

        This is the programmatic equivalent of pressing the "Fit dipole" button. The
        dipole is fitted at the time currently shown in the GUI (see :meth:`set_time`),
        using the sensors that are currently selected in the sensor data window (or all
        sensors when no selection is active). The newly fitted dipole is appended to the
        :attr:`dipoles` attribute.
        """
        with self._busy("Fitting dipole..."):
            evoked_picked = self._evoked.copy()
            cov_picked = self._cov.copy()
            if self._fig_sensors is not None:
                picks = self._fig_sensors.lasso.selection
                if len(picks) > 0:
                    evoked_picked = evoked_picked.pick(picks)
                    evoked_picked.info.normalize_proj()
                    cov_picked = cov_picked.pick_channels(picks, ordered=False)
                    cov_picked["projs"] = evoked_picked.info["projs"]
            evoked_picked.crop(self._current_time, self._current_time)

            dip = fit_dipole(
                evoked_picked,
                cov_picked,
                self._bem,
                trans=self._head_mri_t,
                rank=self._rank,
                n_jobs=self._n_jobs,
                verbose=False,
            )[0]

            self.add_dipole(dip)

    def add_dipole(self, dipole, name=None):
        """Add a dipole (or multiple dipoles) to the GUI.

        Parameters
        ----------
        dipole : Dipole
            The dipole to add. If the ``Dipole`` object defines multiple dipoles, they
            will all be added.
        name : str | list of str | None
            The name of the dipole. When the ``Dipole`` object defines multiple dipoles,
            this should be a list containing the name for each dipole. When ``None``,
            the ``.name`` attribute of the ``Dipole`` object itself will be used.
        """
        from matplotlib.colors import to_hex

        _validate_type(name, (str, list, None), "name")
        if isinstance(name, str):
            names = [name]
        elif name is None:
            # Try to obtain names from `dipole.name`. When multiple dipoles are saved,
            # the names are concatenated with `;` marks.
            if dipole.name is None:
                names = [None] * len(dipole)
            elif len(dipole.name.split(";")) == len(dipole):
                names = dipole.name.split(";")
            else:
                names = [dipole.name] * len(dipole)
        else:
            names = name
        if len(names) != len(dipole):
            raise ValueError(
                f"Number of names ({len(names)}) does not match the number of dipoles "
                f"({len(dipole)})."
            )

        # Ensure orientations are unit vectors. Due to rounding issues this is sometimes
        # not the case.
        dipole._ori /= np.linalg.norm(dipole._ori, axis=1, keepdims=True)

        @_auto_weakref
        def _on_dipole_toggle(active, dip_num):
            return self._on_dipole_toggle(active, dip_num)

        @_auto_weakref
        def _on_dipole_set_name(name, dip_num):
            return self._on_dipole_set_name(name, dip_num)

        @_auto_weakref
        def _on_dipole_toggle_fix_orientation(fix, dip_num):
            return self._on_dipole_toggle_fix_orientation(fix, dip_num)

        @_auto_weakref
        def _on_dipole_delete(dip_num):
            return self._on_dipole_delete(dip_num)

        @_auto_weakref
        def _on_dipole_hover(dip_num, hover):
            return self._on_dipole_hover(dip_num, hover)

        new_dipoles = list()
        for dip, name in zip(dipole, names):
            # Coordinates needed to draw the big arrow on the helmet.
            helmet_coords, helmet_pos = self._get_helmet_coords(dip)

            # Collect all relevant information on the dipole in a dict.
            colors = _get_color_list()
            if len(self._dipoles) == 0:
                dip_num = 0
            else:
                dip_num = max(self._dipoles.keys()) + 1
            if name is None:
                dip.name = f"dip{dip_num}"
            else:
                dip.name = name
            dip_color = colors[dip_num % len(colors)]
            if helmet_coords is not None:
                arrow_mesh = pyvista.PolyData(*_arrow_mesh())
            else:
                arrow_mesh = None
            dipole_dict = dict(
                active=True,
                brain_arrow_actor=None,
                helmet_arrow_actor=None,
                arrow_mesh=arrow_mesh,
                color=dip_color,
                dip=dip,
                fix_ori=True,
                fix_position=True,
                helmet_coords=helmet_coords,
                helmet_pos=helmet_pos,
                num=dip_num,
                # fit_time=self._current_time,
            )
            self._dipoles[dip_num] = dipole_dict

            # Add a row to the dipole list
            r = self._renderer
            hlayout = r._dock_add_layout(vertical=False)
            widgets = []
            widgets.append(
                r._dock_add_check_box(
                    name="",
                    value=True,
                    callback=partial(_on_dipole_toggle, dip_num=dip_num),
                    layout=hlayout,
                )
            )
            widgets.append(
                r._dock_add_text(
                    name=dip.name,
                    value=dip.name,
                    placeholder="name",
                    callback=partial(_on_dipole_set_name, dip_num=dip_num),
                    layout=hlayout,
                )
            )
            # Give the name field the color of the dipole's trace, so the rows in the
            # dipole list can be matched up with the traces at a glance.
            widgets[-1].set_style(
                {
                    "background-color": to_hex(dip_color),
                    "color": "white" if _is_dark(dip_color) else "black",
                }
            )
            # Hovering the row emphasizes the traces belonging to this dipole.
            widgets[-1].set_hover_callbacks(
                enter=partial(_on_dipole_hover, dip_num=dip_num, hover=True),
                leave=partial(_on_dipole_hover, dip_num=dip_num, hover=False),
            )
            widgets.append(
                r._dock_add_check_box(
                    name="Fix ori",
                    value=True,
                    callback=partial(
                        _on_dipole_toggle_fix_orientation, dip_num=dip_num
                    ),
                    layout=hlayout,
                )
            )
            widgets.append(
                r._dock_add_button(
                    name="",
                    icon="clear",
                    callback=partial(_on_dipole_delete, dip_num=dip_num),
                    layout=hlayout,
                )
            )
            dipole_dict["widgets"] = widgets
            r._layout_add_widget(self._dipole_box, hlayout)
            new_dipoles.append(dipole_dict)

        # Show the dipoles and arrows in the 3D view. Only do this after
        # `_fit_timecourses` so that they have the correct size straight away.
        self._fit_timecourses()
        for dipole_dict in new_dipoles:
            dip = dipole_dict["dip"]
            dipole_dict["brain_arrow_actor"] = self._renderer.plotter.add_arrows(
                apply_trans(self._head_mri_t, dip.pos[0]),
                apply_trans(self._head_mri_t, dip.ori[0]),
                color=dipole_dict["color"],
                mag=0.05,
            )
            if dipole_dict["arrow_mesh"] is not None:
                dipole_dict["helmet_arrow_actor"] = self._renderer.plotter.add_mesh(
                    dipole_dict["arrow_mesh"],
                    color=dipole_dict["color"],
                    culling="front",
                )
        self._update_arrows()

    def _get_helmet_coords(self, dip):
        """Compute the coordinate system used for drawing the big arrows on the helmet.

        In this coordinate system, Z is normal to the helmet surface, and XY
        are tangential to the helmet surface.
        """
        if "helmet" not in self._actors:
            return None, None

        # Get the closest vertex (=point) of the helmet mesh
        dip_pos = apply_trans(self._head_mri_t, dip.pos[0])
        points = self._helmet_surf["rr"]
        normals = self._helmet_surf["nn"]
        distances = ((points - dip_pos) * normals).sum(axis=1)
        closest_point = np.argmin(distances)

        # Compute the position of the projected dipole on the helmet
        norm = normals[closest_point]
        helmet_pos = dip_pos + (distances[closest_point] + 0.003) * norm

        # Create a coordinate system where X and Y are tangential to the helmet
        helmet_coords = _normal_orth(norm)

        return helmet_coords, helmet_pos

    def _fit_timecourses(self):
        """Compute (or re-compute) dipole timecourses.

        Called whenever something changes to the multi-dipole situation, i.e. a dipole
        is added, removed, (de-)activated or the "Fix pos" box is toggled.
        """
        self._save_button.set_enabled(len(self.dipoles) > 0)
        active_dips = [d for d in self._dipoles.values() if d["active"]]
        if len(active_dips) == 0:
            if self._gof_line is not None:
                self._gof_line.set_visible(False)
                self._update_time_text()
                self._renderer._mplcanvas.update_plot()
            return

        with self._busy(f"Fitting {self._multi_dipole_method} model..."):
            # Forward solution for the active dipoles. It is needed for the multi-dipole
            # fit below, and in both fitting modes for computing the goodness-of-fit.
            # TODO: When two active dipoles have (nearly) identical positions, they
            # collapse to a single point in the discrete source space below, which
            # errors out. Ideal behavior unclear: merge them, or error informatively?
            this_src = setup_volume_source_space(
                "sample",
                pos=dict(
                    rr=apply_trans(
                        self._head_mri_t,
                        np.vstack([d["dip"].pos[0] for d in active_dips]),
                    ),
                    nn=apply_trans(
                        self._head_mri_t,
                        np.vstack([d["dip"].ori[0] for d in active_dips]),
                    ),
                ),
            )
            this_fwd = self.fwd.compute(this_src)
            this_fwd = convert_forward_solution(this_fwd, surf_ori=False)

            if self._multi_dipole_method == "Multi dipole (MNE)":
                inv = make_inverse_operator(
                    self._evoked.info,
                    # fwd,
                    this_fwd,
                    self._cov,
                    fixed=False,
                    loose=1.0,
                    depth=0,
                    rank=self._rank,
                )
                stc = apply_inverse(
                    self._evoked,
                    inv,
                    method="MNE",
                    lambda2=1e-6,
                    pick_ori="vector",
                )

                timecourses = stc.magnitude().data
                orientations = (stc.data / timecourses[:, np.newaxis, :]).transpose(
                    0, 2, 1
                )
                fixed_timecourses = stc.project(
                    np.array([dip["dip"].ori[0] for dip in active_dips])
                )[0].data

                for i, dip in enumerate(active_dips):
                    if dip["fix_ori"]:
                        dip["timecourse"] = fixed_timecourses[i]
                        dip["orientation"] = dip["dip"].ori.repeat(
                            len(stc.times), axis=0
                        )
                    else:
                        dip["timecourse"] = timecourses[i]
                        dip["orientation"] = orientations[i]
            else:
                assert self._multi_dipole_method == "Single dipole"  # only other option
                for dip in active_dips:
                    dip_with_timecourse, _ = fit_dipole(
                        self._evoked,
                        self._cov,
                        self._bem,
                        pos=dip["dip"].pos[0],  # position is always fixed
                        ori=dip["dip"].ori[0] if dip["fix_ori"] else None,
                        trans=self._head_mri_t,
                        rank=self._rank,
                        n_jobs=self._n_jobs,
                        verbose=True,
                    )
                    if dip["fix_ori"]:
                        dip["timecourse"] = dip_with_timecourse.data[0]
                        dip["orientation"] = dip["dip"].ori.repeat(
                            len(dip_with_timecourse.times), axis=0
                        )
                    else:
                        dip["timecourse"] = dip_with_timecourse.amplitude
                        dip["orientation"] = dip_with_timecourse.ori

            # Update matplotlib canvas at the bottom of the window. Timecourses are
            # stored in SI units (Am), but shown in nAm, hence the 1e9 scaling at the
            # display boundary.
            canvas = self._setup_mplcanvas()
            ymin, ymax = 0, 0
            for dip in active_dips:
                # The dot marks the time at which the dipole was fitted.
                fit_time = dip["dip"].times[0]
                fit_value = np.interp(
                    fit_time, self._evoked.times, dip["timecourse"] * 1e9
                )
                if "line_artist" in dip:
                    dip["line_artist"].set_ydata(dip["timecourse"] * 1e9)
                    dip["dot_artist"].set_ydata([fit_value])
                else:
                    dip["line_artist"] = canvas.plot(
                        self._evoked.times,
                        dip["timecourse"] * 1e9,
                        label=dip["dip"].name,
                        color=dip["color"],
                        linewidth=_TRACE_LINEWIDTH,
                        update=False,
                    )
                    dip["dot_artist"] = canvas.axes.plot(
                        [fit_time],
                        [fit_value],
                        "o",
                        color=dip["color"],
                        markersize=_TRACE_MARKERSIZE,
                        zorder=dip["line_artist"].get_zorder() + 1,
                    )[0]
                ymin = min(ymin, 1.1 * dip["timecourse"].min() * 1e9)
                ymax = max(ymax, 1.1 * dip["timecourse"].max() * 1e9)
            canvas.axes.set_ylim(ymin, ymax)
            self._update_gof(canvas, active_dips, this_fwd)
            canvas.update_plot()
            self._update_arrows()

    def _update_gof(self, canvas, active_dips, fwd):
        """Draw the goodness-of-fit of the combined dipole model on a twin axis."""
        gof = self._compute_gof(active_dips, fwd)
        if self._gof_ax is None:
            self._gof_ax = canvas.axes.twinx()
            self._gof_ax.set_ylim(0, 100)
            self._gof_ax.set_ylabel("GOF (%)", color="gray")
            self._gof_ax.tick_params(axis="y", colors="gray")
            self._gof_ax.spines["top"].set_visible(False)
            self._gof_ax.spines["right"].set_visible(True)
            self._gof_ax.spines["bottom"].set_visible(False)
            self._gof_ax.spines["left"].set_visible(False)
            # Twin axes are drawn on top by default. Flip that around (the classic
            # matplotlib recipe) so the activation traces stay on top of the GOF line.
            canvas.axes.set_zorder(self._gof_ax.get_zorder() + 1)
            canvas.axes.patch.set_visible(False)
        if self._gof_line is None:
            (self._gof_line,) = self._gof_ax.plot(
                self._evoked.times, gof, color="gray", alpha=0.5
            )
        else:
            self._gof_line.set_ydata(gof)
            self._gof_line.set_visible(True)
        self._update_time_text()

    def _compute_gof(self, active_dips, fwd):
        """Compute the goodness-of-fit timecourse of the combined dipole model."""
        # Moments (in head coordinates, like `fwd["sol"]["data"]`) of all dipoles.
        q = np.concatenate(
            [
                (dip["orientation"] * dip["timecourse"][:, np.newaxis]).T
                for dip in active_dips
            ]
        )
        # Bad channels are in the forward solution, but never in the whitener (nor in
        # the channels `fit_dipole` uses), so drop them before whitening.
        picks = [
            c for c in fwd["sol"]["row_names"] if c not in self._evoked.info["bads"]
        ]
        W, ch_names = compute_whitener(
            self._cov, self._evoked.info, picks=picks, rank=self._rank, verbose=False
        )
        data = self._evoked.data[[self._evoked.ch_names.index(c) for c in ch_names]]
        gain = fwd["sol"]["data"][[fwd["sol"]["row_names"].index(c) for c in ch_names]]
        residual = W @ (data - gain @ q)
        data = W @ data
        gof = np.zeros(data.shape[1])
        denom = np.sum(data**2, axis=0)
        good = denom > 0  # a field of exactly zero has no fit quality to speak of
        gof[good] = 100 * (1 - np.sum(residual[:, good] ** 2, axis=0) / denom[good])
        return gof

    @verbose
    def save(self, fname, verbose=None):
        """Save the fitted dipoles to a file.

        Parameters
        ----------
        fname : path-like
            The name of the file. Should end in ``'.dip'`` to save in plain text format,
            or in ``'.bdip'`` to save in binary format.
        %(verbose)s
        """
        if len(self.dipoles) == 0:
            logger.info("No dipoles to save.")
            return

        logger.info(f"Saving dipoles as: {fname}")
        fname = Path(fname)

        # Pack the dipoles into a single mne.Dipole object.
        if all(d.khi2 is not None for d in self.dipoles):
            khi2 = np.array([d.khi2[0] for d in self.dipoles])
        else:
            khi2 = None

        if all(d.nfree is not None for d in self.dipoles):
            nfree = np.array([d.nfree[0] for d in self.dipoles])
        else:
            nfree = None

        dip = Dipole(
            times=np.array([d.times[0] for d in self.dipoles]),
            pos=np.array([d.pos[0] for d in self.dipoles]),
            amplitude=np.array([d.amplitude[0] for d in self.dipoles]),
            ori=np.array([d.ori[0] for d in self.dipoles]),
            gof=np.array([d.gof[0] for d in self.dipoles]),
            khi2=khi2,
            nfree=nfree,
            conf={
                key: np.array([d.conf[key][0] for d in self.dipoles])
                for key in self.dipoles[0].conf.keys()
            },
            name=";".join(d.name if hasattr(d, "name") else "" for d in self.dipoles),
        )
        dip.save(fname, overwrite=True, verbose=verbose)

    def _update_arrows(self):
        """Update the arrows to have the correct size and orientation."""
        active_dips = [d for d in self._dipoles.values() if d["active"]]
        if len(active_dips) == 0:
            return
        orientations = [dip["orientation"] for dip in active_dips]
        timecourses = [dip["timecourse"] for dip in active_dips]
        arrow_scaling = 0.05 / np.max(np.abs(timecourses))
        for dip, ori, timecourse in zip(active_dips, orientations, timecourses):
            helmet_coords = dip["helmet_coords"]
            if helmet_coords is None:
                continue

            dip_ori = apply_trans(
                self._head_mri_t,
                [np.interp(self._current_time, self._evoked.times, o) for o in ori.T],
            )
            dip_moment = np.interp(self._current_time, self._evoked.times, timecourse)
            arrow_size = dip_moment * arrow_scaling
            arrow_mesh = dip["arrow_mesh"]

            # Project the orientation of the dipole tangential to the helmet
            dip_ori_tan = helmet_coords[:2] @ dip_ori @ helmet_coords[:2]

            # Rotate the coordinate system such that Y lies along the dipole
            # orientation, now we have our desired coordinate system for the
            # arrows.
            arrow_coords = np.array(
                [np.cross(dip_ori_tan, helmet_coords[2]), dip_ori_tan, helmet_coords[2]]
            )
            arrow_coords /= np.linalg.norm(arrow_coords, axis=1, keepdims=True)

            # Update the arrow mesh to point in the right directions
            arrow_mesh.points = (_arrow_mesh()[0] * arrow_size) @ arrow_coords
            arrow_mesh.points += dip["helmet_pos"]
        self._renderer._update()

    # TODO: Need to expose a public method for setting the multi-dipole method
    def _on_select_method(self, method):
        """Select the method to use for multi-dipole timecourse fitting."""
        _check_option("method", method, ("Multi dipole (MNE)", "Single dipole"))
        if method == self._multi_dipole_method:
            return
        self._multi_dipole_method = method
        # Defer the (slow) refit to the event loop instead of running it here, inside
        # the combo box's signal handler: this lets the combo box finish closing its
        # popup and repaint before the computation starts.
        if not self._refit_pending:
            self._refit_pending = True
            self._renderer._window_defer(self._deferred_refit)

    def _deferred_refit(self):
        """Refit the timecourses, deferred so that widgets can settle first."""
        self._refit_pending = False
        self._fit_timecourses()

    # TODO: Need to expose public methods for toggling, renaming, (un)fixing the
    # orientation of, and deleting a dipole (probably addressed by name or index).
    def _on_dipole_toggle(self, active, dip_num):
        """Toggle a dipole on or off."""
        dipole = self._dipoles[dip_num]
        active = bool(active)
        dipole["active"] = active
        dipole["line_artist"].set_visible(active)
        dipole["dot_artist"].set_visible(active)
        dipole["brain_arrow_actor"].visibility = active
        dipole["helmet_arrow_actor"].visibility = active
        self._fit_timecourses()
        self._renderer._update()
        self._renderer._mplcanvas.update_plot()

    def _on_dipole_set_name(self, name, dip_num):
        """Set the name of a dipole."""
        self._dipoles[dip_num]["dip"].name = name
        self._renderer._mplcanvas.update_plot()

    def _on_dipole_toggle_fix_orientation(self, fix, dip_num):
        """Fix dipole orientation when fitting timecourse."""
        self._dipoles[dip_num]["fix_ori"] = bool(fix)
        self._fit_timecourses()

    def _on_dipole_delete(self, dip_num):
        """Delete previously fitted dipole."""
        dipole = self._dipoles[dip_num]
        dipole["line_artist"].remove()
        dipole["dot_artist"].remove()
        dipole["brain_arrow_actor"].visibility = False
        if dipole["helmet_arrow_actor"] is not None:  # no helmet arrow for EEG
            dipole["helmet_arrow_actor"].visibility = False
        for widget in dipole["widgets"]:
            widget.hide()
        del self._dipoles[dip_num]
        self._fit_timecourses()
        self._renderer._update()
        self._renderer._mplcanvas.update_plot()

    def _on_dipole_hover(self, dip_num, hover):
        """Emphasize the traces of the dipole whose row is being hovered."""
        dipole = self._dipoles.get(dip_num)
        if dipole is None or "line_artist" not in dipole:
            return
        dipole["line_artist"].set_linewidth(
            _TRACE_LINEWIDTH_HOVER if hover else _TRACE_LINEWIDTH
        )
        dipole["dot_artist"].set_markersize(
            _TRACE_MARKERSIZE_HOVER if hover else _TRACE_MARKERSIZE
        )
        self._renderer._mplcanvas.update_plot()

    def _setup_mplcanvas(self):
        """Configure the matplotlib canvas at the bottom of the window."""
        from matplotlib.transforms import offset_copy

        if self._renderer._mplcanvas is None:
            self._renderer._mplcanvas = self._renderer._window_get_mplcanvas(
                self._fig, 0.22, False, False
            )
            self._renderer._window_adjust_mplcanvas_layout()
            canvas = self._renderer._mplcanvas
            # Dipole moments are stored in Am, but displayed in nAm (see
            # `_fit_timecourses`).
            canvas.axes.set_ylabel("Activation (nAm)")
            canvas.axes.set_xlim(self._evoked.times[0], self._evoked.times[-1])
            canvas.axes.spines["top"].set_visible(False)
            canvas.axes.spines["right"].set_visible(False)
            canvas.axes.axhline(0, linewidth=1, color="gray", zorder=0)
        if self._time_line is None:
            canvas = self._renderer._mplcanvas
            self._time_line = canvas.plot_time_line(
                self._current_time,
                label="time",
                color="black",
                linewidth=1,
            )
            # Label the time line, with a small offset so it does not overlap the line.
            self._time_text = canvas.axes.text(
                self._current_time,
                0.97,
                f"{self._current_time * 1e3:.0f} ms",
                transform=offset_copy(
                    canvas.axes.get_xaxis_transform(),
                    fig=canvas.fig,
                    x=3,
                    units="points",
                ),
                va="top",
                ha="left",
                fontsize=8,
                color="black",
            )
            # the label travels with the time line, so it is drawn along with it
            canvas.add_blit_artist(self._time_text)
        return self._renderer._mplcanvas

    def close(self):
        """Close the dipole fitting GUI."""
        if self._renderer is not None:
            try:
                self._renderer.close()
            except AttributeError:  # maybe already closed
                pass


def _arrow_mesh():
    """Obtain a mesh of an arrow."""
    vertices = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.3, 0.7, 0.0],
            [0.1, 0.7, 0.0],
            [0.1, -1.0, 0.0],
            [-0.1, -1.0, 0.0],
            [-0.1, 0.7, 0.0],
            [-0.3, 0.7, 0.0],
        ]
    )
    faces = np.array([[7, 0, 1, 2, 3, 4, 5, 6]])
    return vertices, faces
