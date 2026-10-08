# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import copy
import os.path as op
import warnings
from functools import partial

import numpy as np

from ..._freesurfer import read_talxfm, vertex_to_mni
from ...transforms import apply_trans
from ...utils import (
    _auto_weakref,
    _ensure_int,
    _ReuseCycle,
    _validate_type,
    fill_doc_static,
)
from ..ui_events import (
    PlaybackSpeed,
    VertexSelect,
    disable_ui_events,
    publish,
    subscribe,
)
from ..utils import _get_color_list, safe_event
from .colormap import _CORTEX_PRESETS
from .view import _lh_views_dict


class _TimeViewerMixin:
    """Mixin providing the interactive time viewer (GUI) of Brain."""

    def setup_time_viewer(self, time_viewer=True, show_traces=True):
        """Configure the time viewer parameters.

        Parameters
        ----------
        time_viewer : bool
            If True, enable widgets interaction. Defaults to True.

        show_traces : bool
            If True, enable visualization of time traces. Defaults to True.

        Notes
        -----
        The keyboard shortcuts are the following:

        '?': Display help window
        'i': Toggle interface
        's': Apply auto-scaling
        'r': Restore original clim
        'c': Clear all traces
        'n': Shift the time forward by the playback speed
        'b': Shift the time backward by the playback speed
        'Space': Start/Pause playback
        'Up': Decrease camera elevation angle
        'Down': Increase camera elevation angle
        'Left': Decrease camera azimuth angle
        'Right': Increase camera azimuth angle

        When multiple overlays are active (added via
        :meth:`add_data` with ``remove_existing=False``), an **Overlay**
        drop-down menu appears in the *Color Limits* dock panel.  Selecting
        an entry from that menu switches which overlay's ``fmin`` / ``fmid``
        / ``fmax`` sliders and smoothing control are active.
        """
        if self.time_viewer:
            return
        if not self._all_data:
            raise ValueError("No data to visualize. See ``add_data``.")
        self.time_viewer = time_viewer
        self.orientation = list(_lh_views_dict.keys())

        # Default configuration
        self.visibility = False
        self.default_playback_speed_range = [0.01, 1]
        self.default_playback_speed_value = 0.01
        self.default_status_bar_msg = "Press ? for help"
        self.default_label_extract_modes = {
            "stc": ["mean", "max"],
            "src": ["mean_flip", "pca_flip", "auto"],
        }
        self.annot = None
        self.label_extract_mode = None
        all_keys = ("lh", "rh", "vol")
        self.act_data_smooth = {key: (None, None) for key in all_keys}
        # remove grey for better contrast on the brain
        self.color_list = _get_color_list(remove=("#7f7f7f",))
        self.color_cycle = _ReuseCycle(self.color_list)
        self.mpl_canvas = None
        self.help_canvas = None
        self.rms = None
        self._picked_patches = {key: list() for key in all_keys}
        self._picked_points = dict()
        self._peak_vertices = {}
        self._auto_peak_points = set()
        self._trace_meta = {}
        self._label_trace_meta = {}
        self._mouse_no_mvt = -1
        self._show_hover_info = False
        self._hover_caption = None

        # Derived parameters:
        self.playback_speed = self.default_playback_speed_value
        _validate_type(show_traces, (bool, str, "numeric"), "show_traces")
        self.interactor_fraction = 0.25
        if isinstance(show_traces, str):
            self.show_traces = True
            self.separate_canvas = False
            self.traces_mode = "vertex"
            if show_traces == "separate":
                self.separate_canvas = True
            elif show_traces == "label":
                self.traces_mode = "label"
            else:
                assert show_traces == "vertex"  # guaranteed above
        else:
            if isinstance(show_traces, bool):
                self.show_traces = show_traces
            else:
                show_traces = float(show_traces)
                if not 0 < show_traces < 1:
                    raise ValueError(
                        "show traces, if numeric, must be between 0 and 1, "
                        f"got {show_traces}"
                    )
                self.show_traces = True
                self.interactor_fraction = show_traces
            self.traces_mode = "vertex"
            self.separate_canvas = False
        del show_traces

        # Start with the first-added overlay active (the colormap dock's
        # default) so that the scalar bar, picking, and traces are all
        # configured against the same overlay
        self._active_data_key = next(iter(self._all_data))
        self._configure_time_label()
        self._configure_scalar_bar()
        # keyboard shortcuts, picking and hover all need mouse and key events
        # from the interactor, plus VTK actors (see the TODO in _add_volume_data),
        # which a page drawn by the notebook_js backend has neither of
        if self._renderer._kind != "notebook_js":
            self._configure_shortcuts()
            self._configure_picking()
            self._configure_hover()
        self._configure_dock()
        self._configure_tool_bar()
        self._configure_status_bar()
        self._configure_help()
        # show everything at the end
        self.toggle_interface()
        self._renderer.show()

        # sizes could change, update views
        for hemi in ("lh", "rh"):
            for ri, ci, v in self._iter_views(hemi):
                self.show_view(view=v, row=ri, col=ci)

        self._renderer._update()
        # finally, show the MplCanvas
        if self.show_traces:
            self.mpl_canvas.show()

    def toggle_interface(self, value=None):
        """Toggle the interface.

        Parameters
        ----------
        value : bool | None
            If True, the widgets are shown and if False, they
            are hidden. If None, the state of the widgets is
            toggled. Defaults to None.
        """
        if value is None:
            self.visibility = not self.visibility
        else:
            self.visibility = value

        # update tool bar and dock
        with self._renderer._window_ensure_minimum_sizes():
            if self.visibility:
                self._renderer._dock_show()
                self._renderer._tool_bar_update_button_icon(
                    name="visibility", icon_name="visibility_on"
                )
            else:
                self._renderer._dock_hide()
                self._renderer._tool_bar_update_button_icon(
                    name="visibility", icon_name="visibility_off"
                )

        self._renderer._update()

    def toggle_playback(self, value=None):
        """Toggle time playback.

        Parameters
        ----------
        value : bool | None
            If True, automatic time playback is enabled and if False,
            it's disabled. If None, the state of time playback is toggled.
            Defaults to None.
        """
        self._renderer._toggle_playback(value)

    def reset(self):
        """Reset view, current time and time step."""
        self.reset_view()
        self._renderer._reset_time()

    def set_playback_speed(self, speed):
        """Set the time playback speed.

        Parameters
        ----------
        speed : float
            The speed of the playback.
        """
        publish(self, PlaybackSpeed(speed=speed))

    def _configure_time_label(self):
        self.time_actor = self._data.get("time_actor")
        if self.time_actor is not None:
            self.time_actor.SetPosition(0.5, 0.03)
            self.time_actor.GetTextProperty().SetJustificationToCentered()
            self.time_actor.GetTextProperty().BoldOn()

    def _configure_scalar_bar(self):
        if self._scalar_bar is not None:
            self._scalar_bar.SetOrientationToVertical()
            self._scalar_bar.SetHeight(0.6)
            self._scalar_bar.SetWidth(0.05)
            self._scalar_bar.SetPosition(0.02, 0.2)

    def _configure_dock_playback_widget(self, name):
        len_time = len(self._data["time"]) - 1

        # Time widget
        if len_time < 1:
            self.widgets["time"] = None
            self.widgets["min_time"] = None
            self.widgets["max_time"] = None
            self.widgets["current_time"] = None
        else:

            @_auto_weakref
            def current_time_func():
                return self._current_time

            self._renderer._enable_time_interaction(
                self,
                current_time_func,
                self._data["time"],
                self.default_playback_speed_value,
                self.default_playback_speed_range,
            )

        # Time label
        current_time = self._current_time
        assert current_time is not None  # should never be the case, float
        time_label = self._data["time_label"]
        if callable(time_label):
            current_time = time_label(current_time)
        else:
            current_time = time_label
        if self.time_actor is not None:
            self.time_actor.SetInput(current_time)
        del current_time

    def _configure_dock_orientation_widget(self, name):
        layout = self._renderer._dock_add_group_box(name, collapse=True)
        # Renderer widget
        # one entry per subplot, which is what VTK's renderers are (asking the
        # renderer for them would reach past its interface, see _add_volume_data)
        rends = [str(i) for i in range(int(np.prod(self._subplot_shape)))]
        if len(rends) > 1:

            @_auto_weakref
            def select_renderer(idx):
                idx = int(idx)
                loc = self._renderer._index_to_loc(idx)
                self.plotter.subplot(*loc)

            self.widgets["renderer"] = self._renderer._dock_add_combo_box(
                name="Renderer",
                value="0",
                rng=rends,
                callback=select_renderer,
                layout=layout,
            )

        # Use 'lh' as a reference for orientation for 'both'
        if self._hemi == "both":
            hemis_ref = ["lh"]
        else:
            hemis_ref = self._hemis
        orientation_data = [None] * len(rends)
        for hemi in hemis_ref:
            for ri, ci, v in self._iter_views(hemi):
                idx = self._renderer._loc_to_index((ri, ci))
                if v == "flat":
                    _data = None
                else:
                    _data = dict(default=v, hemi=hemi, row=ri, col=ci)
                orientation_data[idx] = _data

        @_auto_weakref
        def set_orientation(value, orientation_data=orientation_data):
            if "renderer" in self.widgets:
                idx = int(self.widgets["renderer"].get_value())
            else:
                idx = 0
            if orientation_data[idx] is not None:
                self.show_view(
                    value,
                    row=orientation_data[idx]["row"],
                    col=orientation_data[idx]["col"],
                    hemi=orientation_data[idx]["hemi"],
                )

        self.widgets["orientation"] = self._renderer._dock_add_combo_box(
            name=None,
            value=self.orientation[0],
            rng=self.orientation,
            callback=set_orientation,
            layout=layout,
        )

    def _update_flat_widgets(self):
        """Grey out the dock controls that cannot act on a flat patch."""
        enabled = self._surf != "flat"
        for key in ("orientation", "silhouette"):
            if key in self.widgets:
                self.widgets[key].set_enabled(enabled)

    def _configure_dock_surface_widget(self, name):
        layout = self._renderer._dock_add_group_box(name, collapse=True)
        surfs = ["pial", "white", "inflated"]
        if self._has_flatmaps():
            surfs.append("flat")
        if self._surf in surfs:
            self.widgets["surf"] = self._renderer._dock_add_combo_box(
                name="Surf",
                value=self._surf,
                rng=surfs,
                callback=self.set_surf,
                layout=layout,
            )
        self.widgets["cortex"] = self._renderer._dock_add_combo_box(
            name="Cortex",
            value=self._cortex_preset,
            rng=_CORTEX_PRESETS,
            callback=self.set_cortex_colormap,
            layout=layout,
        )
        self.widgets["cortex_alpha"] = self._renderer._dock_add_slider(
            name="Alpha",
            value=self._alpha,
            rng=[0.0, 1.0],
            callback=self.set_cortex_alpha,
            double=True,
            layout=layout,
        )
        self.widgets["silhouette"] = self._renderer._dock_add_spin_box(
            name="Silhouette",
            value=self._silhouette["line_width"] if self.silhouette else 0.0,
            rng=[0.0, 10.0],
            callback=self.set_silhouette_line_width,
            layout=layout,
        )
        # controls that cannot act on a flat patch are greyed out rather than
        # dropped, since the surface can be switched back and forth live
        self._update_flat_widgets()

    def _configure_dock_colormap_widget(self, name):
        self._active_data_key = next(iter(self._all_data))
        fmax, fscale, fscale_power = _get_range(self)
        rng = [0, fmax * fscale]
        self._data["fscale"] = fscale

        layout = self._renderer._dock_add_group_box(name, collapse=False)

        @_auto_weakref
        def select_data_key(value):
            self._active_data_key = value
            self._refresh_colormap_widgets()
            self._update_act_data_smooth()
            if self.show_traces:
                self._update_peak_vertices()
            if self.mpl_canvas is not None:
                self.mpl_canvas.axes.relim()
                self.mpl_canvas.axes.autoscale_view()
                self.mpl_canvas.update_plot()

        self.widgets["data_key"] = self._renderer._dock_add_combo_box(
            name="Overlay",
            value=self._active_data_key,
            rng=list(self._all_data.keys()),
            callback=select_data_key,
            layout=layout,
        )
        if len(self._all_data) <= 1:
            self.widgets["data_key"].hide()

        text = "min / mid / max"
        if fscale_power != 0:
            text += f" (×1e{fscale_power:d})"
        self._renderer._dock_add_label(
            value=text,
            align=True,
            layout=layout,
        )

        @_auto_weakref
        def update_single_lut_value(value, key):
            # Called by the sliders and spin boxes.
            self.update_lut(**{key: value / self._data["fscale"]})

        keys = ("fmin", "fmid", "fmax")
        for key in keys:
            hlayout = self._renderer._dock_add_layout(vertical=False)
            self.widgets[key] = self._renderer._dock_add_slider(
                name=None,
                value=self._data[key] * self._data["fscale"],
                rng=rng,
                callback=partial(update_single_lut_value, key=key),
                double=True,
                layout=hlayout,
            )
            self.widgets[f"entry_{key}"] = self._renderer._dock_add_spin_box(
                name=None,
                value=self._data[key] * self._data["fscale"],
                callback=partial(update_single_lut_value, key=key),
                rng=rng,
                layout=hlayout,
            )
            self._renderer._layout_add_widget(layout, hlayout)

        # reset / minus / plus
        hlayout = self._renderer._dock_add_layout(vertical=False)
        self._renderer._dock_add_label(
            value="Rescale",
            align=True,
            layout=hlayout,
        )
        self.widgets["reset"] = self._renderer._dock_add_button(
            name="↺",
            callback=self.apply_auto_scaling,
            layout=hlayout,
            style="toolbutton",
        )

        @_auto_weakref
        def fminus():
            self._update_fscale(1.2**-0.25)

        self.widgets["fminus"] = self._renderer._dock_add_button(
            name="➖",
            callback=fminus,
            layout=hlayout,
            style="toolbutton",
        )

        @_auto_weakref
        def fplus():
            self._update_fscale(1.2**0.25)

        self.widgets["fplus"] = self._renderer._dock_add_button(
            name="➕",
            callback=fplus,
            layout=hlayout,
            style="toolbutton",
        )
        self._renderer._layout_add_widget(layout, hlayout)

        self.widgets["smoothing"] = self._renderer._dock_add_spin_box(
            name="Smoothing",
            value=self._data["smoothing_steps"],
            rng=[-1, 15],
            callback=self.set_data_smoothing,
            double=False,
            layout=layout,
        )

        self._update_colormap_range()

    def _refresh_colormap_widgets(self):
        """Sync colormap dock widgets with the currently active overlay."""
        if self._data is None or "fmin" not in self.widgets:
            return
        fmax, fscale, fscale_power = _get_range(self)
        self._data["fscale"] = fscale
        rng = [0, fmax * fscale]
        with disable_ui_events(self):
            for key in ("fmin", "fmid", "fmax"):
                val = self._data[key] * fscale
                self.widgets[key].set_range(rng)
                self.widgets[key].set_value(val)
                self.widgets[f"entry_{key}"].set_range(rng)
                self.widgets[f"entry_{key}"].set_value(val)
            if "smoothing" in self.widgets:
                self.widgets["smoothing"].set_value(self._data["smoothing_steps"])
        # Force the brain and colorbar to reflect the newly active overlay.
        self._update_colormap_range(
            fmin=self._data["fmin"],
            fmid=self._data["fmid"],
            fmax=self._data["fmax"],
        )
        self._renderer._update()

    def _configure_dock_trace_widget(self, name):
        if not self.show_traces:
            return
        # do not show trace mode for volumes
        if (
            self._data.get("src", None) is not None
            and self._data["src"].kind == "volume"
        ):
            self._configure_vertex_time_course()
            return

        layout = self._renderer._dock_add_group_box(name, collapse=True)

        # setup candidate annots
        @safe_event
        @_auto_weakref
        def _set_annot(annot):
            self.clear_glyphs()
            self.remove_labels()
            self.remove_annotations()
            self.annot = annot

            if annot == "None":
                self.traces_mode = "vertex"
                self._configure_vertex_time_course()
            else:
                self.traces_mode = "label"
                self._configure_label_time_course()
            self._renderer._update()

        # setup label extraction parameters
        @safe_event
        @_auto_weakref
        def _set_label_mode(mode):
            if self.traces_mode != "label":
                return
            glyphs = copy.deepcopy(self._picked_patches)
            self.label_extract_mode = mode
            self.clear_glyphs()
            for hemi in self._hemis:
                for label_id in glyphs[hemi]:
                    label = self._annotation_labels[hemi][label_id]
                    vertex_id = label.vertices[0]
                    self._add_label_glyph(hemi, None, vertex_id)
            self.mpl_canvas.axes.relim()
            self.mpl_canvas.axes.autoscale_view()
            self.mpl_canvas.update_plot()
            self._renderer._update()

        from ...label import _read_annot_cands
        from ...source_estimate import _get_allowed_label_modes

        dir_name = op.join(self._subjects_dir, self._subject, "label")
        cands = _read_annot_cands(dir_name, raise_error=False)
        cands = cands + ["None"]
        self.annot = cands[0]
        stc = self._data["stc"]
        # None (no extraction) is allowed by _get_allowed_label_modes but is
        # not a valid choice here; with src=None it would otherwise end up
        # last and become the default, breaking label extraction
        modes = [m for m in _get_allowed_label_modes(stc) if m is not None]
        if self._data["src"] is None:
            modes = [
                m for m in modes if m not in self.default_label_extract_modes["src"]
            ]
        self.label_extract_mode = modes[-1]
        if self.traces_mode == "vertex":
            _set_annot("None")
        else:
            _set_annot(self.annot)
        self.widgets["annotation"] = self._renderer._dock_add_combo_box(
            name="Annotation",
            value=self.annot,
            rng=cands,
            callback=_set_annot,
            layout=layout,
        )
        self.widgets["extract_mode"] = self._renderer._dock_add_combo_box(
            name="Extract mode",
            value=self.label_extract_mode,
            rng=modes,
            callback=_set_label_mode,
            layout=layout,
        )

    def _configure_dock(self):
        self._renderer._dock_initialize()
        self._configure_dock_playback_widget(name="Playback")
        self._configure_dock_colormap_widget(name="Color Limits")
        self._configure_dock_orientation_widget(name="Orientation")
        self._configure_dock_surface_widget(name="Surface")
        self._configure_dock_trace_widget(name="Atlas")
        self._configure_dock_trace_list_widget(name="Trace List")
        self._renderer._dock_finalize()

    def _configure_dock_trace_list_widget(self, name):
        if not self.show_traces or self.mpl_canvas is None:
            return
        add_trace_list = getattr(self._renderer, "_dock_add_trace_list", None)
        if add_trace_list is None:
            return
        self.mpl_canvas._trace_list = add_trace_list(name, collapse=False)
        self.mpl_canvas.sync_traces()

    def _configure_mplcanvas(self):
        # Get the fractional components for the brain and mpl
        self.mpl_canvas = self._renderer._window_get_mplcanvas(
            brain=self,
            interactor_fraction=self.interactor_fraction,
            show_traces=self.show_traces,
            separate_canvas=self.separate_canvas,
        )
        xlim = [np.min(self._data["time"]), np.max(self._data["time"])]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.mpl_canvas.axes.set(xlim=xlim)
        if not self.separate_canvas:
            self._renderer._window_adjust_mplcanvas_layout()
        self.mpl_canvas.set_color(
            bg_color=self._bg_color,
            fg_color=self._fg_color,
        )

    def _configure_vertex_time_course(self):
        if not self.show_traces:
            return
        if self.mpl_canvas is None:
            self._configure_mplcanvas()
        else:
            self.clear_glyphs()

        # Plot one RMS curve per overlay so the viewer shows all overlays.
        self.rms = []
        self._peak_vertices = {}
        multi = len(self._all_data) > 1
        for overlay_key, overlay_data in self._all_data.items():
            y_parts = []
            for hemi_key in ["lh", "rh", "vol"]:
                hemi_data = overlay_data.get(hemi_key)
                if hemi_data is None:
                    continue
                arr = hemi_data["array"]
                if arr.ndim == 1:
                    continue  # static data — no time axis
                if arr.ndim == 3:
                    arr = np.linalg.norm(arr, axis=1)
                y_parts.append(arr)
            if not y_parts:
                continue
            y = np.concatenate(y_parts)
            rms = np.linalg.norm(y, axis=0) / np.sqrt(len(y))
            del y
            label = f"RMS ({overlay_key})" if multi else "RMS"
            (line,) = self.mpl_canvas.axes.plot(
                overlay_data["time"],
                rms,
                lw=3.5,
                label=label,
                zorder=3,
                color=next(self.color_cycle),
                alpha=0.5,
            )
            self.rms.append(line)

        # now plot the time line
        self.plot_time_line(update=False)

        # then the picked points
        self._update_peak_vertices()

    def _update_peak_vertices(self):
        """(Re)compute the peak vertex per hemi for the active overlay."""
        if self.traces_mode != "vertex":
            # in label mode a VertexSelect would toggle the label containing
            # the peak (and label extraction may not even be possible, e.g.,
            # data added without an src)
            return
        old_peak_vertices = self._peak_vertices
        self._peak_vertices = {}
        for idx, hemi in enumerate(["lh", "rh", "vol"]):
            act_data = self.act_data_smooth.get(hemi, [None])[0]
            if act_data is None:
                continue
            hemi_data = self._data[hemi]
            vertices = hemi_data["vertices"]

            # simulate a picked renderer
            if self._hemi in ("both", "rh") or hemi == "vol":
                idx = 0
            self._picked_renderer = self._renderer._all_renderers[idx]

            # initialize the default point
            if self._data["initial_time"] is not None:
                # pick at that time
                use_data = act_data[:, [np.round(self._data["time_idx"]).astype(int)]]
            else:
                use_data = act_data
            ind = np.unravel_index(
                np.argmax(np.abs(use_data), axis=None), use_data.shape
            )
            vertex_id = vertices[ind[0]]
            self._peak_vertices[hemi] = vertex_id

            old_vertex_id = old_peak_vertices.get(hemi)
            if old_vertex_id == vertex_id:
                # same peak vertex but possibly different data: refresh the
                # auto-picked trace in place (manually picked traces keep
                # showing the overlay they were picked from)
                spheres = self._picked_points.get((hemi, vertex_id))
                if (hemi, vertex_id) in self._auto_peak_points and spheres is not None:
                    spheres[0]["line"].set_ydata(
                        self._vertex_trace_data(hemi, vertex_id)
                    )
                continue
            was_auto_picked = (hemi, old_vertex_id) in self._auto_peak_points
            if old_vertex_id is not None and was_auto_picked:
                self._remove_vertex_glyph(hemi=hemi, vertex_id=old_vertex_id)
            # a vertex the user already picked stays a manual pick (and must
            # not be auto-removed on the next overlay switch)
            if (hemi, vertex_id) not in self._picked_points:
                self._auto_peak_points.add((hemi, vertex_id))
            publish(
                self,
                VertexSelect(hemi=hemi, vertex_id=vertex_id, source_id=ind[0]),
            )
        if self.mpl_canvas is not None:
            self.mpl_canvas.sync_traces()

    def _vertex_trace_data(self, hemi, vertex_id):
        """Get the active overlay's time course at a mesh vertex."""
        act_data, smooth = self.act_data_smooth[hemi]
        if smooth is not None:
            act_data = (smooth[[vertex_id]] @ act_data)[0]
        else:  # full-resolution data
            act_data = act_data[vertex_id].copy()
        return act_data

    def _update_act_data_smooth(self):
        # get data for each hemi
        from scipy.sparse import csr_array

        for hemi in ["vol", "lh", "rh"]:
            hemi_data = self._data.get(hemi)
            if hemi_data is not None:
                act_data = hemi_data["array"]
                if act_data.ndim == 3:
                    act_data = np.linalg.norm(act_data, axis=1)
                smooth_mat = hemi_data.get("smooth_mat")
                vertices = hemi_data["vertices"]
                if hemi == "vol":
                    assert smooth_mat is None
                    smooth_mat = csr_array(
                        (np.ones(len(vertices)), (vertices, np.arange(len(vertices))))
                    )
                self.act_data_smooth[hemi] = (act_data, smooth_mat)

    def _configure_picking(self):
        self._update_act_data_smooth()

        self._renderer._update_picking_callback(
            self._on_mouse_move,
            self._on_button_press,
            self._on_button_release,
            self._on_pick,
        )
        subscribe(self, "vertex_select", self._on_vertex_select)

    def _configure_hover(self):
        self._hover_caption = self._create_caption()
        self.plotter.add_actor(
            self._hover_caption,
            name=None,
            culling=False,
            pickable=False,
            reset_camera=False,
            render=False,
        )

        @_auto_weakref
        def on_surface_hover(iren, event):
            self._on_surface_hover(iren, event)

        self.plotter.iren.add_observer("MouseMoveEvent", on_surface_hover)

    def _on_surface_hover(self, iren, event):  # event == "MouseMoveEvent"
        if not self._show_hover_info:
            return
        from pyvista import DataSetMapper

        x, y = iren.GetEventPosition()
        picked_renderer = iren.FindPokedRenderer(x, y)
        vtk_picker = self._renderer._hover_picker
        vtk_picker.Pick(x, y, 0, picked_renderer)
        cell_id = vtk_picker.GetCellId()
        mapper = vtk_picker.GetMapper()
        if not isinstance(mapper, DataSetMapper) or cell_id == -1:
            if self._hover_caption.GetVisibility():
                self._hover_caption.SetVisibility(False)
                self._renderer._update()
            return  # didn't find a mesh
        for _, this_mesh in self.layered_meshes.items():
            if this_mesh._polydata is mapper.dataset:
                mesh = this_mesh._polydata
                break
        else:
            return
        pos = np.array(vtk_picker.GetPickPosition())
        vtk_cell = mesh.GetCell(cell_id)
        cell = [
            vtk_cell.GetPointId(point_id)
            for point_id in range(vtk_cell.GetNumberOfPoints())
        ]
        vert_pos = mesh.points[cell]
        vertex_id = cell[np.argmin(np.linalg.norm(vert_pos - pos, axis=1))]
        _, _, azimuth, elevation, _ = self._renderer.get_camera(rigid=self._rigid)
        text = (
            f"vertex {vertex_id}\n"
            f"({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) mm\n"
            f"az {azimuth:.0f}\N{DEGREE SIGN}  el {elevation:.0f}\N{DEGREE SIGN}"
        )
        self._hover_caption.SetCaption(text)
        self._hover_caption.SetAttachmentPoint(*pos)
        self._hover_caption.SetVisibility(True)
        actor = self._hover_caption.GetTextActor()
        wh = np.zeros(2)
        actor.GetSize(self.plotter.renderer, wh)
        self._hover_caption.SetPosition2(wh)
        self._renderer._update()

    def _toggle_hover_info(self):
        self._show_hover_info = not self._show_hover_info
        if not self._show_hover_info and self._hover_caption is not None:
            self._hover_caption.SetVisibility(False)
            self._renderer._update()

    def _configure_tool_bar(self):
        if not hasattr(self._renderer, "_tool_bar") or self._renderer._tool_bar is None:
            self._renderer._tool_bar_initialize(name="Toolbar")

        @_auto_weakref
        def save_image(filename):
            self.save_image(filename)

        self._renderer._tool_bar_add_file_button(
            name="screenshot",
            desc="Take a screenshot",
            func=save_image,
        )

        @_auto_weakref
        def save_movie(filename):
            self.save_movie(
                filename=filename, time_dilation=(1.0 / self.playback_speed)
            )

        self._renderer._tool_bar_add_file_button(
            name="movie",
            desc="Save movie...",
            func=save_movie,
            shortcut="ctrl+shift+s",
        )
        self._renderer._tool_bar_add_button(
            name="visibility",
            desc="Toggle Controls",
            func=self.toggle_interface,
            icon_name="visibility_on",
        )
        self._renderer._tool_bar_add_button(
            name="scale",
            desc="Auto-Scale",
            func=self.apply_auto_scaling,
        )
        self._renderer._tool_bar_add_button(
            name="clear",
            desc="Clear traces",
            func=self.clear_glyphs,
        )
        self._renderer._tool_bar_add_button(
            name="hover_info",
            desc="Toggle vertex/camera hover info",
            func=self._toggle_hover_info,
            icon_name="information",
        )
        self._renderer._tool_bar_add_spacer()
        self._renderer._tool_bar_add_button(
            name="help",
            desc="Help",
            func=self.help,
            shortcut="?",
        )

    def _rotate_camera(self, which, value):
        _, _, azimuth, elevation, _ = self._renderer.get_camera(rigid=self._rigid)
        kwargs = dict(update=True)
        if which == "azimuth":
            value = azimuth + value
            # Our view_up threshold is 5/175, so let's be safe here
            if elevation < 7.5 or elevation > 172.5:
                kwargs["elevation"] = np.clip(elevation, 10, 170)
        else:
            value = np.clip(elevation + value, 10, 170)
        kwargs[which] = value
        self._set_camera(**kwargs)

    def _configure_shortcuts(self):
        # Remove the default key binding
        if getattr(self.plotter, "iren", None) is not None:
            self.plotter.iren.clear_key_event_callbacks()
        # Then, we add our own:
        self.plotter.add_key_event("i", self.toggle_interface)
        self.plotter.add_key_event("s", self.apply_auto_scaling)
        self.plotter.add_key_event("r", self.restore_user_scaling)
        self.plotter.add_key_event("c", self.clear_glyphs)
        self.plotter.add_key_event("v", self._toggle_hover_info)
        self._configure_arrow_keys()

    def _configure_arrow_keys(self):
        """(Re)bind the arrow keys, which cannot rotate a flat patch."""
        if getattr(self.plotter, "iren", None) is None:
            return
        for key, which, amt in (
            ("Left", "azimuth", 10),
            ("Right", "azimuth", -10),
            ("Up", "elevation", 10),
            ("Down", "elevation", -10),
        ):
            # always clear, so PyVista's own bindings cannot rotate a flat map
            self.plotter.clear_events_for_key(key)
            if self._surf != "flat":
                func = partial(self._rotate_camera, which, amt)
                self.plotter.add_key_event(key, func)

    def _configure_status_bar(self):
        self._renderer._status_bar_initialize()
        self.status_msg = self._renderer._status_bar_add_label(
            self.default_status_bar_msg, stretch=1, on_click=self.help
        )
        self.status_progress = self._renderer._status_bar_add_progress_bar()
        if self.status_progress is not None:
            self.status_progress.hide()

    def _on_mouse_move(self, vtk_picker, event):
        if self._mouse_no_mvt:
            self._mouse_no_mvt -= 1

    def _on_button_press(self, vtk_picker, event):
        self._mouse_no_mvt = 2

    def _on_button_release(self, vtk_picker, event):
        if self._mouse_no_mvt > 0:
            x, y = vtk_picker.GetEventPosition()
            # programmatically detect the picked renderer
            self._picked_renderer = self.plotter.iren.interactor.FindPokedRenderer(x, y)
            # trigger the pick
            self._renderer._picker.Pick(x, y, 0, self._picked_renderer)
        self._mouse_no_mvt = 0

    def _on_pick(self, vtk_picker, event):
        if not self.show_traces:
            return

        # vtk_picker is a vtkCellPicker
        cell_id = vtk_picker.GetCellId()
        mesh = vtk_picker.GetDataSet()

        if mesh is None or cell_id == -1 or not self._mouse_no_mvt:
            return  # don't pick

        # 1) Check to see if there are any spheres along the ray and remove if so
        if len(self._picked_points):
            collection = vtk_picker.GetProp3Ds()
            for ii in range(collection.GetNumberOfItems()):
                actor = collection.GetItemAsObject(ii)
                for (hemi, vertex_id), spheres in self._picked_points.items():
                    if any(sphere["actor"] is actor for sphere in spheres):
                        self._remove_vertex_glyph(hemi=hemi, vertex_id=vertex_id)
                        return

        # 2) Otherwise, pick the objects in the scene
        # PyVista can give the actor's mapper an internal copy of our polydata
        # (e.g., when RGBA scalars are used), in which case the picked dataset
        # is not _polydata itself, so compare against the mapper's dataset too
        mapper_dataset = getattr(vtk_picker.GetMapper(), "dataset", None)
        for hemi, this_mesh in self.layered_meshes.items():
            assert hemi in ("lh", "rh"), f"Unexpected {hemi=}"
            if this_mesh._polydata is mesh or this_mesh._polydata is mapper_dataset:
                mesh = this_mesh._polydata
                break
        else:
            hemi = "vol"
        if self.act_data_smooth[hemi][0] is None:  # no data to add for hemi
            return
        pos = np.array(vtk_picker.GetPickPosition())
        if hemi == "vol":
            # VTK will give us the point closest to the viewer in the vol.
            # We want to pick the point with the maximum value along the
            # camera-to-click array, which fortunately we can get "just"
            # by inspecting the points that are sufficiently close to the
            # ray.
            grid = self._data[hemi]["grid"]
            vertices = self._data[hemi]["vertices"]
            coords = self._data[hemi]["grid_coords"][vertices]
            scalars = grid.point_data["values"][vertices]
            spacing = np.array(grid.GetSpacing())
            max_dist = np.max(spacing) / 2.0
            origin = vtk_picker.GetRenderer().GetActiveCamera().GetPosition()
            ori = pos - origin
            ori /= np.linalg.norm(ori)
            # the magic formula: distance from a ray to a given point
            dists = np.linalg.norm(np.cross(ori, coords - pos), axis=1)
            assert dists.shape == (len(coords),)
            mask = dists <= max_dist
            idx = np.where(mask)[0]
            if len(idx) == 0:
                return  # weird point on edge of volume?
            # useful for debugging the ray by mapping it into the volume, should
            # create a blob near the click point:
            # dists = dists - dists.min()
            # dists = (1. - dists / dists.max()) * self._cmap_range[1]
            # grid.point_data['values'][vertices] = dists * mask
            source_id = idx[np.argmax(np.abs(scalars[idx]))]
            vertex_id = vertices[source_id]
            # Naive way: convert pos directly to idx; i.e., apply mri_src_t
            # shape = self._data[hemi]['grid_shape']
            # taking into account the cell vs point difference (spacing/2)
            # shift = np.array(grid.GetOrigin()) + spacing / 2.
            # ijk = np.round((pos - shift) / spacing).astype(int)
            # vertex_id = np.ravel_multi_index(ijk, shape, order='F')
        else:
            vtk_cell = mesh.GetCell(cell_id)
            cell = [
                vtk_cell.GetPointId(point_id)
                for point_id in range(vtk_cell.GetNumberOfPoints())
            ]
            vert_pos = mesh.points[cell]
            vertex_id = cell[np.argmin(np.linalg.norm(vert_pos - pos, axis=1))]

            # retrieve the nearest source_id from the smooth_mat
            smooth_mat = self.act_data_smooth[hemi][1]
            if smooth_mat is None:  # full-resolution data, no smoothing matrix
                source_id = vertex_id
            else:
                row = smooth_mat[vertex_id]
                source_id = row.argmax() if row.nnz else None

        publish(self, VertexSelect(hemi=hemi, vertex_id=vertex_id, source_id=source_id))

    def _on_vertex_select(self, event):
        """Respond to vertex_select UI event."""
        if self._data is None:
            return
        if event.hemi == "vol":
            try:
                mesh = self._data[event.hemi]["grid"]
            except KeyError:
                return
        else:
            try:
                mesh = self.layered_meshes[event.hemi]._polydata
            except KeyError:
                return
        if self.traces_mode == "label":
            self._add_label_glyph(event.hemi, mesh, event.vertex_id)
        else:
            self._add_vertex_glyph(event.hemi, mesh, event.vertex_id)

    def _add_label_glyph(self, hemi, mesh, vertex_id):
        if hemi == "vol":
            return
        label_id = self._vertex_to_label_id[hemi][vertex_id]
        label = self._annotation_labels[hemi][label_id]

        # remove the patch if already picked
        if label_id in self._picked_patches[hemi]:
            self._remove_label_glyph(hemi, label_id)
            return

        if hemi == label.hemi:
            self.add_label(label, borders=True)
            self._picked_patches[hemi].append(label_id)

    def _remove_label_glyph(self, hemi, label_id):
        label = self._annotation_labels[hemi][label_id]
        # do the bookkeeping first so that a failure partway cannot leave a
        # picked label whose line is already detached, which would make every
        # subsequent removal (and clear_glyphs at annotation changes) fail too
        self._picked_patches[hemi].remove(label_id)
        line, label._line = label._line, None
        self._label_trace_meta.pop(line, None)
        if line is not None:
            try:
                line.remove()
            except ValueError:  # already detached from the axes
                pass
        self.color_cycle.restore(label._color)
        self.mpl_canvas.update_plot()
        self.layered_meshes[hemi].remove_overlay(label.name)
        self._renderer._update()  # mirrors add_label; see _add_vertex_glyph

    def _add_vertex_glyph(self, hemi, mesh, vertex_id, update=True):
        _ensure_int(vertex_id)
        if (hemi, vertex_id) in self._picked_points:
            return

        # skip if the wrong hemi is selected
        if self.act_data_smooth[hemi][0] is None:
            return
        color = next(self.color_cycle)
        line = self.plot_time_course(hemi, vertex_id, color, update=update)
        if hemi == "vol":
            ijk = np.unravel_index(vertex_id, mesh.dimensions, order="F")
            voxel = mesh.GetCell(*ijk)
            center = np.empty(3)
            voxel.GetCentroid(center)
            center -= np.array(mesh.spacing, float) / 2.0
            # In case we ever need to debug problems with this, the following code
            # can be uncommented (it should match!)
            # want = apply_trans(self._data[hemi]["grid_src_mri_t"], ijk)
            # assert np.allclose(center, want, atol=1e-6), f"{center=} vs {want=}"
            #
            # And to make the value at vert 21334 for fsaverage-5 visible:
            # self._renderer._sphere(
            #     center=[0, -5, 5],
            #     color="w",
            #     radius=3.0,
            # )
        else:
            center = mesh.GetPoints().GetPoint(vertex_id)
        del mesh

        # from the picked renderer to the subplot coords
        try:
            lst = self._renderer._all_renderers._renderers
        except AttributeError:
            lst = self._renderer._all_renderers
        rindex = lst.index(self._picked_renderer)
        row, col = self._renderer._index_to_loc(rindex)

        is_peak = self._peak_vertices.get(hemi) == vertex_id
        spheres = list()
        for _ in self._iter_views(hemi):
            # Using _sphere() instead of renderer.sphere() for 2 reasons:
            # 1) renderer.sphere() fails on Windows in a scenario where a lot
            #    of picking requests are done in a short span of time (could be
            #    mitigated with synchronization/delay?)
            # 2) the glyph filter is used in renderer.sphere() but only one
            #    sphere is required in this function.
            actor, mesh = self._renderer._sphere(
                center=np.array(center),
                color=color,
                radius=4.5 if is_peak else 3.0,
                resolution=24 if is_peak else 8,
            )
            if is_peak:
                prop = actor.GetProperty()
                prop.SetSpecular(0.6)
                prop.SetSpecularPower(40)
                prop.SetSpecularColor(1, 1, 1)
            spheres.append(dict(mesh=mesh, actor=actor))

        # add metadata for picking
        for sphere in spheres:
            sphere.update(hemi=hemi, line=line, color=color, vertex_id=vertex_id)

        _ensure_int(vertex_id)
        self._picked_points[(hemi, vertex_id)] = spheres
        if update:
            self._renderer._update()
        return sphere

    def _remove_vertex_glyph(self, *, hemi, vertex_id, render=True):
        _ensure_int(vertex_id)
        assert isinstance(hemi, str), f"got {type(hemi)} for {hemi=}"
        # When linked via _LinkViewer, removing a vertex on one brain cascades
        # to all linked brains, so by the time a given brain's own loop (e.g.
        # in clear_glyphs) reaches this (hemi, vertex_id) it may already be
        # gone; just no-op in that case.
        self._auto_peak_points.discard((hemi, vertex_id))
        spheres = self._picked_points.pop((hemi, vertex_id), None)
        if spheres is None:
            return
        color, line = spheres[0]["color"], spheres[0]["line"]
        line.remove()
        self._trace_meta.pop(line, None)
        self.mpl_canvas.update_plot()

        with warnings.catch_warnings(record=True):
            # We intentionally ignore these in case we have traversed the
            # entire color cycle
            warnings.simplefilter("ignore")
            self.color_cycle.restore(color)
        for sphere in spheres:
            # remove all actors
            self.plotter.remove_actor(sphere.pop("actor"), render=False)
        if render:
            self._renderer._update()

    def _set_trace_visible(self, line, visible):
        """Toggle a trace's 3D glyph visibility to match its plot visibility."""
        for spheres in self._picked_points.values():
            if spheres[0]["line"] is line:
                for sphere in spheres:
                    sphere["actor"].SetVisibility(visible)
                self._renderer._update()
                return

    def _set_trace_highlight(self, line):
        """Dim the 3D glyphs of every picked trace except the highlighted one."""
        if not self._picked_points:
            return
        for spheres in self._picked_points.values():
            opacity = 1.0 if line in (None, spheres[0]["line"]) else 0.3
            for sphere in spheres:
                sphere["actor"].GetProperty().SetOpacity(opacity)
        self._renderer._update()

    def _trace_display_label(self, line):
        """Return a short, dock-friendly trace-list label.

        The vertex auto-picked at peak activation for each hemisphere gets a
        "Peak (LH) 1000"-style name; other picked vertices get a compact
        "LH 1000"-style name instead of the full MNI-coordinate string (still
        available as the row's tooltip). A picked label gets a
        "superiortemporal (LH)"-style name, moving its name's hemisphere
        suffix into the parentheses. RMS curves are returned unchanged.
        """
        meta = self._trace_meta.get(line)
        if meta is not None:
            hemi, vertex_id, _ = meta
            hemi_names = {"lh": "LH", "rh": "RH", "vol": "Vol"}
            if self._peak_vertices.get(hemi) == vertex_id:
                return f"Peak ({hemi_names[hemi]}) {vertex_id}"
            return f"{hemi_names[hemi]} {vertex_id}"
        label_meta = self._label_trace_meta.get(line)
        if label_meta is not None:
            hemi, label_name, _, _ = label_meta
            return f"{label_name.removesuffix(f'-{hemi}')} ({hemi.upper()})"
        return line.get_label()

    def _trace_display_subtitle(self, line):
        """Return an optional small subtitle line for a trace-list row."""
        meta = self._trace_meta.get(line)
        if meta is not None:
            mni_str = meta[2]
            return f"MNI: {mni_str}" if mni_str else None
        label_meta = self._label_trace_meta.get(line)
        if label_meta is not None:
            _, _, mode, n_vertices = label_meta
            return f"{n_vertices} vertices, mode: {mode}"
        return None

    def clear_glyphs(self):
        """Clear the picking glyphs."""
        if not self.time_viewer:
            return
        for hemi, vertex_id in list(self._picked_points):
            self._remove_vertex_glyph(hemi=hemi, vertex_id=vertex_id, render=False)
        assert len(self._picked_points) == 0
        for hemi in self._hemis:
            for label_id in list(self._picked_patches[hemi]):
                self._remove_label_glyph(hemi, label_id)
        assert sum(len(v) for v in self._picked_patches.values()) == 0
        if self.rms is not None:
            for line in self.rms:
                line.remove()
                self.color_cycle.restore(line.get_color())
            self.rms = None
        self._renderer._update()

    @fill_doc_static("brain_update")
    def plot_time_course(self, hemi, vertex_id, color, update=True):
        """Plot the vertex time course.

        Parameters
        ----------
        hemi : str
            The hemisphere id of the vertex.
        vertex_id : int
            The vertex identifier in the mesh.
        color : matplotlib color
            The color of the time course.
        update : bool
            Force an update of the plot. Defaults to True.

        Returns
        -------
        line : matplotlib object
            The time line object.
        """
        if self.mpl_canvas is None:
            return
        time = self._data["time"].copy()  # avoid circular ref
        mni = None
        if hemi == "vol":
            hemi_str = "V"
            xfm = read_talxfm(self._subject, self._subjects_dir)
            if self._units == "mm":
                xfm["trans"][:3, 3] *= 1000.0
            ijk = np.unravel_index(vertex_id, self._data[hemi]["grid_shape"], order="F")
            src_mri_t = self._data[hemi]["grid_src_mri_t"]
            mni = apply_trans(xfm["trans"] @ src_mri_t, ijk)
        else:
            hemi_str = "L" if hemi == "lh" else "R"
            try:
                mni = vertex_to_mni(
                    vertices=vertex_id,
                    hemis=0 if hemi == "lh" else 1,
                    subject=self._subject,
                    subjects_dir=self._subjects_dir,
                )
            except Exception:
                mni = None
        if mni is not None:
            mni_str = ", ".join(f"{m:5.1f}" for m in mni)
            mni_suffix = " MNI: " + mni_str
        else:
            mni_str = None
            mni_suffix = ""
        label = f"{hemi_str}:{str(vertex_id).ljust(6)}{mni_suffix}"
        act_data = self._vertex_trace_data(hemi, vertex_id)
        line = self.mpl_canvas.plot(
            time,
            act_data,
            label=label,
            lw=1.8,
            color=color,
            zorder=4,
            update=False,
        )
        self._trace_meta[line] = (hemi, vertex_id, mni_str)
        if update:
            self.mpl_canvas.axes.relim()
            self.mpl_canvas.axes.autoscale_view()
            self.mpl_canvas.update_plot()
        return line

    @fill_doc_static("brain_update")
    def plot_time_line(self, update=True):
        """Add the time line to the MPL widget.

        Parameters
        ----------
        update : bool
            Force an update of the plot. Defaults to True.
        """
        if self.mpl_canvas is None:
            return
        if isinstance(self.show_traces, bool) and self.show_traces:
            # add time information
            current_time = self._current_time
            if not hasattr(self, "time_line"):
                self.time_line = self.mpl_canvas.plot_time_line(
                    x=current_time,
                    label="time",
                    color=self._fg_color,
                    lw=1.5,
                    ls="--",
                    alpha=0.7,
                    update=update,
                )
            self.time_line.set_xdata([current_time])
            if update:
                # only the time line moved, so the rest of the figure can be
                # blitted from the cached background instead of being redrawn
                self.mpl_canvas.update_blit_artists()

    def _configure_help(self):
        pairs = [
            ("?", "Display help window"),
            ("i", "Toggle interface"),
            ("s", "Apply auto-scaling"),
            ("r", "Restore original clim"),
            ("c", "Clear all traces"),
            ("v", "Toggle vertex/camera hover info"),
            ("n", "Shift the time forward by the playback speed"),
            ("b", "Shift the time backward by the playback speed"),
            ("Space", "Start/Pause playback"),
        ]
        if self._surf == "flat":
            # a flat map is 2D: the arrow keys are not bound and the camera
            # uses the rubber-band style rather than rotation
            mouse_pairs = [
                ("Middle-click-and-drag", "Pan the view"),
                ("Right-click-and-drag / scroll", "Zoom the view"),
            ]
        else:
            pairs += [
                ("Up", "Decrease camera elevation angle"),
                ("Down", "Increase camera elevation angle"),
                ("Left", "Decrease camera azimuth angle"),
                ("Right", "Increase camera azimuth angle"),
            ]
            mouse_pairs = [
                ("Left-click-and-drag", "Rotate the view"),
                ("Middle-click-and-drag", "Pan the view"),
                ("Right-click-and-drag / scroll", "Zoom the view"),
            ]
        if self.help_canvas is not None:  # rebuilt when the bindings change
            close = getattr(self.help_canvas, "close", None)
            if close is not None:
                close()
        self.help_canvas = self._renderer._window_get_help_canvas(pairs, mouse_pairs)

    def help(self):
        """Display the help window."""
        self.help_canvas.show()

    def _configure_label_time_course(self):
        from ...label import read_labels_from_annot

        if not self.show_traces:
            return
        if self.mpl_canvas is None:
            self._configure_mplcanvas()
        else:
            self.clear_glyphs()
        self.traces_mode = "label"
        self.add_annotation(self.annot, color="w", alpha=0.75)

        # now plot the time line
        self.plot_time_line(update=False)
        self.mpl_canvas.update_plot()

        for hemi in self._hemis:
            labels = read_labels_from_annot(
                subject=self._subject,
                parc=self.annot,
                hemi=hemi,
                subjects_dir=self._subjects_dir,
            )
            self._vertex_to_label_id[hemi] = np.full(self.geo[hemi].coords.shape[0], -1)
            self._annotation_labels[hemi] = labels
            for idx, label in enumerate(labels):
                self._vertex_to_label_id[hemi][label.vertices] = idx

    def _update_fscale(self, fscale):
        """Scale the colorbar points."""
        fmin = self._data["fmin"] * fscale
        fmid = self._data["fmid"] * fscale
        fmax = self._data["fmax"] * fscale
        self.update_lut(fmin=fmin, fmid=fmid, fmax=fmax)

    def get_picked_points(self):
        """Return the vertices of the picked points.

        Returns
        -------
        points : dict | None
            The vertices picked by the time viewer, one key per hemisphere with
            a list of vertex indices.
        """
        out = dict(lh=[], rh=[], vol=[])
        for hemi, vertex_id in self._picked_points:
            out[hemi].append(vertex_id)
        return out


def _get_range(brain):
    """Get the data limits.

    Since they may be very small (1E-10 and such), we apply a scaling factor
    such that the data range lies somewhere between 0.01 and 100. This makes
    for more usable sliders. When setting a value on the slider, the value is
    multiplied by the scaling factor and when getting a value, this value
    should be divided by the scaling factor.
    """
    fmax = abs(brain._data["fmax"])
    if 1e-02 <= fmax <= 1e02:
        fscale_power = 0
    else:
        fscale_power = int(np.log10(max(fmax, np.finfo("float32").smallest_normal)))
        if fscale_power < 0:
            fscale_power -= 1
    fscale = 10**-fscale_power
    return fmax, fscale, fscale_power
