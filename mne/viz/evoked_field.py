"""Class to draw evoked MEG and EEG fieldlines, with a GUI to control the figure.

author: Marijn van Vliet <w.m.vanvliet@gmail.com>
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from copy import deepcopy
from functools import partial

import numpy as np
from scipy.interpolate import interp1d

from .._fiff.pick import pick_types
from ..defaults import DEFAULTS
from ..utils import (
    _auto_weakref,
    _check_option,
    _ensure_int,
    _to_rgb,
    _validate_type,
    fill_doc,
)
from ._3d_overlay import _LayeredMesh
from .ui_events import (
    ColormapRange,
    Contours,
    TimeChange,
    disable_ui_events,
    publish,
    subscribe,
)
from .utils import mne_analyze_colormap


@fill_doc
class EvokedField:
    """Plot MEG/EEG fields on head surface and helmet in 3D.

    Parameters
    ----------
    evoked : instance of mne.Evoked
        The evoked object.
    surf_maps : list
        The surface mapping information obtained with make_field_map.
    time : float | None
        The time point at which the field map shall be displayed. If None,
        the average peak latency (across sensor types) is used.
    time_label : str | None
        How to print info about the time instant visualized.
    %(n_jobs)s
    fig : instance of Figure3D | None
        If None (default), a new figure will be created, otherwise it will
        plot into the given figure.

        .. versionadded:: 0.20
    vmax : float | dict | None
        Maximum intensity. Can be a dictionary with two entries ``"eeg"`` and ``"meg"``
        to specify separate values for EEG and MEG fields respectively. Can be
        ``None`` to use the maximum value of the data.

        .. versionadded:: 0.21
        .. versionadded:: 1.4
            ``vmax`` can be a dictionary to specify separate values for EEG and
            MEG fields.
    n_contours : int
        The number of contours.

        .. versionadded:: 0.21
    show_density : bool
        Whether to draw the field density as an overlay on top of the helmet/head
        surface. Defaults to ``True``.
    alpha : float | dict | None
        Opacity of the meshes (between 0 and 1). Can be a dictionary with two
        entries ``"eeg"`` and ``"meg"`` to specify separate values for EEG and
        MEG fields respectively. Can be ``None`` to use 1.0 when a single field
        map is shown, or ``dict(eeg=1.0, meg=0.5)`` when both field maps are shown.

        .. versionadded:: 1.4
    %(interpolation_brain_time)s

        .. versionadded:: 1.6
    %(interaction_scene)s
        Defaults to ``'terrain'``.

        .. versionadded:: 1.1
    time_viewer : bool | str
        Display time viewer GUI. Can also be ``"auto"``, which will mean
        ``True`` if there is more than one time point and ``False`` otherwise.

        .. versionadded:: 1.6
    %(verbose)s

    Notes
    -----
    The figure will publish and subscribe to the following UI events:

    * :class:`~mne.viz.ui_events.TimeChange`
    * :class:`~mne.viz.ui_events.Contours`, ``kind="field_strength_meg" | "field_strength_eeg"``
    * :class:`~mne.viz.ui_events.ColormapRange`, ``kind="field_strength_meg" | "field_strength_eeg"``
    """  # noqa

    def __init__(
        self,
        evoked,
        surf_maps,
        *,
        time=None,
        time_label="t = %0.0f ms",
        n_jobs=None,
        fig=None,
        vmax=None,
        n_contours=21,
        show_density=True,
        alpha=None,
        interpolation="nearest",
        interaction="terrain",
        time_viewer="auto",
        verbose=None,
    ):
        from .backends.renderer import _get_3d_backend, _get_renderer

        # Setup figure parameters
        self._evoked = evoked
        if time is None:
            types = [t for t in ["eeg", "grad", "mag"] if t in evoked]
            time = np.mean([evoked.get_peak(ch_type=t)[1] for t in types])
        self._current_time = time
        if not evoked.times[0] <= time <= evoked.times[-1]:
            raise ValueError(f"`time` ({time:0.3f}) must be inside `evoked.times`")
        self._time_label = time_label

        self._vmax = _validate_type(vmax, (None, "numeric", dict), "vmax")
        self._n_contours = _ensure_int(n_contours, "n_contours")
        self._time_interpolation = _check_option(
            "interpolation",
            interpolation,
            ("linear", "nearest", "zero", "slinear", "quadratic", "cubic"),
        )
        self._interaction = _check_option(
            "interaction", interaction, ["trackball", "terrain"]
        )

        surf_map_kinds = [surf_map["kind"] for surf_map in surf_maps]
        if vmax is None:
            self._vmax = {kind: None for kind in surf_map_kinds}
        elif isinstance(vmax, dict):
            for kind in surf_map_kinds:
                if kind not in vmax:
                    raise ValueError(
                        f'No entry for "{kind}" found in the vmax dictionary'
                    )
            self._vmax = vmax
        else:  # float value
            self._vmax = {kind: vmax for kind in surf_map_kinds}

        if alpha is None:
            self._alpha = {
                surf_map["kind"]: val for surf_map, val in zip(surf_maps, [1.0, 0.5])
            }
        elif isinstance(alpha, dict):
            for kind in surf_map_kinds:
                if kind not in alpha:
                    raise ValueError(
                        f'No entry for "{kind}" found in the alpha dictionary'
                    )
            self._alpha = alpha
        else:  # float value
            self._alpha = {kind: alpha for kind in surf_map_kinds}

        self._colors = [(0.6, 0.6, 0.6), (1.0, 1.0, 1.0)]
        self._colormap = mne_analyze_colormap(format="vtk")
        self._colormap_lines = np.concatenate(
            [
                np.tile([0.0, 0.0, 255.0, 255.0], (127, 1)),
                np.tile([0.0, 0.0, 0.0, 255.0], (2, 1)),
                np.tile([255.0, 0.0, 0.0, 255.0], (127, 1)),
            ]
        )
        self._show_density = show_density

        from ._brain import Brain

        if isinstance(fig, Brain):
            self._renderer = fig._renderer
            self._in_brain_figure = True
            self._units = fig._units
            if _get_3d_backend() == "notebook":
                raise NotImplementedError(
                    "Plotting on top of an existing Brain figure "
                    "is currently not supported inside a notebook."
                )
        else:
            self._renderer = _get_renderer(
                fig, bgcolor=(0.0, 0.0, 0.0), size=(600, 600)
            )
            self._in_brain_figure = False
            self._units = "m"

        self.plotter = self._renderer.plotter
        self.interaction = interaction

        # Prepare the surface maps
        self._surf_maps = [
            self._prepare_surf_map(surf_map, color, self._alpha[surf_map["kind"]])
            for surf_map, color in zip(surf_maps, self._colors)
        ]

        # Do we want the time viewer?
        if time_viewer == "auto":
            time_viewer = len(evoked.times) > 1
        self.time_viewer = time_viewer

        # Configure UI events
        @_auto_weakref
        def current_time_func():
            return self._current_time

        self._widgets = dict()
        if self.time_viewer:
            # Draw widgets only if not inside a figure that already has them.
            if (
                not hasattr(self._renderer, "_widgets")
                or "time_slider" not in self._renderer._widgets
            ):
                self._renderer._enable_time_interaction(
                    self,
                    current_time_func=current_time_func,
                    times=evoked.times,
                )
            if not self._in_brain_figure or "time_slider" not in fig.widgets:
                # Draw the time label
                self._time_label = time_label
                if time_label is not None:
                    if "%" in time_label:
                        time_label = time_label % np.round(1e3 * time)
                    self._time_label_actor = self._renderer.text2d(
                        x_window=0.01, y_window=0.01, text=time_label
                    )
            self._configure_dock()

        subscribe(self, "time_change", self._on_time_change)
        subscribe(self, "colormap_range", self._on_colormap_range)
        subscribe(self, "contours", self._on_contours)

        if not self._in_brain_figure:
            self._renderer.set_interaction(interaction)
            self._renderer.set_camera(azimuth=10, elevation=60, distance="auto")
            self._renderer.show()

    def _prepare_surf_map(self, surf_map, color, alpha):
        """Compute all the data required to render a fieldlines map."""
        if surf_map["kind"] == "eeg":
            pick = pick_types(self._evoked.info, meg=False, eeg=True)
        else:
            pick = pick_types(self._evoked.info, meg=True, eeg=False, ref_meg=False)

        evoked_ch_names = set([self._evoked.ch_names[k] for k in pick])
        map_ch_names = set(surf_map["ch_names"])
        if evoked_ch_names != map_ch_names:
            message = ["Channels in map and data do not match."]
            diff = map_ch_names - evoked_ch_names
            if len(diff):
                message += [f"{list(diff)} not in data file. "]
            diff = evoked_ch_names - map_ch_names
            if len(diff):
                message += [f"{list(diff)} not in map file."]
            raise RuntimeError(" ".join(message))

        data = surf_map["data"] @ self._evoked.data[pick]
        data_interp = interp1d(
            self._evoked.times,
            data,
            kind=self._time_interpolation,
            assume_sorted=True,
        )
        current_data = data_interp(self._current_time)

        # Make a solid surface
        surf = surf_map["surf"]
        if self._units == "mm":
            surf = deepcopy(surf)
            surf["rr"] *= 1000
        map_vmax = self._vmax.get(surf_map["kind"])
        if map_vmax is None:
            map_vmax = float(np.max(current_data))
        mesh = _LayeredMesh(
            renderer=self._renderer,
            vertices=surf["rr"],
            triangles=surf["tris"],
            normals=surf["nn"],
        )
        mesh.map()
        color = _to_rgb(color, alpha=True)
        cmap = np.array([(0, 0, 0, 0), color])
        ctable = np.round(cmap * 255).astype(np.uint8)
        mesh.add_overlay(
            scalars=np.ones(len(current_data)),
            colormap=ctable,
            rng=[0, 1],
            opacity=alpha,
            name="surf",
        )

        # Show the field density
        if self._show_density:
            mesh.add_overlay(
                scalars=current_data,
                colormap=self._colormap,
                rng=[-map_vmax, map_vmax],
                opacity=1.0,
                name="field",
            )

        # And the field lines on top
        if self._n_contours > 1:
            contours = np.linspace(-map_vmax, map_vmax, self._n_contours)
            contours_actor, _ = self._renderer.contour(
                surface=surf,
                scalars=current_data,
                contours=contours,
                vmin=-map_vmax,
                vmax=map_vmax,
                colormap=self._colormap_lines,
            )
        else:
            contours = None  # noqa
            contours_actor = None

        return dict(
            pick=pick,
            data=data,
            data_interp=data_interp,
            map_kind=surf_map["kind"],
            mesh=mesh,
            contours=contours,
            contours_actor=contours_actor,
            surf=surf,
            map_vmax=map_vmax,
        )

    def _update(self):
        """Update the figure to reflect the current settings."""
        for surf_map in self._surf_maps:
            current_data = surf_map["data_interp"](self._current_time)
            surf_map["mesh"].update_overlay(name="field", scalars=current_data)

            if surf_map["contours"] is not None:
                self._renderer.plotter.remove_actor(
                    surf_map["contours_actor"], render=False
                )
                if self._n_contours > 1:
                    surf_map["contours_actor"], _ = self._renderer.contour(
                        surface=surf_map["surf"],
                        scalars=current_data,
                        contours=surf_map["contours"],
                        vmin=-surf_map["map_vmax"],
                        vmax=surf_map["map_vmax"],
                        colormap=self._colormap_lines,
                    )
        if self._time_label is not None:
            if hasattr(self, "_time_label_actor"):
                self._renderer.plotter.remove_actor(
                    self._time_label_actor, render=False
                )
            time_label = self._time_label
            if "%" in self._time_label:
                time_label = self._time_label % np.round(1e3 * self._current_time)
            self._time_label_actor = self._renderer.text2d(
                x_window=0.01, y_window=0.01, text=time_label
            )

        self._renderer.plotter.update()

    def _configure_dock(self):
        """Configure the widgets shown in the dock on the left."""
        r = self._renderer

        if not hasattr(r, "_dock"):
            r._dock_initialize()

        # Fieldline configuration
        layout = r._dock_add_group_box("Fieldlines")

        r._dock_add_label(value="max value", align=True, layout=layout)

        @_auto_weakref
        def _callback(vmax, kind, scaling):
            self.set_vmax(vmax / scaling, kind=kind)

        for surf_map in self._surf_maps:
            if surf_map["map_kind"] == "meg":
                scaling = DEFAULTS["scalings"]["grad"]
            else:
                scaling = DEFAULTS["scalings"]["eeg"]
            rng = [0, np.max(np.abs(surf_map["data"])) * scaling]
            hlayout = r._dock_add_layout(vertical=False)

            self._widgets[f"vmax_slider_{surf_map['map_kind']}"] = r._dock_add_slider(
                name=surf_map["map_kind"].upper(),
                value=surf_map["map_vmax"] * scaling,
                rng=rng,
                callback=partial(_callback, kind=surf_map["map_kind"], scaling=scaling),
                double=True,
                layout=hlayout,
            )
            self._widgets[f"vmax_spin_{surf_map['map_kind']}"] = r._dock_add_spin_box(
                name="",
                value=surf_map["map_vmax"] * scaling,
                rng=rng,
                callback=partial(_callback, kind=surf_map["map_kind"], scaling=scaling),
                layout=hlayout,
            )
            r._layout_add_widget(layout, hlayout)

        hlayout = r._dock_add_layout(vertical=False)
        r._dock_add_label(
            value="Rescale",
            align=True,
            layout=hlayout,
        )
        r._dock_add_button(
            name="↺",
            callback=self._rescale,
            layout=hlayout,
            style="toolbutton",
        )
        r._layout_add_widget(layout, hlayout)

        self._widgets["contours"] = r._dock_add_spin_box(
            name="Contour lines",
            value=21,
            rng=[0, 99],
            step=1,
            double=False,
            callback=self.set_contours,
            layout=layout,
        )
        r._dock_finalize()

    def _on_time_change(self, event):
        """Respond to time_change UI event."""
        new_time = np.clip(event.time, self._evoked.times[0], self._evoked.times[-1])
        if new_time == self._current_time:
            return
        self._current_time = new_time
        self._update()

    def _on_colormap_range(self, event):
        """Response to the colormap_range UI event."""
        if event.kind == "field_strength_meg":
            kind = "meg"
        elif event.kind == "field_strength_eeg":
            kind = "eeg"
        else:
            return

        for surf_map in self._surf_maps:
            if surf_map["map_kind"] == kind:
                break
        else:
            # No field map currently shown of the requested type.
            return

        vmin = event.fmin
        vmax = event.fmax
        surf_map["contours"] = np.linspace(vmin, vmax, self._n_contours)

        if self._show_density:
            surf_map["mesh"].update_overlay(name="field", rng=[vmin, vmax])
            # Update the GUI widgets
            if kind == "meg":
                scaling = DEFAULTS["scalings"]["grad"]
            else:
                scaling = DEFAULTS["scalings"]["eeg"]
            with disable_ui_events(self):
                widget = self._widgets.get(f"vmax_slider_{kind}", None)
                if widget is not None:
                    widget.set_value(vmax * scaling)
                widget = self._widgets.get(f"vmax_spin_{kind}", None)
                if widget is not None:
                    widget.set_value(vmax * scaling)

        self._update()

    def _on_contours(self, event):
        """Respond to the contours UI event."""
        if event.kind == "field_strength_meg":
            kind = "meg"
        elif event.kind == "field_strength_eeg":
            kind = "eeg"
        else:
            return

        for surf_map in self._surf_maps:
            if surf_map["map_kind"] == kind:
                break
        surf_map["contours"] = event.contours
        self._n_contours = len(event.contours)
        with disable_ui_events(self):
            if "contours" in self._widgets:
                self._widgets["contours"].set_value(len(event.contours))
        self._update()

    def set_time(self, time):
        """Set the time to display (in seconds).

        Parameters
        ----------
        time : float
            The time to show, in seconds.
        """
        if self._evoked.times[0] <= time <= self._evoked.times[-1]:
            publish(self, TimeChange(time=time))
        else:
            raise ValueError(
                f"Requested time ({time} s) is outside the range of "
                f"available times ({self._evoked.times[0]}-{self._evoked.times[-1]} s)."
            )

    def set_contours(self, n_contours):
        """Adjust the number of contour lines to use when drawing the fieldlines.

        Parameters
        ----------
        n_contours : int
            The number of contour lines to use.
        """
        for surf_map in self._surf_maps:
            publish(
                self,
                Contours(
                    kind=f"field_strength_{surf_map['map_kind']}",
                    contours=np.linspace(
                        -surf_map["map_vmax"], surf_map["map_vmax"], n_contours
                    ).tolist(),
                ),
            )

    def set_vmax(self, vmax, kind="meg"):
        """Change the color range of the density maps.

        Parameters
        ----------
        vmax : float
            The new maximum value of the color range.
        kind : 'meg' | 'eeg'
            Which field map to apply the new color range to.
        """
        _check_option("type", kind, ["eeg", "meg"])
        for surf_map in self._surf_maps:
            if surf_map["map_kind"] == kind:
                publish(
                    self,
                    ColormapRange(
                        kind=f"field_strength_{kind}",
                        fmin=-vmax,
                        fmax=vmax,
                    ),
                )
                break
        else:
            raise ValueError(f"No {type.upper()} field map currently shown.")

    def _rescale(self):
        """Rescale the fieldlines and density maps to the current time point."""
        for surf_map in self._surf_maps:
            current_data = surf_map["data_interp"](self._current_time)
            vmax = float(np.max(current_data))
            self.set_vmax(vmax, kind=surf_map["map_kind"])
