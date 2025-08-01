# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import copy
import os
import os.path as op
import time
import traceback
import warnings
from functools import partial
from io import BytesIO

import numpy as np
from scipy.interpolate import interp1d
from scipy.sparse import csr_array
from scipy.spatial.distance import cdist

from ..._fiff.meas_info import Info
from ..._fiff.pick import pick_types
from ..._freesurfer import (
    _estimate_talxfm_rigid,
    _get_aseg,
    _get_head_surface,
    _get_skull_surface,
    read_freesurfer_lut,
    read_talxfm,
    vertex_to_mni,
)
from ...defaults import DEFAULTS, _handle_default
from ...surface import _marching_cubes, _mesh_borders, mesh_edges
from ...transforms import (
    Transform,
    _frame_to_str,
    _get_trans,
    _get_transforms_to_coord_frame,
    apply_trans,
)
from ...utils import (
    Bunch,
    _auto_weakref,
    _check_fname,
    _check_option,
    _ensure_int,
    _path_like,
    _ReuseCycle,
    _to_rgb,
    _validate_type,
    fill_doc,
    get_subjects_dir,
    logger,
    use_log_level,
    verbose,
    warn,
)
from .._3d import (
    _check_views,
    _handle_sensor_types,
    _handle_time,
    _plot_forward,
    _plot_helmet,
    _plot_sensors_3d,
    _process_clim,
)
from .._3d_overlay import _LayeredMesh
from ..ui_events import (
    ColormapRange,
    PlaybackSpeed,
    TimeChange,
    VertexSelect,
    _get_event_channel,
    disable_ui_events,
    publish,
    subscribe,
    unsubscribe,
)
from ..utils import (
    _generate_default_filename,
    _get_color_list,
    _save_ndarray_img,
    _show_help_fig,
    concatenate_images,
    safe_event,
)
from .colormap import calculate_lut
from .surface import _Surface
from .view import _lh_views_dict, views_dicts


@fill_doc
class Brain:
    """Class for visualizing a brain.

    .. warning::
       The API for this class is not currently complete. We suggest using
       :meth:`mne.viz.plot_source_estimates` with the PyVista backend
       enabled to obtain a ``Brain`` instance.

    Parameters
    ----------
    subject : str
        Subject name in Freesurfer subjects dir.

        .. versionchanged:: 1.2
           This parameter was renamed from ``subject_id`` to ``subject``.
    hemi : str
        Hemisphere id (ie 'lh', 'rh', 'both', or 'split'). In the case
        of 'both', both hemispheres are shown in the same window.
        In the case of 'split' hemispheres are displayed side-by-side
        in different viewing panes.
    surf : str
        FreeSurfer surface mesh name (ie 'white', 'inflated', etc.).
    title : str
        Title for the window.
    cortex : str, list, dict
        Specifies how the cortical surface is rendered. Options:

        1. The name of one of the preset cortex styles:
            ``'classic'`` (default), ``'high_contrast'``,
            ``'low_contrast'``, or ``'bone'``.
        2. A single color-like argument to render the cortex as a single
            color, e.g. ``'red'`` or ``(0.1, 0.4, 1.)``.
        3. A list of two color-like used to render binarized curvature
            values for gyral (first) and sulcal (second). regions, e.g.,
            ``['red', 'blue']`` or ``[(1, 0, 0), (0, 0, 1)]``.
        4. A dict containing keys ``'vmin', 'vmax', 'colormap'`` with
            values used to render the binarized curvature (where 0 is gyral,
            1 is sulcal).

        .. versionchanged:: 0.24
           Add support for non-string arguments.
    alpha : float in [0, 1]
        Alpha level to control opacity of the cortical surface.
    size : int | array-like, shape (2,)
        The size of the window, in pixels. can be one number to specify
        a square window, or a length-2 sequence to specify (width, height).
    background : tuple(int, int, int)
        The color definition of the background: (red, green, blue).
    foreground : matplotlib color
        Color of the foreground (will be used for colorbars and text).
        None (default) will use black or white depending on the value
        of ``background``.
    figure : list of Figure | None
        If None (default), a new window will be created with the appropriate
        views.
    subjects_dir : str | None
        If not None, this directory will be used as the subjects directory
        instead of the value set using the SUBJECTS_DIR environment
        variable.
    %(views)s
    offset : bool | str
        If True, shifts the right- or left-most x coordinate of the left and
        right surfaces, respectively, to be at zero. This is useful for viewing
        inflated surface where hemispheres typically overlap. Can be "auto"
        (default) use True with inflated surfaces and False otherwise
        (Default: 'auto'). Only used when ``hemi='both'``.

        .. versionchanged:: 0.23
           Default changed to "auto".
    interaction : str
        Can be "trackball" (default) or "terrain", i.e. a turntable-style
        camera.
    units : str
        Can be 'm' or 'mm' (default).
    %(view_layout)s
    silhouette : dict | bool
       As a dict, it contains the ``color``, ``linewidth``, ``alpha`` opacity
       and ``decimate`` (level of decimation between 0 and 1 or None) of the
       brain's silhouette to display. If True, the default values are used
       and if False, no silhouette will be displayed. Defaults to False.
    %(theme_3d)s
    show : bool
        Display the window as soon as it is ready. Defaults to True.

    Attributes
    ----------
    geo : dict
        A dictionary of PyVista surface objects for each hemisphere.
    overlays : dict
        The overlays.

    Notes
    -----
    The figure will publish and subscribe to the following UI events:

    * :class:`~mne.viz.ui_events.TimeChange`
    * :class:`~mne.viz.ui_events.PlaybackSpeed`
    * :class:`~mne.viz.ui_events.ColormapRange`, ``kind="distributed_source_power"``
    * :class:`~mne.viz.ui_events.VertexSelect`

    This table shows the capabilities of each Brain backend ("✓" for full
    support, and "-" for partial support):

    .. table::
       :widths: auto

       +-------------------------------------+--------------+---------------+
       | 3D function:                        | surfer.Brain | mne.viz.Brain |
       +=====================================+==============+===============+
       | :meth:`add_annotation`              | ✓            | ✓             |
       +-------------------------------------+--------------+---------------+
       | :meth:`add_data`                    | ✓            | ✓             |
       +-------------------------------------+--------------+---------------+
       | :meth:`add_dipole`                  |              | ✓             |
       +-------------------------------------+--------------+---------------+
       | :meth:`add_foci`                    | ✓            | ✓             |
       +-------------------------------------+--------------+---------------+
       | :meth:`add_forward`                 |              | ✓             |
       +-------------------------------------+--------------+---------------+
       | :meth:`add_head`                    |              | ✓             |
       +-------------------------------------+--------------+---------------+
       | :meth:`add_label`                   | ✓            | ✓             |
       +-------------------------------------+--------------+---------------+
       | :meth:`add_sensors`                 |              | ✓             |
       +-------------------------------------+--------------+---------------+
       | :meth:`add_skull`                   |              | ✓             |
       +-------------------------------------+--------------+---------------+
       | :meth:`add_text`                    | ✓            | ✓             |
       +-------------------------------------+--------------+---------------+
       | :meth:`add_volume_labels`           |              | ✓             |
       +-------------------------------------+--------------+---------------+
       | :meth:`close`                       | ✓            | ✓             |
       +-------------------------------------+--------------+---------------+
       | data                                | ✓            | ✓             |
       +-------------------------------------+--------------+---------------+
       | foci                                | ✓            |               |
       +-------------------------------------+--------------+---------------+
       | labels                              | ✓            | ✓             |
       +-------------------------------------+--------------+---------------+
       | :meth:`remove_data`                 |              | ✓             |
       +-------------------------------------+--------------+---------------+
       | :meth:`remove_dipole`               |              | ✓             |
       +-------------------------------------+--------------+---------------+
       | :meth:`remove_forward`              |              | ✓             |
       +-------------------------------------+--------------+---------------+
       | :meth:`remove_head`                 |              | ✓             |
       +-------------------------------------+--------------+---------------+
       | :meth:`remove_labels`               | ✓            | ✓             |
       +-------------------------------------+--------------+---------------+
       | :meth:`remove_annotations`          | -            | ✓             |
       +-------------------------------------+--------------+---------------+
       | :meth:`remove_sensors`              |              | ✓             |
       +-------------------------------------+--------------+---------------+
       | :meth:`remove_skull`                |              | ✓             |
       +-------------------------------------+--------------+---------------+
       | :meth:`remove_text`                 |              | ✓             |
       +-------------------------------------+--------------+---------------+
       | :meth:`remove_volume_labels`        |              | ✓             |
       +-------------------------------------+--------------+---------------+
       | :meth:`save_image`                  | ✓            | ✓             |
       +-------------------------------------+--------------+---------------+
       | :meth:`save_movie`                  | ✓            | ✓             |
       +-------------------------------------+--------------+---------------+
       | :meth:`screenshot`                  | ✓            | ✓             |
       +-------------------------------------+--------------+---------------+
       | :meth:`show_view`                   | ✓            | ✓             |
       +-------------------------------------+--------------+---------------+
       | TimeViewer                          | ✓            | ✓             |
       +-------------------------------------+--------------+---------------+
       | :meth:`get_picked_points`           |              | ✓             |
       +-------------------------------------+--------------+---------------+
       | :meth:`add_data(volume) <add_data>` |              | ✓             |
       +-------------------------------------+--------------+---------------+
       | view_layout                         |              | ✓             |
       +-------------------------------------+--------------+---------------+
       | flatmaps                            |              | ✓             |
       +-------------------------------------+--------------+---------------+
       | vertex picking                      |              | ✓             |
       +-------------------------------------+--------------+---------------+
       | label picking                       |              | ✓             |
       +-------------------------------------+--------------+---------------+
    """

    def __init__(
        self,
        subject,
        hemi="both",
        surf="pial",
        title=None,
        cortex="classic",
        alpha=1.0,
        size=800,
        background="black",
        foreground=None,
        figure=None,
        subjects_dir=None,
        views="auto",
        *,
        offset="auto",
        interaction="trackball",
        units="mm",
        view_layout="vertical",
        silhouette=False,
        theme=None,
        show=True,
    ):
        from ..backends.renderer import _get_renderer, backend

        _validate_type(subject, str, "subject")
        self._surf = surf
        if hemi is None:
            hemi = "vol"
        hemi = self._check_hemi(hemi, extras=("both", "split", "vol"))
        if hemi in ("both", "split"):
            self._hemis = ("lh", "rh")
        else:
            assert hemi in ("lh", "rh", "vol")
            self._hemis = (hemi,)
        self._view_layout = _check_option(
            "view_layout", view_layout, ("vertical", "horizontal")
        )

        if figure is not None and not isinstance(figure, int):
            backend._check_3d_figure(figure)
        if title is None:
            self._title = subject
        else:
            self._title = title
        self._interaction = "trackball"

        self._bg_color = _to_rgb(background, name="background")
        if foreground is None:
            foreground = "w" if sum(self._bg_color) < 2 else "k"
        self._fg_color = _to_rgb(foreground, name="foreground")
        del background, foreground
        views = _check_views(surf, views, hemi)
        col_dict = dict(lh=1, rh=1, both=1, split=2, vol=1)
        shape = (len(views), col_dict[hemi])
        if self._view_layout == "horizontal":
            shape = shape[::-1]
        self._subplot_shape = shape

        size = tuple(np.atleast_1d(size).round(0).astype(int).flat)
        if len(size) not in (1, 2):
            raise ValueError(
                '"size" parameter must be an int or length-2 sequence of ints.'
            )
        size = size if len(size) == 2 else size * 2  # 1-tuple to 2-tuple
        subjects_dir = get_subjects_dir(subjects_dir)
        if subjects_dir is not None:
            subjects_dir = str(subjects_dir)

        self.time_viewer = False
        self._hash = time.time_ns()
        self._hemi = hemi
        self._units = units
        self._alpha = float(alpha)
        self._subject = subject
        self._subjects_dir = subjects_dir
        self._views = views
        self._times = None
        self._vertex_to_label_id = dict()
        self._annotation_labels = dict()
        self._labels = {"lh": list(), "rh": list()}
        self._unnamed_label_id = 0  # can only grow
        self._annots = {"lh": list(), "rh": list()}
        self._layered_meshes = dict()
        self._actors = dict()
        self._cleaned = False
        # default values for silhouette
        self._silhouette = {
            "color": self._bg_color,
            "line_width": 2,
            "alpha": alpha,
            "decimate": 0.9,
        }
        _validate_type(silhouette, (dict, bool), "silhouette")
        if isinstance(silhouette, dict):
            self._silhouette.update(silhouette)
            self.silhouette = True
        else:
            self.silhouette = silhouette
        self._scalar_bar = None
        # for now only one time label can be added
        # since it is the same for all figures
        self._time_label_added = False
        # array of data used by TimeViewer
        self._data = {}
        self.geo = {}
        self.set_time_interpolation("nearest")

        geo_kwargs = self._cortex_colormap(cortex)
        # evaluate at the midpoint of the used colormap
        val = -geo_kwargs["vmin"] / (geo_kwargs["vmax"] - geo_kwargs["vmin"])
        self._brain_color = geo_kwargs["colormap"](val)

        # load geometry for one or both hemispheres as necessary
        _validate_type(offset, (str, bool), "offset")
        if isinstance(offset, str):
            _check_option("offset", offset, ("auto",), extra="when str")
            offset = surf in ("inflated", "flat")
        offset = None if (not offset or hemi != "both") else 0.0
        logger.debug(f"Hemi offset: {offset}")
        _validate_type(theme, (str, None), "theme")
        self._renderer = _get_renderer(
            name=self._title, size=size, bgcolor=self._bg_color, shape=shape, fig=figure
        )
        self._renderer._window_close_connect(self._clean)
        self._renderer._window_set_theme(theme)
        self.plotter = self._renderer.plotter
        self.widgets = dict()

        self._setup_canonical_rotation()

        # plot hemis
        for h in ("lh", "rh"):
            if h not in self._hemis:
                continue  # don't make surface if not chosen
            # Initialize a Surface object as the geometry
            geo = _Surface(
                self._subject,
                h,
                surf,
                self._subjects_dir,
                offset,
                units=self._units,
                x_dir=self._rigid[0, :3],
            )
            # Load in the geometry and curvature
            geo.load_geometry()
            geo.load_curvature()
            self.geo[h] = geo
            for _, _, v in self._iter_views(h):
                if self._layered_meshes.get(h) is None:
                    mesh = _LayeredMesh(
                        renderer=self._renderer,
                        vertices=self.geo[h].coords,
                        triangles=self.geo[h].faces,
                        normals=self.geo[h].nn,
                    )
                    mesh.map()  # send to GPU
                    if self.geo[h].bin_curv is None:
                        scalars = mesh._default_scalars[:, 0]
                    else:
                        scalars = self.geo[h].bin_curv
                    mesh.add_overlay(
                        scalars=scalars,
                        colormap=geo_kwargs["colormap"],
                        rng=[geo_kwargs["vmin"], geo_kwargs["vmax"]],
                        opacity=alpha,
                        name="curv",
                    )
                    self._layered_meshes[h] = mesh
                else:
                    actor = self._layered_meshes[h]._actor
                    self._renderer.plotter.add_actor(actor, render=False)
                if self.silhouette:
                    mesh = self._layered_meshes[h]
                    self._renderer._silhouette(
                        mesh=mesh._polydata,
                        color=self._silhouette["color"],
                        line_width=self._silhouette["line_width"],
                        alpha=self._silhouette["alpha"],
                        decimate=self._silhouette["decimate"],
                    )
                self._set_camera(**views_dicts[h][v])

        self.interaction = interaction
        self._closed = False
        if show:
            self.show()
        # update the views once the geometry is all set
        for h in self._hemis:
            for ri, ci, v in self._iter_views(h):
                self.show_view(v, row=ri, col=ci, hemi=h, update=False)

        if surf == "flat":
            self._renderer.set_interaction("rubber_band_2d")

        self._renderer._update()

    def _setup_canonical_rotation(self):
        self._rigid = np.eye(4)
        try:
            xfm = _estimate_talxfm_rigid(self._subject, self._subjects_dir)
        except Exception:
            logger.info(
                "Could not estimate rigid Talairach alignment, using identity matrix"
            )
        else:
            self._rigid[:] = xfm

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
        """
        if self.time_viewer:
            return
        if not self._data:
            raise ValueError("No data to visualize. See ``add_data``.")
        self.time_viewer = time_viewer
        self.orientation = list(_lh_views_dict.keys())
        self.default_smoothing_range = [-1, 15]

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
        self._mouse_no_mvt = -1

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

        self._configure_time_label()
        self._configure_scalar_bar()
        self._configure_shortcuts()
        self._configure_picking()
        self._configure_dock()
        self._configure_tool_bar()
        self._configure_menu()
        self._configure_status_bar()
        self._configure_help()
        # show everything at the end
        self.toggle_interface()
        self._renderer.show()

        # sizes could change, update views
        for hemi in ("lh", "rh"):
            for ri, ci, v in self._iter_views(hemi):
                self.show_view(view=v, row=ri, col=ci)
        self._renderer._process_events()

        self._renderer._update()
        # finally, show the MplCanvas
        if self.show_traces:
            self.mpl_canvas.show()

    @safe_event
    def _clean(self):
        # resolve the reference cycle
        self._renderer._window_close_disconnect()
        self.clear_glyphs()
        self.remove_annotations()
        # clear init actors
        for hemi in self._layered_meshes:
            self._layered_meshes[hemi]._clean()
        self._clear_callbacks()
        self._clear_widgets()
        if getattr(self, "mpl_canvas", None) is not None:
            self.mpl_canvas.clear()
        if getattr(self, "act_data_smooth", None) is not None:
            for key in list(self.act_data_smooth.keys()):
                self.act_data_smooth[key] = None
        # XXX this should be done in PyVista
        for renderer in self._renderer._all_renderers:
            renderer.RemoveAllLights()
        # app_window cannot be set to None because it is used in __del__
        for key in ("lighting", "interactor", "_RenderWindow"):
            setattr(self.plotter, key, None)
        # Qt LeaveEvent requires _Iren so we use _FakeIren instead of None
        # to resolve the ref to vtkGenericRenderWindowInteractor
        self.plotter._Iren = _FakeIren()
        if getattr(self.plotter, "picker", None) is not None:
            self.plotter.picker = None
        if getattr(self._renderer, "_picker", None) is not None:
            self._renderer._picker = None
        # XXX end PyVista
        for key in (
            "plotter",
            "window",
            "dock",
            "tool_bar",
            "menu_bar",
            "interactor",
            "mpl_canvas",
            "time_actor",
            "picked_renderer",
            "act_data_smooth",
            "_scalar_bar",
            "actions",
            "widgets",
            "geo",
            "_data",
        ):
            setattr(self, key, None)
        self._cleaned = True

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

    def apply_auto_scaling(self):
        """Detect automatically fitting scaling parameters."""
        self._update_auto_scaling()

    def restore_user_scaling(self):
        """Restore original scaling parameters."""
        self._update_auto_scaling(restore=True)

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
        layout = self._renderer._dock_add_group_box(name)
        # Renderer widget
        rends = [str(i) for i in range(len(self._renderer._all_renderers))]
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

    def _configure_dock_colormap_widget(self, name):
        fmax, fscale, fscale_power = _get_range(self)
        rng = [0, fmax * fscale]
        self._data["fscale"] = fscale

        layout = self._renderer._dock_add_group_box(name)
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
            callback=self.restore_user_scaling,
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

        layout = self._renderer._dock_add_group_box(name)

        # setup candidate annots
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
        modes = _get_allowed_label_modes(stc)
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
        self._configure_dock_orientation_widget(name="Orientation")
        self._configure_dock_colormap_widget(name="Color Limits")
        self._configure_dock_trace_widget(name="Trace")

        # Smoothing widget
        self.widgets["smoothing"] = self._renderer._dock_add_spin_box(
            name="Smoothing",
            value=self._data["smoothing_steps"],
            rng=self.default_smoothing_range,
            callback=self.set_data_smoothing,
            double=False,
        )

        self._renderer._dock_finalize()

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

        # plot RMS of the activation
        y = np.concatenate(
            list(v[0] for v in self.act_data_smooth.values() if v[0] is not None)
        )
        rms = np.linalg.norm(y, axis=0) / np.sqrt(len(y))
        del y

        (self.rms,) = self.mpl_canvas.axes.plot(
            self._data["time"],
            rms,
            lw=3,
            label="RMS",
            zorder=3,
            color=self._fg_color,
            alpha=0.5,
            ls=":",
        )

        # now plot the time line
        self.plot_time_line(update=False)

        # then the picked points
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
            publish(self, VertexSelect(hemi=hemi, vertex_id=vertex_id))

    def _configure_picking(self):
        # get data for each hemi
        for idx, hemi in enumerate(["vol", "lh", "rh"]):
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

        self._renderer._update_picking_callback(
            self._on_mouse_move,
            self._on_button_press,
            self._on_button_release,
            self._on_pick,
        )
        subscribe(self, "vertex_select", self._on_vertex_select)

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
        if getattr(self, "iren", None) is not None:
            self.plotter.iren.clear_key_event_callbacks()
        # Then, we add our own:
        self.plotter.add_key_event("i", self.toggle_interface)
        self.plotter.add_key_event("s", self.apply_auto_scaling)
        self.plotter.add_key_event("r", self.restore_user_scaling)
        self.plotter.add_key_event("c", self.clear_glyphs)
        for key, which, amt in (
            ("Left", "azimuth", 10),
            ("Right", "azimuth", -10),
            ("Up", "elevation", 10),
            ("Down", "elevation", -10),
        ):
            self.plotter.clear_events_for_key(key)
            self.plotter.add_key_event(key, partial(self._rotate_camera, which, amt))

    def _configure_menu(self):
        self._renderer._menu_initialize()
        self._renderer._menu_add_submenu(
            name="help",
            desc="Help",
        )
        self._renderer._menu_add_button(
            menu_name="help",
            name="help",
            desc="Show MNE key bindings\t?",
            func=self.help,
        )

    def _configure_status_bar(self):
        self._renderer._status_bar_initialize()
        self.status_msg = self._renderer._status_bar_add_label(
            self.default_status_bar_msg, stretch=1
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
            self._renderer._picker.Pick(x, y, 0, self.picked_renderer)
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
        for hemi, this_mesh in self._layered_meshes.items():
            assert hemi in ("lh", "rh"), f"Unexpected {hemi=}"
            if this_mesh._polydata is mesh:
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
            grid = mesh = self._data[hemi]["grid"]
            vertices = self._data[hemi]["vertices"]
            coords = self._data[hemi]["grid_coords"][vertices]
            scalars = grid.cell_data["values"][vertices]
            spacing = np.array(grid.GetSpacing())
            max_dist = np.linalg.norm(spacing) / 2.0
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
            # useful for debugging the ray by mapping it into the volume:
            # dists = dists - dists.min()
            # dists = (1. - dists / dists.max()) * self._cmap_range[1]
            # grid.cell_data['values'][vertices] = dists * mask
            idx = idx[np.argmax(np.abs(scalars[idx]))]
            vertex_id = vertices[idx]
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
            vertices = mesh.points[cell]
            idx = np.argmin(abs(vertices - pos), axis=0)
            vertex_id = cell[idx[0]]

        publish(self, VertexSelect(hemi=hemi, vertex_id=vertex_id))

    def _on_time_change(self, event):
        """Respond to a time change UI event."""
        if event.time == self._current_time:
            return
        time_idx = self._to_time_index(event.time)
        self._update_current_time_idx(time_idx)
        if self.time_viewer:
            with disable_ui_events(self):
                if "time" in self.widgets:
                    self.widgets["time"].set_value(time_idx)
                if "current_time" in self.widgets:
                    self.widgets["current_time"].set_value(f"{self._current_time: .3f}")
            self.plot_time_line(update=True)

    def _on_colormap_range(self, event):
        """Respond to the colormap_range UI event."""
        if event.kind != "distributed_source_power":
            return
        lims = {key: getattr(event, key) for key in ("fmin", "fmid", "fmax", "alpha")}
        # Check if limits have changed at all.
        if all(val is None or val == self._data[key] for key, val in lims.items()):
            return
        # Update the GUI elements.
        with disable_ui_events(self):
            for key, val in lims.items():
                if val is not None:
                    if key in self.widgets:
                        self.widgets[key].set_value(val * self._data["fscale"])
                    entry_key = "entry_" + key
                    if entry_key in self.widgets:
                        self.widgets[entry_key].set_value(val * self._data["fscale"])
        # Update the render.
        self._update_colormap_range(**lims)

    def _on_vertex_select(self, event):
        """Respond to vertex_select UI event."""
        if event.hemi == "vol":
            try:
                mesh = self._data[event.hemi]["grid"]
            except KeyError:
                return
        else:
            try:
                mesh = self._layered_meshes[event.hemi]._polydata
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
        label._line.remove()
        self.color_cycle.restore(label._color)
        self.mpl_canvas.update_plot()
        self._layered_meshes[hemi].remove_overlay(label.name)
        self._picked_patches[hemi].remove(label_id)

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
            ijk = np.unravel_index(
                vertex_id, np.array(mesh.GetDimensions()) - 1, order="F"
            )
            voxel = mesh.GetCell(*ijk)
            center = np.empty(3)
            voxel.GetCentroid(center)
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
                radius=4.0,
            )
            spheres.append(dict(mesh=mesh, actor=actor))

        # add metadata for picking
        for sphere in spheres:
            sphere.update(hemi=hemi, line=line, color=color, vertex_id=vertex_id)

        _ensure_int(vertex_id)
        self._picked_points[(hemi, vertex_id)] = spheres
        return sphere

    def _remove_vertex_glyph(self, *, hemi, vertex_id, render=True):
        _ensure_int(vertex_id)
        assert isinstance(hemi, str), f"got {type(hemi)} for {hemi=}"
        spheres = self._picked_points.pop((hemi, vertex_id))
        color, line = spheres[0]["color"], spheres[0]["line"]
        line.remove()
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
            self.rms.remove()
            self.rms = None
        self._renderer._update()

    @fill_doc
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
        %(brain_update)s

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
            mni = " MNI: " + ", ".join(f"{m:5.1f}" for m in mni)
        else:
            mni = ""
        label = f"{hemi_str}:{str(vertex_id).ljust(6)}{mni}"
        act_data, smooth = self.act_data_smooth[hemi]
        if smooth is not None:
            act_data = (smooth[[vertex_id]] @ act_data)[0]
        else:
            act_data = act_data[vertex_id].copy()
        line = self.mpl_canvas.plot(
            time,
            act_data,
            label=label,
            lw=1.0,
            color=color,
            zorder=4,
            update=update,
        )
        return line

    @fill_doc
    def plot_time_line(self, update=True):
        """Add the time line to the MPL widget.

        Parameters
        ----------
        %(brain_update)s
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
                    lw=1,
                    update=update,
                )
            self.time_line.set_xdata([current_time])
            if update:
                self.mpl_canvas.update_plot()

    def _configure_help(self):
        pairs = [
            ("?", "Display help window"),
            ("i", "Toggle interface"),
            ("s", "Apply auto-scaling"),
            ("r", "Restore original clim"),
            ("c", "Clear all traces"),
            ("n", "Shift the time forward by the playback speed"),
            ("b", "Shift the time backward by the playback speed"),
            ("Space", "Start/Pause playback"),
            ("Up", "Decrease camera elevation angle"),
            ("Down", "Increase camera elevation angle"),
            ("Left", "Decrease camera azimuth angle"),
            ("Right", "Increase camera azimuth angle"),
        ]
        text1, text2 = zip(*pairs)
        text1 = "\n".join(text1)
        text2 = "\n".join(text2)
        self.help_canvas = self._renderer._window_get_simple_canvas(
            width=5, height=2, dpi=80
        )
        _show_help_fig(
            col1=text1,
            col2=text2,
            fig_help=self.help_canvas.fig,
            ax=self.help_canvas.axes,
            show=False,
        )

    def help(self):
        """Display the help window."""
        self.help_canvas.show()

    def _clear_callbacks(self):
        # Remove the default key binding
        if getattr(self, "iren", None) is not None:
            self.plotter.iren.clear_key_event_callbacks()

    def _clear_widgets(self):
        if not hasattr(self, "widgets"):
            return
        for widget in self.widgets.values():
            if widget is not None:
                for key in ("triggered", "floatValueChanged"):
                    setattr(widget, key, None)
        self.widgets.clear()

    @property
    def interaction(self):
        """The interaction style."""
        return self._interaction

    @interaction.setter
    def interaction(self, interaction):
        """Set the interaction style."""
        _validate_type(interaction, str, "interaction")
        _check_option("interaction", interaction, ("trackball", "terrain"))
        for _ in self._iter_views("vol"):  # will traverse all
            self._renderer.set_interaction(interaction)

    def _cortex_colormap(self, cortex):
        """Return the colormap corresponding to the cortex."""
        from matplotlib.colors import ListedColormap

        from .._3d import _get_cmap

        colormap_map = dict(
            classic=dict(colormap="Greys", vmin=-1, vmax=2),
            high_contrast=dict(colormap="Greys", vmin=-0.1, vmax=1.3),
            low_contrast=dict(colormap="Greys", vmin=-5, vmax=5),
            bone=dict(colormap="bone_r", vmin=-0.2, vmax=2),
        )
        _validate_type(cortex, (str, dict, list, tuple), "cortex")
        if isinstance(cortex, str):
            if cortex in colormap_map:
                cortex = colormap_map[cortex]
            else:
                cortex = [cortex] * 2
        if isinstance(cortex, list | tuple):
            _check_option(
                "len(cortex)",
                len(cortex),
                (2, 3),
                extra="when cortex is a list or tuple",
            )
            if len(cortex) == 3:
                cortex = [cortex] * 2
            cortex = list(cortex)
            for ci, c in enumerate(cortex):
                cortex[ci] = _to_rgb(c, name="cortex")
            cortex = dict(
                colormap=ListedColormap(cortex, name="custom binary"), vmin=0, vmax=1
            )
        cortex = dict(
            vmin=float(cortex["vmin"]),
            vmax=float(cortex["vmax"]),
            colormap=_get_cmap(cortex["colormap"]),
        )
        return cortex

    def _remove(self, item, render=False):
        """Remove actors from the rendered scene."""
        if item in self._actors:
            logger.debug(f"Removing {len(self._actors[item])} {item} actor(s)")
            for actor in self._actors[item]:
                self._renderer.plotter.remove_actor(actor, render=False)
            self._actors.pop(item)  # remove actor list
            if render:
                self._renderer._update()

    def _add_actor(self, item, actor):
        """Add an actor to the internal register."""
        if item in self._actors:  # allows adding more than one
            self._actors[item].append(actor)
        else:
            self._actors[item] = [actor]

    @verbose
    def add_data(
        self,
        array,
        fmin=None,
        fmid=None,
        fmax=None,
        thresh=None,
        center=None,
        transparent=False,
        colormap="auto",
        alpha=1,
        vertices=None,
        smoothing_steps=None,
        time=None,
        time_label="auto",
        colorbar=True,
        hemi=None,
        remove_existing=None,
        time_label_size=None,
        initial_time=None,
        scale_factor=None,
        vector_alpha=None,
        clim=None,
        src=None,
        volume_options=0.4,
        colorbar_kwargs=None,
        verbose=None,
    ):
        """Display data from a numpy array on the surface or volume.

        This provides a similar interface to PySurfer, but it displays
        it with a single colormap. It offers more flexibility over the
        colormap, and provides a way to display four-dimensional data
        (i.e., a timecourse) or five-dimensional data (i.e., a
        vector-valued timecourse).

        .. note:: ``fmin`` sets the low end of the colormap, and is separate
                  from thresh (this is a different convention from PySurfer).

        Parameters
        ----------
        array : numpy array, shape (n_vertices[, 3][, n_times])
            Data array. For the data to be understood as vector-valued
            (3 values per vertex corresponding to X/Y/Z surface RAS),
            then ``array`` must be have all 3 dimensions.
            If vectors with no time dimension are desired, consider using a
            singleton (e.g., ``np.newaxis``) to create a "time" dimension
            and pass ``time_label=None`` (vector values are not supported).
        %(fmin_fmid_fmax)s
        %(thresh)s
        %(center)s
        %(transparent)s
        colormap : str, list of color, or array
            Name of matplotlib colormap to use, a list of matplotlib colors,
            or a custom look up table (an n x 4 array coded with RBGA values
            between 0 and 255), the default "auto" chooses a default divergent
            colormap, if "center" is given (currently "icefire"), otherwise a
            default sequential colormap (currently "rocket").
        alpha : float in [0, 1]
            Alpha level to control opacity of the overlay.
        vertices : numpy array
            Vertices for which the data is defined (needed if
            ``len(data) < nvtx``).
        smoothing_steps : int or None
            Number of smoothing steps (smoothing is used if len(data) < nvtx)
            The value 'nearest' can be used too. None (default) will use as
            many as necessary to fill the surface.
        time : numpy array
            Time points in the data array (if data is 2D or 3D).
        %(time_label)s
        colorbar : bool
            Whether to add a colorbar to the figure. Can also be a tuple
            to give the (row, col) index of where to put the colorbar.
        hemi : str | None
            If None, it is assumed to belong to the hemisphere being
            shown. If two hemispheres are being shown, an error will
            be thrown.
        remove_existing : bool
            Not supported yet.
            Remove surface added by previous "add_data" call. Useful for
            conserving memory when displaying different data in a loop.
        time_label_size : int
            Font size of the time label (default 14).
        initial_time : float | None
            Time initially shown in the plot. ``None`` to use the first time
            sample (default).
        scale_factor : float | None (default)
            The scale factor to use when displaying glyphs for vector-valued
            data.
        vector_alpha : float | None
            Alpha level to control opacity of the arrows. Only used for
            vector-valued data. If None (default), ``alpha`` is used.
        clim : dict
            Original clim arguments.
        %(src_volume_options)s
        colorbar_kwargs : dict | None
            Options to pass to ``pyvista.Plotter.add_scalar_bar``
            (e.g., ``dict(title_font_size=10)``).
        %(verbose)s

        Notes
        -----
        If the data is defined for a subset of vertices (specified
        by the "vertices" parameter), a smoothing method is used to interpolate
        the data onto the high resolution surface. If the data is defined for
        subsampled version of the surface, smoothing_steps can be set to None,
        in which case only as many smoothing steps are applied until the whole
        surface is filled with non-zeros.

        Due to a VTK alpha rendering bug, ``vector_alpha`` is
        clamped to be strictly < 1.
        """
        _validate_type(transparent, bool, "transparent")
        _validate_type(vector_alpha, ("numeric", None), "vector_alpha")
        _validate_type(scale_factor, ("numeric", None), "scale_factor")

        # those parameters are not supported yet, only None is allowed
        _check_option("thresh", thresh, [None])
        _check_option("remove_existing", remove_existing, [None])
        _validate_type(time_label_size, (None, "numeric"), "time_label_size")
        if time_label_size is not None:
            time_label_size = float(time_label_size)
            if time_label_size < 0:
                raise ValueError(
                    f"time_label_size must be positive, got {time_label_size}"
                )

        hemi = self._check_hemi(hemi, extras=["vol"])
        stc, array, vertices = self._check_stc(hemi, array, vertices)
        array = np.asarray(array)
        vector_alpha = alpha if vector_alpha is None else vector_alpha
        self._data["vector_alpha"] = vector_alpha
        self._data["scale_factor"] = scale_factor

        # Create time array and add label if > 1D
        if array.ndim <= 1:
            time_idx = 0
        else:
            # check time array
            if time is None:
                time = np.arange(array.shape[-1])
            else:
                time = np.asarray(time)
                if time.shape != (array.shape[-1],):
                    raise ValueError(
                        f"time has shape {time.shape}, but need shape "
                        f"{(array.shape[-1],)} (array.shape[-1])"
                    )
            self._data["time"] = time

            if self._n_times is None:
                self._times = time
            elif len(time) != self._n_times:
                raise ValueError("New n_times is different from previous n_times")
            elif not np.array_equal(time, self._times):
                raise ValueError(
                    "Not all time values are consistent with previously set times."
                )

            # initial time
            if initial_time is None:
                time_idx = 0
            else:
                time_idx = self._to_time_index(initial_time)

        # time label
        time_label, _ = _handle_time(time_label, "s", time)
        y_txt = 0.05 + 0.1 * bool(colorbar)

        if array.ndim == 3:
            if array.shape[1] != 3:
                raise ValueError(
                    "If array has 3 dimensions, array.shape[1] must equal 3, got "
                    f"{array.shape[1]}"
                )
        fmin, fmid, fmax = _update_limits(fmin, fmid, fmax, center, array)
        if colormap == "auto":
            colormap = "mne" if center is not None else "hot"

        if smoothing_steps is None:
            smoothing_steps = 7
        elif smoothing_steps == "nearest":
            smoothing_steps = -1
        elif isinstance(smoothing_steps, int):
            if smoothing_steps < 0:
                raise ValueError(
                    "Expected value of `smoothing_steps` is positive but "
                    f"{smoothing_steps} was given."
                )
        else:
            raise TypeError(
                "Expected type of `smoothing_steps` is int or NoneType but "
                f"{type(smoothing_steps)} was given."
            )

        self._data["stc"] = stc
        self._data["src"] = src
        self._data["smoothing_steps"] = smoothing_steps
        self._data["clim"] = clim
        self._data["time"] = time
        self._data["initial_time"] = initial_time
        self._data["time_label"] = time_label
        self._data["initial_time_idx"] = time_idx
        self._data["time_idx"] = time_idx
        self._data["transparent"] = transparent
        # data specific for a hemi
        self._data[hemi] = dict()
        self._data[hemi]["glyph_dataset"] = None
        self._data[hemi]["glyph_mapper"] = None
        self._data[hemi]["glyph_actor"] = None
        self._data[hemi]["array"] = array
        self._data[hemi]["vertices"] = vertices
        self._data["alpha"] = alpha
        self._data["colormap"] = colormap
        self._data["center"] = center
        self._data["fmin"] = fmin
        self._data["fmid"] = fmid
        self._data["fmax"] = fmax
        self._update_colormap_range()

        # 1) add the surfaces first
        actor = None
        for _ in self._iter_views(hemi):
            if hemi in ("lh", "rh"):
                actor = self._layered_meshes[hemi]._actor
            else:
                src_vol = src[2:] if src.kind == "mixed" else src
                actor, _ = self._add_volume_data(hemi, src_vol, volume_options)
        assert actor is not None  # should have added one
        self._add_actor("data", actor)

        # 2) update time and smoothing properties
        # set_data_smoothing calls "_update_current_time_idx" for us, which will set
        # _current_time
        self.set_time_interpolation(self.time_interpolation)
        self.set_data_smoothing(self._data["smoothing_steps"])

        # 3) add the other actors
        if colorbar is True:
            # bottom left by default
            colorbar = (self._subplot_shape[0] - 1, 0)
        for ri, ci, v in self._iter_views(hemi):
            # Add the time label to the bottommost view
            do = (ri, ci) == colorbar
            if not self._time_label_added and time_label is not None and do:
                time_actor = self._renderer.text2d(
                    x_window=0.95,
                    y_window=y_txt,
                    color=self._fg_color,
                    size=time_label_size,
                    text=time_label(self._current_time),
                    justification="right",
                )
                self._data["time_actor"] = time_actor
                self._time_label_added = True
            if colorbar and self._scalar_bar is None and do:
                kwargs = dict(
                    source=actor,
                    n_labels=8,
                    color=self._fg_color,
                    bgcolor=self._brain_color[:3],
                )
                kwargs.update(colorbar_kwargs or {})
                self._scalar_bar = self._renderer.scalarbar(**kwargs)
            self._set_camera(**views_dicts[hemi][v])

        # 4) update the scalar bar and opacity (and render)
        self._update_colormap_range(alpha=alpha)

        # 5) enable UI events to interact with the data
        subscribe(self, "colormap_range", self._on_colormap_range)
        if time is not None and len(time) > 1:
            subscribe(self, "time_change", self._on_time_change)

    def remove_data(self):
        """Remove rendered data from the mesh."""
        self._remove("data", render=True)

        # Stop listening to events
        if "time_change" in _get_event_channel(self):
            unsubscribe(self, "time_change")

    def _iter_views(self, hemi):
        """Iterate over rows and columns that need to be added to."""
        hemi_dict = dict(lh=[0], rh=[0], vol=[0])
        if self._hemi == "split":
            hemi_dict.update(rh=[1], vol=[0, 1])
        for vi, view in enumerate(self._views):
            view_dict = dict(lh=[vi], rh=[vi], vol=[vi])
            if self._hemi == "split":
                view_dict.update(vol=[vi, vi])
            if self._view_layout == "vertical":
                rows, cols = view_dict, hemi_dict  # views are rows, hemis cols
            else:
                rows, cols = hemi_dict, view_dict  # hemis are rows, views cols
            for ri, ci in zip(rows[hemi], cols[hemi]):
                self._renderer.subplot(ri, ci)
                yield ri, ci, view

    def remove_labels(self):
        """Remove all the ROI labels from the image."""
        for hemi in self._hemis:
            mesh = self._layered_meshes[hemi]
            for label in self._labels[hemi]:
                mesh.remove_overlay(label.name)
            self._labels[hemi].clear()
        self._renderer._update()

    def remove_annotations(self):
        """Remove all annotations from the image."""
        for hemi in self._hemis:
            if hemi in self._layered_meshes:
                mesh = self._layered_meshes[hemi]
                mesh.remove_overlay(self._annots[hemi])
            if hemi in self._annots:
                self._annots[hemi].clear()
        self._renderer._update()

    def _add_volume_data(self, hemi, src, volume_options):
        from ...source_space import SourceSpaces
        from ..backends._pyvista import _hide_testing_actor

        _validate_type(src, SourceSpaces, "src")
        _check_option("src.kind", src.kind, ("volume",))
        _validate_type(volume_options, (dict, "numeric", None), "volume_options")
        assert hemi == "vol"
        if not isinstance(volume_options, dict):
            volume_options = dict(
                resolution=float(volume_options) if volume_options is not None else None
            )
        volume_options = _handle_default("volume_options", volume_options)
        allowed_types = (
            ["resolution", (None, "numeric")],
            ["blending", (str,)],
            ["alpha", ("numeric", None)],
            ["surface_alpha", (None, "numeric")],
            ["silhouette_alpha", (None, "numeric")],
            ["silhouette_linewidth", ("numeric",)],
        )
        for key, types in allowed_types:
            _validate_type(volume_options[key], types, f"volume_options[{repr(key)}]")
        extra_keys = set(volume_options) - set(a[0] for a in allowed_types)
        if len(extra_keys):
            raise ValueError(f"volume_options got unknown keys {sorted(extra_keys)}")
        blending = _check_option(
            'volume_options["blending"]',
            volume_options["blending"],
            ("composite", "mip"),
        )
        alpha = volume_options["alpha"]
        if alpha is None:
            alpha = 0.4 if self._data[hemi]["array"].ndim == 3 else 1.0
        alpha = np.clip(float(alpha), 0.0, 1.0)
        resolution = volume_options["resolution"]
        surface_alpha = volume_options["surface_alpha"]
        if surface_alpha is None:
            surface_alpha = min(alpha / 2.0, 0.1)
        silhouette_alpha = volume_options["silhouette_alpha"]
        if silhouette_alpha is None:
            silhouette_alpha = surface_alpha / 4.0
        silhouette_linewidth = volume_options["silhouette_linewidth"]
        del volume_options
        volume_pos = self._data[hemi].get("grid_volume_pos")
        volume_neg = self._data[hemi].get("grid_volume_neg")
        center = self._data["center"]
        if volume_pos is None:
            xyz = np.meshgrid(*[np.arange(s) for s in src[0]["shape"]], indexing="ij")
            dimensions = np.array(src[0]["shape"], int)
            mult = 1000 if self._units == "mm" else 1
            src_mri_t = src[0]["src_mri_t"]["trans"].copy()
            src_mri_t[:3] *= mult
            if resolution is not None:
                resolution = resolution * mult / 1000.0  # to mm
            del src, mult
            coords = np.array([c.ravel(order="F") for c in xyz]).T
            coords = apply_trans(src_mri_t, coords)
            self.geo[hemi] = Bunch(coords=coords)
            vertices = self._data[hemi]["vertices"]
            assert self._data[hemi]["array"].shape[0] == len(vertices)
            # MNE constructs the source space on a uniform grid in MRI space,
            # but mne coreg can change it to be non-uniform, so we need to
            # use all three elements here
            assert np.allclose(src_mri_t[:3, :3], np.diag(np.diag(src_mri_t)[:3]))
            spacing = np.diag(src_mri_t)[:3]
            origin = src_mri_t[:3, 3] - spacing / 2.0
            scalars = np.zeros(np.prod(dimensions))
            scalars[vertices] = 1.0  # for the outer mesh
            grid, grid_mesh, volume_pos, volume_neg = self._renderer._volume(
                dimensions,
                origin,
                spacing,
                scalars,
                surface_alpha,
                resolution,
                blending,
                center,
            )
            self._data[hemi]["alpha"] = alpha  # incorrectly set earlier
            self._data[hemi]["grid"] = grid
            self._data[hemi]["grid_mesh"] = grid_mesh
            self._data[hemi]["grid_coords"] = coords
            self._data[hemi]["grid_src_mri_t"] = src_mri_t
            self._data[hemi]["grid_shape"] = dimensions
            self._data[hemi]["grid_volume_pos"] = volume_pos
            self._data[hemi]["grid_volume_neg"] = volume_neg
        actor_pos, _ = self._renderer.plotter.add_actor(
            volume_pos, name=None, culling=False, reset_camera=False, render=False
        )
        actor_neg = actor_mesh = None
        if volume_neg is not None:
            actor_neg, _ = self._renderer.plotter.add_actor(
                volume_neg, name=None, culling=False, reset_camera=False, render=False
            )
        grid_mesh = self._data[hemi]["grid_mesh"]
        if grid_mesh is not None:
            actor_mesh, prop = self._renderer.plotter.add_actor(
                grid_mesh,
                name=None,
                culling=False,
                pickable=False,
                reset_camera=False,
                render=False,
            )
            prop.SetColor(*self._brain_color[:3])
            prop.SetOpacity(surface_alpha)
            if silhouette_alpha > 0 and silhouette_linewidth > 0:
                for _ in self._iter_views("vol"):
                    self._renderer._silhouette(
                        mesh=grid_mesh.GetInput(),
                        color=self._brain_color[:3],
                        line_width=silhouette_linewidth,
                        alpha=silhouette_alpha,
                    )
        for actor in (actor_pos, actor_neg, actor_mesh):
            if actor is not None:
                _hide_testing_actor(actor)

        return actor_pos, actor_neg

    def add_label(
        self,
        label,
        color=None,
        alpha=1,
        scalar_thresh=None,
        borders=False,
        hemi=None,
        subdir=None,
    ):
        """Add an ROI label to the image.

        Parameters
        ----------
        label : str | instance of Label
            Label filepath or name. Can also be an instance of
            an object with attributes "hemi", "vertices", "name", and
            optionally "color" and "values" (if scalar_thresh is not None).
        color : matplotlib-style color | None
            Anything matplotlib accepts: string, RGB, hex, etc. (default
            "crimson").
        alpha : float in [0, 1]
            Alpha level to control opacity.
        scalar_thresh : None | float
            Threshold the label ids using this value in the label
            file's scalar field (i.e. label only vertices with
            scalar >= thresh).
        borders : bool | int
            Show only label borders. If int, specify the number of steps
            (away from the true border) along the cortical mesh to include
            as part of the border definition.
        hemi : str | None
            If None, it is assumed to belong to the hemisphere being
            shown.
        subdir : None | str
            If a label is specified as name, subdir can be used to indicate
            that the label file is in a sub-directory of the subject's
            label directory rather than in the label directory itself (e.g.
            for ``$SUBJECTS_DIR/$SUBJECT/label/aparc/lh.cuneus.label``
            ``brain.add_label('cuneus', subdir='aparc')``).

        Notes
        -----
        To remove previously added labels, run Brain.remove_labels().
        """
        from ...label import read_label

        if isinstance(label, str):
            if color is None:
                color = "crimson"

            if os.path.isfile(label):
                filepath = label
                label = read_label(filepath)
                hemi = label.hemi
                label_name = os.path.basename(filepath).split(".")[1]
            else:
                hemi = self._check_hemi(hemi)
                label_name = label
                label_fname = ".".join([hemi, label_name, "label"])
                if subdir is None:
                    filepath = op.join(
                        self._subjects_dir, self._subject, "label", label_fname
                    )
                else:
                    filepath = op.join(
                        self._subjects_dir, self._subject, "label", subdir, label_fname
                    )
                if not os.path.exists(filepath):
                    raise ValueError(f"Label file {filepath} does not exist")
                label = read_label(filepath)
            ids = label.vertices
            scalars = label.values
        else:
            # try to extract parameters from label instance
            try:
                hemi = label.hemi
                ids = label.vertices
                if label.name is None:
                    label.name = "unnamed" + str(self._unnamed_label_id)
                    self._unnamed_label_id += 1
                label_name = str(label.name)

                if color is None:
                    if hasattr(label, "color") and label.color is not None:
                        color = label.color
                    else:
                        color = "crimson"

                if scalar_thresh is not None:
                    scalars = label.values
            except Exception:
                raise ValueError(
                    "Label was not a filename (str), and could "
                    "not be understood as a class. The class "
                    'must have attributes "hemi", "vertices", '
                    '"name", and (if scalar_thresh is not None)'
                    '"values"'
                )
            hemi = self._check_hemi(hemi)

        if scalar_thresh is not None:
            ids = ids[scalars >= scalar_thresh]

        if self.time_viewer and self.show_traces and self.traces_mode == "label":
            stc = self._data["stc"]
            src = self._data["src"]
            tc = stc.extract_label_time_course(
                label, src=src, mode=self.label_extract_mode
            )
            tc = tc[0] if tc.ndim == 2 else tc[0, 0, :]
            color = next(self.color_cycle)
            line = self.mpl_canvas.plot(
                self._data["time"], tc, label=label_name, color=color
            )
        else:
            line = None

        orig_color = color
        color = _to_rgb(color, alpha, alpha=True)
        cmap = np.array(
            [
                (
                    0,
                    0,
                    0,
                    0,
                ),
                color,
            ]
        )
        ctable = np.round(cmap * 255).astype(np.uint8)

        scalars = np.zeros(self.geo[hemi].coords.shape[0])
        scalars[ids] = 1
        if borders:
            keep_idx = _mesh_borders(self.geo[hemi].faces, scalars)
            show = np.zeros(scalars.size, dtype=np.int64)
            if isinstance(borders, int):
                for _ in range(borders):
                    keep_idx = np.isin(self.geo[hemi].faces.ravel(), keep_idx)
                    keep_idx.shape = self.geo[hemi].faces.shape
                    keep_idx = self.geo[hemi].faces[np.any(keep_idx, axis=1)]
                    keep_idx = np.unique(keep_idx)
            show[keep_idx] = 1
            scalars *= show
        for _, _, v in self._iter_views(hemi):
            mesh = self._layered_meshes[hemi]
            mesh.add_overlay(
                scalars=scalars,
                colormap=ctable,
                rng=[np.min(scalars), np.max(scalars)],
                opacity=alpha,
                name=label_name,
            )
            if self.time_viewer and self.show_traces and self.traces_mode == "label":
                label._color = orig_color
                label._line = line
            self._labels[hemi].append(label)
        self._renderer._update()

    @fill_doc
    def add_forward(self, fwd, trans, alpha=1, scale=None):
        """Add a quiver to render positions of dipoles.

        Parameters
        ----------
        %(fwd)s
        %(trans_not_none)s
        %(alpha)s Default 1.
        scale : None | float
            The size of the arrow representing the dipoles in
            :class:`mne.viz.Brain` units. Default 1.5mm.

        Notes
        -----
        .. versionadded:: 1.0
        """
        head_mri_t = _get_trans(trans, "head", "mri", allow_none=False)[0]
        del trans
        if scale is None:
            scale = 1.5 if self._units == "mm" else 1.5e-3
        error_msg = (
            'Unexpected forward model coordinate frame {}, must be "head" or "mri"'
        )
        if fwd["coord_frame"] in _frame_to_str:
            fwd_frame = _frame_to_str[fwd["coord_frame"]]
            if fwd_frame == "mri":
                fwd_trans = Transform("mri", "mri")
            elif fwd_frame == "head":
                fwd_trans = head_mri_t
            else:
                raise RuntimeError(error_msg.format(fwd_frame))
        else:
            raise RuntimeError(error_msg.format(fwd["coord_frame"]))
        for actor in _plot_forward(
            self._renderer,
            fwd,
            fwd_trans,
            fwd_scale=1e3 if self._units == "mm" else 1,
            scale=scale,
            alpha=alpha,
        ):
            self._add_actor("forward", actor)

        self._renderer._update()

    def remove_forward(self):
        """Remove forward sources from the rendered scene."""
        self._remove("forward", render=True)

    @fill_doc
    def add_dipole(
        self, dipole, trans, colors="red", alpha=1, scales=None, *, mode="arrow"
    ):
        """Add a quiver to render positions of dipoles.

        Parameters
        ----------
        dipole : instance of Dipole
            Dipole object containing position, orientation and amplitude of
            one or more dipoles or in the forward solution.
        %(trans_not_none)s
        colors : list | matplotlib-style color | None
            A single color or list of anything matplotlib accepts:
            string, RGB, hex, etc. Default red.
        %(alpha)s Default 1.
        scales : list | float | None
            The size of the arrow representing the dipole in
            :class:`mne.viz.Brain` units. Default 5mm.
        mode : "2darrow" | "arrow" | "cone" | "cylinder" | "sphere" | "oct"
            The drawing mode for the dipole to render.
            Defaults to ``"arrow"``.

        Notes
        -----
        .. versionadded:: 1.0
        """
        head_mri_t = _get_trans(trans, "head", "mri", allow_none=False)[0]
        del trans
        n_dipoles = len(dipole)
        if not isinstance(colors, list | tuple):
            colors = [colors] * n_dipoles  # make into list
        if len(colors) != n_dipoles:
            raise ValueError(
                f"The number of colors ({len(colors)}) "
                f"and dipoles ({n_dipoles}) must match"
            )
        colors = [
            _to_rgb(color, name=f"colors[{ci}]") for ci, color in enumerate(colors)
        ]
        if scales is None:
            scales = 5 if self._units == "mm" else 5e-3
        if not isinstance(scales, list | tuple):
            scales = [scales] * n_dipoles  # make into list
        if len(scales) != n_dipoles:
            raise ValueError(
                f"The number of scales ({len(scales)}) "
                f"and dipoles ({n_dipoles}) must match"
            )
        pos = apply_trans(head_mri_t, dipole.pos)
        pos *= 1e3 if self._units == "mm" else 1
        for _ in self._iter_views("vol"):
            for this_pos, this_ori, color, scale in zip(
                pos, dipole.ori, colors, scales
            ):
                actor, _ = self._renderer.quiver3d(
                    *this_pos,
                    *this_ori,
                    color=color,
                    opacity=alpha,
                    mode=mode,
                    scale=scale,
                )
                self._add_actor("dipole", actor)

        self._renderer._update()

    def remove_dipole(self):
        """Remove dipole objects from the rendered scene."""
        self._remove("dipole", render=True)

    @fill_doc
    def add_head(self, dense=True, color="gray", alpha=0.5):
        """Add a mesh to render the outer head surface.

        Parameters
        ----------
        dense : bool
            Whether to plot the dense head (``seghead``) or the less dense head
            (``head``).
        %(color_matplotlib)s
        %(alpha)s

        Notes
        -----
        .. versionadded:: 0.24
        """
        # load head
        surf = _get_head_surface(
            "seghead" if dense else "head", self._subject, self._subjects_dir
        )
        verts, triangles = surf["rr"], surf["tris"]
        verts *= 1e3 if self._units == "mm" else 1
        color = _to_rgb(color)

        for _ in self._iter_views("vol"):
            actor, _ = self._renderer.mesh(
                *verts.T,
                triangles=triangles,
                color=color,
                opacity=alpha,
                render=False,
            )
            self._add_actor("head", actor)

        self._renderer._update()

    def remove_head(self):
        """Remove head objects from the rendered scene."""
        self._remove("head", render=True)

    @fill_doc
    def add_skull(self, outer=True, color="gray", alpha=0.5):
        """Add a mesh to render the skull surface.

        Parameters
        ----------
        outer : bool
            Adds the outer skull if ``True``, otherwise adds the inner skull.
        %(color_matplotlib)s
        %(alpha)s

        Notes
        -----
        .. versionadded:: 0.24
        """
        surf = _get_skull_surface(
            "outer" if outer else "inner", self._subject, self._subjects_dir
        )
        verts, triangles = surf["rr"], surf["tris"]
        verts *= 1e3 if self._units == "mm" else 1
        color = _to_rgb(color)

        for _ in self._iter_views("vol"):
            actor, _ = self._renderer.mesh(
                *verts.T,
                triangles=triangles,
                color=color,
                opacity=alpha,
                reset_camera=False,
                render=False,
            )
            self._add_actor("skull", actor)

        self._renderer._update()

    def remove_skull(self):
        """Remove skull objects from the rendered scene."""
        self._remove("skull", render=True)

    @fill_doc
    def add_volume_labels(
        self,
        aseg="auto",
        labels=None,
        colors=None,
        alpha=0.5,
        smooth=0.9,
        fill_hole_size=None,
        legend=None,
    ):
        """Add labels to the rendering from an anatomical segmentation.

        Parameters
        ----------
        %(aseg)s
        labels : list
            Labeled regions of interest to plot. See
            :func:`mne.get_montage_volume_labels`
            for one way to determine regions of interest. Regions can also be
            chosen from the :term:`FreeSurfer LUT`.
        colors : list | matplotlib-style color | None
            A list of anything matplotlib accepts: string, RGB, hex, etc.
            (default :term:`FreeSurfer LUT` colors).
        %(alpha)s
        %(smooth)s
        fill_hole_size : int | None
            The size of holes to remove in the mesh in voxels. Default is None,
            no holes are removed. Warning, this dilates the boundaries of the
            surface by ``fill_hole_size`` number of voxels so use the minimal
            size.
        legend : bool | None | dict
            Add a legend displaying the names of the ``labels``. Default (None)
            is ``True`` if the number of ``labels`` is 10 or fewer.
            Can also be a dict of ``kwargs`` to pass to
            ``pyvista.Plotter.add_legend``.

        Notes
        -----
        .. versionadded:: 0.24
        """
        aseg, aseg_data = _get_aseg(aseg, self._subject, self._subjects_dir)

        vox_mri_t = aseg.header.get_vox2ras_tkr()
        mult = 1e-3 if self._units == "m" else 1
        vox_mri_t[:3] *= mult
        del aseg

        # read freesurfer lookup table
        lut, fs_colors = read_freesurfer_lut()
        if labels is None:  # assign default ROI labels based on indices
            lut_r = {v: k for k, v in lut.items()}
            labels = [lut_r[idx] for idx in DEFAULTS["volume_label_indices"]]

        _validate_type(fill_hole_size, (int, None), "fill_hole_size")
        _validate_type(legend, (bool, None, dict), "legend")
        if legend is None:
            legend = len(labels) < 11

        if colors is None:
            colors = [fs_colors[label] / 255 for label in labels]
        elif not isinstance(colors, list | tuple):
            colors = [colors] * len(labels)  # make into list
        colors = [
            _to_rgb(color, name=f"colors[{ci}]") for ci, color in enumerate(colors)
        ]
        surfs = _marching_cubes(
            aseg_data,
            [lut[label] for label in labels],
            smooth=smooth,
            fill_hole_size=fill_hole_size,
        )
        for label, color, (verts, triangles) in zip(labels, colors, surfs):
            if len(verts) == 0:  # not in aseg vals
                warn(
                    f"Value {lut[label]} not found for label "
                    f"{repr(label)} in anatomical segmentation file "
                )
                continue
            verts = apply_trans(vox_mri_t, verts)
            for _ in self._iter_views("vol"):
                actor, _ = self._renderer.mesh(
                    *verts.T,
                    triangles=triangles,
                    color=color,
                    opacity=alpha,
                    reset_camera=False,
                    render=False,
                )
                self._add_actor("volume_labels", actor)

        if legend or isinstance(legend, dict):
            # use empty kwargs for legend = True
            legend = legend if isinstance(legend, dict) else dict()
            self._renderer.plotter.add_legend(list(zip(labels, colors)), **legend)

        self._renderer._update()

    def remove_volume_labels(self):
        """Remove the volume labels from the rendered scene."""
        self._remove("volume_labels", render=True)
        self._renderer.plotter.remove_legend()

    @fill_doc
    def add_foci(
        self,
        coords,
        coords_as_verts=False,
        map_surface=None,
        scale_factor=1,
        color="white",
        alpha=1,
        name=None,
        hemi=None,
        resolution=50,
    ):
        """Add spherical foci, possibly mapping to displayed surf.

        The foci spheres can be displayed at the coordinates given, or
        mapped through a surface geometry. In other words, coordinates
        from a volume-based analysis in MNI space can be displayed on an
        inflated average surface by finding the closest vertex on the
        white surface and mapping to that vertex on the inflated mesh.

        Parameters
        ----------
        coords : ndarray, shape (n_coords, 3)
            Coordinates in stereotaxic space (default) or array of
            vertex ids (with ``coord_as_verts=True``).
        coords_as_verts : bool
            Whether the coords parameter should be interpreted as vertex ids.
        map_surface : str | None
            Surface to project the coordinates to, or None to use raw coords.
            When set to a surface, each foci is positioned at the closest
            vertex in the mesh.
        scale_factor : float
            Controls the size of the foci spheres (relative to 1cm).
        %(color_matplotlib)s
        %(alpha)s Default is 1.
        name : str
            Internal name to use.
        hemi : str | None
            If None, it is assumed to belong to the hemisphere being
            shown. If two hemispheres are being shown, an error will
            be thrown.
        resolution : int
            The resolution of the spheres.
        """
        hemi = self._check_hemi(hemi, extras=["vol"])

        # Figure out how to interpret the first parameter
        if coords_as_verts:
            coords = self.geo[hemi].coords[coords]
            map_surface = None

        # Possibly map the foci coords through a surface
        if map_surface is not None:
            foci_surf = _Surface(
                self._subject,
                hemi,
                map_surface,
                self._subjects_dir,
                offset=0,
                units=self._units,
                x_dir=self._rigid[0, :3],
            )
            foci_surf.load_geometry()
            foci_vtxs = np.argmin(cdist(foci_surf.coords, coords), axis=0)
            coords = self.geo[hemi].coords[foci_vtxs]

        # Convert the color code
        color = _to_rgb(color)

        if self._units == "m":
            scale_factor = scale_factor / 1000.0

        for _, _, v in self._iter_views(hemi):
            self._renderer.sphere(
                center=coords,
                color=color,
                scale=(10.0 * scale_factor),
                opacity=alpha,
                resolution=resolution,
            )
            self._set_camera(**views_dicts[hemi][v])
        self._renderer._update()

        # Store the foci in the Brain._data dictionary
        data_foci = coords
        if "foci" in self._data.get(hemi, []):
            data_foci = np.vstack((self._data[hemi]["foci"], data_foci))
        self._data[hemi] = self._data.get(hemi, dict())  # no data added yet
        self._data[hemi]["foci"] = data_foci

    @verbose
    def add_sensors(
        self,
        info,
        trans,
        meg=None,
        eeg="original",
        fnirs=True,
        ecog=True,
        seeg=True,
        dbs=True,
        max_dist=0.004,
        *,
        sensor_colors=None,
        sensor_scales=None,
        verbose=None,
    ):
        """Add mesh objects to represent sensor positions.

        Parameters
        ----------
        %(info_not_none)s
        %(trans_not_none)s
        %(meg)s
        %(eeg)s
        %(fnirs)s
        %(ecog)s
        %(seeg)s
        %(dbs)s
        %(max_dist_ieeg)s
        %(sensor_colors)s

            .. versionadded:: 1.6
        %(sensor_scales)s

            .. versionadded:: 1.9
        %(verbose)s

        Notes
        -----
        .. versionadded:: 0.24
        """
        from ...preprocessing.ieeg._projection import _project_sensors_onto_inflated

        _validate_type(info, Info, "info")
        meg, eeg, fnirs, warn_meg, sensor_alpha = _handle_sensor_types(meg, eeg, fnirs)
        picks = pick_types(
            info,
            meg=("sensors" in meg),
            ref_meg=("ref" in meg),
            eeg=(len(eeg) > 0),
            ecog=ecog,
            seeg=seeg,
            dbs=dbs,
            fnirs=(len(fnirs) > 0),
        )
        head_mri_t = _get_trans(trans, "head", "mri", allow_none=False)[0]
        if self._surf in ("inflated", "flat"):
            for modality, check in dict(seeg=seeg, ecog=ecog).items():
                if pick_types(info, **{modality: check}).size > 0:
                    info = _project_sensors_onto_inflated(
                        info.copy(),
                        head_mri_t,
                        subject=self._subject,
                        subjects_dir=self._subjects_dir,
                        picks=modality,
                        max_dist=max_dist,
                        flat=self._surf == "flat",
                    )
        del trans
        # get transforms to "mri" window
        to_cf_t = _get_transforms_to_coord_frame(info, head_mri_t, coord_frame="mri")
        if pick_types(info, eeg=True, exclude=()).size > 0 and "projected" in eeg:
            head_surf = _get_head_surface("seghead", self._subject, self._subjects_dir)
        else:
            head_surf = None
        # Do the main plotting
        for _ in self._iter_views("vol"):
            if picks.size > 0:
                sensors_actors = _plot_sensors_3d(
                    self._renderer,
                    info,
                    to_cf_t,
                    picks,
                    meg,
                    eeg,
                    fnirs,
                    warn_meg,
                    head_surf,
                    self._units,
                    sensor_alpha=sensor_alpha,
                    sensor_colors=sensor_colors,
                    sensor_scales=sensor_scales,
                )
                # sensors_actors can still be None
                for item, actors in (sensors_actors or {}).items():
                    for actor in actors:
                        self._add_actor(item, actor)

            if "helmet" in meg and pick_types(info, meg=True).size > 0:
                actor, _, _ = _plot_helmet(
                    self._renderer,
                    info,
                    to_cf_t,
                    head_mri_t,
                    "mri",
                    alpha=sensor_alpha["meg_helmet"],
                    scale=1 if self._units == "m" else 1e3,
                )
                self._add_actor("helmet", actor)

        self._renderer._update()

    def remove_sensors(self, kind=None):
        """Remove sensors from the rendered scene.

        Parameters
        ----------
        kind : str | list | None
            If None, removes all sensor-related data including the helmet.
            Can be "meg", "eeg", "fnirs", "ecog", "seeg", "dbs" or "helmet"
            to remove that item.
        """
        all_kinds = ("meg", "eeg", "fnirs", "ecog", "seeg", "dbs", "helmet")
        if kind is None:
            for item in all_kinds:
                self._remove(item, render=False)
        else:
            if isinstance(kind, str):
                kind = [kind]
            for this_kind in kind:
                _check_option("kind", this_kind, all_kinds)
            self._remove(this_kind, render=False)
        self._renderer._update()

    def add_text(
        self,
        x,
        y,
        text,
        name=None,
        color=None,
        opacity=1.0,
        row=0,
        col=0,
        font_size=None,
        justification=None,
    ):
        """Add a text to the visualization.

        Parameters
        ----------
        x : float
            X coordinate.
        y : float
            Y coordinate.
        text : str
            Text to add.
        name : str
            Name of the text (text label can be updated using update_text()).
        color : tuple
            Color of the text. Default is the foreground color set during
            initialization (default is black or white depending on the
            background color).
        opacity : float
            Opacity of the text (default 1.0).
        row : int | None
            Row index of which brain to use. Default is the top row.
        col : int | None
            Column index of which brain to use. Default is the left-most
            column.
        font_size : float | None
            The font size to use.
        justification : str | None
            The text justification.
        """
        _validate_type(name, (str, None), "name")
        name = text if name is None else name
        if "text" in self._actors and name in self._actors["text"]:
            raise ValueError(f"Text with the name {name} already exists")
        if color is None:
            color = self._fg_color
        for ri, ci, _ in self._iter_views("vol"):
            if (row is None or row == ri) and (col is None or col == ci):
                actor = self._renderer.text2d(
                    x_window=x,
                    y_window=y,
                    text=text,
                    color=color,
                    size=font_size,
                    justification=justification,
                )
                if "text" not in self._actors:
                    self._actors["text"] = dict()
                self._actors["text"][name] = actor

    def remove_text(self, name=None):
        """Remove text from the rendered scene.

        Parameters
        ----------
        name : str | None
            Remove specific text by name. If None, all text will be removed.
        """
        _validate_type(name, (str, None), "name")
        if name is None:
            for actor in self._actors["text"].values():
                self._renderer.plotter.remove_actor(actor, render=False)
            self._actors.pop("text")
        else:
            names = [None]
            if "text" in self._actors:
                names += list(self._actors["text"].keys())
            _check_option("name", name, names)
            self._renderer.plotter.remove_actor(
                self._actors["text"][name], render=False
            )
            self._actors["text"].pop(name)
        self._renderer._update()

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

    @fill_doc
    def add_annotation(
        self, annot, borders=True, alpha=1, hemi=None, remove_existing=True, color=None
    ):
        """Add an annotation file.

        Parameters
        ----------
        annot : str | tuple
            Either path to annotation file or annotation name. Alternatively,
            the annotation can be specified as a ``(labels, ctab)`` tuple per
            hemisphere, i.e. ``annot=(labels, ctab)`` for a single hemisphere
            or ``annot=((lh_labels, lh_ctab), (rh_labels, rh_ctab))`` for both
            hemispheres. ``labels`` and ``ctab`` should be arrays as returned
            by :func:`nibabel.freesurfer.io.read_annot`.
        borders : bool | int
            Show only label borders. If int, specify the number of steps
            (away from the true border) along the cortical mesh to include
            as part of the border definition.
        %(alpha)s Default is 1.
        hemi : str | None
            If None, it is assumed to belong to the hemisphere being
            shown. If two hemispheres are being shown, data must exist
            for both hemispheres.
        remove_existing : bool
            If True (default), remove old annotations.
        color : matplotlib-style color code
            If used, show all annotations in the same (specified) color.
            Probably useful only when showing annotation borders.
        """
        from ...label import _read_annot

        hemis = self._check_hemis(hemi)

        # Figure out where the data is coming from
        if _path_like(annot):
            if os.path.isfile(annot):
                filepath = _check_fname(annot, overwrite="read")
                file_hemi, annot = filepath.name.split(".", 1)
                if len(hemis) > 1:
                    if file_hemi == "lh":
                        filepaths = [filepath, filepath.parent / ("rh." + annot)]
                    elif file_hemi == "rh":
                        filepaths = [filepath.parent / ("lh." + annot), filepath]
                    else:
                        raise RuntimeError(
                            "To add both hemispheres simultaneously, filename must "
                            'begin with "lh." or "rh."'
                        )
                else:
                    filepaths = [filepath]
            else:
                filepaths = []
                for hemi in hemis:
                    filepath = op.join(
                        self._subjects_dir,
                        self._subject,
                        "label",
                        ".".join([hemi, annot, "annot"]),
                    )
                    if not os.path.exists(filepath):
                        raise ValueError(f"Annotation file {filepath} does not exist")
                    filepaths += [filepath]
            annots = []
            for hemi, filepath in zip(hemis, filepaths):
                # Read in the data
                labels, cmap, _ = _read_annot(filepath)
                annots.append((labels, cmap))
        else:
            annots = [annot] if len(hemis) == 1 else annot
            annot = "annotation"

        for hemi, (labels, cmap) in zip(hemis, annots):
            # Maybe zero-out the non-border vertices
            self._to_borders(labels, hemi, borders)

            # Handle null labels properly
            cmap[:, 3] = 255
            bgcolor = np.round(np.array(self._brain_color) * 255).astype(int)
            bgcolor[-1] = 0
            cmap[cmap[:, 4] < 0, 4] += 2**24  # wrap to positive
            cmap[cmap[:, 4] <= 0, :4] = bgcolor
            if np.any(labels == 0) and not np.any(cmap[:, -1] <= 0):
                cmap = np.vstack((cmap, np.concatenate([bgcolor, [0]])))

            # Set label ids sensibly
            order = np.argsort(cmap[:, -1])
            cmap = cmap[order]
            ids = np.searchsorted(cmap[:, -1], labels)
            cmap = cmap[:, :4]

            #  Set the alpha level
            alpha_vec = cmap[:, 3]
            alpha_vec[alpha_vec > 0] = alpha * 255

            # Override the cmap when a single color is used
            if color is not None:
                rgb = np.round(np.multiply(_to_rgb(color), 255))
                cmap[:, :3] = rgb.astype(cmap.dtype)

            ctable = cmap.astype(np.float64)
            for _ in self._iter_views(hemi):
                mesh = self._layered_meshes[hemi]
                mesh.add_overlay(
                    scalars=ids,
                    colormap=ctable,
                    rng=[np.min(ids), np.max(ids)],
                    opacity=alpha,
                    name=annot,
                )
                self._annots[hemi].append(annot)
                if not self.time_viewer or self.traces_mode == "vertex":
                    self._renderer._set_colormap_range(
                        mesh._actor, cmap.astype(np.uint8), None
                    )

        self._renderer._update()

    def close(self):
        """Close all figures and cleanup data structure."""
        self._closed = True
        self._renderer.close()

    def show(self):
        """Display the window."""
        self._renderer.show()

    @fill_doc
    def get_view(self, row=0, col=0, *, align=True):
        """Get the camera orientation for a given subplot display.

        Parameters
        ----------
        row : int
            The row to use, default is the first one.
        col : int
            The column to check, the default is the first one.
        %(align_view)s

        Returns
        -------
        %(roll)s
        %(distance)s
        %(azimuth)s
        %(elevation)s
        %(focalpoint)s
        """
        row = _ensure_int(row, "row")
        col = _ensure_int(col, "col")
        rigid = self._rigid if align else None
        for h in self._hemis:
            for ri, ci, _ in self._iter_views(h):
                if (row == ri) and (col == ci):
                    return self._renderer.get_camera(rigid=rigid)
        return (None,) * 5

    @verbose
    def show_view(
        self,
        view=None,
        roll=None,
        distance=None,
        *,
        row=None,
        col=None,
        hemi=None,
        align=True,
        azimuth=None,
        elevation=None,
        focalpoint=None,
        update=True,
        verbose=None,
    ):
        """Orient camera to display view.

        Parameters
        ----------
        %(view)s
        %(roll)s
        %(distance)s
        row : int | None
            The row to set. Default all rows.
        col : int | None
            The column to set. Default all columns.
        hemi : str | None
            Which hemi to use for view lookup (when in "both" mode).
        %(align_view)s
        %(azimuth)s
        %(elevation)s
        %(focalpoint)s
        %(brain_update)s

            .. versionadded:: 1.6
        %(verbose)s

        Notes
        -----
        The builtin string views are the following perspectives, based on the
        :term:`RAS` convention. If not otherwise noted, the view will have the
        top of the brain (superior, +Z) in 3D space shown upward in the 2D
        perspective:

        ``'lateral'``
            From the left or right side such that the lateral (outside)
            surface of the given hemisphere is visible.
        ``'medial'``
            From the left or right side such that the medial (inside)
            surface of the given hemisphere is visible (at least when in split
            or single-hemi mode).
        ``'rostral'``
            From the front.
        ``'caudal'``
            From the rear.
        ``'dorsal'``
            From above, with the front of the brain pointing up.
        ``'ventral'``
            From below, with the front of the brain pointing up.
        ``'frontal'``
            From the front and slightly lateral, with the brain slightly
            tilted forward (yielding a view from slightly above).
        ``'parietal'``
            From the rear and slightly lateral, with the brain slightly tilted
            backward (yielding a view from slightly above).
        ``'axial'``
            From above with the brain pointing up (same as ``'dorsal'``).
        ``'sagittal'``
            From the right side.
        ``'coronal'``
            From the rear.

        Three letter abbreviations (e.g., ``'lat'``) of all of the above are
        also supported.
        """
        _validate_type(row, ("int-like", None), "row")
        _validate_type(col, ("int-like", None), "col")
        hemi = self._hemi if hemi is None else hemi
        if hemi == "split":
            if (
                self._view_layout == "vertical"
                and col == 1
                or self._view_layout == "horizontal"
                and row == 1
            ):
                hemi = "rh"
            else:
                hemi = "lh"
        _validate_type(view, (str, None), "view")
        view_params = dict(
            azimuth=azimuth,
            elevation=elevation,
            roll=roll,
            distance=distance,
            focalpoint=focalpoint,
        )
        if view is not None:  # view_params take precedence
            view_params = {
                param: val for param, val in view_params.items() if val is not None
            }  # no overwriting with None
            view_params = dict(views_dicts[hemi].get(view), **view_params)
        for h in self._hemis:
            for ri, ci, _ in self._iter_views(h):
                if (row is None or row == ri) and (col is None or col == ci):
                    self._set_camera(**view_params, align=align)
        if update:
            self._renderer._update()

    def _set_camera(
        self,
        *,
        distance=None,
        focalpoint=None,
        update=False,
        align=True,
        verbose=None,
        **kwargs,
    ):
        # Wrap to self._renderer.set_camera safely, always passing self._rigid
        # and using better no-op-like defaults
        return self._renderer.set_camera(
            distance=distance,
            focalpoint=focalpoint,
            update=update,
            rigid=self._rigid if align else None,
            **kwargs,
        )

    def reset_view(self):
        """Reset the camera."""
        for h in self._hemis:
            for _, _, v in self._iter_views(h):
                self._set_camera(**views_dicts[h][v])
        self._renderer._update()

    def save_image(self, filename=None, mode="rgb"):
        """Save view from all panels to disk.

        Parameters
        ----------
        filename : path-like
            Path to new image file.
        mode : str
            Either ``'rgb'`` or ``'rgba'`` for values to return.
        """
        if filename is None:
            filename = _generate_default_filename(".png")
        _save_ndarray_img(filename, self.screenshot(mode=mode, time_viewer=True))

    @fill_doc
    def screenshot(self, mode="rgb", time_viewer=False):
        """Generate a screenshot of current view.

        Parameters
        ----------
        mode : str
            Either ``'rgb'`` or ``'rgba'`` for values to return.
        %(time_viewer_brain_screenshot)s

        Returns
        -------
        screenshot : array
            Image pixel values.
        """
        n_channels = 3 if mode == "rgb" else 4
        img = self._renderer.screenshot(mode)
        logger.debug(f"Got screenshot of size {img.shape}")
        if (
            time_viewer
            and self.time_viewer
            and self.show_traces
            and not self.separate_canvas
        ):
            from matplotlib.image import imread

            canvas = self.mpl_canvas.fig.canvas
            canvas.draw_idle()
            fig = self.mpl_canvas.fig
            with BytesIO() as output:
                # Need to pass dpi here so it uses the physical (HiDPI) DPI
                # rather than logical DPI when saving in most cases.
                # But when matplotlib uses HiDPI and VTK doesn't
                # (e.g., macOS w/Qt 5.14+ and VTK9) then things won't work,
                # so let's just calculate the DPI we need to get
                # the correct size output based on the widths being equal
                size_in = fig.get_size_inches()
                dpi = fig.get_dpi()
                want_size = tuple(x * dpi for x in size_in)
                n_pix = want_size[0] * want_size[1]
                logger.debug(
                    f"Saving figure of size {size_in} @ {dpi} DPI "
                    f"({want_size} = {n_pix} pixels)"
                )
                # Sometimes there can be off-by-one errors here (e.g.,
                # if in mpl int() rather than int(round()) is used to
                # compute the number of pixels) so rather than use "raw"
                # format and try to reshape ourselves, just write to PNG
                # and read it, which has the dimensions encoded for us.
                fig.savefig(
                    output,
                    dpi=dpi,
                    format="png",
                    facecolor=self._bg_color,
                    edgecolor="none",
                )
                output.seek(0)
                trace_img = imread(output, format="png")[:, :, :n_channels]
                trace_img = np.clip(np.round(trace_img * 255), 0, 255).astype(np.uint8)
            bgcolor = np.array(self._brain_color[:n_channels]) / 255
            img = concatenate_images(
                [img, trace_img], bgcolor=bgcolor, n_channels=n_channels
            )
        return img

    @fill_doc
    def update_lut(self, fmin=None, fmid=None, fmax=None, alpha=None):
        """Update the range of the color map.

        Parameters
        ----------
        %(fmin_fmid_fmax)s
        %(alpha)s
        """
        publish(
            self,
            ColormapRange(
                kind="distributed_source_power",
                fmin=fmin,
                fmid=fmid,
                fmax=fmax,
                alpha=alpha,
            ),
        )

    @fill_doc
    def _update_colormap_range(self, fmin=None, fmid=None, fmax=None, alpha=None):
        """Update the range of the color map.

        Parameters
        ----------
        %(fmin_fmid_fmax)s
        %(alpha)s
        """
        args = f"{fmin}, {fmid}, {fmax}, {alpha}"
        logger.debug(f"Updating LUT with {args}")
        center = self._data["center"]
        colormap = self._data["colormap"]
        transparent = self._data["transparent"]
        lims = {key: self._data[key] for key in ("fmin", "fmid", "fmax")}
        _update_monotonic(lims, fmin=fmin, fmid=fmid, fmax=fmax)
        assert all(val is not None for val in lims.values())

        self._data.update(lims)
        self._data["ctable"] = np.round(
            calculate_lut(
                colormap, alpha=1.0, center=center, transparent=transparent, **lims
            )
            * 255
        ).astype(np.uint8)
        # update our values
        rng = self._cmap_range
        ctable = self._data["ctable"]
        for hemi in ["lh", "rh", "vol"]:
            hemi_data = self._data.get(hemi)
            if hemi_data is not None:
                if hemi in self._layered_meshes:
                    mesh = self._layered_meshes[hemi]
                    mesh.update_overlay(
                        name="data",
                        colormap=self._data["ctable"],
                        opacity=alpha,
                        rng=rng,
                    )
                    self._renderer._set_colormap_range(
                        mesh._actor, ctable, self._scalar_bar, rng, self._brain_color
                    )

                grid_volume_pos = hemi_data.get("grid_volume_pos")
                grid_volume_neg = hemi_data.get("grid_volume_neg")
                for grid_volume in (grid_volume_pos, grid_volume_neg):
                    if grid_volume is not None:
                        self._renderer._set_volume_range(
                            grid_volume,
                            ctable,
                            hemi_data["alpha"],
                            self._scalar_bar,
                            rng,
                        )

                glyph_actor = hemi_data.get("glyph_actor")
                if glyph_actor is not None:
                    for glyph_actor_ in glyph_actor:
                        self._renderer._set_colormap_range(
                            glyph_actor_, ctable, self._scalar_bar, rng
                        )
        self._renderer._update()

    def set_data_smoothing(self, n_steps):
        """Set the number of smoothing steps.

        Parameters
        ----------
        n_steps : int
            Number of smoothing steps.
        """
        from ...morph import _hemi_morph

        for hemi in ["lh", "rh"]:
            hemi_data = self._data.get(hemi)
            if hemi_data is not None:
                if len(hemi_data["array"]) >= self.geo[hemi].x.shape[0]:
                    continue
                vertices = hemi_data["vertices"]
                if vertices is None:
                    raise ValueError(
                        f"len(data) < nvtx ({len(hemi_data)} < "
                        f"{self.geo[hemi].x.shape[0]}): the vertices "
                        "parameter must not be None"
                    )
                morph_n_steps = "nearest" if n_steps == -1 else n_steps
                with use_log_level(False):
                    smooth_mat = _hemi_morph(
                        self.geo[hemi].orig_faces,
                        np.arange(len(self.geo[hemi].coords)),
                        vertices,
                        morph_n_steps,
                        maps=None,
                        warn=False,
                    )
                self._data[hemi]["smooth_mat"] = smooth_mat
        self._update_current_time_idx(self._data["time_idx"])
        self._data["smoothing_steps"] = n_steps

    @property
    def _n_times(self):
        return len(self._times) if self._times is not None else None

    @property
    def time_interpolation(self):
        """The interpolation mode."""
        return self._time_interpolation

    @fill_doc
    def set_time_interpolation(self, interpolation):
        """Set the interpolation mode.

        Parameters
        ----------
        %(interpolation_brain_time)s
        """
        self._time_interpolation = _check_option(
            "interpolation",
            interpolation,
            ("linear", "nearest", "zero", "slinear", "quadratic", "cubic"),
        )
        self._time_interp_funcs = dict()
        self._time_interp_inv = None
        if self._times is not None:
            idx = np.arange(self._n_times)
            for hemi in ["lh", "rh", "vol"]:
                hemi_data = self._data.get(hemi)
                if hemi_data is not None:
                    array = hemi_data["array"]
                    self._time_interp_funcs[hemi] = _safe_interp1d(
                        idx,
                        array,
                        self._time_interpolation,
                        axis=-1,
                        assume_sorted=True,
                    )
            self._time_interp_inv = _safe_interp1d(idx, self._times)

    def _update_current_time_idx(self, time_idx):
        """Update all widgets in the figure to reflect a new time point.

        Parameters
        ----------
        time_idx : int | float
            The time index to use. Can be a float to use interpolation
            between indices.
        """
        self._current_act_data = dict()
        time_actor = self._data.get("time_actor", None)
        time_label = self._data.get("time_label", None)
        for hemi in ["lh", "rh", "vol"]:
            hemi_data = self._data.get(hemi)
            if hemi_data is not None:
                array = hemi_data["array"]
                # interpolate in time
                vectors = None
                if array.ndim == 1:
                    act_data = array
                    self._current_time = 0
                else:
                    act_data = self._time_interp_funcs[hemi](time_idx)
                    self._current_time = self._time_interp_inv(time_idx)
                    if array.ndim == 3:
                        vectors = act_data
                        act_data = np.linalg.norm(act_data, axis=1)
                    self._current_time = self._time_interp_inv(time_idx)
                self._current_act_data[hemi] = act_data
                if time_actor is not None and time_label is not None:
                    time_actor.SetInput(time_label(self._current_time))

                # update the volume interpolation
                grid = hemi_data.get("grid")
                if grid is not None:
                    vertices = self._data["vol"]["vertices"]
                    values = self._current_act_data["vol"]
                    rng = self._cmap_range
                    fill = 0 if self._data["center"] is not None else rng[0]
                    grid.cell_data["values"].fill(fill)
                    # XXX for sided data, we probably actually need two
                    # volumes as composite/MIP needs to look at two
                    # extremes... for now just use abs. Eventually we can add
                    # two volumes if we want.
                    grid.cell_data["values"][vertices] = values

                # interpolate in space
                smooth_mat = hemi_data.get("smooth_mat")
                if smooth_mat is not None:
                    act_data = smooth_mat.dot(act_data)

                # update the mesh scalar values
                if hemi in self._layered_meshes:
                    mesh = self._layered_meshes[hemi]
                    if "data" in mesh._overlays:
                        mesh.update_overlay(name="data", scalars=act_data)
                    else:
                        mesh.add_overlay(
                            scalars=act_data,
                            colormap=self._data["ctable"],
                            rng=self._cmap_range,
                            opacity=None,
                            name="data",
                        )

                # update the glyphs
                if vectors is not None:
                    self._update_glyphs(hemi, vectors)

        self._data["time_idx"] = time_idx
        self._renderer._update()

    def set_time_point(self, time_idx):
        """Set the time point to display (can be a float to interpolate).

        Parameters
        ----------
        time_idx : int | float
            The time index to use. Can be a float to use interpolation
            between indices.
        """
        if self._times is None:
            raise ValueError("Cannot set time when brain has no defined times.")
        elif 0 <= time_idx <= len(self._times):
            publish(self, TimeChange(time=self._time_interp_inv(time_idx)))
        else:
            raise ValueError(
                f"Requested time point ({time_idx}) is outside the range of "
                f"available time points (0-{len(self._times)})."
            )

    def set_time(self, time):
        """Set the time to display (in seconds).

        Parameters
        ----------
        time : float
            The time to show, in seconds.
        """
        if self._times is None:
            raise ValueError("Cannot set time when brain has no defined times.")
        elif min(self._times) <= time <= max(self._times):
            publish(self, TimeChange(time=time))
        else:
            raise ValueError(
                f"Requested time ({time} s) is outside the range of "
                f"available times ({min(self._times)}-{max(self._times)} s)."
            )

    def _update_glyphs(self, hemi, vectors):
        hemi_data = self._data.get(hemi)
        assert hemi_data is not None
        vertices = hemi_data["vertices"]
        vector_alpha = self._data["vector_alpha"]
        scale_factor = self._data["scale_factor"]
        vertices = slice(None) if vertices is None else vertices
        x, y, z = np.array(self.geo[hemi].coords)[vertices].T

        if hemi_data["glyph_actor"] is None:
            add = True
            hemi_data["glyph_actor"] = list()
        else:
            add = False
        count = 0
        for _ in self._iter_views(hemi):
            if hemi_data["glyph_dataset"] is None:
                glyph_mapper, glyph_dataset = self._renderer.quiver3d(
                    x,
                    y,
                    z,
                    vectors[:, 0],
                    vectors[:, 1],
                    vectors[:, 2],
                    color=None,
                    mode="2darrow",
                    scale_mode="vector",
                    scale=scale_factor,
                    opacity=vector_alpha,
                )
                hemi_data["glyph_dataset"] = glyph_dataset
                hemi_data["glyph_mapper"] = glyph_mapper
            else:
                glyph_dataset = hemi_data["glyph_dataset"]
                glyph_dataset.point_data["vec"] = vectors
                glyph_mapper = hemi_data["glyph_mapper"]
            if add:
                glyph_actor = self._renderer._actor(glyph_mapper)
                prop = glyph_actor.GetProperty()
                prop.SetLineWidth(2.0)
                prop.SetOpacity(vector_alpha)
                self._renderer.plotter.add_actor(glyph_actor, render=False)
                hemi_data["glyph_actor"].append(glyph_actor)
            else:
                glyph_actor = hemi_data["glyph_actor"][count]
            count += 1
            self._renderer._set_colormap_range(
                actor=glyph_actor,
                ctable=self._data["ctable"],
                scalar_bar=None,
                rng=self._cmap_range,
            )

    @property
    def _cmap_range(self):
        dt_max = self._data["fmax"]
        if self._data["center"] is None:
            dt_min = self._data["fmin"]
        else:
            dt_min = -1 * dt_max
        rng = [dt_min, dt_max]
        return rng

    def _update_fscale(self, fscale):
        """Scale the colorbar points."""
        fmin = self._data["fmin"] * fscale
        fmid = self._data["fmid"] * fscale
        fmax = self._data["fmax"] * fscale
        self.update_lut(fmin=fmin, fmid=fmid, fmax=fmax)

    def _update_auto_scaling(self, restore=False):
        user_clim = self._data["clim"]
        if user_clim is not None and "lims" in user_clim:
            allow_pos_lims = False
        else:
            allow_pos_lims = True
        if user_clim is not None and restore:
            clim = user_clim
        else:
            clim = "auto"
        colormap = self._data["colormap"]
        transparent = self._data["transparent"]
        mapdata = _process_clim(
            clim,
            colormap,
            transparent,
            np.concatenate(list(self._current_act_data.values())),
            allow_pos_lims,
        )
        diverging = "pos_lims" in mapdata["clim"]
        colormap = mapdata["colormap"]
        scale_pts = mapdata["clim"]["pos_lims" if diverging else "lims"]
        transparent = mapdata["transparent"]
        del mapdata
        fmin, fmid, fmax = scale_pts
        center = 0.0 if diverging else None
        self._data["center"] = center
        self._data["colormap"] = colormap
        self._data["transparent"] = transparent
        self.update_lut(fmin=fmin, fmid=fmid, fmax=fmax)

    def _to_time_index(self, value):
        """Return the interpolated time index of the given time value."""
        time = self._data["time"]
        value = np.interp(value, time, np.arange(len(time)))
        return value

    @property
    def data(self):
        """Data used by time viewer and color bar widgets."""
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def views(self):
        return self._views

    @property
    def hemis(self):
        return self._hemis

    def _save_movie(
        self,
        filename,
        time_dilation=4.0,
        tmin=None,
        tmax=None,
        framerate=24,
        interpolation=None,
        codec=None,
        bitrate=None,
        callback=None,
        time_viewer=False,
        **kwargs,
    ):
        import imageio

        with self._renderer._disabled_interaction():
            images = self._make_movie_frames(
                time_dilation,
                tmin,
                tmax,
                framerate,
                interpolation,
                callback,
                time_viewer,
            )
        # find imageio FFMPEG parameters
        if "fps" not in kwargs:
            kwargs["fps"] = framerate
        if codec is not None:
            kwargs["codec"] = codec
        if bitrate is not None:
            kwargs["bitrate"] = bitrate
        # when using GIF we need to convert FPS to duration in milliseconds for Pillow
        if str(filename).endswith(".gif"):
            kwargs["duration"] = 1000 * len(images) / kwargs.pop("fps")
        imageio.mimwrite(filename, images, **kwargs)

    def _save_movie_tv(
        self,
        filename,
        time_dilation=4.0,
        tmin=None,
        tmax=None,
        framerate=24,
        interpolation=None,
        codec=None,
        bitrate=None,
        callback=None,
        time_viewer=False,
        **kwargs,
    ):
        def frame_callback(frame, n_frames):
            if frame == n_frames:
                # On the ImageIO step
                self.status_msg.set_value(f"Saving with ImageIO: {filename}")
                self.status_msg.show()
                self.status_progress.hide()
                self._renderer._status_bar_update()
            else:
                self.status_msg.set_value(
                    f"Rendering images (frame {frame + 1} / {n_frames}) ..."
                )
                self.status_msg.show()
                self.status_progress.show()
                self.status_progress.set_range([0, n_frames - 1])
                self.status_progress.set_value(frame)
                self.status_progress.update()
            self.status_msg.update()
            self._renderer._status_bar_update()

        # set cursor to busy
        default_cursor = self._renderer._window_get_cursor()
        self._renderer._window_set_cursor(
            self._renderer._window_new_cursor("WaitCursor")
        )

        try:
            self._save_movie(
                filename,
                time_dilation,
                tmin,
                tmax,
                framerate,
                interpolation,
                codec,
                bitrate,
                frame_callback,
                time_viewer,
                **kwargs,
            )
        except (Exception, KeyboardInterrupt):
            warn("Movie saving aborted:\n" + traceback.format_exc())
        finally:
            self._renderer._window_set_cursor(default_cursor)

    @fill_doc
    def save_movie(
        self,
        filename=None,
        time_dilation=4.0,
        tmin=None,
        tmax=None,
        framerate=24,
        interpolation=None,
        codec=None,
        bitrate=None,
        callback=None,
        time_viewer=False,
        **kwargs,
    ):
        """Save a movie (for data with a time axis).

        The movie is created through the :mod:`imageio` module. The format is
        determined by the extension, and additional options can be specified
        through keyword arguments that depend on the format, see
        :doc:`imageio's format page <imageio:formats/index>`.

        .. Warning::
            This method assumes that time is specified in seconds when adding
            data. If time is specified in milliseconds this will result in
            movies 1000 times longer than expected.

        Parameters
        ----------
        filename : str
            Path at which to save the movie. The extension determines the
            format (e.g., ``'*.mov'``, ``'*.gif'``, ...; see the :mod:`imageio`
            documentation for available formats).
        time_dilation : float
            Factor by which to stretch time (default 4). For example, an epoch
            from -100 to 600 ms lasts 700 ms. With ``time_dilation=4`` this
            would result in a 2.8 s long movie.
        tmin : float
            First time point to include (default: all data).
        tmax : float
            Last time point to include (default: all data).
        framerate : float
            Framerate of the movie (frames per second, default 24).
        %(interpolation_brain_time)s
            If None, it uses the current ``brain.interpolation``,
            which defaults to ``'nearest'``. Defaults to None.
        codec : str | None
            The codec to use.
        bitrate : float | None
            The bitrate to use.
        callback : callable | None
            A function to call on each iteration. Useful for status message
            updates. It will be passed keyword arguments ``frame`` and
            ``n_frames``.
        %(time_viewer_brain_screenshot)s
        **kwargs : dict
            Specify additional options for :mod:`imageio`.
        """
        if filename is None:
            filename = _generate_default_filename(".mp4")
        func = self._save_movie_tv if self.time_viewer else self._save_movie
        func(
            filename,
            time_dilation,
            tmin,
            tmax,
            framerate,
            interpolation,
            codec,
            bitrate,
            callback,
            time_viewer,
            **kwargs,
        )

    def _make_movie_frames(
        self, time_dilation, tmin, tmax, framerate, interpolation, callback, time_viewer
    ):
        from math import floor

        # find tmin
        if tmin is None:
            tmin = self._times[0]
        elif tmin < self._times[0]:
            raise ValueError(
                f"tmin={repr(tmin)} is smaller than the first time point "
                f"({repr(self._times[0])})"
            )

        # find indexes at which to create frames
        if tmax is None:
            tmax = self._times[-1]
        elif tmax > self._times[-1]:
            raise ValueError(
                f"tmax={repr(tmax)} is greater than the latest time point "
                f"({repr(self._times[-1])})"
            )
        n_frames = floor((tmax - tmin) * time_dilation * framerate)
        times = np.arange(n_frames, dtype=float)
        times /= framerate * time_dilation
        times += tmin
        time_idx = np.interp(times, self._times, np.arange(self._n_times))

        n_times = len(time_idx)
        if n_times == 0:
            raise ValueError("No time points selected")

        logger.debug(f"Save movie for time points/samples\n{times}\n{time_idx}")
        # Sometimes the first screenshot is rendered with a different
        # resolution on OS X
        self.screenshot(time_viewer=time_viewer)
        old_mode = self.time_interpolation
        if interpolation is not None:
            self.set_time_interpolation(interpolation)
        try:
            images = [
                self.screenshot(time_viewer=time_viewer)
                for _ in self._iter_time(time_idx, callback)
            ]
        finally:
            self.set_time_interpolation(old_mode)
        if callback is not None:
            callback(frame=len(time_idx), n_frames=len(time_idx))
        return images

    def _iter_time(self, time_idx, callback):
        """Iterate through time points, then reset to current time.

        Parameters
        ----------
        time_idx : array_like
            Time point indexes through which to iterate.
        callback : callable | None
            Callback to call before yielding each frame.

        Yields
        ------
        idx : int | float
            Current index.

        Notes
        -----
        Used by movie and image sequence saving functions.
        """
        current_time_idx = self._data["time_idx"]
        for ii, idx in enumerate(time_idx):
            self.set_time_point(idx)
            if callback is not None:
                callback(frame=ii, n_frames=len(time_idx))
            yield idx

        # Restore original time index
        self.set_time_point(current_time_idx)

    def _check_stc(self, hemi, array, vertices):
        from ...source_estimate import (
            _BaseMixedSourceEstimate,
            _BaseSourceEstimate,
            _BaseSurfaceSourceEstimate,
            _BaseVolSourceEstimate,
        )

        if isinstance(array, _BaseSourceEstimate):
            stc = array
            stc_surf = stc_vol = None
            if isinstance(stc, _BaseSurfaceSourceEstimate):
                stc_surf = stc
            elif isinstance(stc, _BaseMixedSourceEstimate):
                stc_surf = stc.surface() if hemi != "vol" else None
                stc_vol = stc.volume() if hemi == "vol" else None
            elif isinstance(stc, _BaseVolSourceEstimate):
                stc_vol = stc if hemi == "vol" else None
            else:
                raise TypeError("stc not supported")

            if stc_surf is None and stc_vol is None:
                raise ValueError("No data to be added")
            if stc_surf is not None:
                array = getattr(stc_surf, hemi + "_data")
                vertices = stc_surf.vertices[0 if hemi == "lh" else 1]
            if stc_vol is not None:
                array = stc_vol.data
                vertices = np.concatenate(stc_vol.vertices)
        else:
            stc = None
        return stc, array, vertices

    def _check_hemi(self, hemi, extras=()):
        """Check for safe single-hemi input, returns str."""
        _validate_type(hemi, (None, str), "hemi")
        if hemi is None:
            if self._hemi not in ["lh", "rh"]:
                raise ValueError(
                    "hemi must not be None when both hemispheres are displayed"
                )
            hemi = self._hemi
        _check_option("hemi", hemi, ("lh", "rh") + tuple(extras))
        return hemi

    def _check_hemis(self, hemi):
        """Check for safe dual or single-hemi input, returns list."""
        if hemi is None:
            if self._hemi not in ["lh", "rh"]:
                hemi = ["lh", "rh"]
            else:
                hemi = [self._hemi]
        elif hemi not in ["lh", "rh"]:
            extra = " or None" if self._hemi in ["lh", "rh"] else ""
            raise ValueError('hemi must be either "lh" or "rh"' + extra)
        else:
            hemi = [hemi]
        return hemi

    def _to_borders(self, label, hemi, borders, restrict_idx=None):
        """Convert a label/parc to borders."""
        if not isinstance(borders, bool | int) or borders < 0:
            raise ValueError("borders must be a bool or positive integer")
        if borders:
            n_vertices = label.size
            edges = mesh_edges(self.geo[hemi].orig_faces)
            edges = edges.tocoo()
            border_edges = label[edges.row] != label[edges.col]
            show = np.zeros(n_vertices, dtype=np.int64)
            keep_idx = np.unique(edges.row[border_edges])
            if isinstance(borders, int):
                for _ in range(borders):
                    keep_idx = np.isin(self.geo[hemi].orig_faces.ravel(), keep_idx)
                    keep_idx.shape = self.geo[hemi].orig_faces.shape
                    keep_idx = self.geo[hemi].orig_faces[np.any(keep_idx, axis=1)]
                    keep_idx = np.unique(keep_idx)
                if restrict_idx is not None:
                    keep_idx = keep_idx[np.isin(keep_idx, restrict_idx)]
            show[keep_idx] = 1
            label *= show

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

    def __hash__(self):
        """Hash the object."""
        return self._hash


def _safe_interp1d(x, y, kind="linear", axis=-1, assume_sorted=False):
    """Work around interp1d not liking singleton dimensions."""
    if y.shape[axis] == 1:

        def func(x):
            return np.take(y, np.zeros(np.asarray(x).shape, int), axis=axis)

        return func
    else:
        return interp1d(x, y, kind, axis=axis, assume_sorted=assume_sorted)


def _update_limits(fmin, fmid, fmax, center, array):
    if center is None:
        if fmin is None:
            fmin = array.min() if array.size > 0 else 0
        if fmax is None:
            fmax = array.max() if array.size > 0 else 1
    else:
        if fmin is None:
            fmin = 0
        if fmax is None:
            fmax = np.abs(center - array).max() if array.size > 0 else 1
    if fmid is None:
        fmid = (fmin + fmax) / 2.0

    if fmin >= fmid:
        raise RuntimeError(f"min must be < mid, got {fmin:0.4g} >= {fmid:0.4g}")
    if fmid >= fmax:
        raise RuntimeError(f"mid must be < max, got {fmid:0.4g} >= {fmax:0.4g}")

    return fmin, fmid, fmax


def _update_monotonic(lims, fmin, fmid, fmax):
    if fmin is not None:
        lims["fmin"] = fmin
        if lims["fmax"] < fmin:
            logger.debug(f"    Bumping fmax = {lims['fmax']} to {fmin}")
            lims["fmax"] = fmin
        if lims["fmid"] < fmin:
            logger.debug(f"    Bumping fmid = {lims['fmid']} to {fmin}")
            lims["fmid"] = fmin
    assert lims["fmin"] <= lims["fmid"] <= lims["fmax"]
    if fmid is not None:
        lims["fmid"] = fmid
        if lims["fmin"] > fmid:
            logger.debug(f"    Bumping fmin = {lims['fmin']} to {fmid}")
            lims["fmin"] = fmid
        if lims["fmax"] < fmid:
            logger.debug(f"    Bumping fmax = {lims['fmax']} to {fmid}")
            lims["fmax"] = fmid
    assert lims["fmin"] <= lims["fmid"] <= lims["fmax"]
    if fmax is not None:
        lims["fmax"] = fmax
        if lims["fmin"] > fmax:
            logger.debug(f"    Bumping fmin = {lims['fmin']} to {fmax}")
            lims["fmin"] = fmax
        if lims["fmid"] > fmax:
            logger.debug(f"    Bumping fmid = {lims['fmid']} to {fmax}")
            lims["fmid"] = fmax
    assert lims["fmin"] <= lims["fmid"] <= lims["fmax"]


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


class _FakeIren:
    def EnterEvent(self):
        pass

    def MouseMoveEvent(self):
        pass

    def LeaveEvent(self):
        pass

    def SetEventInformation(self, *args, **kwargs):
        pass

    def CharEvent(self):
        pass

    def KeyPressEvent(self, *args, **kwargs):
        pass

    def KeyReleaseEvent(self, *args, **kwargs):
        pass
