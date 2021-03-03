# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Oleh Kozynets <ok7mailbox@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#          jona-sassenhagen <jona.sassenhagen@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#
# License: Simplified BSD

import contextlib
from functools import partial
import os
import os.path as op
import sys
import time
import traceback
import warnings

import numpy as np
from scipy import sparse
from collections import OrderedDict

from .colormap import calculate_lut
from .surface import Surface
from .view import views_dicts, _lh_views_dict
from .mplcanvas import MplCanvas
from .callback import (ShowView, IntSlider, TimeSlider, SmartSlider,
                       BumpColorbarPoints, UpdateColorbarScale)

from ..utils import _show_help, _get_color_list
from .._3d import _process_clim, _handle_time, _check_views

from ...externals.decorator import decorator
from ...defaults import _handle_default
from ...surface import mesh_edges
from ...source_space import SourceSpaces, vertex_to_mni, read_talxfm
from ...transforms import apply_trans
from ...utils import (_check_option, logger, verbose, fill_doc, _validate_type,
                      use_log_level, Bunch, _ReuseCycle, warn)


@decorator
def safe_event(fun, *args, **kwargs):
    """Protect against PyQt5 exiting on event-handling errors."""
    try:
        return fun(*args, **kwargs)
    except Exception:
        traceback.print_exc(file=sys.stderr)


class _Overlay(object):
    def __init__(self, scalars, colormap, rng, opacity, name):
        self._scalars = scalars
        self._colormap = colormap
        assert rng is not None
        self._rng = rng
        self._opacity = opacity
        self._name = name

    def to_colors(self):
        from .._3d import _get_cmap
        from matplotlib.colors import ListedColormap

        if isinstance(self._colormap, str):
            kind = self._colormap
            cmap = _get_cmap(self._colormap)
        else:
            cmap = ListedColormap(self._colormap / 255.)
            kind = str(type(self._colormap))
        logger.debug(
            f'Color mapping {repr(self._name)} with {kind} '
            f'colormap and range {self._rng}')

        rng = self._rng
        assert rng is not None
        scalars = _norm(self._scalars, rng)

        colors = cmap(scalars)
        if self._opacity is not None:
            colors[:, 3] *= self._opacity
        return colors


def _range(x):
    return np.max(x) - np.min(x)


def _norm(x, rng):
    if rng[0] == rng[1]:
        factor = factor = 1 if rng[0] == 0 else 1e-6 * rng[0]
    else:
        factor = rng[1] - rng[0]
    return (x - rng[0]) / factor


class _LayeredMesh(object):
    def __init__(self, renderer, vertices, triangles, normals):
        self._renderer = renderer
        self._vertices = vertices
        self._triangles = triangles
        self._normals = normals

        self._polydata = None
        self._actor = None
        self._is_mapped = False

        self._cache = None
        self._overlays = OrderedDict()

        self._default_scalars = np.ones(vertices.shape)
        self._default_scalars_name = 'Data'

    def map(self):
        kwargs = {
            "color": None,
            "pickable": True,
            "rgba": True,
        }
        mesh_data = self._renderer.mesh(
            x=self._vertices[:, 0],
            y=self._vertices[:, 1],
            z=self._vertices[:, 2],
            triangles=self._triangles,
            normals=self._normals,
            scalars=self._default_scalars,
            **kwargs
        )
        self._actor, self._polydata = mesh_data
        self._is_mapped = True

    def _compute_over(self, B, A):
        assert A.ndim == B.ndim == 2
        assert A.shape[1] == B.shape[1] == 4
        A_w = A[:, 3:]  # * 1
        B_w = B[:, 3:] * (1 - A_w)
        C = A.copy()
        C[:, :3] *= A_w
        C[:, :3] += B[:, :3] * B_w
        C[:, 3:] += B_w
        C[:, :3] /= C[:, 3:]
        return np.clip(C, 0, 1, out=C)

    def _compose_overlays(self):
        B = None
        for overlay in self._overlays.values():
            A = overlay.to_colors()
            if B is None:
                B = A
            else:
                B = self._compute_over(B, A)
        return B

    def add_overlay(self, scalars, colormap, rng, opacity, name):
        overlay = _Overlay(
            scalars=scalars,
            colormap=colormap,
            rng=rng,
            opacity=opacity,
            name=name,
        )
        self._overlays[name] = overlay
        colors = overlay.to_colors()

        # save colors in cache
        if self._cache is None:
            self._cache = colors
        else:
            self._cache = self._compute_over(self._cache, colors)

        # update the texture
        self._update()

    def remove_overlay(self, names):
        if not isinstance(names, list):
            names = [names]
        for name in names:
            if name in self._overlays:
                del self._overlays[name]
        self.update()

    def _update(self):
        if self._cache is None:
            return
        from ..backends._pyvista import _set_mesh_scalars
        _set_mesh_scalars(
            mesh=self._polydata,
            scalars=self._cache,
            name=self._default_scalars_name,
        )

    def update(self):
        self._cache = self._compose_overlays()
        self._update()

    def _clean(self):
        mapper = self._actor.GetMapper()
        mapper.SetLookupTable(None)
        self._actor.SetMapper(None)
        self._actor = None
        self._polydata = None
        self._renderer = None

    def update_overlay(self, name, scalars=None, colormap=None,
                       opacity=None, rng=None):
        overlay = self._overlays.get(name, None)
        if overlay is None:
            return
        if scalars is not None:
            overlay._scalars = scalars
        if colormap is not None:
            overlay._colormap = colormap
        if opacity is not None:
            overlay._opacity = opacity
        if rng is not None:
            overlay._rng = rng
        self.update()


@fill_doc
class Brain(object):
    """Class for visualizing a brain.

    .. warning::
       The API for this class is not currently complete. We suggest using
       :meth:`mne.viz.plot_source_estimates` with the PyVista backend
       enabled to obtain a ``Brain`` instance.

    Parameters
    ----------
    subject_id : str
        Subject name in Freesurfer subjects dir.
    hemi : str
        Hemisphere id (ie 'lh', 'rh', 'both', or 'split'). In the case
        of 'both', both hemispheres are shown in the same window.
        In the case of 'split' hemispheres are displayed side-by-side
        in different viewing panes.
    surf : str
        FreeSurfer surface mesh name (ie 'white', 'inflated', etc.).
    title : str
        Title for the window.
    cortex : str or None
        Specifies how the cortical surface is rendered.
        The name of one of the preset cortex styles can be:
        ``'classic'`` (default), ``'high_contrast'``,
        ``'low_contrast'``, or ``'bone'`` or a valid color name.
        Setting this to ``None`` is equivalent to ``(0.5, 0.5, 0.5)``.
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
    figure : list of Figure | None | int
        If None (default), a new window will be created with the appropriate
        views. For single view plots, the figure can be specified as int to
        retrieve the corresponding Mayavi window.
    subjects_dir : str | None
        If not None, this directory will be used as the subjects directory
        instead of the value set using the SUBJECTS_DIR environment
        variable.
    views : list | str
        The views to use.
    offset : bool
        If True, aligs origin with medial wall. Useful for viewing inflated
        surface where hemispheres typically overlap (Default: True).
    show_toolbar : bool
        If True, toolbars will be shown for each view.
    offscreen : bool
        If True, rendering will be done offscreen (not shown). Useful
        mostly for generating images or screenshots, but can be buggy.
        Use at your own risk.
    interaction : str
        Can be "trackball" (default) or "terrain", i.e. a turntable-style
        camera.
    units : str
        Can be 'm' or 'mm' (default).
    %(view_layout)s
    show : bool
        Display the window as soon as it is ready. Defaults to True.

    Attributes
    ----------
    geo : dict
        A dictionary of pysurfer.Surface objects for each hemisphere.
    overlays : dict
        The overlays.

    Notes
    -----
    This table shows the capabilities of each Brain backend ("✓" for full
    support, and "-" for partial support):

    .. table::
       :widths: auto

       +---------------------------+--------------+---------------+
       | 3D function:              | surfer.Brain | mne.viz.Brain |
       +===========================+==============+===============+
       | add_annotation            | ✓            | ✓             |
       +---------------------------+--------------+---------------+
       | add_data                  | ✓            | ✓             |
       +---------------------------+--------------+---------------+
       | add_foci                  | ✓            | ✓             |
       +---------------------------+--------------+---------------+
       | add_label                 | ✓            | ✓             |
       +---------------------------+--------------+---------------+
       | add_text                  | ✓            | ✓             |
       +---------------------------+--------------+---------------+
       | close                     | ✓            | ✓             |
       +---------------------------+--------------+---------------+
       | data                      | ✓            | ✓             |
       +---------------------------+--------------+---------------+
       | foci                      | ✓            |               |
       +---------------------------+--------------+---------------+
       | labels                    | ✓            | ✓             |
       +---------------------------+--------------+---------------+
       | remove_foci               | ✓            |               |
       +---------------------------+--------------+---------------+
       | remove_labels             | ✓            | ✓             |
       +---------------------------+--------------+---------------+
       | remove_annotations        | -            | ✓             |
       +---------------------------+--------------+---------------+
       | scale_data_colormap       | ✓            |               |
       +---------------------------+--------------+---------------+
       | save_image                | ✓            | ✓             |
       +---------------------------+--------------+---------------+
       | save_movie                | ✓            | ✓             |
       +---------------------------+--------------+---------------+
       | screenshot                | ✓            | ✓             |
       +---------------------------+--------------+---------------+
       | show_view                 | ✓            | ✓             |
       +---------------------------+--------------+---------------+
       | TimeViewer                | ✓            | ✓             |
       +---------------------------+--------------+---------------+
       | enable_depth_peeling      |              | ✓             |
       +---------------------------+--------------+---------------+
       | get_picked_points         |              | ✓             |
       +---------------------------+--------------+---------------+
       | add_data(volume)          |              | ✓             |
       +---------------------------+--------------+---------------+
       | view_layout               |              | ✓             |
       +---------------------------+--------------+---------------+
       | flatmaps                  |              | ✓             |
       +---------------------------+--------------+---------------+
       | vertex picking            |              | ✓             |
       +---------------------------+--------------+---------------+
       | label picking             |              | ✓             |
       +---------------------------+--------------+---------------+
    """

    def __init__(self, subject_id, hemi, surf, title=None,
                 cortex="classic", alpha=1.0, size=800, background="black",
                 foreground=None, figure=None, subjects_dir=None,
                 views='auto', offset=True, show_toolbar=False,
                 offscreen=False, interaction='trackball', units='mm',
                 view_layout='vertical', show=True):
        from ..backends.renderer import backend, _get_renderer, _get_3d_backend
        from .._3d import _get_cmap
        from matplotlib.colors import colorConverter

        if hemi in ('both', 'split'):
            self._hemis = ('lh', 'rh')
        elif hemi in ('lh', 'rh'):
            self._hemis = (hemi, )
        else:
            raise KeyError('hemi has to be either "lh", "rh", "split", '
                           'or "both"')
        self._view_layout = _check_option('view_layout', view_layout,
                                          ('vertical', 'horizontal'))

        if figure is not None and not isinstance(figure, int):
            backend._check_3d_figure(figure)
        if title is None:
            self._title = subject_id
        else:
            self._title = title
        self._interaction = 'trackball'

        if isinstance(background, str):
            background = colorConverter.to_rgb(background)
        self._bg_color = background
        if foreground is None:
            foreground = 'w' if sum(self._bg_color) < 2 else 'k'
        if isinstance(foreground, str):
            foreground = colorConverter.to_rgb(foreground)
        self._fg_color = foreground

        if isinstance(views, str):
            views = [views]
        views = _check_views(surf, views, hemi)
        col_dict = dict(lh=1, rh=1, both=1, split=2)
        shape = (len(views), col_dict[hemi])
        if self._view_layout == 'horizontal':
            shape = shape[::-1]
        self._subplot_shape = shape

        size = tuple(np.atleast_1d(size).round(0).astype(int).flat)
        if len(size) not in (1, 2):
            raise ValueError('"size" parameter must be an int or length-2 '
                             'sequence of ints.')
        self._size = size if len(size) == 2 else size * 2  # 1-tuple to 2-tuple

        self.time_viewer = False
        self.notebook = (_get_3d_backend() == "notebook")
        self._hemi = hemi
        self._units = units
        self._alpha = float(alpha)
        self._subject_id = subject_id
        self._subjects_dir = subjects_dir
        self._views = views
        self._times = None
        self._vertex_to_label_id = dict()
        self._annotation_labels = dict()
        self._labels = {'lh': list(), 'rh': list()}
        self._annots = {'lh': list(), 'rh': list()}
        self._layered_meshes = {}
        # for now only one color bar can be added
        # since it is the same for all figures
        self._colorbar_added = False
        # for now only one time label can be added
        # since it is the same for all figures
        self._time_label_added = False
        # array of data used by TimeViewer
        self._data = {}
        self.geo = {}
        self.set_time_interpolation('nearest')

        geo_kwargs = self._cortex_colormap(cortex)
        # evaluate at the midpoint of the used colormap
        val = -geo_kwargs['vmin'] / (geo_kwargs['vmax'] - geo_kwargs['vmin'])
        self._brain_color = _get_cmap(geo_kwargs['colormap'])(val)

        # load geometry for one or both hemispheres as necessary
        offset = None if (not offset or hemi != 'both') else 0.0

        self._renderer = _get_renderer(name=self._title, size=self._size,
                                       bgcolor=background,
                                       shape=shape,
                                       fig=figure)

        if _get_3d_backend() == "pyvista":
            self.plotter = self._renderer.plotter
            self.window = self.plotter.app_window
            self.window.signal_close.connect(self._clean)

        for h in self._hemis:
            # Initialize a Surface object as the geometry
            geo = Surface(subject_id, h, surf, subjects_dir, offset,
                          units=self._units)
            # Load in the geometry and curvature
            geo.load_geometry()
            geo.load_curvature()
            self.geo[h] = geo
            for ri, ci, v in self._iter_views(h):
                self._renderer.subplot(ri, ci)
                if self._layered_meshes.get(h) is None:
                    mesh = _LayeredMesh(
                        renderer=self._renderer,
                        vertices=self.geo[h].coords,
                        triangles=self.geo[h].faces,
                        normals=self.geo[h].nn,
                    )
                    mesh.map()  # send to GPU
                    mesh.add_overlay(
                        scalars=self.geo[h].bin_curv,
                        colormap=geo_kwargs["colormap"],
                        rng=[geo_kwargs["vmin"], geo_kwargs["vmax"]],
                        opacity=alpha,
                        name='curv',
                    )
                    self._layered_meshes[h] = mesh
                    # add metadata to the mesh for picking
                    mesh._polydata._hemi = h
                else:
                    actor = self._layered_meshes[h]._actor
                    self._renderer.plotter.add_actor(actor)
                self._renderer.set_camera(**views_dicts[h][v])

        self.interaction = interaction
        self._closed = False
        if show:
            self.show()
        # update the views once the geometry is all set
        for h in self._hemis:
            for ri, ci, v in self._iter_views(h):
                self.show_view(v, row=ri, col=ci, hemi=h)

        if surf == 'flat':
            self._renderer.set_interaction("rubber_band_2d")

        if hemi == 'rh' and hasattr(self._renderer, "_orient_lights"):
            self._renderer._orient_lights()

    def setup_time_viewer(self, time_viewer=True, show_traces=True):
        """Configure the time viewer parameters.

        Parameters
        ----------
        time_viewer : bool
            If True, enable widgets interaction. Defaults to True.

        show_traces : bool
            If True, enable visualization of time traces. Defaults to True.
        """
        if self.time_viewer:
            return
        if not self._data:
            raise ValueError("No data to visualize. See ``add_data``.")
        self.time_viewer = time_viewer
        self.orientation = list(_lh_views_dict.keys())
        self.default_smoothing_range = [0, 15]

        # setup notebook
        if self.notebook:
            self._configure_notebook()
            return

        # Default configuration
        self.playback = False
        self.visibility = False
        self.refresh_rate_ms = max(int(round(1000. / 60.)), 1)
        self.default_scaling_range = [0.2, 2.0]
        self.default_playback_speed_range = [0.01, 1]
        self.default_playback_speed_value = 0.05
        self.default_status_bar_msg = "Press ? for help"
        self.default_label_extract_modes = {
            "stc": ["mean", "max"],
            "src": ["mean_flip", "pca_flip", "auto"],
        }
        self.default_trace_modes = ('vertex', 'label')
        self.annot = None
        self.label_extract_mode = None
        all_keys = ('lh', 'rh', 'vol')
        self.act_data_smooth = {key: (None, None) for key in all_keys}
        self.color_list = _get_color_list()
        # remove grey for better contrast on the brain
        self.color_list.remove("#7f7f7f")
        self.color_cycle = _ReuseCycle(self.color_list)
        self.mpl_canvas = None
        self.gfp = None
        self.picked_patches = {key: list() for key in all_keys}
        self.picked_points = {key: list() for key in all_keys}
        self.pick_table = dict()
        self._spheres = list()
        self._mouse_no_mvt = -1
        self.icons = dict()
        self.actions = dict()
        self.callbacks = dict()
        self.sliders = dict()
        self.keys = ('fmin', 'fmid', 'fmax')
        self.slider_length = 0.02
        self.slider_width = 0.04
        self.slider_color = (0.43137255, 0.44313725, 0.45882353)
        self.slider_tube_width = 0.04
        self.slider_tube_color = (0.69803922, 0.70196078, 0.70980392)
        self._trace_mode_widget = None
        self._annot_cands_widget = None
        self._label_mode_widget = None

        # Direct access parameters:
        self._iren = self._renderer.plotter.iren
        self.main_menu = self.plotter.main_menu
        self.tool_bar = self.window.addToolBar("toolbar")
        self.status_bar = self.window.statusBar()
        self.interactor = self.plotter.interactor

        # Derived parameters:
        self.playback_speed = self.default_playback_speed_value
        _validate_type(show_traces, (bool, str, 'numeric'), 'show_traces')
        self.interactor_fraction = 0.25
        if isinstance(show_traces, str):
            self.show_traces = True
            self.separate_canvas = False
            self.traces_mode = 'vertex'
            if show_traces == 'separate':
                self.separate_canvas = True
            elif show_traces == 'label':
                self.traces_mode = 'label'
            else:
                assert show_traces == 'vertex'  # guaranteed above
        else:
            if isinstance(show_traces, bool):
                self.show_traces = show_traces
            else:
                show_traces = float(show_traces)
                if not 0 < show_traces < 1:
                    raise ValueError(
                        'show traces, if numeric, must be between 0 and 1, '
                        f'got {show_traces}')
                self.show_traces = True
                self.interactor_fraction = show_traces
            self.traces_mode = 'vertex'
            self.separate_canvas = False
        del show_traces

        self._load_icons()
        self._configure_time_label()
        self._configure_sliders()
        self._configure_scalar_bar()
        self._configure_playback()
        self._configure_menu()
        self._configure_tool_bar()
        self._configure_status_bar()
        self._configure_picking()
        self._configure_trace_mode()

        # show everything at the end
        self.toggle_interface()
        with self.ensure_minimum_sizes():
            self.show()

    @safe_event
    def _clean(self):
        # resolve the reference cycle
        self.clear_glyphs()
        self.remove_annotations()
        # clear init actors
        for hemi in self._hemis:
            self._layered_meshes[hemi]._clean()
        self._clear_callbacks()
        if getattr(self, 'mpl_canvas', None) is not None:
            self.mpl_canvas.clear()
        if getattr(self, 'act_data_smooth', None) is not None:
            for key in list(self.act_data_smooth.keys()):
                self.act_data_smooth[key] = None
        # XXX this should be done in PyVista
        for renderer in self.plotter.renderers:
            renderer.RemoveAllLights()
        # app_window cannot be set to None because it is used in __del__
        for key in ('lighting', 'interactor', '_RenderWindow'):
            setattr(self.plotter, key, None)
        # Qt LeaveEvent requires _Iren so we use _FakeIren instead of None
        # to resolve the ref to vtkGenericRenderWindowInteractor
        self.plotter._Iren = _FakeIren()
        if getattr(self.plotter, 'scalar_bar', None) is not None:
            self.plotter.scalar_bar = None
        if getattr(self.plotter, 'picker', None) is not None:
            self.plotter.picker = None
        # XXX end PyVista
        for key in ('reps', 'plotter', 'main_menu', 'window', 'tool_bar',
                    'status_bar', 'interactor', 'mpl_canvas', 'time_actor',
                    'picked_renderer', 'act_data_smooth', '_iren',
                    'actions', 'sliders', 'geo', '_hemi_actors', '_data'):
            setattr(self, key, None)

    @contextlib.contextmanager
    def ensure_minimum_sizes(self):
        """Ensure that widgets respect the windows size."""
        from ..backends._pyvista import _process_events
        sz = self._size
        adjust_mpl = self.show_traces and not self.separate_canvas
        if not adjust_mpl:
            yield
        else:
            mpl_h = int(round((sz[1] * self.interactor_fraction) /
                              (1 - self.interactor_fraction)))
            self.mpl_canvas.canvas.setMinimumSize(sz[0], mpl_h)
            try:
                yield
            finally:
                self.splitter.setSizes([sz[1], mpl_h])
                _process_events(self.plotter)
                _process_events(self.plotter)
                self.mpl_canvas.canvas.setMinimumSize(0, 0)
            _process_events(self.plotter)
            _process_events(self.plotter)
            # sizes could change, update views
            for hemi in ('lh', 'rh'):
                for ri, ci, v in self._iter_views(hemi):
                    self.show_view(view=v, row=ri, col=ci)
            _process_events(self.plotter)

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

        # update tool bar icon
        if self.visibility:
            self.actions["visibility"].setIcon(self.icons["visibility_on"])
        else:
            self.actions["visibility"].setIcon(self.icons["visibility_off"])

        # manage sliders
        for slider in self.plotter.slider_widgets:
            slider_rep = slider.GetRepresentation()
            if self.visibility:
                slider_rep.VisibilityOn()
            else:
                slider_rep.VisibilityOff()

        # manage time label
        time_label = self._data['time_label']
        # if we actually have time points, we will show the slider so
        # hide the time actor
        have_ts = self._times is not None and len(self._times) > 1
        if self.time_actor is not None:
            if self.visibility and time_label is not None and not have_ts:
                self.time_actor.SetInput(time_label(self._current_time))
                self.time_actor.VisibilityOn()
            else:
                self.time_actor.VisibilityOff()

        self._update()

    def apply_auto_scaling(self):
        """Detect automatically fitting scaling parameters."""
        self._update_auto_scaling()
        for key in ('fmin', 'fmid', 'fmax'):
            self.reps[key].SetValue(self._data[key])
        self._update()

    def restore_user_scaling(self):
        """Restore original scaling parameters."""
        self._update_auto_scaling(restore=True)
        for key in ('fmin', 'fmid', 'fmax'):
            self.reps[key].SetValue(self._data[key])
        self._update()

    def toggle_playback(self, value=None):
        """Toggle time playback.

        Parameters
        ----------
        value : bool | None
            If True, automatic time playback is enabled and if False,
            it's disabled. If None, the state of time playback is toggled.
            Defaults to None.
        """
        if value is None:
            self.playback = not self.playback
        else:
            self.playback = value

        # update tool bar icon
        if self.playback:
            self.actions["play"].setIcon(self.icons["pause"])
        else:
            self.actions["play"].setIcon(self.icons["play"])

        if self.playback:
            time_data = self._data['time']
            max_time = np.max(time_data)
            if self._current_time == max_time:  # start over
                self.set_time_point(0)  # first index
            self._last_tick = time.time()

    def reset(self):
        """Reset view and time step."""
        self.reset_view()
        max_time = len(self._data['time']) - 1
        if max_time > 0:
            self.callbacks["time"](
                self._data["initial_time_idx"],
                update_widget=True,
            )
        self._update()

    def set_playback_speed(self, speed):
        """Set the time playback speed.

        Parameters
        ----------
        speed : float
            The speed of the playback.
        """
        self.playback_speed = speed

    @safe_event
    def _play(self):
        if self.playback:
            try:
                self._advance()
            except Exception:
                self.toggle_playback(value=False)
                raise

    def _advance(self):
        this_time = time.time()
        delta = this_time - self._last_tick
        self._last_tick = time.time()
        time_data = self._data['time']
        times = np.arange(self._n_times)
        time_shift = delta * self.playback_speed
        max_time = np.max(time_data)
        time_point = min(self._current_time + time_shift, max_time)
        # always use linear here -- this does not determine the data
        # interpolation mode, it just finds where we are (in time) in
        # terms of the time indices
        idx = np.interp(time_point, time_data, times)
        self.callbacks["time"](idx, update_widget=True)
        if time_point == max_time:
            self.toggle_playback(value=False)

    def _set_slider_style(self):
        for slider in self.sliders.values():
            if slider is not None:
                slider_rep = slider.GetRepresentation()
                slider_rep.SetSliderLength(self.slider_length)
                slider_rep.SetSliderWidth(self.slider_width)
                slider_rep.SetTubeWidth(self.slider_tube_width)
                slider_rep.GetSliderProperty().SetColor(self.slider_color)
                slider_rep.GetTubeProperty().SetColor(self.slider_tube_color)
                slider_rep.GetLabelProperty().SetShadow(False)
                slider_rep.GetLabelProperty().SetBold(True)
                slider_rep.GetLabelProperty().SetColor(self._fg_color)
                slider_rep.GetTitleProperty().ShallowCopy(
                    slider_rep.GetLabelProperty()
                )
                slider_rep.GetCapProperty().SetOpacity(0)

    def _configure_notebook(self):
        from ._notebook import _NotebookInteractor
        self._renderer.figure.display = _NotebookInteractor(self)

    def _configure_time_label(self):
        self.time_actor = self._data.get('time_actor')
        if self.time_actor is not None:
            self.time_actor.SetPosition(0.5, 0.03)
            self.time_actor.GetTextProperty().SetJustificationToCentered()
            self.time_actor.GetTextProperty().BoldOn()
            self.time_actor.VisibilityOff()

    def _configure_scalar_bar(self):
        if self._colorbar_added:
            scalar_bar = self.plotter.scalar_bar
            scalar_bar.SetOrientationToVertical()
            scalar_bar.SetHeight(0.6)
            scalar_bar.SetWidth(0.05)
            scalar_bar.SetPosition(0.02, 0.2)

    def _configure_sliders(self):
        # Orientation slider
        # Use 'lh' as a reference for orientation for 'both'
        if self._hemi == 'both':
            hemis_ref = ['lh']
        else:
            hemis_ref = self._hemis
        for hemi in hemis_ref:
            for ri, ci, view in self._iter_views(hemi):
                orientation_name = f"orientation_{hemi}_{ri}_{ci}"
                self.plotter.subplot(ri, ci)
                if view == 'flat':
                    self.callbacks[orientation_name] = None
                    continue
                self.callbacks[orientation_name] = ShowView(
                    plotter=self.plotter,
                    brain=self,
                    orientation=self.orientation,
                    hemi=hemi,
                    row=ri,
                    col=ci,
                )
                self.sliders[orientation_name] = \
                    self.plotter.add_text_slider_widget(
                    self.callbacks[orientation_name],
                    value=0,
                    data=self.orientation,
                    pointa=(0.82, 0.74),
                    pointb=(0.98, 0.74),
                    event_type='always'
                )
                orientation_rep = \
                    self.sliders[orientation_name].GetRepresentation()
                orientation_rep.ShowSliderLabelOff()
                self.callbacks[orientation_name].slider_rep = orientation_rep
                self.callbacks[orientation_name](view, update_widget=True)

        # Put other sliders on the bottom right view
        ri, ci = np.array(self._subplot_shape) - 1
        self.plotter.subplot(ri, ci)

        # Smoothing slider
        self.callbacks["smoothing"] = IntSlider(
            plotter=self.plotter,
            callback=self.set_data_smoothing,
            first_call=False,
        )
        self.sliders["smoothing"] = self.plotter.add_slider_widget(
            self.callbacks["smoothing"],
            value=self._data['smoothing_steps'],
            rng=self.default_smoothing_range, title="smoothing",
            pointa=(0.82, 0.90),
            pointb=(0.98, 0.90)
        )
        self.callbacks["smoothing"].slider_rep = \
            self.sliders["smoothing"].GetRepresentation()

        # Time slider
        max_time = len(self._data['time']) - 1
        # VTK on macOS bombs if we create these then hide them, so don't
        # even create them
        if max_time < 1:
            self.callbacks["time"] = None
            self.sliders["time"] = None
        else:
            self.callbacks["time"] = TimeSlider(
                plotter=self.plotter,
                brain=self,
                first_call=False,
                callback=self.plot_time_line,
            )
            self.sliders["time"] = self.plotter.add_slider_widget(
                self.callbacks["time"],
                value=self._data['time_idx'],
                rng=[0, max_time],
                pointa=(0.23, 0.1),
                pointb=(0.77, 0.1),
                event_type='always'
            )
            self.callbacks["time"].slider_rep = \
                self.sliders["time"].GetRepresentation()
            # configure properties of the time slider
            self.sliders["time"].GetRepresentation().SetLabelFormat(
                'idx=%0.1f')

        current_time = self._current_time
        assert current_time is not None  # should never be the case, float
        time_label = self._data['time_label']
        if callable(time_label):
            current_time = time_label(current_time)
        else:
            current_time = time_label
        if self.sliders["time"] is not None:
            self.sliders["time"].GetRepresentation().SetTitleText(current_time)
        if self.time_actor is not None:
            self.time_actor.SetInput(current_time)
        del current_time

        # Playback speed slider
        if self.sliders["time"] is None:
            self.callbacks["playback_speed"] = None
            self.sliders["playback_speed"] = None
        else:
            self.callbacks["playback_speed"] = SmartSlider(
                plotter=self.plotter,
                callback=self.set_playback_speed,
            )
            self.sliders["playback_speed"] = self.plotter.add_slider_widget(
                self.callbacks["playback_speed"],
                value=self.default_playback_speed_value,
                rng=self.default_playback_speed_range, title="speed",
                pointa=(0.02, 0.1),
                pointb=(0.18, 0.1),
                event_type='always'
            )
            self.callbacks["playback_speed"].slider_rep = \
                self.sliders["playback_speed"].GetRepresentation()

        # Colormap slider
        pointa = np.array((0.82, 0.26))
        pointb = np.array((0.98, 0.26))
        shift = np.array([0, 0.1])

        for idx, key in enumerate(self.keys):
            title = "clim" if not idx else ""
            rng = _get_range(self)
            self.callbacks[key] = BumpColorbarPoints(
                plotter=self.plotter,
                brain=self,
                name=key
            )
            self.sliders[key] = self.plotter.add_slider_widget(
                self.callbacks[key],
                value=self._data[key],
                rng=rng, title=title,
                pointa=pointa + idx * shift,
                pointb=pointb + idx * shift,
                event_type="always",
            )

        # fscale
        self.callbacks["fscale"] = UpdateColorbarScale(
            plotter=self.plotter,
            brain=self,
        )
        self.sliders["fscale"] = self.plotter.add_slider_widget(
            self.callbacks["fscale"],
            value=1.0,
            rng=self.default_scaling_range, title="fscale",
            pointa=(0.82, 0.10),
            pointb=(0.98, 0.10)
        )
        self.callbacks["fscale"].slider_rep = \
            self.sliders["fscale"].GetRepresentation()

        # register colorbar slider representations
        self.reps = \
            {key: self.sliders[key].GetRepresentation() for key in self.keys}
        for name in ("fmin", "fmid", "fmax", "fscale"):
            self.callbacks[name].reps = self.reps

        # set the slider style
        self._set_slider_style()

    def _configure_playback(self):
        self.plotter.add_callback(self._play, self.refresh_rate_ms)

    def _configure_mplcanvas(self):
        win = self.plotter.app_window
        dpi = win.windowHandle().screen().logicalDotsPerInch()
        ratio = (1 - self.interactor_fraction) / self.interactor_fraction
        w = self.interactor.geometry().width()
        h = self.interactor.geometry().height() / ratio
        # Get the fractional components for the brain and mpl
        self.mpl_canvas = MplCanvas(self, w / dpi, h / dpi, dpi)
        xlim = [np.min(self._data['time']),
                np.max(self._data['time'])]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.mpl_canvas.axes.set(xlim=xlim)
        if not self.separate_canvas:
            from PyQt5.QtWidgets import QSplitter
            from PyQt5.QtCore import Qt
            canvas = self.mpl_canvas.canvas
            vlayout = self.plotter.frame.layout()
            vlayout.removeWidget(self.interactor)
            self.splitter = splitter = QSplitter(
                orientation=Qt.Vertical, parent=self.plotter.frame)
            vlayout.addWidget(splitter)
            splitter.addWidget(self.interactor)
            splitter.addWidget(canvas)
        self.mpl_canvas.set_color(
            bg_color=self._bg_color,
            fg_color=self._fg_color,
        )
        self.mpl_canvas.show()

    def _configure_vertex_time_course(self):
        if not self.show_traces:
            return
        if self.mpl_canvas is None:
            self._configure_mplcanvas()
        else:
            self.clear_glyphs()

        # plot the GFP
        y = np.concatenate(list(v[0] for v in self.act_data_smooth.values()
                                if v[0] is not None))
        y = np.linalg.norm(y, axis=0) / np.sqrt(len(y))
        self.gfp, = self.mpl_canvas.axes.plot(
            self._data['time'], y,
            lw=3, label='GFP', zorder=3, color=self._fg_color,
            alpha=0.5, ls=':')

        # now plot the time line
        self.plot_time_line()

        # then the picked points
        for idx, hemi in enumerate(['lh', 'rh', 'vol']):
            act_data = self.act_data_smooth.get(hemi, [None])[0]
            if act_data is None:
                continue
            hemi_data = self._data[hemi]
            vertices = hemi_data['vertices']

            # simulate a picked renderer
            if self._hemi in ('both', 'rh') or hemi == 'vol':
                idx = 0
            self.picked_renderer = self.plotter.renderers[idx]

            # initialize the default point
            if self._data['initial_time'] is not None:
                # pick at that time
                use_data = act_data[
                    :, [np.round(self._data['time_idx']).astype(int)]]
            else:
                use_data = act_data
            ind = np.unravel_index(np.argmax(np.abs(use_data), axis=None),
                                   use_data.shape)
            if hemi == 'vol':
                mesh = hemi_data['grid']
            else:
                mesh = self._layered_meshes[hemi]._polydata
            vertex_id = vertices[ind[0]]
            self._add_vertex_glyph(hemi, mesh, vertex_id)

    def _configure_picking(self):
        from ..backends._pyvista import _update_picking_callback

        # get data for each hemi
        for idx, hemi in enumerate(['vol', 'lh', 'rh']):
            hemi_data = self._data.get(hemi)
            if hemi_data is not None:
                act_data = hemi_data['array']
                if act_data.ndim == 3:
                    act_data = np.linalg.norm(act_data, axis=1)
                smooth_mat = hemi_data.get('smooth_mat')
                vertices = hemi_data['vertices']
                if hemi == 'vol':
                    assert smooth_mat is None
                    smooth_mat = sparse.csr_matrix(
                        (np.ones(len(vertices)),
                         (vertices, np.arange(len(vertices)))))
                self.act_data_smooth[hemi] = (act_data, smooth_mat)

        _update_picking_callback(
            self.plotter,
            self._on_mouse_move,
            self._on_button_press,
            self._on_button_release,
            self._on_pick
        )

    def _configure_trace_mode(self):
        from ...source_estimate import _get_allowed_label_modes
        from ...label import _read_annot_cands
        from PyQt5.QtWidgets import QComboBox, QLabel
        if not self.show_traces:
            return

        # do not show trace mode for volumes
        if (self._data.get('src', None) is not None and
                self._data['src'].kind == 'volume'):
            self._configure_vertex_time_course()
            return

        # setup candidate annots
        def _set_annot(annot):
            self.clear_glyphs()
            self.remove_labels()
            self.remove_annotations()
            self.annot = annot

            if annot == 'None':
                self.traces_mode = 'vertex'
                self._configure_vertex_time_course()
            else:
                self.traces_mode = 'label'
                self._configure_label_time_course()
            self._update()

        dir_name = op.join(self._subjects_dir, self._subject_id, 'label')
        cands = _read_annot_cands(dir_name)
        self.tool_bar.addSeparator()
        self.tool_bar.addWidget(QLabel("Annotation"))
        self._annot_cands_widget = QComboBox()
        self.tool_bar.addWidget(self._annot_cands_widget)
        self._annot_cands_widget.addItem('None')
        for cand in cands:
            self._annot_cands_widget.addItem(cand)
        self.annot = cands[0]

        # setup label extraction parameters
        def _set_label_mode(mode):
            if self.traces_mode != 'label':
                return
            import copy
            glyphs = copy.deepcopy(self.picked_patches)
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
            self._update()

        self.tool_bar.addSeparator()
        self.tool_bar.addWidget(QLabel("Label extraction mode"))
        self._label_mode_widget = QComboBox()
        self.tool_bar.addWidget(self._label_mode_widget)
        stc = self._data["stc"]
        modes = _get_allowed_label_modes(stc)
        if self._data["src"] is None:
            modes = [m for m in modes if m not in
                     self.default_label_extract_modes["src"]]
        for mode in modes:
            self._label_mode_widget.addItem(mode)
            self.label_extract_mode = mode

        if self.traces_mode == 'vertex':
            _set_annot('None')
        else:
            _set_annot(self.annot)
        self._annot_cands_widget.setCurrentText(self.annot)
        self._label_mode_widget.setCurrentText(self.label_extract_mode)
        self._annot_cands_widget.currentTextChanged.connect(_set_annot)
        self._label_mode_widget.currentTextChanged.connect(_set_label_mode)

    def _load_icons(self):
        from PyQt5.QtGui import QIcon
        from ..backends._pyvista import _init_resources
        _init_resources()
        self.icons["help"] = QIcon(":/help.svg")
        self.icons["play"] = QIcon(":/play.svg")
        self.icons["pause"] = QIcon(":/pause.svg")
        self.icons["reset"] = QIcon(":/reset.svg")
        self.icons["scale"] = QIcon(":/scale.svg")
        self.icons["clear"] = QIcon(":/clear.svg")
        self.icons["movie"] = QIcon(":/movie.svg")
        self.icons["restore"] = QIcon(":/restore.svg")
        self.icons["screenshot"] = QIcon(":/screenshot.svg")
        self.icons["visibility_on"] = QIcon(":/visibility_on.svg")
        self.icons["visibility_off"] = QIcon(":/visibility_off.svg")

    def _save_movie_noname(self):
        return self.save_movie(None)

    def _configure_tool_bar(self):
        self.actions["screenshot"] = self.tool_bar.addAction(
            self.icons["screenshot"],
            "Take a screenshot",
            self.plotter._qt_screenshot
        )
        self.actions["movie"] = self.tool_bar.addAction(
            self.icons["movie"],
            "Save movie...",
            self._save_movie_noname,
        )
        self.actions["visibility"] = self.tool_bar.addAction(
            self.icons["visibility_on"],
            "Toggle Visibility",
            self.toggle_interface
        )
        self.actions["play"] = self.tool_bar.addAction(
            self.icons["play"],
            "Play/Pause",
            self.toggle_playback
        )
        self.actions["reset"] = self.tool_bar.addAction(
            self.icons["reset"],
            "Reset",
            self.reset
        )
        self.actions["scale"] = self.tool_bar.addAction(
            self.icons["scale"],
            "Auto-Scale",
            self.apply_auto_scaling
        )
        self.actions["restore"] = self.tool_bar.addAction(
            self.icons["restore"],
            "Restore scaling",
            self.restore_user_scaling
        )
        self.actions["clear"] = self.tool_bar.addAction(
            self.icons["clear"],
            "Clear traces",
            self.clear_glyphs
        )
        self.actions["help"] = self.tool_bar.addAction(
            self.icons["help"],
            "Help",
            self.help
        )

        self.actions["movie"].setShortcut("ctrl+shift+s")
        self.actions["visibility"].setShortcut("i")
        self.actions["play"].setShortcut(" ")
        self.actions["scale"].setShortcut("s")
        self.actions["restore"].setShortcut("r")
        self.actions["clear"].setShortcut("c")
        self.actions["help"].setShortcut("?")

    def _configure_menu(self):
        # remove default picking menu
        to_remove = list()
        for action in self.main_menu.actions():
            if action.text() == "Tools":
                to_remove.append(action)
        for action in to_remove:
            self.main_menu.removeAction(action)

        # add help menu
        menu = self.main_menu.addMenu('Help')
        menu.addAction('Show MNE key bindings\t?', self.help)

    def _configure_status_bar(self):
        from PyQt5.QtWidgets import QLabel, QProgressBar
        self.status_msg = QLabel(self.default_status_bar_msg)
        self.status_progress = QProgressBar()
        self.status_bar.layout().addWidget(self.status_msg, 1)
        self.status_bar.layout().addWidget(self.status_progress, 0)
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
            self.picked_renderer = self.plotter.iren.FindPokedRenderer(x, y)
            # trigger the pick
            self.plotter.picker.Pick(x, y, 0, self.picked_renderer)
        self._mouse_no_mvt = 0

    def _on_pick(self, vtk_picker, event):
        if not self.show_traces:
            return

        # vtk_picker is a vtkCellPicker
        cell_id = vtk_picker.GetCellId()
        mesh = vtk_picker.GetDataSet()

        if mesh is None or cell_id == -1 or not self._mouse_no_mvt:
            return  # don't pick

        # 1) Check to see if there are any spheres along the ray
        if len(self._spheres):
            collection = vtk_picker.GetProp3Ds()
            found_sphere = None
            for ii in range(collection.GetNumberOfItems()):
                actor = collection.GetItemAsObject(ii)
                for sphere in self._spheres:
                    if any(a is actor for a in sphere._actors):
                        found_sphere = sphere
                        break
                if found_sphere is not None:
                    break
            if found_sphere is not None:
                assert found_sphere._is_glyph
                mesh = found_sphere

        # 2) Remove sphere if it's what we have
        if hasattr(mesh, "_is_glyph"):
            self._remove_vertex_glyph(mesh)
            return

        # 3) Otherwise, pick the objects in the scene
        try:
            hemi = mesh._hemi
        except AttributeError:  # volume
            hemi = 'vol'
        else:
            assert hemi in ('lh', 'rh')
        if self.act_data_smooth[hemi][0] is None:  # no data to add for hemi
            return
        pos = np.array(vtk_picker.GetPickPosition())
        if hemi == 'vol':
            # VTK will give us the point closest to the viewer in the vol.
            # We want to pick the point with the maximum value along the
            # camera-to-click array, which fortunately we can get "just"
            # by inspecting the points that are sufficiently close to the
            # ray.
            grid = mesh = self._data[hemi]['grid']
            vertices = self._data[hemi]['vertices']
            coords = self._data[hemi]['grid_coords'][vertices]
            scalars = grid.cell_arrays['values'][vertices]
            spacing = np.array(grid.GetSpacing())
            max_dist = np.linalg.norm(spacing) / 2.
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
            # grid.cell_arrays['values'][vertices] = dists * mask
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
            cell = [vtk_cell.GetPointId(point_id) for point_id
                    in range(vtk_cell.GetNumberOfPoints())]
            vertices = mesh.points[cell]
            idx = np.argmin(abs(vertices - pos), axis=0)
            vertex_id = cell[idx[0]]

        if self.traces_mode == 'label':
            self._add_label_glyph(hemi, mesh, vertex_id)
        else:
            self._add_vertex_glyph(hemi, mesh, vertex_id)

    def _add_label_glyph(self, hemi, mesh, vertex_id):
        if hemi == 'vol':
            return
        label_id = self._vertex_to_label_id[hemi][vertex_id]
        label = self._annotation_labels[hemi][label_id]

        # remove the patch if already picked
        if label_id in self.picked_patches[hemi]:
            self._remove_label_glyph(hemi, label_id)
            return

        if hemi == label.hemi:
            self.add_label(label, borders=True, reset_camera=False)
            self.picked_patches[hemi].append(label_id)

    def _remove_label_glyph(self, hemi, label_id):
        label = self._annotation_labels[hemi][label_id]
        label._line.remove()
        self.color_cycle.restore(label._color)
        self.mpl_canvas.update_plot()
        self._layered_meshes[hemi].remove_overlay(label.name)
        self.picked_patches[hemi].remove(label_id)

    def _add_vertex_glyph(self, hemi, mesh, vertex_id):
        if vertex_id in self.picked_points[hemi]:
            return

        # skip if the wrong hemi is selected
        if self.act_data_smooth[hemi][0] is None:
            return
        from ..backends._pyvista import _sphere
        color = next(self.color_cycle)
        line = self.plot_time_course(hemi, vertex_id, color)
        if hemi == 'vol':
            ijk = np.unravel_index(
                vertex_id, np.array(mesh.GetDimensions()) - 1, order='F')
            # should just be GetCentroid(center), but apparently it's VTK9+:
            # center = np.empty(3)
            # voxel.GetCentroid(center)
            voxel = mesh.GetCell(*ijk)
            pts = voxel.GetPoints()
            n_pts = pts.GetNumberOfPoints()
            center = np.empty((n_pts, 3))
            for ii in range(pts.GetNumberOfPoints()):
                pts.GetPoint(ii, center[ii])
            center = np.mean(center, axis=0)
        else:
            center = mesh.GetPoints().GetPoint(vertex_id)
        del mesh

        # from the picked renderer to the subplot coords
        rindex = self.plotter.renderers.index(self.picked_renderer)
        row, col = self.plotter.index_to_loc(rindex)

        actors = list()
        spheres = list()
        for ri, ci, _ in self._iter_views(hemi):
            self.plotter.subplot(ri, ci)
            # Using _sphere() instead of renderer.sphere() for 2 reasons:
            # 1) renderer.sphere() fails on Windows in a scenario where a lot
            #    of picking requests are done in a short span of time (could be
            #    mitigated with synchronization/delay?)
            # 2) the glyph filter is used in renderer.sphere() but only one
            #    sphere is required in this function.
            actor, sphere = _sphere(
                plotter=self.plotter,
                center=np.array(center),
                color=color,
                radius=4.0,
            )
            actors.append(actor)
            spheres.append(sphere)

        # add metadata for picking
        for sphere in spheres:
            sphere._is_glyph = True
            sphere._hemi = hemi
            sphere._line = line
            sphere._actors = actors
            sphere._color = color
            sphere._vertex_id = vertex_id

        self.picked_points[hemi].append(vertex_id)
        self._spheres.extend(spheres)
        self.pick_table[vertex_id] = spheres
        return sphere

    def _remove_vertex_glyph(self, mesh, render=True):
        vertex_id = mesh._vertex_id
        if vertex_id not in self.pick_table:
            return

        hemi = mesh._hemi
        color = mesh._color
        spheres = self.pick_table[vertex_id]
        spheres[0]._line.remove()
        self.mpl_canvas.update_plot()
        self.picked_points[hemi].remove(vertex_id)

        with warnings.catch_warnings(record=True):
            # We intentionally ignore these in case we have traversed the
            # entire color cycle
            warnings.simplefilter('ignore')
            self.color_cycle.restore(color)
        for sphere in spheres:
            # remove all actors
            self.plotter.remove_actor(sphere._actors, render=render)
            sphere._actors = None
            self._spheres.pop(self._spheres.index(sphere))
        self.pick_table.pop(vertex_id)

    def clear_glyphs(self):
        """Clear the picking glyphs."""
        if not self.time_viewer:
            return
        for sphere in list(self._spheres):  # will remove itself, so copy
            self._remove_vertex_glyph(sphere, render=False)
        assert sum(len(v) for v in self.picked_points.values()) == 0
        assert len(self.pick_table) == 0
        assert len(self._spheres) == 0
        for hemi in self._hemis:
            for label_id in list(self.picked_patches[hemi]):
                self._remove_label_glyph(hemi, label_id)
        assert sum(len(v) for v in self.picked_patches.values()) == 0
        if self.gfp is not None:
            self.gfp.remove()
            self.gfp = None
        self._update()

    def plot_time_course(self, hemi, vertex_id, color):
        """Plot the vertex time course.

        Parameters
        ----------
        hemi : str
            The hemisphere id of the vertex.
        vertex_id : int
            The vertex identifier in the mesh.
        color : matplotlib color
            The color of the time course.

        Returns
        -------
        line : matplotlib object
            The time line object.
        """
        if self.mpl_canvas is None:
            return
        time = self._data['time'].copy()  # avoid circular ref
        if hemi == 'vol':
            hemi_str = 'V'
            xfm = read_talxfm(
                self._subject_id, self._subjects_dir)
            if self._units == 'mm':
                xfm['trans'][:3, 3] *= 1000.
            ijk = np.unravel_index(
                vertex_id, self._data[hemi]['grid_shape'], order='F')
            src_mri_t = self._data[hemi]['grid_src_mri_t']
            mni = apply_trans(np.dot(xfm['trans'], src_mri_t), ijk)
        else:
            hemi_str = 'L' if hemi == 'lh' else 'R'
            mni = vertex_to_mni(
                vertices=vertex_id,
                hemis=0 if hemi == 'lh' else 1,
                subject=self._subject_id,
                subjects_dir=self._subjects_dir
            )
        label = "{}:{} MNI: {}".format(
            hemi_str, str(vertex_id).ljust(6),
            ', '.join('%5.1f' % m for m in mni))
        act_data, smooth = self.act_data_smooth[hemi]
        if smooth is not None:
            act_data = smooth[vertex_id].dot(act_data)[0]
        else:
            act_data = act_data[vertex_id].copy()
        line = self.mpl_canvas.plot(
            time,
            act_data,
            label=label,
            lw=1.,
            color=color,
            zorder=4,
        )
        return line

    def plot_time_line(self):
        """Add the time line to the MPL widget."""
        if self.mpl_canvas is None:
            return
        if isinstance(self.show_traces, bool) and self.show_traces:
            # add time information
            current_time = self._current_time
            if not hasattr(self, "time_line"):
                self.time_line = self.mpl_canvas.plot_time_line(
                    x=current_time,
                    label='time',
                    color=self._fg_color,
                    lw=1,
                )
            self.time_line.set_xdata(current_time)
            self.mpl_canvas.update_plot()

    def help(self):
        """Display the help window."""
        pairs = [
            ('?', 'Display help window'),
            ('i', 'Toggle interface'),
            ('s', 'Apply auto-scaling'),
            ('r', 'Restore original clim'),
            ('c', 'Clear all traces'),
            ('Space', 'Start/Pause playback'),
        ]
        text1, text2 = zip(*pairs)
        text1 = '\n'.join(text1)
        text2 = '\n'.join(text2)
        _show_help(
            col1=text1,
            col2=text2,
            width=5,
            height=2,
        )

    def _clear_callbacks(self):
        from ..backends._pyvista import _remove_picking_callback
        if not hasattr(self, 'callbacks'):
            return
        for callback in self.callbacks.values():
            if callback is not None:
                if hasattr(callback, "plotter"):
                    callback.plotter = None
                if hasattr(callback, "brain"):
                    callback.brain = None
                if hasattr(callback, "slider_rep"):
                    callback.slider_rep = None
        self.callbacks.clear()
        if self.show_traces:
            _remove_picking_callback(self._iren, self.plotter.picker)

    @property
    def interaction(self):
        """The interaction style."""
        return self._interaction

    @interaction.setter
    def interaction(self, interaction):
        """Set the interaction style."""
        _validate_type(interaction, str, 'interaction')
        _check_option('interaction', interaction, ('trackball', 'terrain'))
        for ri, ci, _ in self._iter_views('vol'):  # will traverse all
            self._renderer.subplot(ri, ci)
            self._renderer.set_interaction(interaction)

    def _cortex_colormap(self, cortex):
        """Return the colormap corresponding to the cortex."""
        colormap_map = dict(classic=dict(colormap="Greys",
                                         vmin=-1, vmax=2),
                            high_contrast=dict(colormap="Greys",
                                               vmin=-.1, vmax=1.3),
                            low_contrast=dict(colormap="Greys",
                                              vmin=-5, vmax=5),
                            bone=dict(colormap="bone_r",
                                      vmin=-.2, vmax=2),
                            )
        return colormap_map[cortex]

    @verbose
    def add_data(self, array, fmin=None, fmid=None, fmax=None,
                 thresh=None, center=None, transparent=False, colormap="auto",
                 alpha=1, vertices=None, smoothing_steps=None, time=None,
                 time_label="auto", colorbar=True,
                 hemi=None, remove_existing=None, time_label_size=None,
                 initial_time=None, scale_factor=None, vector_alpha=None,
                 clim=None, src=None, volume_options=0.4, colorbar_kwargs=None,
                 verbose=None):
        """Display data from a numpy array on the surface or volume.

        This provides a similar interface to
        :meth:`surfer.Brain.add_overlay`, but it displays
        it with a single colormap. It offers more flexibility over the
        colormap, and provides a way to display four-dimensional data
        (i.e., a timecourse) or five-dimensional data (i.e., a
        vector-valued timecourse).

        .. note:: ``fmin`` sets the low end of the colormap, and is separate
                  from thresh (this is a different convention from
                  :meth:`surfer.Brain.add_overlay`).

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
            Options to pass to :meth:`pyvista.BasePlotter.add_scalar_bar`
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

        Due to a Mayavi (or VTK) alpha rendering bug, ``vector_alpha`` is
        clamped to be strictly < 1.
        """
        _validate_type(transparent, bool, 'transparent')
        _validate_type(vector_alpha, ('numeric', None), 'vector_alpha')
        _validate_type(scale_factor, ('numeric', None), 'scale_factor')

        # those parameters are not supported yet, only None is allowed
        _check_option('thresh', thresh, [None])
        _check_option('remove_existing', remove_existing, [None])
        _validate_type(time_label_size, (None, 'numeric'), 'time_label_size')
        if time_label_size is not None:
            time_label_size = float(time_label_size)
            if time_label_size < 0:
                raise ValueError('time_label_size must be positive, got '
                                 f'{time_label_size}')

        hemi = self._check_hemi(hemi, extras=['vol'])
        stc, array, vertices = self._check_stc(hemi, array, vertices)
        array = np.asarray(array)
        vector_alpha = alpha if vector_alpha is None else vector_alpha
        self._data['vector_alpha'] = vector_alpha
        self._data['scale_factor'] = scale_factor

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
                    raise ValueError('time has shape %s, but need shape %s '
                                     '(array.shape[-1])' %
                                     (time.shape, (array.shape[-1],)))
            self._data["time"] = time

            if self._n_times is None:
                self._times = time
            elif len(time) != self._n_times:
                raise ValueError("New n_times is different from previous "
                                 "n_times")
            elif not np.array_equal(time, self._times):
                raise ValueError("Not all time values are consistent with "
                                 "previously set times.")

            # initial time
            if initial_time is None:
                time_idx = 0
            else:
                time_idx = self._to_time_index(initial_time)

        # time label
        time_label, _ = _handle_time(time_label, 's', time)
        y_txt = 0.05 + 0.1 * bool(colorbar)

        if array.ndim == 3:
            if array.shape[1] != 3:
                raise ValueError('If array has 3 dimensions, array.shape[1] '
                                 'must equal 3, got %s' % (array.shape[1],))
        fmin, fmid, fmax = _update_limits(
            fmin, fmid, fmax, center, array
        )
        if colormap == 'auto':
            colormap = 'mne' if center is not None else 'hot'

        if smoothing_steps is None:
            smoothing_steps = 7
        elif smoothing_steps == 'nearest':
            smoothing_steps = 0
        elif isinstance(smoothing_steps, int):
            if smoothing_steps < 0:
                raise ValueError('Expected value of `smoothing_steps` is'
                                 ' positive but {} was given.'.format(
                                     smoothing_steps))
        else:
            raise TypeError('Expected type of `smoothing_steps` is int or'
                            ' NoneType but {} was given.'.format(
                                type(smoothing_steps)))

        self._data['stc'] = stc
        self._data['src'] = src
        self._data['smoothing_steps'] = smoothing_steps
        self._data['clim'] = clim
        self._data['time'] = time
        self._data['initial_time'] = initial_time
        self._data['time_label'] = time_label
        self._data['initial_time_idx'] = time_idx
        self._data['time_idx'] = time_idx
        self._data['transparent'] = transparent
        # data specific for a hemi
        self._data[hemi] = dict()
        self._data[hemi]['glyph_dataset'] = None
        self._data[hemi]['glyph_mapper'] = None
        self._data[hemi]['glyph_actor'] = None
        self._data[hemi]['array'] = array
        self._data[hemi]['vertices'] = vertices
        self._data['alpha'] = alpha
        self._data['colormap'] = colormap
        self._data['center'] = center
        self._data['fmin'] = fmin
        self._data['fmid'] = fmid
        self._data['fmax'] = fmax
        self.update_lut()

        # 1) add the surfaces first
        actor = None
        for ri, ci, _ in self._iter_views(hemi):
            self._renderer.subplot(ri, ci)
            if hemi in ('lh', 'rh'):
                actor = self._layered_meshes[hemi]._actor
            else:
                src_vol = src[2:] if src.kind == 'mixed' else src
                actor, _ = self._add_volume_data(hemi, src_vol, volume_options)
        assert actor is not None  # should have added one

        # 2) update time and smoothing properties
        # set_data_smoothing calls "set_time_point" for us, which will set
        # _current_time
        self.set_time_interpolation(self.time_interpolation)
        self.set_data_smoothing(self._data['smoothing_steps'])

        # 3) add the other actors
        if colorbar is True:
            # botto left by default
            colorbar = (self._subplot_shape[0] - 1, 0)
        for ri, ci, v in self._iter_views(hemi):
            self._renderer.subplot(ri, ci)
            # Add the time label to the bottommost view
            do = (ri, ci) == colorbar
            if not self._time_label_added and time_label is not None and do:
                time_actor = self._renderer.text2d(
                    x_window=0.95, y_window=y_txt,
                    color=self._fg_color,
                    size=time_label_size,
                    text=time_label(self._current_time),
                    justification='right'
                )
                self._data['time_actor'] = time_actor
                self._time_label_added = True
            if colorbar and not self._colorbar_added and do:
                kwargs = dict(source=actor, n_labels=8, color=self._fg_color,
                              bgcolor=self._brain_color[:3])
                kwargs.update(colorbar_kwargs or {})
                self._renderer.scalarbar(**kwargs)
                self._colorbar_added = True
            self._renderer.set_camera(**views_dicts[hemi][v])

        # 4) update the scalar bar and opacity
        self.update_lut(alpha=alpha)
        self._update()

    def _iter_views(self, hemi):
        # which rows and columns each type of visual needs to be added to
        if self._hemi == 'split':
            hemi_dict = dict(lh=[0], rh=[1], vol=[0, 1])
        else:
            hemi_dict = dict(lh=[0], rh=[0], vol=[0])
        for vi, view in enumerate(self._views):
            if self._hemi == 'split':
                view_dict = dict(lh=[vi], rh=[vi], vol=[vi, vi])
            else:
                view_dict = dict(lh=[vi], rh=[vi], vol=[vi])
            if self._view_layout == 'vertical':
                rows = view_dict  # views are rows
                cols = hemi_dict  # hemis are columns
            else:
                rows = hemi_dict  # hemis are rows
                cols = view_dict  # views are columns
            for ri, ci in zip(rows[hemi], cols[hemi]):
                yield ri, ci, view

    def remove_labels(self):
        """Remove all the ROI labels from the image."""
        for hemi in self._hemis:
            mesh = self._layered_meshes[hemi]
            mesh.remove_overlay(self._labels[hemi])
            self._labels[hemi].clear()
        self._update()

    def remove_annotations(self):
        """Remove all annotations from the image."""
        for hemi in self._hemis:
            mesh = self._layered_meshes[hemi]
            mesh.remove_overlay(self._annots[hemi])
            self._annots[hemi].clear()
        self._update()

    def _add_volume_data(self, hemi, src, volume_options):
        from ..backends._pyvista import _volume
        _validate_type(src, SourceSpaces, 'src')
        _check_option('src.kind', src.kind, ('volume',))
        _validate_type(
            volume_options, (dict, 'numeric', None), 'volume_options')
        assert hemi == 'vol'
        if not isinstance(volume_options, dict):
            volume_options = dict(
                resolution=float(volume_options) if volume_options is not None
                else None)
        volume_options = _handle_default('volume_options', volume_options)
        allowed_types = (
            ['resolution', (None, 'numeric')],
            ['blending', (str,)],
            ['alpha', ('numeric', None)],
            ['surface_alpha', (None, 'numeric')],
            ['silhouette_alpha', (None, 'numeric')],
            ['silhouette_linewidth', ('numeric',)],
        )
        for key, types in allowed_types:
            _validate_type(volume_options[key], types,
                           f'volume_options[{repr(key)}]')
        extra_keys = set(volume_options) - set(a[0] for a in allowed_types)
        if len(extra_keys):
            raise ValueError(
                f'volume_options got unknown keys {sorted(extra_keys)}')
        blending = _check_option('volume_options["blending"]',
                                 volume_options['blending'],
                                 ('composite', 'mip'))
        alpha = volume_options['alpha']
        if alpha is None:
            alpha = 0.4 if self._data[hemi]['array'].ndim == 3 else 1.
        alpha = np.clip(float(alpha), 0., 1.)
        resolution = volume_options['resolution']
        surface_alpha = volume_options['surface_alpha']
        if surface_alpha is None:
            surface_alpha = min(alpha / 2., 0.1)
        silhouette_alpha = volume_options['silhouette_alpha']
        if silhouette_alpha is None:
            silhouette_alpha = surface_alpha / 4.
        silhouette_linewidth = volume_options['silhouette_linewidth']
        del volume_options
        volume_pos = self._data[hemi].get('grid_volume_pos')
        volume_neg = self._data[hemi].get('grid_volume_neg')
        center = self._data['center']
        if volume_pos is None:
            xyz = np.meshgrid(
                *[np.arange(s) for s in src[0]['shape']], indexing='ij')
            dimensions = np.array(src[0]['shape'], int)
            mult = 1000 if self._units == 'mm' else 1
            src_mri_t = src[0]['src_mri_t']['trans'].copy()
            src_mri_t[:3] *= mult
            if resolution is not None:
                resolution = resolution * mult / 1000.  # to mm
            del src, mult
            coords = np.array([c.ravel(order='F') for c in xyz]).T
            coords = apply_trans(src_mri_t, coords)
            self.geo[hemi] = Bunch(coords=coords)
            vertices = self._data[hemi]['vertices']
            assert self._data[hemi]['array'].shape[0] == len(vertices)
            # MNE constructs the source space on a uniform grid in MRI space,
            # but mne coreg can change it to be non-uniform, so we need to
            # use all three elements here
            assert np.allclose(
                src_mri_t[:3, :3], np.diag(np.diag(src_mri_t)[:3]))
            spacing = np.diag(src_mri_t)[:3]
            origin = src_mri_t[:3, 3] - spacing / 2.
            scalars = np.zeros(np.prod(dimensions))
            scalars[vertices] = 1.  # for the outer mesh
            grid, grid_mesh, volume_pos, volume_neg = \
                _volume(dimensions, origin, spacing, scalars, surface_alpha,
                        resolution, blending, center)
            self._data[hemi]['alpha'] = alpha  # incorrectly set earlier
            self._data[hemi]['grid'] = grid
            self._data[hemi]['grid_mesh'] = grid_mesh
            self._data[hemi]['grid_coords'] = coords
            self._data[hemi]['grid_src_mri_t'] = src_mri_t
            self._data[hemi]['grid_shape'] = dimensions
            self._data[hemi]['grid_volume_pos'] = volume_pos
            self._data[hemi]['grid_volume_neg'] = volume_neg
        actor_pos, _ = self._renderer.plotter.add_actor(
            volume_pos, reset_camera=False, name=None, culling=False)
        if volume_neg is not None:
            actor_neg, _ = self._renderer.plotter.add_actor(
                volume_neg, reset_camera=False, name=None, culling=False)
        else:
            actor_neg = None
        grid_mesh = self._data[hemi]['grid_mesh']
        if grid_mesh is not None:
            import vtk
            _, prop = self._renderer.plotter.add_actor(
                grid_mesh, reset_camera=False, name=None, culling=False,
                pickable=False)
            prop.SetColor(*self._brain_color[:3])
            prop.SetOpacity(surface_alpha)
            if silhouette_alpha > 0 and silhouette_linewidth > 0:
                for ri, ci, v in self._iter_views('vol'):
                    self._renderer.subplot(ri, ci)
                    grid_silhouette = vtk.vtkPolyDataSilhouette()
                    grid_silhouette.SetInputData(grid_mesh.GetInput())
                    grid_silhouette.SetCamera(
                        self._renderer.plotter.renderer.GetActiveCamera())
                    grid_silhouette.SetEnableFeatureAngle(0)
                    grid_silhouette_mapper = vtk.vtkPolyDataMapper()
                    grid_silhouette_mapper.SetInputConnection(
                        grid_silhouette.GetOutputPort())
                    _, prop = self._renderer.plotter.add_actor(
                        grid_silhouette_mapper, reset_camera=False, name=None,
                        culling=False, pickable=False)
                    prop.SetColor(*self._brain_color[:3])
                    prop.SetOpacity(silhouette_alpha)
                    prop.SetLineWidth(silhouette_linewidth)

        return actor_pos, actor_neg

    def add_label(self, label, color=None, alpha=1, scalar_thresh=None,
                  borders=False, hemi=None, subdir=None,
                  reset_camera=True):
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
            If None, it is assumed to belong to the hemipshere being
            shown.
        subdir : None | str
            If a label is specified as name, subdir can be used to indicate
            that the label file is in a sub-directory of the subject's
            label directory rather than in the label directory itself (e.g.
            for ``$SUBJECTS_DIR/$SUBJECT/label/aparc/lh.cuneus.label``
            ``brain.add_label('cuneus', subdir='aparc')``).
        reset_camera : bool
            If True, reset the camera view after adding the label. Defaults
            to True.

        Notes
        -----
        To remove previously added labels, run Brain.remove_labels().
        """
        from matplotlib.colors import colorConverter
        from ...label import read_label
        if isinstance(label, str):
            if color is None:
                color = "crimson"

            if os.path.isfile(label):
                filepath = label
                label = read_label(filepath)
                hemi = label.hemi
                label_name = os.path.basename(filepath).split('.')[1]
            else:
                hemi = self._check_hemi(hemi)
                label_name = label
                label_fname = ".".join([hemi, label_name, 'label'])
                if subdir is None:
                    filepath = op.join(self._subjects_dir, self._subject_id,
                                       'label', label_fname)
                else:
                    filepath = op.join(self._subjects_dir, self._subject_id,
                                       'label', subdir, label_fname)
                if not os.path.exists(filepath):
                    raise ValueError('Label file %s does not exist'
                                     % filepath)
                label = read_label(filepath)
            ids = label.vertices
            scalars = label.values
        else:
            # try to extract parameters from label instance
            try:
                hemi = label.hemi
                ids = label.vertices
                if label.name is None:
                    label_name = 'unnamed'
                else:
                    label_name = str(label.name)

                if color is None:
                    if hasattr(label, 'color') and label.color is not None:
                        color = label.color
                    else:
                        color = "crimson"

                if scalar_thresh is not None:
                    scalars = label.values
            except Exception:
                raise ValueError('Label was not a filename (str), and could '
                                 'not be understood as a class. The class '
                                 'must have attributes "hemi", "vertices", '
                                 '"name", and (if scalar_thresh is not None)'
                                 '"values"')
            hemi = self._check_hemi(hemi)

        if scalar_thresh is not None:
            ids = ids[scalars >= scalar_thresh]

        scalars = np.zeros(self.geo[hemi].coords.shape[0])
        scalars[ids] = 1

        if self.time_viewer and self.show_traces:
            stc = self._data["stc"]
            src = self._data["src"]
            tc = stc.extract_label_time_course(label, src=src,
                                               mode=self.label_extract_mode)
            tc = tc[0] if tc.ndim == 2 else tc[0, 0, :]
            color = next(self.color_cycle)
            line = self.mpl_canvas.plot(
                self._data['time'], tc, label=label_name,
                color=color)
        else:
            line = None

        orig_color = color
        color = colorConverter.to_rgba(color, alpha)
        cmap = np.array([(0, 0, 0, 0,), color])
        ctable = np.round(cmap * 255).astype(np.uint8)

        for ri, ci, v in self._iter_views(hemi):
            self._renderer.subplot(ri, ci)
            if borders:
                n_vertices = scalars.size
                edges = mesh_edges(self.geo[hemi].faces)
                edges = edges.tocoo()
                border_edges = scalars[edges.row] != scalars[edges.col]
                show = np.zeros(n_vertices, dtype=np.int64)
                keep_idx = np.unique(edges.row[border_edges])
                if isinstance(borders, int):
                    for _ in range(borders):
                        keep_idx = np.in1d(
                            self.geo[hemi].faces.ravel(), keep_idx)
                        keep_idx.shape = self.geo[hemi].faces.shape
                        keep_idx = self.geo[hemi].faces[np.any(
                            keep_idx, axis=1)]
                        keep_idx = np.unique(keep_idx)
                show[keep_idx] = 1
                scalars *= show

            mesh = self._layered_meshes[hemi]
            mesh.add_overlay(
                scalars=scalars,
                colormap=ctable,
                rng=[np.min(scalars), np.max(scalars)],
                opacity=alpha,
                name=label_name,
            )
            if reset_camera:
                self._renderer.set_camera(**views_dicts[hemi][v])
            if self.time_viewer and self.traces_mode == 'label':
                label._color = orig_color
                label._line = line
            self._labels[hemi].append(label)
        self._update()

    def add_foci(self, coords, coords_as_verts=False, map_surface=None,
                 scale_factor=1, color="white", alpha=1, name=None,
                 hemi=None, resolution=50):
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
        map_surface : None
            Surface to map coordinates through, or None to use raw coords.
        scale_factor : float
            Controls the size of the foci spheres (relative to 1cm).
        color : matplotlib color code
            HTML name, RBG tuple, or hex code.
        alpha : float in [0, 1]
            Opacity of focus gylphs.
        name : str
            Internal name to use.
        hemi : str | None
            If None, it is assumed to belong to the hemipshere being
            shown. If two hemispheres are being shown, an error will
            be thrown.
        resolution : int
            The resolution of the spheres.
        """
        from matplotlib.colors import colorConverter
        hemi = self._check_hemi(hemi, extras=['vol'])

        # those parameters are not supported yet, only None is allowed
        _check_option('map_surface', map_surface, [None])

        # Figure out how to interpret the first parameter
        if coords_as_verts:
            coords = self.geo[hemi].coords[coords]

        # Convert the color code
        if not isinstance(color, tuple):
            color = colorConverter.to_rgb(color)

        if self._units == 'm':
            scale_factor = scale_factor / 1000.
        for ri, ci, v in self._iter_views(hemi):
            self._renderer.subplot(ri, ci)
            self._renderer.sphere(center=coords, color=color,
                                  scale=(10. * scale_factor),
                                  opacity=alpha, resolution=resolution)
            self._renderer.set_camera(**views_dicts[hemi][v])

    def add_text(self, x, y, text, name=None, color=None, opacity=1.0,
                 row=-1, col=-1, font_size=None, justification=None):
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
        row : int
            Row index of which brain to use.
        col : int
            Column index of which brain to use.
        font_size : float | None
            The font size to use.
        justification : str | None
            The text justification.
        """
        # XXX: support `name` should be added when update_text/remove_text
        # are implemented
        # _check_option('name', name, [None])

        self._renderer.text2d(x_window=x, y_window=y, text=text, color=color,
                              size=font_size, justification=justification)

    def _configure_label_time_course(self):
        from ...label import read_labels_from_annot
        if not self.show_traces:
            return
        if self.mpl_canvas is None:
            self._configure_mplcanvas()
        else:
            self.clear_glyphs()
        self.traces_mode = 'label'
        self.add_annotation(self.annot, color="w", alpha=0.75)

        # now plot the time line
        self.plot_time_line()
        self.mpl_canvas.update_plot()

        for hemi in self._hemis:
            labels = read_labels_from_annot(
                subject=self._subject_id,
                parc=self.annot,
                hemi=hemi,
                subjects_dir=self._subjects_dir
            )
            self._vertex_to_label_id[hemi] = np.full(
                self.geo[hemi].coords.shape[0], -1)
            self._annotation_labels[hemi] = labels
            for idx, label in enumerate(labels):
                self._vertex_to_label_id[hemi][label.vertices] = idx

    def add_annotation(self, annot, borders=True, alpha=1, hemi=None,
                       remove_existing=True, color=None, **kwargs):
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
        alpha : float in [0, 1]
            Alpha level to control opacity.
        hemi : str | None
            If None, it is assumed to belong to the hemipshere being
            shown. If two hemispheres are being shown, data must exist
            for both hemispheres.
        remove_existing : bool
            If True (default), remove old annotations.
        color : matplotlib-style color code
            If used, show all annotations in the same (specified) color.
            Probably useful only when showing annotation borders.
        **kwargs : dict
            These are passed to the underlying
            ``mayavi.mlab.pipeline.surface`` call.
        """
        from ...label import _read_annot
        hemis = self._check_hemis(hemi)

        # Figure out where the data is coming from
        if isinstance(annot, str):
            if os.path.isfile(annot):
                filepath = annot
                path = os.path.split(filepath)[0]
                file_hemi, annot = os.path.basename(filepath).split('.')[:2]
                if len(hemis) > 1:
                    if annot[:2] == 'lh.':
                        filepaths = [filepath, op.join(path, 'rh' + annot[2:])]
                    elif annot[:2] == 'rh.':
                        filepaths = [op.join(path, 'lh' + annot[2:], filepath)]
                    else:
                        raise RuntimeError('To add both hemispheres '
                                           'simultaneously, filename must '
                                           'begin with "lh." or "rh."')
                else:
                    filepaths = [filepath]
            else:
                filepaths = []
                for hemi in hemis:
                    filepath = op.join(self._subjects_dir,
                                       self._subject_id,
                                       'label',
                                       ".".join([hemi, annot, 'annot']))
                    if not os.path.exists(filepath):
                        raise ValueError('Annotation file %s does not exist'
                                         % filepath)
                    filepaths += [filepath]
            annots = []
            for hemi, filepath in zip(hemis, filepaths):
                # Read in the data
                labels, cmap, _ = _read_annot(filepath)
                annots.append((labels, cmap))
        else:
            annots = [annot] if len(hemis) == 1 else annot
            annot = 'annotation'

        for hemi, (labels, cmap) in zip(hemis, annots):
            # Maybe zero-out the non-border vertices
            self._to_borders(labels, hemi, borders)

            # Handle null labels properly
            cmap[:, 3] = 255
            bgcolor = np.round(np.array(self._brain_color) * 255).astype(int)
            bgcolor[-1] = 0
            cmap[cmap[:, 4] < 0, 4] += 2 ** 24  # wrap to positive
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
                from matplotlib.colors import colorConverter
                rgb = np.round(np.multiply(colorConverter.to_rgb(color), 255))
                cmap[:, :3] = rgb.astype(cmap.dtype)

            ctable = cmap.astype(np.float64)
            for ri, ci, _ in self._iter_views(hemi):
                self._renderer.subplot(ri, ci)
                mesh = self._layered_meshes[hemi]
                mesh.add_overlay(
                    scalars=ids,
                    colormap=ctable,
                    rng=[np.min(ids), np.max(ids)],
                    opacity=alpha,
                    name=annot,
                )
                self._annots[hemi].append(annot)
                if not self.time_viewer or self.traces_mode == 'vertex':
                    from ..backends._pyvista import _set_colormap_range
                    _set_colormap_range(mesh._actor, cmap.astype(np.uint8),
                                        None)

        self._update()

    def close(self):
        """Close all figures and cleanup data structure."""
        self._closed = True
        self._renderer.close()

    def show(self):
        """Display the window."""
        self._renderer.show()

    def show_view(self, view=None, roll=None, distance=None, row=0, col=0,
                  hemi=None):
        """Orient camera to display view.

        Parameters
        ----------
        view : str | dict
            String view, or a dict with azimuth and elevation.
        roll : float | None
            The roll.
        distance : float | None
            The distance.
        row : int
            The row to set.
        col : int
            The column to set.
        hemi : str
            Which hemi to use for string lookup (when in "both" mode).
        """
        hemi = self._hemi if hemi is None else hemi
        if hemi == 'split':
            if (self._view_layout == 'vertical' and col == 1 or
                    self._view_layout == 'horizontal' and row == 1):
                hemi = 'rh'
            else:
                hemi = 'lh'
        if isinstance(view, str):
            view = views_dicts[hemi].get(view)
        view = view.copy()
        if roll is not None:
            view.update(roll=roll)
        if distance is not None:
            view.update(distance=distance)
        self._renderer.subplot(row, col)
        self._renderer.set_camera(**view, reset_camera=False)
        self._update()

    def reset_view(self):
        """Reset the camera."""
        for h in self._hemis:
            for ri, ci, v in self._iter_views(h):
                self._renderer.subplot(ri, ci)
                self._renderer.set_camera(**views_dicts[h][v],
                                          reset_camera=False)

    def save_image(self, filename, mode='rgb'):
        """Save view from all panels to disk.

        Parameters
        ----------
        filename : str
            Path to new image file.
        mode : str
            Either 'rgb' or 'rgba' for values to return.
        """
        self._renderer.screenshot(mode=mode, filename=filename)

    @fill_doc
    def screenshot(self, mode='rgb', time_viewer=False):
        """Generate a screenshot of current view.

        Parameters
        ----------
        mode : str
            Either 'rgb' or 'rgba' for values to return.
        %(brain_screenshot_time_viewer)s

        Returns
        -------
        screenshot : array
            Image pixel values.
        """
        img = self._renderer.screenshot(mode)
        if time_viewer and self.time_viewer and \
                self.show_traces and \
                not self.separate_canvas:
            canvas = self.mpl_canvas.fig.canvas
            canvas.draw_idle()
            # In theory, one of these should work:
            #
            # trace_img = np.frombuffer(
            #     canvas.tostring_rgb(), dtype=np.uint8)
            # trace_img.shape = canvas.get_width_height()[::-1] + (3,)
            #
            # or
            #
            # trace_img = np.frombuffer(
            #     canvas.tostring_rgb(), dtype=np.uint8)
            # size = time_viewer.mpl_canvas.getSize()
            # trace_img.shape = (size.height(), size.width(), 3)
            #
            # But in practice, sometimes the sizes does not match the
            # renderer tostring_rgb() size. So let's directly use what
            # matplotlib does in lib/matplotlib/backends/backend_agg.py
            # before calling tobytes():
            trace_img = np.asarray(
                canvas.renderer._renderer).take([0, 1, 2], axis=2)
            # need to slice into trace_img because generally it's a bit
            # smaller
            delta = trace_img.shape[1] - img.shape[1]
            if delta > 0:
                start = delta // 2
                trace_img = trace_img[:, start:start + img.shape[1]]
            img = np.concatenate([img, trace_img], axis=0)
        return img

    @fill_doc
    def update_lut(self, fmin=None, fmid=None, fmax=None, alpha=None):
        """Update color map.

        Parameters
        ----------
        %(fmin_fmid_fmax)s
        alpha : float | None
            Alpha to use in the update.
        """
        from ..backends._pyvista import _set_colormap_range, _set_volume_range
        center = self._data['center']
        colormap = self._data['colormap']
        transparent = self._data['transparent']
        lims = dict(fmin=fmin, fmid=fmid, fmax=fmax)
        lims = {key: self._data[key] if val is None else val
                for key, val in lims.items()}
        assert all(val is not None for val in lims.values())
        if lims['fmin'] > lims['fmid']:
            lims['fmin'] = lims['fmid']
        if lims['fmax'] < lims['fmid']:
            lims['fmax'] = lims['fmid']
        self._data.update(lims)
        self._data['ctable'] = np.round(
            calculate_lut(colormap, alpha=1., center=center,
                          transparent=transparent, **lims) *
            255).astype(np.uint8)
        # update our values
        rng = self._cmap_range
        ctable = self._data['ctable']
        # in testing, no plotter; if colorbar=False, no scalar_bar
        scalar_bar = getattr(
            getattr(self._renderer, 'plotter', None), 'scalar_bar', None)
        for hemi in ['lh', 'rh', 'vol']:
            hemi_data = self._data.get(hemi)
            if hemi_data is not None:
                if hemi in self._layered_meshes:
                    mesh = self._layered_meshes[hemi]
                    mesh.update_overlay(name='data',
                                        colormap=self._data['ctable'],
                                        opacity=alpha,
                                        rng=rng)
                    _set_colormap_range(mesh._actor, ctable, scalar_bar, rng,
                                        self._brain_color)
                    scalar_bar = None

                grid_volume_pos = hemi_data.get('grid_volume_pos')
                grid_volume_neg = hemi_data.get('grid_volume_neg')
                for grid_volume in (grid_volume_pos, grid_volume_neg):
                    if grid_volume is not None:
                        _set_volume_range(
                            grid_volume, ctable, hemi_data['alpha'],
                            scalar_bar, rng)
                        scalar_bar = None

                glyph_actor = hemi_data.get('glyph_actor')
                if glyph_actor is not None:
                    for glyph_actor_ in glyph_actor:
                        _set_colormap_range(
                            glyph_actor_, ctable, scalar_bar, rng)
                        scalar_bar = None

    def set_data_smoothing(self, n_steps):
        """Set the number of smoothing steps.

        Parameters
        ----------
        n_steps : int
            Number of smoothing steps.
        """
        from ...morph import _hemi_morph
        for hemi in ['lh', 'rh']:
            hemi_data = self._data.get(hemi)
            if hemi_data is not None:
                if len(hemi_data['array']) >= self.geo[hemi].x.shape[0]:
                    continue
                vertices = hemi_data['vertices']
                if vertices is None:
                    raise ValueError(
                        'len(data) < nvtx (%s < %s): the vertices '
                        'parameter must not be None'
                        % (len(hemi_data), self.geo[hemi].x.shape[0]))
                morph_n_steps = 'nearest' if n_steps == 0 else n_steps
                maps = sparse.eye(len(self.geo[hemi].coords), format='csr')
                with use_log_level(False):
                    smooth_mat = _hemi_morph(
                        self.geo[hemi].orig_faces,
                        np.arange(len(self.geo[hemi].coords)),
                        vertices, morph_n_steps, maps, warn=False)
                self._data[hemi]['smooth_mat'] = smooth_mat
        self.set_time_point(self._data['time_idx'])
        self._data['smoothing_steps'] = n_steps

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
        %(brain_time_interpolation)s
        """
        self._time_interpolation = _check_option(
            'interpolation',
            interpolation,
            ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic')
        )
        self._time_interp_funcs = dict()
        self._time_interp_inv = None
        if self._times is not None:
            idx = np.arange(self._n_times)
            for hemi in ['lh', 'rh', 'vol']:
                hemi_data = self._data.get(hemi)
                if hemi_data is not None:
                    array = hemi_data['array']
                    self._time_interp_funcs[hemi] = _safe_interp1d(
                        idx, array, self._time_interpolation, axis=-1,
                        assume_sorted=True)
            self._time_interp_inv = _safe_interp1d(idx, self._times)

    def set_time_point(self, time_idx):
        """Set the time point shown (can be a float to interpolate).

        Parameters
        ----------
        time_idx : int | float
            The time index to use. Can be a float to use interpolation
            between indices.
        """
        self._current_act_data = dict()
        time_actor = self._data.get('time_actor', None)
        time_label = self._data.get('time_label', None)
        for hemi in ['lh', 'rh', 'vol']:
            hemi_data = self._data.get(hemi)
            if hemi_data is not None:
                array = hemi_data['array']
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
                grid = hemi_data.get('grid')
                if grid is not None:
                    vertices = self._data['vol']['vertices']
                    values = self._current_act_data['vol']
                    rng = self._cmap_range
                    fill = 0 if self._data['center'] is not None else rng[0]
                    grid.cell_arrays['values'].fill(fill)
                    # XXX for sided data, we probably actually need two
                    # volumes as composite/MIP needs to look at two
                    # extremes... for now just use abs. Eventually we can add
                    # two volumes if we want.
                    grid.cell_arrays['values'][vertices] = values

                # interpolate in space
                smooth_mat = hemi_data.get('smooth_mat')
                if smooth_mat is not None:
                    act_data = smooth_mat.dot(act_data)

                # update the mesh scalar values
                if hemi in self._layered_meshes:
                    mesh = self._layered_meshes[hemi]
                    if 'data' in mesh._overlays:
                        mesh.update_overlay(name='data', scalars=act_data)
                    else:
                        mesh.add_overlay(
                            scalars=act_data,
                            colormap=self._data['ctable'],
                            rng=self._cmap_range,
                            opacity=None,
                            name='data',
                        )

                # update the glyphs
                if vectors is not None:
                    self._update_glyphs(hemi, vectors)

        self._data['time_idx'] = time_idx
        self._update()

    def set_time(self, time):
        """Set the time to display (in seconds).

        Parameters
        ----------
        time : float
            The time to show, in seconds.
        """
        if self._times is None:
            raise ValueError(
                'Cannot set time when brain has no defined times.')
        elif min(self._times) <= time <= max(self._times):
            self.set_time_point(np.interp(float(time), self._times,
                                          np.arange(self._n_times)))
        else:
            raise ValueError(
                f'Requested time ({time} s) is outside the range of '
                f'available times ({min(self._times)}-{max(self._times)} s).')

    def _update_glyphs(self, hemi, vectors):
        from ..backends._pyvista import _set_colormap_range, _create_actor
        hemi_data = self._data.get(hemi)
        assert hemi_data is not None
        vertices = hemi_data['vertices']
        vector_alpha = self._data['vector_alpha']
        scale_factor = self._data['scale_factor']
        vertices = slice(None) if vertices is None else vertices
        x, y, z = np.array(self.geo[hemi].coords)[vertices].T

        if hemi_data['glyph_actor'] is None:
            add = True
            hemi_data['glyph_actor'] = list()
        else:
            add = False
        count = 0
        for ri, ci, _ in self._iter_views(hemi):
            self._renderer.subplot(ri, ci)
            if hemi_data['glyph_dataset'] is None:
                glyph_mapper, glyph_dataset = self._renderer.quiver3d(
                    x, y, z,
                    vectors[:, 0], vectors[:, 1], vectors[:, 2],
                    color=None,
                    mode='2darrow',
                    scale_mode='vector',
                    scale=scale_factor,
                    opacity=vector_alpha,
                    name=str(hemi) + "_glyph"
                )
                hemi_data['glyph_dataset'] = glyph_dataset
                hemi_data['glyph_mapper'] = glyph_mapper
            else:
                glyph_dataset = hemi_data['glyph_dataset']
                glyph_dataset.point_arrays['vec'] = vectors
                glyph_mapper = hemi_data['glyph_mapper']
            if add:
                glyph_actor = _create_actor(glyph_mapper)
                prop = glyph_actor.GetProperty()
                prop.SetLineWidth(2.)
                prop.SetOpacity(vector_alpha)
                self._renderer.plotter.add_actor(glyph_actor)
                hemi_data['glyph_actor'].append(glyph_actor)
            else:
                glyph_actor = hemi_data['glyph_actor'][count]
            count += 1
            _set_colormap_range(
                actor=glyph_actor,
                ctable=self._data['ctable'],
                scalar_bar=None,
                rng=self._cmap_range,
            )

    @property
    def _cmap_range(self):
        dt_max = self._data['fmax']
        if self._data['center'] is None:
            dt_min = self._data['fmin']
        else:
            dt_min = -1 * dt_max
        rng = [dt_min, dt_max]
        return rng

    def _update_fscale(self, fscale):
        """Scale the colorbar points."""
        fmin = self._data['fmin'] * fscale
        fmid = self._data['fmid'] * fscale
        fmax = self._data['fmax'] * fscale
        self.update_lut(fmin=fmin, fmid=fmid, fmax=fmax)

    def _update_auto_scaling(self, restore=False):
        user_clim = self._data['clim']
        if user_clim is not None and 'lims' in user_clim:
            allow_pos_lims = False
        else:
            allow_pos_lims = True
        if user_clim is not None and restore:
            clim = user_clim
        else:
            clim = 'auto'
        colormap = self._data['colormap']
        transparent = self._data['transparent']
        mapdata = _process_clim(
            clim, colormap, transparent,
            np.concatenate(list(self._current_act_data.values())),
            allow_pos_lims)
        diverging = 'pos_lims' in mapdata['clim']
        colormap = mapdata['colormap']
        scale_pts = mapdata['clim']['pos_lims' if diverging else 'lims']
        transparent = mapdata['transparent']
        del mapdata
        fmin, fmid, fmax = scale_pts
        center = 0. if diverging else None
        self._data['center'] = center
        self._data['colormap'] = colormap
        self._data['transparent'] = transparent
        self.update_lut(fmin=fmin, fmid=fmid, fmax=fmax)

    def _to_time_index(self, value):
        """Return the interpolated time index of the given time value."""
        time = self._data['time']
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

    def _save_movie(self, filename, time_dilation=4., tmin=None, tmax=None,
                    framerate=24, interpolation=None, codec=None,
                    bitrate=None, callback=None, time_viewer=False, **kwargs):
        import imageio
        from ..backends._pyvista import _disabled_interaction
        with _disabled_interaction(self._renderer):
            images = self._make_movie_frames(
                time_dilation, tmin, tmax, framerate, interpolation, callback,
                time_viewer)
        # find imageio FFMPEG parameters
        if 'fps' not in kwargs:
            kwargs['fps'] = framerate
        if codec is not None:
            kwargs['codec'] = codec
        if bitrate is not None:
            kwargs['bitrate'] = bitrate
        imageio.mimwrite(filename, images, **kwargs)

    @fill_doc
    def save_movie(self, filename, time_dilation=4., tmin=None, tmax=None,
                   framerate=24, interpolation=None, codec=None,
                   bitrate=None, callback=None, time_viewer=False, **kwargs):
        """Save a movie (for data with a time axis).

        The movie is created through the :mod:`imageio` module. The format is
        determined by the extension, and additional options can be specified
        through keyword arguments that depend on the format. For available
        formats and corresponding parameters see the imageio documentation:
        http://imageio.readthedocs.io/en/latest/formats.html#multiple-images

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
        %(brain_time_interpolation)s
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
        %(brain_screenshot_time_viewer)s
        **kwargs : dict
            Specify additional options for :mod:`imageio`.

        Returns
        -------
        dialog : object
            The opened dialog is returned for testing purpose only.
        """
        if self.time_viewer:
            try:
                from pyvista.plotting.qt_plotting import FileDialog
            except ImportError:
                from pyvistaqt.plotting import FileDialog

            if filename is None:
                self.status_msg.setText("Choose movie path ...")
                self.status_msg.show()
                self.status_progress.setValue(0)

                def _post_setup(unused):
                    del unused
                    self.status_msg.hide()
                    self.status_progress.hide()

                dialog = FileDialog(
                    self.plotter.app_window,
                    callback=partial(self._save_movie, **kwargs)
                )
                dialog.setDirectory(os.getcwd())
                dialog.finished.connect(_post_setup)
                return dialog
            else:
                from PyQt5.QtCore import Qt
                from PyQt5.QtGui import QCursor

                def frame_callback(frame, n_frames):
                    if frame == n_frames:
                        # On the ImageIO step
                        self.status_msg.setText(
                            "Saving with ImageIO: %s"
                            % filename
                        )
                        self.status_msg.show()
                        self.status_progress.hide()
                        self.status_bar.layout().update()
                    else:
                        self.status_msg.setText(
                            "Rendering images (frame %d / %d) ..."
                            % (frame + 1, n_frames)
                        )
                        self.status_msg.show()
                        self.status_progress.show()
                        self.status_progress.setRange(0, n_frames - 1)
                        self.status_progress.setValue(frame)
                        self.status_progress.update()
                        self.status_progress.repaint()
                    self.status_msg.update()
                    self.status_msg.parent().update()
                    self.status_msg.repaint()

                # temporarily hide interface
                default_visibility = self.visibility
                self.toggle_interface(value=False)
                # set cursor to busy
                default_cursor = self.interactor.cursor()
                self.interactor.setCursor(QCursor(Qt.WaitCursor))

                try:
                    self._save_movie(
                        filename=filename,
                        time_dilation=(1. / self.playback_speed),
                        callback=frame_callback,
                        **kwargs
                    )
                except (Exception, KeyboardInterrupt):
                    warn('Movie saving aborted:\n' + traceback.format_exc())

                # restore visibility
                self.toggle_interface(value=default_visibility)
                # restore cursor
                self.interactor.setCursor(default_cursor)
        else:
            self._save_movie(filename, time_dilation, tmin, tmax,
                             framerate, interpolation, codec,
                             bitrate, callback, time_viewer, **kwargs)

    def _make_movie_frames(self, time_dilation, tmin, tmax, framerate,
                           interpolation, callback, time_viewer):
        from math import floor

        # find tmin
        if tmin is None:
            tmin = self._times[0]
        elif tmin < self._times[0]:
            raise ValueError("tmin=%r is smaller than the first time point "
                             "(%r)" % (tmin, self._times[0]))

        # find indexes at which to create frames
        if tmax is None:
            tmax = self._times[-1]
        elif tmax > self._times[-1]:
            raise ValueError("tmax=%r is greater than the latest time point "
                             "(%r)" % (tmax, self._times[-1]))
        n_frames = floor((tmax - tmin) * time_dilation * framerate)
        times = np.arange(n_frames, dtype=float)
        times /= framerate * time_dilation
        times += tmin
        time_idx = np.interp(times, self._times, np.arange(self._n_times))

        n_times = len(time_idx)
        if n_times == 0:
            raise ValueError("No time points selected")

        logger.debug("Save movie for time points/samples\n%s\n%s"
                     % (times, time_idx))
        # Sometimes the first screenshot is rendered with a different
        # resolution on OS X
        self.screenshot(time_viewer=time_viewer)
        old_mode = self.time_interpolation
        if interpolation is not None:
            self.set_time_interpolation(interpolation)
        try:
            images = [
                self.screenshot(time_viewer=time_viewer)
                for _ in self._iter_time(time_idx, callback)]
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
        if self.time_viewer:
            func = partial(self.callbacks["time"],
                           update_widget=True)
        else:
            func = self.set_time_point
        current_time_idx = self._data["time_idx"]
        for ii, idx in enumerate(time_idx):
            func(idx)
            if callback is not None:
                callback(frame=ii, n_frames=len(time_idx))
            yield idx

        # Restore original time index
        func(current_time_idx)

    def _show(self):
        """Request rendering of the window."""
        try:
            return self._renderer.show()
        except RuntimeError:
            logger.info("No active/running renderer available.")

    def _check_stc(self, hemi, array, vertices):
        from ...source_estimate import (
            _BaseSourceEstimate, _BaseSurfaceSourceEstimate,
            _BaseMixedSourceEstimate, _BaseVolSourceEstimate
        )
        if isinstance(array, _BaseSourceEstimate):
            stc = array
            stc_surf = stc_vol = None
            if isinstance(stc, _BaseSurfaceSourceEstimate):
                stc_surf = stc
            elif isinstance(stc, _BaseMixedSourceEstimate):
                stc_surf = stc.surface() if hemi != 'vol' else None
                stc_vol = stc.volume() if hemi == 'vol' else None
            elif isinstance(stc, _BaseVolSourceEstimate):
                stc_vol = stc if hemi == 'vol' else None
            else:
                raise TypeError("stc not supported")

            if stc_surf is None and stc_vol is None:
                raise ValueError("No data to be added")
            if stc_surf is not None:
                array = getattr(stc_surf, hemi + '_data')
                vertices = stc_surf.vertices[0 if hemi == 'lh' else 1]
            if stc_vol is not None:
                array = stc_vol.data
                vertices = np.concatenate(stc_vol.vertices)
        else:
            stc = None
        return stc, array, vertices

    def _check_hemi(self, hemi, extras=()):
        """Check for safe single-hemi input, returns str."""
        if hemi is None:
            if self._hemi not in ['lh', 'rh']:
                raise ValueError('hemi must not be None when both '
                                 'hemispheres are displayed')
            else:
                hemi = self._hemi
        elif hemi not in ['lh', 'rh'] + list(extras):
            extra = ' or None' if self._hemi in ['lh', 'rh'] else ''
            raise ValueError('hemi must be either "lh" or "rh"' +
                             extra + ", got " + str(hemi))
        return hemi

    def _check_hemis(self, hemi):
        """Check for safe dual or single-hemi input, returns list."""
        if hemi is None:
            if self._hemi not in ['lh', 'rh']:
                hemi = ['lh', 'rh']
            else:
                hemi = [self._hemi]
        elif hemi not in ['lh', 'rh']:
            extra = ' or None' if self._hemi in ['lh', 'rh'] else ''
            raise ValueError('hemi must be either "lh" or "rh"' + extra)
        else:
            hemi = [hemi]
        return hemi

    def _to_borders(self, label, hemi, borders, restrict_idx=None):
        """Convert a label/parc to borders."""
        if not isinstance(borders, (bool, int)) or borders < 0:
            raise ValueError('borders must be a bool or positive integer')
        if borders:
            n_vertices = label.size
            edges = mesh_edges(self.geo[hemi].orig_faces)
            edges = edges.tocoo()
            border_edges = label[edges.row] != label[edges.col]
            show = np.zeros(n_vertices, dtype=np.int64)
            keep_idx = np.unique(edges.row[border_edges])
            if isinstance(borders, int):
                for _ in range(borders):
                    keep_idx = np.in1d(
                        self.geo[hemi].orig_faces.ravel(), keep_idx)
                    keep_idx.shape = self.geo[hemi].orig_faces.shape
                    keep_idx = self.geo[hemi].orig_faces[
                        np.any(keep_idx, axis=1)]
                    keep_idx = np.unique(keep_idx)
                if restrict_idx is not None:
                    keep_idx = keep_idx[np.in1d(keep_idx, restrict_idx)]
            show[keep_idx] = 1
            label *= show

    def enable_depth_peeling(self):
        """Enable depth peeling."""
        self._renderer.enable_depth_peeling()

    def _update(self):
        from ..backends import renderer
        if renderer.get_3d_backend() in ['pyvista', 'notebook']:
            if self.notebook and self._renderer.figure.display is not None:
                self._renderer.figure.display.update()
            else:
                self._renderer.plotter.update()

    def get_picked_points(self):
        """Return the vertices of the picked points.

        Returns
        -------
        points : list of int | None
            The vertices picked by the time viewer.
        """
        if hasattr(self, "time_viewer"):
            return self.picked_points

    def __hash__(self):
        """Hash the object."""
        raise NotImplementedError


def _safe_interp1d(x, y, kind='linear', axis=-1, assume_sorted=False):
    """Work around interp1d not liking singleton dimensions."""
    from scipy.interpolate import interp1d
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
        fmid = (fmin + fmax) / 2.

    if fmin >= fmid:
        raise RuntimeError('min must be < mid, got %0.4g >= %0.4g'
                           % (fmin, fmid))
    if fmid >= fmax:
        raise RuntimeError('mid must be < max, got %0.4g >= %0.4g'
                           % (fmid, fmax))

    return fmin, fmid, fmax


def _get_range(brain):
    val = np.abs(np.concatenate(list(brain._current_act_data.values())))
    return [np.min(val), np.max(val)]


class _FakeIren():
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
