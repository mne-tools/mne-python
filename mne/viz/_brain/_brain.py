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
from io import BytesIO
import os
import os.path as op
import sys
import time
import copy
import traceback
import warnings

import numpy as np
from collections import OrderedDict

from .colormap import calculate_lut
from .surface import _Surface
from .view import views_dicts, _lh_views_dict
from .callback import (ShowView, TimeCallBack, SmartCallBack,
                       UpdateLUT, UpdateColorbarScale)

from ..utils import (_show_help_fig, _get_color_list, concatenate_images,
                     _generate_default_filename, _save_ndarray_img)
from .._3d import (_process_clim, _handle_time, _check_views,
                   _handle_sensor_types, _plot_sensors)
from ...defaults import _handle_default, DEFAULTS
from ...externals.decorator import decorator
from ...fixes import _point_data, _cell_data
from ..._freesurfer import (vertex_to_mni, read_talxfm, read_freesurfer_lut,
                            _get_head_surface, _get_skull_surface)
from ...io.pick import pick_types
from ...io.meas_info import Info
from ...surface import (mesh_edges, _mesh_borders, _marching_cubes,
                        get_meg_helmet_surf)
from ...source_space import SourceSpaces
from ...transforms import (apply_trans, invert_transform, _get_trans,
                           _get_transforms_to_coord_frame)
from ...utils import (_check_option, logger, verbose, fill_doc, _validate_type,
                      use_log_level, Bunch, _ReuseCycle, warn,
                      get_subjects_dir, _check_fname, _to_rgb)


_ARROW_MOVE = 10  # degrees per press


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
        from matplotlib.colors import Colormap, ListedColormap

        if isinstance(self._colormap, str):
            cmap = _get_cmap(self._colormap)
        elif isinstance(self._colormap, Colormap):
            cmap = self._colormap
        else:
            cmap = ListedColormap(
                self._colormap / 255., name=str(type(self._colormap)))
        logger.debug(
            f'Color mapping {repr(self._name)} with {cmap.name} '
            f'colormap and range {self._rng}')

        rng = self._rng
        assert rng is not None
        scalars = _norm(self._scalars, rng)

        colors = cmap(scalars)
        if self._opacity is not None:
            colors[:, 3] *= self._opacity
        return colors


def _norm(x, rng):
    if rng[0] == rng[1]:
        factor = 1 if rng[0] == 0 else 1e-6 * rng[0]
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

        self._current_colors = None
        self._cached_colors = None
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
        B = cache = None
        for overlay in self._overlays.values():
            A = overlay.to_colors()
            if B is None:
                B = A
            else:
                cache = B
                B = self._compute_over(cache, A)
        return B, cache

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
        if self._current_colors is None:
            self._current_colors = colors
        else:
            # save previous colors to cache
            self._cached_colors = self._current_colors
            self._current_colors = self._compute_over(
                self._cached_colors, colors)

        # apply the texture
        self._apply()

    def remove_overlay(self, names):
        to_update = False
        if not isinstance(names, list):
            names = [names]
        for name in names:
            if name in self._overlays:
                del self._overlays[name]
                to_update = True
        if to_update:
            self.update()

    def _apply(self):
        if self._current_colors is None or self._renderer is None:
            return
        self._renderer._set_mesh_scalars(
            mesh=self._polydata,
            scalars=self._current_colors,
            name=self._default_scalars_name,
        )

    def update(self, colors=None):
        if colors is not None and self._cached_colors is not None:
            self._current_colors = self._compute_over(
                self._cached_colors, colors)
        else:
            self._current_colors, self._cached_colors = \
                self._compose_overlays()
        self._apply()

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
        # partial update: use cache if possible
        if name == list(self._overlays.keys())[-1]:
            self.update(colors=overlay.to_colors())
        else:  # full update
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
    offset : bool | str
        If True, shifts the right- or left-most x coordinate of the left and
        right surfaces, respectively, to be at zero. This is useful for viewing
        inflated surface where hemispheres typically overlap. Can be "auto"
        (default) use True with inflated surfaces and False otherwise
        (Default: 'auto'). Only used when ``hemi='both'``.

        .. versionchanged:: 0.23
           Default changed to "auto".
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
    silhouette : dict | bool
       As a dict, it contains the ``color``, ``linewidth``, ``alpha`` opacity
       and ``decimate`` (level of decimation between 0 and 1 or None) of the
       brain's silhouette to display. If True, the default values are used
       and if False, no silhouette will be displayed. Defaults to False.
    theme : str | path-like
        Can be "auto" (default), "light", or "dark" or a path-like to a
        custom stylesheet. For Dark-Mode and automatic Dark-Mode-Detection,
        :mod:`qdarkstyle` respectively and `darkdetect
        <https://github.com/albertosottile/darkdetect>`__ is required.
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
       | add_head                  |              | ✓             |
       +---------------------------+--------------+---------------+
       | add_label                 | ✓            | ✓             |
       +---------------------------+--------------+---------------+
       | add_sensors               |              | ✓             |
       +---------------------------+--------------+---------------+
       | add_skull                 |              | ✓             |
       +---------------------------+--------------+---------------+
       | add_text                  | ✓            | ✓             |
       +---------------------------+--------------+---------------+
       | add_volume_labels         |              | ✓             |
       +---------------------------+--------------+---------------+
       | close                     | ✓            | ✓             |
       +---------------------------+--------------+---------------+
       | data                      | ✓            | ✓             |
       +---------------------------+--------------+---------------+
       | foci                      | ✓            |               |
       +---------------------------+--------------+---------------+
       | labels                    | ✓            | ✓             |
       +---------------------------+--------------+---------------+
       | remove_data               |              | ✓             |
       +---------------------------+--------------+---------------+
       | remove_foci               | ✓            |               |
       +---------------------------+--------------+---------------+
       | remove_head               |              | ✓             |
       +---------------------------+--------------+---------------+
       | remove_labels             | ✓            | ✓             |
       +---------------------------+--------------+---------------+
       | remove_annotations        | -            | ✓             |
       +---------------------------+--------------+---------------+
       | remove_sensors            |              | ✓             |
       +---------------------------+--------------+---------------+
       | remove_skull              |              | ✓             |
       +---------------------------+--------------+---------------+
       | remove_text               |              | ✓             |
       +---------------------------+--------------+---------------+
       | remove_volume_labels      |              | ✓             |
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

    def __init__(self, subject_id, hemi='both', surf='pial', title=None,
                 cortex="classic", alpha=1.0, size=800, background="black",
                 foreground=None, figure=None, subjects_dir=None,
                 views='auto', offset='auto', show_toolbar=False,
                 offscreen=False, interaction='trackball', units='mm',
                 view_layout='vertical', silhouette=False, theme='auto',
                 show=True):
        from ..backends.renderer import backend, _get_renderer

        if hemi is None:
            hemi = 'vol'
        hemi = self._check_hemi(hemi, extras=('both', 'split', 'vol'))
        if hemi in ('both', 'split'):
            self._hemis = ('lh', 'rh')
        else:
            assert hemi in ('lh', 'rh', 'vol')
            self._hemis = (hemi, )
        self._view_layout = _check_option('view_layout', view_layout,
                                          ('vertical', 'horizontal'))

        if figure is not None and not isinstance(figure, int):
            backend._check_3d_figure(figure)
        if title is None:
            self._title = subject_id
        else:
            self._title = title
        self._interaction = 'trackball'

        self._bg_color = _to_rgb(background, name='background')
        if foreground is None:
            foreground = 'w' if sum(self._bg_color) < 2 else 'k'
        self._fg_color = _to_rgb(foreground, name='foreground')
        del background, foreground
        views = _check_views(surf, views, hemi)
        col_dict = dict(lh=1, rh=1, both=1, split=2, vol=1)
        shape = (len(views), col_dict[hemi])
        if self._view_layout == 'horizontal':
            shape = shape[::-1]
        self._subplot_shape = shape

        size = tuple(np.atleast_1d(size).round(0).astype(int).flat)
        if len(size) not in (1, 2):
            raise ValueError('"size" parameter must be an int or length-2 '
                             'sequence of ints.')
        size = size if len(size) == 2 else size * 2  # 1-tuple to 2-tuple
        subjects_dir = get_subjects_dir(subjects_dir)

        self.theme = theme

        self.time_viewer = False
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
        self._unnamed_label_id = 0  # can only grow
        self._annots = {'lh': list(), 'rh': list()}
        self._layered_meshes = dict()
        self._actors = dict()
        self._elevation_rng = [15, 165]  # range of motion of camera on theta
        self._lut_locked = None
        self._cleaned = False
        # default values for silhouette
        self._silhouette = {
            'color': self._bg_color,
            'line_width': 2,
            'alpha': alpha,
            'decimate': 0.9,
        }
        _validate_type(silhouette, (dict, bool), 'silhouette')
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
        self.set_time_interpolation('nearest')

        geo_kwargs = self._cortex_colormap(cortex)
        # evaluate at the midpoint of the used colormap
        val = -geo_kwargs['vmin'] / (geo_kwargs['vmax'] - geo_kwargs['vmin'])
        self._brain_color = geo_kwargs['colormap'](val)

        # load geometry for one or both hemispheres as necessary
        _validate_type(offset, (str, bool), 'offset')
        if isinstance(offset, str):
            _check_option('offset', offset, ('auto',), extra='when str')
            offset = (surf in ('inflated', 'flat'))
        offset = None if (not offset or hemi != 'both') else 0.0
        logger.debug(f'Hemi offset: {offset}')

        self._renderer = _get_renderer(name=self._title, size=size,
                                       bgcolor=self._bg_color,
                                       shape=shape,
                                       fig=figure)
        self._renderer._window_close_connect(self._clean)
        self._renderer._window_set_theme(theme)
        self.plotter = self._renderer.plotter

        self._setup_canonical_rotation()

        # plot hemis
        for h in ('lh', 'rh'):
            if h not in self._hemis:
                continue  # don't make surface if not chosen
            # Initialize a Surface object as the geometry
            geo = _Surface(self._subject_id, h, surf, self._subjects_dir,
                           offset, units=self._units, x_dir=self._rigid[0, :3])
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
                self._renderer.set_camera(update=False, reset_camera=False,
                                          **views_dicts[h][v])

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

    def _setup_canonical_rotation(self):
        from ...coreg import fit_matched_points, _trans_from_params
        self._rigid = np.eye(4)
        try:
            xfm = read_talxfm(self._subject_id, self._subjects_dir)
        except Exception:
            return
        # XYZ+origin + halfway
        pts_tal = np.concatenate([np.eye(4)[:, :3], np.eye(3) * 0.5])
        pts_subj = apply_trans(invert_transform(xfm), pts_tal)
        # we fit with scaling enabled, but then discard it (we just need
        # the rigid-body components)
        params = fit_matched_points(pts_subj, pts_tal, scale=3, out='params')
        self._rigid[:] = _trans_from_params((True, True, False), params[:6])

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
        self.playback = False
        self.visibility = False
        self.refresh_rate_ms = max(int(round(1000. / 60.)), 1)
        self.default_scaling_range = [0.2, 2.0]
        self.default_playback_speed_range = [0.01, 1]
        self.default_playback_speed_value = 0.01
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
        self.help_canvas = None
        self.rms = None
        self.picked_patches = {key: list() for key in all_keys}
        self.picked_points = {key: list() for key in all_keys}
        self.pick_table = dict()
        self._spheres = list()
        self._mouse_no_mvt = -1
        self.callbacks = dict()
        self.widgets = dict()
        self.keys = ('fmin', 'fmid', 'fmax')

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

        self._configure_time_label()
        self._configure_scalar_bar()
        self._configure_shortcuts()
        self._configure_picking()
        self._configure_tool_bar()
        self._configure_dock()
        self._configure_menu()
        self._configure_status_bar()
        self._configure_playback()
        self._configure_help()
        # show everything at the end
        self.toggle_interface()
        self._renderer.show()

        # sizes could change, update views
        for hemi in ('lh', 'rh'):
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
        self.clear_glyphs()
        self.remove_annotations()
        # clear init actors
        for hemi in self._hemis:
            self._layered_meshes[hemi]._clean()
        self._clear_callbacks()
        self._clear_widgets()
        if getattr(self, 'mpl_canvas', None) is not None:
            self.mpl_canvas.clear()
        if getattr(self, 'act_data_smooth', None) is not None:
            for key in list(self.act_data_smooth.keys()):
                self.act_data_smooth[key] = None
        # XXX this should be done in PyVista
        for renderer in self._renderer._all_renderers:
            renderer.RemoveAllLights()
        # app_window cannot be set to None because it is used in __del__
        for key in ('lighting', 'interactor', '_RenderWindow'):
            setattr(self.plotter, key, None)
        # Qt LeaveEvent requires _Iren so we use _FakeIren instead of None
        # to resolve the ref to vtkGenericRenderWindowInteractor
        self.plotter._Iren = _FakeIren()
        if getattr(self.plotter, 'picker', None) is not None:
            self.plotter.picker = None
        # XXX end PyVista
        for key in ('plotter', 'window', 'dock', 'tool_bar', 'menu_bar',
                    'interactor', 'mpl_canvas', 'time_actor',
                    'picked_renderer', 'act_data_smooth', '_scalar_bar',
                    'actions', 'widgets', 'geo', '_data'):
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
                    name="visibility", icon_name="visibility_on")
            else:
                self._renderer._dock_hide()
                self._renderer._tool_bar_update_button_icon(
                    name="visibility", icon_name="visibility_off")

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
        if value is None:
            self.playback = not self.playback
        else:
            self.playback = value

        # update tool bar icon
        if self.playback:
            self._renderer._tool_bar_update_button_icon(
                name="play", icon_name="pause")
        else:
            self._renderer._tool_bar_update_button_icon(
                name="play", icon_name="play")

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
        self._renderer._update()

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

    def _configure_time_label(self):
        self.time_actor = self._data.get('time_actor')
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

    def _configure_dock_time_widget(self, layout=None):
        len_time = len(self._data['time']) - 1
        if len_time < 1:
            return
        layout = self._renderer.dock_layout if layout is None else layout
        hlayout = self._renderer._dock_add_layout(vertical=False)
        self.widgets["min_time"] = self._renderer._dock_add_label(
            value="-", layout=hlayout)
        self._renderer._dock_add_stretch(hlayout)
        self.widgets["current_time"] = self._renderer._dock_add_label(
            value="x", layout=hlayout)
        self._renderer._dock_add_stretch(hlayout)
        self.widgets["max_time"] = self._renderer._dock_add_label(
            value="+", layout=hlayout)
        self._renderer._layout_add_widget(layout, hlayout)
        min_time = float(self._data['time'][0])
        max_time = float(self._data['time'][-1])
        self.widgets["min_time"].set_value(f"{min_time: .3f}")
        self.widgets["max_time"].set_value(f"{max_time: .3f}")
        self.widgets["current_time"].set_value(f"{self._current_time: .3f}")

    def _configure_dock_playback_widget(self, name):
        layout = self._renderer._dock_add_group_box(name)
        len_time = len(self._data['time']) - 1

        # Time widget
        if len_time < 1:
            self.callbacks["time"] = None
            self.widgets["time"] = None
        else:
            self.callbacks["time"] = TimeCallBack(
                brain=self,
                callback=self.plot_time_line,
            )
            self.widgets["time"] = self._renderer._dock_add_slider(
                name="Time (s)",
                value=self._data['time_idx'],
                rng=[0, len_time],
                double=True,
                callback=self.callbacks["time"],
                compact=False,
                layout=layout,
            )
            self.callbacks["time"].widget = self.widgets["time"]

        # Time labels
        if len_time < 1:
            self.widgets["min_time"] = None
            self.widgets["max_time"] = None
            self.widgets["current_time"] = None
        else:
            self._configure_dock_time_widget(layout)
            self.callbacks["time"].label = self.widgets["current_time"]

        # Playback speed widget
        if len_time < 1:
            self.callbacks["playback_speed"] = None
            self.widgets["playback_speed"] = None
        else:
            self.callbacks["playback_speed"] = SmartCallBack(
                callback=self.set_playback_speed,
            )
            self.widgets["playback_speed"] = self._renderer._dock_add_spin_box(
                name="Speed",
                value=self.default_playback_speed_value,
                rng=self.default_playback_speed_range,
                callback=self.callbacks["playback_speed"],
                layout=layout,
            )
            self.callbacks["playback_speed"].widget = \
                self.widgets["playback_speed"]

        # Time label
        current_time = self._current_time
        assert current_time is not None  # should never be the case, float
        time_label = self._data['time_label']
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
            def select_renderer(idx):
                idx = int(idx)
                loc = self._renderer._index_to_loc(idx)
                self.plotter.subplot(*loc)

            self.callbacks["renderer"] = SmartCallBack(
                callback=select_renderer,
            )
            self.widgets["renderer"] = self._renderer._dock_add_combo_box(
                name="Renderer",
                value="0",
                rng=rends,
                callback=self.callbacks["renderer"],
                layout=layout,
            )
            self.callbacks["renderer"].widget = \
                self.widgets["renderer"]

        # Use 'lh' as a reference for orientation for 'both'
        if self._hemi == 'both':
            hemis_ref = ['lh']
        else:
            hemis_ref = self._hemis
        orientation_data = [None] * len(rends)
        for hemi in hemis_ref:
            for ri, ci, v in self._iter_views(hemi):
                idx = self._renderer._loc_to_index((ri, ci))
                if v == 'flat':
                    _data = None
                else:
                    _data = dict(default=v, hemi=hemi, row=ri, col=ci)
                orientation_data[idx] = _data
        self.callbacks["orientation"] = ShowView(
            brain=self,
            data=orientation_data,
        )
        self.widgets["orientation"] = self._renderer._dock_add_combo_box(
            name=None,
            value=self.orientation[0],
            rng=self.orientation,
            callback=self.callbacks["orientation"],
            layout=layout,
        )

    def _configure_dock_colormap_widget(self, name):
        layout = self._renderer._dock_add_group_box(name)
        self._renderer._dock_add_label(
            value="min / mid / max",
            align=True,
            layout=layout,
        )
        up = UpdateLUT(brain=self)
        for key in self.keys:
            hlayout = self._renderer._dock_add_layout(vertical=False)
            rng = _get_range(self)
            self.callbacks[key] = lambda value, key=key: up(**{key: value})
            self.widgets[key] = self._renderer._dock_add_slider(
                name=None,
                value=self._data[key],
                rng=rng,
                callback=self.callbacks[key],
                double=True,
                layout=hlayout,
            )
            self.widgets[f"entry_{key}"] = self._renderer._dock_add_spin_box(
                name=None,
                value=self._data[key],
                callback=self.callbacks[key],
                rng=rng,
                layout=hlayout,
            )
            up.widgets[key] = [self.widgets[key], self.widgets[f"entry_{key}"]]
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
        )
        for key, char, val in (("fminus", "➖", 1.2 ** -0.25),
                               ("fplus", "➕", 1.2 ** 0.25)):
            self.callbacks[key] = UpdateColorbarScale(
                brain=self,
                factor=val,
            )
            self.widgets[key] = self._renderer._dock_add_button(
                name=char,
                callback=self.callbacks[key],
                layout=hlayout,
            )
        self._renderer._layout_add_widget(layout, hlayout)

        # register colorbar slider representations
        widgets = {key: self.widgets[key] for key in self.keys}
        for name in ("fmin", "fmid", "fmax", "fminus", "fplus"):
            self.callbacks[name].widgets = widgets

    def _configure_dock_trace_widget(self, name):
        if not self.show_traces:
            return
        # do not show trace mode for volumes
        if (self._data.get('src', None) is not None and
                self._data['src'].kind == 'volume'):
            self._configure_vertex_time_course()
            return

        layout = self._renderer._dock_add_group_box(name)

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
            self._renderer._update()

        # setup label extraction parameters
        def _set_label_mode(mode):
            if self.traces_mode != 'label':
                return
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
            self._renderer._update()

        from ...source_estimate import _get_allowed_label_modes
        from ...label import _read_annot_cands
        dir_name = op.join(self._subjects_dir, self._subject_id, 'label')
        cands = _read_annot_cands(dir_name, raise_error=False)
        cands = cands + ['None']
        self.annot = cands[0]
        stc = self._data["stc"]
        modes = _get_allowed_label_modes(stc)
        if self._data["src"] is None:
            modes = [m for m in modes if m not in
                     self.default_label_extract_modes["src"]]
        self.label_extract_mode = modes[-1]
        if self.traces_mode == 'vertex':
            _set_annot('None')
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
        self.callbacks["smoothing"] = SmartCallBack(
            callback=self.set_data_smoothing,
        )
        self.widgets["smoothing"] = self._renderer._dock_add_spin_box(
            name="Smoothing",
            value=self._data['smoothing_steps'],
            rng=self.default_smoothing_range,
            callback=self.callbacks["smoothing"],
            double=False
        )
        self.callbacks["smoothing"].widget = \
            self.widgets["smoothing"]

        self._renderer._dock_finalize()

    def _configure_playback(self):
        self._renderer._playback_initialize(
            func=self._play,
            timeout=self.refresh_rate_ms,
            value=self._data['time_idx'],
            rng=[0, len(self._data['time']) - 1],
            time_widget=self.widgets["time"],
            play_widget=self.widgets["play"],
        )

    def _configure_mplcanvas(self):
        # Get the fractional components for the brain and mpl
        self.mpl_canvas = self._renderer._window_get_mplcanvas(
            brain=self,
            interactor_fraction=self.interactor_fraction,
            show_traces=self.show_traces,
            separate_canvas=self.separate_canvas
        )
        xlim = [np.min(self._data['time']),
                np.max(self._data['time'])]
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
        y = np.concatenate(list(v[0] for v in self.act_data_smooth.values()
                                if v[0] is not None))
        rms = np.linalg.norm(y, axis=0) / np.sqrt(len(y))
        del y

        self.rms, = self.mpl_canvas.axes.plot(
            self._data['time'], rms,
            lw=3, label='RMS', zorder=3, color=self._fg_color,
            alpha=0.5, ls=':')

        # now plot the time line
        self.plot_time_line(update=False)

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
            self.picked_renderer = self._renderer._all_renderers[idx]

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
            self._add_vertex_glyph(hemi, mesh, vertex_id, update=False)

    def _configure_picking(self):
        # get data for each hemi
        from scipy import sparse
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

        self._renderer._update_picking_callback(
            self._on_mouse_move,
            self._on_button_press,
            self._on_button_release,
            self._on_pick
        )

    def _configure_tool_bar(self):
        self._renderer._tool_bar_load_icons()
        self._renderer._tool_bar_set_theme(self.theme)
        self._renderer._tool_bar_initialize(name="Toolbar")
        self._renderer._tool_bar_add_file_button(
            name="screenshot",
            desc="Take a screenshot",
            func=self.save_image,
        )
        self._renderer._tool_bar_add_file_button(
            name="movie",
            desc="Save movie...",
            func=lambda filename: self.save_movie(
                filename=filename,
                time_dilation=(1. / self.playback_speed)),
            shortcut="ctrl+shift+s",
        )
        self._renderer._tool_bar_add_button(
            name="visibility",
            desc="Toggle Controls",
            func=self.toggle_interface,
            icon_name="visibility_on"
        )
        self.widgets["play"] = self._renderer._tool_bar_add_play_button(
            name="play",
            desc="Play/Pause",
            func=self.toggle_playback,
            shortcut=" ",
        )
        self._renderer._tool_bar_add_button(
            name="reset",
            desc="Reset",
            func=self.reset,
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

    def _shift_time(self, op):
        self.callbacks["time"](
            value=(op(self._current_time, self.playback_speed)),
            time_as_index=False,
            update_widget=True,
        )

    def _rotate_azimuth(self, value):
        azimuth = (self._renderer.figure._azimuth + value) % 360
        self._renderer.set_camera(azimuth=azimuth, reset_camera=False)

    def _rotate_elevation(self, value):
        elevation = np.clip(
            self._renderer.figure._elevation + value,
            self._elevation_rng[0],
            self._elevation_rng[1],
        )
        self._renderer.set_camera(elevation=elevation, reset_camera=False)

    def _configure_shortcuts(self):
        # First, we remove the default bindings:
        self._clear_callbacks()
        # Then, we add our own:
        self.plotter.add_key_event("i", self.toggle_interface)
        self.plotter.add_key_event("s", self.apply_auto_scaling)
        self.plotter.add_key_event("r", self.restore_user_scaling)
        self.plotter.add_key_event("c", self.clear_glyphs)
        self.plotter.add_key_event("n", partial(self._shift_time,
                                   op=lambda x, y: x + y))
        self.plotter.add_key_event("b", partial(self._shift_time,
                                   op=lambda x, y: x - y))
        for key, func, sign in (("Left", self._rotate_azimuth, 1),
                                ("Right", self._rotate_azimuth, -1),
                                ("Up", self._rotate_elevation, 1),
                                ("Down", self._rotate_elevation, -1)):
            self.plotter.add_key_event(key, partial(func, sign * _ARROW_MOVE))

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
            self.default_status_bar_msg, stretch=1)
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
            try:
                # pyvista<0.30.0
                self.picked_renderer = \
                    self.plotter.iren.FindPokedRenderer(x, y)
            except AttributeError:
                # pyvista>=0.30.0
                self.picked_renderer = \
                    self.plotter.iren.interactor.FindPokedRenderer(x, y)
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
            scalars = _cell_data(grid)['values'][vertices]
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
            # _cell_data(grid)['values'][vertices] = dists * mask
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

    def _add_vertex_glyph(self, hemi, mesh, vertex_id, update=True):
        if vertex_id in self.picked_points[hemi]:
            return

        # skip if the wrong hemi is selected
        if self.act_data_smooth[hemi][0] is None:
            return
        color = next(self.color_cycle)
        line = self.plot_time_course(hemi, vertex_id, color, update=update)
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
        try:
            lst = self._renderer._all_renderers._renderers
        except AttributeError:
            lst = self._renderer._all_renderers
        rindex = lst.index(self.picked_renderer)
        row, col = self._renderer._index_to_loc(rindex)

        actors = list()
        spheres = list()
        for _ in self._iter_views(hemi):
            # Using _sphere() instead of renderer.sphere() for 2 reasons:
            # 1) renderer.sphere() fails on Windows in a scenario where a lot
            #    of picking requests are done in a short span of time (could be
            #    mitigated with synchronization/delay?)
            # 2) the glyph filter is used in renderer.sphere() but only one
            #    sphere is required in this function.
            actor, sphere = self._renderer._sphere(
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
        if self.rms is not None:
            self.rms.remove()
            self.rms = None
        self._renderer._update()

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
        time = self._data['time'].copy()  # avoid circular ref
        mni = None
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
            try:
                mni = vertex_to_mni(
                    vertices=vertex_id,
                    hemis=0 if hemi == 'lh' else 1,
                    subject=self._subject_id,
                    subjects_dir=self._subjects_dir
                )
            except Exception:
                mni = None
        if mni is not None:
            mni = ' MNI: ' + ', '.join('%5.1f' % m for m in mni)
        else:
            mni = ''
        label = "{}:{}{}".format(hemi_str, str(vertex_id).ljust(6), mni)
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
            update=update,
        )
        return line

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
                    label='time',
                    color=self._fg_color,
                    lw=1,
                    update=update,
                )
            self.time_line.set_xdata(current_time)
            if update:
                self.mpl_canvas.update_plot()

    def _configure_help(self):
        pairs = [
            ('?', 'Display help window'),
            ('i', 'Toggle interface'),
            ('s', 'Apply auto-scaling'),
            ('r', 'Restore original clim'),
            ('c', 'Clear all traces'),
            ('n', 'Shift the time forward by the playback speed'),
            ('b', 'Shift the time backward by the playback speed'),
            ('Space', 'Start/Pause playback'),
            ('Up', 'Decrease camera elevation angle'),
            ('Down', 'Increase camera elevation angle'),
            ('Left', 'Decrease camera azimuth angle'),
            ('Right', 'Increase camera azimuth angle'),
        ]
        text1, text2 = zip(*pairs)
        text1 = '\n'.join(text1)
        text2 = '\n'.join(text2)
        self.help_canvas = self._renderer._window_get_simple_canvas(
            width=5, height=2, dpi=80)
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
        if not hasattr(self, 'callbacks'):
            return
        for callback in self.callbacks.values():
            if callback is not None:
                for key in ('plotter', 'brain', 'callback',
                            'widget', 'widgets'):
                    setattr(callback, key, None)
        self.callbacks.clear()
        # Remove the default key binding
        if getattr(self, "iren", None) is not None:
            self.plotter.iren.clear_key_event_callbacks()

    def _clear_widgets(self):
        if not hasattr(self, 'widgets'):
            return
        for widget in self.widgets.values():
            if widget is not None:
                for key in ('triggered', 'valueChanged'):
                    setattr(widget, key, None)
        self.widgets.clear()

    @property
    def interaction(self):
        """The interaction style."""
        return self._interaction

    @interaction.setter
    def interaction(self, interaction):
        """Set the interaction style."""
        _validate_type(interaction, str, 'interaction')
        _check_option('interaction', interaction, ('trackball', 'terrain'))
        for _ in self._iter_views('vol'):  # will traverse all
            self._renderer.set_interaction(interaction)

    def _cortex_colormap(self, cortex):
        """Return the colormap corresponding to the cortex."""
        from .._3d import _get_cmap
        from matplotlib.colors import ListedColormap
        colormap_map = dict(classic=dict(colormap="Greys",
                                         vmin=-1, vmax=2),
                            high_contrast=dict(colormap="Greys",
                                               vmin=-.1, vmax=1.3),
                            low_contrast=dict(colormap="Greys",
                                              vmin=-5, vmax=5),
                            bone=dict(colormap="bone_r",
                                      vmin=-.2, vmax=2),
                            )
        _validate_type(cortex, (str, dict, list, tuple), 'cortex')
        if isinstance(cortex, str):
            if cortex in colormap_map:
                cortex = colormap_map[cortex]
            else:
                cortex = [cortex] * 2
        if isinstance(cortex, (list, tuple)):
            _check_option('len(cortex)', len(cortex), (2, 3),
                          extra='when cortex is a list or tuple')
            if len(cortex) == 3:
                cortex = [cortex] * 2
            cortex = list(cortex)
            for ci, c in enumerate(cortex):
                cortex[ci] = _to_rgb(c, name='cortex')
            cortex = dict(
                colormap=ListedColormap(cortex, name='custom binary'),
                vmin=0, vmax=1)
        cortex = dict(
            vmin=float(cortex['vmin']),
            vmax=float(cortex['vmax']),
            colormap=_get_cmap(cortex['colormap']),
        )
        return cortex

    def _remove(self, item, render=False):
        """Remove actors from the rendered scene."""
        if item in self._actors:
            logger.debug(
                f'Removing {len(self._actors[item])} {item} actor(s)')
            for actor in self._actors[item]:
                self._renderer.plotter.remove_actor(actor)
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
            Options to pass to :meth:`pyvista.Plotter.add_scalar_bar`
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
            smoothing_steps = -1
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
        for _ in self._iter_views(hemi):
            if hemi in ('lh', 'rh'):
                actor = self._layered_meshes[hemi]._actor
            else:
                src_vol = src[2:] if src.kind == 'mixed' else src
                actor, _ = self._add_volume_data(hemi, src_vol, volume_options)
        assert actor is not None  # should have added one
        self._add_actor('data', actor)

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
            if colorbar and self._scalar_bar is None and do:
                kwargs = dict(source=actor, n_labels=8, color=self._fg_color,
                              bgcolor=self._brain_color[:3])
                kwargs.update(colorbar_kwargs or {})
                self._scalar_bar = self._renderer.scalarbar(**kwargs)
            self._renderer.set_camera(
                update=False, reset_camera=False, **views_dicts[hemi][v])

        # 4) update the scalar bar and opacity
        self.update_lut(alpha=alpha)

    def remove_data(self):
        """Remove rendered data from the mesh."""
        self._remove('data', render=True)

    def _iter_views(self, hemi):
        """Iterate over rows and columns that need to be added to."""
        hemi_dict = dict(lh=[0], rh=[0], vol=[0])
        if self._hemi == 'split':
            hemi_dict.update(rh=[1], vol=[0, 1])
        for vi, view in enumerate(self._views):
            view_dict = dict(lh=[vi], rh=[vi], vol=[vi])
            if self._hemi == 'split':
                view_dict.update(vol=[vi, vi])
            if self._view_layout == 'vertical':
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
            mesh = self._layered_meshes[hemi]
            mesh.remove_overlay(self._annots[hemi])
            self._annots[hemi].clear()
        self._renderer._update()

    def _add_volume_data(self, hemi, src, volume_options):
        from ..backends._pyvista import _hide_testing_actor
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
                self._renderer._volume(dimensions, origin, spacing, scalars,
                                       surface_alpha, resolution, blending,
                                       center)
            self._data[hemi]['alpha'] = alpha  # incorrectly set earlier
            self._data[hemi]['grid'] = grid
            self._data[hemi]['grid_mesh'] = grid_mesh
            self._data[hemi]['grid_coords'] = coords
            self._data[hemi]['grid_src_mri_t'] = src_mri_t
            self._data[hemi]['grid_shape'] = dimensions
            self._data[hemi]['grid_volume_pos'] = volume_pos
            self._data[hemi]['grid_volume_neg'] = volume_neg
        actor_pos, _ = self._renderer.plotter.add_actor(
            volume_pos, reset_camera=False, name=None, culling=False,
            render=False)
        actor_neg = actor_mesh = None
        if volume_neg is not None:
            actor_neg, _ = self._renderer.plotter.add_actor(
                volume_neg, reset_camera=False, name=None, culling=False,
                render=False)
        grid_mesh = self._data[hemi]['grid_mesh']
        if grid_mesh is not None:
            actor_mesh, prop = self._renderer.plotter.add_actor(
                grid_mesh, reset_camera=False, name=None, culling=False,
                pickable=False, render=False)
            prop.SetColor(*self._brain_color[:3])
            prop.SetOpacity(surface_alpha)
            if silhouette_alpha > 0 and silhouette_linewidth > 0:
                for _ in self._iter_views('vol'):
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
                    label.name = 'unnamed' + str(self._unnamed_label_id)
                    self._unnamed_label_id += 1
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

        if self.time_viewer and self.show_traces \
                and self.traces_mode == 'label':
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
        color = _to_rgb(color, alpha, alpha=True)
        cmap = np.array([(0, 0, 0, 0,), color])
        ctable = np.round(cmap * 255).astype(np.uint8)

        scalars = np.zeros(self.geo[hemi].coords.shape[0])
        scalars[ids] = 1
        if borders:
            keep_idx = _mesh_borders(self.geo[hemi].faces, scalars)
            show = np.zeros(scalars.size, dtype=np.int64)
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
        for _, _, v in self._iter_views(hemi):
            mesh = self._layered_meshes[hemi]
            mesh.add_overlay(
                scalars=scalars,
                colormap=ctable,
                rng=[np.min(scalars), np.max(scalars)],
                opacity=alpha,
                name=label_name,
            )
            if reset_camera:
                self._renderer.set_camera(update=False, **views_dicts[hemi][v])
            if self.time_viewer and self.show_traces \
                    and self.traces_mode == 'label':
                label._color = orig_color
                label._line = line
            self._labels[hemi].append(label)
        self._renderer._update()

    @fill_doc
    def add_head(self, dense=True, color='gray', alpha=0.5):
        """Add a mesh to render the outer head surface.

        Parameters
        ----------
        dense : bool
            Whether to plot the dense head (``seghead``) or the less dense head
            (``head``).
        color : color
            A list of anything matplotlib accepts: string, RGB, hex, etc.
        alpha : float in [0, 1]
            Alpha level to control opacity.

        Notes
        -----
        .. versionadded:: 0.24
        """
        # load head
        surf = _get_head_surface('seghead' if dense else 'head',
                                 self._subject_id, self._subjects_dir)
        verts, triangles = surf['rr'], surf['tris']
        verts *= 1e3 if self._units == 'mm' else 1
        color = _to_rgb(color, alpha, alpha=True)

        for _ in self._iter_views('vol'):
            actor, _ = self._renderer.mesh(
                *verts.T, triangles=triangles, color=color,
                opacity=alpha, reset_camera=False, render=False)
            self._add_actor('head', actor)

        self._renderer._update()

    def remove_head(self):
        """Remove head objects from the rendered scene."""
        self._remove('head', render=True)

    @fill_doc
    def add_skull(self, outer=True, color='gray', alpha=0.5):
        """Add a mesh to render the skull surface.

        Parameters
        ----------
        outer : bool
            Adds the outer skull if ``True``, otherwise adds the inner skull.
        color : color
            A list of anything matplotlib accepts: string, RGB, hex, etc.
        alpha : float in [0, 1]
            Alpha level to control opacity.

        Notes
        -----
        .. versionadded:: 0.24
        """
        surf = _get_skull_surface('outer' if outer else 'inner',
                                  self._subject_id, self._subjects_dir)
        verts, triangles = surf['rr'], surf['tris']
        verts *= 1e3 if self._units == 'mm' else 1
        color = _to_rgb(color, alpha, alpha=True)

        for _ in self._iter_views('vol'):
            actor, _ = self._renderer.mesh(
                *verts.T, triangles=triangles, color=color,
                opacity=alpha, reset_camera=False, render=False)
            self._add_actor('skull', actor)

        self._renderer._update()

    def remove_skull(self):
        """Remove skull objects from the rendered scene."""
        self._remove('skull', render=True)

    @fill_doc
    def add_volume_labels(self, aseg='aparc+aseg', labels=None, colors=None,
                          alpha=0.5, smooth=0.9, legend=None):
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
        alpha : float in [0, 1]
            Alpha level to control opacity.
        %(smooth)s
        legend : bool | None | dict
            Add a legend displaying the names of the ``labels``. Default (None)
            is ``True`` if the number of ``labels`` is 10 or fewer.
            Can also be a dict of ``kwargs`` to pass to
            :meth:`pyvista.Plotter.add_legend`.

        Notes
        -----
        .. versionadded:: 0.24
        """
        import nibabel as nib

        # load anatomical segmentation image
        if not aseg.endswith('aseg'):
            raise RuntimeError(
                f'`aseg` file path must end with "aseg", got {aseg}')
        aseg = _check_fname(op.join(self._subjects_dir, self._subject_id,
                                    'mri', aseg + '.mgz'),
                            overwrite='read', must_exist=True)
        aseg_fname = aseg
        aseg = nib.load(aseg_fname)
        aseg_data = np.asarray(aseg.dataobj)
        vox_mri_t = aseg.header.get_vox2ras_tkr()
        mult = 1e-3 if self._units == 'm' else 1
        vox_mri_t[:3] *= mult
        del aseg

        # read freesurfer lookup table
        lut, fs_colors = read_freesurfer_lut()
        if labels is None:  # assign default ROI labels based on indices
            lut_r = {v: k for k, v in lut.items()}
            labels = [lut_r[idx] for idx in DEFAULTS['volume_label_indices']]

        _validate_type(legend, (bool, None), 'legend')
        if legend is None:
            legend = len(labels) < 11

        if colors is None:
            colors = [fs_colors[label] / 255 for label in labels]
        elif not isinstance(colors, (list, tuple)):
            colors = [colors] * len(labels)  # make into list
        colors = [_to_rgb(color, alpha, name=f'colors[{ci}]', alpha=True)
                  for ci, color in enumerate(colors)]
        surfs = _marching_cubes(
            aseg_data, [lut[label] for label in labels], smooth=smooth)
        for label, color, (verts, triangles) in zip(labels, colors, surfs):
            if len(verts) == 0:  # not in aseg vals
                warn(f'Value {lut[label]} not found for label '
                     f'{repr(label)} in: {aseg_fname}')
                continue
            verts = apply_trans(vox_mri_t, verts)
            for _ in self._iter_views('vol'):
                actor, _ = self._renderer.mesh(
                    *verts.T, triangles=triangles, color=color,
                    opacity=alpha, reset_camera=False, render=False)
                self._add_actor('volume_labels', actor)

        if legend or isinstance(legend, dict):
            # use empty kwargs for legend = True
            legend = legend if isinstance(legend, dict) else dict()
            self._renderer.plotter.add_legend(
                list(zip(labels, colors)), **legend)

        self._renderer._update()

    def remove_volume_labels(self):
        """Remove the volume labels from the rendered scene."""
        self._remove('volume_labels', render=True)
        self._renderer.plotter.remove_legend()

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
        hemi = self._check_hemi(hemi, extras=['vol'])

        # those parameters are not supported yet, only None is allowed
        _check_option('map_surface', map_surface, [None])

        # Figure out how to interpret the first parameter
        if coords_as_verts:
            coords = self.geo[hemi].coords[coords]

        # Convert the color code
        color = _to_rgb(color)

        if self._units == 'm':
            scale_factor = scale_factor / 1000.
        for _, _, v in self._iter_views(hemi):
            self._renderer.sphere(center=coords, color=color,
                                  scale=(10. * scale_factor),
                                  opacity=alpha, resolution=resolution)
            self._renderer.set_camera(**views_dicts[hemi][v])

    @verbose
    def add_sensors(self, info, trans, meg=None, eeg='original', fnirs=True,
                    ecog=True, seeg=True, dbs=True, verbose=None):
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
        %(verbose)s

        Notes
        -----
        .. versionadded:: 0.24
        """
        _validate_type(info, Info, 'info')
        meg, eeg, fnirs, warn_meg = _handle_sensor_types(meg, eeg, fnirs)
        picks = pick_types(info, meg=('sensors' in meg),
                           ref_meg=('ref' in meg), eeg=(len(eeg) > 0),
                           ecog=ecog, seeg=seeg, dbs=dbs,
                           fnirs=(len(fnirs) > 0))
        head_mri_t = _get_trans(trans, 'head', 'mri', allow_none=False)[0]
        del trans
        # get transforms to "mri"window
        to_cf_t = _get_transforms_to_coord_frame(
            info, head_mri_t, coord_frame='mri')
        if pick_types(info, eeg=True, exclude=()).size > 0 and \
                'projected' in eeg:
            head_surf = _get_head_surface(
                'seghead', self._subject_id, self._subjects_dir)
        else:
            head_surf = None
        # Do the main plotting
        for _ in self._iter_views('vol'):
            if picks.size > 0:
                sensors_actors = _plot_sensors(
                    self._renderer, info, to_cf_t, picks, meg, eeg,
                    fnirs, warn_meg, head_surf, self._units)
                for item, actors in sensors_actors.items():
                    for actor in actors:
                        self._add_actor(item, actor)

            if 'helmet' in meg and pick_types(info, meg=True).size > 0:
                surf = get_meg_helmet_surf(info, head_mri_t)
                verts = surf['rr'] * (1 if self._units == 'm' else 1e3)
                actor, _ = self._renderer.mesh(
                    *verts.T, surf['tris'],
                    color=DEFAULTS['coreg']['helmet_color'],
                    opacity=0.25, reset_camera=False, render=False)
                self._add_actor('helmet', actor)

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
        all_kinds = ('meg', 'eeg', 'fnirs', 'ecog', 'seeg', 'dbs', 'helmet')
        if kind is None:
            for item in all_kinds:
                self._remove(item, render=False)
        else:
            if isinstance(kind, str):
                kind = [kind]
            for this_kind in kind:
                _check_option('kind', this_kind, all_kinds)
            self._remove(this_kind, render=False)
        self._renderer._update()

    def add_text(self, x, y, text, name=None, color=None, opacity=1.0,
                 row=0, col=0, font_size=None, justification=None):
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
        _validate_type(name, (str, None), 'name')
        name = text if name is None else name
        if 'text' in self._actors and name in self._actors['text']:
            raise ValueError(f'Text with the name {name} already exists')
        for ri, ci, _ in self._iter_views('vol'):
            if (row is None or row == ri) and (col is None or col == ci):
                actor = self._renderer.text2d(
                    x_window=x, y_window=y, text=text, color=color,
                    size=font_size, justification=justification)
                if 'text' not in self._actors:
                    self._actors['text'] = dict()
                self._actors['text'][name] = actor

    def remove_text(self, name=None):
        """Remove text from the rendered scene.

        Parameters
        ----------
        name : str | None
            Remove specific text by name. If None, all text will be removed.
        """
        _validate_type(name, (str, None), 'name')
        if name is None:
            for actor in self._actors['text'].values():
                self._renderer.plotter.remove_actor(actor)
            self._actors.pop('text')
        else:
            names = [None]
            if 'text' in self._actors:
                names += list(self._actors['text'].keys())
            _check_option('name', name, names)
            self._renderer.plotter.remove_actor(
                self._actors['text'][name])
            self._actors['text'].pop(name)
        self._renderer._update()

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
        self.plot_time_line(update=False)
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
        alpha : float
            Opacity of the head surface. Must be between 0 and 1 (inclusive).
            Default is 0.5.
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
                if not self.time_viewer or self.traces_mode == 'vertex':
                    self._renderer._set_colormap_range(
                        mesh._actor, cmap.astype(np.uint8), None)

        self._renderer._update()

    def close(self):
        """Close all figures and cleanup data structure."""
        self._closed = True
        self._renderer.close()

    def show(self):
        """Display the window."""
        self._renderer.show()

    @fill_doc
    def show_view(self, view=None, roll=None, distance=None,
                  row='deprecated', col='deprecated', hemi=None, align=True,
                  azimuth=None, elevation=None, focalpoint=None):
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
        align : bool
            If True, consider view arguments relative to canonical MRI
            directions (closest to MNI for the subject) rather than native MRI
            space. This helps when MRIs are not in standard orientation (e.g.,
            have large rotations).
        %(azimuth)s
        %(elevation)s
        %(focalpoint)s
        """
        hemi = self._hemi if hemi is None else hemi
        if hemi == 'split':
            if (self._view_layout == 'vertical' and col == 1 or
                    self._view_layout == 'horizontal' and row == 1):
                hemi = 'rh'
            else:
                hemi = 'lh'
        if isinstance(view, dict):  # deprecate at version 0.25
            warn('`view` is a dict is deprecated, please use `azimuth` and '
                 '`elevation` as arguments directly to `show_view`',
                 DeprecationWarning)
            if azimuth is None and 'azimuth' in view:
                azimuth = view['azimuth']
            if elevation is None and 'elevation' in view:
                elevation = view['elevation']
            view = None
        if (row == 'deprecated' or col == 'deprecated') and \
                len(set([_ for h in self._hemis
                         for _ in self._iter_views(h)])) > 1:
            warn('`row` and `col` default behavior is changing, in version '
                 '0.25 the default behavior will be to apply `show_view` to '
                 'all views', DeprecationWarning)
        if row == 'deprecated':
            row = None
        if col == 'deprecated':
            col = None
        view_params = dict(azimuth=azimuth, elevation=elevation, roll=roll,
                           distance=distance, focalpoint=focalpoint)
        if view is not None:  # view string takes precedence
            view_params = {param: val for param, val in view_params.items()
                           if val is not None}  # no overwriting with None
            view_params = dict(views_dicts[hemi].get(view), **view_params)
        xfm = self._rigid if align else None
        for h in self._hemis:
            for ri, ci, _ in self._iter_views(h):
                if (row is None or row == ri) and (col is None or col == ci):
                    self._renderer.set_camera(
                        **view_params, reset_camera=False, rigid=xfm)
        self._renderer._update()

    def reset_view(self):
        """Reset the camera."""
        for h in self._hemis:
            for _, _, v in self._iter_views(h):
                self._renderer.set_camera(**views_dicts[h][v],
                                          reset_camera=False)

    def save_image(self, filename=None, mode='rgb'):
        """Save view from all panels to disk.

        Parameters
        ----------
        filename : str
            Path to new image file.
        mode : str
            Either 'rgb' or 'rgba' for values to return.
        """
        if filename is None:
            filename = _generate_default_filename(".png")
        _save_ndarray_img(
            filename, self.screenshot(mode=mode, time_viewer=True))

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
        n_channels = 3 if mode == 'rgb' else 4
        img = self._renderer.screenshot(mode)
        logger.debug(f'Got screenshot of size {img.shape}')
        if time_viewer and self.time_viewer and \
                self.show_traces and \
                not self.separate_canvas:
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
                    f'Saving figure of size {size_in} @ {dpi} DPI '
                    f'({want_size} = {n_pix} pixels)')
                # Sometimes there can be off-by-one errors here (e.g.,
                # if in mpl int() rather than int(round()) is used to
                # compute the number of pixels) so rather than use "raw"
                # format and try to reshape ourselves, just write to PNG
                # and read it, which has the dimensions encoded for us.
                fig.savefig(output, dpi=dpi, format='png',
                            facecolor=self._bg_color, edgecolor='none')
                output.seek(0)
                trace_img = imread(output, format='png')[:, :, :n_channels]
                trace_img = np.clip(
                    np.round(trace_img * 255), 0, 255).astype(np.uint8)
            bgcolor = np.array(self._brain_color[:n_channels]) / 255
            img = concatenate_images([img, trace_img], bgcolor=bgcolor,
                                     n_channels=n_channels)
        return img

    @contextlib.contextmanager
    def _no_lut_update(self, why):
        orig = self._lut_locked
        self._lut_locked = why
        try:
            yield
        finally:
            self._lut_locked = orig

    @fill_doc
    def update_lut(self, fmin=None, fmid=None, fmax=None, alpha=None):
        """Update color map.

        Parameters
        ----------
        %(fmin_fmid_fmax)s
        alpha : float | None
            Alpha to use in the update.
        """
        args = f'{fmin}, {fmid}, {fmax}, {alpha}'
        if self._lut_locked is not None:
            logger.debug(f'LUT update postponed with {args}')
            return
        logger.debug(f'Updating LUT with {args}')
        center = self._data['center']
        colormap = self._data['colormap']
        transparent = self._data['transparent']
        lims = {key: self._data[key] for key in ('fmin', 'fmid', 'fmax')}
        _update_monotonic(lims, fmin=fmin, fmid=fmid, fmax=fmax)
        assert all(val is not None for val in lims.values())

        self._data.update(lims)
        self._data['ctable'] = np.round(
            calculate_lut(colormap, alpha=1., center=center,
                          transparent=transparent, **lims) *
            255).astype(np.uint8)
        # update our values
        rng = self._cmap_range
        ctable = self._data['ctable']
        for hemi in ['lh', 'rh', 'vol']:
            hemi_data = self._data.get(hemi)
            if hemi_data is not None:
                if hemi in self._layered_meshes:
                    mesh = self._layered_meshes[hemi]
                    mesh.update_overlay(name='data',
                                        colormap=self._data['ctable'],
                                        opacity=alpha,
                                        rng=rng)
                    self._renderer._set_colormap_range(
                        mesh._actor, ctable, self._scalar_bar, rng,
                        self._brain_color)

                grid_volume_pos = hemi_data.get('grid_volume_pos')
                grid_volume_neg = hemi_data.get('grid_volume_neg')
                for grid_volume in (grid_volume_pos, grid_volume_neg):
                    if grid_volume is not None:
                        self._renderer._set_volume_range(
                            grid_volume, ctable, hemi_data['alpha'],
                            self._scalar_bar, rng)

                glyph_actor = hemi_data.get('glyph_actor')
                if glyph_actor is not None:
                    for glyph_actor_ in glyph_actor:
                        self._renderer._set_colormap_range(
                            glyph_actor_, ctable, self._scalar_bar, rng)
        if self.time_viewer:
            with self._no_lut_update(f'update_lut {args}'):
                for key in ('fmin', 'fmid', 'fmax'):
                    self.callbacks[key](lims[key])
        self._renderer._update()

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
                morph_n_steps = 'nearest' if n_steps == -1 else n_steps
                with use_log_level(False):
                    smooth_mat = _hemi_morph(
                        self.geo[hemi].orig_faces,
                        np.arange(len(self.geo[hemi].coords)),
                        vertices, morph_n_steps, maps=None, warn=False)
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
                    _cell_data(grid)['values'].fill(fill)
                    # XXX for sided data, we probably actually need two
                    # volumes as composite/MIP needs to look at two
                    # extremes... for now just use abs. Eventually we can add
                    # two volumes if we want.
                    _cell_data(grid)['values'][vertices] = values

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
        self._renderer._update()

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
        for _ in self._iter_views(hemi):
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
                _point_data(glyph_dataset)['vec'] = vectors
                glyph_mapper = hemi_data['glyph_mapper']
            if add:
                glyph_actor = self._renderer._actor(glyph_mapper)
                prop = glyph_actor.GetProperty()
                prop.SetLineWidth(2.)
                prop.SetOpacity(vector_alpha)
                self._renderer.plotter.add_actor(glyph_actor, render=False)
                hemi_data['glyph_actor'].append(glyph_actor)
            else:
                glyph_actor = hemi_data['glyph_actor'][count]
            count += 1
            self._renderer._set_colormap_range(
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
        with self._renderer._disabled_interaction():
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

    def _save_movie_tv(self, filename, time_dilation=4., tmin=None, tmax=None,
                       framerate=24, interpolation=None, codec=None,
                       bitrate=None, callback=None, time_viewer=False,
                       **kwargs):
        def frame_callback(frame, n_frames):
            if frame == n_frames:
                # On the ImageIO step
                self.status_msg.set_value(
                    "Saving with ImageIO: %s"
                    % filename
                )
                self.status_msg.show()
                self.status_progress.hide()
                self._renderer._status_bar_update()
            else:
                self.status_msg.set_value(
                    "Rendering images (frame %d / %d) ..."
                    % (frame + 1, n_frames)
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
            self._renderer._window_new_cursor("WaitCursor"))

        try:
            self._save_movie(filename, time_dilation, tmin, tmax,
                             framerate, interpolation, codec,
                             bitrate, frame_callback, time_viewer, **kwargs)
        except (Exception, KeyboardInterrupt):
            warn('Movie saving aborted:\n' + traceback.format_exc())
        finally:
            self._renderer._window_set_cursor(default_cursor)

    @fill_doc
    def save_movie(self, filename=None, time_dilation=4., tmin=None, tmax=None,
                   framerate=24, interpolation=None, codec=None,
                   bitrate=None, callback=None, time_viewer=False, **kwargs):
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
        """
        if filename is None:
            filename = _generate_default_filename(".mp4")
        func = self._save_movie_tv if self.time_viewer else self._save_movie
        func(filename, time_dilation, tmin, tmax,
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
        _validate_type(hemi, (None, str), 'hemi')
        if hemi is None:
            if self._hemi not in ['lh', 'rh']:
                raise ValueError('hemi must not be None when both '
                                 'hemispheres are displayed')
            hemi = self._hemi
        _check_option('hemi', hemi, ('lh', 'rh') + tuple(extras))
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


def _update_monotonic(lims, fmin, fmid, fmax):
    if fmin is not None:
        lims['fmin'] = fmin
        if lims['fmax'] < fmin:
            logger.debug(f'    Bumping fmax = {lims["fmax"]} to {fmin}')
            lims['fmax'] = fmin
        if lims['fmid'] < fmin:
            logger.debug(f'    Bumping fmid = {lims["fmid"]} to {fmin}')
            lims['fmid'] = fmin
    assert lims['fmin'] <= lims['fmid'] <= lims['fmax']
    if fmid is not None:
        lims['fmid'] = fmid
        if lims['fmin'] > fmid:
            logger.debug(f'    Bumping fmin = {lims["fmin"]} to {fmid}')
            lims['fmin'] = fmid
        if lims['fmax'] < fmid:
            logger.debug(f'    Bumping fmax = {lims["fmax"]} to {fmid}')
            lims['fmax'] = fmid
    assert lims['fmin'] <= lims['fmid'] <= lims['fmax']
    if fmax is not None:
        lims['fmax'] = fmax
        if lims['fmin'] > fmax:
            logger.debug(f'    Bumping fmin = {lims["fmin"]} to {fmax}')
            lims['fmin'] = fmax
        if lims['fmid'] > fmax:
            logger.debug(f'    Bumping fmid = {lims["fmid"]} to {fmax}')
            lims['fmid'] = fmax
    assert lims['fmin'] <= lims['fmid'] <= lims['fmax']


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
