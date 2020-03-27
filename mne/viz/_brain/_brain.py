# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Oleh Kozynets <ok7mailbox@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#          jona-sassenhagen <jona.sassenhagen@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#
# License: Simplified BSD

import os
import os.path as op

import numpy as np
from scipy import sparse

from .colormap import calculate_lut
from .surface import Surface
from .view import lh_views_dict, rh_views_dict, View

from .._3d import _process_clim, _handle_time

from ...morph import _hemi_morph
from ...label import read_label
from ...utils import _check_option, logger, verbose, fill_doc, _validate_type


class _Brain(object):
    """Class for visualizing a brain.

    It is used for creating meshes of the given subject's
    cortex. The activation data can be shown on a mesh using add_data
    method. Figures, meshes, activation data and other information
    are stored as attributes of a class instance.

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
        freesurfer surface mesh name (ie 'white', 'inflated', etc.).
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
        views to use.
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
        Not supported yet.
        Can be "trackball" (default) or "terrain", i.e. a turntable-style
        camera.
    units : str
        Can be 'm' or 'mm' (default).

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

       +---------------------------+--------------+-----------------------+
       | 3D function:              | surfer.Brain | mne.viz._brain._Brain |
       +===========================+==============+=======================+
       | add_data                  | ✓            | -                     |
       +---------------------------+--------------+-----------------------+
       | add_foci                  | ✓            | -                     |
       +---------------------------+--------------+-----------------------+
       | add_label                 | ✓            | -                     |
       +---------------------------+--------------+-----------------------+
       | add_text                  | ✓            | -                     |
       +---------------------------+--------------+-----------------------+
       | close                     | ✓            | ✓                     |
       +---------------------------+--------------+-----------------------+
       | data                      | ✓            | ✓                     |
       +---------------------------+--------------+-----------------------+
       | foci                      | ✓            |                       |
       +---------------------------+--------------+-----------------------+
       | labels                    | ✓            |                       |
       +---------------------------+--------------+-----------------------+
       | labels_dict               | ✓            |                       |
       +---------------------------+--------------+-----------------------+
       | remove_data               | ✓            |                       |
       +---------------------------+--------------+-----------------------+
       | remove_foci               | ✓            |                       |
       +---------------------------+--------------+-----------------------+
       | remove_labels             | ✓            | -                     |
       +---------------------------+--------------+-----------------------+
       | save_image                | ✓            | ✓                     |
       +---------------------------+--------------+-----------------------+
       | screenshot                | ✓            | ✓                     |
       +---------------------------+--------------+-----------------------+
       | show_view                 | ✓            | -                     |
       +---------------------------+--------------+-----------------------+
       | TimeViewer                | ✓            | ✓                     |
       +---------------------------+--------------+-----------------------+

    """

    def __init__(self, subject_id, hemi, surf, title=None,
                 cortex=None, alpha=1.0, size=800, background="black",
                 foreground=None, figure=None, subjects_dir=None,
                 views=['lateral'], offset=True, show_toolbar=False,
                 offscreen=False, interaction=None, units='mm'):
        from ..backends.renderer import backend, _get_renderer
        from matplotlib.colors import colorConverter

        if interaction is not None:
            raise ValueError('"interaction" parameter is not supported.')

        if hemi in ('both', 'split'):
            self._hemis = ('lh', 'rh')
        elif hemi in ('lh', 'rh'):
            self._hemis = (hemi, )
        else:
            raise KeyError('hemi has to be either "lh", "rh", "split", '
                           'or "both"')

        if isinstance(background, str):
            background = colorConverter.to_rgb(background)
        if isinstance(foreground, str):
            foreground = colorConverter.to_rgb(foreground)
        if isinstance(views, str):
            views = [views]
        n_row = len(views)
        col_dict = dict(lh=1, rh=1, both=1, split=2)
        n_col = col_dict[hemi]

        size = tuple(np.atleast_1d(size).round(0).astype(int).flat)
        if len(size) not in (1, 2):
            raise ValueError('"size" parameter must be an int or length-2 '
                             'sequence of ints.')
        fig_size = size if len(size) == 2 else size * 2  # 1-tuple to 2-tuple

        self._foreground = foreground
        self._hemi = hemi
        self._units = units
        self._title = title
        self._subject_id = subject_id
        self._subjects_dir = subjects_dir
        self._views = views
        self._times = None
        # for now only one color bar can be added
        # since it is the same for all figures
        self._colorbar_added = False
        # for now only one time label can be added
        # since it is the same for all figures
        self._time_label_added = False
        # array of data used by TimeViewer
        self._data = {}
        self.geo, self._hemi_meshes, self._overlays = {}, {}, {}
        self.set_time_interpolation('nearest')

        # load geometry for one or both hemispheres as necessary
        offset = None if (not offset or hemi != 'both') else 0.0

        if figure is not None and not isinstance(figure, int):
            backend._check_3d_figure(figure)
        if title is None:
            self._title = subject_id
        else:
            self._title = title
        self._renderer = _get_renderer(name=self._title, size=fig_size,
                                       bgcolor=background,
                                       shape=(n_row, n_col),
                                       fig=figure)

        for h in self._hemis:
            # Initialize a Surface object as the geometry
            geo = Surface(subject_id, h, surf, subjects_dir, offset,
                          units=self._units)
            # Load in the geometry and curvature
            geo.load_geometry()
            geo.load_curvature()
            self.geo[h] = geo

        for ri, v in enumerate(views):
            for hi, h in enumerate(['lh', 'rh']):
                views_dict = lh_views_dict if hemi == 'lh' else rh_views_dict
                if not (hemi in ['lh', 'rh'] and h != hemi):
                    ci = hi if hemi == 'split' else 0
                    self._renderer.subplot(ri, ci)
                    mesh_data = self._renderer.mesh(
                        x=self.geo[h].coords[:, 0],
                        y=self.geo[h].coords[:, 1],
                        z=self.geo[h].coords[:, 2],
                        triangles=self.geo[h].faces,
                        color=self.geo[h].grey_curv,
                        opacity=alpha,
                    )
                    if isinstance(mesh_data, tuple):
                        _, mesh = mesh_data
                        # add metadata to the mesh for picking
                        mesh._hemi = h
                    else:
                        _, mesh = mesh_data, None
                    self._hemi_meshes[h] = mesh
                    self._renderer.set_camera(azimuth=views_dict[v].azim,
                                              elevation=views_dict[v].elev)

        # Force rendering
        self._renderer.show()

    @verbose
    def add_data(self, array, fmin=None, fmid=None, fmax=None,
                 thresh=None, center=None, transparent=False, colormap="auto",
                 alpha=1, vertices=None, smoothing_steps=None, time=None,
                 time_label="auto", colorbar=True,
                 hemi=None, remove_existing=None, time_label_size=None,
                 initial_time=None, scale_factor=None, vector_alpha=None,
                 clim=None, verbose=None):
        """Display data from a numpy array on the surface.

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
        fmin : float
            Minimum value in colormap (uses real fmin if None).
        fmid : float
            Intermediate value in colormap (fmid between fmin and
            fmax if None).
        fmax : float
            Maximum value in colormap (uses real max if None).
        thresh : None or float
            Not supported yet.
            if not None, values below thresh will not be visible
        center : float or None
            if not None, center of a divergent colormap, changes the meaning of
            fmin, fmax and fmid.
        transparent : bool
            if True: use a linear transparency between fmin and fmid
            and make values below fmin fully transparent (symmetrically for
            divergent colormaps)
        colormap : str, list of color, or array
            name of matplotlib colormap to use, a list of matplotlib colors,
            or a custom look up table (an n x 4 array coded with RBGA values
            between 0 and 255), the default "auto" chooses a default divergent
            colormap, if "center" is given (currently "icefire"), otherwise a
            default sequential colormap (currently "rocket").
        alpha : float in [0, 1]
            alpha level to control opacity of the overlay.
        vertices : numpy array
            vertices for which the data is defined (needed if len(data) < nvtx)
        smoothing_steps : int or None
            number of smoothing steps (smoothing is used if len(data) < nvtx)
            The value 'nearest' can be used too.
            Default : 7
        time : numpy array
            time points in the data array (if data is 2D or 3D)
        %(time_label)s
        colorbar : bool
            whether to add a colorbar to the figure
        hemi : str | None
            If None, it is assumed to belong to the hemisphere being
            shown. If two hemispheres are being shown, an error will
            be thrown.
        remove_existing : bool
            Not supported yet.
            Remove surface added by previous "add_data" call. Useful for
            conserving memory when displaying different data in a loop.
        time_label_size : int
            Font size of the time label (default 14)
        initial_time : float | None
            Time initially shown in the plot. ``None`` to use the first time
            sample (default).
        scale_factor : float | None (default)
            Not supported yet.
            The scale factor to use when displaying glyphs for vector-valued
            data.
        vector_alpha : float | None
            Not supported yet.
            alpha level to control opacity of the arrows. Only used for
            vector-valued data. If None (default), ``alpha`` is used.
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
        _check_option('time_label_size', time_label_size, [None])

        hemi = self._check_hemi(hemi)
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

        self._data['smoothing_steps'] = smoothing_steps
        self._data['clim'] = clim
        self._data['time'] = time
        self._data['initial_time'] = initial_time
        self._data['time_label'] = time_label
        self._data['time_idx'] = time_idx
        self._data['transparent'] = transparent
        # data specific for a hemi
        self._data[hemi] = dict()
        self._data[hemi]['actor'] = list()
        self._data[hemi]['mesh'] = list()
        self._data[hemi]['glyph_actor'] = None
        self._data[hemi]['glyph_mesh'] = None
        self._data[hemi]['array'] = array
        self._data[hemi]['vertices'] = vertices
        self._data['alpha'] = alpha
        self._data['colormap'] = colormap
        self._data['center'] = center
        self._data['fmin'] = fmin
        self._data['fmid'] = fmid
        self._data['fmax'] = fmax

        dt_max = fmax
        dt_min = fmin if center is None else -1 * fmax

        ctable = self.update_lut()
        self._data['ctable'] = ctable

        # 1) add the surfaces first
        for ri, v in enumerate(self._views):
            views_dict = lh_views_dict if hemi == 'lh' else rh_views_dict
            if self._hemi != 'split':
                ci = 0
            else:
                ci = 0 if hemi == 'lh' else 1
            self._renderer.subplot(ri, ci)

            mesh_data = self._renderer.mesh(
                x=self.geo[hemi].coords[:, 0],
                y=self.geo[hemi].coords[:, 1],
                z=self.geo[hemi].coords[:, 2],
                triangles=self.geo[hemi].faces,
                color=None,
                colormap=ctable,
                vmin=dt_min,
                vmax=dt_max,
                scalars=np.zeros(len(self.geo[hemi].coords)),
            )
            if isinstance(mesh_data, tuple):
                actor, mesh = mesh_data
                # add metadata to the mesh for picking
                mesh._hemi = hemi
            else:
                actor, mesh = mesh_data, None
            self._data[hemi]['actor'].append(actor)
            self._data[hemi]['mesh'].append(mesh)

        # 2) update time and smoothing properties
        # set_data_smoothing calls "set_time_point" for us, which will set
        # _current_time
        self.set_time_interpolation(self.time_interpolation)
        self.set_data_smoothing(smoothing_steps)

        # 3) add the other actors
        for ri, v in enumerate(self._views):
            views_dict = lh_views_dict if hemi == 'lh' else rh_views_dict
            if self._hemi != 'split':
                ci = 0
            else:
                ci = 0 if hemi == 'lh' else 1
            if not self._time_label_added and time_label is not None:
                time_actor = self._renderer.text2d(
                    x_window=0.95, y_window=y_txt,
                    size=time_label_size,
                    text=time_label(self._current_time),
                    justification='right'
                )
                self._data['time_actor'] = time_actor
                self._time_label_added = True
            if colorbar and not self._colorbar_added:
                self._renderer.scalarbar(source=actor, n_labels=8,
                                         bgcolor=(0.5, 0.5, 0.5))
                self._colorbar_added = True
            self._renderer.set_camera(azimuth=views_dict[v].azim,
                                      elevation=views_dict[v].elev)

    def add_label(self, label, color=None, alpha=1, scalar_thresh=None,
                  borders=False, hemi=None, subdir=None):
        """Add an ROI label to the image.

        Parameters
        ----------
        label : str | instance of Label
            label filepath or name. Can also be an instance of
            an object with attributes "hemi", "vertices", "name", and
            optionally "color" and "values" (if scalar_thresh is not None).
        color : matplotlib-style color | None
            anything matplotlib accepts: string, RGB, hex, etc. (default
            "crimson")
        alpha : float in [0, 1]
            alpha level to control opacity
        scalar_thresh : None or number
            threshold the label ids using this value in the label
            file's scalar field (i.e. label only vertices with
            scalar >= thresh)
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

        Notes
        -----
        To remove previously added labels, run Brain.remove_labels().
        """
        from matplotlib.colors import colorConverter
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

        # XXX: add support for label_name
        self._label_name = label_name

        label = np.zeros(self.geo[hemi].coords.shape[0])
        label[ids] = 1
        color = colorConverter.to_rgba(color, alpha)
        cmap = np.array([(0, 0, 0, 0,), color])
        ctable = np.round(cmap * 255).astype(np.uint8)

        for ri, v in enumerate(self._views):
            if self._hemi != 'split':
                ci = 0
            else:
                ci = 0 if hemi == 'lh' else 1
            views_dict = lh_views_dict if hemi == 'lh' else rh_views_dict
            self._renderer.subplot(ri, ci)
            if borders:
                surface = {
                    'rr': self.geo[hemi].coords,
                    'tris': self.geo[hemi].faces,
                }
                self._renderer.contour(surface, label, [1.0], color=color,
                                       kind='tube')
            else:
                self._renderer.mesh(x=self.geo[hemi].coords[:, 0],
                                    y=self.geo[hemi].coords[:, 1],
                                    z=self.geo[hemi].coords[:, 2],
                                    triangles=self.geo[hemi].faces,
                                    scalars=label,
                                    color=None,
                                    colormap=ctable,
                                    backface_culling=False)
            self._renderer.set_camera(azimuth=views_dict[v].azim,
                                      elevation=views_dict[v].elev)

    def add_foci(self, coords, coords_as_verts=False, map_surface=None,
                 scale_factor=1, color="white", alpha=1, name=None,
                 hemi=None):
        """Add spherical foci, possibly mapping to displayed surf.

        The foci spheres can be displayed at the coordinates given, or
        mapped through a surface geometry. In other words, coordinates
        from a volume-based analysis in MNI space can be displayed on an
        inflated average surface by finding the closest vertex on the
        white surface and mapping to that vertex on the inflated mesh.

        Parameters
        ----------
        coords : numpy array
            x, y, z coordinates in stereotaxic space (default) or array of
            vertex ids (with ``coord_as_verts=True``)
        coords_as_verts : bool
            whether the coords parameter should be interpreted as vertex ids
        map_surface : Freesurfer surf or None
            surface to map coordinates through, or None to use raw coords
        scale_factor : float
            Controls the size of the foci spheres (relative to 1cm).
        color : matplotlib color code
            HTML name, RBG tuple, or hex code
        alpha : float in [0, 1]
            opacity of focus gylphs
        name : str
            internal name to use
        hemi : str | None
            If None, it is assumed to belong to the hemipshere being
            shown. If two hemispheres are being shown, an error will
            be thrown.
        """
        from matplotlib.colors import colorConverter
        hemi = self._check_hemi(hemi)

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
        for ri, v in enumerate(self._views):
            views_dict = lh_views_dict if hemi == 'lh' else rh_views_dict
            if self._hemi != 'split':
                ci = 0
            else:
                ci = 0 if hemi == 'lh' else 1
            self._renderer.subplot(ri, ci)
            self._renderer.sphere(center=coords, color=color,
                                  scale=(10. * scale_factor),
                                  opacity=alpha)
            self._renderer.set_camera(azimuth=views_dict[v].azim,
                                      elevation=views_dict[v].elev)

    def add_text(self, x, y, text, name=None, color=None, opacity=1.0,
                 row=-1, col=-1, font_size=None, justification=None):
        """Add a text to the visualization.

        Parameters
        ----------
        x : Float
            x coordinate
        y : Float
            y coordinate
        text : str
            Text to add
        name : str
            Name of the text (text label can be updated using update_text())
        color : Tuple
            Color of the text. Default is the foreground color set during
            initialization (default is black or white depending on the
            background color).
        opacity : Float
            Opacity of the text. Default: 1.0
        row : int
            Row index of which brain to use
        col : int
            Column index of which brain to use
        """
        # XXX: support `name` should be added when update_text/remove_text
        # are implemented
        # _check_option('name', name, [None])

        self._renderer.text2d(x_window=x, y_window=y, text=text, color=color,
                              size=font_size, justification=justification)

    def remove_labels(self, labels=None):
        """Remove one or more previously added labels from the image.

        Parameters
        ----------
        labels : None | str | list of str
            Labels to remove. Can be a string naming a single label, or None to
            remove all labels. Possible names can be found in the Brain.labels
            attribute.
        """
        pass

    def close(self):
        """Close all figures and cleanup data structure."""
        self._renderer.close()

    def show_view(self, view=None, roll=None, distance=None, row=0, col=0,
                  hemi=None):
        """Orient camera to display view."""
        hemi = self._hemi if hemi is None else hemi
        views_dict = lh_views_dict if hemi == 'lh' else rh_views_dict
        if isinstance(view, str):
            view = views_dict.get(view)
        elif isinstance(view, dict):
            view = View(azim=view['azimuth'],
                        elev=view['elevation'])
        self._renderer.subplot(row, col)
        self._renderer.set_camera(azimuth=view.azim,
                                  elevation=view.elev)
        self._renderer.reset_camera()

    def save_image(self, filename, mode='rgb'):
        """Save view from all panels to disk.

        Parameters
        ----------
        filename: string
            path to new image file
        mode : string
            Either 'rgb' or 'rgba' for values to return.
        """
        self._renderer.screenshot(mode=mode, filename=filename)

    def screenshot(self, mode='rgb'):
        """Generate a screenshot of current view.

        Parameters
        ----------
        mode : string
            Either 'rgb' or 'rgba' for values to return.

        Returns
        -------
        screenshot : array
            Image pixel values.
        """
        return self._renderer.screenshot(mode)

    def update_lut(self, fmin=None, fmid=None, fmax=None, alpha=None):
        """Update color map.

        Parameters
        ----------
        fmin : float | None
            Minimum value in colormap.
        fmid : float | None
            Intermediate value in colormap (fmid between fmin and
            fmax).
        fmax : float | None
            Maximum value in colormap.
        """
        alpha = alpha if alpha is not None else self._data['alpha']
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
        self._data['ctable'] = \
            calculate_lut(colormap, alpha=alpha, center=center,
                          transparent=transparent, **lims)
        return self._data['ctable']

    def set_data_smoothing(self, n_steps):
        """Set the number of smoothing steps.

        Parameters
        ----------
        n_steps : int
            Number of smoothing steps
        """
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
                smooth_mat = _hemi_morph(
                    self.geo[hemi].faces,
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
        _check_option('interpolation', interpolation,
                      ('linear', 'nearest', 'zero', 'slinear', 'quadratic',
                       'cubic'))
        self._time_interpolation = str(interpolation)
        del interpolation
        self._time_interp_funcs = dict()
        self._time_interp_inv = None
        if self._times is not None:
            idx = np.arange(self._n_times)
            for hemi in ['lh', 'rh']:
                hemi_data = self._data.get(hemi)
                if hemi_data is not None:
                    array = hemi_data['array']
                    self._time_interp_funcs[hemi] = _safe_interp1d(
                        idx, array, self._time_interpolation, axis=-1,
                        assume_sorted=True)
            self._time_interp_inv = _safe_interp1d(idx, self._times)

    def set_time_point(self, time_idx):
        """Set the time point shown (can be a float to interpolate)."""
        from ..backends._pyvista import _set_mesh_scalars
        current_act_data = list()
        time_actor = self._data.get('time_actor', None)
        time_label = self._data.get('time_label', None)
        for hemi in ['lh', 'rh']:
            hemi_data = self._data.get(hemi)
            if hemi_data is not None:
                array = hemi_data['array']
                # interpolate in time
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
                current_act_data.append(act_data)
                if time_actor is not None and time_label is not None:
                    time_actor.SetInput(time_label(self._current_time))

                # interpolate in space
                smooth_mat = hemi_data['smooth_mat']
                if smooth_mat is not None:
                    act_data = smooth_mat.dot(act_data)

                # update the mesh scalar values
                for mesh in hemi_data['mesh']:
                    if mesh is not None:
                        _set_mesh_scalars(mesh, act_data, 'Data')

                # update the glyphs
                if array.ndim == 3:
                    self.update_glyphs(hemi, vectors)
        self._current_act_data = np.concatenate(current_act_data)
        self._data['time_idx'] = time_idx

    def update_glyphs(self, hemi, vectors):
        from ..backends._pyvista import (_set_colormap_range,
                                         _add_polydata_actor)
        hemi_data = self._data.get(hemi)
        if hemi_data is not None:
            vertices = hemi_data['vertices']
            fmin = self._data['fmin']
            fmid = self._data['fmid']
            fmax = self._data['fmax']
            ctable = self.update_lut(fmin=fmin, fmid=fmid, fmax=fmax, alpha=1)
            ctable = (ctable * 255).astype(np.uint8)
            vector_alpha = self._data['vector_alpha']
            scale_factor = self._data['scale_factor']
            rng = [fmin, fmax]
            vertices = slice(None) if vertices is None else vertices
            x, y, z = np.array(self.geo[hemi].coords)[vertices].T

            polydata = self._renderer.quiver3d(
                x, y, z,
                vectors[:, 0], vectors[:, 1], vectors[:, 2],
                color=None,
                mode='2darrow',
                scale_mode='vector',
                scale=scale_factor,
                opacity=vector_alpha,
                name=str(hemi) + "_glyph"
            )
            if polydata is not None:
                if hemi_data['glyph_mesh'] is None:
                    hemi_data['glyph_mesh'] = polydata
                    glyph_actor = _add_polydata_actor(
                        plotter=self._renderer.plotter,
                        polydata=polydata,
                        hide=True
                    )
                    hemi_data['glyph_actor'] = glyph_actor
                    glyph_actor.GetProperty().SetLineWidth(2.)
                else:
                    glyph_actor = hemi_data['glyph_actor']
                    glyph_mesh = hemi_data['glyph_mesh']
                    glyph_mesh.shallow_copy(polydata)
                _set_colormap_range(glyph_actor, ctable, None, rng)
                # the glyphs are now ready to be displayed
                glyph_actor.VisibilityOn()

    def update_fmax(self, fmax):
        """Set the colorbar max point."""
        from ..backends._pyvista import _set_colormap_range
        ctable = self.update_lut(fmax=fmax)
        ctable = (ctable * 255).astype(np.uint8)
        center = self._data['center']
        fmin = self._data['fmin']
        for hemi in ['lh', 'rh']:
            hemi_data = self._data.get(hemi)
            if hemi_data is not None:
                for actor in hemi_data['actor']:
                    dt_max = fmax
                    dt_min = fmin if center is None else -1 * fmax
                    rng = [dt_min, dt_max]
                    if self._colorbar_added:
                        scalar_bar = self._renderer.plotter.scalar_bar
                    else:
                        scalar_bar = None
                    _set_colormap_range(actor, ctable, scalar_bar, rng)
                    self._data['fmax'] = fmax
                    self._data['ctable'] = ctable

    def update_fmid(self, fmid):
        """Set the colorbar mid point."""
        from ..backends._pyvista import _set_colormap_range
        ctable = self.update_lut(fmid=fmid)
        ctable = (ctable * 255).astype(np.uint8)
        for hemi in ['lh', 'rh']:
            hemi_data = self._data.get(hemi)
            if hemi_data is not None:
                for actor in hemi_data['actor']:
                    if self._colorbar_added:
                        scalar_bar = self._renderer.plotter.scalar_bar
                    else:
                        scalar_bar = None
                    _set_colormap_range(actor, ctable, scalar_bar)
                    self._data['fmid'] = fmid
                    self._data['ctable'] = ctable

    def update_fmin(self, fmin):
        """Set the colorbar min point."""
        from ..backends._pyvista import _set_colormap_range
        ctable = self.update_lut(fmin=fmin)
        ctable = (ctable * 255).astype(np.uint8)
        center = self._data['center']
        fmax = self._data['fmax']
        for hemi in ['lh', 'rh']:
            hemi_data = self._data.get(hemi)
            if hemi_data is not None:
                for actor in hemi_data['actor']:
                    dt_max = fmax
                    dt_min = fmin if center is None else -1 * fmax
                    rng = [dt_min, dt_max]
                    if self._colorbar_added:
                        scalar_bar = self._renderer.plotter.scalar_bar
                    else:
                        scalar_bar = None
                    _set_colormap_range(actor, ctable, scalar_bar, rng)
                    self._data['fmin'] = fmin
                    self._data['ctable'] = ctable

    def update_fscale(self, fscale):
        """Scale the colorbar points."""
        from ..backends._pyvista import _set_colormap_range
        center = self._data['center']
        fmin = self._data['fmin'] * fscale
        fmid = self._data['fmid'] * fscale
        fmax = self._data['fmax'] * fscale
        ctable = self.update_lut(fmin=fmin, fmid=fmid, fmax=fmax)
        ctable = (ctable * 255).astype(np.uint8)
        for hemi in ['lh', 'rh']:
            hemi_data = self._data.get(hemi)
            if hemi_data is not None:
                for actor in hemi_data['actor']:
                    dt_max = fmax
                    dt_min = fmin if center is None else -1 * fmax
                    rng = [dt_min, dt_max]
                    if self._colorbar_added:
                        scalar_bar = self._renderer.plotter.scalar_bar
                    else:
                        scalar_bar = None
                    _set_colormap_range(actor, ctable, scalar_bar, rng)
                    self._data['ctable'] = ctable
                    self._data['fmin'] = fmin
                    self._data['fmid'] = fmid
                    self._data['fmax'] = fmax

    def update_auto_scaling(self, restore=False):
        from ..backends._pyvista import _set_colormap_range
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
            clim, colormap, transparent, self._current_act_data,
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
        ctable = self.update_lut(fmin=fmin, fmid=fmid, fmax=fmax)
        ctable = (ctable * 255).astype(np.uint8)
        for hemi in ['lh', 'rh']:
            hemi_data = self._data.get(hemi)
            if hemi_data is not None:
                for actor in hemi_data['actor']:
                    dt_max = fmax
                    dt_min = fmin if center is None else -1 * fmax
                    rng = [dt_min, dt_max]
                    if self._colorbar_added:
                        scalar_bar = self._renderer.plotter.scalar_bar
                    else:
                        scalar_bar = None
                    _set_colormap_range(actor, ctable, scalar_bar, rng)
                    self._data['ctable'] = ctable

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
    def views(self):
        return self._views

    @property
    def hemis(self):
        return self._hemis

    def _show(self):
        """Request rendering of the window."""
        try:
            return self._renderer.show()
        except RuntimeError:
            logger.info("No active/running renderer available.")

    def _check_hemi(self, hemi):
        """Check for safe single-hemi input, returns str."""
        if hemi is None:
            if self._hemi not in ['lh', 'rh']:
                raise ValueError('hemi must not be None when both '
                                 'hemispheres are displayed')
            else:
                hemi = self._hemi
        elif hemi not in ['lh', 'rh']:
            extra = ' or None' if self._hemi in ['lh', 'rh'] else ''
            raise ValueError('hemi must be either "lh" or "rh"' +
                             extra + ", got " + str(hemi))
        return hemi

    def scale_data_colormap(self, fmin, fmid, fmax, transparent,
                            center=None, alpha=1.0, data=None, verbose=None):
        """Scale the data colormap."""
        lut_lst = self._data['ctable']
        n_col = len(lut_lst)

        # apply the lut on every surfaces
        for hemi in ['lh', 'rh']:
            hemi_data = self._data.get(hemi)
            if hemi_data is not None:
                for actor in hemi_data['actor']:
                    vtk_lut = actor.GetMapper().GetLookupTable()
                    vtk_lut.SetNumberOfColors(n_col)
                    vtk_lut.SetRange([fmin, fmax])
                    vtk_lut.Build()
                    for i in range(0, n_col):
                        lt = lut_lst[i]
                        vtk_lut.SetTableValue(i, lt[0], lt[1], lt[2], alpha)
        self.update_fscale(1.0)

    def enable_depth_peeling(self):
        """Enable depth peeling."""
        self._renderer.enable_depth_peeling()


def _safe_interp1d(x, y, kind='linear', axis=-1, assume_sorted=False):
    """Work around interp1d not liking singleton dimensions."""
    from scipy.interpolate import interp1d
    if y.shape[axis] == 1:
        def func(x):
            return y.copy()
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
