# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Oleh Kozynets <ok7mailbox@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#          jona-sassenhagen <jona.sassenhagen@gmail.com>
#
# License: Simplified BSD

import numpy as np

from .colormap import _calculate_lut
from .view import views_dict
from .surface import Surface


class Brain(object):
    u"""Class for visualizing a brain using ipyvolume.

    It is used for creating ipyvolume meshes of the given subject's
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
    alpha : float in [0, 1]
        Alpha level to control opacity of the cortical surface.
    size : float | tuple(float, float)
        The size of the window, in pixels. can be one number to specify
        a square window, or the (width, height) of a rectangular window.
    background : tuple(int, int, int)
        The color definition of the background: (red, green, blue).
    foreground : matplotlib color
        Color of the foreground (will be used for colorbars and text).
        None (default) will use black or white depending on the value
        of ``background``.
    figure : list of Figure | None | int
        Not supported yet.
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
        A dictionary of ipysurfer.Surface objects for each hemisphere.
    overlays : dict
        The overlays.
    """

    def __init__(self, subject_id, hemi, surf, title=None,
                 alpha=1.0, size=800, background=(0, 0, 0),
                 foreground=None, figure=None, subjects_dir=None,
                 views=['lateral'], offset=True, show_toolbar=False,
                 offscreen=False, interaction=None, units='mm'):
        if hemi == 'split':
            raise ValueError('Option hemi="split" is not supported yet.')

        if figure is not None:
            raise ValueError('figure parameter is not supported yet.')

        if interaction is not None:
            raise ValueError('"interaction" parameter is not supported.')

        from ..backends.renderer import _Renderer

        self._foreground = foreground
        self._hemi = hemi
        self._units = units
        self._title = title
        self._subject_id = subject_id
        self._views = views
        # for now only one color bar can be added
        # since it is the same for all figures
        self._colorbar_added = False
        # array of data used by TimeViewer
        self._data = {}

        # load geometry for one or both hemispheres as necessary
        offset = None if (not offset or hemi != 'both') else 0.0

        if hemi in ('both', 'split'):
            self._hemis = ('lh', 'rh')
        elif hemi in ('lh', 'rh'):
            self._hemis = (hemi, )
        else:
            raise KeyError('hemi has to be either "lh", "rh", "split", '
                           'or "both"')

        if isinstance(size, int):
            fig_size = (size, size)
        elif isinstance(size, tuple):
            fig_size = size

        self.geo, self._hemi_meshes, self._overlays = {}, {}, {}
        self._renderers = [[] for _ in views]

        for h in self._hemis:
            # Initialize a Surface object as the geometry
            geo = Surface(subject_id, h, surf, subjects_dir, offset,
                          units=self._units)
            # Load in the geometry and curvature
            geo.load_geometry()
            geo.load_curvature()
            self.geo[h] = geo

        for ri, v in enumerate(views):
            renderer = _Renderer(size=fig_size, bgcolor=background)
            self._renderers[ri].append(renderer)
            renderer.set_camera(azimuth=views_dict[v].azim,
                                elevation=views_dict[v].elev)

            for ci, h in enumerate(self._hemis):
                if ci == 1 and hemi == 'split':
                    # create a separate figure for right hemisphere
                    renderer = _Renderer(size=fig_size, bgcolor=background)
                    self._renderers[ri].append(renderer)
                    renderer.set_camera(azimuth=views_dict[v].azim,
                                        elevation=views_dict[v].elev)

                mesh = renderer.mesh(x=self.geo[h].coords[:, 0],
                                     y=self.geo[h].coords[:, 1],
                                     z=self.geo[h].coords[:, 2],
                                     triangles=self.geo[h].faces,
                                     color=self.geo[h].grey_curv)

                self._hemi_meshes[h + '_' + v] = mesh

    def add_data(self, array, min=None, mid=None, max=None, thresh=None,
                 center=None, transparent=None, colormap="auto", alpha=1,
                 vertices=None, smoothing_steps=None, time=None,
                 time_label="time index=%d", colorbar=True,
                 hemi=None, remove_existing=None, time_label_size=None,
                 initial_time=None, scale_factor=None, vector_alpha=None,
                 verbose=None):
        u"""Display data from a numpy array on the surface.

        This provides a similar interface to
        :meth:`surfer.Brain.add_overlay`, but it displays
        it with a single colormap. It offers more flexibility over the
        colormap, and provides a way to display four-dimensional data
        (i.e., a timecourse) or five-dimensional data (i.e., a
        vector-valued timecourse).

        .. note:: ``min`` sets the low end of the colormap, and is separate
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
        min : float
            Min value in colormap (uses real min if None).
        mid : float
            Intermediate value in colormap (middle between min and max
            if None).
        max : float
            Max value in colormap (uses real max if None).
        thresh : None or float
            Not supported yet.
            if not None, values below thresh will not be visible
        center : float or None
            if not None, center of a divergent colormap, changes the meaning of
            min, max and mid.
        transparent : bool
            Not supported yet.
            if True: use a linear transparency between min and mid and make
            values below min fully transparent (symmetrically for divergent
            colormaps)
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
            Default : 20
        time : numpy array
            time points in the data array (if data is 2D or 3D)
        time_label : str | callable | None
            format of the time label (a format string, a function that maps
            floating point time values to strings, or None for no label)
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
            Not supported yet.
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
        verbose : bool, str, int, or None
            Not supported yet.
            If not None, override default verbose level (see surfer.verbose).

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
        if len(array.shape) == 3:
            raise ValueError('Vector values in "array" are not supported.')
        if thresh is not None:
            raise ValueError('"threshold" parameter is not supported yet.')
        if transparent is not None:
            raise ValueError('"transparent" is not supported yet.')
        if remove_existing is not None:
            raise ValueError('"remove_existing" is not supported yet.')
        if time_label_size is not None:
            raise ValueError('"time_label_size" is not supported yet.')
        if scale_factor is not None:
            raise ValueError('"scale_factor" is not supported yet.')
        if vector_alpha is not None:
            raise ValueError('"vector_alpha" is not supported yet.')
        if verbose is not None:
            raise ValueError('"verbose" is not supported yet.')

        from surfer.utils import mesh_edges, smoothing_matrix

        hemi = self._check_hemi(hemi)
        array = np.asarray(array)

        if initial_time is None:
            time_idx = 0
        else:
            time_idx = np.argmin(abs(time - initial_time))

        self._data['time'] = time
        self._data['initial_time'] = initial_time
        self._data['time_label'] = time_label
        self._data['time_idx'] = time_idx
        # data specific for a hemi
        self._data[hemi + '_array'] = array

        if time is not None and len(array.shape) == 2:
            # we have scalar_data with time dimension
            act_data = array[:, time_idx]
        else:
            # we have scalar data without time
            act_data = array

        if center is None:
            if min is None:
                min = array.min() if array.size > 0 else 0
            if max is None:
                max = array.max() if array.size > 0 else 1
        else:
            if min is None:
                min = 0
            if max is None:
                max = np.abs(center - array).max() if array.size > 0 else 1
        if mid is None:
            mid = (min + max) / 2.
        _check_limits(min, mid, max, extra='')
        self._data['alpha'] = alpha
        self._data['colormap'] = colormap
        self._data['center'] = center
        self._data['min'] = min
        self._data['mid'] = mid
        self._data['max'] = max

        lut = self.update_lut()

        # Create smoothing matrix if necessary
        if len(act_data) < self.geo[hemi].x.shape[0]:
            if vertices is None:
                raise ValueError('len(data) < nvtx (%s < %s): the vertices '
                                 'parameter must not be None'
                                 % (len(act_data), self.geo[hemi].x.shape[0]))
            adj_mat = mesh_edges(self.geo[hemi].faces)
            smooth_mat = smoothing_matrix(vertices,
                                          adj_mat,
                                          smoothing_steps)
            act_data = smooth_mat.dot(act_data)
            self._data[hemi + '_smooth_mat'] = smooth_mat

        # data mapping into [0, 1] interval
        dt_max = max
        dt_min = min if center is None else -1 * max
        k = 1 / (dt_max - dt_min)
        b = 1 - k * dt_max
        act_data = k * act_data + b
        act_data = np.clip(act_data, 0, 1)

        act_color = lut(act_data)

        self._data['k'] = k
        self._data['b'] = b

        for ri, v in enumerate(self._views):
            if self._hemi != 'split':
                ci = 0
            else:
                ci = 0 if hemi == 'lh' else 1
            renderer = self._renderers[ri][ci]
            mesh = renderer.mesh(x=self.geo[hemi].coords[:, 0],
                                 y=self.geo[hemi].coords[:, 1],
                                 z=self.geo[hemi].coords[:, 2],
                                 triangles=self.geo[hemi].faces,
                                 color=act_color)
            self._overlays[hemi + '_' + v] = mesh

        # How can we make this bit universal as well???
        # if colorbar and not self._colorbar_added:
        #     ColorBar(self)
        #     self._colorbar_added = True

    def show(self):
        u"""Display widget."""
        self._renderers[0][0].show()

    def update_lut(self, min=None, mid=None, max=None):
        u"""Update color map.

        Parameters
        ----------
        min : float | None
            Min value in colormap.
        mid : float | None
            Intermediate value in colormap (middle between min and
            max).
        max : float | None
            Max value in colormap.
        """
        alpha = self._data['alpha']
        center = self._data['center']
        colormap = self._data['colormap']
        min = self._data['min'] if min is None else min
        mid = self._data['mid'] if mid is None else mid
        max = self._data['max'] if max is None else max

        self._data['lut'] = _calculate_lut(
            colormap, alpha=alpha, fmin=min, fmid=mid, fmax=max, center=center)

        return self._data['lut']

    @property
    def overlays(self):
        return self._overlays

    @property
    def data(self):
        u"""Data used by time viewer and color bar widgets."""
        return self._data

    @property
    def views(self):
        return self._views

    @property
    def hemis(self):
        return self._hemis

    def _check_hemi(self, hemi):
        u"""Check for safe single-hemi input, returns str."""
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


def _check_limits(min, mid, max, extra='f'):
    u"""Check for monotonicity."""
    if min >= mid:
        raise ValueError('%smin must be < %smid, got %0.4g >= %0.4g'
                         % (extra, extra, min, mid))
    if mid >= max:
        raise ValueError('%smid must be < %smax, got %0.4g >= %0.4g'
                         % (extra, extra, mid, max))
