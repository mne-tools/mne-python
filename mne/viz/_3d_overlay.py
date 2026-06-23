"""Classes to handle overlapping surfaces."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from collections import OrderedDict

import numpy as np

from ..utils import logger


class _Overlay:
    def __init__(self, scalars, colormap, rng, opacity, name):
        self._scalars = scalars
        self._colormap = colormap
        assert rng is not None
        self._rng = rng
        self._opacity = opacity
        self._name = name

    def to_colors(self):
        from matplotlib.colors import Colormap, ListedColormap

        from ._3d import _get_cmap

        if isinstance(self._colormap, str):
            cmap = _get_cmap(self._colormap)
        elif isinstance(self._colormap, Colormap):
            cmap = self._colormap
        else:
            cmap = ListedColormap(
                self._colormap / 255.0, name=str(type(self._colormap))
            )
        logger.debug(
            f"Color mapping {repr(self._name)} with {cmap.name} "
            f"colormap and range {self._rng}"
        )

        rng = self._rng
        assert rng is not None
        scalars = self._norm(rng)

        colors = cmap(scalars)
        opacity = self._check_opacity(colors.shape[0])
        if opacity is not None:
            colors[:, 3] *= opacity
        return colors

    def _check_opacity(self, n_vertices):
        if self._opacity is None:
            return None
        opacity = np.asarray(self._opacity)
        if opacity.ndim == 0:
            return float(opacity)
        if opacity.ndim != 1:
            raise ValueError(
                "opacity must be a scalar or a 1D array with one value per "
                f"vertex, got an array with shape {opacity.shape}"
            )
        if len(opacity) != n_vertices:
            raise ValueError(
                "opacity array must have one value per vertex, got "
                f"{len(opacity)} values for {n_vertices} vertices"
            )
        return opacity

    def _norm(self, rng):
        if rng[0] == rng[1]:
            factor = 1 if rng[0] == 0 else 1e-6 * rng[0]
        else:
            factor = rng[1] - rng[0]
        return (self._scalars - rng[0]) / factor


class LayeredMesh:
    """A mesh with support for layered RGBA overlays and optional smoothing.

    This class manages a single brain-surface mesh and composites multiple
    named overlays (e.g., curvature, data, labels) on top of each other using
    alpha blending.  It is the object stored in
    ``Brain.layered_meshes``.

    .. warning::
        This class is not meant to be instantiated directly, and the
        initialization API could change at any time! Use
        :meth:`mne.viz.Brain.add_data` to create instances.

    Parameters
    ----------
    renderer : instance of _Renderer
        The renderer used to create the underlying mesh polydata.
    vertices : array, shape (n_vertices, 3)
        Vertex coordinates in metres.
    triangles : array, shape (n_triangles, 3)
        Triangle indices into ``vertices``.
    normals : array, shape (n_vertices, 3)
        Vertex normals.

    Attributes
    ----------
    smooth_mat : scipy.sparse.csr_array or None
        Optional spatial smoother / upsampler applied to scalar data before
        rendering.  When set (e.g., by
        :meth:`~mne.viz.Brain.set_data_smoothing`), every call to
        :meth:`update_overlay` will multiply the incoming scalars by this
        matrix before storing them.  Shape must be
        ``(n_surface_vertices, n_source_vertices)``.

    Notes
    -----
    .. versionadded:: 1.13
    """

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
        self._default_scalars_name = "Data"
        self.smooth_mat = None

    def _map(self):
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
            **kwargs,
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
        C_alpha_zero = C[:, 3] == 0
        C[~C_alpha_zero, :3] /= C[~C_alpha_zero, 3:]
        C[C_alpha_zero, :3] = 0
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

    def add_overlay(self, scalars, colormap, rng, opacity, name, smooth=False):
        """Add a named overlay to the mesh.

        Parameters
        ----------
        scalars : array, shape (n_vertices,)
            Scalar values to display. If ``smooth=True`` and
            ``smooth_mat`` is set, shape must be ``(n_src_vertices,)``.
        colormap : array, shape (n_colors, 4)
            RGBA colormap table (values in ``[0, 255]``).
        rng : array-like, shape (2,)
            ``[min, max]`` range for colormap mapping.
        opacity : float | None
            Overlay opacity in ``[0, 1]``. ``None`` keeps the existing value.
        name : str
            Unique key identifying this overlay.
        smooth : bool
            If ``True`` and ``smooth_mat`` is set, multiply ``scalars``
            by ``smooth_mat`` before rendering. Use ``True`` only for
            source-space data; surface-space overlays (curvature, labels,
            annotations) should use the default ``False``.
        """
        if smooth and self.smooth_mat is not None:
            scalars = self.smooth_mat.dot(scalars)
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
            self._current_colors = self._compute_over(self._cached_colors, colors)

        # apply the texture
        self._apply()

    def remove_overlay(self, names):
        """Remove one or more overlays by name.

        Parameters
        ----------
        names : str | list of str
            Name(s) of the overlay(s) to remove.
        """
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
        self._polydata[self._default_scalars_name] = self._current_colors

    def update(self, colors=None):
        """Recompose all overlays and refresh the mesh texture.

        Parameters
        ----------
        colors : array-like of shape (n_triangles, 4) | None
            Pre-composited RGBA colors to blend over the cached layer stack.
            If ``None``, all overlays are recomposed from scratch.
        """
        if colors is not None and self._cached_colors is not None:
            self._current_colors = self._compute_over(self._cached_colors, colors)
        else:
            self._current_colors, self._cached_colors = self._compose_overlays()
        self._apply()

    def _clean(self):
        mapper = self._actor.GetMapper()
        mapper.SetLookupTable(None)
        self._actor.SetMapper(None)
        self._actor = None
        self._polydata = None
        self._renderer = None

    def update_overlay(
        self, name, scalars=None, colormap=None, opacity=None, rng=None, smooth=False
    ):
        """Update an existing overlay in-place.

        Parameters
        ----------
        name : str
            Key of the overlay to update (must already exist).
        scalars : array, shape (n_vertices,) | None
            New scalar values. If ``None``, scalars are unchanged.
        colormap : array, shape (n_colors, 4) | None
            New RGBA colormap table. If ``None``, colormap is unchanged.
        opacity : float | None
            New opacity in ``[0, 1]``. If ``None``, opacity is unchanged.
        rng : array-like, shape (2,) | None
            New ``[min, max]`` colormap range. If ``None``, range is unchanged.
        smooth : bool
            If ``True`` and ``smooth_mat`` is set, multiply ``scalars``
            by ``smooth_mat`` before rendering. Use ``True`` only for
            source-space data.
        """
        overlay = self._overlays.get(name, None)
        if overlay is None:
            return
        if scalars is not None:
            if smooth and self.smooth_mat is not None:
                scalars = self.smooth_mat.dot(scalars)
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
