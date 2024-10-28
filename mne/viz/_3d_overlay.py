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
        if self._opacity is not None:
            colors[:, 3] *= self._opacity
        return colors

    def _norm(self, rng):
        if rng[0] == rng[1]:
            factor = 1 if rng[0] == 0 else 1e-6 * rng[0]
        else:
            factor = rng[1] - rng[0]
        return (self._scalars - rng[0]) / factor


class _LayeredMesh:
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
            self._current_colors = self._compute_over(self._cached_colors, colors)

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
        self._polydata[self._default_scalars_name] = self._current_colors

    def update(self, colors=None):
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

    def update_overlay(self, name, scalars=None, colormap=None, opacity=None, rng=None):
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
