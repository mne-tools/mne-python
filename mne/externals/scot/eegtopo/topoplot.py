# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 Martin Billinger

from __future__ import division

import numpy as np
from scipy.interpolate import interp1d
#noinspection PyPep8Naming
import matplotlib.pyplot as plot
#noinspection PyPep8Naming
import matplotlib.path as path
#noinspection PyPep8Naming
import matplotlib.patches as patches
from .projections import (array_project_radial_to3d,
                                       array_project_radial_to2d)
from .geometry.euclidean import Vector
from scipy.spatial import ConvexHull


class Topoplot:
    """ Creates 2D scalp maps. """

    def __init__(self, m=4, num_lterms=10, headcolor=[0, 0, 0, 1], clipping='head', electrodescale=1, interpolationrange=np.pi * 3 / 4, head_radius=np.pi * 3 / 4):
        self.interprange = interpolationrange
        self.head_radius = head_radius
        self.nose_angle = 15
        self.nose_length = 0.12

        self.headcolor = headcolor

        self.clipping = clipping
        self.electrodescale = np.asarray(electrodescale)

        verts = np.array([
            (1, 0),
            (1, 0.5535714285714286), (0.5535714285714286, 1), (0, 1),
            (-0.5535714285714286, 1), (-1, 0.5535714285714286), (-1, 0),
            (-1, -0.5535714285714286), (-0.5535714285714286, -1), (0, -1),
            (0.5535714285714286, -1), (1, -0.5535714285714286), (1, 0),
        ]) * self.head_radius
        codes = [path.Path.MOVETO,
                 path.Path.CURVE4, path.Path.CURVE4, path.Path.CURVE4,
                 path.Path.CURVE4, path.Path.CURVE4, path.Path.CURVE4,
                 path.Path.CURVE4, path.Path.CURVE4, path.Path.CURVE4,
                 path.Path.CURVE4, path.Path.CURVE4, path.Path.CURVE4,
        ]
        self.path_head = path.Path(verts, codes)

        x = self.head_radius * np.cos((90.0 - self.nose_angle / 2) * np.pi / 180.0)
        y = self.head_radius * np.sin((90.0 - self.nose_angle / 2) * np.pi / 180.0)
        verts = np.array([(x, y), (0, self.head_radius * (1 + self.nose_length)), (-x, y)])
        codes = [path.Path.MOVETO, path.Path.LINETO, path.Path.LINETO]
        self.path_nose = path.Path(verts, codes)

        self.legendre_factors = self.calc_legendre_factors(m, num_lterms)

        self.channel_fence = None
        self.locations = None
        self.g = None
        self.z = None
        self.c = None
        self.image = None

        self.g_map = {}

    @staticmethod
    def calc_legendre_factors(m, num_lterms):
        return [(2 * n + 1) / (n ** m * (n + 1) ** m * 4 * np.pi) for n in range(1, num_lterms + 1)]

    def calc_g(self, x):
        return np.polynomial.legendre.legval(x, self.legendre_factors)

    def set_locations(self, locations):
        n = len(locations)

        g = np.zeros((1 + n, 1 + n))
        g[:, 0] = np.ones(1 + n)
        g[-1, :] = np.ones(1 + n)
        g[-1, 0] = 0
        for i in range(n):
            for j in range(n):
                g[i, j + 1] = self.calc_g(np.dot(locations[i], locations[j]))

        self.channel_fence = None
        self.locations = locations
        self.g = g

    def set_values(self, z):
        self.z = z
        self.c = np.linalg.solve(self.g, np.concatenate((z, [0])))

    def get_map(self):
        return self.image

    def set_map(self, img):
        self.image = img

    def calc_gmap(self, pixels):

        try:
            return self.g_map[pixels]
        except KeyError:
            pass

        x = np.linspace(-self.interprange, self.interprange, pixels)
        y = np.linspace(self.interprange, -self.interprange, pixels)

        xy = np.transpose(np.meshgrid(x, y)) / self.electrodescale

        e = array_project_radial_to3d(xy)

        gmap = self.calc_g(e.dot(np.transpose(self.locations)))
        self.g_map[pixels] = gmap
        return gmap

    def create_map(self, pixels=32):
        gm = self.calc_gmap(pixels)
        self.image = gm.dot(self.c[1:]) + self.c[0]

    def plot_map(self, axes=None, crange=None, offset=(0,0)):
        if axes is None: axes = plot.gca()
        if crange is str:
            if crange.lower() == 'channels':
                crange = None
            elif crange.lower() in ['full', 'map']:
                vru = np.nanmax(np.abs(self.image))
                vrl = -vru
        if crange is None:
            vru = np.nanmax(np.abs(self.z))
            vrl = -vru
        else:
            vrl, vru = crange
        head = self.path_head.deepcopy()
        head.vertices += offset

        if self.clipping == 'head':
            clip_path = (head, axes.transData)
        elif self.clipping == 'electrodes':
            verts = self._get_fence()
            codes = [path.Path.LINETO] * (len(verts) - 1)
            codes.insert(0, path.Path.MOVETO)
            clip_path = (path.Path(verts, codes), axes.transData)
        else:
            raise ValueError('unknown clipping mode: ', self.clipping)

        return axes.imshow(self.image, vmin=vrl, vmax=vru, clip_path=clip_path,
                           extent=(offset[0]-self.interprange, offset[0]+self.interprange,
                                   offset[1]-self.interprange, offset[1]+self.interprange))

    def plot_locations(self, axes=None, offset=(0,0)):
        if axes is None: axes = plot.gca()
        p2 = array_project_radial_to2d(self.locations) * self.electrodescale + offset
        axes.plot(p2[:, 0], p2[:, 1], 'k.')

    def plot_head(self, axes=None, offset=(0,0)):
        if axes is None: axes = plot.gca()
        head = self.path_head.deepcopy()
        nose = self.path_nose.deepcopy()
        head.vertices += offset
        nose.vertices += offset
        axes.add_patch(patches.PathPatch(head, facecolor='none', edgecolor=self.headcolor, lw=2))
        axes.add_patch(patches.PathPatch(nose, facecolor='none', edgecolor=self.headcolor, lw=2))

    def plot_circles(self, radius, axes=None, offset=(0,0)):
        if axes is None: axes = plot.gca()
        col = interp1d([-1, 0, 1], [[0, 1, 1], [0, 1, 0], [1, 1, 0]])
        for i in range(len(self.locations)):
            p3 = self.locations[i]
            p2 = array_project_radial_to2d(Vector.fromiterable(p3)) * self.electrodescale + offset
            circ = plot.Circle((p2[0, 0], p2[0, 1]), radius=radius, color=col(self.z[i]))
            axes.add_patch(circ)

    def _get_fence(self):
        if self.channel_fence is None:
            points = array_project_radial_to2d(self.locations) * self.electrodescale
            hull = ConvexHull(points)
            self.channel_fence = points[hull.vertices]
        return self.channel_fence


def topoplot(values, locations, headcolor=[0, 0, 0, 1], axes=None, offset=(0, 0)):
    """ Wrapper function for :class:`Topoplot.
    
    This function is only provided for convenience.
    """
    topo = Topoplot(headcolor=headcolor)
    topo.set_locations(locations)
    topo.set_values(values)
    topo.create_map()
    #h = topo.plot_map(axes, offset)
    topo.plot_map(axes=axes, offset=offset)
    topo.plot_locations(axes=axes, offset=offset)
    topo.plot_head(axes=axes, offset=offset)
    #plot.colorbar(h)
    return topo
