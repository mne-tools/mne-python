"""
Core visualization operations based on ipyvolume.

Actual implementation of _Renderer and _Projection classes.
"""

import ipyvolume as ipv
import numpy as np
from pythreejs import (BlendFactors, BlendingMode, Equations, ShaderMaterial,
                       Side)


class _Renderer(object):
    """Ipyvolume-based rendering."""
    def __init__(self, fig=None, size=(600, 600), bgcolor=(0., 0., 0.),
                 name="Ipyvolume Scene", show=False):
        # why do I need to check for it?
        from mne.viz.backends.renderer import MNE_3D_BACKEND_TEST_DATA
        self.off_screen = False
        self.name = name

        if MNE_3D_BACKEND_TEST_DATA:
            self.off_screen = True

        if fig is None:
            fig_w, fig_h = size
            self.plotter = ipv.figure(width=fig_w, height=fig_h)
            self.plotter.animation = 0
            ipv.style.box_off()
            ipv.style.axes_off()
            ipv.style.background_color(bgcolor)
        else:
            # not sure this is the exact behavior, needs a thorough look
            self.plotter = ipv.figure(key=fig)

    def scene(self):
        return self.plotter

    def set_interactive(self):
        # what is this for?
        #self.plotter.enable_terrain_style()
        pass

    def mesh(self, x, y, z, triangles, color, opacity=1.0, shading=False,
             backface_culling=False, **kwargs):
        color = np.append(color, opacity)
        mesh = ipv.plot_trisurf(x, y, z, triangles=triangles, color=color)
        _add_transperent_material(mesh)

    def contour(self, surface, scalars, contours, line_width=1.0, opacity=1.0,
                vmin=None, vmax=None, colormap=None):
        from matplotlib import cm
        from matplotlib.colors import ListedColormap
        from ipyvolume.pylab import plot

        vertices = surface['rr']
        tris = surface['tris']

        if colormap is None:
            cmap = cm.get_cmap('coolwarm')
        else:
            cmap = ListedColormap(colormap / 255.0)

        if isinstance(contours, int):
            levels = np.linspace(vmin, vmax, num=contours)
        else:
            levels = np.array(contours)

        if scalars is not None:
            if vmin is None:
                vmin = min(scalars)
            if vmax is None:
                vmax = max(scalars)
            nscalars = (scalars - vmin) / (vmax - vmin)
            color = cmap(nscalars)
        else:
            color = np.append(color, opacity)

        verts, faces, _, _ = _isoline(vertices=vertices, tris=tris,
                                vertex_data=scalars,
                                levels=levels)

        x = verts[:, 0]
        y = verts[:, 1]
        z = verts[:, 2]
        plot(x, y, z, color=color)

    def surface(self, surface, color=None, opacity=1.0,
                vmin=None, vmax=None, colormap=None, scalars=None,
                backface_culling=False):
        # what is vmin, vmax for?
        # should we use ipv.plot_surface?
        from matplotlib import cm
        from matplotlib.colors import ListedColormap

        if colormap is None:
            cmap = cm.get_cmap('coolwarm')
        else:
            cmap = ListedColormap(colormap / 255.0)

        vertices = np.array(surface['rr'])
        x = vertices[:, 0]
        y = vertices[:, 1]
        z = vertices[:, 2]
        triangles = np.array(surface['tris'])

        if scalars is not None:
            if vmin is None:
                vmin = min(scalars)
            if vmax is None:
                vmax = max(scalars)
            nscalars = (scalars - vmin) / (vmax - vmin)
            color = cmap(nscalars)
        else:
            color = np.append(color, opacity)

        mesh = ipv.plot_trisurf(x, y, z, triangles=triangles, color=color)
        _add_transperent_material(mesh)

    def sphere(self, center, color, scale, opacity=1.0, resolution=8,
               backface_culling=False):
        default_sphere_radius = 0.5
        if center.ndim == 1:
            center = np.array([center])

        # fetch original data
        orig_vertices, orig_faces = \
            _create_sphere(radius=scale * default_sphere_radius,
                           cols=resolution, rows=resolution)
        n_vertices = len(orig_vertices)
        orig_vertices = np.c_[orig_vertices, np.ones(n_vertices)].T
        n_faces = len(orig_faces)
        n_center = len(center)
        voffset = 0
        foffset = 0

        # accumulate mesh data
        acc_vertices = np.empty((n_center * n_vertices, 3), dtype=np.float32)
        acc_faces = np.empty((n_center * n_faces, 3), dtype=np.uint32)
        for c in center:
            # apply index shifting and accumulate faces
            current_faces = orig_faces + voffset
            acc_faces[foffset:foffset + n_faces, :] = current_faces
            foffset += n_faces

            # apply translation and accumulate vertices
            mat = np.eye(4)
            mat[:-1, 3] = c
            current_vertices = mat.dot(orig_vertices)
            acc_vertices[voffset:voffset + n_vertices, :] = \
                current_vertices.T[:, 0:3]
            voffset += n_vertices

        x = acc_vertices[:, 0]
        y = acc_vertices[:, 1]
        z = acc_vertices[:, 2]
        color = np.append(color, opacity)

        ipv.xyzlim(-1, 1)
        mesh = ipv.plot_trisurf(x, y, z, triangles=acc_faces, color=color)
        _add_transperent_material(mesh)

    def quiver3d(self, x, y, z, u, v, w, color, scale, mode, resolution=8,
                 glyph_height=None, glyph_center=None, glyph_resolution=None,
                 opacity=1.0, scale_mode='none', scalars=None,
                 backface_culling=False):
        # possible geo/marker values
        # this.geos = {
        #     diamond: this.geo_diamond,
        #     box: this.geo_box,
        #     arrow: this.geo_arrow,
        #     sphere: this.geo_sphere,
        #     cat: this.geo_cat,
        #     square_2d: this.geo_square_2d,
        #     point_2d: this.geo_point_2d,
        #     circle_2d: this.geo_circle_2d,
        #     triangle_2d: this.geo_triangle_2d
        # }
        color = np.append(color, opacity)
        scatter = ipv.quiver(x, y, z, u, v, w, marker=mode, color=color)
        _add_transperent_material(scatter)

    def text(self, x, y, text, width, color=(1.0, 1.0, 1.0)):
        pass
        # how to add text in ipyvolume/pythreejs?
        # self.plotter.add_text(text, position=(x, y),
        #                       font_size=int(width * 100),
        #                       color=color)

    def show(self):
        ipv.show()

    def close(self):
        pass
        # ???
        # self.plotter.close()

    def set_camera(self, azimuth=0.0, elevation=0.0, distance=1.0,
                   focalpoint=(0, 0, 0)):
        ipv.view(azimuth=azimuth, elevation=elevation, distance=distance)


def _create_sphere(rows, cols, radius, offset=True):
    verts = np.empty((rows + 1, cols, 3), dtype=np.float32)

    # compute vertices
    phi = (np.arange(rows + 1) * np.pi / rows).reshape(rows + 1, 1)
    s = radius * np.sin(phi)
    verts[..., 2] = radius * np.cos(phi)
    th = ((np.arange(cols) * 2 * np.pi / cols).reshape(1, cols))
    if offset:
        # rotate each row by 1/2 column
        th = th + ((np.pi / cols) * np.arange(rows+1).reshape(rows+1, 1))
    verts[..., 0] = s * np.cos(th)
    verts[..., 1] = s * np.sin(th)
    # remove redundant vertices from top and bottom
    verts = verts.reshape((rows+1)*cols, 3)[cols-1:-(cols-1)]

    # compute faces
    faces = np.empty((rows*cols*2, 3), dtype=np.uint32)
    rowtemplate1 = (((np.arange(cols).reshape(cols, 1) +
                      np.array([[1, 0, 0]])) % cols) +
                    np.array([[0, 0, cols]]))
    rowtemplate2 = (((np.arange(cols).reshape(cols, 1) +
                      np.array([[1, 0, 1]])) % cols) +
                    np.array([[0, cols, cols]]))
    for row in range(rows):
        start = row * cols * 2
        faces[start:start+cols] = rowtemplate1 + row * cols
        faces[start+cols:start+(cols*2)] = rowtemplate2 + row * cols
    # cut off zero-area triangles at top and bottom
    faces = faces[cols:-cols]

    # adjust for redundant vertices that were removed from top and bottom
    vmin = cols-1
    faces[faces < vmin] = vmin
    faces -= vmin
    vmax = verts.shape[0]-1
    faces[faces > vmax] = vmax
    return verts, faces


def _add_transperent_material(mesh):
    """Changes the mesh material so it will support transparency."""
    mat = ShaderMaterial()
    mat.alphaTest = 0.1
    mat.blending = BlendingMode.CustomBlending
    mat.blendDst = BlendFactors.OneMinusSrcAlphaFactor
    mat.blendEquation = Equations.AddEquation
    mat.blendSrc = BlendFactors.SrcAlphaFactor
    mat.transparent = True
    mat.side = Side.DoubleSide

    mesh.material = mat


def _isoline(vertices, tris, vertex_data, levels):
    """Generate an isocurve from vertex data in a surface mesh.
    Parameters
    ----------
    vertices : ndarray, shape (Nv, 3)
        Vertex coordinates.
    tris : ndarray, shape (Nf, 3)
        Indices of triangular element into the vertices array.
    vertex_data : ndarray, shape (Nv,)
        data at vertex.
    levels : ndarray, shape (Nl,)
        Levels at which to generate an isocurve
    Returns
    -------
    lines : ndarray, shape (Nvout, 3)
        Vertex coordinates for lines points
    connects : ndarray, shape (Ne, 2)
        Indices of line element into the vertex array.
    vertex_level: ndarray, shape (Nvout,)
        level for vertex in lines
    Notes
    -----
    Uses a marching squares algorithm to generate the isolines.
    """

    lines = None
    connects = None
    vertex_level = None
    level_index = None
    if not all([isinstance(x, np.ndarray) for x in (vertices, tris,
                vertex_data, levels)]):
        raise ValueError('all inputs must be numpy arrays')
    if vertices.shape[1] <= 3:
        verts = vertices
    elif vertices.shape[1] == 4:
        verts = vertices[:, :-1]
    else:
        verts = None
    if (verts is not None and tris.shape[1] == 3 and
            vertex_data.shape[0] == verts.shape[0]):
        edges = np.vstack((tris.reshape((-1)),
                           np.roll(tris, -1, axis=1).reshape((-1)))).T
        edge_datas = vertex_data[edges]
        edge_coors = verts[edges].reshape(tris.shape[0]*3, 2, 3)
        for lev in levels:
            # index for select edges with vertices have only False - True
            # or True - False at extremity
            index = (edge_datas >= lev)
            index = index[:, 0] ^ index[:, 1]  # xor calculation
            # Selectect edge
            edge_datas_Ok = edge_datas[index, :]
            xyz = edge_coors[index]
            # Linear interpolation
            ratio = np.array([(lev - edge_datas_Ok[:, 0]) /
                              (edge_datas_Ok[:, 1] - edge_datas_Ok[:, 0])])
            point = xyz[:, 0, :] + ratio.T * (xyz[:, 1, :] - xyz[:, 0, :])
            nbr = point.shape[0]//2
            if connects is not None:
                connect = np.arange(0, nbr*2).reshape((nbr, 2)) + \
                    len(lines)
                connects = np.append(connects, connect, axis=0)
                lines = np.append(lines, point, axis=0)
                vertex_level = np.append(vertex_level,
                                         np.zeros(len(point)) +
                                         lev)
                level_index = np.append(level_index, np.array(len(point)))
            else:
                lines = point
                connects = np.arange(0, nbr*2).reshape((nbr, 2))
                vertex_level = np.zeros(len(point)) + lev
                level_index = np.array(len(point))

            vertex_level = vertex_level.reshape((vertex_level.size, 1))

    return lines, connects, vertex_level, level_index
