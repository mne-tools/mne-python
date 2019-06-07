"""Core visualization operations based on ipyvolume."""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Oleh Kozynets <ok7mailbox@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#
# License: Simplified BSD

import warnings
import numpy as np
from ._utils import _translate
from ._utils import _rotate
from ._utils import _create_sphere
from ._utils import _create_cone
from ._utils import _create_cylinder
from ._utils import _create_arrow
from ._utils import _isoline

from ..base_renderer import _BaseRenderer
from .._utils import _get_colormap_from_array, _get_color_from_scalars
from ....utils import copy_base_doc_to_subclass_doc
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import ipyvolume as ipv
    from pythreejs import (BlendFactors, BlendingMode, Equations,
                           ShaderMaterial, Side)


class _Projection(object):
    """Class storing projection information.

    Attributes
    ----------
    xy : array
        Result of 2d projection of 3d data.
    pts : None
        Scene sensors handle.
    """

    def __init__(self, xy=None, pts=None):
        """Store input projection information into attributes."""
        raise NotImplementedError('Projection class is not implemented yet')

    def visible(self, state):
        """Modify visibility attribute of the sensors."""
        raise NotImplementedError('This feature is not implemented yet')


@copy_base_doc_to_subclass_doc
class _Renderer(_BaseRenderer):
    """Class managing rendering scene created with ipyvolume.

    Attributes
    ----------
    plotter: ipyvolume.widgets.Figure
        The scene handler.
    """

    def __init__(self, fig=None, size=(600, 600), bgcolor=(0., 0., 0.),
                 name="Ipyvolume Scene", show=False):
        """Set up the scene.

        Parameters
        ----------
        fig: ipyvolume.widgets.Figure
            The scene handler.
        size: tuple
            The dimensions of the context window: (width, height).
        bgcolor: tuple
            The color definition of the background: (red, green, blue).
        name: str | None
            The name of the scene.
        """
        self.off_screen = False
        self.name = name
        self.vmin = 0.0
        self.vmax = 0.0

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            if fig is None:
                fig_w, fig_h = size
                self.plotter = ipv.figure(width=fig_w, height=fig_h)
                self.plotter.animation = 0
                ipv.style.box_off()
                ipv.style.axes_off()
                bgcolor = tuple(int(c) for c in bgcolor)
                ipv.style.background_color('#%02x%02x%02x' % bgcolor)
            else:
                self.plotter = ipv.figure(key=fig)
            self.plotter.camera.up = (0.0, 0.0, 1.0)

    def update_limits(self, x, y, z):
        self.vmin = np.min([self.vmin, x.min(), y.min(), z.min()])
        self.vmax = np.max([self.vmax, x.max(), y.max(), z.max()])

    def scene(self):
        return self.plotter

    def set_interactive(self):
        pass

    def mesh(self, x, y, z, triangles, color, opacity=1.0, shading=False,
             backface_culling=False, **kwargs):
        # opacity for overlays will be provided as part of color
        color = _color2rgba(color, opacity)
        mesh = ipv.plot_trisurf(x, y, z, triangles=triangles, color=color)
        _add_transparent_material(mesh, opacity, backface_culling)
        self.update_limits(x, y, z)
        return mesh

    def contour(self, surface, scalars, contours, line_width=1.0, opacity=1.0,
                vmin=None, vmax=None, colormap=None,
                normalized_colormap=False):
        surf_verts = surface['rr']
        surf_faces = surface['tris']

        cmap = _get_colormap_from_array(colormap, normalized_colormap)

        if isinstance(contours, int):
            n_levels = contours
            levels = np.linspace(vmin, vmax, num=contours)
        else:
            n_levels = len(contours)
            levels = np.array(contours)

        if scalars is not None:
            color = _get_color_from_scalars(cmap, scalars, vmin, vmax)
        if len(color) == 3:
            color = np.append(color, opacity)

        verts, faces, vertex_level, _ = \
            _isoline(vertices=surf_verts, tris=surf_faces,
                     vertex_data=scalars, levels=levels)

        vertex_level = _normalize_array(vertex_level)
        vertex_level = np.array(vertex_level * n_levels).astype(np.int)
        vertex_level[(vertex_level == n_levels)] = n_levels - 1

        x, y, z = verts.T
        colors = [color[level] for level in vertex_level]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            ipv.plot_trisurf(x, y, z, color=colors, lines=faces.flatten())

        self.update_limits(x, y, z)

    def surface(self, surface, color=None, opacity=1.0,
                vmin=None, vmax=None, colormap=None,
                normalized_colormap=False, scalars=None,
                backface_culling=False):
        cmap = _get_colormap_from_array(colormap, normalized_colormap)

        vertices = np.array(surface['rr'])
        x, y, z = vertices.T
        triangles = np.array(surface['tris'])

        if scalars is not None:
            color = _get_color_from_scalars(cmap, scalars, vmin, vmax)
        if len(color) == 3:
            color = np.append(color, opacity)

        mesh = ipv.plot_trisurf(x, y, z, triangles=triangles, color=color)
        _add_transparent_material(mesh, opacity, backface_culling)
        self.update_limits(x, y, z)

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

        x, y, z = acc_vertices.T
        color = np.append(color, opacity)

        mesh = ipv.plot_trisurf(x, y, z, triangles=acc_faces, color=color)
        _add_transparent_material(mesh, opacity, backface_culling)
        self.update_limits(x, y, z)

    def quiver3d(self, x, y, z, u, v, w, color, scale, mode, resolution=8,
                 glyph_height=None, glyph_center=None, glyph_resolution=None,
                 opacity=1.0, scale_mode='none', scalars=None,
                 backface_culling=False):
        source = np.column_stack((x, y, z))
        destination = source + np.column_stack((u, v, w))
        arr_pos = np.array(list(zip(source, destination)))
        arr_pos.reshape(-1, 3)

        acc_vertices = list()
        acc_faces = list()
        foffset = 0
        voffset = 0
        for i in range(len(source)):
            scalar = scalars[i] if scalars is not None else None
            orig_vertices, orig_faces, mat = \
                _create_quiver(mode=mode, source=source[i, :],
                               destination=destination[i, :],
                               scale=scale,
                               scale_mode=scale_mode,
                               scalar=scalar)
            n_faces = len(orig_faces)
            n_vertices = len(orig_vertices)

            # accumulate faces
            current_faces = orig_faces + voffset
            acc_faces.append(current_faces)
            foffset += n_faces

            # accumulate vertices
            mat = mat.T
            current_vertices = np.c_[orig_vertices, np.ones(n_vertices)]
            current_vertices = mat.dot(current_vertices.T)
            current_vertices = current_vertices.T[:, 0:3]
            acc_vertices.append(current_vertices)
            voffset += n_vertices

        # concatenate
        faces = np.concatenate(acc_faces, axis=0)
        vertices = np.concatenate(acc_vertices, axis=0)

        x, y, z = vertices.T
        mesh = ipv.plot_trisurf(x, y, z, triangles=faces, color=color)
        _add_transparent_material(mesh, opacity, backface_culling)
        self.update_limits(x, y, z)

    def text(self, x, y, text, width, color=(1.0, 1.0, 1.0)):
        pass

    def show(self):
        ipv.xyzlim(self.vmin, self.vmax)
        ipv.show()

    def close(self):
        ipv.clear()

    def set_camera(self, azimuth=0.0, elevation=0.0, distance=1.0,
                   focalpoint=(0, 0, 0)):
        ipv.view(azimuth=azimuth,
                 elevation=elevation,
                 distance=distance * 2.0)

    def screenshot(self):
        pass

    def project(self):
        pass


def _create_quiver(mode, source, destination, scale, scale_mode='none',
                   scalar=None, resolution=8):
    v1 = destination - source
    vn = np.linalg.norm(v1)
    v1 = v1 / vn

    v2 = np.array([0, 0, 1])

    cosangle = np.dot(v1, v2)
    axis = np.cross(v2, v1)

    length = vn
    if scale_mode == 'none':
        length = scale
    elif scale_mode == 'scalar' and scalar is not None:
        length = scale * scalar

    radius = length / 20.0

    if mode == 'arrow':
        cone_radius = radius * 3.0
        cone_length = length / 4.0
        vertices, faces = _create_arrow(rows=resolution, cols=resolution,
                                        length=length, radius=radius,
                                        cone_radius=cone_radius,
                                        cone_length=cone_length)
    elif mode == 'cone':
        vertices, faces = _create_cone(cols=resolution, length=length,
                                       radius=radius)
    elif mode == 'cylinder':
        vertices, faces = _create_cylinder(rows=resolution, cols=resolution,
                                           length=length,
                                           radius=[radius, radius])

    # apply transform
    mat = np.identity(4)
    if cosangle != 1:
        mat = np.dot(mat, _rotate(np.degrees(np.arccos(cosangle)), axis))
    mat = np.dot(mat, _translate(source))
    return vertices, faces, mat


def _add_transparent_material(mesh, opacity, backface_culling):
    """Change the mesh material so it will support transparency."""
    mat = ShaderMaterial()
    mat.alphaTest = opacity
    mat.depthTest = True
    mat.blending = BlendingMode.CustomBlending
    mat.blendDst = BlendFactors.OneMinusSrcAlphaFactor
    mat.blendEquation = Equations.AddEquation
    mat.blendSrc = BlendFactors.SrcAlphaFactor
    mat.transparent = True
    if backface_culling:
        mat.side = Side.BackSide
    else:
        mat.side = Side.DoubleSide

    mesh.material = mat


def _color2rgba(color, opacity):
    """Update color opacity values."""
    if opacity is None:
        # no need to update colors
        return color

    color = np.array(color)
    if color.ndim == 1:
        color = np.expand_dims(color, axis=0)
    _, n_components = color.shape
    if n_components == 4:
        color[:, -1] = opacity
    elif n_components == 3:
        # add new axis
        rgba_color = np.zeros((len(color), 4))
        rgba_color[:, :-1] = color
        rgba_color[:, -1] = opacity
        color = rgba_color

    return color


def _normalize_array(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


def _close_all():
    # XXX This is not implemented yet
    pass
