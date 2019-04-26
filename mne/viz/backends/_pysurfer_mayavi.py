"""
Core visualization operations based on Mayavi.

Actual implementation of _Renderer and _Projection classes.
"""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Mainak Jas <mainak@neuro.hut.fi>
#          Mark Wronkiewicz <wronk.mark@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

import warnings
import numpy as np
from .base_renderer import _BaseRenderer
from ...surface import _normalize_vectors
from ...utils import (_import_mlab, _validate_type, SilenceStdout,
                      copy_base_doc_to_subclass_doc)


class _Projection(object):
    """Class storing projection information.

    Attributes
    ----------
    xy : array
        Result of 2d projection of 3d data.
    pts : Source
        Mayavi source handle.
    """

    def __init__(self, xy=None, pts=None):
        """Store input projection information into attributes."""
        self.xy = xy
        self.pts = pts

    def visible(self, state):
        """Modify visibility attribute of the source."""
        if self.pts is not None:
            self.pts.visible = state


@copy_base_doc_to_subclass_doc
class _Renderer(_BaseRenderer):
    """Class managing rendering scene.

    Attributes
    ----------
    mlab: mayavi.mlab
        Main Mayavi access point.
    fig: mlab.Figure
        Mayavi scene handle.
    """

    def __init__(self, fig=None, size=(600, 600), bgcolor=(0., 0., 0.),
                 name=None, show=False):
        self.mlab = _import_mlab()
        if fig is None:
            self.fig = _mlab_figure(figure=name, bgcolor=bgcolor, size=size)
        else:
            self.fig = fig
        if show is False:
            _toggle_mlab_render(self.fig, False)

    def scene(self):
        return self.fig

    def set_interactive(self):
        from tvtk.api import tvtk
        if self.fig.scene is not None:
            self.fig.scene.interactor.interactor_style = \
                tvtk.InteractorStyleTerrain()

    def mesh(self, x, y, z, triangles, color, opacity=1.0, shading=False,
             backface_culling=False, **kwargs):
        with warnings.catch_warnings(record=True):  # traits
            surface = self.mlab.triangular_mesh(x, y, z, triangles,
                                                color=color,
                                                opacity=opacity,
                                                figure=self.fig,
                                                **kwargs)
            surface.actor.property.shading = shading
            surface.actor.property.backface_culling = backface_culling
            return surface

    def contour(self, surface, scalars, contours, line_width=1.0, opacity=1.0,
                vmin=None, vmax=None, colormap=None):
        mesh = _create_mesh_surf(surface, self.fig, scalars=scalars)
        with warnings.catch_warnings(record=True):  # traits
            cont = self.mlab.pipeline.contour_surface(
                mesh, contours=contours, line_width=1.0, vmin=vmin, vmax=vmax,
                opacity=opacity, figure=self.fig)
            cont.module_manager.scalar_lut_manager.lut.table = colormap

    def surface(self, surface, color=None, opacity=1.0,
                vmin=None, vmax=None, colormap=None, scalars=None,
                backface_culling=False):
        # Make a solid surface
        mesh = _create_mesh_surf(surface, self.fig, scalars=scalars)
        with warnings.catch_warnings(record=True):  # traits
            surface = self.mlab.pipeline.surface(
                mesh, color=color, opacity=opacity, vmin=vmin, vmax=vmax,
                figure=self.fig)
            if colormap is not None:
                surface.module_manager.scalar_lut_manager.lut.table = colormap
            surface.actor.property.backface_culling = backface_culling

    def sphere(self, center, color, scale, opacity=1.0,
               resolution=8, backface_culling=False):
        if center.ndim == 1:
            x = center[0]
            y = center[1]
            z = center[2]
        elif center.ndim == 2:
            x = center[:, 0]
            y = center[:, 1]
            z = center[:, 2]
        surface = self.mlab.points3d(x, y, z, color=color,
                                     resolution=resolution,
                                     scale_factor=scale, opacity=opacity,
                                     figure=self.fig)
        surface.actor.property.backface_culling = backface_culling

    def quiver3d(self, x, y, z, u, v, w, color, scale, mode, resolution=8,
                 glyph_height=None, glyph_center=None, glyph_resolution=None,
                 opacity=1.0, scale_mode='none', scalars=None,
                 backface_culling=False):
        with warnings.catch_warnings(record=True):  # traits
            if mode == 'arrow':
                self.mlab.quiver3d(x, y, z, u, v, w, mode=mode,
                                   color=color, scale_factor=scale,
                                   scale_mode=scale_mode,
                                   resolution=resolution, scalars=scalars,
                                   opacity=opacity, figure=self.fig)
            elif mode == 'cone':
                self.mlab.quiver3d(x, y, z, u, v, w, color=color,
                                   mode=mode, scale_factor=scale,
                                   opacity=opacity, figure=self.fig)
            elif mode == 'cylinder':
                quiv = self.mlab.quiver3d(x, y, z, u, v, w, mode=mode,
                                          color=color, scale_factor=scale,
                                          opacity=opacity, figure=self.fig)
                quiv.glyph.glyph_source.glyph_source.height = glyph_height
                quiv.glyph.glyph_source.glyph_source.center = glyph_center
                quiv.glyph.glyph_source.glyph_source.resolution = \
                    glyph_resolution
                quiv.actor.property.backface_culling = backface_culling

    def text(self, x, y, text, width, color=(1.0, 1.0, 1.0)):
        with warnings.catch_warnings(record=True):  # traits
            self.mlab.text(x, y, text, width=width, color=color,
                           figure=self.fig)

    def show(self):
        if self.fig is not None:
            _toggle_mlab_render(self.fig, True)

    def close(self):
        self.mlab.close(self.fig)

    def set_camera(self, azimuth=None, elevation=None, distance=None,
                   focalpoint=None):
        with warnings.catch_warnings(record=True):  # traits
            with SilenceStdout():  # setting roll
                self.mlab.view(azimuth, elevation, distance,
                               focalpoint=focalpoint, figure=self.fig)

    def screenshot(self):
        with warnings.catch_warnings(record=True):  # traits
            return self.mlab.screenshot(self.fig)

    def project(self, xyz, ch_names):
        xy = _3d_to_2d(self.fig, xyz)
        xy = dict(zip(ch_names, xy))
        pts = self.fig.children[-1]

        return _Projection(xy=xy, pts=pts)


def _mlab_figure(**kwargs):
    """Create a Mayavi figure using our defaults."""
    from mayavi import mlab
    fig = mlab.figure(**kwargs)
    # If using modern VTK/Mayavi, improve rendering with FXAA
    if hasattr(getattr(fig.scene, 'renderer', None), 'use_fxaa'):
        fig.scene.renderer.use_fxaa = True
    return fig


def _toggle_mlab_render(fig, render):
    mlab = _import_mlab()
    if mlab.options.backend != 'test':
        fig.scene.disable_render = not render


def _create_mesh_surf(surf, fig=None, scalars=None, vtk_normals=True):
    """Create Mayavi mesh from MNE surf."""
    mlab = _import_mlab()
    x, y, z = surf['rr'].T
    with warnings.catch_warnings(record=True):  # traits
        mesh = mlab.pipeline.triangular_mesh_source(
            x, y, z, surf['tris'], scalars=scalars, figure=fig)
    if vtk_normals:
        mesh = mlab.pipeline.poly_data_normals(mesh)
        mesh.filter.compute_cell_normals = False
        mesh.filter.consistency = False
        mesh.filter.non_manifold_traversal = False
        mesh.filter.splitting = False
    else:
        # make absolutely sure these are normalized for Mayavi
        nn = surf['nn'].copy()
        _normalize_vectors(nn)
        mesh.data.point_data.normals = nn
        mesh.data.cell_data.normals = None
    return mesh


def _3d_to_2d(fig, xyz):
    """Convert 3d points to a 2d perspective using a Mayavi Scene."""
    from mayavi.core.scene import Scene

    _validate_type(fig, Scene, "fig", "Scene")
    xyz = np.column_stack([xyz, np.ones(xyz.shape[0])])

    # Transform points into 'unnormalized' view coordinates
    comb_trans_mat = _get_world_to_view_matrix(fig.scene)
    view_coords = np.dot(comb_trans_mat, xyz.T).T

    # Divide through by the fourth element for normalized view coords
    norm_view_coords = view_coords / (view_coords[:, 3].reshape(-1, 1))

    # Transform from normalized view coordinates to display coordinates.
    view_to_disp_mat = _get_view_to_display_matrix(fig.scene)
    xy = np.dot(view_to_disp_mat, norm_view_coords.T).T

    # Pull the first two columns since they're meaningful for 2d plotting
    xy = xy[:, :2]
    return xy


def _get_world_to_view_matrix(scene):
    """Return the 4x4 matrix to transform xyz space to the current view.

    This is a concatenation of the model view and perspective transforms.
    """
    from mayavi.core.ui.mayavi_scene import MayaviScene
    from tvtk.pyface.tvtk_scene import TVTKScene

    _validate_type(scene, (MayaviScene, TVTKScene), "scene",
                   "TVTKScene/MayaviScene")
    cam = scene.camera

    # The VTK method needs the aspect ratio and near and far
    # clipping planes in order to return the proper transform.
    scene_size = tuple(scene.get_size())
    clip_range = cam.clipping_range
    aspect_ratio = float(scene_size[0]) / scene_size[1]

    # Get the vtk matrix object using the aspect ratio we defined
    vtk_comb_trans_mat = cam.get_composite_projection_transform_matrix(
        aspect_ratio, clip_range[0], clip_range[1])
    vtk_comb_trans_mat = vtk_comb_trans_mat.to_array()
    return vtk_comb_trans_mat


def _get_view_to_display_matrix(scene):
    """Return the 4x4 matrix to convert view coordinates to display coordinates.

    It's assumed that the view should take up the entire window and that the
    origin of the window is in the upper left corner.
    """  # noqa: E501
    from mayavi.core.ui.mayavi_scene import MayaviScene
    from tvtk.pyface.tvtk_scene import TVTKScene

    _validate_type(scene, (MayaviScene, TVTKScene), "scene",
                   "TVTKScene/MayaviScene")

    # normalized view coordinates have the origin in the middle of the space
    # so we need to scale by width and height of the display window and shift
    # by half width and half height. The matrix accomplishes that.
    x, y = tuple(scene.get_size())
    view_to_disp_mat = np.array([[x / 2.0,       0.,   0.,   x / 2.0],
                                 [0.,      -y / 2.0,   0.,   y / 2.0],
                                 [0.,            0.,   1.,        0.],
                                 [0.,            0.,   0.,        1.]])
    return view_to_disp_mat
