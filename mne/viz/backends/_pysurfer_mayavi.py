"""
Core visualization operations.
"""

import warnings
from ...surface import _normalize_vectors
from ...utils import _import_mlab


class Renderer:
    def __init__(self, size=(600, 600), bgcolor=(0., 0., 0.)):
        self.mlab = _import_mlab()
        self.fig = _mlab_figure(bgcolor=bgcolor, size=size)
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
        surface = self.mlab.triangular_mesh(x, y, z, color=color,
                                            triangles=triangles,
                                            opacity=opacity,
                                            figure=self.fig,
                                            **kwargs)
        surface.actor.property.shading = shading
        surface.actor.property.backface_culling = backface_culling

    def contour(self, surface, scalars, contours, line_width=1.0, opacity=1.0,
                vmin=None, vmax=None, colormap=None):
        mesh = _create_mesh_surf(surface, self.fig, scalars=scalars)
        cont = self.mlab.pipeline.contour_surface(
            mesh, contours=contours, line_width=1.0, vmin=vmin, vmax=vmax,
            opacity=opacity, figure=self.fig)
        cont.module_manager.scalar_lut_manager.lut.table = colormap

    def surface(self, surface, color=(0.7, 0.7, 0.7), opacity=1.0,
                vmin=None, vmax=None, colormap=None,
                backface_culling=False):
        # Make a solid surface
        mesh = _create_mesh_surf(surface, self.fig)
        surface = self.mlab.pipeline.surface(
            mesh, color=color, opacity=opacity, vmin=vmin, vmax=vmax,
            figure=self.fig)
        if colormap is not None:
            surface.module_manager.scalar_lut_manager.lut.table = colormap
        surface.actor.property.backface_culling = backface_culling

    def sphere(self, center, color, scale, opacity=1.0,
               backface_culling=False):
        surface = self.mlab.points3d(center[:, 0], center[:, 1],
                                     center[:, 2], color=color,
                                     scale_factor=scale, opacity=opacity,
                                     figure=self.fig)
        surface.actor.property.backface_culling = backface_culling

    def quiver3d(self, x, y, z, u, v, w, color, scale, mode, resolution=8,
                 glyph_height=None, glyph_center=None, glyph_resolution=None,
                 opacity=1.0, scale_mode='none', scalars=None,
                 backface_culling=False):
        if mode == 'arrow':
            self.mlab.quiver3d(x, y, z, u, v, w, mode=mode,
                               color=color, scale_factor=scale,
                               scale_mode=scale_mode,
                               resolution=resolution, scalars=scalars,
                               opacity=opacity, figure=self.fig)
        elif mode == 'cylinder':
            quiv = self.mlab.quiver3d(x, y, z, u, v, w, mode=mode,
                                      opacity=opacity, figure=self.fig)
            quiv.glyph.glyph_source.glyph_source.height = glyph_height
            quiv.glyph.glyph_source.glyph_source.center = glyph_center
            quiv.glyph.glyph_source.glyph_source.resolution = glyph_resolution
            quiv.actor.property.backface_culling = backface_culling

    def text(self, x, y, text, width):
        self.mlab.text(x, y, text, width=width, figure=self.fig)

    def show(self):
        _toggle_mlab_render(self.fig, True)

    def set_camera(self, azimuth=None, elevation=None, distance=None,
                   focalpoint=None):
        self.mlab.view(azimuth, elevation, distance,
                       focalpoint=focalpoint, figure=self.fig)

    def screenshot(self):
        return self.mlab.screenshot(self.fig)


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
