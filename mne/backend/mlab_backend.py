"""
Core visualization operations.
"""

import warnings
from ..surface import _normalize_vectors
from ..utils import _import_mlab
from tvtk.api import tvtk


class Renderer:
    def __init__(self):
        self.mlab = None
        self.fig = None

renderer = Renderer()


def _mlab_figure(**kwargs):
    """Create a Mayavi figure using our defaults."""
    from mayavi import mlab
    fig = mlab.figure(**kwargs)
    # If using modern VTK/Mayavi, improve rendering with FXAA
    if hasattr(getattr(fig.scene, 'renderer', None), 'use_fxaa'):
        fig.scene.renderer.use_fxaa = True
    return fig


def _toggle_mlab_render(fig, render):
    if renderer.mlab.options.backend != 'test':
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


def init(wsize, bg):
    renderer.mlab = _import_mlab()
    renderer.fig = _mlab_figure(bgcolor=bg, size=wsize)
    _toggle_mlab_render(renderer.fig, False)
    return 0


def set_interactive():
    renderer.fig.scene.interactor.interactor_style = \
        tvtk.InteractorStyleTerrain()


def add_surface(surf, color, opacity=1.0, backface_culling=False):
    mesh = _create_mesh_surf(surf, renderer.fig)
    surface = renderer.mlab.pipeline.surface(
        mesh, color=color, opacity=opacity, figure=renderer.fig)
    surface.actor.property.backface_culling = backface_culling


def add_spheres(centers, color, scale, opacity=1.0, backface_culling=False):
    surface = renderer.mlab.points3d(centers[:, 0], centers[:, 1],
                                     centers[:, 2], color=color,
                                     scale_factor=scale, opacity=opacity,
                                     figure=renderer.fig)
    surface.actor.property.backface_culling = backface_culling


def add_arrows(x, y, z, u, v, w, color, scale, resolution, opacity=1.0,
               scale_mode='none', scalars=None):
    renderer.mlab.quiver3d(x, y, z, u, v, w, mode='arrow', scale_factor=scale,
                           color=color, scale_mode=scale_mode,
                           resolution=resolution, scalars=scalars,
                           opacity=opacity, figure=renderer.fig)


def add_3d_arrows(x, y, z, u, v, w, color, scale, height, resolution,
                  center=None, opacity=1.0, backface_culling=False):
    quiv = renderer.mlab.quiver3d(x, y, z, u, v, w, mode='cylinder',
                                  color=color, scale_factor=scale,
                                  opacity=opacity, figure=renderer.fig)
    quiv.glyph.glyph_source.glyph_source.height = height
    quiv.glyph.glyph_source.glyph_source.center = center
    quiv.glyph.glyph_source.glyph_source.resolution = resolution
    quiv.actor.property.backface_culling = backface_culling


def show():
    _toggle_mlab_render(renderer.fig, True)
