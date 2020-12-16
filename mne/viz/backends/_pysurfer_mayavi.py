"""
Core visualization operations based on Mayavi.

Actual implementation of _Renderer and _Projection classes.
"""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Mainak Jas <mainak@neuro.hut.fi>
#          Mark Wronkiewicz <wronk.mark@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

from contextlib import contextmanager
import warnings
import numpy as np

from mayavi.core.scene import Scene
from mayavi.core.ui.mayavi_scene import MayaviScene
from tvtk.pyface.tvtk_scene import TVTKScene

from .base_renderer import _BaseRenderer
from ._utils import _check_color, _alpha_blend_background, ALLOWED_QUIVER_MODES
from ...surface import _normalize_vectors
from ...utils import (_import_mlab, _validate_type, SilenceStdout,
                      copy_base_doc_to_subclass_doc, _check_option)


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

    def __init__(self, fig=None, size=(600, 600), bgcolor='black',
                 name=None, show=False, shape=(1, 1), smooth_shading=True):
        if bgcolor is not None:
            bgcolor = _check_color(bgcolor)
        self.mlab = _import_mlab()
        self.shape = shape
        if fig is None:
            self.fig = _mlab_figure(figure=name, bgcolor=bgcolor, size=size)
        elif isinstance(fig, int):
            self.fig = _mlab_figure(figure=fig, bgcolor=bgcolor, size=size)
        else:
            self.fig = fig
        self.fig._window_size = size
        _toggle_mlab_render(self.fig, show)

    @property
    def figure(self):  # cross-compat w/PyVista
        return self.fig

    def subplot(self, x, y):
        pass

    def scene(self):
        return self.fig

    def set_interaction(self, interaction):
        from tvtk.api import tvtk
        if self.fig.scene is not None:
            self.fig.scene.interactor.interactor_style = \
                getattr(tvtk, f'InteractorStyle{interaction.capitalize()}')()

    def mesh(self, x, y, z, triangles, color, opacity=1.0, shading=False,
             backface_culling=False, scalars=None, colormap=None,
             vmin=None, vmax=None, interpolate_before_map=True,
             representation='surface', line_width=1., normals=None,
             polygon_offset=None, **kwargs):
        # normals and pickable are unused
        kwargs.pop('pickable', None)
        del normals

        if color is not None:
            color = _check_color(color)
        if color is not None and isinstance(color, np.ndarray) \
           and color.ndim > 1:
            if color.shape[1] == 3:
                vertex_color = np.c_[color, np.ones(len(color))] * 255.0
            else:
                vertex_color = color * 255.0
            # create a lookup table to enable one color per vertex
            scalars = np.arange(len(color))
            color = None
        else:
            vertex_color = None
        with warnings.catch_warnings(record=True):  # traits
            surface = self.mlab.triangular_mesh(x, y, z, triangles,
                                                color=color,
                                                scalars=scalars,
                                                opacity=opacity,
                                                figure=self.fig,
                                                vmin=vmin,
                                                vmax=vmax,
                                                representation=representation,
                                                line_width=line_width,
                                                **kwargs)

            l_m = surface.module_manager.scalar_lut_manager
            if vertex_color is not None:
                l_m.lut.table = vertex_color
            elif isinstance(colormap, np.ndarray):
                if colormap.dtype == np.uint8:
                    l_m.lut.table = colormap
                elif colormap.dtype == np.float64:
                    l_m.load_lut_from_list(colormap)
                else:
                    raise TypeError('Expected type for colormap values are'
                                    ' np.float64 or np.uint8: '
                                    '{} was given'.format(colormap.dtype))
            elif colormap is not None:
                from matplotlib.cm import get_cmap
                l_m.load_lut_from_list(
                    get_cmap(colormap)(np.linspace(0, 1, 256)))
            else:
                assert color is not None
            surface.actor.property.shading = shading
            surface.actor.property.backface_culling = backface_culling
        return surface

    def contour(self, surface, scalars, contours, width=1.0, opacity=1.0,
                vmin=None, vmax=None, colormap=None,
                normalized_colormap=False, kind='line', color=None):
        mesh = _create_mesh_surf(surface, self.fig, scalars=scalars)
        with warnings.catch_warnings(record=True):  # traits
            cont = self.mlab.pipeline.contour_surface(
                mesh, contours=contours, line_width=width, vmin=vmin,
                vmax=vmax, opacity=opacity, figure=self.fig)
            cont.module_manager.scalar_lut_manager.lut.table = colormap
            return cont

    def surface(self, surface, color=None, opacity=1.0,
                vmin=None, vmax=None, colormap=None,
                normalized_colormap=False, scalars=None,
                backface_culling=False, polygon_offset=None):
        if color is not None:
            color = _check_color(color)
        if normalized_colormap:
            colormap = colormap * 255.0
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
               resolution=8, backface_culling=False,
               radius=None):
        color = _check_color(color)
        center = np.atleast_2d(center)
        x, y, z = center.T
        surface = self.mlab.points3d(x, y, z, color=color,
                                     resolution=resolution,
                                     scale_factor=scale, opacity=opacity,
                                     figure=self.fig)
        surface.actor.property.backface_culling = backface_culling

    def tube(self, origin, destination, radius=0.001, color='white',
             scalars=None, vmin=None, vmax=None, colormap='RdBu',
             normalized_colormap=False, reverse_lut=False):
        color = _check_color(color)
        origin = np.atleast_2d(origin)
        destination = np.atleast_2d(destination)
        if scalars is None:
            # TODO: iterating over each tube rather than plotting in
            #       one call may be slow.
            #       See https://github.com/mne-tools/mne-python/issues/7644
            for idx in range(origin.shape[0]):
                surface = self.mlab.plot3d([origin[idx, 0],
                                            destination[idx, 0]],
                                           [origin[idx, 1],
                                            destination[idx, 1]],
                                           [origin[idx, 2],
                                            destination[idx, 2]],
                                           tube_radius=radius,
                                           color=color,
                                           figure=self.fig)
        else:
            for idx in range(origin.shape[0]):
                surface = self.mlab.plot3d([origin[idx, 0],
                                            destination[idx, 0]],
                                           [origin[idx, 1],
                                            destination[idx, 1]],
                                           [origin[idx, 2],
                                            destination[idx, 2]],
                                           [scalars[idx, 0],
                                            scalars[idx, 1]],
                                           tube_radius=radius,
                                           vmin=vmin,
                                           vmax=vmax,
                                           colormap=colormap,
                                           figure=self.fig)
        surface.module_manager.scalar_lut_manager.reverse_lut = reverse_lut
        return surface

    def quiver3d(self, x, y, z, u, v, w, color, scale, mode, resolution=8,
                 glyph_height=None, glyph_center=None, glyph_resolution=None,
                 opacity=1.0, scale_mode='none', scalars=None,
                 backface_culling=False, colormap=None, vmin=None, vmax=None,
                 line_width=2., name=None, solid_transform=None):
        _check_option('mode', mode, ALLOWED_QUIVER_MODES)
        color = _check_color(color)
        with warnings.catch_warnings(record=True):  # traits
            if mode in ('arrow', '2darrow'):
                self.mlab.quiver3d(x, y, z, u, v, w, mode=mode,
                                   color=color, scale_factor=scale,
                                   scale_mode=scale_mode,
                                   resolution=resolution, scalars=scalars,
                                   opacity=opacity, figure=self.fig)
            elif mode in ('cone', 'sphere', 'oct'):
                use_mode = 'sphere' if mode == 'oct' else mode
                quiv = self.mlab.quiver3d(x, y, z, u, v, w, color=color,
                                          mode=use_mode, scale_factor=scale,
                                          opacity=opacity, figure=self.fig)
                if mode == 'sphere':
                    quiv.glyph.glyph_source.glyph_source.center = 0., 0., 0.
                elif mode == 'oct':
                    _oct_glyph(quiv.glyph.glyph_source, solid_transform)
            else:
                assert mode == 'cylinder', mode  # should be guaranteed above
                quiv = self.mlab.quiver3d(x, y, z, u, v, w, mode=mode,
                                          color=color, scale_factor=scale,
                                          opacity=opacity, figure=self.fig)
                if glyph_height is not None:
                    quiv.glyph.glyph_source.glyph_source.height = glyph_height
                if glyph_center is not None:
                    quiv.glyph.glyph_source.glyph_source.center = glyph_center
                if glyph_resolution is not None:
                    quiv.glyph.glyph_source.glyph_source.resolution = \
                        glyph_resolution
                quiv.actor.property.backface_culling = backface_culling

    def text2d(self, x_window, y_window, text, size=14, color='white',
               justification=None):
        if color is not None:
            color = _check_color(color)
        size = 14 if size is None else size
        with warnings.catch_warnings(record=True):  # traits
            text = self.mlab.text(x_window, y_window, text, color=color,
                                  figure=self.fig)
            text.property.font_size = size
            text.actor.text_scale_mode = 'viewport'
            if isinstance(justification, str):
                text.property.justification = justification

    def text3d(self, x, y, z, text, scale, color='white'):
        color = _check_color(color)
        with warnings.catch_warnings(record=True):  # traits
            self.mlab.text3d(x, y, z, text, scale=scale, color=color,
                             figure=self.fig)

    def scalarbar(self, source, color="white", title=None, n_labels=4,
                  bgcolor=None):
        with warnings.catch_warnings(record=True):  # traits
            bar = self.mlab.scalarbar(source, title=title, nb_labels=n_labels)
        if color is not None:
            bar.label_text_property.color = _check_color(color)
        if bgcolor is not None:
            from tvtk.api import tvtk
            bgcolor = np.asarray(bgcolor)
            bgcolor = np.append(bgcolor, 1.0) * 255.
            cmap = source.module_manager.scalar_lut_manager
            lut = cmap.lut
            ctable = lut.table.to_array()
            cbar_lut = tvtk.LookupTable()
            cbar_lut.deep_copy(lut)
            vals = _alpha_blend_background(ctable, bgcolor)
            cbar_lut.table.from_array(vals)
            cmap.scalar_bar.lookup_table = cbar_lut

    def show(self):
        if self.fig is not None:
            _toggle_mlab_render(self.fig, True)

    def close(self):
        _close_3d_figure(figure=self.fig)

    def set_camera(self, azimuth=None, elevation=None, distance=None,
                   focalpoint=None, roll=None, reset_camera=None):
        _set_3d_view(figure=self.fig, azimuth=azimuth,
                     elevation=elevation, distance=distance,
                     focalpoint=focalpoint, roll=roll)

    def reset_camera(self):
        renderer = getattr(self.fig.scene, 'renderer', None)
        if renderer is not None:
            renderer.reset_camera()

    def screenshot(self, mode='rgb', filename=None):
        return _take_3d_screenshot(figure=self.fig, mode=mode,
                                   filename=filename)

    def project(self, xyz, ch_names):
        xy = _3d_to_2d(self.fig, xyz)
        xy = dict(zip(ch_names, xy))
        pts = self.fig.children[-1]

        return _Projection(xy=xy, pts=pts)

    def enable_depth_peeling(self):
        if self.fig.scene is not None:
            self.fig.scene.renderer.use_depth_peeling = True

    def remove_mesh(self, surface):
        if self.fig.scene is not None:
            self.fig.scene.renderer.remove_actor(surface.actor)


def _mlab_figure(**kwargs):
    """Create a Mayavi figure using our defaults."""
    from .._3d import _get_3d_option
    fig = _import_mlab().figure(**kwargs)
    # If using modern VTK/Mayavi, improve rendering with FXAA
    antialias = _get_3d_option('antialias')
    if antialias and hasattr(getattr(fig.scene, 'renderer', None), 'use_fxaa'):
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
    """Return the 4x4 matrix to convert view coordinates to display coords.

    It's assumed that the view should take up the entire window and that the
    origin of the window is in the upper left corner.
    """
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


def _close_all():
    from mayavi import mlab
    mlab.close(all=True)


def _set_3d_view(figure, azimuth, elevation, focalpoint, distance, roll=None,
                 reset_camera=True):
    from mayavi import mlab
    with warnings.catch_warnings(record=True):  # traits
        with SilenceStdout():
            mlab.view(azimuth, elevation, distance,
                      focalpoint=focalpoint, figure=figure, roll=roll)
            mlab.draw(figure)


def _set_3d_title(figure, title, size=40):
    from mayavi import mlab
    text = mlab.title(text='', figure=figure)
    text.property.vertical_justification = 'top'
    text.property.font_size = size
    mlab.draw(figure)


def _check_3d_figure(figure):
    try:
        import mayavi  # noqa F401
    except Exception:
        raise TypeError('figure must be a mayavi scene but the'
                        'mayavi package is not found.')
    else:
        from mayavi.core.scene import Scene
        if not isinstance(figure, Scene):
            raise TypeError('figure must be a mayavi scene.')


def _save_figure(img, filename):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(frameon=False)
    FigureCanvasAgg(fig)
    fig.figimage(img, resize=True)
    fig.savefig(filename)


def _close_3d_figure(figure):
    from mayavi import mlab
    mlab.close(figure)


def _take_3d_screenshot(figure, mode='rgb', filename=None):
    from mayavi import mlab
    from mne.viz.backends.renderer import MNE_3D_BACKEND_TESTING
    if MNE_3D_BACKEND_TESTING:
        ndim = 3 if mode == 'rgb' else 4
        if figure.scene is None:
            figure_size = figure._window_size
        else:
            figure_size = figure.scene._renwin.size
        img = np.zeros(tuple(figure_size) + (ndim,), np.uint8)
    else:
        from pyface.api import GUI
        gui = GUI()
        gui.process_events()
        with warnings.catch_warnings(record=True):  # traits
            img = mlab.screenshot(figure, mode=mode)
    if isinstance(filename, str):
        _save_figure(img, filename)
    return img


@contextmanager
def _testing_context(interactive):
    mlab = _import_mlab()
    orig_backend = mlab.options.backend
    mlab.options.backend = 'test'
    try:
        yield
    finally:
        mlab.options.backend = orig_backend


def _oct_glyph(glyph_source, transform):
    from tvtk.api import tvtk
    from tvtk.common import configure_input
    from traits.api import Array
    gs = tvtk.PlatonicSolidSource()

    # Workaround for:
    #  File "mayavi/components/glyph_source.py", line 231, in _glyph_position_changed  # noqa: E501
    #    g.center = 0.0, 0.0, 0.0
    # traits.trait_errors.TraitError: Cannot set the undefined 'center' attribute of a 'TransformPolyDataFilter' object.  # noqa: E501
    class SafeTransformPolyDataFilter(tvtk.TransformPolyDataFilter):
        center = Array(shape=(3,), value=np.zeros(3))

    gs.solid_type = 'octahedron'
    if transform is not None:
        # glyph:             mayavi.modules.vectors.Vectors
        # glyph.glyph:       vtkGlyph3D
        # glyph.glyph.glyph: mayavi.components.glyph.Glyph
        assert transform.shape == (4, 4)
        tr = tvtk.Transform()
        tr.set_matrix(transform.ravel())
        trp = SafeTransformPolyDataFilter()
        configure_input(trp, gs)
        trp.transform = tr
        trp.update()
        gs = trp
    glyph_source.glyph_source = gs
