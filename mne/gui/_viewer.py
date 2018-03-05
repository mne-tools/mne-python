"""Mayavi/traits GUI visualization elements."""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import numpy as np

from mayavi.mlab import pipeline, text3d
from mayavi.modules.glyph import Glyph
from mayavi.modules.surface import Surface
from mayavi.sources.vtk_data_source import VTKDataSource
from mayavi.tools.mlab_scene_model import MlabSceneModel
from traits.api import (HasTraits, HasPrivateTraits, on_trait_change,
                        Instance, Array, Bool, Button, Enum, Float, Int, List,
                        Range, Str, RGBColor, Property, cached_property)
from traitsui.api import View, Item, HGroup, VGrid, VGroup
from tvtk.api import tvtk

from ..defaults import DEFAULTS
from ..surface import (complete_surface_info, _project_onto_surface,
                       _normalize_vectors)
from ..source_space import _points_outside_surface
from ..utils import SilenceStdout
from ..viz._3d import _create_mesh_surf, _toggle_mlab_render


headview_borders = VGroup(Item('headview', style='custom', show_label=False),
                          show_border=True, label='View')


class HeadViewController(HasTraits):
    """Set head views for the given coordinate system.

    Parameters
    ----------
    system : 'RAS' | 'ALS' | 'ARI'
        Coordinate system described as initials for directions associated with
        the x, y, and z axes. Relevant terms are: Anterior, Right, Left,
        Superior, Inferior.
    """

    system = Enum("RAS", "ALS", "ARI", desc="Coordinate system: directions of "
                  "the x, y, and z axis.")

    right = Button()
    front = Button()
    left = Button()
    top = Button()
    interaction = Enum('Trackball', 'Terrain')

    scale = Float(0.16)

    scene = Instance(MlabSceneModel)

    view = View(VGrid('0', 'top', '0', Item('scale', label='Scale',
                                            show_label=True),
                      'right', 'front', 'left', 'interaction',
                      show_labels=False, columns=4))

    @on_trait_change('scene.activated')
    def _init_view(self):
        self.scene.parallel_projection = True
        self._trackball_interactor = None

        # apparently scene,activated happens several times
        if self.scene.renderer:
            self.sync_trait('scale', self.scene.camera, 'parallel_scale')
            # and apparently this does not happen by default:
            self.on_trait_change(self.scene.render, 'scale')

    @on_trait_change('interaction')
    def on_set_interaction(self, _, interaction):
        if self.scene is None:
            return
        if interaction == 'Terrain':
            # Ensure we're in the correct orientatino for the
            # InteractorStyleTerrain to have the correct "up"
            if self._trackball_interactor is None:
                self._trackball_interactor = \
                    self.scene.interactor.interactor_style
            self.on_set_view('front', '')
            self.scene.mlab.draw()
            self.scene.interactor.interactor_style = \
                tvtk.InteractorStyleTerrain()
            self.on_set_view('front', '')
            self.scene.mlab.draw()
        else:  # interaction == 'trackball'
            self.scene.interactor.interactor_style = self._trackball_interactor

    @on_trait_change('top,left,right,front')
    def on_set_view(self, view, _):
        if self.scene is None:
            return

        system = self.system
        kwargs = dict(ALS=dict(front=(0, 90, -90),
                               left=(90, 90, 180),
                               right=(-90, 90, 0),
                               top=(0, 0, -90)),
                      RAS=dict(front=(90., 90., 180),
                               left=(180, 90, 90),
                               right=(0., 90, 270),
                               top=(90, 0, 180)),
                      ARI=dict(front=(0, 90, 90),
                               left=(-90, 90, 180),
                               right=(90, 90, 0),
                               top=(0, 180, 90)))
        if system not in kwargs:
            raise ValueError("Invalid system: %r" % system)
        if view not in kwargs[system]:
            raise ValueError("Invalid view: %r" % view)
        kwargs = dict(zip(('azimuth', 'elevation', 'roll'),
                          kwargs[system][view]))
        with SilenceStdout():
            self.scene.mlab.view(distance=None, reset_roll=True,
                                 figure=self.scene.mayavi_scene, **kwargs)


class Object(HasPrivateTraits):
    """Represent a 3d object in a mayavi scene."""

    points = Array(float, shape=(None, 3))
    nn = Array(float, shape=(None, 3))
    name = Str

    scene = Instance(MlabSceneModel, ())
    src = Instance(VTKDataSource)

    # This should be Tuple, but it is broken on Anaconda as of 2016/12/16
    color = RGBColor((1., 1., 1.))
    opacity = Range(low=0., high=1., value=1.)
    visible = Bool(True)

    def _update_points(self):
        """Update the location of the plotted points."""
        if not hasattr(self.src, 'data'):
            return
        self.src.data.points = self.points
        return True


class PointObject(Object):
    """Represent a group of individual points in a mayavi scene."""

    label = Bool(False)
    projectable = Bool(False)  # set based on type of points
    orientable = Property(depends_on=['project_to_points', 'project_to_tris'])
    text3d = List
    point_scale = Float(10, label='Point Scale')

    # projection onto a surface
    project_to_points = Array(float, shape=(None, 3))
    project_to_tris = Array(int, shape=(None, 3))
    project_to_surface = Bool(False, label='Project', desc='project points '
                              'onto the surface')
    orient_to_surface = Bool(False, label='Orient', desc='orient points '
                             'toward the surface')
    scale_by_distance = Bool(False, label='Dist.', desc='scale points by '
                             'distance from the surface')
    mark_inside = Bool(False, label='Mark', desc='mark points inside the '
                       'surface in a different color')
    inside_color = RGBColor((0., 0., 0.))

    glyph = Instance(Glyph)
    resolution = Int(8)

    view = View(HGroup(Item('visible', show_label=False),
                       Item('color', show_label=False),
                       Item('opacity')))

    def __init__(self, view='points', *args, **kwargs):
        """Init.

        Parameters
        ----------
        view : 'points' | 'cloud'
            Whether the view options should be tailored to individual points
            or a point cloud.
        """
        self._view = view
        super(PointObject, self).__init__(*args, **kwargs)

    def default_traits_view(self):  # noqa: D102
        color = Item('color', show_label=False)
        scale = Item('point_scale', label='Size')
        orient = Item('orient_to_surface',
                      enabled_when='orientable and not project_to_surface')
        dist = Item('scale_by_distance',
                    enabled_when='orientable and not project_to_surface')
        mark = Item('mark_inside',
                    enabled_when='orientable and not project_to_surface')
        if self._view == 'points':
            visible = Item('visible', label='Show', show_label=True)
            views = (visible, color, scale, 'label')
        elif self._view == 'cloud':
            visible = Item('visible', show_label=False)
            views = (visible, color, scale)
        else:
            raise ValueError("PointObject(view = %r)" % self._view)
        group2 = HGroup(dist, Item('project_to_surface', show_label=True,
                                   enabled_when='projectable'),
                        orient, mark, show_left=False)
        return View(HGroup(HGroup(*views), group2))

    @on_trait_change('label')
    def _show_labels(self, show):
        _toggle_mlab_render(self, False)
        while self.text3d:
            text = self.text3d.pop()
            text.remove()

        if show:
            fig = self.scene.mayavi_scene
            for i, pt in enumerate(np.array(self.src.data.points)):
                x, y, z = pt
                t = text3d(x, y, z, ' %i' % i, scale=.01, color=self.color,
                           figure=fig)
                self.text3d.append(t)
        _toggle_mlab_render(self, True)

    @on_trait_change('visible')
    def _on_hide(self):
        if not self.visible:
            self.label = False

    @on_trait_change('scene.activated')
    def _plot_points(self):
        """Add the points to the mayavi pipeline"""
        if self.scene is None:
            return
        if hasattr(self.glyph, 'remove'):
            self.glyph.remove()
        if hasattr(self.src, 'remove'):
            self.src.remove()

        _toggle_mlab_render(self, False)
        x, y, z = self.points.T
        fig = self.scene.mayavi_scene
        scatter = pipeline.scalar_scatter(x, y, z, fig=fig)
        if not scatter.running:
            # this can occur sometimes during testing w/ui.dispose()
            return
        # fig.scene.engine.current_object is scatter
        glyph = pipeline.glyph(scatter, color=self.color,
                               figure=fig, scale_factor=self.point_scale,
                               opacity=1., resolution=self.resolution,
                               mode='sphere')
        glyph.actor.property.backface_culling = True
        glyph.glyph.glyph.vector_mode = 'use_normal'
        glyph.glyph.glyph.clamping = False

        glyph.actor.mapper.color_mode = 'map_scalars'
        glyph.actor.mapper.scalar_mode = 'use_point_data'
        glyph.actor.mapper.use_lookup_table_scalar_range = False

        self.src = scatter
        self.glyph = glyph

        self.sync_trait('point_scale', self.glyph.glyph.glyph, 'scale_factor')
        self.sync_trait('color', self.glyph.actor.property, mutual=False)
        self.sync_trait('visible', self.glyph)
        self.sync_trait('opacity', self.glyph.actor.property)
        self.sync_trait('mark_inside', self.glyph.actor.mapper,
                        'scalar_visibility')
        self.on_trait_change(self._update_points, 'points')
        self._update_markers()
        self._update_colors()
        _toggle_mlab_render(self, True)
        # self.scene.camera.parallel_scale = _scale

    # don't put project_to_tris here, just always set project_to_points second
    @on_trait_change('points,project_to_points,project_to_surface,mark_inside')
    def _update_projections(self):
        """Update the styles of the plotted points."""
        if not hasattr(self.src, 'data'):
            return
        if len(self.project_to_points) <= 1 or len(self.points) == 0:
            return

        # Do the projections
        pts = self.points
        surf = dict(rr=np.array(self.project_to_points),
                    tris=np.array(self.project_to_tris))
        method = 'accurate' if len(surf['rr']) <= 20484 else 'nearest'
        proj_pts, proj_nn = _project_onto_surface(
            pts, surf, project_rrs=True, return_nn=True,
            method=method)[2:4]
        vec = pts - proj_pts  # point to the surface
        if self.project_to_surface:
            pts = proj_pts
            nn = proj_nn
        else:
            nn = vec.copy()
            _normalize_vectors(nn)
        if self.mark_inside and not self.project_to_surface:
            scalars = _points_outside_surface(pts, surf).astype(int)
        else:
            scalars = np.ones(len(pts))
        # With this, a point exactly on the surface is of size point_scale
        dist = np.linalg.norm(vec, axis=-1, keepdims=True)
        self.src.data.point_data.normals = (250 * dist + 1) * nn
        self.src.data.point_data.scalars = scalars
        self.glyph.actor.mapper.scalar_range = [0., 1.]
        self.src.data.point_data.update()

    @on_trait_change('color,inside_color')
    def _update_colors(self):
        if self.glyph is None:
            return
        # inside_color is the surface color, let's try to get far
        # from that
        inside = np.array(self.inside_color)
        # if it's too close to gray, just use black:
        if np.mean(np.abs(inside - 0.5)) < 0.2:
            inside.fill(0.)
        else:
            inside = 1 - inside
        colors = np.array([tuple(inside) + (1,),
                           tuple(self.color) + (1,)]) * 255.
        self.glyph.module_manager.scalar_lut_manager.lut.table = colors

    @on_trait_change('project_to_surface,orient_to_surface,scale_by_distance')
    def _update_markers(self):
        if self.glyph is None:
            return
        defaults = DEFAULTS['coreg']
        gs = self.glyph.glyph.glyph_source
        res = getattr(gs.glyph_source, 'theta_resolution',
                      getattr(gs.glyph_source, 'resolution', None))
        if self.project_to_surface or self.orient_to_surface:
            gs.glyph_source = tvtk.CylinderSource()
            gs.glyph_source.height = defaults['eegp_height']
            gs.glyph_source.center = (0., -defaults['eegp_height'], 0)
            gs.glyph_source.resolution = res
        else:
            gs.glyph_source = tvtk.SphereSource()
            gs.glyph_source.phi_resolution = res
            gs.glyph_source.theta_resolution = res
        if self.scale_by_distance:
            self.glyph.glyph.scale_mode = 'scale_by_vector'
        else:
            self.glyph.glyph.scale_mode = 'data_scaling_off'

    def _resolution_changed(self, new):
        if not self.glyph:
            return

        gs = self.glyph.glyph.glyph_source.glyph_source
        if isinstance(gs, tvtk.SphereSource):
            gs.phi_resolution = new
            gs.theta_resolution = new
        else:
            gs.resolution = new

    @cached_property
    def _get_orientable(self):
        orientable = (len(self.project_to_points) > 0 and
                      len(self.project_to_tris) > 0)
        return orientable


# XXX eventually we should update the normals whenever "points" is changed


class SurfaceObject(Object):
    """Represent a solid object in a mayavi scene.

    Notes
    -----
    Doesn't automatically update plot because update requires both
    :attr:`points` and :attr:`tri`. Call :meth:`plot` after updateing both
    attributes.
    """

    rep = Enum("Surface", "Wireframe")
    tri = Array(int, shape=(None, 3))

    surf = Instance(Surface)
    surf_rear = Instance(Surface)

    view = View(HGroup(Item('visible', show_label=False),
                       Item('color', show_label=False),
                       Item('opacity')))

    def __init__(self, block_behind=False, **kwargs):  # noqa: D102
        self._block_behind = block_behind
        super(SurfaceObject, self).__init__(**kwargs)

    def clear(self):  # noqa: D102
        if hasattr(self.src, 'remove'):
            self.src.remove()
        if hasattr(self.surf, 'remove'):
            self.surf.remove()
        if hasattr(self.surf_rear, 'remove'):
            self.surf_rear.remove()
        self.reset_traits(['src', 'surf'])

    @on_trait_change('scene.activated')
    def plot(self):
        """Add the points to the mayavi pipeline"""
        _scale = self.scene.camera.parallel_scale
        self.clear()

        if not np.any(self.tri):
            return

        fig = self.scene.mayavi_scene
        surf = complete_surface_info(dict(rr=self.points, tris=self.tri),
                                     verbose='error')
        self.src = _create_mesh_surf(surf, fig=fig)
        rep = 'wireframe' if self.rep == 'Wireframe' else 'surface'
        surf = pipeline.surface(
            self.src, figure=fig, color=self.color, representation=rep,
            line_width=1)
        surf.actor.property.backface_culling = True
        self.surf = surf
        self.sync_trait('visible', self.surf, 'visible')
        self.sync_trait('color', self.surf.actor.property, mutual=False)
        self.sync_trait('opacity', self.surf.actor.property)
        if self._block_behind:
            surf_rear = pipeline.surface(
                self.src, figure=fig, color=self.color, representation=rep,
                line_width=1)
            surf_rear.actor.property.frontface_culling = True
            self.surf_rear = surf_rear
            self.sync_trait('color', self.surf_rear.actor.property,
                            mutual=False)
            self.sync_trait('visible', self.surf_rear, 'visible')
            self.surf_rear.actor.property.opacity = 1.

        self.scene.camera.parallel_scale = _scale

    @on_trait_change('points')
    def _update_points(self):
        if Object._update_points(self):
            self.src.update()  # necessary for SurfaceObject since Mayavi 4.5.0
