# -*- coding: utf-8 -*-
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
                        Range, Str, Property, cached_property, ArrayOrNone)
from traitsui.api import (View, Item, HGroup, VGrid, VGroup, Spring,
                          TextEditor)
from tvtk.api import tvtk

from ..defaults import DEFAULTS
from ..surface import _CheckInside, _DistanceQuery
from ..transforms import apply_trans, rotation
from ..utils import SilenceStdout
from ..viz.backends._pysurfer_mayavi import (_create_mesh_surf, _oct_glyph,
                                             _toggle_mlab_render)

try:
    from traitsui.api import RGBColor
except ImportError:
    from traits.api import RGBColor

headview_borders = VGroup(Item('headview', style='custom', show_label=False),
                          show_border=True, label='View')


def _mm_fmt(x):
    """Format data in units of mm."""
    return '%0.1f' % x


laggy_float_editor_mm = TextEditor(auto_set=False, enter_set=True,
                                   evaluate=float,
                                   format_func=lambda x: '%0.1f' % x)

laggy_float_editor_scale = TextEditor(auto_set=False, enter_set=True,
                                      evaluate=float,
                                      format_func=lambda x: '%0.1f' % x)

laggy_float_editor_headscale = TextEditor(auto_set=False, enter_set=True,
                                          evaluate=float,
                                          format_func=lambda x: '%0.3f' % x)

laggy_float_editor_weight = TextEditor(auto_set=False, enter_set=True,
                                       evaluate=float,
                                       format_func=lambda x: '%0.2f' % x)

laggy_float_editor_deg = TextEditor(auto_set=False, enter_set=True,
                                    evaluate=float,
                                    format_func=lambda x: '%0.1f' % x)

_BUTTON_WIDTH = -80
_DEG_WIDTH = -50  # radian floats
_MM_WIDTH = _DEG_WIDTH  # mm floats
_SCALE_WIDTH = _DEG_WIDTH  # scale floats
_INC_BUTTON_WIDTH = -25  # inc/dec buttons
_DEG_STEP_WIDTH = -50
_MM_STEP_WIDTH = _DEG_STEP_WIDTH
_SCALE_STEP_WIDTH = _DEG_STEP_WIDTH
_WEIGHT_WIDTH = -60  # weight floats
_VIEW_BUTTON_WIDTH = -60
# width is optimized for macOS and Linux avoid a horizontal scroll-bar
# even when a vertical one is present
_COREG_WIDTH = -290
_TEXT_WIDTH = -260
_REDUCED_TEXT_WIDTH = _TEXT_WIDTH - 40 * np.sign(_TEXT_WIDTH)
_DIG_SOURCE_WIDTH = _TEXT_WIDTH - 50 * np.sign(_TEXT_WIDTH)
_MRI_FIDUCIALS_WIDTH = _TEXT_WIDTH - 60 * np.sign(_TEXT_WIDTH)
_SHOW_BORDER = True
_RESET_LABEL = u"↻"
_RESET_WIDTH = _INC_BUTTON_WIDTH


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
    interaction = Enum('trackball', 'terrain')

    scale = Float(0.16)

    scene = Instance(MlabSceneModel)

    view = View(VGroup(
        VGrid('0', Item('top', width=_VIEW_BUTTON_WIDTH), '0',
              Item('right', width=_VIEW_BUTTON_WIDTH),
              Item('front', width=_VIEW_BUTTON_WIDTH),
              Item('left', width=_VIEW_BUTTON_WIDTH),
              columns=3, show_labels=False),
        '_',
        HGroup(Item('scale', label='Scale',
                    editor=laggy_float_editor_headscale,
                    width=_SCALE_WIDTH, show_label=True),
               Item('interaction', tooltip='Mouse interaction mode',
                    show_label=False), Spring()),
        show_labels=False))

    @on_trait_change('scene.activated')
    def _init_view(self):
        self.scene.parallel_projection = True

        # apparently scene,activated happens several times
        if self.scene.renderer:
            self.sync_trait('scale', self.scene.camera, 'parallel_scale')
            # and apparently this does not happen by default:
            self.on_trait_change(self.scene.render, 'scale')
            self.interaction = self.interaction  # could be delayed

    @on_trait_change('interaction')
    def on_set_interaction(self, _, interaction):
        if self.scene is None or self.scene.interactor is None:
            return
        # Ensure we're in the correct orientation for the
        # InteractorStyleTerrain to have the correct "up"
        self.on_set_view('front', '')
        self.scene.mlab.draw()
        self.scene.interactor.interactor_style = \
            tvtk.InteractorStyleTerrain() if interaction == 'terrain' else \
            tvtk.InteractorStyleTrackballCamera()
        # self.scene.interactor.interactor_style.
        self.on_set_view('front', '')
        self.scene.mlab.draw()

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
        kwargs['focalpoint'] = (0., 0., 0.)
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
    # Due to a MESA bug, we use 0.99 opacity to force alpha blending
    opacity = Range(low=0., high=1., value=0.99)
    visible = Bool(True)

    def _update_points(self):
        """Update the location of the plotted points."""
        if hasattr(self.src, 'data'):
            self.src.data.points = self.points
            return True


class PointObject(Object):
    """Represent a group of individual points in a mayavi scene."""

    label = Bool(False)
    label_scale = Float(0.01)
    projectable = Bool(False)  # set based on type of points
    orientable = Property(depends_on=['nearest'])
    text3d = List
    point_scale = Float(10, label='Point Scale')

    # projection onto a surface
    nearest = Instance(_DistanceQuery)
    check_inside = Instance(_CheckInside)
    project_to_trans = ArrayOrNone(float, shape=(4, 4))
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

    def __init__(self, view='points', has_norm=False, *args, **kwargs):
        """Init.

        Parameters
        ----------
        view : 'points' | 'cloud' | 'arrow' | 'oct'
            Whether the view options should be tailored to individual points
            or a point cloud.
        has_norm : bool
            Whether a norm can be defined; adds view options based on point
            norms (default False).
        """
        assert view in ('points', 'cloud', 'arrow', 'oct')
        self._view = view
        self._has_norm = bool(has_norm)
        super(PointObject, self).__init__(*args, **kwargs)

    def default_traits_view(self):  # noqa: D102
        color = Item('color', show_label=False)
        scale = Item('point_scale', label='Size', width=_SCALE_WIDTH,
                     editor=laggy_float_editor_headscale)
        orient = Item('orient_to_surface',
                      enabled_when='orientable and not project_to_surface',
                      tooltip='Orient points toward the surface')
        dist = Item('scale_by_distance',
                    enabled_when='orientable and not project_to_surface',
                    tooltip='Scale points by distance from the surface')
        mark = Item('mark_inside',
                    enabled_when='orientable and not project_to_surface',
                    tooltip='Mark points inside the surface using a different '
                    'color')
        if self._view == 'arrow':
            visible = Item('visible', label='Show', show_label=False)
            return View(HGroup(visible, scale, 'opacity', 'label', Spring()))
        elif self._view in ('points', 'oct'):
            visible = Item('visible', label='Show', show_label=True)
            views = (visible, color, scale, 'label')
        else:
            assert self._view == 'cloud'
            visible = Item('visible', show_label=False)
            views = (visible, color, scale)

        if not self._has_norm:
            return View(HGroup(*views))

        group2 = HGroup(dist, Item('project_to_surface', show_label=True,
                                   enabled_when='projectable',
                                   tooltip='Project points onto the surface '
                                   '(for visualization, does not affect '
                                   'fitting)'),
                        orient, mark, Spring(), show_left=False)
        return View(HGroup(HGroup(*views), group2))

    @on_trait_change('label')
    def _show_labels(self, show):
        _toggle_mlab_render(self, False)
        while self.text3d:
            text = self.text3d.pop()
            text.remove()

        if show and len(self.src.data.points) > 0:
            fig = self.scene.mayavi_scene
            if self._view == 'arrow':  # for axes
                x, y, z = self.src.data.points[0]
                self.text3d.append(text3d(
                    x, y, z, self.name, scale=self.label_scale,
                    color=self.color, figure=fig))
            else:
                for i, (x, y, z) in enumerate(np.array(self.src.data.points)):
                    self.text3d.append(text3d(
                        x, y, z, ' %i' % i, scale=self.label_scale,
                        color=self.color, figure=fig))
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
        mode = {'cloud': 'sphere', 'points': 'sphere', 'oct': 'sphere'}.get(
            self._view, self._view)
        assert mode in ('sphere', 'arrow')
        glyph = pipeline.glyph(scatter, color=self.color,
                               figure=fig, scale_factor=self.point_scale,
                               opacity=1., resolution=self.resolution,
                               mode=mode)
        if self._view == 'oct':
            _oct_glyph(glyph.glyph.glyph_source, rotation(0, 0, np.pi / 4))
        glyph.actor.property.backface_culling = True
        glyph.glyph.glyph.vector_mode = 'use_normal'
        glyph.glyph.glyph.clamping = False
        if mode == 'arrow':
            glyph.glyph.glyph_source.glyph_position = 'tail'

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
        self._update_marker_scaling()
        self._update_marker_type()
        self._update_colors()
        _toggle_mlab_render(self, True)
        # self.scene.camera.parallel_scale = _scale

    def _nearest_default(self):
        return _DistanceQuery(np.zeros((1, 3)))

    def _get_nearest(self, proj_rr):
        idx = self.nearest.query(proj_rr)[1]
        proj_pts = apply_trans(
            self.project_to_trans, self.nearest.data[idx])
        proj_nn = apply_trans(
            self.project_to_trans, self.check_inside.surf['nn'][idx],
            move=False)
        return proj_pts, proj_nn

    @on_trait_change('points,project_to_trans,project_to_surface,mark_inside,'
                     'nearest')
    def _update_projections(self):
        """Update the styles of the plotted points."""
        if not hasattr(self.src, 'data'):
            return
        if self._view == 'arrow':
            self.src.data.point_data.normals = self.nn
            self.src.data.point_data.update()
            return
        # projections
        if len(self.nearest.data) <= 1 or len(self.points) == 0:
            return

        # Do the projections
        pts = self.points
        inv_trans = np.linalg.inv(self.project_to_trans)
        proj_rr = apply_trans(inv_trans, self.points)
        proj_pts, proj_nn = self._get_nearest(proj_rr)
        vec = pts - proj_pts  # point to the surface
        if self.project_to_surface:
            pts = proj_pts
        nn = proj_nn
        if self.mark_inside and not self.project_to_surface:
            scalars = (~self.check_inside(proj_rr, verbose=False)).astype(int)
        else:
            scalars = np.ones(len(pts))
        # With this, a point exactly on the surface is of size point_scale
        dist = np.linalg.norm(vec, axis=-1, keepdims=True)
        self.src.data.point_data.normals = (250 * dist + 1) * nn
        self.src.data.point_data.scalars = scalars
        self.glyph.actor.mapper.scalar_range = [0., 1.]
        self.src.data.points = pts  # projection can change this
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

    @on_trait_change('project_to_surface,orient_to_surface')
    def _update_marker_type(self):
        # not implemented for arrow
        if self.glyph is None or self._view == 'arrow':
            return
        defaults = DEFAULTS['coreg']
        gs = self.glyph.glyph.glyph_source
        res = getattr(gs.glyph_source, 'theta_resolution',
                      getattr(gs.glyph_source, 'resolution', None))
        if res is None:
            return
        if self.project_to_surface or self.orient_to_surface:
            gs.glyph_source = tvtk.CylinderSource()
            gs.glyph_source.height = defaults['eegp_height']
            gs.glyph_source.center = (0., -defaults['eegp_height'], 0)
            gs.glyph_source.resolution = res
        else:
            gs.glyph_source = tvtk.SphereSource()
            gs.glyph_source.phi_resolution = res
            gs.glyph_source.theta_resolution = res

    @on_trait_change('scale_by_distance,project_to_surface')
    def _update_marker_scaling(self):
        if self.glyph is None:
            return
        if self.scale_by_distance and not self.project_to_surface:
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
        elif isinstance(gs, tvtk.CylinderSource):
            gs.resolution = new
        else:  # ArrowSource
            gs.tip_resolution = new
            gs.shaft_resolution = new

    @cached_property
    def _get_orientable(self):
        return len(self.nearest.data) > 1


class SurfaceObject(Object):
    """Represent a solid object in a mayavi scene.

    Notes
    -----
    Doesn't automatically update plot because update requires both
    :attr:`points` and :attr:`tris`. Call :meth:`plot` after updating both
    attributes.
    """

    rep = Enum("Surface", "Wireframe")
    tris = Array(int, shape=(None, 3))

    surf = Instance(Surface)
    surf_rear = Instance(Surface)

    view = View(HGroup(Item('visible', show_label=False),
                       Item('color', show_label=False),
                       Item('opacity')))

    def __init__(self, block_behind=False, **kwargs):  # noqa: D102
        self._block_behind = block_behind
        self._deferred_tri_update = False
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

        if not np.any(self.tris):
            return

        fig = self.scene.mayavi_scene
        surf = dict(rr=self.points, tris=self.tris)
        normals = _create_mesh_surf(surf, fig=fig)
        self.src = normals.parent
        rep = 'wireframe' if self.rep == 'Wireframe' else 'surface'
        # Add the opaque "inside" first to avoid the translucent "outside"
        # from being occluded (gh-5152)
        if self._block_behind:
            surf_rear = pipeline.surface(
                normals, figure=fig, color=self.color, representation=rep,
                line_width=1)
            surf_rear.actor.property.frontface_culling = True
            self.surf_rear = surf_rear
            self.sync_trait('color', self.surf_rear.actor.property,
                            mutual=False)
            self.sync_trait('visible', self.surf_rear, 'visible')
            self.surf_rear.actor.property.opacity = 1.
        surf = pipeline.surface(
            normals, figure=fig, color=self.color, representation=rep,
            line_width=1)
        surf.actor.property.backface_culling = True
        self.surf = surf
        self.sync_trait('visible', self.surf, 'visible')
        self.sync_trait('color', self.surf.actor.property, mutual=False)
        self.sync_trait('opacity', self.surf.actor.property)

        self.scene.camera.parallel_scale = _scale

    @on_trait_change('tris')
    def _update_tris(self):
        self._deferred_tris_update = True

    @on_trait_change('points')
    def _update_points(self):
        # Nuke the tris before setting the points otherwise we can get
        # a nasty segfault (gh-5728)
        if self._deferred_tris_update and self.src is not None:
            self.src.data.polys = None
        if Object._update_points(self):
            if self._deferred_tris_update:
                self.src.data.polys = self.tris
                self._deffered_tris_update = False
            self.src.update()  # necessary for SurfaceObject since Mayavi 4.5.0
