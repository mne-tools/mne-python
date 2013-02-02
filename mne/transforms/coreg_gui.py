"""GUI for coregistration between different coordinate frames"""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os

import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree, leaves_list
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist

from mayavi.core.ui.mayavi_scene import MayaviScene
from mayavi.tools import pipeline
from mayavi.tools.mlab_scene_model import MlabSceneModel
from pyface.api import error, confirm, YES, NO, CANCEL, ProgressDialog
import traits.api as traits
from traitsui.api import View, Item, HGroup, VGroup
from tvtk.pyface.scene_editor import SceneEditor

from .coreg import MriHeadFitter
from ..fiff import Raw


def raw_find_point(raw):
    "Open FindDigPoint with the dig info from a raw file"
    raw = Raw(raw)
    dig = raw.info['dig']
    pts = filter(lambda d: d['kind'] == 4, dig)
    pts = np.array([d['r'] for d in pts])
    return FindDigPoint(pts)

class FindDigPoint(traits.HasTraits):
    """
    Mayavi viewer for visualizing specific points in an object.

    """
    right = traits.Button()
    front = traits.Button()
    left = traits.Button()
    top = traits.Button()

    # parameters
    point = traits.Range(low=0, high=10000, is_float=True, mode='spinner')

    scene = traits.Instance(MlabSceneModel, ())

    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=500, width=500, show_label=False),
                HGroup('72', 'top', show_labels=False),
                HGroup('right', 'front', 'left', show_labels=False),
                '_',
                VGroup('point'),
                )

    def __init__(self, pts):
        self._orig_pts = pts
        pts = pts * 1000
        self.pts = pts

        traits.HasTraits.__init__(self)
        self.configure_traits()

    @traits.on_trait_change('scene.activated')
    def on_init(self):
        d = Delaunay(self.pts)
        tri = d.convex_hull
        x, y, z = self.pts.T

        fig = self.scene.mayavi_scene
        mesh = pipeline.triangular_mesh_source(x, y, z, tri, figure=fig)
        surf = pipeline.surface(mesh, figure=fig, color=(1, 1, 1),
                                representation='wireframe', line_width=1)
        surf.actor.property.lighting = False

        self.src = pipeline.scalar_scatter(0, 0, 0)
        self.glyph = pipeline.glyph(self.src, color=(1, 0, 0), figure=fig,
                                    scale_factor=10)

        self.point = 0

    @traits.on_trait_change('point')
    def on_update_point(self):
        self.scene.disable_render = True

        self.src.data.points[0] = self.pts[int(self.point)]
        self.glyph.remove()
        fig = self.scene.mayavi_scene
        self.glyph = pipeline.glyph(self.src, color=(1, 0, 0), figure=fig,
                                    scale_factor=10)

        self.scene.disable_render = False

    @traits.on_trait_change('top,left,right,front')
    def on_set_view(self, view='front', info=None):
        self.scene.parallel_projection = True
        self.scene.camera.parallel_scale = 150
        kwargs = dict(azimuth=90, elevation=90, distance=None, roll=180,
                      reset_roll=True, figure=self.scene.mayavi_scene)
        if view == 'left':
            kwargs.update(azimuth=180, roll=90)
        elif view == 'right':
            kwargs.update(azimuth=0, roll=270)
        elif view == 'top':
            kwargs.update(elevation=0)
        self.scene.mlab.view(**kwargs)



def raw_hs(raw):
    "Open FixDigHeadShape with the dig info from a raw file"
    raw = Raw(raw)
    dig = raw.info['dig']
    pts = filter(lambda d: d['kind'] == 4, dig)
    pts = np.array([d['r'] for d in pts])
    return FixDigHeadShape(pts)

class FixDigHeadShape(traits.HasTraits):
    """
    Mayavi viewer for decomposing an object based on clustering

    """
    right = traits.Button()
    front = traits.Button()
    left = traits.Button()
    top = traits.Button()

    # parameters
    clusters = traits.Range(low=0, high=20, is_float=False, mode='spinner')

    # saving
    cancel = traits.Button()
    ok = traits.Button()

    scene = traits.Instance(MlabSceneModel, ())

    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=500, width=500, show_label=False),
                HGroup('72', 'top', show_labels=False),
                HGroup('right', 'front', 'left', show_labels=False),
                '_',
                VGroup('clusters'),
                '_',
                HGroup('cancel', 'ok', show_labels=False),
                )

    def __init__(self, pts):
        self._orig_pts = pts
        pts = pts * 1000
        self.pts = pts

        cdist = pdist(pts, metric='euclidean', p=2, w=None, V=None, VI=None)
        z = linkage(cdist, method='single')
        self._root = to_tree(z)
        self._leaves = leaves_list(z)

        self._sel = None
        self._plots = []
        self._all_idx = np.arange(len(pts))
        traits.HasTraits.__init__(self)
        self.configure_traits()

    @traits.on_trait_change('scene.activated')
    def on_init(self):
        self.clusters = 0

    @traits.on_trait_change('clusters')
    def on_update_clusters(self):
        self.scene.disable_render = True
        if hasattr(self, 'mesh'):
            self.mesh.remove()
            del self.mesh
            self.surf.remove()
            del self.surf
        if hasattr(self, 'rmesh'):
            self.rmesh.remove()
            del self.rmesh
            self.rsurf.remove()
            del self.rsurf

        node = self._root
        rnode = None  # rejected node
        i = 0
        i_last = 0
        for _ in xrange(self.clusters):
            rnode = node.left
            node = node.right
            i_last = i
            i += rnode.count
            if node.is_leaf():
                return

        self.i = (i, i_last)
        sel = np.sort(self._leaves[i:])
        pts = self.pts[sel]
        self.mesh, self.surf = self._plot_pts(pts)

        if rnode is not None:
            rsel = np.sort(self._leaves[i_last:i])
            pts = self.pts[rsel]
            if len(pts) > 2:
                self.rmesh, self.rsurf = self._plot_pts(pts, color=(1, 0, 0))

        self.scene.disable_render = False

    @traits.on_trait_change('top,left,right,front')
    def on_set_view(self, view='front', info=None):
        self.scene.parallel_projection = True
        self.scene.camera.parallel_scale = 150
        kwargs = dict(azimuth=90, elevation=90, distance=None, roll=180,
                      reset_roll=True, figure=self.scene.mayavi_scene)
        if view == 'left':
            kwargs.update(azimuth=180, roll=90)
        elif view == 'right':
            kwargs.update(azimuth=0, roll=270)
        elif view == 'top':
            kwargs.update(elevation=0)
        self.scene.mlab.view(**kwargs)

    def _plot_pts(self, pts, color=(1, 1, 1)):
        d = Delaunay(pts)
        tri = d.convex_hull
        x, y, z = pts.T

        fig = self.scene.mayavi_scene
        mesh = pipeline.triangular_mesh_source(x, y, z, tri, figure=fig)
        surf = pipeline.surface(mesh, figure=fig, color=color,  # opacity=1,
                                representation='wireframe', line_width=1)
        surf.actor.property.lighting = False

        return mesh, surf



class MriHeadCoreg(traits.HasTraits):
    """
    Mayavi viewer for adjusting an MRI to a digitized head shape.

    """
    # views
    right = traits.Button()
    front = traits.Button()
    left = traits.Button()
    top = traits.Button()

    # parameters
    nasion = traits.Array(float, (1, 3))
    rotation = traits.Array(float, (1, 3))
    scale = traits.Array(float, (1, 3), [[1, 1, 1]])
    shrink = traits.Float(1)
    restore_fit = traits.Button()

    # fitting
    fit_scale = traits.Button()
    fit_no_scale = traits.Button()

    # saving
    s_to = traits.String()
    save = traits.Button()

    scene = traits.Instance(MlabSceneModel, ())

    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=500, width=500, show_label=False),
                HGroup('72', 'top', show_labels=False),
                HGroup('right', 'front', 'left', show_labels=False),
                HGroup('fit_scale', 'fit_no_scale', 'restore_fit',
                       show_labels=False),
                HGroup('nasion'),
                HGroup('scale', 'shrink'),
                HGroup('rotation'),
                HGroup('s_to', HGroup('save', show_labels=False)),
                )

    def __init__(self, raw, s_from=None, s_to=None, subjects_dir=None):
        """
        Parameters
        ----------
        raw : str(path)
            path to a raw file containing the digitizer data.
        s_from : str
            name of the source subject (e.g., 'fsaverage').
            Can be None if the raw file-name starts with "{subject}_".
        s_to : str | None
            Name of the the subject for which the MRI is destined (used to
            save MRI and in the trans file's file name).
            Can be None if the raw file-name starts with "{subject}_".
        subjects_dir : None | path
            Override the SUBJECTS_DIR environment variable
            (sys.environ['SUBJECTS_DIR'])

        """
        self.fitter = MriHeadFitter(raw, s_from, s_to, subjects_dir)

        traits.HasTraits.__init__(self)
        self.configure_traits()

    @traits.on_trait_change('scene.activated')
    def on_init(self):
        fig = self.scene.mayavi_scene
        self.scene.disable_render = True
        self.fitter.plot(fig=fig)

        self._text = None
        s_to = self.fitter.s_to
        self.s_to = s_to

        self.front = True
        self.scene.disable_render = False
        self._last_fit = None

    @traits.on_trait_change('fit_scale,fit_no_scale')
    def _fit(self, caller, info2):
        if caller == 'fit_scale':
            self.fitter.fit(method='sr')
        elif caller == 'fit_no_scale':
            self.fitter.fit(method='r')
        else:
            error(self, "Unknown caller for _fit(): %r" % caller, "Error")
            return

        rotation = self.fitter.get_rot()
        scale = self.fitter.get_scale()
        self._last_fit = ([scale], [rotation])
        self.on_restore_fit()

    @traits.on_trait_change('restore_fit')
    def on_restore_fit(self):
        if self._last_fit is None:
            error("No fit has been performed", "No Fit")
            return

        self.scale, self.rotation = self._last_fit
        self.shrink = 1

    @traits.on_trait_change('save')
    def on_save(self):
        s_to = self.s_to

        trans_fname = self.fitter.get_trans_fname(s_to)
        if os.path.exists(trans_fname):
            title = "Replace trans file for %s?" % s_to
            msg = ("A trans file already exists at %r. Replace "
                   "it?" % trans_fname)
            answer = confirm(None, msg, title, cancel=False, default=NO)
            if answer != YES:
                return

        s_to_dir = self.fitter.get_mri_dir(s_to)
        if os.path.exists(s_to_dir):
            title = "Replace %r?" % s_to
            msg = ("The mri subject %r already exists. Replace "
                   "%r?" % (s_to, s_to_dir))
            answer = confirm(None, msg, title, cancel=False, default=NO)
            if answer != YES:
                return

        try:
            self.fitter.save(s_to, overwrite=True)
        except Exception as e:
            error(None, str(e))

    @traits.on_trait_change('nasion')
    def on_set_nasion(self):
        args = tuple(self.nasion[0])
        self.fitter.set_nasion(*args)

    @traits.on_trait_change('s_to')
    def on_set_s_to(self, s_to):
        s_from = self.fitter.s_from
        fig = self.scene.mayavi_scene
        if s_to == s_from:
            text = "%s" % s_from
            width = .2
        else:
            text = "%s -> %s" % (s_from, s_to)
            width = .5

        if self._text is None:
            self._text = self.scene.mlab.text(0.01, 0.01, text, figure=fig,
                                              width=width)
        else:
            self._text.text = text
            self._text.width = width

    @traits.on_trait_change('scale,rotation,shrink')
    def on_set_trans(self):
        scale = np.array(self.scale[0])
        scale_scale = (1 - self.shrink) * np.array([1, .4, 1])
        scale *= (1 - scale_scale)
        args = tuple(self.rotation[0]) + tuple(scale)
        self.fitter.set(*args)

    @traits.on_trait_change('top,left,right,front')
    def on_set_view(self, view='front', info=None):
        self.scene.parallel_projection = True
        self.scene.camera.parallel_scale = 150
        kwargs = dict(azimuth=90, elevation=90, distance=None, roll=180,
                      reset_roll=True, figure=self.scene.mayavi_scene)
        if view == 'left':
            kwargs.update(azimuth=180, roll=90)
        elif view == 'right':
            kwargs.update(azimuth=0, roll=270)
        elif view == 'top':
            kwargs.update(elevation=0)
        self.scene.mlab.view(**kwargs)
