"""GUI for coregistration between different coordinate frames"""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os

import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree, leaves_list
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist

from mayavi import mlab
from mayavi.core.ui.mayavi_scene import MayaviScene
from mayavi.tools import pipeline
from mayavi.tools.mlab_scene_model import MlabSceneModel
from pyface.api import error, confirm, YES, NO, CANCEL, ProgressDialog
import traits.api as traits
from traitsui.api import View, Item, Group, HGroup, VGroup
from tvtk.pyface.scene_editor import SceneEditor

from .coreg import MriHeadFitter, HeadMriFitter, BemGeom
from ..fiff import Raw, FIFF, read_fiducials, write_fiducials
from ..utils import get_subjects_dir



class HeadViewer(traits.HasTraits):
    "Baseclass for GUIs working on head models"
    right = traits.Button()
    front = traits.Button()
    left = traits.Button()
    top = traits.Button()
    view_scale = traits.Float(0.13)

    scene = traits.Instance(MlabSceneModel, ())

    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=600, width=600, show_label=False),
                # # HeadViewer Traits
                Group(HGroup('72', Item('top', show_label=False), '100',
                             Item('view_scale', label='Scale')),
                      HGroup('right', 'front', 'left', show_labels=False),
                      label='View', show_border=True),
                # # end HeadViewer traits
                )

    @traits.on_trait_change('scene.activated')
    def _init_view(self):
        self.sync_trait('view_scale', self.scene.camera, 'parallel_scale')
        self.view_scale = 0.16

    @traits.on_trait_change('view_scale')
    def _on_view_scale_update(self):
        self.scene.camera.parallel_scale = self.view_scale
        self.scene.render()

    @traits.on_trait_change('top,left,right,front')
    def on_set_view(self, view='front', info=None):
        self.scene.parallel_projection = True
        kwargs = dict(azimuth=90, elevation=90, distance=None, roll=180,
                      reset_roll=True, figure=self.scene.mayavi_scene)
        if view == 'left':
            kwargs.update(azimuth=180, roll=90)
        elif view == 'right':
            kwargs.update(azimuth=0, roll=270)
        elif view == 'top':
            kwargs.update(elevation=0)
        self.scene.mlab.view(**kwargs)



def raw_find_point(raw):
    "Open FindDigPoint with the dig info from a raw file"
    raw = Raw(raw)
    dig = raw.info['dig']
    pts = filter(lambda d: d['kind'] == 4, dig)
    pts = np.array([d['r'] for d in pts])
    gui = FindDigPoint(pts)
    gui.configure_traits()
    return gui

class FindDigPoint(HeadViewer):
    """
    Mayavi viewer for visualizing specific points in an object.

    """
    point = traits.Range(low=0, high=10000, is_float=True, mode='spinner')

    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=500, width=500, show_label=False),
                # # HeadViewer Traits
                Group(HGroup('72', Item('top', show_label=False), '100',
                             Item('view_scale', label='Scale')),
                      HGroup('right', 'front', 'left', show_labels=False),
                      label='View', show_border=True),
                # # end HeadViewer traits
                '_',
                VGroup('point'),
                )

    def __init__(self, pts):
        self.pts = pts

        traits.HasTraits.__init__(self)

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
                                    scale_factor=.01)

        self.point = 0

    @traits.on_trait_change('point')
    def on_update_point(self):
        self.scene.disable_render = True

        self.src.data.points[0] = self.pts[int(self.point)]
        self.glyph.remove()
        fig = self.scene.mayavi_scene
        self.glyph = pipeline.glyph(self.src, color=(1, 0, 0), figure=fig,
                                    scale_factor=.01)

        self.scene.disable_render = False



def raw_hs(raw):
    "Open FixDigHeadShape with the dig info from a raw file"
    raw = Raw(raw)
    dig = raw.info['dig']
    pts = filter(lambda d: d['kind'] == 4, dig)
    pts = np.array([d['r'] for d in pts])
    gui = FixDigHeadShape(pts)
    gui.configure_traits()
    return gui


class FixDigHeadShape(HeadViewer):
    """
    Mayavi viewer for decomposing an object based on clustering

    """
    # parameters
    clusters = traits.Range(low=0, high=20, is_float=False, mode='spinner')

    # saving
    cancel = traits.Button()
    ok = traits.Button()

    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=500, width=500, show_label=False),
                # # HeadViewer Traits
                Group(HGroup('72', Item('top', show_label=False), '100',
                             Item('view_scale', label='Scale')),
                      HGroup('right', 'front', 'left', show_labels=False),
                      label='View', show_border=True),
                # # end HeadViewer traits
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



class Fiducials(HeadViewer):
    """
    Mayavi viewer for creating a fiducials file.

    Parameters
    ----------
    subject : str
        The mri subject.
    fid : None | str
        Fiducials file for initial positions.
    subjects_dir : None | str
        Overrule the subjects_dir environment variable.

    """
    set = traits.Enum('RAP', 'Nasion', 'LAP')
    nasion = traits.Array(float, (1, 3))
    LAP = traits.Array(float, (1, 3))
    RAP = traits.Array(float, (1, 3))

    _save = traits.Button()
    scene = traits.Instance(MlabSceneModel, ())

    # the layout of the dialog created
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=500, width=500, show_label=False),
                # # HeadViewer Traits
                Group(HGroup('72', Item('top', show_label=False), '100',
                             Item('view_scale', label='Scale')),
                      HGroup('right', 'front', 'left', show_labels=False),
                      label='View', show_border=True),
                # # end HeadViewer traits
                VGroup(Item('set', style='custom'), 'nasion', 'LAP', 'RAP',
                       label='Fiducials', show_border=True),
                HGroup(HGroup('_save', show_labels=False)),
                )


    def __init__(self, subject, fid=None, subjects_dir=None):
        self.subjects_dir = get_subjects_dir(subjects_dir)
        self.subject = subject

        self._fid_file = fid

        traits.HasTraits.__init__(self)

    @traits.on_trait_change('scene.activated')
    def on_init(self):
        fig = self.scene.mayavi_scene
        self.scene.disable_render = True

        fname = os.path.join(self.subjects_dir, self.subject, 'bem',
                             self.subject + '-head.fif')
        self.head = BemGeom(fname)
        self._pts = self.head.get_pts()
        self.head_mesh, _ = self.head.plot_solid(self.scene.mayavi_scene,
                                                 color=(.7, .7, .6))

        nasion = pipeline.scalar_scatter(0, 0, 0)
        pipeline.glyph(nasion.data, figure=fig, color=(0, 1, 0), opacity=0.8,
                       scale_factor=0.01)
        self._nasion = nasion

        LAP = pipeline.scalar_scatter(0, 0, 0)
        pipeline.glyph(LAP, figure=fig, color=(0, 0, 1), opacity=0.8,
                       scale_factor=0.01)
        self._LAP = LAP

        RAP = pipeline.scalar_scatter(0, 0, 0)
        pipeline.glyph(RAP, figure=fig, color=(1, 0, 0), opacity=0.8,
                       scale_factor=0.01)
        self._RAP = RAP

        self.scene.mayavi_scene.on_mouse_pick(self._on_mouse_click)

        if self._fid_file is not None:
            fids, _ = read_fiducials(self._fid_file)
            for fid in fids:
                ident = fid['ident']
                r = [ fid['r']]
                if ident == 1:
                    self.LAP = r
                elif ident == 2:
                    self.nasion = r
                elif ident == 3:
                    self.RAP = r

        self.front = True
        self.scene.disable_render = False

    def _on_mouse_click(self, picker):
        pid = picker.point_id
        pts = [self._pts[pid]]
        if self.set == 'Nasion':
            self.nasion = pts
        elif self.set == 'LAP':
            self.LAP = pts
        elif self.set == 'RAP':
            self.RAP = pts

    @traits.on_trait_change('nasion')
    def on_nasion_change(self):
        self._nasion.data.points = self.nasion

    @traits.on_trait_change('LAP')
    def on_LAP_change(self):
        self._LAP.data.points = self.LAP

    @traits.on_trait_change('RAP')
    def on_RAP_change(self):
        self._RAP.data.points = self.RAP

    @traits.on_trait_change('set')
    def on_set_change(self):
        if self.set == 'Nasion':
            self.front = True
        elif self.set == 'LAP':
            self.left = True
        elif self.set == 'RAP':
            self.right = True

    @traits.on_trait_change('_save')
    def on_save(self):
        self.save()

    def save(self, fname=None, overwrite=False):
        if fname is None:
            fname = os.path.join(self.subjects_dir, self.subject, 'bem',
                                 self.subject + '-fiducials.fif')

        if os.path.exists(fname) and not overwrite:
            title = "Replace %s Fiducials?" % self.subject
            msg = ("The mri subject %s already has a fiducials file. \nReplace "
                   "%r?" % (self.subject, fname))
            answer = confirm(None, msg, title, cancel=False, default=NO)
            if answer != YES:
                return

        dig = [
               {'kind': 1, 'ident': 1, 'r': np.array(self.LAP[0])},
               {'kind': 1, 'ident': 2, 'r': np.array(self.nasion[0])},
               {'kind': 1, 'ident': 3, 'r': np.array(self.RAP[0])},
               ]
        write_fiducials(fname, dig, FIFF.FIFFV_COORD_MRI)



class HeadMriCoreg(HeadViewer):
    """
    Mayavi viewer for estimating the head mri transform.

    Parameters
    ----------
    raw : str(path)
        path to a raw file containing the digitizer data.
    subject : str
        name of the mri subject.
        Can be None if the raw file-name starts with "{subject}_".
    subjects_dir : None | path
        Override the SUBJECTS_DIR environment variable
        (sys.environ['SUBJECTS_DIR'])

    """
    # parameters
    nasion = traits.Array(float, (1, 3), label='Digitizer Position Adjustment')
    rotation = traits.Array(float, (1, 3))

    # fitting
    fit = traits.Button(label='Fit')
    fit_fid = traits.Button(label='Fit Fiducials')
    restore_fit = traits.Button(label='Restore Last Fit')

    # saving
    save = traits.Button(label='Save Trans')

    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=500, width=500, show_label=False),
                # # HeadViewer Traits
                Group(HGroup('72', Item('top', show_label=False), '100',
                             Item('view_scale', label='Scale')),
                      HGroup('right', 'front', 'left', show_labels=False),
                      label='View', show_border=True),
                # # end HeadViewer traits
                VGroup('nasion',
                       HGroup('fit', 'fit_fid', 'restore_fit', show_labels=False),
                       'rotation', label='Transform', show_border=True),
                HGroup('save', show_labels=False),
                )

    def __init__(self, raw, subject=None, subjects_dir=None):
        self.fitter = HeadMriFitter(raw, subject, subjects_dir=subjects_dir)
        self._last_fit = None

        traits.HasTraits.__init__(self)

    @traits.on_trait_change('scene.activated')
    def on_init(self):
        fig = self.scene.mayavi_scene
        self.scene.disable_render = True

        self.fitter.plot(fig=fig)
        mlab.text(0.01, 0.01, self.fitter.subject, figure=fig, width=0.1)

        self.left = True
        self.scene.disable_render = False
        self._last_fit = None

    @traits.on_trait_change('fit')
    def on_fit(self):
        prog = ProgressDialog(title="Fitting...", message="Fitting head to "
                              "mri...")
        prog.open()
        prog.update(0)
        self._last_fit = self.fitter.fit()
        self.on_restore_fit()
        prog.close()

    @traits.on_trait_change('fit_fid')
    def on_fit_fid(self):
        self._last_fit = self.fitter.fit_fiducials(fixed_nas=True)
        self.on_restore_fit()

    @traits.on_trait_change('restore_fit')
    def on_restore_fit(self):
        if self._last_fit is None:
            error(None, "No fit has been performed", "No Fit")
            return

        self.rotation = [self._last_fit]

    @traits.on_trait_change('save')
    def on_save(self):
        trans_fname = self.fitter.get_trans_fname()
        if os.path.exists(trans_fname):
            title = "Replace trans file for %s?" % self.fitter.subject
            msg = ("A trans file named %r already exists. \nReplace "
                   "%r?" % (os.path.basename(trans_fname), trans_fname))
            answer = confirm(None, msg, title, cancel=False, default=NO)
            if answer != YES:
                return

        try:
            self.fitter.save_trans(trans_fname, overwrite=True)
        except Exception as e:
            error(None, str(e), "Error while Saving")

    @traits.on_trait_change('rotation')
    def on_set_rot(self):
        rot = np.array(self.rotation[0])
        self.fitter.set(rot=rot)

    @traits.on_trait_change('nasion')
    def on_set_trans(self):
        trans = np.array(self.nasion[0])
        self.fitter.set(trans=trans)



class MriHeadCoreg(HeadViewer):
    """
    Mayavi viewer for adjusting an MRI to a digitized head shape.

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
    # parameters
    nasion = traits.Array(float, (1, 3), label='Digitizer Position Adjustment')
    n_scale_params = traits.Enum(1, 3, label='N Scaling Parameters')

    scale3 = traits.Array(float, (1, 3), [[1, 1, 1]], label='Scale',
                          enabled_when='n_scale_params == 3')
    shrink = traits.Float(0., enabled_when='n_scale_params == 3')

    scale1 = traits.Float(1, label='Scale')

    rotation = traits.Array(float, (1, 3))
    restore_fit = traits.Button(label='Restore Last Fit')

    # fitting
    fit_scale = traits.Button(label='Fit with Scaling')
    fit_no_scale = traits.Button(label='Fit Rotation Only')

    # saving
    s_to = traits.String('NONE', label='Subject Name')
    save = traits.Button()

    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=500, width=500, show_label=False),
                # # HeadViewer Traits
                Group(HGroup('72', Item('top', show_label=False), '100',
                             Item('view_scale', label='Scale')),
                      HGroup('right', 'front', 'left', show_labels=False),
                      label='View', show_border=True),
                # # end HeadViewer traits
                Group(HGroup('nasion'),
                      HGroup(Item('n_scale_params', style='custom'),
                             Item('fit_scale', show_label=False)),
                      label='Initial Parameters', show_border=True),
                Group(HGroup('restore_fit', show_labels=False),
                      HGroup('scale1', enabled_when='n_scale_params == 1'),
                      HGroup('scale3', 'shrink', enabled_when='n_scale_params == 3'),
                      HGroup('rotation', Item('fit_no_scale', show_label=False)),
                      label='Adjust Fit', show_border=True),
                HGroup('s_to', HGroup('save', show_labels=False)),
                )

    def __init__(self, raw, s_from=None, s_to=None, subjects_dir=None):
        self.fitter = MriHeadFitter(raw, s_from, subjects_dir=subjects_dir)

        if s_to is None:
            try:
                s_to = self.fitter._raw_name.split('_')[0]
            except:
                pass

        self._s_to_arg = s_to

        traits.HasTraits.__init__(self)

    @traits.on_trait_change('scene.activated')
    def on_init(self):
        fig = self.scene.mayavi_scene
        self.scene.disable_render = True

        self.fitter.plot(fig=fig)
        self._text = mlab.text(0.01, 0.01, '_' * 60, figure=fig, width=0.4)
        if isinstance(self._s_to_arg, str):
            self.s_to = self._s_to_arg
        else:
            self.s_to = ''

        self.left = True
        self.scene.disable_render = False
        self._last_fit = None

    @traits.on_trait_change('fit_scale,fit_no_scale')
    def on_fit(self, caller, info):
        if caller == 'fit_scale':
            n_scale = self.n_scale_params
        elif caller == 'fit_no_scale':
            n_scale = 0
        else:
            error(None, "Unknown caller for fit: %r" % caller, "Error")
            return

        prog = ProgressDialog(title="Fitting...", message="Fitting %s to "
                              "%s" % (self.fitter.subject, self.s_to))
        prog.open()
        prog.update(0)
        rotation, scale = self.fitter.fit(scale=n_scale)
        self._last_fit = dict(n=n_scale, s=scale, r=rotation)
        self.on_restore_fit()
        prog.close()

    @traits.on_trait_change('restore_fit')
    def on_restore_fit(self):
        if self._last_fit is None:
            error(None, "No fit has been performed", "No Fit")
            return

        self.scene.disable_render = True
        lf = self._last_fit
        self.rotation = [lf['r']]
        if lf['n'] == 1:
            self.n_scale_params = 1
            self.scale1 = lf['s']
        elif lf['n'] == 3:
            self.n_scale_params = 3
            self.scale3 = [lf['s']]

        self.scene.disable_render = False

    @traits.on_trait_change('save')
    def on_save(self):
        s_from = self.fitter.subject
        s_to = self.s_to

        trans_fname = self.fitter.get_trans_fname(s_to)
        if os.path.exists(trans_fname):
            title = "Replace trans file for %s?" % s_to
            msg = ("A trans file named %r already exists. \nReplace "
                   "%r?" % (os.path.basename(trans_fname), trans_fname))
            answer = confirm(None, msg, title, cancel=False, default=NO)
            if answer != YES:
                return

        s_to_dir = self.fitter.get_mri_dir(s_to)
        if os.path.exists(s_to_dir):
            title = "Replace %r?" % s_to
            msg = ("The mri subject %r already exists. \nReplace "
                   "%r?" % (s_to, s_to_dir))
            answer = confirm(None, msg, title, cancel=False, default=NO)
            if answer != YES:
                return

        prog = ProgressDialog(title="Saving...", message="Saving scaled %s to "
                              "%s" % (s_from, s_to))
        prog.open()
        prog.update(0)
        try:
            self.fitter.save_all(s_to, overwrite=True)
        except Exception as e:
            error(None, str(e), "Error while Saving")
        prog.close()

    @traits.on_trait_change('n_scale_params')
    def on_set_n_scale(self):
        if self.n_scale_params == 1:
            self.scale1 = np.mean(self.scale3)
        else:
            s = self.scale1
            self.scale3 = [[s, s, s]]

    @traits.on_trait_change('rotation')
    def on_set_rot(self):
        rot = np.array(self.rotation[0])
        self.fitter.set(rot=rot)

    @traits.on_trait_change('s_to')
    def on_set_s_to(self, s_to):
        s_from = self.fitter.subject
        if s_to:
            text = "%s -> %s" % (s_from, s_to)
        else:
            text = "%s" % s_from

        self._text.text = text

    @traits.on_trait_change('scale1')
    def on_set_scale1(self):
        self.fitter.set(scale=self.scale1)

    @traits.on_trait_change('scale3,shrink')
    def on_set_scale3(self):
        scale = np.array(self.scale3[0])
        if self.shrink:
            scale -= scale * self.shrink * np.array([1, .4, 1])
        self.fitter.set(scale=scale)

    @traits.on_trait_change('nasion')
    def on_set_trans(self):
        trans = np.array(self.nasion[0])
        self.fitter.set(trans=trans)
