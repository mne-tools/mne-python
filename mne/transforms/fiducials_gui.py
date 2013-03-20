"""Mayavi/traits GUI for averaging two sets of KIT marker points"""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

from glob import glob
import cPickle as pickle
import os

from mayavi.core.ui.mayavi_scene import MayaviScene
from mayavi.modules.glyph import Glyph
from mayavi.sources.vtk_data_source import VTKDataSource
from mayavi.tools.mlab_scene_model import MlabSceneModel
import numpy as np
from pyface.api import confirm, error, FileDialog, OK, YES
from traits.api import HasTraits, HasPrivateTraits, on_trait_change, \
                       cached_property, Instance, Property, \
                       Array, Bool, Button, Color, Enum, File, Float, List, \
                       Str, Tuple
from traitsui.api import View, Item, Group, HGroup, VGroup, CheckListEditor
from traitsui.menu import NoButtons
from tvtk.pyface.scene_editor import SceneEditor

from .coreg import fit_matched_pts
from .file_traits import SubjectSelector, BemSource
from .transforms import apply_trans, rotation, translation
from .viewer import HeadViewController, PointObject, SurfaceObject, headview_borders
from ..fiff import Raw, FIFF, read_fiducials, write_fiducials
from ..fiff.kit.coreg import read_mrk
from ..utils import get_config, get_subjects_dir



fid_fname = "{subjects_dir}/{subject}/bem/{name}-fiducials.fif"



class FiducialsPanel(HasPrivateTraits):
    """Set fiducials on an MRI surface"""
    fid_file = File(filter=["Fiducials FIFF file (*.fif)|*.fif"])
    fid_name = Property(depends_on='fid_file')
    fid_pts = Property(depends_on=['fid_file'])

    subjects_dir = Str()
    subject = Str()

    locked = Bool(False)
    set = Enum('LAP', 'Nasion', 'RAP')
    LAP = Array(float, (1, 3))
    nasion = Array(float, (1, 3))
    RAP = Array(float, (1, 3))

    can_save_as = Property(depends_on=['LAP', 'nasion', 'RAP'])
    save_as = Button(label='Save As...')
    can_save = Property(depends_on=['fid_file', 'can_save_as'])
    save = Button(label='Save')
    reset_fid = Button(label="Reset to File")
    can_reset = Property(depends_on=['fid_file', 'fid_pts', 'LAP', 'nasion',
                                     'RAP'])

    scene = Instance(MlabSceneModel)
    headview = Instance(HeadViewController)
    hsp_obj = Instance(SurfaceObject)

    # the layout of the dialog created
    view = View(VGroup(Item('fid_file', label='Fiducials File'),
                       Item('fid_name', show_label=False, style='readonly'),
                       Item('set', style='custom'),  # 'nasion', 'LAP', 'RAP',
                       HGroup(Item('save', enabled_when='can_save'),
                              Item('save_as', enabled_when='can_save_as'),
                              Item('reset_fid', enabled_when='can_reset'),
                              show_labels=False),
                       enabled_when="locked==False"))

    @cached_property
    def _get_fid_name(self):
        name = os.path.basename(self.fid_file)
        return name

    @cached_property
    def _get_fid_pts(self):
        fname = self.fid_file
        pts = np.zeros((3, 3))
        if os.path.exists(fname):
            fids, _ = read_fiducials(fname)
            for fid in fids:
                ident = fid['ident']
                r = fid['r']
                if ident == 1:
                    pts[0] = r
                elif ident == 2:
                    pts[1] = r
                elif ident == 3:
                    pts[2] = r
        return pts

    @on_trait_change('fid_pts')
    def reset_fid_to_file(self):
        self.LAP = self.fid_pts[0:1]
        self.nasion = self.fid_pts[1:2]
        self.RAP = self.fid_pts[2:3]

    def _reset_fid_fired(self):
        self.reset_fid_to_file()

    @cached_property
    def _get_can_reset(self):
        if not self.fid_file:
            return False
        elif np.any(self.LAP != self.fid_pts[0:1]):
            return True
        elif np.any(self.nasion != self.fid_pts[1:2]):
            return True
        elif np.any(self.RAP != self.fid_pts[2:3]):
            return True
        return False

    @cached_property
    def _get_can_save_as(self):
        can = not (np.all(self.nasion == self.LAP)
                   or np.all(self.nasion == self.RAP)
                   or np.all(self.LAP == self.RAP))
        return can

    @cached_property
    def _get_can_save(self):
        if not self.can_save_as:
            return False
        elif self.fid_file:
            return True
        elif self.subjects_dir and self.subject:
            return True
        else:
            return False

    def get_dig_list(self):
        dig = [{'kind': 1, 'ident': 1, 'r': np.array(self.LAP[0])},
               {'kind': 1, 'ident': 2, 'r': np.array(self.nasion[0])},
               {'kind': 1, 'ident': 3, 'r': np.array(self.RAP[0])}]
        return dig

    def _save_fired(self):
        if self.fid_file:
            fname = self.fid_file
        elif self.subjects_dir and self.subject:
            fname = fid_fname.format(subjects_dir=self.subjects_dir,
                                     subject=self.subject, name=self.subject)
        else:
            self._save_as_fired()
            return
        write_fiducials(fname, self.get_dig_list(), FIFF.FIFFV_COORD_MRI)
        self.fid_file = fname

    def _save_as_fired(self):
        if self.fid_file:
            default_path = self.fid_file
        else:
            default_path = ''

        dlg = FileDialog(action="save as",
                         wildcard="Ficudials FIFF file (*.fif)|*.fif",
                         default_path=default_path)
        dlg.open()
        if dlg.return_code != OK:
            return

        path = dlg.path
        if not path.endswith('.fif'):
            path = path + '.fif'
            if os.path.exists(path):
                answer = confirm(None, "The file %r already exists. Should it be "
                                 "replaced?", "Overwrite File?")
                if answer != YES:
                    return

        write_fiducials(path, self.get_dig_list(), FIFF.FIFFV_COORD_MRI)

    def _on_mouse_click(self, picker):
        if self.locked:
            return

        pid = picker.point_id
        pt = [self.hsp_obj.src.data.points[pid]]
        if self.set == 'Nasion':
            self.nasion = pt
        elif self.set == 'LAP':
            self.LAP = pt
        elif self.set == 'RAP':
            self.RAP = pt
        else:
            raise ValueError("set = %r" % self.set)

    @on_trait_change('set')
    def _on_set_change(self):
        if self.set == 'Nasion':
            self.headview.front = True
        elif self.set == 'LAP':
            self.headview.left = True
        elif self.set == 'RAP':
            self.headview.right = True



# FiducialsPanel view that allows manipulating coordinates numerically
view2 = View(VGroup(Item('fid_file', label='Fiducials File'),
                    Item('fid_name', show_label=False, style='readonly'),
                    Item('set', style='custom'), 'nasion', 'LAP', 'RAP',
                    HGroup(Item('save', enabled_when='can_save'),
                           Item('save_as', enabled_when='can_save_as'),
                           show_labels=False),
                    enabled_when="locked==False"))


class MainWindow(HasTraits):
    """GUI for interpolating between two KIT marker files

    Parameters
    ----------
    mrk1, mrk2 : str
        Path to pre- and post measurement marker files (*.sqd) or empty string.
    """
    headview = Instance(HeadViewController)
    mri_src = Instance(BemSource, ())
    mri_obj = Instance(SurfaceObject)
    s_sel = Instance(SubjectSelector, ())
    scene = Instance(MlabSceneModel, ())
    fid_panel = Instance(FiducialsPanel)

    point_scale = float(5e-3)
    lap_obj = Instance(PointObject)
    nas_obj = Instance(PointObject)
    rap_obj = Instance(PointObject)

    def _headview_default(self):
        return HeadViewController(scene=self.scene, system='RAS')

    def _fid_panel_default(self):
        pnl = FiducialsPanel(headview=self.headview, scene=self.scene)
        pnl.trait_view('view', view2)
        return pnl

    view = View(HGroup(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                            dock='vertical'),
                       VGroup(headview_borders,
                              VGroup(Item('s_sel', style='custom'),
                                     label="Subject", show_border=True,
                                     show_labels=False),
                              VGroup(Item('fid_panel', style="custom"),
                                     label="Fiducials", show_border=True,
                                     show_labels=False),
                              show_labels=False),
                       show_labels=False),
                resizable=True,
                height=0.75, width=0.75,
                buttons=NoButtons)

    def __init__(self, subject=None, subjects_dir=None, fid_file=None):
        subjects_dir = get_subjects_dir(subjects_dir)
        super(MainWindow, self).__init__()

        # set initial parameters
        if subjects_dir is not None:
            self.s_sel.subjects_dir = subjects_dir
        if subject is not None:
            if subject in self.s_sel.subjects:
                self.s_sel.subject = subject

    @on_trait_change('scene.activated')
    def _init_plot(self):
        self.scene.disable_render = True

        # fiducials
        self.lap_obj = PointObject(scene=self.scene, color=(255, 0, 0),
                                   point_scale=self.point_scale)
        self.fid_panel.sync_trait('LAP', self.lap_obj, 'points', mutual=False)

        self.nas_obj = PointObject(scene=self.scene, color=(0, 255, 0),
                                   point_scale=self.point_scale)
        self.fid_panel.sync_trait('nasion', self.nas_obj, 'points', mutual=False)

        self.rap_obj = PointObject(scene=self.scene, color=(0, 0, 255),
                                   point_scale=self.point_scale)
        self.fid_panel.sync_trait('RAP', self.rap_obj, 'points', mutual=False)

        # bem
        self.mri_obj = SurfaceObject(points=self.mri_src.pts, tri=self.mri_src.tri,
                                     scene=self.scene, color=(255, 255, 255))
        self.mri_src.on_trait_change(self._on_mri_src_change, 'tri')
        self.fid_panel.hsp_obj = self.mri_obj

        self.scene.disable_render = False
        self.headview.front = True

        # picker
        self.scene.mayavi_scene.on_mouse_pick(self.fid_panel._on_mouse_click)

    def _on_mri_src_change(self):
        if (not np.any(self.mri_src.pts)) or (not np.any(self.mri_src.tri)):
            self.mri_obj.clear()
            return

        self.mri_obj.points = self.mri_src.pts
        self.mri_obj.tri = self.mri_src.tri
        self.mri_obj.plot()

    @on_trait_change('s_sel.mri_dir')
    def _on_subject_change(self):
        subjects_dir = self.s_sel.subjects_dir
        subject = self.s_sel.subject
        if not subjects_dir or not subject:
            return

        # update bem head
        fname = os.path.join(subjects_dir, subject, 'bem',
                             subject + '-head.fif')
        self.mri_src.file = fname

        # find fiducials file
        path = fid_fname.format(subjects_dir=subjects_dir, subject=subject,
                                name=subject)
        if os.path.exists(path):
            self.fid_panel.fid_file = path
        else:
            path = fid_fname.format(subjects_dir=subjects_dir, subject=subject,
                                    name='*')
            fnames = glob(path)
            if fnames:
                path = fnames[0]
                self.fid_panel.fid_file = path
