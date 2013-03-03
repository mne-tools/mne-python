"""Mayavi/traits GUI for averaging two sets of KIT marker points"""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

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
from .transforms import apply_trans, rotation, translation
from .viewer import HeadViewController, PointObject, SurfaceObject
from ..fiff import Raw, FIFF, read_fiducials, write_fiducials
from ..fiff.kit.coreg import read_mrk
from ..surface import read_bem_surfaces
from ..utils import get_config, get_subjects_dir



fid_fname = "{subjects_dir}/{subject}/bem/{subject}-fiducials.fif"



class FiducialsPanel(HasPrivateTraits):
    """Set fiducials on an MRI surface"""
    fid_file = File(wildcard="Ficudials FIFF file (*.fif)|*.fif")
    hsp_file = File(wildcard="Ficudials FIFF file (*.fif)|*.fif")
    fid_pts = Property(depends_on=['fid_file'])
    set = Enum('LAP', 'Nasion', 'RAP')
    LAP = Array(float, (1, 3))
    nasion = Array(float, (1, 3))
    RAP = Array(float, (1, 3))
    reset_fid = Button(label='Reset Fiducials')

    can_save_as = Property(depends_on=['LAP', 'nasion', 'RAP'])
    save_as = Button(label='Save As...')
    can_save = Property(depends_on=['fid_file', 'can_save_as'])
    save = Button(label='Save')

    scene = Instance(MlabSceneModel)
    head_view = Instance(HeadViewController)
    point_scale = float(5e-3)
    hsp_obj = Instance(SurfaceObject)
    lap_obj = Instance(PointObject)
    nas_obj = Instance(PointObject)
    rap_obj = Instance(PointObject)

    # the layout of the dialog created
    view = View(VGroup(Item('fid_file', label='Fiducials File'),
                       Item('set', style='custom'), 'nasion', 'LAP', 'RAP',
                       HGroup(Item('save', enabled_when='can_save'),
                              Item('save_as', enabled_when='can_save_as'),
                              show_labels=False),
                       label='Fiducials', show_border=True))

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
    def _reset_fid_fired(self):
        self.LAP = self.fid_pts[0:1]
        self.nasion = self.fid_pts[1:2]
        self.RAP = self.fid_pts[2:3]

    @cached_property
    def _get_can_save_as(self):
        can = not (np.all(self.nasion == self.LAP)
                    or np.all(self.nasion == self.RAP)
                    or np.all(self.LAP == self.RAP))
        return can

    @cached_property
    def _get_can_save(self):
        return (self.can_save_as and self.fid_file)

    def get_dig_list(self):
        dig = [{'kind': 1, 'ident': 1, 'r': np.array(self.LAP[0])},
               {'kind': 1, 'ident': 2, 'r': np.array(self.nasion[0])},
               {'kind': 1, 'ident': 3, 'r': np.array(self.RAP[0])}]
        return dig

    def _save_fired(self):
        write_fiducials(self.fid_file, self.get_dig_list(), FIFF.FIFFV_COORD_MRI)

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

    def __init__(self, subject, subjects_dir, **kwargs):
        self._subject = subject
        self._subjects_dir = subjects_dir
        self._default_path = fid_fname.format(subjects_dir=subjects_dir, subject=subject)
        if ('fid_file' not in kwargs) or (kwargs['fid_file'] is None):
            if os.path.exists(self._default_path):
                kwargs['fid_file'] = self._default_path
        super(FiducialsPanel, self).__init__(**kwargs)

    @on_trait_change('scene.activated')
    def _init_plot(self):
        self.scene.disable_render = True

        fname = os.path.join(self._subjects_dir, self._subject, 'bem',
                             self._subject + '-head.fif')
        bem = read_bem_surfaces(fname)[0]
        self.hsp_obj = SurfaceObject(scene=self.scene, color=(255, 225, 200),
                                     points=bem['rr'], tri=bem['tris'])

        self.lap_obj = PointObject(scene=self.scene, color=(255, 0, 0),
                                   point_scale=self.point_scale)
        self.sync_trait('LAP', self.lap_obj, 'points', mutual=False)

        self.nas_obj = PointObject(scene=self.scene, color=(0, 255, 0),
                                   point_scale=self.point_scale)
        self.sync_trait('nasion', self.nas_obj, 'points', mutual=False)

        self.rap_obj = PointObject(scene=self.scene, color=(0, 0, 255),
                                   point_scale=self.point_scale)
        self.sync_trait('RAP', self.rap_obj, 'points', mutual=False)

        self.scene.mayavi_scene.on_mouse_pick(self._on_mouse_click)
        self.scene.disable_render = False

    def _on_mouse_click(self, picker):
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
            self.head_view.front = True
        elif self.set == 'LAP':
            self.head_view.left = True
        elif self.set == 'RAP':
            self.head_view.right = True



view2 = View(VGroup(VGroup(Item('fid_file', label='Fiducials File'),
                           Item('set', style='custom'), 'nasion', 'LAP', 'RAP',
                           HGroup(Item('save', enabled_when='can_save'),
                                  Item('save_as', enabled_when='can_save_as'),
                                  show_labels=False),
                           label='Fiducials', show_border=True),
                    VGroup(Item('hsp_obj', style='custom'),
                           label='Head Shape View Options',
                           show_border=True, show_labels=False)
                    ))

class MainWindow(HasTraits):
    """GUI for interpolating between two KIT marker files

    Parameters
    ----------
    mrk1, mrk2 : str
        Path to pre- and post measurement marker files (*.sqd) or empty string.
    """
    scene = Instance(MlabSceneModel, ())
    head_view = Instance(HeadViewController)
    panel = Instance(FiducialsPanel)

    def _head_view_default(self):
        return HeadViewController(scene=self.scene, system='RAS')

    def _panel_default(self):
        pnl = FiducialsPanel(self._subject, self._subjects_dir,
                             fid_file=self._fid_file, scene=self.scene,
                             head_view=self.head_view)
        pnl.trait_view('view', view2)
        return pnl

    view = View(HGroup(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                            dock='vertical'),
                       VGroup(Item('panel', style="custom"),
#                              VGroup(Item('panel.hsp_obj', style='custom'),
#                                     label='Head Shape View Options',
#                                     show_border=True, show_labels=False),
                              Item('head_view', style='custom'),
                              show_labels=False),
                       show_labels=False),
                resizable=True,
                height=0.75, width=0.75,
                buttons=NoButtons)

    def __init__(self, subject, subjects_dir=None, fid_file=None):
        self._subject = subject
        self._subjects_dir = get_subjects_dir(subjects_dir)
        self._fid_file = fid_file
        super(MainWindow, self).__init__()

    @on_trait_change('scene.activated')
    def _init_plot(self):
        self.head_view.front = True
