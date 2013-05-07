"""GUI for coregistration between different coordinate frames"""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os
from Queue import Queue
from threading import Thread

import numpy as np
from numpy import dot
from scipy.cluster.hierarchy import linkage, to_tree, leaves_list
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist

from mayavi import mlab
from mayavi.core.ui.mayavi_scene import MayaviScene
from mayavi.tools import pipeline
from mayavi.tools.mlab_scene_model import MlabSceneModel
from pyface.api import error, confirm, OK, YES, NO, CANCEL, ProgressDialog, FileDialog
import traits.api as traits
from traits.api import HasTraits, HasPrivateTraits, cached_property, on_trait_change, Instance, Property, \
                       Any, Array, Bool, Button, Color, Directory, Enum, File, Float, Int, List, \
                       Range, Str, Tuple
from traitsui.api import View, Item, Group, HGroup, VGroup, EnumEditor
from traitsui.menu import Action, UndoButton, NoButtons, OKCancelButtons, OKButton, \
                          CancelButton, HelpButton
from tvtk.pyface.scene_editor import SceneEditor

from .coreg import trans_fname, fit_matched_pts, fit_point_cloud, scale_mri
from .fiducials_gui import FiducialsPanel
from .file_traits import BemSource, FidSource, RawHspSource, SubjectSelector
from .transforms import rotation, rotation3d, translation, apply_trans, write_trans
from .viewer import HeadViewController, PointObject, SurfaceObject, headview_item
from ..fiff import FIFF
from ..source_space import prepare_bem_model, setup_source_space
from ..utils import get_config, get_subjects_dir



class CoregControl(HasPrivateTraits):
    """Traits object for estimating the head mri transform.

    Notes
    -----
    Transform from head to mri space is modeled with the following steps:

     * move the head shape to its nasion position
     * rotate the head shape with user defined rotation around its nasion
     * move the head shape by user defined translation
     * move the head shape origin to the mri nasion

    If MRI scaling is enabled,

     * the MRI is scaled relative to its origin center (prior to any
       transformation of the digitizer head)
    """
    raw_dir = Str
    subject = Any  # MRI subject name (used for constructing trans file name)
    tgt_subject = Str  # subject for which the MRI will be scaled
    subjects_dir = Str

    # data source
    mri_pts = Array(shape=(None, 3))
    mri_fid = Array(shape=(3, 3))
    hsp_pts = Array(shape=(None, 3))
    hsp_fid = Array(shape=(3, 3))
    dig = List  # digitizer fiducials for info['dig']

    # dependsnt
    src_pts = Property(depends_on=['hsp_pts', 'hsp_fid'])  # hsp points moved to their nasion
    src_fid = Property(depends_on='hsp_fid')
    tgt_origin = Property(depends_on=['mri_fid', 'scale'])
    tgt_pts = Property(depends_on=['mri_pts', 'tgt_origin'])  # mri_pts scaled and moved to their nasion
    tgt_fid = Property(depends_on=['tgt_origin'])

    # parameters
    reset_params = Button(label='Reset')
    n_scale_params = Enum(0, 1, 3, desc="Scale the MRI to better fit the "
                          "subject's head shape (a new MRI subject will be "
                          "created with a name specified upon saving)")
    scale3 = Array(float, (1, 3), [[1, 1, 1]], label='Scale by 3')
    scale1 = Float(1, label='Scale by 1')
    rotation = Array(float, (1, 3))
    translation = Array(float, (1, 3))

    # transforms
    scale = Property(depends_on=['n_scale_params', 'scale3', 'scale1'])
    head_mri_trans = Property(depends_on=['tgt_origin', 'hsp_fid', 'translation',
                                          'rotation', 'scale'])

    # fitting
    has_fid_data = Property(Bool, depends_on=['mri_fid', 'hsp_fid'])
    has_pts_data = Property(Bool, depends_on=['mri_pts', 'hsp_pts'])
    # fitting with scaling
    fits_rot = Button(label='Fit Head Shape')
    fits_fid = Button(label='Fit Fiducials')
    fits_ap = Button(label='Fit LAP/RAP')
    # fitting without scaling
    fit_rot = Button(label='Fit Rotation')
    fit_fid = Button(label='Fit Fiducials')
    fit_ap = Button(label='Fit LAP/RAP')

    # saving
    can_save = Property(Bool, depends_on='dig')
    save = Button
    queue = Instance(Queue, ())
    has_worker = Bool(False)

    background_processing = Bool(False)

    # View Element
    axis_labels = Str("Right   \t\tAnterior\t\tSuperior")

    view = View(VGroup(HGroup(Item('reset_params', tooltip="Reset all "
                                   "coregistration parameters"),
                              show_labels=False),
                       '_',
                       Item('n_scale_params', label='MRI Scaling',
                            style='custom', show_label=True,
                            editor=EnumEditor(values={0: '1:No Scaling',
                                                      1: '2:1 Parameter',
                                                      3: '3:3 Parameters'},
                                              cols=3)),
                       Item('scale1', enabled_when='n_scale_params == 1',
                            label="Scale (x1)", show_label=True,
                            tooltip="Scale along all axes with this factor"),
                       Item('scale3', enabled_when='n_scale_params == 3',
                            label="Scale (x3)", show_label=True,
                            tooltip="Scaling along x (right), y (anterior) "
                            "and z (superior) axes"),
                       HGroup(Item('fits_rot', enabled_when='n_scale_params',
                                   tooltip="Rotate the digitizer "
                                   "head shape and scale the MRI so as to minimize the distance "
                                   "from each digitizer point to the closest MRI point"),
                              Item('fits_ap', enabled_when='n_scale_params == 1',
                                   tooltip="While leaving the nasion in "
                                   "place, rotate the digitizer head shape and scale the MRI so as to "
                                   "minimize the distance of the two "
                                   "auricular points"),
                              Item('fits_fid', enabled_when='n_scale_params == 1',
                                   tooltip="Move and rotate the digitizer "
                                   "head shape, and scale the MRI so as to "
                                   "minimize the distance of the three "
                                   "fiducials."),
                              show_labels=False),
                       '_',
                       Item('axis_labels', show_label=False, style='readonly'),
                       Item('translation', show_label=True,
                            tooltip="Movement into x (right), y (anterior) "
                            "and z (superior) direction"),
                       Item('rotation', show_label=True,
                            tooltip="Rotation around the right, anterior "
                            "and superior axes"),
                       HGroup(Item('fit_rot', enabled_when='has_pts_data',
                                   tooltip="Rotate the head shape (around the "
                                   "nasion) so as to minimize the distance from "
                                   "each head shape point to its closest MRI "
                                   "point"),
                              Item('fit_ap', enabled_when='has_fid_data',
                                   tooltip="Try to match the LAP and the RAP, "
                                   "leaving the Nasion in place"),
                              Item('fit_fid', enabled_when='has_fid_data',
                                   tooltip="Move and rotate the head shape so "
                                   "as to minimize the distance between the "
                                   "MRI and head shape fiducials"),
                              show_labels=False),
                       Item('save', enabled_when='can_save', show_label=False),
                       show_labels=False),
                buttons=[UndoButton])

    @cached_property
    def _get_can_save(self):
        return len(self.dig) > 0

    @cached_property
    def _get_has_fid_data(self):
        has = (np.any(self.mri_fid) and np.any(self.hsp_fid))
        return has

    @cached_property
    def _get_has_pts_data(self):
        has = (np.any(self.mri_pts) and np.any(self.hsp_pts))
        return has

    @cached_property
    def _get_head_mri_trans(self):
        if not self.has_fid_data:
            return np.eye(3)
        x, y, z = -self.hsp_fid[0]
        trans = translation(x, y, z)
        x, y, z = self.rotation[0]
        trans = dot(rotation(x, y, z), trans)
        x, y, z = self.translation[0] + self.tgt_origin
        trans = dot(translation(x, y, z), trans)
        return trans

    @cached_property
    def _get_scale(self):
        if self.n_scale_params == 0:
            return np.array(1)
        elif self.n_scale_params == 1:
            return np.array(self.scale1)
        else:
            return self.scale3[0]

    @cached_property
    def _get_src_pts(self):
        return self.hsp_pts - self.hsp_fid[0]

    @cached_property
    def _get_src_fid(self):
        return self.hsp_fid - self.hsp_fid[0]

    @cached_property
    def _get_tgt_origin(self):
        return self.mri_fid[0] * self.scale

    @cached_property
    def _get_tgt_pts(self):
        pts = self.mri_pts * self.scale
        pts -= self.tgt_origin
        return pts

    @cached_property
    def _get_tgt_fid(self):
        fid = self.mri_fid * self.scale
        fid -= self.tgt_origin
        return fid

    def _fit_ap_fired(self):
        tgt_fid = self.tgt_fid[1:] - self.translation[0]
        x0 = tuple(self.rotation[0])
        rot = fit_matched_pts(self.src_fid[1:], tgt_fid, rotate=True,
                              translate=False, x0=x0)
        self.rotation = [rot]

    def _fit_fid_fired(self):
        x0 = tuple(self.rotation[0]) + tuple(self.translation[0])
        est = fit_matched_pts(self.src_fid, self.tgt_fid, x0=x0)
        self.rotation = [est[:3]]
        self.translation = [est[3:]]

    def _fit_rot_fired(self):
        tgt_pts = self.tgt_pts - self.translation[0]
        x0 = tuple(self.rotation[0])
        rot = fit_point_cloud(self.src_pts, tgt_pts, rotate=True,
                              translate=False, x0=x0)
        self.rotation = [rot]

    def _fits_ap_fired(self):
        tgt_fid = self.mri_fid[1:] - self.mri_fid[0]
        tgt_fid -= self.translation[0]
        x0 = tuple(self.rotation[0]) + (1 / self.scale1,)
        x = fit_matched_pts(self.src_fid[1:], tgt_fid, rotate=True,
                            translate=False, scale=1, x0=x0)
        self.scale1 = 1 / x[3]
        self.rotation = [x[:3]]

    def _fits_fid_fired(self):
        tgt_fid = self.mri_fid - self.mri_fid[0]
        x0 = tuple(self.rotation[0]) + tuple(self.translation[0]) + (1 / self.scale1,)
        x = fit_matched_pts(self.src_fid, tgt_fid, rotate=True,
                            translate=True, scale=1, x0=x0)
        self.scale1 = 1 / x[6]
        self.rotation = [x[:3]]
        self.translation = [x[3:6]]

    def _fits_rot_fired(self):
        if self.n_scale_params == 1:
            tgt_pts = self.mri_pts - self.tgt_origin
            x0 = tuple(self.rotation[0]) + (1 / self.scale1,)
            est = fit_point_cloud(self.src_pts, tgt_pts, rotate=True,
                                  translate=False, scale=1, x0=x0)
            self.scale1 = 1 / est[3]
        else:
            x0 = tuple(self.rotation[0]) + tuple(1 / self.scale)
            est = fit_point_cloud(self.src_pts, self.tgt_pts, rotate=True,
                                  translate=False, scale=3, x0=x0)
            self.scale3 = [1 / est[3:]]
        self.rotation = [est[:3]]

    def _reset_params_fired(self):
        self.reset_traits(('n_scaling_params', 'scale1', 'scale3',
                           'translation', 'rotation'))

    def _save_fired(self):
        # find target subject and MRI options
        if self.n_scale_params:
            mridlg = NewMriDialog(subjects_dir=self.subjects_dir,
                                  subject=self.tgt_subject,
                                  src_subject=self.subject)
            ui = mridlg.edit_traits(kind='modal')
            if ui.result != True:
                return
            subject = mridlg.subject
        else:
            subject = self.subject

        # find trans file destination
        trans_file = trans_fname.format(raw_dir=self.raw_dir, subject=subject)
        dlg = FileDialog(action="save as", wildcard="Trans File (*.fif)|"
                         "*.fif", default_path=trans_file)
        dlg.open()
        if dlg.return_code != OK:
            return

        dest = dlg.path
        if not dest.endswith('.fif'):
            dest = dest + '.fif'
            if os.path.exists(dest):
                answer = confirm(None, "The file %r already exists. Should it be "
                                 "replaced?", "Overwrite File?")
                if answer != YES:
                    return

        # save the trans file
        trans = self.head_mri_trans
        dig = self.dig
        if trans is not None:
            for i in xrange(len(dig)):
                dig[i]['r'] = apply_trans(trans, dig[i]['r'])

        info = {'to': FIFF.FIFFV_COORD_MRI, 'from': FIFF.FIFFV_COORD_HEAD,
                'trans': trans, 'dig': dig}
        try:
            write_trans(dest, info)
        except Exception as e:
            error(None, str(e), "Error Saving Trans File")

        # save the scaled MRI
        if self.n_scale_params:
            bemdir = os.path.join(self.subjects_dir, subject, 'bem')
            bem = os.path.join(bemdir, '%s-inner_skull-bem.fif' % subject)

            if mridlg.background or self.background_processing:
                if not self.has_worker:
                    def worker():
                        while True:
                            cmd, args, kwargs = self.queue.get()
                            cmd(*args, **kwargs)
                            self.queue.task_done()

                    t = Thread(target=worker)
                    t.daemon = True
                    t.start()
                    self.has_worker = True

                self.queue.put((scale_mri, (self.subject, subject),
                                dict(scale=self.scale, overwrite=True)))

                if mridlg.prepare_bem_model:
                    self.queue.put((prepare_bem_model, (bem,), {}))

                if mridlg.setup_source_space:
                    if mridlg.ss_subd == 'ico':
                        self.queue.put((setup_source_space, (subject,),
                                        dict(ico=mridlg.ss_param,
                                             subjects_dir=self.subjects_dir)))
                    else:
                        raise NotImplementedError

                return

            # progress dialog
            title = "Saving %s..." % subject
            message = "Saving scaled MRI %s for %s" % (self.subject, subject)
            vmax = 1 + mridlg.prepare_bem_model + mridlg.setup_source_space
            if vmax == 1:  # indefinite progress bar
                prog = ProgressDialog(title=title, message=message)
            else:
                prog = ProgressDialog(title=title, message=message, min=0,
                                      max=vmax)

            progi = 0
            prog.open()
            prog.update(0)

            try:
                scale_mri(self.subject, subject, scale=self.scale,
                          overwrite=True, subjects_dir=self.subjects_dir)
            except Exception as e:
                error(None, str(e), "Error while Saving Scaled MRI")

            if mridlg.prepare_bem_model:
                progi += 1
                prog.update(progi)
                prog.change_message("Running mne_prepare_bem_model...")

                try:
                    prepare_bem_model(bem)
                except Exception as e:
                    err = "%s\n\nSee log for more information." % str(e)
                    error(None, err, "Error in mne_prepare_bem_model")

            if mridlg.setup_source_space:
                progi += 1
                prog.update(progi)
                prog.change_message("Running mne_setup_source_space...")

                try:
                    if mridlg.ss_subd == 'ico':
                        setup_source_space(subject, ico=mridlg.ss_param,
                                           subjects_dir=self.subjects_dir)
                    else:
                        err = ("Can only use ico parameter, not "
                               "%s" % mridlg.ss_subd)
                        raise NotImplementedError(err)
                except Exception as e:
                    err = "%s\n\nSee log for more information." % str(e)
                    error(None, err, "Error in mne_setup_source_space")

            prog.close()



class NewMriDialog(HasPrivateTraits):
    subjects_dir = Directory
    subject = Str
    src_subject = Str
    sdir = Property(depends_on=['subjects_dir', 'subject'])
    subject_exists = Property(Bool, depends_on='sdir')

    feedback = Str(' ' * 100)
    can_delete = Bool
    overwrite = Bool
    can_save = Bool

    prepare_bem_model = Bool(True)
    setup_source_space = Bool(True)
    ss_subd = Enum('ico', 'spacing')
    ss_param = Int(4)

    background = Bool(False)

    view = View(Item('subject', label='New MRI Subject Name', tooltip="A new "
                     "folder with this name will be created in the current "
                     "subjects_dir for the scaled MRI files"),
                Item('feedback', show_label=False, style='readonly'),
                Item('overwrite', enabled_when='can_delete', tooltip="If a "
                     "subject with the chosen name exists, delete the old "
                     "subject"),
                '_',
                Item('prepare_bem_model', tooltip="Execute "
                     "mne_prepare_bem_model after saving the scaled brain. "
                     "This step is required before a forward solution can be "
                     "generated"),
                Item('setup_source_space', tooltip="Execute "
                     "mne_setup_source_space after saving the scaled brain. "
                     "This step is required before a forward solution can be "
                     "generated"),
                Item('ss_subd', label='Subdivision Method',
                     enabled_when='setup_source_space'),
                Item('ss_param', label='Subdivision Parameter',
                     enabled_when='setup_source_space'),
#                '_',
#                Item('background', label='Run in Background', tooltip="Exe"
#                     "cute MRI preparation in the background (in a searate "
#                     "thread) without blocking the the current "
#                     "GUI/interpreter"),
                width=500,
                buttons=[CancelButton,
                           Action(name='OK', enabled_when='can_save')])

    @cached_property
    def _get_sdir(self):
        return os.path.join(self.subjects_dir, self.subject)

    @cached_property
    def _get_subject_exists(self):
        if not self.subject:
            return False
        elif os.path.exists(self.sdir):
            return True
        else:
            return False

    @on_trait_change('subjects_dir,subject')
    def update_feedback(self):
        self.overwrite = False
        if not self.subject:
            self.feedback = "No subject specified..."
            self.can_save = False
            self.can_delete = False
        elif self.subject == self.src_subject:
            self.feedback = "Must be different from MRI source subject..."
            self.can_save = False
            self.can_delete = False
        elif self.subject_exists:
            self.feedback = "Subject already exists..."
            self.can_save = False
            self.can_delete = True
        else:
            self.feedback = "Name ok."
            self.can_save = True
            self.can_delete = False

    @on_trait_change('overwrite')
    def on_overwrite_change(self, new):
        if not new:
            self.update_feedback()
            return

        subject = self.subject
        title = "Overwrite %s?" % subject
        msg = ("The current MRI subject %s will be deleted. This can not be "
               "undone." % subject)
        answer = confirm(None, msg, title, cancel=False, default=NO)
        if answer == YES:
            self.feedback = "%s will be overwritten." % subject
            self.can_save = True
        else:
#            self.reset_traits(['overwrite'])
#            self.overwrite_status = 'declined'
#            obj.trait_set(overwrite=False)
            self.overwrite = False
            self.update_feedback()
            self.feedback = "Will NOT overwrite"
#            self.trait_property_changed('overwrite', False, False)

#    @on_trait_change('overwrite_status')
#    def on_overwrite_status(self, new):
#        if new == 'declined':
#            self.overwrite = False
#            self.overwrite_status = 'None'


# view for high resolution (all controls on the right)
view_hrs = View(HGroup(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                            dock='vertical'),
                       VGroup(VGroup(headview_item,
                                     Item('mri_obj', label='MRI', style='custom'),
                                     label='View', show_labels=False,
                                     show_border=True),
                              VGroup(Item('hsp_src', style="custom"),
                                     Item('s_sel', style="custom"),
                                     label='Data Source', show_labels=False,
                                     show_border=True),
                              VGroup(Item('lock_fiducials', style='custom',
                                          editor=EnumEditor(values={False: '2:Edit',
                                                                    True: '1:Lock'},
                                                            cols=2)),
                                     Item('fid_panel', style='custom'),
                                     label='MRI Fiducials', show_labels=False,
                                     show_border=True),
                              VGroup(Item('coreg', style='custom'),
                                     label='Coregistration', show_labels=False,
                                     show_border=True, enabled_when='lock_fiducials'),
                              show_labels=False),
                       show_labels=False,
                      ),
                resizable=True,  # height=0.75,
                width=1100,  # 0.75,
                buttons=[UndoButton])  # HelpButton


class CoregFrame(HasTraits):
    """GUI for interpolating between two KIT marker files

    Parameters
    ----------
    mrk1, mrk2 : str
        Path to pre- and post measurement marker files (*.sqd) or empty string.
    """
    # controls
    headview = Instance(HeadViewController)
    mri_src = Instance(BemSource, ())
    hsp_src = Instance(RawHspSource, ())
#    mri_fid_src = Instance(FidSource, ())
    s_sel = Instance(SubjectSelector, ())
    coreg = Instance(CoregControl, ())

    pick_tolerance = Float(.0025)

    # fiducials
    lock_fiducials = Bool(True)
    fid_panel = Instance(FiducialsPanel)

    # visualization
    scene = Instance(MlabSceneModel, ())
    hsp_obj = Instance(PointObject)
    mri_obj = Instance(SurfaceObject)
    hsp_fid_obj = Instance(PointObject)
    lap_obj = Instance(PointObject)
    nas_obj = Instance(PointObject)
    rap_obj = Instance(PointObject)

    view = View(HGroup(VGroup(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                                   dock='vertical', show_label=False),
                              VGroup(headview_item,
                                     Item('mri_obj', label='MRI', style='custom'),
                                     Item('hsp_obj', label='Head Shape', style='custom'),
                                     label='View Options',
                                     show_border=True
#                                      show_labels=False,
                                     ),
                              ),
                       VGroup(
                              VGroup(Item('hsp_src', style="custom"),
                                     Item('s_sel', style="custom"),
                                     label='Data Source', show_labels=False,
                                     show_border=True),
                              VGroup(HGroup(Item('lock_fiducials', style='custom',
                                                 editor=EnumEditor(values={False: '2:Edit',
                                                                           True: '1:Lock'},
                                                            cols=2), show_label=False)),
                                     Item('fid_panel', style='custom'),
                                     label='MRI Fiducials', show_labels=False,
                                     show_border=True),
                              VGroup(Item('coreg', style='custom'),
                                     label='Coregistration', show_labels=False,
                                     show_border=True, enabled_when='lock_fiducials'),
                              show_labels=False),
                       show_labels=False,
                      ),
                resizable=True, buttons=[UndoButton])  # HelpButton

    def _fid_panel_default(self):
        return FiducialsPanel(scene=self.scene, headview=self.headview)

    def _headview_default(self):
        return HeadViewController(scene=self.scene, system='RAS')

    def __init__(self, raw=None, subject=None, subjects_dir=None):
        super(CoregFrame, self).__init__()

        # process parameters
        subjects_dir = get_subjects_dir(subjects_dir)

        # sync mri subject
        self.s_sel.on_trait_change(self._on_bem_file_change, 'bem_file')
        self.s_sel.sync_trait('subjects_dir', self.coreg, mutual=False)

        # sync data to coreg panel
        self.mri_src.sync_trait('pts', self.coreg, 'mri_pts', mutual=False)
#        self.mri_fid_src.sync_trait('fid', self.coreg, 'mri_fid', mutual=False)
        self.hsp_src.sync_trait('pts', self.coreg, 'hsp_pts', mutual=False)
        self.hsp_src.sync_trait('fid', self.coreg, 'hsp_fid', mutual=False)
        self.hsp_src.sync_trait('fid_dig', self.coreg, 'dig', mutual=False)

        # sync ficudials panel
        self.sync_trait('lock_fiducials', self.fid_panel, 'locked', mutual=True)

        # sync path source to coreg panel
        self.hsp_src.sync_trait('raw_dir', self.coreg, mutual=False)
        self.s_sel.sync_trait('subject', self.coreg, mutual=False)

        # set initial parameters
        if raw is not None:
            self.hsp_src.raw_file = raw
            if subject is None:
                raw_name = os.path.basename(raw)
                if '_' in raw_name:
                    subject, _ = raw_name.split('_', 1)
        if subjects_dir is not None:
            self.s_sel.subjects_dir = subjects_dir
        if subject is not None:
            if subject in self.s_sel.subjects:
                self.s_sel.subject = subject

        # sync path components to fiducials panel
        self.s_sel.sync_trait('subjects_dir', self.fid_panel, mutual=False)
        self.s_sel.sync_trait('subject', self.fid_panel, mutual=False)

        # lock fiducials if file is found
        if self.fid_panel.fid_file:
            self.lock_fiducials = True

    @on_trait_change('s_sel.subject')
    def _on_subject_changes(self, new):
        if new:
            self.fid_panel.subject = new

    @on_trait_change('scene.activated')
    def init_plot(self):
        self.scene.disable_render = True

        # MRI scalp
        self.mri_obj = SurfaceObject(points=self.mri_src.pts, tri=self.mri_src.tri,
                                     scene=self.scene, color=(255, 255, 255))
        self.coreg.sync_trait('scale', self.mri_obj, 'trans', mutual=False)
        self.fid_panel.hsp_obj = self.mri_obj

        # MRI Fiducials
#        self.mri_fid_obj = PointObject(scene=self.scene, color=(255, 0, 0), point_scale=1e-2)
#        self.mri_fid_src.sync_trait('fid', self.mri_fid_obj, 'points', mutual=False)
#        self.coreg.sync_trait('scale', self.mri_fid_obj, 'trans', mutual=False)
        self.lap_obj = PointObject(scene=self.scene, color=(255, 0, 0),
                                   point_scale=1e-2)
        self.fid_panel.sync_trait('LAP', self.lap_obj, 'points', mutual=False)
        self.coreg.sync_trait('scale', self.lap_obj, 'trans', mutual=False)

        self.nas_obj = PointObject(scene=self.scene, color=(0, 255, 0),
                                   point_scale=1e-2)
        self.fid_panel.sync_trait('nasion', self.nas_obj, 'points', mutual=False)
        self.coreg.sync_trait('scale', self.nas_obj, 'trans', mutual=False)

        self.rap_obj = PointObject(scene=self.scene, color=(0, 0, 255),
                                   point_scale=1e-2)
        self.fid_panel.sync_trait('RAP', self.rap_obj, 'points', mutual=False)
        self.coreg.sync_trait('scale', self.rap_obj, 'trans', mutual=False)

        # Digitizer Head Shape
        self.hsp_obj = PointObject(scene=self.scene, color=(255, 255, 255),
                                   point_scale=5e-3)
        self.hsp_src.sync_trait('pts', self.hsp_obj, 'points', mutual=False)
        self.coreg.sync_trait('head_mri_trans', self.hsp_obj, 'trans',
                              mutual=False)

        # Digitizer Fiducials
        self.hsp_fid_obj = PointObject(scene=self.scene, color=(0, 0, 255),
                                       opacity=0.3, point_scale=3e-2)
        self.hsp_src.sync_trait('fid', self.hsp_fid_obj, 'points', mutual=False)
        self.coreg.sync_trait('head_mri_trans', self.hsp_fid_obj, 'trans', mutual=False)

        mscene = self.scene.mayavi_scene
        self.picker = mscene.on_mouse_pick(self.fid_panel._on_mouse_click)
        self.headview.left = True
        self.scene.disable_render = False

        # adapt picker sensitivity when zooming
        self.scene.camera.on_trait_change(self._on_view_scale_change, 'parallel_scale')

    @on_trait_change('lock_fiducials')
    def _on_lock_fiducials(self):
        if (not self.hsp_obj) or (not self.hsp_fid_obj):
            return

        if self.lock_fiducials:
            self.hsp_obj.visible = True
            self.hsp_fid_obj.visible = True
            fid = np.vstack((self.fid_panel.nasion,
                             self.fid_panel.LAP,
                             self.fid_panel.RAP))
            self.coreg.mri_fid = fid
        else:
            self.hsp_obj.visible = False
            self.hsp_fid_obj.visible = False

#     @on_trait_change('scene.camera.parallel_scale', post_init=True)
    def _on_view_scale_change(self, scale):
        self.picker.tolerance = self.pick_tolerance / scale

    def _on_bem_file_change(self):
        bem_file = self.s_sel.bem_file
        if not bem_file:
            self.mri_src.reset_traits(['file'])
            self.fid_panel.reset_traits(['fid_file'])
            return
        self.mri_src.file = bem_file % 'head'
        fid_file = bem_file % 'fiducials'
        if os.path.exists(fid_file):
            self.fid_panel.fid_file = fid_file
            if self.lock_fiducials:
                self._on_lock_fiducials()
            else:
                self.lock_fiducials = True
        else:
            self.fid_panel.reset_traits(('fid_file',))
            self.lock_fiducials = False

    @on_trait_change('mri_src.tri')
    def _on_mri_src_change(self):
        if not self.mri_obj:
            return
        elif (not np.any(self.mri_src.pts)) or (not np.any(self.mri_src.tri)):
            self.mri_obj.clear()
            return

        self.mri_obj.points = self.mri_src.pts
        self.mri_obj.tri = self.mri_src.tri
        self.mri_obj.plot()

    @on_trait_change('hsp_src.raw_fname')
    def _on_raw_change(self, fname):
        # try to guess subject from fname
        if '_' in fname:
            subject, _ = fname.split('_', 1)
            self.coreg.tgt_subject = subject
            if subject in self.s_sel.subjects:
                self.s_sel.subject = subject
            elif 'fsaverage' in self.s_sel.subjects:
                self.s_sel.subject = 'fsaverage'
