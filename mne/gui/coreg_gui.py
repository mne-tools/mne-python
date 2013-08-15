"""Traits-based GUI for head-MRI coregistration"""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os
from Queue import Queue
from threading import Thread

import numpy as np
from numpy import dot
from scipy.spatial.distance import cdist

from mayavi.core.ui.mayavi_scene import MayaviScene
from mayavi.tools.mlab_scene_model import MlabSceneModel
from pyface.api import error, confirm, warning, OK, YES, information, \
                       FileDialog, GUI
from traits.api import HasTraits, HasPrivateTraits, cached_property, \
                       on_trait_change, Instance, Property, Any, Array, Bool, \
                       Button, Directory, Enum, Float, Int, List, Str
from traitsui.api import View, Item, HGroup, VGroup, EnumEditor, TextEditor, \
                         Handler, Label, Spring
from traitsui.menu import Action, UndoButton, CancelButton
from tvtk.pyface.scene_editor import SceneEditor

from ..fiff import FIFF
from ..source_space import prepare_bem_model, setup_source_space
from ..transforms import write_trans, rotation, translation, apply_trans
from ..transforms.coreg import trans_fname, fit_matched_pts, fit_point_cloud, \
                               scale_mri
from ..utils import get_subjects_dir
from .fiducials_gui import FiducialsPanel
from .file_traits import assert_env_set, BemSource, RawHspSource, \
                         SubjectSelector
from .viewer import HeadViewController, PointObject, SurfaceObject, \
                    headview_item


laggy_float_editor = TextEditor(auto_set=False, enter_set=True, evaluate=float)


class CoregFrameHandler(Handler):
    """Handler that checks for unfinished processes before closing its window
    """
    def close(self, info, is_ok):
        if info.object.coreg.queue.unfinished_tasks:
            msg = ("Can not close the window while saving is still in "
                   "progress. Please wait until all MRIs are processed.")
            title = "Saving Still in Progress"
            information(None, msg, title)
            return False
        else:
            return True


class CoregPanel(HasPrivateTraits):
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
    subject = Any("MRI subject name")
    tgt_subject = Str(desc="subject for which the MRI will be scaled, "
                      "inferred from raw file name")
    subjects_dir = Str

    # data source
    mri_pts = Array(shape=(None, 3))
    mri_fid = Array(shape=(3, 3))
    hsp_pts = Array(shape=(None, 3))
    hsp_fid = Array(shape=(3, 3))
    dig = List(desc="digitizer fiducials for info['dig']")

    # dependent
    src_pts = Property(depends_on=['hsp_pts', 'hsp_fid'], desc="hsp points "
                       "moved to their nasion")
    src_fid = Property(depends_on='hsp_fid')
    tgt_origin = Property(depends_on=['mri_fid', 'scale'])
    tgt_pts = Property(depends_on=['mri_pts', 'tgt_origin'], desc="mri_pts "
                       "scaled and moved to their nasion")
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
    head_mri_trans = Property(depends_on=['tgt_origin', 'hsp_fid',
                                          'translation', 'rotation', 'scale'])

    # fitting
    has_fid_data = Property(Bool, depends_on=['mri_fid', 'hsp_fid'])
    has_pts_data = Property(Bool, depends_on=['mri_pts', 'hsp_pts'])
    # fitting with scaling
    fits_rot = Button(label='Fit Head Shape')
    fits_fid = Button(label='Fit Fiducials')
    fits_ap = Button(label='Fit LAP/RAP')
    # fitting without scaling
    fit_rot = Button(label='Fit Head Shape')
    fit_fid = Button(label='Fit Fiducials')
    fit_ap = Button(label='Fit LAP/RAP')

    # saving
    can_save = Property(Bool, depends_on='dig')
    save = Button
    queue = Instance(Queue, ())
    queue_feedback = Str('')
    queue_current = Str('')
    queue_len = Int(0)
    queue_len_str = Property(Str, depends_on=['queue_len'])
    error = Str('')

    view = View(VGroup(Item('n_scale_params', label='MRI Scaling',
                            style='custom', show_label=True,
                            editor=EnumEditor(values={0: '1:No Scaling',
                                                      1: '2:1 Param',
                                                      3: '3:3 Params'},
                                              cols=3)),
                       Item('scale1', enabled_when='n_scale_params == 1',
                            label="Scale (x1)", show_label=True,
                            editor=laggy_float_editor,
                            tooltip="Scale along all axes with this factor"),
                       Item('scale3', enabled_when='n_scale_params == 3',
                            label="Scale (x3)", show_label=True,
                            tooltip="Scaling along x (right), y (anterior) "
                            "and z (superior) axes"),
                       HGroup(Item('fits_rot', enabled_when='n_scale_params',
                                   tooltip="Rotate the digitizer head shape "
                                   "and scale the MRI so as to minimize the "
                                   "distance from each digitizer point to the "
                                   "closest MRI point"),
                              Item('fits_ap',
                                   enabled_when='n_scale_params == 1',
                                   tooltip="While leaving the nasion in "
                                   "place, rotate the digitizer head shape "
                                   "and scale the MRI so as to minimize the "
                                   "distance of the two auricular points"),
                              Item('fits_fid',
                                   enabled_when='n_scale_params == 1',
                                   tooltip="Move and rotate the digitizer "
                                   "head shape, and scale the MRI so as to "
                                   "minimize the distance of the three "
                                   "fiducials."),
                              show_labels=False),
                       '_',
                       HGroup(Label("Axis:"), Spring(), Spring(),
                              Label("Right"), Spring(), Label("Anterior"),
                              Spring(), Label("Superior"), Spring()),
                       Item('translation', show_label=True,
                            tooltip="Movement into x (right), y (anterior) "
                            "and z (superior) direction"),
                       Item('rotation', show_label=True,
                            tooltip="Rotation around the right, anterior "
                            "and superior axes"),
                       HGroup(Item('fit_rot', enabled_when='has_pts_data',
                                   tooltip="Rotate the head shape (around the "
                                   "nasion) so as to minimize the distance "
                                   "from each head shape point to its closest "
                                   "MRI point"),
                              Item('fit_ap', enabled_when='has_fid_data',
                                   tooltip="Try to match the LAP and the RAP, "
                                   "leaving the Nasion in place"),
                              Item('fit_fid', enabled_when='has_fid_data',
                                   tooltip="Move and rotate the head shape so "
                                   "as to minimize the distance between the "
                                   "MRI and head shape fiducials"),
                              show_labels=False),
                       '_',
                       HGroup(Item('save', enabled_when='can_save',
                                   tooltip="Save the trans file and (if "
                                   "scaling is enabled) the scaled MRI"),
                              Item('reset_params', tooltip="Reset all "
                                   "coregistration parameters"),
                              show_labels=False),
                       Item('queue_feedback', style='readonly'),
                       Item('queue_current', style='readonly'),
                       Item('queue_len_str', style='readonly'),
                       show_labels=False),
                kind='panel', buttons=[UndoButton])

    def __init__(self, *args, **kwargs):
        super(CoregPanel, self).__init__(*args, **kwargs)

        # setup save worker
        def worker():
            while True:
                desc, cmd, args, kwargs = self.queue.get()

                self.queue_len -= 1
                self.queue_current = 'Processing: %s' % desc

                # task
                try:
                    cmd(*args, **kwargs)
                except Exception as err:
                    self.error = str(err)
                    res = "Error in %s"
                else:
                    res = "Done: %s"

                # finalize
                self.queue_current = ''
                self.queue_feedback = res % desc
                self.queue.task_done()

        t = Thread(target=worker)
        t.daemon = True
        t.start()

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
    def _get_queue_len_str(self):
        if self.queue_len:
            return "Queue length: %i" % self.queue_len
        else:
            return ''

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
        GUI.set_busy()
        tgt_fid = self.tgt_fid[1:] - self.translation[0]
        x0 = tuple(self.rotation[0])
        rot = fit_matched_pts(self.src_fid[1:], tgt_fid, rotate=True,
                              translate=False, x0=x0)
        self.rotation = [rot]
        GUI.set_busy(False)

    def _fit_fid_fired(self):
        GUI.set_busy()
        x0 = tuple(self.rotation[0]) + tuple(self.translation[0])
        est = fit_matched_pts(self.src_fid, self.tgt_fid, x0=x0)
        self.rotation = [est[:3]]
        self.translation = [est[3:]]
        GUI.set_busy(False)

    def _fit_rot_fired(self):
        GUI.set_busy()
        tgt_pts = self.tgt_pts - self.translation[0]
        x0 = tuple(self.rotation[0])
        rot = fit_point_cloud(self.src_pts, tgt_pts, rotate=True,
                              translate=False, x0=x0)
        self.rotation = [rot]
        GUI.set_busy(False)

    def _fits_ap_fired(self):
        GUI.set_busy()
        tgt_fid = self.mri_fid[1:] - self.mri_fid[0]
        tgt_fid -= self.translation[0]
        x0 = tuple(self.rotation[0]) + (1 / self.scale1,)
        x = fit_matched_pts(self.src_fid[1:], tgt_fid, rotate=True,
                            translate=False, scale=1, x0=x0)
        self.scale1 = 1 / x[3]
        self.rotation = [x[:3]]
        GUI.set_busy(False)

    def _fits_fid_fired(self):
        GUI.set_busy()
        tgt_fid = self.mri_fid - self.mri_fid[0]
        x0 = tuple(self.rotation[0]) + tuple(self.translation[0]) \
             + (1 / self.scale1,)
        x = fit_matched_pts(self.src_fid, tgt_fid, rotate=True,
                            translate=True, scale=1, x0=x0)
        self.scale1 = 1 / x[6]
        self.rotation = [x[:3]]
        self.translation = [x[3:6]]
        GUI.set_busy(False)

    def _fits_rot_fired(self):
        GUI.set_busy()
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
        GUI.set_busy(False)

    def _n_scale_params_changed(self, new):
        if not new:
            return

        # Make sure that MNE_ROOT environment variable is set
        if not assert_env_set(mne_root=True):
            err = ("MNE_ROOT environment variable could not be set. "
                   "You will be able to scale MRIs, but the preparatory mne "
                   "tools will fail. Please specify the MNE_ROOT environment "
                   "variable. In Python this can be done using:\n\n"
                   ">>> os.environ['MNE_ROOT'] = '/Applications/mne-2.7.3'")
            warning(None, err, "MNE_ROOT Not Set")

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
                answer = confirm(None, "The file %r already exists. Should it "
                                 "be replaced?", "Overwrite File?")
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

            self.queue.put(('Scaling %s' % subject, scale_mri,
                            (self.subject, subject),
                            dict(scale=self.scale, overwrite=True,
                                 subjects_dir=self.subjects_dir)))
            self.queue_len += 1

            if mridlg.prepare_bem_model:
                self.queue.put(('mne_prepare_bem_model %s' % subject,
                                prepare_bem_model, (bem,), {}))
                self.queue_len += 1

            if mridlg.prepare_bem_model and mridlg.setup_source_space:
                if mridlg.ss_subd == 'ico':
                    self.queue.put(('mne_setup_source_space %s' % subject,
                                    setup_source_space, (subject,),
                                    dict(ico=mridlg.ss_param,
                                         subjects_dir=self.subjects_dir)))
                    self.queue_len += 1
                elif mridlg.ss_subd == 'spacing':
                    self.queue.put(('mne_setup_source_space %s' % subject,
                                    setup_source_space, (subject,),
                                    dict(spacing=mridlg.ss_param,
                                         subjects_dir=self.subjects_dir)))
                    self.queue_len += 1
                else:
                    err = ("ss_param needs to be 'ico' or 'spacing', can "
                           "not be %s" % mridlg.ss_subd)
                    raise ValueError(err)


class NewMriDialog(HasPrivateTraits):
    subjects_dir = Directory
    subject = Str
    src_subject = Str
    sdir = Property(depends_on=['subjects_dir', 'subject'])
    subject_exists = Property(Bool, depends_on='sdir')

    feedback = Str(' ' * 100)
    can_overwrite = Bool
    overwrite = Bool
    can_save = Bool

    prepare_bem_model = Bool(True)
    setup_source_space = Bool(True)
    ss_subd = Enum('ico', 'spacing')
    ss_param = Int(4)

    view = View(Item('subject', label='New MRI Subject Name', tooltip="A new "
                     "folder with this name will be created in the current "
                     "subjects_dir for the scaled MRI files"),
                Item('feedback', show_label=False, style='readonly'),
                Item('overwrite', enabled_when='can_overwrite', tooltip="If a "
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
                     "generated", enabled_when='prepare_bem_model'),
                Item('ss_subd', label='Subdivision Method', tooltip="mne_"
                     "setup_source_space parameter",
                     enabled_when='setup_source_space'),
                Item('ss_param', label='Subdivision Parameter', tooltip="mne_"
                     "setup_source_space parameter",
                     enabled_when='setup_source_space'),
                width=500,
                buttons=[CancelButton,
                           Action(name='OK', enabled_when='can_save')])

    def _can_overwrite_changed(self, new):
        if not new:
            self.overwrite = False

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

    @on_trait_change('sdir,overwrite')
    def update_dialog(self):
        if not self.subject:
            self.feedback = "No subject specified..."
            self.can_save = False
            self.can_overwrite = False
        elif self.subject == self.src_subject:
            self.feedback = "Must be different from MRI source subject..."
            self.can_save = False
            self.can_overwrite = False
        elif self.subject_exists:
            if self.overwrite:
                self.feedback = "%s will be overwritten." % self.subject
                self.can_save = True
                self.can_overwrite = True
            else:
                self.feedback = "Subject already exists..."
                self.can_save = False
                self.can_overwrite = True
        else:
            self.feedback = "Name ok."
            self.can_save = True
            self.can_overwrite = False


class CoregFrame(HasTraits):
    """GUI for interpolating between two KIT marker files
    """
    # controls
    headview = Instance(HeadViewController)
    mri_src = Instance(BemSource, ())
    hsp_src = Instance(RawHspSource, ())
    s_sel = Instance(SubjectSelector, ())
    coreg = Instance(CoregPanel, ())

    pick_tolerance = Float(.0025)

    # fiducials
    fid_panel = Instance(FiducialsPanel)
    fid_ok = Bool(False)
    lock_fiducials = Bool(True)

    # visualization
    scene = Instance(MlabSceneModel, ())
    hsp_obj = Instance(PointObject)
    mri_obj = Instance(SurfaceObject)
    hsp_fid_obj = Instance(PointObject)
    lap_obj = Instance(PointObject)
    nas_obj = Instance(PointObject)
    rap_obj = Instance(PointObject)

    # feedback strings
    fit_eval_fid = Str('-')
    fit_eval_pts = Str('-')

    view = View(
        HGroup(
            VGroup(
                VGroup(Item('hsp_src', style="custom"),
                       Item('s_sel', style="custom"),
                       label='Data Source', show_labels=False,
                       show_border=True),
                VGroup(HGroup(Item('lock_fiducials', style='custom',
                                   editor=EnumEditor(cols=2,
                                                     values={False: '2:Edit',
                                                             True: '1:Lock'}),
                                   show_label=False,
                                   enabled_when='fid_ok')),
                       Item('fid_panel', style='custom'),
                       label='MRI Fiducials', show_labels=False,
                       show_border=True)),
            VGroup(Item('scene',
                        editor=SceneEditor(scene_class=MayaviScene),
                        dock='vertical', show_label=False),
                   VGroup(headview_item,
                          VGroup(
                                Item('mri_obj', label='MRI', style='custom'),
                                Item('hsp_obj', label='HSP', style='custom')),
                          label='View Options', show_border=True,
                          show_labels=False)),
            VGroup(VGroup(Item('coreg', style='custom'),
                          label='Coregistration', show_labels=False,
                          show_border=True,
                          enabled_when='lock_fiducials'),
                   VGroup(Item('fit_eval_fid', style='readonly'),
                          Item('fit_eval_pts', style='readonly'),
                          label='Fit', show_labels=False,
                          show_border=True),
                   show_labels=False),
            show_labels=False),
        resizable=True, buttons=[UndoButton], handler=CoregFrameHandler(),
        height=700)

    def _fid_panel_default(self):
        return FiducialsPanel(scene=self.scene, headview=self.headview)

    @on_trait_change('hsp_fid_obj.src.data.points', True)
    def _update_fit_eval_fid(self):
        if np.all(self.hsp_fid_obj.points == 0):
            self.fit_eval_fid = '-'
            return

        mri_fid = np.vstack((self.nas_obj.src.data.points,
                             self.lap_obj.src.data.points,
                             self.rap_obj.src.data.points))
        head_fid = np.array(self.hsp_fid_obj.src.data.points)
        dists = np.sqrt(np.sum((mri_fid - head_fid) ** 2, 1))
        dists *= 1000
        self.fit_eval_fid = ("Fiducials Error: NAS %.1f mm, LAP %.1f mm, RAP "
                             "%.1f mm" % tuple(dists))

    @on_trait_change('hsp_obj.src.data.points', True)
    def _update_fit_eval_pts(self):
        if np.all(self.hsp_obj.points == 0) or (self.mri_obj.src is None):
            self.fit_eval_pts = '-'
            return

        mri_pts = np.array(self.mri_obj.src.data.points)
        head_pts = np.array(self.hsp_obj.src.data.points)
        dists = cdist(head_pts, mri_pts, 'euclidean')
        dists = np.min(dists, 1)
        av_dist = np.mean(dists)
        self.fit_eval_pts = "Average Points Error: %.1f mm" % (av_dist * 1000)

    def _headview_default(self):
        return HeadViewController(scene=self.scene, system='RAS')

    def __init__(self, raw=None, subject=None, subjects_dir=None):
        super(CoregFrame, self).__init__()

        # process parameters
        subjects_dir = get_subjects_dir(subjects_dir)
        if (subjects_dir is not None) and not os.path.isdir(subjects_dir):
            subjects_dir = None

        # sync mri subject
        self.s_sel.on_trait_change(self._on_bem_file_change, 'bem_file')
        self.s_sel.sync_trait('subjects_dir', self.coreg, mutual=False)
        self.s_sel.sync_trait('subject', self.coreg, mutual=False)
        self.s_sel.sync_trait('subjects_dir', self.fid_panel, mutual=False)
        self.s_sel.sync_trait('subject', self.fid_panel, mutual=False)

        # sync data to coreg panel
        self.mri_src.sync_trait('pts', self.coreg, 'mri_pts', mutual=False)
        self.hsp_src.sync_trait('pts', self.coreg, 'hsp_pts', mutual=False)
        self.hsp_src.sync_trait('fid', self.coreg, 'hsp_fid', mutual=False)
        self.hsp_src.sync_trait('fid_dig', self.coreg, 'dig', mutual=False)

        # sync ficudials panel
        self.sync_trait('lock_fiducials', self.fid_panel, 'locked',
                        mutual=True)

        # sync path source to coreg panel
        self.hsp_src.sync_trait('raw_dir', self.coreg, mutual=False)

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
            else:
                msg = ("No MRI subject named %r; ignoring subject "
                       "argument." % subject)
                error(msg, "Subject Not Found")

        # lock fiducials if file is found
        if self.fid_panel.fid_file:
            self.lock_fiducials = True

    @on_trait_change('fid_panel.fid_ok', post_init=True)
    def _on_fid_ok_change(self, new):
        # simply using 'fid_panel.fid_ok' resulted in delayed updates
        self.fid_ok = new

    @on_trait_change('scene.activated')
    def init_plot(self):
        self.scene.disable_render = True

        # MRI scalp
        self.mri_obj = SurfaceObject(points=self.mri_src.pts,
                                     tri=self.mri_src.tri,
                                     scene=self.scene, color=(252, 227, 191))
        self.coreg.sync_trait('scale', self.mri_obj, 'trans', mutual=False)
        self.fid_panel.hsp_obj = self.mri_obj

        # MRI Fiducials
        self.lap_obj = PointObject(scene=self.scene, color=(255, 0, 0),
                                   point_scale=1e-2)
        self.fid_panel.sync_trait('lap', self.lap_obj, 'points', mutual=False)
        self.coreg.sync_trait('scale', self.lap_obj, 'trans', mutual=False)

        self.nas_obj = PointObject(scene=self.scene, color=(0, 255, 0),
                                   point_scale=1e-2)
        self.fid_panel.sync_trait('nasion', self.nas_obj, 'points',
                                  mutual=False)
        self.coreg.sync_trait('scale', self.nas_obj, 'trans', mutual=False)

        self.rap_obj = PointObject(scene=self.scene, color=(0, 0, 255),
                                   point_scale=1e-2)
        self.fid_panel.sync_trait('rap', self.rap_obj, 'points', mutual=False)
        self.coreg.sync_trait('scale', self.rap_obj, 'trans', mutual=False)

        # Digitizer Head Shape
        self.hsp_obj = PointObject(view='cloud', scene=self.scene,
                                   color=(255, 255, 255), point_scale=2e-3,
                                   resolution=5)
        self.hsp_src.sync_trait('pts', self.hsp_obj, 'points', mutual=False)
        self.coreg.sync_trait('head_mri_trans', self.hsp_obj, 'trans',
                              mutual=False)

        # Digitizer Fiducials
        self.hsp_fid_obj = PointObject(scene=self.scene, color=(0, 0, 255),
                                       opacity=0.3, point_scale=3e-2)
        self.hsp_src.sync_trait('fid', self.hsp_fid_obj, 'points',
                                mutual=False)
        self.coreg.sync_trait('head_mri_trans', self.hsp_fid_obj, 'trans',
                              mutual=False)

        self.headview.left = True
        self.scene.disable_render = False

    @on_trait_change('lock_fiducials')
    def _on_lock_fiducials(self, lock):
        if (not self.hsp_obj) or (not self.hsp_fid_obj):
            return

        if lock:
            pass  # removing the picker here leads to exception
        else:
            on_pick = self.scene.mayavi_scene.on_mouse_pick
            self.picker = on_pick(self.fid_panel._on_pick, type='cell')

        if lock:
            self.hsp_obj.visible = True
            self.hsp_fid_obj.visible = True
            fid = np.vstack((self.fid_panel.nasion,
                             self.fid_panel.lap,
                             self.fid_panel.rap))
            self.coreg.mri_fid = fid
            self._update_fit_eval_fid()
        else:
            self.hsp_obj.visible = False
            self.hsp_fid_obj.visible = False

    def _on_bem_file_change(self):
        bem_file = self.s_sel.bem_file
        if not bem_file:
            self.mri_src.reset_traits(['file'])
            self.fid_panel.reset_traits(['fid_file'])
            return
        self.mri_src.file = bem_file % 'head'
        fid_file = bem_file % 'fiducials'
        self.lock_fiducials = False
        if os.path.exists(fid_file):
            self.fid_panel.fid_file = fid_file
            self.lock_fiducials = True
        else:
            self.fid_panel.reset_traits(('fid_file',))

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
