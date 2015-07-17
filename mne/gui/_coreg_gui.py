"""Traits-based GUI for head-MRI coregistration"""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os
from ..externals.six.moves import queue
import re
from threading import Thread
import warnings

import numpy as np
from scipy.spatial.distance import cdist

# allow import without traits
try:
    from mayavi.core.ui.mayavi_scene import MayaviScene
    from mayavi.tools.mlab_scene_model import MlabSceneModel
    from pyface.api import (error, confirm, warning, OK, YES, information,
                            FileDialog, GUI)
    from traits.api import (Bool, Button, cached_property, DelegatesTo,
                            Directory, Enum, Float, HasTraits,
                            HasPrivateTraits, Instance, Int, on_trait_change,
                            Property, Str)
    from traitsui.api import (View, Item, Group, HGroup, VGroup, VGrid,
                              EnumEditor, Handler, Label, TextEditor)
    from traitsui.menu import Action, UndoButton, CancelButton, NoButtons
    from tvtk.pyface.scene_editor import SceneEditor
except:
    from ..utils import trait_wraith
    HasTraits = HasPrivateTraits = Handler = object
    cached_property = on_trait_change = MayaviScene = MlabSceneModel =\
        Bool = Button = DelegatesTo = Directory = Enum = Float = Instance =\
        Int = Property = Str = View = Item = Group = HGroup = VGroup = VGrid =\
        EnumEditor = Label = TextEditor = Action = UndoButton = CancelButton =\
        NoButtons = SceneEditor = trait_wraith


from ..coreg import bem_fname, trans_fname
from ..io.constants import FIFF
from ..forward import prepare_bem_model
from ..transforms import (write_trans, read_trans, apply_trans, rotation,
                          translation, scaling, rotation_angles)
from ..coreg import (fit_matched_points, fit_point_cloud, scale_mri,
                     _point_cloud_error)
from ..utils import get_subjects_dir, logger
from ._fiducials_gui import MRIHeadWithFiducialsModel, FiducialsPanel
from ._file_traits import (set_mne_root, trans_wildcard, InstSource,
                           SubjectSelectorPanel)
from ._viewer import (defaults, HeadViewController, PointObject, SurfaceObject,
                      _testing_mode)


laggy_float_editor = TextEditor(auto_set=False, enter_set=True, evaluate=float)


class CoregModel(HasPrivateTraits):
    """Traits object for estimating the head mri transform.

    Notes
    -----
    Transform from head to mri space is modelled with the following steps:

     * move the head shape to its nasion position
     * rotate the head shape with user defined rotation around its nasion
     * move the head shape by user defined translation
     * move the head shape origin to the mri nasion

    If MRI scaling is enabled,

     * the MRI is scaled relative to its origin center (prior to any
       transformation of the digitizer head)


    Don't sync transforms to anything to prevent them from being recomputed
    upon every parameter change.
    """
    # data sources
    mri = Instance(MRIHeadWithFiducialsModel, ())
    hsp = Instance(InstSource, ())

    # parameters
    grow_hair = Float(label="Grow Hair [mm]", desc="Move the back of the MRI "
                      "head outwards to compensate for hair on the digitizer "
                      "head shape")
    n_scale_params = Enum(0, 1, 3, desc="Scale the MRI to better fit the "
                          "subject's head shape (a new MRI subject will be "
                          "created with a name specified upon saving)")
    scale_x = Float(1, label="Right (X)")
    scale_y = Float(1, label="Anterior (Y)")
    scale_z = Float(1, label="Superior (Z)")
    rot_x = Float(0, label="Right (X)")
    rot_y = Float(0, label="Anterior (Y)")
    rot_z = Float(0, label="Superior (Z)")
    trans_x = Float(0, label="Right (X)")
    trans_y = Float(0, label="Anterior (Y)")
    trans_z = Float(0, label="Superior (Z)")

    prepare_bem_model = Bool(True, desc="whether to run mne_prepare_bem_model "
                             "after scaling the MRI")

    # secondary to parameters
    scale = Property(depends_on=['n_scale_params', 'scale_x', 'scale_y',
                                 'scale_z'])
    has_fid_data = Property(Bool, depends_on=['mri_origin', 'hsp.nasion'],
                            desc="Required fiducials data is present.")
    has_pts_data = Property(Bool, depends_on=['mri.points', 'hsp.points'])

    # MRI dependent
    mri_origin = Property(depends_on=['mri.nasion', 'scale'],
                          desc="Coordinates of the scaled MRI's nasion.")

    # target transforms
    mri_scale_trans = Property(depends_on=['scale'])
    head_mri_trans = Property(depends_on=['hsp.nasion', 'rot_x', 'rot_y',
                                          'rot_z', 'trans_x', 'trans_y',
                                          'trans_z', 'mri_origin'],
                              desc="Transformaiton of the head shape to "
                              "match the scaled MRI.")

    # info
    subject_has_bem = DelegatesTo('mri')
    lock_fiducials = DelegatesTo('mri')
    can_prepare_bem_model = Property(Bool, depends_on=['n_scale_params',
                                                       'subject_has_bem'])
    can_save = Property(Bool, depends_on=['head_mri_trans'])
    raw_subject = Property(depends_on='hsp.inst_fname', desc="Subject guess "
                           "based on the raw file name.")

    # transformed geometry
    processed_mri_points = Property(depends_on=['mri.points', 'grow_hair'])
    transformed_mri_points = Property(depends_on=['processed_mri_points',
                                                  'mri_scale_trans'])
    transformed_hsp_points = Property(depends_on=['hsp.points',
                                                  'head_mri_trans'])
    transformed_mri_lpa = Property(depends_on=['mri.lpa', 'mri_scale_trans'])
    transformed_hsp_lpa = Property(depends_on=['hsp.lpa', 'head_mri_trans'])
    transformed_mri_nasion = Property(depends_on=['mri.nasion',
                                                  'mri_scale_trans'])
    transformed_hsp_nasion = Property(depends_on=['hsp.nasion',
                                                  'head_mri_trans'])
    transformed_mri_rpa = Property(depends_on=['mri.rpa', 'mri_scale_trans'])
    transformed_hsp_rpa = Property(depends_on=['hsp.rpa', 'head_mri_trans'])

    # fit properties
    lpa_distance = Property(depends_on=['transformed_mri_lpa',
                                        'transformed_hsp_lpa'])
    nasion_distance = Property(depends_on=['transformed_mri_nasion',
                                           'transformed_hsp_nasion'])
    rpa_distance = Property(depends_on=['transformed_mri_rpa',
                                        'transformed_hsp_rpa'])
    point_distance = Property(depends_on=['transformed_mri_points',
                                          'transformed_hsp_points'])

    # fit property info strings
    fid_eval_str = Property(depends_on=['lpa_distance', 'nasion_distance',
                                        'rpa_distance'])
    points_eval_str = Property(depends_on='point_distance')

    @cached_property
    def _get_can_prepare_bem_model(self):
        return self.subject_has_bem and self.n_scale_params > 0

    @cached_property
    def _get_can_save(self):
        return np.any(self.head_mri_trans != np.eye(4))

    @cached_property
    def _get_has_pts_data(self):
        has = (np.any(self.mri.points) and np.any(self.hsp.points))
        return has

    @cached_property
    def _get_has_fid_data(self):
        has = (np.any(self.mri_origin) and np.any(self.hsp.nasion))
        return has

    @cached_property
    def _get_scale(self):
        if self.n_scale_params == 0:
            return np.array(1)
        elif self.n_scale_params == 1:
            return np.array(self.scale_x)
        else:
            return np.array([self.scale_x, self.scale_y, self.scale_z])

    @cached_property
    def _get_mri_scale_trans(self):
        if np.isscalar(self.scale) or self.scale.ndim == 0:
            if self.scale == 1:
                return np.eye(4)
            else:
                s = self.scale
                return scaling(s, s, s)
        else:
            return scaling(*self.scale)

    @cached_property
    def _get_mri_origin(self):
        if np.isscalar(self.scale) and self.scale == 1:
            return self.mri.nasion
        else:
            return self.mri.nasion * self.scale

    @cached_property
    def _get_head_mri_trans(self):
        if not self.has_fid_data:
            return np.eye(4)

        # move hsp so that its nasion becomes the origin
        x, y, z = -self.hsp.nasion[0]
        trans = translation(x, y, z)

        # rotate hsp by rotation parameters
        rot = rotation(self.rot_x, self.rot_y, self.rot_z)
        trans = np.dot(rot, trans)

        # move hsp by translation parameters
        transl = translation(self.trans_x, self.trans_y, self.trans_z)
        trans = np.dot(transl, trans)

        # move the hsp origin(/nasion) to the MRI's nasion
        x, y, z = self.mri_origin[0]
        tgt_mri_trans = translation(x, y, z)
        trans = np.dot(tgt_mri_trans, trans)

        return trans

    @cached_property
    def _get_processed_mri_points(self):
        if self.grow_hair:
            if len(self.mri.norms):
                if self.n_scale_params == 0:
                    scaled_hair_dist = self.grow_hair / 1000
                else:
                    scaled_hair_dist = self.grow_hair / self.scale / 1000

                points = self.mri.points.copy()
                hair = points[:, 2] > points[:, 1]
                points[hair] += self.mri.norms[hair] * scaled_hair_dist
                return points
            else:
                error(None, "Norms missing form bem, can't grow hair")
                self.grow_hair = 0
        return self.mri.points

    @cached_property
    def _get_transformed_mri_points(self):
        points = apply_trans(self.mri_scale_trans, self.processed_mri_points)
        return points

    @cached_property
    def _get_transformed_mri_lpa(self):
        return apply_trans(self.mri_scale_trans, self.mri.lpa)

    @cached_property
    def _get_transformed_mri_nasion(self):
        return apply_trans(self.mri_scale_trans, self.mri.nasion)

    @cached_property
    def _get_transformed_mri_rpa(self):
        return apply_trans(self.mri_scale_trans, self.mri.rpa)

    @cached_property
    def _get_transformed_hsp_points(self):
        return apply_trans(self.head_mri_trans, self.hsp.points)

    @cached_property
    def _get_transformed_hsp_lpa(self):
        return apply_trans(self.head_mri_trans, self.hsp.lpa)

    @cached_property
    def _get_transformed_hsp_nasion(self):
        return apply_trans(self.head_mri_trans, self.hsp.nasion)

    @cached_property
    def _get_transformed_hsp_rpa(self):
        return apply_trans(self.head_mri_trans, self.hsp.rpa)

    @cached_property
    def _get_lpa_distance(self):
        d = np.ravel(self.transformed_mri_lpa - self.transformed_hsp_lpa)
        return np.sqrt(np.dot(d, d))

    @cached_property
    def _get_nasion_distance(self):
        d = np.ravel(self.transformed_mri_nasion - self.transformed_hsp_nasion)
        return np.sqrt(np.dot(d, d))

    @cached_property
    def _get_rpa_distance(self):
        d = np.ravel(self.transformed_mri_rpa - self.transformed_hsp_rpa)
        return np.sqrt(np.dot(d, d))

    @cached_property
    def _get_point_distance(self):
        if (len(self.transformed_hsp_points) == 0 or
                len(self.transformed_mri_points) == 0):
            return
        dists = cdist(self.transformed_hsp_points, self.transformed_mri_points,
                      'euclidean')
        dists = np.min(dists, 1)
        return dists

    @cached_property
    def _get_fid_eval_str(self):
        d = (self.lpa_distance * 1000, self.nasion_distance * 1000,
             self.rpa_distance * 1000)
        txt = ("Fiducials Error: LPA %.1f mm, NAS %.1f mm, RPA %.1f mm" % d)
        return txt

    @cached_property
    def _get_points_eval_str(self):
        if self.point_distance is None:
            return ""
        av_dist = np.mean(self.point_distance)
        return "Average Points Error: %.1f mm" % (av_dist * 1000)

    def _get_raw_subject(self):
        # subject name guessed based on the inst file name
        if '_' in self.hsp.inst_fname:
            subject, _ = self.hsp.inst_fname.split('_', 1)
            if not subject:
                subject = None
        else:
            subject = None
        return subject

    @on_trait_change('raw_subject')
    def _on_raw_subject_change(self, subject):
        if subject in self.mri.subject_source.subjects:
            self.mri.subject = subject
        elif 'fsaverage' in self.mri.subject_source.subjects:
            self.mri.subject = 'fsaverage'

    def omit_hsp_points(self, distance=0, reset=False):
        """Exclude head shape points that are far away from the MRI head

        Parameters
        ----------
        distance : float
            Exclude all points that are further away from the MRI head than
            this distance. Previously excluded points are still excluded unless
            reset=True is specified. A value of distance <= 0 excludes nothing.
        reset : bool
            Reset the filter before calculating new omission (default is
            False).
        """
        distance = float(distance)
        if reset:
            logger.info("Coregistration: Reset excluded head shape points")
            with warnings.catch_warnings(record=True):  # Traits None comp
                self.hsp.points_filter = None

        if distance <= 0:
            return

        # find the new filter
        hsp_pts = self.transformed_hsp_points
        mri_pts = self.transformed_mri_points
        point_distance = _point_cloud_error(hsp_pts, mri_pts)
        new_sub_filter = point_distance <= distance
        n_excluded = np.sum(new_sub_filter == False)  # noqa
        logger.info("Coregistration: Excluding %i head shape points with "
                    "distance >= %.3f m.", n_excluded, distance)

        # combine the new filter with the previous filter
        old_filter = self.hsp.points_filter
        if old_filter is None:
            new_filter = new_sub_filter
        else:
            new_filter = np.ones(len(self.hsp.raw_points), np.bool8)
            new_filter[old_filter] = new_sub_filter

        # set the filter
        with warnings.catch_warnings(record=True):  # comp to None in Traits
            self.hsp.points_filter = new_filter

    def fit_auricular_points(self):
        "Find rotation to fit LPA and RPA"
        src_fid = np.vstack((self.hsp.lpa, self.hsp.rpa))
        src_fid -= self.hsp.nasion

        tgt_fid = np.vstack((self.mri.lpa, self.mri.rpa))
        tgt_fid -= self.mri.nasion
        tgt_fid *= self.scale
        tgt_fid -= [self.trans_x, self.trans_y, self.trans_z]

        x0 = (self.rot_x, self.rot_y, self.rot_z)
        rot = fit_matched_points(src_fid, tgt_fid, rotate=True,
                                 translate=False, x0=x0, out='params')

        self.rot_x, self.rot_y, self.rot_z = rot

    def fit_fiducials(self):
        "Find rotation and translation to fit all 3 fiducials"
        src_fid = np.vstack((self.hsp.lpa, self.hsp.nasion, self.hsp.rpa))
        src_fid -= self.hsp.nasion

        tgt_fid = np.vstack((self.mri.lpa, self.mri.nasion, self.mri.rpa))
        tgt_fid -= self.mri.nasion
        tgt_fid *= self.scale

        x0 = (self.rot_x, self.rot_y, self.rot_z, self.trans_x, self.trans_y,
              self.trans_z)
        est = fit_matched_points(src_fid, tgt_fid, x0=x0, out='params')

        self.rot_x, self.rot_y, self.rot_z = est[:3]
        self.trans_x, self.trans_y, self.trans_z = est[3:]

    def fit_hsp_points(self):
        "Find rotation to fit head shapes"
        src_pts = self.hsp.points - self.hsp.nasion

        tgt_pts = self.processed_mri_points - self.mri.nasion
        tgt_pts *= self.scale
        tgt_pts -= [self.trans_x, self.trans_y, self.trans_z]

        x0 = (self.rot_x, self.rot_y, self.rot_z)
        rot = fit_point_cloud(src_pts, tgt_pts, rotate=True, translate=False,
                              x0=x0)

        self.rot_x, self.rot_y, self.rot_z = rot

    def fit_scale_auricular_points(self):
        "Find rotation and MRI scaling based on LPA and RPA"
        src_fid = np.vstack((self.hsp.lpa, self.hsp.rpa))
        src_fid -= self.hsp.nasion

        tgt_fid = np.vstack((self.mri.lpa, self.mri.rpa))
        tgt_fid -= self.mri.nasion
        tgt_fid -= [self.trans_x, self.trans_y, self.trans_z]

        x0 = (self.rot_x, self.rot_y, self.rot_z, 1. / self.scale_x)
        x = fit_matched_points(src_fid, tgt_fid, rotate=True, translate=False,
                               scale=1, x0=x0, out='params')

        self.scale_x = 1. / x[3]
        self.rot_x, self.rot_y, self.rot_z = x[:3]

    def fit_scale_fiducials(self):
        "Find translation, rotation and scaling based on the three fiducials"
        src_fid = np.vstack((self.hsp.lpa, self.hsp.nasion, self.hsp.rpa))
        src_fid -= self.hsp.nasion

        tgt_fid = np.vstack((self.mri.lpa, self.mri.nasion, self.mri.rpa))
        tgt_fid -= self.mri.nasion

        x0 = (self.rot_x, self.rot_y, self.rot_z, self.trans_x, self.trans_y,
              self.trans_z, 1. / self.scale_x,)
        est = fit_matched_points(src_fid, tgt_fid, rotate=True, translate=True,
                                 scale=1, x0=x0, out='params')

        self.scale_x = 1. / est[6]
        self.rot_x, self.rot_y, self.rot_z = est[:3]
        self.trans_x, self.trans_y, self.trans_z = est[3:6]

    def fit_scale_hsp_points(self):
        "Find MRI scaling and rotation to match head shape points"
        src_pts = self.hsp.points - self.hsp.nasion

        tgt_pts = self.processed_mri_points - self.mri.nasion

        if self.n_scale_params == 1:
            x0 = (self.rot_x, self.rot_y, self.rot_z, 1. / self.scale_x)
            est = fit_point_cloud(src_pts, tgt_pts, rotate=True,
                                  translate=False, scale=1, x0=x0)

            self.scale_x = 1. / est[3]
        else:
            x0 = (self.rot_x, self.rot_y, self.rot_z, 1. / self.scale_x,
                  1. / self.scale_y, 1. / self.scale_z)
            est = fit_point_cloud(src_pts, tgt_pts, rotate=True,
                                  translate=False, scale=3, x0=x0)
            self.scale_x, self.scale_y, self.scale_z = 1. / est[3:]

        self.rot_x, self.rot_y, self.rot_z = est[:3]

    def get_scaling_job(self, subject_to):
        desc = 'Scaling %s' % subject_to
        func = scale_mri
        args = (self.mri.subject, subject_to, self.scale)
        kwargs = dict(overwrite=True, subjects_dir=self.mri.subjects_dir)
        return (desc, func, args, kwargs)

    def get_prepare_bem_model_job(self, subject_to):
        subjects_dir = self.mri.subjects_dir
        subject_from = self.mri.subject

        bem_name = 'inner_skull-bem'
        bem_file = bem_fname.format(subjects_dir=subjects_dir,
                                    subject=subject_from, name=bem_name)
        if not os.path.exists(bem_file):
            pattern = bem_fname.format(subjects_dir=subjects_dir,
                                       subject=subject_to, name='(.+-bem)')
            bem_dir, bem_file = os.path.split(pattern)
            m = None
            bem_file_pattern = re.compile(bem_file)
            for name in os.listdir(bem_dir):
                m = bem_file_pattern.match(name)
                if m is not None:
                    break

            if m is None:
                pattern = bem_fname.format(subjects_dir=subjects_dir,
                                           subject=subject_to, name='*-bem')
                err = ("No bem file found; looking for files matching "
                       "%s" % pattern)
                error(None, err)

            bem_name = m.group(1)

        bem_file = bem_fname.format(subjects_dir=subjects_dir,
                                    subject=subject_to, name=bem_name)

        # job
        desc = 'mne_prepare_bem_model for %s' % subject_to
        func = prepare_bem_model
        args = (bem_file,)
        kwargs = {}
        return (desc, func, args, kwargs)

    def load_trans(self, fname):
        """Load the head-mri transform from a fif file

        Parameters
        ----------
        fname : str
            File path.
        """
        info = read_trans(fname)
        head_mri_trans = info['trans']
        self.set_trans(head_mri_trans)

    def reset(self):
        """Reset all the parameters affecting the coregistration"""
        self.reset_traits(('grow_hair', 'n_scaling_params', 'scale_x',
                           'scale_y', 'scale_z', 'rot_x', 'rot_y', 'rot_z',
                           'trans_x', 'trans_y', 'trans_z'))

    def set_trans(self, head_mri_trans):
        """Set rotation and translation parameters from a transformation matrix

        Parameters
        ----------
        head_mri_trans : array, shape (4, 4)
            Transformation matrix from head to MRI space.
        """
        x, y, z = -self.mri_origin[0]
        mri_tgt_trans = translation(x, y, z)
        head_tgt_trans = np.dot(mri_tgt_trans, head_mri_trans)

        x, y, z = self.hsp.nasion[0]
        src_hsp_trans = translation(x, y, z)
        src_tgt_trans = np.dot(head_tgt_trans, src_hsp_trans)

        rot_x, rot_y, rot_z = rotation_angles(src_tgt_trans[:3, :3])
        x, y, z = src_tgt_trans[:3, 3]

        self.rot_x = rot_x
        self.rot_y = rot_y
        self.rot_z = rot_z
        self.trans_x = x
        self.trans_y = y
        self.trans_z = z

    def save_trans(self, fname):
        """Save the head-mri transform as a fif file

        Parameters
        ----------
        fname : str
            Target file path.
        """
        if not self.can_save:
            raise RuntimeError("Not enough information for saving transform")
        trans_matrix = self.head_mri_trans
        trans = {'to': FIFF.FIFFV_COORD_MRI, 'from': FIFF.FIFFV_COORD_HEAD,
                 'trans': trans_matrix}
        write_trans(fname, trans)


class CoregFrameHandler(Handler):
    """Handler that checks for unfinished processes before closing its window
    """
    def close(self, info, is_ok):
        if info.object.queue.unfinished_tasks:
            information(None, "Can not close the window while saving is still "
                        "in progress. Please wait until all MRIs are "
                        "processed.", "Saving Still in Progress")
            return False
        else:
            return True


class CoregPanel(HasPrivateTraits):
    model = Instance(CoregModel)

    # parameters
    reset_params = Button(label='Reset')
    grow_hair = DelegatesTo('model')
    n_scale_params = DelegatesTo('model')
    scale_step = Float(1.01)
    scale_x = DelegatesTo('model')
    scale_x_dec = Button('-')
    scale_x_inc = Button('+')
    scale_y = DelegatesTo('model')
    scale_y_dec = Button('-')
    scale_y_inc = Button('+')
    scale_z = DelegatesTo('model')
    scale_z_dec = Button('-')
    scale_z_inc = Button('+')
    rot_step = Float(0.01)
    rot_x = DelegatesTo('model')
    rot_x_dec = Button('-')
    rot_x_inc = Button('+')
    rot_y = DelegatesTo('model')
    rot_y_dec = Button('-')
    rot_y_inc = Button('+')
    rot_z = DelegatesTo('model')
    rot_z_dec = Button('-')
    rot_z_inc = Button('+')
    trans_step = Float(0.001)
    trans_x = DelegatesTo('model')
    trans_x_dec = Button('-')
    trans_x_inc = Button('+')
    trans_y = DelegatesTo('model')
    trans_y_dec = Button('-')
    trans_y_inc = Button('+')
    trans_z = DelegatesTo('model')
    trans_z_dec = Button('-')
    trans_z_inc = Button('+')

    # fitting
    has_fid_data = DelegatesTo('model')
    has_pts_data = DelegatesTo('model')
    # fitting with scaling
    fits_hsp_points = Button(label='Fit Head Shape')
    fits_fid = Button(label='Fit Fiducials')
    fits_ap = Button(label='Fit LPA/RPA')
    # fitting without scaling
    fit_hsp_points = Button(label='Fit Head Shape')
    fit_fid = Button(label='Fit Fiducials')
    fit_ap = Button(label='Fit LPA/RPA')

    # fit info
    fid_eval_str = DelegatesTo('model')
    points_eval_str = DelegatesTo('model')

    # saving
    can_prepare_bem_model = DelegatesTo('model')
    can_save = DelegatesTo('model')
    prepare_bem_model = DelegatesTo('model')
    save = Button(label="Save As...")
    load_trans = Button
    queue = Instance(queue.Queue, ())
    queue_feedback = Str('')
    queue_current = Str('')
    queue_len = Int(0)
    queue_len_str = Property(Str, depends_on=['queue_len'])
    error = Str('')

    view = View(VGroup(Item('grow_hair', show_label=True),
                       Item('n_scale_params', label='MRI Scaling',
                            style='custom', show_label=True,
                            editor=EnumEditor(values={0: '1:No Scaling',
                                                      1: '2:1 Param',
                                                      3: '3:3 Params'},
                                              cols=3)),
                       VGrid(Item('scale_x', editor=laggy_float_editor,
                                  show_label=True, tooltip="Scale along "
                                  "right-left axis",
                                  enabled_when='n_scale_params > 0'),
                             Item('scale_x_dec',
                                  enabled_when='n_scale_params > 0'),
                             Item('scale_x_inc',
                                  enabled_when='n_scale_params > 0'),
                             Item('scale_step', tooltip="Scaling step",
                                  enabled_when='n_scale_params > 0'),
                             Item('scale_y', editor=laggy_float_editor,
                                  show_label=True,
                                  enabled_when='n_scale_params > 1',
                                  tooltip="Scale along anterior-posterior "
                                  "axis"),
                             Item('scale_y_dec',
                                  enabled_when='n_scale_params > 1'),
                             Item('scale_y_inc',
                                  enabled_when='n_scale_params > 1'),
                             Label('(Step)'),
                             Item('scale_z', editor=laggy_float_editor,
                                  show_label=True,
                                  enabled_when='n_scale_params > 1',
                                  tooltip="Scale along anterior-posterior "
                                  "axis"),
                             Item('scale_z_dec',
                                  enabled_when='n_scale_params > 1'),
                             Item('scale_z_inc',
                                  enabled_when='n_scale_params > 1'),
                             show_labels=False, columns=4),
                       HGroup(Item('fits_hsp_points',
                                   enabled_when='n_scale_params',
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
                       Label("Translation:"),
                       VGrid(Item('trans_x', editor=laggy_float_editor,
                                  show_label=True, tooltip="Move along "
                                  "right-left axis"),
                             'trans_x_dec', 'trans_x_inc',
                             Item('trans_step', tooltip="Movement step"),
                             Item('trans_y', editor=laggy_float_editor,
                                  show_label=True, tooltip="Move along "
                                  "anterior-posterior axis"),
                             'trans_y_dec', 'trans_y_inc',
                             Label('(Step)'),
                             Item('trans_z', editor=laggy_float_editor,
                                  show_label=True, tooltip="Move along "
                                  "anterior-posterior axis"),
                             'trans_z_dec', 'trans_z_inc',
                             show_labels=False, columns=4),
                       Label("Rotation:"),
                       VGrid(Item('rot_x', editor=laggy_float_editor,
                                  show_label=True, tooltip="Rotate along "
                                  "right-left axis"),
                             'rot_x_dec', 'rot_x_inc',
                             Item('rot_step', tooltip="Rotation step"),
                             Item('rot_y', editor=laggy_float_editor,
                                  show_label=True, tooltip="Rotate along "
                                  "anterior-posterior axis"),
                             'rot_y_dec', 'rot_y_inc',
                             Label('(Step)'),
                             Item('rot_z', editor=laggy_float_editor,
                                  show_label=True, tooltip="Rotate along "
                                  "anterior-posterior axis"),
                             'rot_z_dec', 'rot_z_inc',
                             show_labels=False, columns=4),
                       # buttons
                       HGroup(Item('fit_hsp_points',
                                   enabled_when='has_pts_data',
                                   tooltip="Rotate the head shape (around the "
                                   "nasion) so as to minimize the distance "
                                   "from each head shape point to its closest "
                                   "MRI point"),
                              Item('fit_ap', enabled_when='has_fid_data',
                                   tooltip="Try to match the LPA and the RPA, "
                                   "leaving the Nasion in place"),
                              Item('fit_fid', enabled_when='has_fid_data',
                                   tooltip="Move and rotate the head shape so "
                                   "as to minimize the distance between the "
                                   "MRI and head shape fiducials"),
                              Item('load_trans', enabled_when='has_fid_data'),
                              show_labels=False),
                       '_',
                       Item('fid_eval_str', style='readonly'),
                       Item('points_eval_str', style='readonly'),
                       '_',
                       HGroup(Item('prepare_bem_model'),
                              Label("Run mne_prepare_bem_model"),
                              show_labels=False,
                              enabled_when='can_prepare_bem_model'),
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
    def _get_queue_len_str(self):
        if self.queue_len:
            return "Queue length: %i" % self.queue_len
        else:
            return ''

    @cached_property
    def _get_rotation(self):
        rot = np.array([self.rot_x, self.rot_y, self.rot_z])
        return rot

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

    @cached_property
    def _get_translation(self):
        trans = np.array([self.trans_x, self.trans_y, self.trans_z])
        return trans

    def _fit_ap_fired(self):
        GUI.set_busy()
        self.model.fit_auricular_points()
        GUI.set_busy(False)

    def _fit_fid_fired(self):
        GUI.set_busy()
        self.model.fit_fiducials()
        GUI.set_busy(False)

    def _fit_hsp_points_fired(self):
        GUI.set_busy()
        self.model.fit_hsp_points()
        GUI.set_busy(False)

    def _fits_ap_fired(self):
        GUI.set_busy()
        self.model.fit_scale_auricular_points()
        GUI.set_busy(False)

    def _fits_fid_fired(self):
        GUI.set_busy()
        self.model.fit_scale_fiducials()
        GUI.set_busy(False)

    def _fits_hsp_points_fired(self):
        GUI.set_busy()
        self.model.fit_scale_hsp_points()
        GUI.set_busy(False)

    def _n_scale_params_changed(self, new):
        if not new:
            return

        # Make sure that MNE_ROOT environment variable is set
        if not set_mne_root(True):
            err = ("MNE_ROOT environment variable could not be set. "
                   "You will be able to scale MRIs, but the "
                   "mne_prepare_bem_model tool will fail. Please install "
                   "MNE.")
            warning(None, err, "MNE_ROOT Not Set")

    def _reset_params_fired(self):
        self.model.reset()

    def _rot_x_dec_fired(self):
        self.rot_x -= self.rot_step

    def _rot_x_inc_fired(self):
        self.rot_x += self.rot_step

    def _rot_y_dec_fired(self):
        self.rot_y -= self.rot_step

    def _rot_y_inc_fired(self):
        self.rot_y += self.rot_step

    def _rot_z_dec_fired(self):
        self.rot_z -= self.rot_step

    def _rot_z_inc_fired(self):
        self.rot_z += self.rot_step

    def _load_trans_fired(self):
        # find trans file destination
        raw_dir = os.path.dirname(self.model.hsp.file)
        subject = self.model.mri.subject
        trans_file = trans_fname.format(raw_dir=raw_dir, subject=subject)
        dlg = FileDialog(action="open", wildcard=trans_wildcard,
                         default_path=trans_file)
        dlg.open()
        if dlg.return_code != OK:
            return
        trans_file = dlg.path
        self.model.load_trans(trans_file)

    def _save_fired(self):
        if self.n_scale_params:
            subjects_dir = self.model.mri.subjects_dir
            subject_from = self.model.mri.subject
            subject_to = self.model.raw_subject or self.model.mri.subject
        else:
            subject_to = self.model.mri.subject

        # ask for target subject
        if self.n_scale_params:
            mridlg = NewMriDialog(subjects_dir=subjects_dir,
                                  subject_from=subject_from,
                                  subject_to=subject_to)
            ui = mridlg.edit_traits(kind='modal')
            if ui.result != True:  # noqa
                return
            subject_to = mridlg.subject_to

        # find bem file to run mne_prepare_bem_model
        if self.can_prepare_bem_model and self.prepare_bem_model:
            bem_job = self.model.get_prepare_bem_model_job(subject_to)
        else:
            bem_job = None

        # find trans file destination
        raw_dir = os.path.dirname(self.model.hsp.file)
        trans_file = trans_fname.format(raw_dir=raw_dir, subject=subject_to)
        dlg = FileDialog(action="save as", wildcard=trans_wildcard,
                         default_path=trans_file)
        dlg.open()
        if dlg.return_code != OK:
            return
        trans_file = dlg.path
        if not trans_file.endswith('.fif'):
            trans_file = trans_file + '.fif'
            if os.path.exists(trans_file):
                answer = confirm(None, "The file %r already exists. Should it "
                                 "be replaced?", "Overwrite File?")
                if answer != YES:
                    return

        # save the trans file
        try:
            self.model.save_trans(trans_file)
        except Exception as e:
            error(None, str(e), "Error Saving Trans File")
            return

        # save the scaled MRI
        if self.n_scale_params:
            job = self.model.get_scaling_job(subject_to)
            self.queue.put(job)
            self.queue_len += 1

            if bem_job is not None:
                self.queue.put(bem_job)
                self.queue_len += 1

    def _scale_x_dec_fired(self):
        step = 1. / self.scale_step
        self.scale_x *= step

    def _scale_x_inc_fired(self):
        self.scale_x *= self.scale_step

    def _scale_x_changed(self, old, new):
        if self.n_scale_params == 1:
            self.scale_y = new
            self.scale_z = new

    def _scale_y_dec_fired(self):
        step = 1. / self.scale_step
        self.scale_y *= step

    def _scale_y_inc_fired(self):
        self.scale_y *= self.scale_step

    def _scale_z_dec_fired(self):
        step = 1. / self.scale_step
        self.scale_z *= step

    def _scale_z_inc_fired(self):
        self.scale_z *= self.scale_step

    def _trans_x_dec_fired(self):
        self.trans_x -= self.trans_step

    def _trans_x_inc_fired(self):
        self.trans_x += self.trans_step

    def _trans_y_dec_fired(self):
        self.trans_y -= self.trans_step

    def _trans_y_inc_fired(self):
        self.trans_y += self.trans_step

    def _trans_z_dec_fired(self):
        self.trans_z -= self.trans_step

    def _trans_z_inc_fired(self):
        self.trans_z += self.trans_step


class NewMriDialog(HasPrivateTraits):
    # Dialog to determine target subject name for a scaled MRI
    subjects_dir = Directory
    subject_to = Str
    subject_from = Str
    subject_to_dir = Property(depends_on=['subjects_dir', 'subject_to'])
    subject_to_exists = Property(Bool, depends_on='subject_to_dir')

    feedback = Str(' ' * 100)
    can_overwrite = Bool
    overwrite = Bool
    can_save = Bool

    view = View(Item('subject_to', label='New MRI Subject Name', tooltip="A "
                     "new folder with this name will be created in the "
                     "current subjects_dir for the scaled MRI files"),
                Item('feedback', show_label=False, style='readonly'),
                Item('overwrite', enabled_when='can_overwrite', tooltip="If a "
                     "subject with the chosen name exists, delete the old "
                     "subject"),
                width=500,
                buttons=[CancelButton,
                         Action(name='OK', enabled_when='can_save')])

    def _can_overwrite_changed(self, new):
        if not new:
            self.overwrite = False

    @cached_property
    def _get_subject_to_dir(self):
        return os.path.join(self.subjects_dir, self.subject_to)

    @cached_property
    def _get_subject_to_exists(self):
        if not self.subject_to:
            return False
        elif os.path.exists(self.subject_to_dir):
            return True
        else:
            return False

    @on_trait_change('subject_to_dir,overwrite')
    def update_dialog(self):
        if not self.subject_to:
            self.feedback = "No subject specified..."
            self.can_save = False
            self.can_overwrite = False
        elif self.subject_to == self.subject_from:
            self.feedback = "Must be different from MRI source subject..."
            self.can_save = False
            self.can_overwrite = False
        elif self.subject_to_exists:
            if self.overwrite:
                self.feedback = "%s will be overwritten." % self.subject_to
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


def _make_view(tabbed=False, split=False, scene_width=-1):
    """Create a view for the CoregFrame

    Parameters
    ----------
    tabbed : bool
        Combine the data source panel and the coregistration panel into a
        single panel with tabs.
    split : bool
        Split the main panels with a movable splitter (good for QT4 but
        unnecessary for wx backend).
    scene_width : int
        Specify a minimum width for the 3d scene (in pixels).

    returns
    -------
    view : traits View
        View object for the CoregFrame.
    """
    view_options = VGroup(Item('headview', style='custom'), 'view_options',
                          show_border=True, show_labels=False, label='View')

    scene = VGroup(Item('scene', show_label=False,
                        editor=SceneEditor(scene_class=MayaviScene),
                        dock='vertical', width=500),
                   view_options)

    data_panel = VGroup(VGroup(Item('subject_panel', style='custom'),
                               label="MRI Subject", show_border=True,
                               show_labels=False),
                        VGroup(Item('lock_fiducials', style='custom',
                                    editor=EnumEditor(cols=2,
                                                      values={False: '2:Edit',
                                                              True: '1:Lock'}),
                                    enabled_when='fid_ok'),
                               HGroup('hsp_always_visible',
                                      Label("Always Show Head Shape Points"),
                                      show_labels=False),
                               Item('fid_panel', style='custom'),
                               label="MRI Fiducials", show_border=True,
                               show_labels=False),
                        VGroup(Item('raw_src', style="custom"),
                               HGroup(Item('distance', show_label=True),
                                      'omit_points', 'reset_omit_points',
                                      show_labels=False),
                               Item('omitted_info', style='readonly',
                                    show_label=False),
                               label='Head Shape Source (Raw/Epochs/Evoked)',
                               show_border=True, show_labels=False),
                        show_labels=False, label="Data Source")

    coreg_panel = VGroup(Item('coreg_panel', style='custom'),
                         label="Coregistration", show_border=True,
                         show_labels=False,
                         enabled_when="fid_panel.locked")

    if split:
        main_layout = 'split'
    else:
        main_layout = 'normal'

    if tabbed:
        main = HGroup(scene,
                      Group(data_panel, coreg_panel, show_labels=False,
                            layout='tabbed'),
                      layout=main_layout)
    else:
        main = HGroup(data_panel, scene, coreg_panel, show_labels=False,
                      layout=main_layout)

    view = View(main, resizable=True, handler=CoregFrameHandler(),
                buttons=NoButtons)
    return view


class ViewOptionsPanel(HasTraits):
    mri_obj = Instance(SurfaceObject)
    hsp_obj = Instance(PointObject)
    view = View(VGroup(Item('mri_obj', style='custom',  # show_border=True,
                            label="MRI Head Surface"),
                       Item('hsp_obj', style='custom',  # show_border=True,
                            label="Head Shape Points")),
                title="View Options")


class CoregFrame(HasTraits):
    """GUI for head-MRI coregistration
    """
    model = Instance(CoregModel, ())

    scene = Instance(MlabSceneModel, ())
    headview = Instance(HeadViewController)

    subject_panel = Instance(SubjectSelectorPanel)
    fid_panel = Instance(FiducialsPanel)
    coreg_panel = Instance(CoregPanel)
    raw_src = DelegatesTo('model', 'hsp')

    # Omit Points
    distance = Float(5., label="Distance [mm]", desc="Maximal distance for "
                     "head shape points from MRI in mm")
    omit_points = Button(label='Omit Points', desc="Omit head shape points "
                         "for the purpose of the automatic coregistration "
                         "procedure.")
    reset_omit_points = Button(label='Reset Omission', desc="Reset the "
                               "omission of head shape points to include all.")
    omitted_info = Property(Str, depends_on=['model.hsp.n_omitted'])

    fid_ok = DelegatesTo('model', 'mri.fid_ok')
    lock_fiducials = DelegatesTo('model')
    hsp_always_visible = Bool(False, label="Always Show Head Shape")

    # visualization
    hsp_obj = Instance(PointObject)
    mri_obj = Instance(SurfaceObject)
    lpa_obj = Instance(PointObject)
    nasion_obj = Instance(PointObject)
    rpa_obj = Instance(PointObject)
    hsp_lpa_obj = Instance(PointObject)
    hsp_nasion_obj = Instance(PointObject)
    hsp_rpa_obj = Instance(PointObject)
    hsp_visible = Property(depends_on=['hsp_always_visible', 'lock_fiducials'])

    view_options = Button(label="View Options")

    picker = Instance(object)

    view_options_panel = Instance(ViewOptionsPanel)

    # Processing
    queue = DelegatesTo('coreg_panel')

    view = _make_view()

    def _subject_panel_default(self):
        return SubjectSelectorPanel(model=self.model.mri.subject_source)

    def _fid_panel_default(self):
        panel = FiducialsPanel(model=self.model.mri, headview=self.headview)
        return panel

    def _coreg_panel_default(self):
        panel = CoregPanel(model=self.model)
        return panel

    def _headview_default(self):
        return HeadViewController(scene=self.scene, system='RAS')

    def __init__(self, raw=None, subject=None, subjects_dir=None):
        super(CoregFrame, self).__init__()

        subjects_dir = get_subjects_dir(subjects_dir)
        if (subjects_dir is not None) and os.path.isdir(subjects_dir):
            self.model.mri.subjects_dir = subjects_dir

        if subject is not None:
            self.model.mri.subject = subject

        if raw is not None:
            self.model.hsp.file = raw

    @on_trait_change('scene.activated')
    def _init_plot(self):
        self.scene.disable_render = True

        lpa_color = defaults['lpa_color']
        nasion_color = defaults['nasion_color']
        rpa_color = defaults['rpa_color']

        # MRI scalp
        color = defaults['mri_color']
        self.mri_obj = SurfaceObject(points=self.model.transformed_mri_points,
                                     color=color, tri=self.model.mri.tris,
                                     scene=self.scene)
        # on_trait_change was unreliable, so link it another way:
        self.model.mri.on_trait_change(self._on_mri_src_change, 'tris')
        self.model.sync_trait('transformed_mri_points', self.mri_obj, 'points',
                              mutual=False)
        self.fid_panel.hsp_obj = self.mri_obj

        # MRI Fiducials
        point_scale = defaults['mri_fid_scale']
        self.lpa_obj = PointObject(scene=self.scene, color=lpa_color,
                                   point_scale=point_scale)
        self.model.mri.sync_trait('lpa', self.lpa_obj, 'points', mutual=False)
        self.model.sync_trait('scale', self.lpa_obj, 'trans', mutual=False)

        self.nasion_obj = PointObject(scene=self.scene, color=nasion_color,
                                      point_scale=point_scale)
        self.model.mri.sync_trait('nasion', self.nasion_obj, 'points',
                                  mutual=False)
        self.model.sync_trait('scale', self.nasion_obj, 'trans', mutual=False)

        self.rpa_obj = PointObject(scene=self.scene, color=rpa_color,
                                   point_scale=point_scale)
        self.model.mri.sync_trait('rpa', self.rpa_obj, 'points', mutual=False)
        self.model.sync_trait('scale', self.rpa_obj, 'trans', mutual=False)

        # Digitizer Head Shape
        color = defaults['hsp_point_color']
        point_scale = defaults['hsp_points_scale']
        p = PointObject(view='cloud', scene=self.scene, color=color,
                        point_scale=point_scale, resolution=5)
        self.hsp_obj = p
        self.model.hsp.sync_trait('points', p, mutual=False)
        self.model.sync_trait('head_mri_trans', p, 'trans', mutual=False)
        self.sync_trait('hsp_visible', p, 'visible', mutual=False)

        # Digitizer Fiducials
        point_scale = defaults['hsp_fid_scale']
        opacity = defaults['hsp_fid_opacity']
        p = PointObject(scene=self.scene, color=lpa_color, opacity=opacity,
                        point_scale=point_scale)
        self.hsp_lpa_obj = p
        self.model.hsp.sync_trait('lpa', p, 'points', mutual=False)
        self.model.sync_trait('head_mri_trans', p, 'trans', mutual=False)
        self.sync_trait('hsp_visible', p, 'visible', mutual=False)

        p = PointObject(scene=self.scene, color=nasion_color, opacity=opacity,
                        point_scale=point_scale)
        self.hsp_nasion_obj = p
        self.model.hsp.sync_trait('nasion', p, 'points', mutual=False)
        self.model.sync_trait('head_mri_trans', p, 'trans', mutual=False)
        self.sync_trait('hsp_visible', p, 'visible', mutual=False)

        p = PointObject(scene=self.scene, color=rpa_color, opacity=opacity,
                        point_scale=point_scale)
        self.hsp_rpa_obj = p
        self.model.hsp.sync_trait('rpa', p, 'points', mutual=False)
        self.model.sync_trait('head_mri_trans', p, 'trans', mutual=False)
        self.sync_trait('hsp_visible', p, 'visible', mutual=False)

        on_pick = self.scene.mayavi_scene.on_mouse_pick
        if not _testing_mode():
            self.picker = on_pick(self.fid_panel._on_pick, type='cell')

        self.headview.left = True
        self.scene.disable_render = False

        self.view_options_panel = ViewOptionsPanel(mri_obj=self.mri_obj,
                                                   hsp_obj=self.hsp_obj)

    @cached_property
    def _get_hsp_visible(self):
        return self.hsp_always_visible or self.lock_fiducials

    @cached_property
    def _get_omitted_info(self):
        if self.model.hsp.n_omitted == 0:
            return "No points omitted"
        elif self.model.hsp.n_omitted == 1:
            return "1 point omitted"
        else:
            return "%i points omitted" % self.model.hsp.n_omitted

    def _omit_points_fired(self):
        distance = self.distance / 1000.
        self.model.omit_hsp_points(distance)

    def _reset_omit_points_fired(self):
        self.model.omit_hsp_points(0, True)

    @on_trait_change('model.mri.tris')
    def _on_mri_src_change(self):
        if self.mri_obj is None:
            return
        if not (np.any(self.model.mri.points) and np.any(self.model.mri.tris)):
            self.mri_obj.clear()
            return

        self.mri_obj.points = self.model.mri.points
        self.mri_obj.tri = self.model.mri.tris
        self.mri_obj.plot()

    # automatically lock fiducials if a good fiducials file is loaded
    @on_trait_change('model.mri.fid_file')
    def _on_fid_file_loaded(self):
        if self.model.mri.fid_file:
            self.fid_panel.locked = True
        else:
            self.fid_panel.locked = False

    def _view_options_fired(self):
        self.view_options_panel.edit_traits()
