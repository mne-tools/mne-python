import os
import re
import time
import warnings
import numpy as np
from ..io import write_fiducials
from ..io.constants import FIFF
from ..surface import complete_surface_info, decimate_surface, _DistanceQuery
from ..utils import warn
from ..coreg import (_find_fiducials_files, _find_head_bem, fit_matched_points,
                     get_mni_fiducials, bem_fname, _DEFAULT_PARAMETERS)
from ..transforms import (write_trans, read_trans, apply_trans, rotation,
                          rotation_angles, Transform, _ensure_trans,
                          rot_to_quat, _angle_between_quats)
from ..source import (_Surf, _MRISubjectSource, _SurfaceSource, _DigSource,
                      _FiducialsSource)
from ..utils import logger


class _MRIHeadWithFiducialsModel(object):
    def __init__(self):
        self.subject_source = _MRISubjectSource()
        self.bem_low_res = _SurfaceSource()
        self.bem_high_res = _SurfaceSource()
        self.fid = _FiducialsSource()

        # self.fid_file = DelegatesTo('fid', 'file')
        # self.fid_fname = DelegatesTo('fid', 'fname')
        # self.fid_points = DelegatesTo('fid', 'points')
        # self.subjects_dir = DelegatesTo('subject_source')
        # self.subject = DelegatesTo('subject_source')
        # self.subject_has_bem = DelegatesTo('subject_source')
        self.lpa = np.empty((1, 3)).astype(float)
        self.nasion = np.empty((1, 3)).astype(float)
        self.rpa = np.empty((1, 3)).astype(float)

        # info
        self.can_save = None
        self.can_save_as = None
        self.can_reset = None
        self.fid_ok = None
        self.default_fid_fname = None

    def reset_fiducials(self):  # noqa: D102
        if self.fid_points is not None:
            self.lpa = self.fid_points[0:1]
            self.nasion = self.fid_points[1:2]
            self.rpa = self.fid_points[2:3]

    def save(self, fname=None):
        """Save the current fiducials to a file.

        Parameters
        ----------
        fname : str
            Destination file path. If None, will use the current fid filename
            if available, or else use the default pattern.
        """
        if fname is None:
            fname = self.fid_file
        if not fname:
            fname = self.default_fid_fname

        dig = [{'kind': FIFF.FIFFV_POINT_CARDINAL,
                'ident': FIFF.FIFFV_POINT_LPA,
                'r': np.array(self.lpa[0])},
               {'kind': FIFF.FIFFV_POINT_CARDINAL,
                'ident': FIFF.FIFFV_POINT_NASION,
                'r': np.array(self.nasion[0])},
               {'kind': FIFF.FIFFV_POINT_CARDINAL,
                'ident': FIFF.FIFFV_POINT_RPA,
                'r': np.array(self.rpa[0])}]
        write_fiducials(fname, dig, FIFF.FIFFV_COORD_MRI)
        self.fid_file = fname

    def _get_can_reset(self):
        if not self.fid_file:
            return False
        elif np.any(self.lpa != self.fid.points[0:1]):
            return True
        elif np.any(self.nasion != self.fid.points[1:2]):
            return True
        elif np.any(self.rpa != self.fid.points[2:3]):
            return True
        return False

    def _get_can_save_as(self):
        can = not (np.all(self.nasion == self.lpa) or
                   np.all(self.nasion == self.rpa) or
                   np.all(self.lpa == self.rpa))
        return can

    def _get_can_save(self):
        if not self.can_save_as:
            return False
        elif self.fid_file:
            return True
        elif self.subjects_dir and self.subject:
            return True
        else:
            return False

    def _get_default_fid_fname(self):
        fname = self.fid.fname.format(subjects_dir=self.subjects_dir,
                                      subject=self.subject)
        return fname

    def _get_fid_ok(self):
        return all(np.any(pt) for pt in (self.nasion, self.lpa, self.rpa))

    def _reset_fired(self):
        self.reset_fiducials()

    def _subject_changed(self):
        subject = self.subject
        subjects_dir = self.subjects_dir
        if not subjects_dir or not subject:
            return

        # find high-res head model (if possible)
        high_res_path = _find_head_bem(subject, subjects_dir, high_res=True)
        low_res_path = _find_head_bem(subject, subjects_dir, high_res=False)
        if high_res_path is None and low_res_path is None:
            raise RuntimeError("No standard head model was found"
                               " for subject %s" % subject)
        if high_res_path is not None:
            self.bem_high_res.file = high_res_path
        else:
            self.bem_high_res.file = low_res_path
        if low_res_path is None:
            # This should be very rare!
            warn('No low-resolution head found, decimating high resolution '
                 'mesh (%d vertices): %s' % (len(self.bem_high_res.surf.rr),
                                             high_res_path,))
            # Create one from the high res one, which we know we have
            rr, tris = decimate_surface(self.bem_high_res.surf.rr,
                                        self.bem_high_res.surf.tris,
                                        n_triangles=5120)
            surf = complete_surface_info(dict(rr=rr, tris=tris),
                                         copy=False, verbose=False)
            # directly set the attributes of bem_low_res
            self.bem_low_res.surf = _Surf(tris=surf['tris'], rr=surf['rr'],
                                          nn=surf['nn'])
        else:
            self.bem_low_res.file = low_res_path

        # Set MNI points
        try:
            fids = get_mni_fiducials(subject, subjects_dir)
        except Exception:  # some problem, leave at origin
            self.fid.mni_points = None
        else:
            self.fid.mni_points = np.array([f['r'] for f in fids], float)

        # find fiducials file
        fid_files = _find_fiducials_files(subject, subjects_dir)
        if len(fid_files) == 0:
            self.fid.reset_traits(['file'])
            self.lock_fiducials = False
        else:
            self.fid_file = fid_files[0].format(subjects_dir=subjects_dir,
                                                subject=subject)
            self.lock_fiducials = True

        # does not seem to happen by itself ... so hard code it:
        self.reset_fiducials()


class _CoregModel():
    def __init__(self):
        # data sources
        self.mri = _MRIHeadWithFiducialsModel()
        self.hsp = _DigSource()

        # parameters
        self.guess_mri_subject = True
        self.grow_hair = None
        self.n_scale_params = (0, 1, 3)

        self.scale_x = 100
        self.scale_y = 100
        self.scale_z = 100
        self.trans_x = 0
        self.trans_y = 0
        self.trans_z = 0
        self.rot_x = 0
        self.rot_y = 0
        self.rot_z = 0
        self.parameters = list()
        self.last_parameters = list()
        self.lpa_weight = 1.
        self.nasion_weight = 10.
        self.rpa_weight = 1.
        self.hsp_weight = 1.
        self.eeg_weight = 1.
        self.hpi_weight = 1.
        self.iteration = -1
        self.icp_iterations = 20
        self.icp_start_time = 0.0
        self.icp_angle = 0.2
        self.icp_distance = 0.2
        self.icp_scale = 0.2
        self.icp_fid_match = ('nearest', 'matched')
        self.fit_icp_running = False
        self.fits_icp_running = False
        self.coord_frame = ('mri', 'head')
        self.status_text = str()

        # options during scaling
        self.scale_labels = True
        self.copy_annot = True
        self.prepare_bem_model = True

        # secondary to parameters
        self.has_nasion_data = None
        self.has_lpa_data = None
        self.has_rpa_data = None
        self.has_fid_data = None
        self.has_mri_data = None
        self.has_hsp_data = None
        self.has_eeg_data = None
        self.has_hpi_data = None
        self.n_icp_points = None
        self.changes = None

        # target transforms
        self.mri_head_t = None
        self.head_mri_t = None
        self.mri_trans_noscale = None
        self.mri_trans = None
        self.hsp_trans = None

        # info
        # subject_has_bem = DelegatesTo('mri')
        # lock_fiducials = DelegatesTo('mri')
        self.can_prepare_bem_model = None
        self.can_save = None
        self.raw_subject = None

        # Always computed in the MRI coordinate frame for speed
        # (building the nearest-neighbor tree is slow!)
        # though it will always need to be rebuilt in (non-uniform)
        # scaling mode
        self.nearest_calc = _DistanceQuery()

        # MRI geometry transformed to viewing coordinate system
        self.processed_high_res_mri_points = None
        self.processed_low_res_mri_points = None
        self.transformed_high_res_mri_points = None
        self.transformed_low_res_mri_points = None
        self.nearest_transformed_high_res_mri_idx_lpa = None
        self.nearest_transformed_high_res_mri_idx_nasion = None
        self.nearest_transformed_high_res_mri_idx_rpa = None
        self.nearest_transformed_high_res_mri_idx_hsp = None
        self.nearest_transformed_high_res_mri_idx_orig_hsp = None
        self.nearest_transformed_high_res_mri_idx_eeg = None
        self.nearest_transformed_high_res_mri_idx_hpi = None
        self.transformed_mri_lpa = None
        self.transformed_mri_nasion = None
        self.transformed_mri_rpa = None
        # HSP geometry transformed to viewing coordinate system
        self.transformed_hsp_points = None
        self.transformed_orig_hsp_points = None
        self.transformed_hsp_lpa = None
        self.transformed_hsp_nasion = None
        self.transformed_hsp_rpa = None
        self.transformed_hsp_eeg_points = None
        self.transformed_hsp_hpi = None

        # fit properties
        self.lpa_distance = None
        self.nasion_distance = None
        self.rpa_distance = None
        self.point_distance = None
        self.orig_hsp_point_distance = None

        # fit property info strings
        self.fid_eval_str = None
        self.points_eval_str = None

    def _parameters_default(self):
        return list(_DEFAULT_PARAMETERS)

    def _last_parameters_default(self):
        return list(_DEFAULT_PARAMETERS)

    def _get_can_prepare_bem_model(self):
        return self.subject_has_bem and self.n_scale_params > 0

    def _get_can_save(self):
        return np.any(self.mri_head_t != np.eye(4))

    def _get_has_lpa_data(self):
        return (np.any(self.mri.lpa) and np.any(self.hsp.lpa))

    def _get_has_nasion_data(self):
        return (np.any(self.mri.nasion) and np.any(self.hsp.nasion))

    def _get_has_rpa_data(self):
        return (np.any(self.mri.rpa) and np.any(self.hsp.rpa))

    def _get_has_fid_data(self):
        return self.has_nasion_data and self.has_lpa_data and self.has_rpa_data

    def _get_has_mri_data(self):
        return len(self.transformed_high_res_mri_points) > 0

    def _get_has_hsp_data(self):
        return (self.has_mri_data and
                len(self.nearest_transformed_high_res_mri_idx_hsp) > 0)

    def _get_has_eeg_data(self):
        return (self.has_mri_data and
                len(self.nearest_transformed_high_res_mri_idx_eeg) > 0)

    def _get_has_hpi_data(self):
        return (self.has_mri_data and
                len(self.nearest_transformed_high_res_mri_idx_hpi) > 0)

    def _get_n_icp_points(self):
        """Get parameters for an ICP iteration."""
        n = (self.hsp_weight > 0) * len(self.hsp.points)
        for key in ('lpa', 'nasion', 'rpa'):
            if getattr(self, 'has_%s_data' % key):
                n += 1
        n += (self.eeg_weight > 0) * len(self.hsp.eeg_points)
        n += (self.hpi_weight > 0) * len(self.hsp.hpi_points)
        return n

    def _get_changes(self):
        new = np.array(self.parameters, float)
        old = np.array(self.last_parameters, float)
        move = np.linalg.norm(old[3:6] - new[3:6]) * 1e3
        angle = np.rad2deg(_angle_between_quats(
            rot_to_quat(rotation(*new[:3])[:3, :3]),
            rot_to_quat(rotation(*old[:3])[:3, :3])))
        percs = 100 * (new[6:] - old[6:]) / old[6:]
        return move, angle, percs

    def _get_mri_head_t(self):
        # rotate and translate hsp
        trans = rotation(*self.parameters[:3])
        trans[:3, 3] = np.array(self.parameters[3:6])
        return trans

    def _get_head_mri_t(self):
        trans = rotation(*self.parameters[:3]).T
        trans[:3, 3] = -np.dot(trans[:3, :3], self.parameters[3:6])
        # should be the same as np.linalg.inv(self.mri_head_t)
        return trans

    def _get_processed_high_res_mri_points(self):
        return self._get_processed_mri_points('high')

    def _get_processed_low_res_mri_points(self):
        return self._get_processed_mri_points('low')

    def _get_processed_mri_points(self, res):
        bem = self.mri.bem_low_res if res == 'low' else self.mri.bem_high_res
        if self.grow_hair:
            if len(bem.surf.nn):
                scaled_hair_dist = (1e-3 * self.grow_hair /
                                    np.array(self.parameters[6:9]))
                points = bem.surf.rr.copy()
                hair = points[:, 2] > points[:, 1]
                points[hair] += bem.surf.nn[hair] * scaled_hair_dist
                return points
            else:
                raise ValueError("Norms missing from bem, can't grow hair")
                self.grow_hair = 0
        else:
            return bem.surf.rr

    def _get_mri_trans(self):
        t = self.mri_trans_noscale.copy()
        t[:, :3] *= self.parameters[6:9]
        return t

    def _get_mri_trans_noscale(self):
        if self.coord_frame == 'head':
            t = self.mri_head_t
        else:
            t = np.eye(4)
        return t

    def _get_hsp_trans(self):
        if self.coord_frame == 'head':
            t = np.eye(4)
        else:
            t = self.head_mri_t
        return t

    def _get_nearest_transformed_high_res_mri_idx_lpa(self):
        return self.nearest_calc.query(
            apply_trans(self.head_mri_t, self.hsp.lpa))[1]

    def _get_nearest_transformed_high_res_mri_idx_nasion(self):
        return self.nearest_calc.query(
            apply_trans(self.head_mri_t, self.hsp.nasion))[1]

    def _get_nearest_transformed_high_res_mri_idx_rpa(self):
        return self.nearest_calc.query(
            apply_trans(self.head_mri_t, self.hsp.rpa))[1]

    def _get_nearest_transformed_high_res_mri_idx_hsp(self):
        return self.nearest_calc.query(
            apply_trans(self.head_mri_t, self.hsp.points))[1]

    def _get_nearest_transformed_high_res_mri_idx_orig_hsp(self):
        # This is redundant to some extent with the one above due to
        # overlapping points, but it's fast and the refactoring to
        # remove redundancy would be a pain.
        return self.nearest_calc.query(
            apply_trans(self.head_mri_t, self.hsp._hsp_points))[1]

    def _get_nearest_transformed_high_res_mri_idx_eeg(self):
        return self.nearest_calc.query(
            apply_trans(self.head_mri_t, self.hsp.eeg_points))[1]

    def _get_nearest_transformed_high_res_mri_idx_hpi(self):
        return self.nearest_calc.query(
            apply_trans(self.head_mri_t, self.hsp.hpi_points))[1]

    # MRI view-transformed data
    def _get_transformed_low_res_mri_points(self):
        points = apply_trans(self.mri_trans,
                             self.processed_low_res_mri_points)
        return points

    def _nearest_calc_default(self):
        return _DistanceQuery(
            self.processed_high_res_mri_points * self.parameters[6:9])

    def _update_nearest_calc(self):
        self.nearest_calc = self._nearest_calc_default()

    def _get_transformed_high_res_mri_points(self):
        points = apply_trans(self.mri_trans,
                             self.processed_high_res_mri_points)
        return points

    def _get_transformed_mri_lpa(self):
        return apply_trans(self.mri_trans, self.mri.lpa)

    def _get_transformed_mri_nasion(self):
        return apply_trans(self.mri_trans, self.mri.nasion)

    def _get_transformed_mri_rpa(self):
        return apply_trans(self.mri_trans, self.mri.rpa)

    # HSP view-transformed data
    def _get_transformed_hsp_points(self):
        return apply_trans(self.hsp_trans, self.hsp.points)

    def _get_transformed_orig_hsp_points(self):
        return apply_trans(self.hsp_trans, self.hsp._hsp_points)

    def _get_transformed_hsp_lpa(self):
        return apply_trans(self.hsp_trans, self.hsp.lpa)

    def _get_transformed_hsp_nasion(self):
        return apply_trans(self.hsp_trans, self.hsp.nasion)

    def _get_transformed_hsp_rpa(self):
        return apply_trans(self.hsp_trans, self.hsp.rpa)

    def _get_transformed_hsp_eeg_points(self):
        return apply_trans(self.hsp_trans, self.hsp.eeg_points)

    def _get_transformed_hsp_hpi(self):
        return apply_trans(self.hsp_trans, self.hsp.hpi_points)

    # Distances, etc.
    def _get_lpa_distance(self):
        d = np.ravel(self.transformed_mri_lpa - self.transformed_hsp_lpa)
        return np.linalg.norm(d)

    def _get_nasion_distance(self):
        d = np.ravel(self.transformed_mri_nasion - self.transformed_hsp_nasion)
        return np.linalg.norm(d)

    def _get_rpa_distance(self):
        d = np.ravel(self.transformed_mri_rpa - self.transformed_hsp_rpa)
        return np.linalg.norm(d)

    def _get_point_distance(self):
        mri_points = list()
        hsp_points = list()
        if self.hsp_weight > 0 and self.has_hsp_data:
            mri_points.append(self.transformed_high_res_mri_points[
                self.nearest_transformed_high_res_mri_idx_hsp])
            hsp_points.append(self.transformed_hsp_points)
            assert len(mri_points[-1]) == len(hsp_points[-1])
        if self.eeg_weight > 0 and self.has_eeg_data:
            mri_points.append(self.transformed_high_res_mri_points[
                self.nearest_transformed_high_res_mri_idx_eeg])
            hsp_points.append(self.transformed_hsp_eeg_points)
            assert len(mri_points[-1]) == len(hsp_points[-1])
        if self.hpi_weight > 0 and self.has_hpi_data:
            mri_points.append(self.transformed_high_res_mri_points[
                self.nearest_transformed_high_res_mri_idx_hpi])
            hsp_points.append(self.transformed_hsp_hpi)
            assert len(mri_points[-1]) == len(hsp_points[-1])
        if all(len(h) == 0 for h in hsp_points):
            return None
        mri_points = np.concatenate(mri_points)
        hsp_points = np.concatenate(hsp_points)
        return np.linalg.norm(mri_points - hsp_points, axis=-1)

    def _get_orig_hsp_point_distance(self):
        mri_points = self.transformed_high_res_mri_points[
            self.nearest_transformed_high_res_mri_idx_orig_hsp]
        hsp_points = self.transformed_orig_hsp_points
        return np.linalg.norm(mri_points - hsp_points, axis=-1)

    def _get_fid_eval_str(self):
        d = (self.lpa_distance * 1000, self.nasion_distance * 1000,
             self.rpa_distance * 1000)
        return u'Fiducials: %.1f, %.1f, %.1f mm' % d

    def _get_points_eval_str(self):
        if self.point_distance is None:
            return ""
        dists = 1000 * self.point_distance
        av_dist = np.mean(dists)
        std_dist = np.std(dists)
        kinds = [kind for kind, check in
                 (('HSP', self.hsp_weight > 0 and self.has_hsp_data),
                  ('EEG', self.eeg_weight > 0 and self.has_eeg_data),
                  ('HPI', self.hpi_weight > 0 and self.has_hpi_data))
                 if check]
        return (u"%s %s: %.1f ± %.1f mm"
                % (len(dists), '+'.join(kinds), av_dist, std_dist))

    def _get_raw_subject(self):
        # subject name guessed based on the inst file name
        if '_' in self.hsp.inst_fname:
            subject, _ = self.hsp.inst_fname.split('_', 1)
            if subject:
                return subject

    def _on_raw_subject_change(self, subject):
        if self.guess_mri_subject:
            if subject in self.mri.subject_source.subjects:
                self.mri.subject = subject
            elif 'fsaverage' in self.mri.subject_source.subjects:
                self.mri.subject = 'fsaverage'

    def omit_hsp_points(self, distance):
        """Exclude head shape points that are far away from the MRI head.

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
        if distance <= 0:
            return

        # find the new filter
        mask = self.orig_hsp_point_distance <= distance
        n_excluded = np.sum(~mask)
        logger.info("Coregistration: Excluding %i head shape points with "
                    "distance >= %.3f m.", n_excluded, distance)
        # set the filter
        with warnings.catch_warnings(record=True):  # comp to None in Traits
            self.hsp.points_filter = mask

    def fit_fiducials(self, n_scale_params=None):
        """Find rotation and translation to fit all 3 fiducials."""
        if n_scale_params is None:
            n_scale_params = self.n_scale_params
        head_pts = np.vstack((self.hsp.lpa, self.hsp.nasion, self.hsp.rpa))
        mri_pts = np.vstack((self.mri.lpa, self.mri.nasion, self.mri.rpa))
        weights = [self.lpa_weight, self.nasion_weight, self.rpa_weight]
        assert n_scale_params in (0, 1)  # guaranteed by GUI
        if n_scale_params == 0:
            mri_pts *= self.parameters[6:9]  # not done in fit_matched_points
        x0 = np.array(self.parameters[:6 + n_scale_params])
        est = fit_matched_points(mri_pts, head_pts, x0=x0, out='params',
                                 scale=n_scale_params, weights=weights)
        if n_scale_params == 0:
            self.parameters[:6] = est
        else:
            self.parameters[:] = np.concatenate([est, [est[-1]] * 2])

    def _setup_icp(self, n_scale_params):
        """Get parameters for an ICP iteration."""
        head_pts = list()
        mri_pts = list()
        weights = list()
        if self.has_hsp_data and self.hsp_weight > 0:  # should be true
            head_pts.append(self.hsp.points)
            mri_pts.append(self.processed_high_res_mri_points[
                self.nearest_transformed_high_res_mri_idx_hsp])
            weights.append(np.full(len(head_pts[-1]), self.hsp_weight))
        for key in ('lpa', 'nasion', 'rpa'):
            if getattr(self, 'has_%s_data' % key):
                head_pts.append(getattr(self.hsp, key))
                if self.icp_fid_match == 'matched':
                    mri_pts.append(getattr(self.mri, key))
                else:
                    assert self.icp_fid_match == 'nearest'
                    mri_pts.append(self.processed_high_res_mri_points[
                        getattr(self, 'nearest_transformed_high_res_mri_idx_%s'
                                % (key,))])
                weights.append(np.full(len(mri_pts[-1]),
                                       getattr(self, '%s_weight' % key)))
        if self.has_eeg_data and self.eeg_weight > 0:
            head_pts.append(self.hsp.eeg_points)
            mri_pts.append(self.processed_high_res_mri_points[
                self.nearest_transformed_high_res_mri_idx_eeg])
            weights.append(np.full(len(mri_pts[-1]), self.eeg_weight))
        if self.has_hpi_data and self.hpi_weight > 0:
            head_pts.append(self.hsp.hpi_points)
            mri_pts.append(self.processed_high_res_mri_points[
                self.nearest_transformed_high_res_mri_idx_hpi])
            weights.append(np.full(len(mri_pts[-1]), self.hpi_weight))
        head_pts = np.concatenate(head_pts)
        mri_pts = np.concatenate(mri_pts)
        weights = np.concatenate(weights)
        if n_scale_params == 0:
            mri_pts *= self.parameters[6:9]  # not done in fit_matched_points
        return head_pts, mri_pts, weights

    def fit_icp(self, n_scale_params=None):
        """Find MRI scaling, translation, and rotation to match HSP."""
        if n_scale_params is None:
            n_scale_params = self.n_scale_params

        # Initial guess (current state)
        assert n_scale_params in (0, 1, 3)
        est = self.parameters[:[6, 7, None, 9][n_scale_params]]

        # Do the fits, assigning and evaluating at each step
        attr = 'fit_icp_running' if n_scale_params == 0 else 'fits_icp_running'
        setattr(self, attr, True)
        self.icp_start_time = time.time()
        for self.iteration in range(self.icp_iterations):
            head_pts, mri_pts, weights = self._setup_icp(n_scale_params)
            est = fit_matched_points(mri_pts, head_pts, scale=n_scale_params,
                                     x0=est, out='params', weights=weights)
            if n_scale_params == 0:
                self.parameters[:6] = est
            elif n_scale_params == 1:
                self.parameters[:] = list(est) + [est[-1]] * 2
            else:
                self.parameters[:] = est
            angle, move, scale = self.changes
            if angle <= self.icp_angle and move <= self.icp_distance and \
                    all(scale <= self.icp_scale):
                self.status_text = self.status_text[:-1] + '; converged)'
                break
            if not getattr(self, attr):  # canceled by user
                self.status_text = self.status_text[:-1] + '; cancelled)'
                break
        else:
            self.status_text = self.status_text[:-1] + '; did not converge)'
        setattr(self, attr, False)
        self.iteration = -1

    def get_scaling_job(self, subject_to, skip_fiducials):
        """Find all arguments needed for the scaling worker."""
        subjects_dir = self.mri.subjects_dir
        subject_from = self.mri.subject
        bem_names = []
        if self.can_prepare_bem_model and self.prepare_bem_model:
            pattern = bem_fname.format(subjects_dir=subjects_dir,
                                       subject=subject_from, name='(.+-bem)')
            bem_dir, pattern = os.path.split(pattern)
            for filename in os.listdir(bem_dir):
                match = re.match(pattern, filename)
                if match:
                    bem_names.append(match.group(1))

        return (subjects_dir, subject_from, subject_to, self.parameters[6:9],
                skip_fiducials, self.scale_labels, self.copy_annot, bem_names)

    def load_trans(self, fname):
        """Load the head-mri transform from a fif file.

        Parameters
        ----------
        fname : str
            File path.
        """
        self.set_trans(_ensure_trans(read_trans(fname, return_all=True),
                                     'mri', 'head')['trans'])

    def reset(self):
        """Reset all the parameters affecting the coregistration."""
        self.reset_traits(('grow_hair', 'n_scaling_params'))
        self.parameters[:] = _DEFAULT_PARAMETERS
        self.omit_hsp_points(np.inf)

    def set_trans(self, mri_head_t):
        """Set rotation and translation params from a transformation matrix.

        Parameters
        ----------
        mri_head_t : array, shape (4, 4)
            Transformation matrix from MRI to head space.
        """
        rot_x, rot_y, rot_z = rotation_angles(mri_head_t)
        x, y, z = mri_head_t[:3, 3]
        self.parameters[:6] = [rot_x, rot_y, rot_z, x, y, z]

    def save_trans(self, fname):
        """Save the head-mri transform as a fif file.

        Parameters
        ----------
        fname : str
            Target file path.
        """
        if not self.can_save:
            raise RuntimeError("Not enough information for saving transform")
        write_trans(fname, Transform('head', 'mri', self.head_mri_t))

    def _parameters_items_changed(self):
        # Update GUI as necessary
        n_scale = self.n_scale_params
        for ii, key in enumerate(('rot_x', 'rot_y', 'rot_z')):
            val = np.rad2deg(self.parameters[ii])
            if val != getattr(self, key):  # prevent circular
                setattr(self, key, val)
        for ii, key in enumerate(('trans_x', 'trans_y', 'trans_z')):
            val = self.parameters[ii + 3] * 1e3
            if val != getattr(self, key):  # prevent circular
                setattr(self, key, val)
        for ii, key in enumerate(('scale_x', 'scale_y', 'scale_z')):
            val = self.parameters[ii + 6] * 1e2
            if val != getattr(self, key):  # prevent circular
                setattr(self, key, val)
        # Only update our nearest-neighbor if necessary
        if self.parameters[6:9] != self.last_parameters[6:9]:
            self._update_nearest_calc()
        # Update the status text
        move, angle, percs = self.changes
        text = u'Change:  Δ=%0.1f mm  ∠=%0.2f°' % (move, angle)
        if n_scale:
            text += '  Scale ' if n_scale == 1 else '  Sx/y/z '
            text += '/'.join(['%+0.1f%%' % p for p in percs[:n_scale]])
        if self.iteration >= 0:
            text += u' (iteration %d/%d, %0.1f sec)' % (
                self.iteration + 1, self.icp_iterations,
                time.time() - self.icp_start_time)
        self.last_parameters[:] = self.parameters[:]
        self.status_text = text

    def _rot_x_changed(self):
        self.parameters[0] = np.deg2rad(self.rot_x)

    def _rot_y_changed(self):
        self.parameters[1] = np.deg2rad(self.rot_y)

    def _rot_z_changed(self):
        self.parameters[2] = np.deg2rad(self.rot_z)

    def _trans_x_changed(self):
        self.parameters[3] = self.trans_x * 1e-3

    def _trans_y_changed(self):
        self.parameters[4] = self.trans_y * 1e-3

    def _trans_z_changed(self):
        self.parameters[5] = self.trans_z * 1e-3

    def _scale_x_changed(self):
        if self.n_scale_params == 1:
            self.parameters[6:9] = [self.scale_x * 1e-2] * 3
        else:
            self.parameters[6] = self.scale_x * 1e-2

    def _scale_y_changed(self):
        self.parameters[7] = self.scale_y * 1e-2

    def _scale_z_changed(self):
        self.parameters[8] = self.scale_z * 1e-2
