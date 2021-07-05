import numpy as np
from ..io import write_fiducials
from ..io.constants import FIFF
from ..surface import complete_surface_info, decimate_surface
from ..utils import warn
from ..coreg import (_find_fiducials_files, _find_head_bem,
                     get_mni_fiducials)
from ..source import _Surf, _MRISubjectSource, _SurfaceSource, _FiducialsSource


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
        self.rpa = np.empy((1, 3)).astype(float)

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
