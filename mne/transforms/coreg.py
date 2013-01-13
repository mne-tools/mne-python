"""Coregistration between different coordinate frames"""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

from copy import deepcopy
import fnmatch
import os
import re
import shutil
import subprocess

import logging
logger = logging.getLogger('mne')

import numpy as np
from numpy import dot
from numpy.linalg import inv
from scipy.optimize import leastsq
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

from ..fiff import Raw, FIFF
from ..fiff.meas_info import read_fiducials, write_fiducials
from ..label import read_label, Label
from ..source_space import read_source_spaces, write_source_spaces
from ..surface import read_surface, write_surface, read_bem_surfaces, \
                      write_bem_surface
from ..utils import get_subjects_dir
from .transforms import apply_trans, rotation, scaling, translation, write_trans



def fit_matched_pts(src_pts, tgt_pts, tol=None, params=False):
    """Find a transform that minimizes the squared distance between two
    matching sets of points.

    Uses :func:`scipy.optimize.leastsq` to find a transformation involving
    rotation and translation.

    Parameters
    ----------
    src_pts : array, shape = (n, 3)
        Points to which the transform should be applied.
    tgt_pts : array, shape = (n, 3)
        Points to which src_pts should be fitted. Each point in tgt_pts should
        correspond to the point in src_pts with the same index.
    tol : scalar | None
        The error tolerance. If the distance between any of the matched points
        exceeds this value in the solution, a RuntimeError is raised. With
        None, no error check is performed.
    params : bool
        Also return the estimated rotation and translation parameters.

    Returns
    -------
    trans : array, shape = (4, 4)
        Transformation that, if applied to src_pts, minimizes the squared
        distance to tgt_pts.
    [rotation : array, len = 3, optional]
        The rotation parameters around the x, y, and z axes (in radians).
    [translation : array, len = 3, optional]
        The translation parameters in x, y, and z direction.
    """
    def error(params):
        trans = dot(translation(*params[:3]), rotation(*params[3:]))
        est = apply_trans(trans, src_pts)
        return (tgt_pts - est).ravel()

    x0 = (0, 0, 0, 0, 0, 0)
    x, _, _, _, _ = leastsq(error, x0, full_output=True)

    transl = x[:3]
    rot = x[3:]
    trans = dot(translation(*transl), rotation(*rot))

    # assess the error of the solution
    if tol is not None:
        est_pts = apply_trans(trans, src_pts)
        err = np.sqrt(np.sum((est_pts - tgt_pts) ** 2, axis=1))
        if np.any(err > tol):
            raise RuntimeError("Error exceeds tolerance. Error = %r" % err)

    if params:
        return trans, rot, transl
    else:
        return trans



def find_mri_paths(subject='fsaverage', subjects_dir=None):
    """Find all files of an mri relevant for source transformation

    Parameters
    ----------
    subject : str
        Name of the mri subject.
    subjects_dir : None | path
        Override the SUBJECTS_DIR environment variable
        (sys.environ['SUBJECTS_DIR'])

    Returns
    -------
    paths | dict
        Dictionary whose keys are relevant file type names (str), and whose
        values are lists of paths.

    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)

    s_dir = os.path.join(subjects_dir, '{sub}')
    paths = {'s_dir': s_dir}

    # directories to create
    bem_dir = os.path.join(s_dir, 'bem')
    surf_dir = os.path.join(s_dir, 'surf')
    lbl_dir = os.path.join(s_dir, 'label')

    paths['dirs'] = [bem_dir, surf_dir]

    bem_path = os.path.join(bem_dir, '{name}.{ext}')
    surf_path = os.path.join(surf_dir, '{name}')

    # surf/ files
    paths['surf'] = surf = []
    surf_names = ('orig', 'orig_avg',
                  'inflated', 'inflated_avg', 'inflated_pre',
                  'pial', 'pial_avg',
                  'smoothwm',
                  'white', 'white_avg',
                  'sphere', 'sphere.reg', 'sphere.reg.avg')
    for name in surf_names:
        for hemi in ('lh.', 'rh.'):
            fname = surf_path.format(sub='{sub}', name=hemi + name)
            surf.append(fname)

    # watershed files
    for name in ['inner_skull', 'outer_skull', 'outer_skin']:
        fname = bem_path.format(sub='{sub}', name=name, ext='surf')
        surf.append(fname)

    # bem files
    paths['bem'] = bem = []
    path = os.path.join(subjects_dir, '{sub}', 'bem', '{sub}-{name}.fif')
    for name in ['head']:
        fname = path.format(sub='{sub}', name=name)
        bem.append(fname)

    # fiducials
    fname = path.format(sub='{sub}', name='fiducials')
    paths['fid'] = [fname]

    # check presence of required files
    for ftype in ['surf', 'bem', 'fid']:
        for fname in paths[ftype]:
            src = fname.format(sub=subject)
            if not os.path.exists(src):
                raise IOError("Required file not found: %r" % src)

    # src
    paths['ico'] = icos = []
    paths['src'] = src = []
    basename = '{sub}-ico-{ico}-src.fif'
    path = os.path.join(bem_dir, basename)
    p = re.compile('\A' + basename.format(sub=subject, ico='(\d+)'))
    for name in os.listdir(bem_dir.format(sub=subject)):
        match = p.match(name)
        if match:
            ico = int(match.group(1))
            icos.append(ico)
            fname = path.format(sub='{sub}', ico=ico)
            src.append(fname)
    if len(src) == 0:
        raise IOError("No source space found.")

    # labels
    paths['lbl'] = lbls = []
    top = lbl_dir.format(sub=subject)
    relpath_start = len(top) + 1
    for dirp, _, files in os.walk(top):
        files = fnmatch.filter(files, '*.label')
        dirname = os.path.join(lbl_dir, dirp[relpath_start:])
        for basename in files:
            lbls.append(os.path.join(dirname, basename))
        paths['dirs'].append(dirname)

    # duplicate curvature files
    paths['duplicate'] = dup = []
    path = os.path.join(surf_dir, '{name}')
    for name in ['lh.curv', 'rh.curv']:
        fname = path.format(sub='{sub}', name=name)
        dup.append(fname)

    return paths



class HeadMriFitter(object):
    """Fit the head shape to an mri (create a head mri trans file)
    """
    def __init__(self, raw, subject=None, subjects_dir=None):
        """
        Parameters
        ----------
        raw : Raw | str(path)
            The Raw object or the path to the raw file containing the digitizer
            data.
        subject : None | str
            name of the mri subject (e.g., 'fsaverage').
            Can be None if the raw file-name starts with "{subject}_".
        subjects_dir : None | path
            Override the SUBJECTS_DIR environment variable
            (sys.environ['SUBJECTS_DIR'])

        """
        subjects_dir = get_subjects_dir(subjects_dir, True)

        # resolve raw
        if isinstance(raw, basestring):
            raw_fname = raw
            raw = Raw(raw_fname)
        else:
            raw_fname = raw.info['filename']
        raw_fname = raw_fname

        # resolve subject
        if subject is None:
            _, tail = os.path.split(raw_fname)
            subject = tail.split('_')[0]

        # resolve mri subject path
        mri_sdir = os.path.join(subjects_dir, subject)
        if not os.path.exists(mri_sdir):
            err = ("Subject mri directory for %r not found "
                   "(%r)" % (subject, mri_sdir))
            raise ValueError(err)

        # mri head shape
        fname = os.path.join(mri_sdir, 'bem', '%s-%s.fif' % (subject, 'head'))
        self.mri_hs = geom_bem(fname, unit='m')

        # mri fiducials
        fname = os.path.join(mri_sdir, 'bem', subject + '-fiducials.fif')
        if not os.path.exists(fname):
            err = ("Now fiducials file found for %r (%r). Use XXX() "
                   "to create one." % (subject, fname))
            raise ValueError(err)
        dig, _ = read_fiducials(fname)
        self.mri_fid = geom_fid(dig, unit='m')

        # digitizer data from raw
        self.dig_hs = geom_dig_hs(raw.info['dig'], unit='m')
        self.dig_fid = geom_fid(raw.info['dig'], unit='m')

        # move to head to the mri's nasion
        self._t_origin_mri = translation(*self.mri_fid.nas)
        self._t_dig_origin = translation(*(-self.dig_fid.nas))

        # store attributes
        self._subject = subject
        self._raw_dir = os.path.dirname(raw_fname)
        self._subjects_dir = subjects_dir
        self.set(rot=(0, 0, 0), trans=(0, 0, 0))

    def _error(self, dig_trans, mri_trans=None):
        """
        For each point in the head shape, the distance to the closest point in
        the mri.

        Parameters
        ----------
        dig_trans : None | array, shape = (4, 4)
            The transformation matrix that is applied to the digitizer head
            shape.
        mri_trans : None | array, shape = (4, 4)
            The transformation matrix that is applied to the mri.
        """
        pts = self.dig_hs.get_pts(dig_trans)
        pts0 = self.mri_hs.get_pts(mri_trans)
        Y = cdist(pts, pts0, 'euclidean')
        dist = Y.min(axis=1)
        return dist

    def fit(self, move=False, **kwargs):
        """Fit the head to the mri using rotation and optionally translation

        Parameters
        ----------
        move : bool
            Also include translation parameters in the fit. If False, only
            rotation around the nasion is permitted.
        kwargs:
            scipy.optimize.leastsq kwargs

        http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html
        """
        if 'epsfcn' not in kwargs:
            kwargs['epsfcn'] = 0.01

        t_dig_origin = self._t_dig_origin
        t_origin_mri = self._t_origin_mri
        if move:
            x0 = np.hstack((self._rot, self._trans))
            def error(x):
                rx, ry, rz, tx, ty, tz = x
                trans = reduce(dot, (t_origin_mri, translation(tx, ty, tz), rotation(rx, ry, rz),
                                     t_dig_origin))
                err = self._error(trans)
                logger.debug("x = %s -> Error = "
                             "%s" % (x, np.sum(err ** 2)))
                return err
        else:
            x0 = self._rot
            t_origin_mri = dot(t_origin_mri, self._t_trans)
            def error(x):
                rx, ry, rz = x
                trans = reduce(dot, (t_origin_mri, rotation(rx, ry, rz),
                                     t_dig_origin))
                err = self._error(trans)
                logger.debug("x = %s -> Error = "
                             "%s" % (x, np.sum(err ** 2)))
                return err

        x_est, self.info = leastsq(error, x0, **kwargs)

        if move:
            self.set(rot=x_est[:3], trans=x_est[3:])
        else:
            self.set(rot=x_est)
        return x_est

    def get_head_mri_trans(self):
        trans = reduce(dot, (self._t_origin_mri, self._t_trans, self._t_rot,
                             self._t_dig_origin))
        return trans

    def plot(self, size=(512, 512), fig=None):
        if fig is None:
            from mayavi import mlab
            fig = mlab.figure(size=size)

        self.fig = fig
        self.mri_hs.plot_solid(fig)
        self.mri_fid.plot_points(fig, scale=.005)
        self.dig_hs.plot_solid(fig, opacity=1., rep='wireframe',
                               color=(.5, .5, 1))
        self.dig_fid.plot_points(fig, scale=.04, opacity=.25,
                                 color=(.5, .5, 1))
        return fig

    def save_trans(self, fname=None, overwrite=False):
        """Save the trans file

        Parameters
        ----------
        fname : str(path) | None
            Target file name. With None, a filename is constructed out of the
            directory of the raw file provided on initialization, the mri
            subject, and the suffix `-trans.fif`.
        overwrite : bool
            If a file already exists at the specified location, overwrite it.

        """
        if fname is None:
            name = self._subject + '-trans.fif'
            fname = os.path.join(self._raw_dir, name)

        if os.path.exists(fname):
            if overwrite:
                os.remove(fname)
            else:
                err = ("File already exists: %r" % fname)
                raise IOError(err)

        # in m
        trans = self.get_head_mri_trans()
        dig = deepcopy(self.dig_fid.source_dig)  # these are in m
        for d in dig:  # [in m]
            d['r'] = apply(trans, d['r'])
        trans = {'to': FIFF.FIFFV_COORD_MRI, 'from': FIFF.FIFFV_COORD_HEAD,
                'trans': trans, 'dig': dig}
        write_trans(fname, trans)

    def set(self, rot=None, trans=None):
        """Set the transformation parameters

        Parameters
        ----------
        rot : None | tuple of (x, y, z)
            Rotation parameters.
        trans : None | tuple of (x, y, z)
            Translation parameters.
        """
        if rot is not None:
            rot = np.asarray(rot, dtype=float)
            if rot.shape != (3,):
                raise ValueError("rot parameter needs to be of shape (3,), "
                                 "not %r" % rot.shape)
            self._t_rot = rotation(*rot)
            self._rot = rot
        if trans is not None:
            trans = np.asarray(trans, dtype=float)
            if trans.shape != (3,):
                raise ValueError("trans parameter needs to be of shape (3,), "
                                 "not %r" % trans.shape)
            self._t_trans = translation(*trans)
            self._trans = trans
        self.update()

    def update(self):
        """Update the transform and any plots"""
        trans = self.get_head_mri_trans()
        for g in [self.dig_hs, self.dig_fid]:
            g.set_trans(trans)



class MriHeadFitter(object):
    """
    Fit an MRI to a head shape model.

    Transforms applied to MRI:

    #. move MRI nasion to origin
    #. move MRI nasion according to the specified parameters
    #. apply scaling
    #. apply rotation
    #. move MRI nasion to headshape nasion

    .. note::
        Distances are internally represented in mm and converted where needed.

    """
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
        subjects_dir = get_subjects_dir(subjects_dir, True)

        # raw
        if isinstance(raw, basestring):
            raw_fname = raw
            raw = Raw(raw_fname)
        else:
            raw_fname = raw.info['filename']
        self._raw_fname = raw_fname

        # subject
        if (s_from is None) or (s_to is None):
            _, tail = os.path.split(raw_fname)
            subject = tail.split('_')[0]
            if s_from is None:
                s_from = subject
            if s_to is None:
                s_to = subject

        # MRI head shape
        mri_sdir = os.path.join(subjects_dir, s_from)
        if not os.path.exists(mri_sdir):
            err = ("MRI-directory for %r not found (%r)" % (s_from, mri_sdir))
            raise ValueError(err)
        fname = os.path.join(mri_sdir, 'bem', 'outer_skin.surf')
        pts, tri = read_surface(fname)
        self.mri_hs = geom(pts, tri)

        fname = os.path.join(mri_sdir, 'bem', s_from + '-fiducials.fif')
        if not os.path.exists(mri_sdir):
            err = ("Fiducials file for %r not found (%r). Use set_nasion() "
                   "to create it." % (s_from, mri_sdir))
            raise ValueError(err)
        dig, _ = read_fiducials(fname)
        self.mri_fid = geom_fid(dig, unit='mm')

        # digitizer data from raw
        self.dig_hs = geom_dig_hs(raw.info['dig'], unit='mm')
        self.dig_fid = geom_fid(raw.info['dig'], unit='mm')

        # move to the origin
        self._t_mri_origin = inv(translation(*self.mri_fid.nas))
        self._t_origin_dig = translation(*self.dig_fid.nas)

        self.subjects_dir = subjects_dir
        self.s_from = s_from
        self.s_to = s_to
        self._paths = find_mri_paths(s_from, subjects_dir)
        raw_dir = os.path.dirname(self._raw_fname)
        self._trans_fname = os.path.join(raw_dir, '{sub}-trans.fif')

        self._t_mri_origin_adjust = translation(0, 0, 0)
        self.set(0, 0, 0, 1, 1, 1)

    def plot(self, size=(512, 512), fig=None):
        if fig is None:
            from mayavi import mlab
            fig = mlab.figure(size=size)

        self.fig = fig
        self.mri_hs.plot_solid(fig)
        self.mri_fid.plot_points(fig, scale=5)
        self.dig_hs.plot_solid(fig, opacity=1., rep='wireframe',
                               color=(.5, .5, 1))
        self.dig_fid.plot_points(fig, scale=40, opacity=.25, color=(.5, .5, 1))
        return fig

    def _error(self, trans):
        "For each point in pts, the distance to the closest point in pts0"
        pts = self.dig_hs.get_pts()
        trans = reduce(dot, (self._t_origin_dig, trans, self._t_mri_origin_adjust,
                                self._t_mri_origin))
        pts0 = self.mri_hs.get_pts(trans)
        Y = cdist(pts, pts0, 'euclidean')
        dist = Y.min(axis=1)
        return dist

    def _dist_fixnas_mr(self, param):
        rx, ry, rz, mx, my, mz = param
        T = dot(rotation(rx, ry, rz), scaling(mx, my, mz))
        err = self._error(T)
        logger.debug("Params = %s -> Error = %s" % (param, np.sum(err ** 2)))
        return err

    def _dist_fixnas_r(self, param):
        rx, ry, rz = param
        T = rotation(rx, ry, rz)
        m = self._params[3:]
        if any(p != 1 for p in m):
            T = dot(T * scaling(*m))
        err = self._error(T)
        logger.debug("Params = %s -> Error = %s" % (param, np.sum(err ** 2)))
        return err

    def _dist_fixnas_1scale(self, param):
        rx, ry, rz, m = param
        T = dot(rotation(rx, ry, rz), scaling(m, m, m))
        err = self._error(T)
        logger.debug("Params = %s -> Error = %s" % (param, np.sum(err ** 2)))
        return err

    def _estimate_fixnas_mr(self, **kwargs):
        params = self._params
        params = np.asarray(params, dtype=float)
        est_params, self.info = leastsq(self._dist_fixnas_mr, params, **kwargs)
        return est_params

    def _estimate_fixnas_r(self, **kwargs):
        params = self._params[:3]
        params = np.asarray(params, dtype=float)
        est_params, self.info = leastsq(self._dist_fixnas_r, params, **kwargs)
        return est_params

    def _estimate_fixnas_1scale(self, **kwargs):
        params = self._params[:4]
        params = np.asarray(params, dtype=float)
        est_params, self.info = leastsq(self._dist_fixnas_1scale, params, **kwargs)
        return est_params

    def fit(self, method='sr', **kwargs):
        """
        Parameters
        ----------
        method : 'sr' | 'r' | '1sr'
            'sr': scaling with 3 parameters and rotation;
            'r': rotation only;
            '1sr': scaling with  single parameters and rotation.
        kwargs:
            scipy.optimize.leastsq kwargs

        http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html
        """
        if 'epsfcn' not in kwargs:
            kwargs['epsfcn'] = 0.01

        if method == '1sr':
            est = self._estimate_fixnas_1scale(**kwargs)
            est = np.hstack((est, np.ones(2) * est[-1:]))
        elif method == 'sr':
            est = self._estimate_fixnas_mr(**kwargs)
        elif method == 'r':
            est = self._estimate_fixnas_r(**kwargs)
            est = tuple(est) + (1, 1, 1)
        else:
            raise ValueError("No method %r" % method)
        self.set(*est)
        return est

    def get_mri_dir(self, s_to):
        return self._paths['s_dir'].format(sub=s_to)

    def get_rot(self):
        return self._params[:3]

    def get_scale(self):
        return self._params[3:]

    def get_t_scale(self, unit='mm'):
        """
        Scaling matrix for adjusting the size of the head model

        Parameters
        ----------
        unit : 'mm' | 'm'
            Unit of the coordinates on which the transform should operate.

        Returns
        -------
        trans : array, shape = (4, 4)
            Matrix to apply the scaling to the MRI.
        """
        T0 = dot(self._t_mri_origin_adjust, self._t_mri_origin)
        if unit == 'mm':
            pass
        elif unit == 'm':
            T0[:3, 3] /= 1000
        else:
            raise ValueError("unit: %r" % unit)

        T = reduce(dot, (inv(T0), self.trans_scale, T0))
        return T

    def get_t_trans(self, unit='mm'):
        """
        Head_mri_t for the trans file (rotation + translation)

        Parameters
        ----------
        unit : 'mm' | 'm'
            Unit of the coordinates on which the transform should operate.

        Returns
        -------
        trans : array, shape = (4, 4)
            The head-MRI transform.
        """
        T0 = dot(self._t_mri_origin_adjust, self._t_mri_origin)
        if unit == 'mm':
            T = reduce(dot, (self._t_origin_dig, self.trans_rot, T0))
        elif unit == 'm':
            T0[:3, 3] /= 1000
            trans1 = self._t_origin_dig.copy()
            trans1[:3, 3] /= 1000
            T = reduce(dot, (trans1, self.trans_rot, T0))
        else:
            raise ValueError('Unknown unit %r' % unit)
        return inv(T)

    def get_trans_fname(self, s_to):
        return self._trans_fname.format(sub=s_to)

    def save(self, s_to=None, surf=True, homog=True,
             setup_fwd=True, overwrite=False):
        """
        Save the scaled MRI as well as the trans file

        Parameters
        ----------
        s_to : None | str
            Subject for which to save MRI. With None (default), use the s_to
            set on initialization.
        surf, homog : bool
            Arguments for the `mne_setup_forward_model` call.
        setup_fwd : bool | 'block'
            Call `mne_setup_forward_model` at the end. With True, the command
            is called and the corresponding Popen object returned. With
            'block', the Python interpreter is blocked until
            `mne_setup_forward_model` finishes.
        overwrite : bool
            If an MRI already exists for this subject, overwrite it.

        See Also
        --------
        MriHeadFitter.save_trans : save only the trans file
        """
        s_from = self.s_from
        if s_to is None:
            s_to = self.s_to
        paths = self._paths

        # make sure we have an empty target directory
        dest = paths['s_dir'].format(sub=s_to)
        if os.path.exists(dest):
            if overwrite:
                shutil.rmtree(dest)
            else:
                err = ("Subject directory for %s already exists: "
                       "%r" % (s_to, dest))
                raise IOError(err)

        # write trans file
        self.save_trans(s_to=s_to, overwrite=overwrite)

        for dirname in paths['dirs']:
            os.makedirs(dirname.format(sub=s_to))

        # MRI Scaling [in m]
        trans_m = self.get_t_scale('m')
        trans_mm = self.get_t_scale('mm')
        trans_normals = inv(trans_m).T

        # save MRI scaling transform
        fname = os.path.join(dest, 'MRI-scale-trans.fif')
        COORD = FIFF.FIFFV_COORD_UNKNOWN
        write_trans(fname, {'trans':trans_m, 'from':COORD, 'to':COORD,
                            'dig':deepcopy(self.dig_fid.source_dig)})

        # surf files [in mm]
        for fname in paths['surf']:
            src = fname.format(sub=s_from)
            dest = fname.format(sub=s_to)
            pts, tri = read_surface(src)
            pts = apply_trans(trans_mm, pts)
            write_surface(dest, pts, tri)

        # bem files [in m]
        for fname in paths['bem']:
            src = fname.format(sub=s_from)
            surfs = read_bem_surfaces(src)
            if len(surfs) != 1:
                err = ("Bem file with more than one surface: %r" % src)
                raise NotImplementedError(err)
            surf0 = surfs[0]
            surf0['rr'] = apply_trans(trans_m, surf0['rr'])
            dest = fname.format(sub=s_to)
            write_bem_surface(dest, surf0)

        # fiducials [in m]
        for fname in paths['fid']:
            src = fname.format(sub=s_from)
            pts, cframe = read_fiducials(src)
            for pt in pts:
                pt['r'] = apply_trans(trans_m, pt['r'])
            dest = fname.format(sub=s_to)
            write_fiducials(dest, pts, cframe)

        # src [in m]
        for fname in paths['src']:
            src = fname.format(sub=s_from)
            sss = read_source_spaces(src)
            for ss in sss:
                ss['rr'] = apply_trans(trans_m, ss['rr'])
                ss['nn'] = apply_trans(trans_normals, ss['nn'])
            dest = fname.format(sub=s_to)
            write_source_spaces(dest, sss)

        # labels [in m]
        for fname in paths['lbl']:
            src = fname.format(sub=s_from)
            l_old = read_label(src)
            pos = apply_trans(trans_m, l_old.pos)
            l_new = Label(l_old.vertices, pos, l_old.values, l_old.hemi,
                          l_old.comment)
            dest = fname.format(sub=s_to)
            l_new.save(dest)

        # duplicate files
        for fname in paths['duplicate']:
            src = fname.format(sub=s_from)
            dest = fname.format(sub=s_to)
            shutil.copyfile(src, dest)

        # run mne_setup_forward_model
        if setup_fwd:
            block = setup_fwd == 'block'
            for ico in paths['ico']:
                self._mne_setup_forward_model(s_to, ico, surf=surf,
                                              homog=homog, block=block)

    def _mne_setup_forward_model(self, s_to, ico, surf=True, homog=True,
                                 block=False):
        "Run mne_setup_forward_model command"
        env = os.environ.copy()
        env['SUBJECTS_DIR'] = self.subjects_dir
        cmd = ["mne_setup_forward_model", "--subject", s_to, "--ico", str(ico)]
        if surf:
            cmd.append('--surf')
        if homog:
            cmd.append('--homog')

        p = subprocess.Popen(cmd, env=env)
        if block:
            p.communicate()
        else:
            return p

    def save_trans(self, fname=None, s_to=None, overwrite=False):
        """
        Save only the trans file

        Parameters
        ----------
        fname : str(path) | None
            Target file name. With None, a filename is constructed out of the
            directory of the raw file provided on initialization, `s_to`, and
            the suffix `-trans.fif`.
        s_to : None | str
            Subject for which to save MRI. With None (default), use the s_to
            set on initialization.
        overwrite : bool
            If a file already exists at the specified loaction, overwrite it.

        See Also
        --------
        MriHeadFitter.save : save the scaled head model and the trans file

        """
        if fname is None:
            if s_to is None:
                s_to = self.s_to
            fname = self.get_trans_fname(s_to)

        if os.path.exists(fname):
            if overwrite:
                os.remove(fname)
            else:
                err = ("Trans file exists: %r" % fname)
                raise IOError(err)

        # in m
        trans = self.get_t_trans('m')
        dig = deepcopy(self.dig_fid.source_dig)  # these are in m
        for d in dig:  # [in m]
            d['r'] = apply_trans(trans, d['r'])
        info = {'to': FIFF.FIFFV_COORD_MRI, 'from': FIFF.FIFFV_COORD_HEAD,
                'trans': trans, 'dig': dig}
        write_trans(fname, info)

    def set(self, rx, ry, rz, sx, sy, sz):
        """Set the rotation and scaling parameters and update any plots"""
        self._params = (rx, ry, rz, sx, sy, sz)
        self.trans_rot = rotation(rx, ry, rz)
        self.trans_scale = scaling(sx, sy, sz)
        self.update()

    def set_nasion(self, x, y, z):
        """Set the nasion position correction and update any plots (in mm)"""
        self._t_mri_origin_adjust = translation(x, y, z)
        self.update()

    def update(self):
        """Update the transform and any plots"""
        T = reduce(dot, (self._t_origin_dig, self.trans_rot, self.trans_scale,
                         self._t_mri_origin_adjust, self._t_mri_origin))
        T = inv(T)
        for g in [self.dig_hs, self.dig_fid]:
            g.set_trans(T)



class geom(object):
    """
    Represents a set of points and transformations.

    A geom object maintains a set of points (optionally with a
    triangularization) and a list of transforms, and can plot itself to a
    mayavi figure. The plot is dynamically updated when the transform changes.

    Parameters
    ----------
    pts : array, shape = (n_pts, 3)
        A list of points
    tri : None | array, shape = (n_tri, 3)
        Triangularization (optional). A list of triangles, each triangle
        composed of the indices of three points forming a triangle
        together.

    """
    def __init__(self, pts, tri=None):
        self.trans = None

        self.pts = pts = np.vstack((pts.T, np.ones(len(pts))))
        self.tri = tri

        self._plots_surf = []
        self._plots_pt = []

    def get_pts(self, trans=None):
        """
        Returns the points contained in the object

        Parameters
        ----------
        trans : None | True | Matrix (4x4)
            None: don't transform the points
            True: apply the transformation matrix that is stored in the object
            array: apply the given array as transformation matrix

        Returns
        -------
        pts : array, shape = (n_pts, 3)
            The points.

        """
        if trans is True:
            trans = self.get_trans()

        if trans is None:
            pts = self.pts
        else:
            pts = dot(trans, self.pts)

        return pts[:3].T

    def get_trans(self):
        "Returns the matrix for the complete transformation"
        return self.trans

    def plot_solid(self, fig, opacity=1., rep='surface', color=(1, 1, 1)):
        "Returns: mesh, surf"
        from mayavi.tools import pipeline

        if self.tri is None:
            d = Delaunay(self.pts[:3].T)
            self.tri = d.convex_hull

        x, y, z, _ = self.pts
        mesh = pipeline.triangular_mesh_source(x, y, z, self.tri, figure=fig)
        surf = pipeline.surface(mesh, figure=fig, color=color, opacity=opacity,
                                representation=rep)

        self._plots_surf.append((mesh, surf))
        if self.trans is not None:
            self.update_plot()

        return mesh, surf

    def plot_points(self, fig, scale=1e-2, opacity=1., color=(1, 0, 0)):
        "Returns: src, glyph"
        from mayavi.tools import pipeline

        x, y, z, _ = self.pts
        src = pipeline.scalar_scatter(x, y, z)
        glyph = pipeline.glyph(src, color=color, figure=fig, scale_factor=scale,
                               opacity=opacity)

        self._plots_pt.append((src, glyph))
        if self.trans is not None:
            self.update_plot()

        return src, glyph

    def set_opacity(self, v=1):
        "Set the opacity of the plot object"
        if v == 1:
            v = True
        elif v == 0:
            v = False
        else:
            raise NotImplementedError

        for _, plt in self._plots_pt + self._plots_surf:
            if isinstance(v, bool):
                plt.visible = v

    def set_trans(self, trans):
        """Set the active transform

        Parameters
        ----------
        trans : None | array, shape = (4,4)
            The transformation matrix to be applied.

        """
        if trans is not None:
            trans = np.asarray(trans)
            if trans.shape != (4, 4):
                err = "Transformation matrix needs to be of shape (4,4)"
                raise ValueError(err)

        self.trans = trans
        self.update_plot()

    def update_plot(self):
        pts = self.get_pts(trans=True)
        for mesh, _ in self._plots_surf:
            mesh.data.points = pts
        for src, _ in self._plots_pt:
            src.data.points = pts



class geom_fid(geom):
    def __init__(self, dig, unit='mm'):
        if unit == 'mm':
            x = 1000
        elif unit == 'm':
            x = 1
        else:
            raise ValueError('Unit: %r' % unit)

        dig = filter(lambda d: d['kind'] == 1, dig)
        pts = np.array([d['r'] for d in dig]) * x

        super(geom_fid, self).__init__(pts)
        self.unit = unit

        self.source_dig = dig
        digs = {d['ident']: d for d in dig}
        self.rap = digs[1]['r'] * x
        self.nas = digs[2]['r'] * x
        self.lap = digs[3]['r'] * x



class geom_dig_hs(geom):
    def __init__(self, dig, unit='mm'):
        if unit == 'mm':
            x = 1000
        elif unit == 'm':
            x = 1
        else:
            raise ValueError('Unit: %r' % unit)

        pts = filter(lambda d: d['kind'] == 4, dig)
        pts = np.array([d['r'] for d in pts]) * x

        super(geom_dig_hs, self).__init__(pts)



class geom_bem(geom):
    def __init__(self, bem, unit='m'):
        if isinstance(bem, basestring):
            bem = read_bem_surfaces(bem)[0]

        pts = bem['rr']
        tri = bem['tris']

        if unit == 'mm':
            pts *= 1000
        elif unit == 'm':
            pass
        else:
            raise ValueError('Unit: %r' % unit)

        super(geom_bem, self).__init__(pts, tri)
