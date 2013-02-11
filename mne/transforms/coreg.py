"""Coregistration between different coordinate frames"""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

from copy import deepcopy
import fnmatch
import os
import re
import shutil
from subprocess import check_output, CalledProcessError

import logging
logger = logging.getLogger('mne')

import numpy as np
from numpy import dot
from scipy.linalg import inv
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
    for name in ['head', 'inner_skull-bem']:
        fname = path.format(sub='{sub}', name=name)
        bem.append(fname)

    # fiducials
    fname = path.format(sub='{sub}', name='fiducials')
    paths['fid'] = [fname]

    # check presence of required files
    for ftype in ['surf', 'bem', 'fid']:
        for fname in paths[ftype]:
            src = os.path.realpath(fname.format(sub=subject))
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
    def __init__(self, raw, subject=None, subjects_dir=None):
        subjects_dir = get_subjects_dir(subjects_dir, True)

        # resolve raw
        if isinstance(raw, basestring):
            raw_fname = raw
            raw = Raw(raw_fname)
        else:
            raw_fname = raw.info['filename']
        raw_dir, raw_name = os.path.split(raw_fname)

        # resolve subject
        if subject is None:
            subject = raw_name.split('_')[0]

        # resolve mri subject path
        mri_sdir = os.path.join(subjects_dir, subject)
        if not os.path.exists(mri_sdir):
            err = ("Subject mri directory for %r not found "
                   "(%r)" % (subject, mri_sdir))
            raise ValueError(err)

        # mri head shape
        fname = os.path.join(mri_sdir, 'bem', '%s-%s.fif' % (subject, 'head'))
        self.mri_hs = BemGeom(fname, unit='m')

        # mri fiducials
        fname = os.path.join(mri_sdir, 'bem', subject + '-fiducials.fif')
        if not os.path.exists(fname):
            err = ("Now fiducials file found for %r (%r). Use mne.gui."
                   "set_fiducials() to create one." % (subject, fname))
            raise ValueError(err)
        dig, _ = read_fiducials(fname)
        self.mri_fid = FidGeom(dig, unit='m')

        # digitizer data from raw
        self.dig_hs = HeadshapeGeom(raw.info['dig'], unit='m')
        self.dig_fid = FidGeom(raw.info['dig'], unit='m')

        # move to head to the mri's nasion
        self._t_origin_mri = translation(*self.mri_fid.nas)
        self._t_dig_origin = translation(*(-self.dig_fid.nas))

        # path patterns
        self._trans_fname = os.path.join(raw_dir, '{subject}-trans.fif')

        # store attributes
        self._raw_dir = raw_dir
        self._raw_fname = raw_fname
        self._raw_name = raw_name
        self.subject = subject
        self.subjects_dir = subjects_dir

        self.reset()

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

    def fit_fiducials(self, fixed_nas=True, **kwargs):
        """Fit the head to the mri using only the fiducials.

        Parameters
        ----------
        fixed_nas : bool
            Keep the nasion position fixed.
        kwargs:
            scipy.optimize.leastsq kwargs

        """
        if 'epsfcn' not in kwargs:
            kwargs['epsfcn'] = 0.01

        t_dig_origin = self._t_dig_origin
        t_origin_mri = self._t_origin_mri

        src_pts = self.dig_fid.get_pts(t_dig_origin)
        t_mri_origin = inv(t_origin_mri)
        tgt_pts = self.mri_fid.get_pts(t_mri_origin)

        if fixed_nas:
            tgt_pts = apply_trans(inv(self._t_trans), tgt_pts)
            def error(params):
                trans = rotation(*params)
                est = apply_trans(trans, src_pts)
                return (tgt_pts - est).ravel()

            x0 = (0, 0, 0)
            rot, _ = leastsq(error, x0)

            self.set(rot=rot)
            return rot
        else:
            _, rot, transl = fit_matched_pts(src_pts, tgt_pts, params=True)
            self.set(rot=rot, trans=transl)
            return rot, transl

    def get_head_mri_trans(self):
        """Returns the head-mri transform

        Returns
        -------
        trans : array, shape = (4, 4)
            The head-MRI transform.
        """
        trans = reduce(dot, (self._t_origin_mri, self._t_trans, self._t_rot,
                             self._t_dig_origin))
        return trans

    def get_trans_fname(self, subject=None):
        if subject is None:
            subject = self.subject
        return self._trans_fname.format(subject=subject)

    def plot(self, size=(512, 512), fig=None):
        if fig is None:
            from mayavi import mlab
            fig = mlab.figure(size=size)

        self.fig = fig
        self.mri_hs.plot_solid(fig)
        self.mri_fid.plot_points(fig, scale=.005)
        self.dig_hs.plot_solid(fig, opacity=1., rep='wireframe',
                               color=(.1, .1, .6))
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
            fname = self.get_trans_fname(self.subject)

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
            d['r'] = apply_trans(trans, d['r'])
        trans = {'to': FIFF.FIFFV_COORD_MRI, 'from': FIFF.FIFFV_COORD_HEAD,
                'trans': trans, 'dig': dig}
        write_trans(fname, trans)

    def reset(self):
        self.set(rot=(0, 0, 0), trans=(0, 0, 0))

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



class MriHeadFitter(HeadMriFitter):
    """
    Fit an MRI to a head shape model.

    Parameters
    ----------
    raw : str(path)
        path to a raw file containing the digitizer data.
    s_from : str
        name of the mri subject providing the mri (e.g., 'fsaverage').
    subjects_dir : None | path
        Override the SUBJECTS_DIR environment variable
        (sys.environ['SUBJECTS_DIR'])

    See Also
    --------
    HeadMriFitter : Create a coregistration -trans file without scaling the mri.

    """
    def __init__(self, raw, s_from, subjects_dir=None):
        super(MriHeadFitter, self).__init__(raw, subject=s_from,
                                            subjects_dir=subjects_dir)

        self._paths = find_mri_paths(s_from, self.subjects_dir)

    def fit(self, scale=3, **kwargs):
        """Fit the head to the mri using rotation and optionally translation

        Parameters
        ----------
        scale : 3 | 1 | 0
            The number of scaling parameters to use.
        kwargs:
            scipy.optimize.leastsq kwargs

        http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html
        """
        if 'epsfcn' not in kwargs:
            kwargs['epsfcn'] = 0.01

        t_dig_origin = self._t_dig_origin
        t_digorigin_adjust = self._t_trans
        t_mri_origin = self._t_mri_origin

        if scale == 3:
            x0 = np.hstack((self._rot, self._scale))
            def error(x):
                rx, ry, rz, sx, sy, sz = x
                trans_dig = reduce(dot, (t_digorigin_adjust,
                                         rotation(rx, ry, rz), t_dig_origin))
                trans_mri = dot(scaling(sx, sy, sz), t_mri_origin)
                err = self._error(trans_dig, trans_mri)
                return err
        elif scale == 1:
            x0 = np.hstack((self._rot, [self._scale.mean()]))
            def error(x):
                rx, ry, rz, s = x
                trans_dig = reduce(dot, (t_digorigin_adjust,
                                         rotation(rx, ry, rz), t_dig_origin))
                trans_mri = dot(scaling(s, s, s), t_mri_origin)
                err = self._error(trans_dig, trans_mri)
                return err
        elif scale == 0:
            x0 = np.hstack((self._rot, [self._scale.mean()]))
            def error(x):
                rx, ry, rz, s = x
                trans_dig = reduce(dot, (t_digorigin_adjust,
                                         rotation(rx, ry, rz), t_dig_origin))
                trans_mri = dot(scaling(s, s, s), t_mri_origin)
                err = self._error(trans_dig, trans_mri)
                return err
        else:
            raise ValueError("Scale must be 1 or 3, not %r" % scale)

        x_est, self.info = leastsq(error, x0, **kwargs)

        rot = x_est[:3]
        if scale == 3:
            scale = x_est[3:]
        elif scale == 1:
            scale = x_est[3]
        else:
            scale = None
        self.set(rot=rot, scale=scale)
        return rot, scale

    def get_mri_dir(self, s_to):
        return self._paths['s_dir'].format(sub=s_to)

    def get_mri_trans(self):
        """Returns the scaling matrix for adjusting the size of the head model

        Returns
        -------
        trans : array, shape = (4, 4)
            Matrix to apply the scaling to the MRI.
        """
        trans = reduce(dot, (self._t_origin_mri, self._t_scale,
                             self._t_mri_origin))
        return trans

    def save_all(self, s_to=None, surf=True, homog=True,
                 setup_fwd=True, overwrite=False, trans_fname=None):
        """
        Save the scaled MRI as well as the trans file

        Parameters
        ----------
        s_to : None | str
            Subject for which to save MRI. With None (default), use the s_to
            set on initialization.
        surf, homog : bool
            Arguments for the `mne_setup_forward_model` call.
        setup_fwd : bool
            Execute `mne_setup_forward_model` after creating the mri files.
        overwrite : bool
            If an MRI already exists for this subject, overwrite it.
        trans_fname : None
            Where to save the head-mri transform -trans file. Default (None) is
            '{raw_dir}/{s_to}-trans.fif'

        See Also
        --------
        MriHeadFitter.save_trans : save only the trans file
        """
        s_from = self.subject
        if s_to is None:
            s_to = self._raw_name.split('_')[0]
        if trans_fname is None:
            trans_fname = self.get_trans_fname(s_to)

        # make sure we have an empty target directory
        paths = self._paths
        dest = paths['s_dir'].format(sub=s_to)
        if os.path.exists(dest):
            if overwrite:
                shutil.rmtree(dest)
            else:
                err = ("Subject directory for %s already exists: "
                       "%r" % (s_to, dest))
                raise IOError(err)

        # write trans file
        self.save_trans(fname=trans_fname, overwrite=overwrite)

        for dirname in paths['dirs']:
            os.makedirs(dirname.format(sub=s_to))

        # MRI Scaling
        trans_m = self.get_mri_trans()
        trans_mm = reduce(dot, (scaling(1e3, 1e3, 1e3), trans_m,
                                scaling(1e-3, 1e-3, 1e-3)))
        trans_normals = inv(trans_m).T

        # save MRI scaling transform
        fname = os.path.join(dest, 'MRI-scale-trans.fif')
        COORD = FIFF.FIFFV_COORD_UNKNOWN
        write_trans(fname, {'trans':trans_m, 'from':COORD, 'to':COORD,
                            'dig':deepcopy(self.dig_fid.source_dig)})

        # surf files [in mm]
        for fname in paths['surf']:
            src = os.path.realpath(fname.format(sub=s_from))
            dest = fname.format(sub=s_to)
            pts, tri = read_surface(src)
            pts = apply_trans(trans_mm, pts)
            write_surface(dest, pts, tri)

        # bem files [in m]
        for fname in paths['bem']:
            src = os.path.realpath(fname.format(sub=s_from))
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
            src = os.path.realpath(fname.format(sub=s_from))
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
            for ico in paths['ico']:
                self._mne_setup_forward_model(s_to, ico, surf=surf,
                                              homog=homog)

    def _mne_setup_forward_model(self, s_to, ico, surf=True, homog=True):
        "Run mne_setup_forward_model command"
        env = os.environ.copy()
        env['SUBJECTS_DIR'] = self.subjects_dir
        cmd = ["mne_setup_forward_model", "--subject", s_to, "--ico", str(ico)]
        if surf:
            cmd.append('--surf')
        if homog:
            cmd.append('--homog')

        try:
            out = check_output(cmd)
        except CalledProcessError as err:
            title = 'Error executing mne_setup_forward_model'
            print os.linesep.join((title, '=' * len(title), err.output))
            raise
        else:
            title = 'mne_setup_forward_model'
            print os.linesep.join((title, '=' * len(title), out))

    def reset(self):
        self._t_mri_origin = translation(*(-self.mri_fid.nas))
        self.set(rot=(0, 0, 0), trans=(0, 0, 0), scale=(1, 1, 1))

    def set(self, rot=None, trans=None, scale=None):
        """Set the rotation and scaling parameters and update any plots"""
        if scale is not None:
            if np.isscalar(scale):
                scale = (scale, scale, scale)
            scale = np.asarray(scale, dtype=float)
            if scale.shape != (3,):
                raise ValueError("scale parameter needs to be scalar or of "
                                 "shape (3,), not %r" % scale)
            self._t_scale = scaling(*scale)
            self._scale = scale
        super(MriHeadFitter, self).set(rot=rot, trans=trans)

    def update(self):
        super(MriHeadFitter, self).update()
        trans = self.get_mri_trans()
        for g in (self.mri_hs, self.mri_fid):
            g.set_trans(trans)



class Geom(object):
    """
    Represents a set of points and transformations.

    A Geom object maintains a set of points (optionally with a
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
            trans = self.trans

        if trans is None:
            pts = self.pts
        else:
            pts = dot(trans, self.pts)

        return pts[:3].T

    def plot_solid(self, fig, opacity=1., rep='surface', color=(1, 1, 1)):
        "Returns: mesh, surf"
        from mayavi.tools import pipeline

        if self.tri is None:
            d = Delaunay(self.pts[:3].T)
            self.tri = d.convex_hull

        x, y, z, _ = self.pts
        mesh = pipeline.triangular_mesh_source(x, y, z, self.tri, figure=fig)
        surf = pipeline.surface(mesh, figure=fig, color=color, opacity=opacity,
                                representation=rep, line_width=1)
        if rep == 'wireframe':
            surf.actor.property.lighting = False

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



class FidGeom(Geom):
    def __init__(self, dig, unit='mm'):
        if unit == 'mm':
            x = 1000
        elif unit == 'm':
            x = 1
        else:
            raise ValueError('Unit: %r' % unit)

        dig = filter(lambda d: d['kind'] == 1, dig)
        pts = np.array([d['r'] for d in dig]) * x

        super(FidGeom, self).__init__(pts)
        self.unit = unit

        self.source_dig = dig
        digs = {d['ident']: d for d in dig}
        self.rap = digs[1]['r'] * x
        self.nas = digs[2]['r'] * x
        self.lap = digs[3]['r'] * x



class HeadshapeGeom(Geom):
    def __init__(self, dig, unit='mm'):
        if unit == 'mm':
            x = 1000
        elif unit == 'm':
            x = 1
        else:
            raise ValueError('Unit: %r' % unit)

        pts = filter(lambda d: d['kind'] == 4, dig)
        pts = np.array([d['r'] for d in pts]) * x

        super(HeadshapeGeom, self).__init__(pts)



class BemGeom(Geom):
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

        super(BemGeom, self).__init__(pts, tri)
