"""Coregistration between different coordinate frames"""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

from ConfigParser import RawConfigParser
import fnmatch
from glob import iglob
import os
import re
import shutil

import logging
logger = logging.getLogger('mne')

import numpy as np
from numpy import dot
from scipy.optimize import leastsq
from scipy.spatial.distance import cdist

from ..fiff.meas_info import read_fiducials, write_fiducials
from ..label import read_label, Label
from ..source_space import read_source_spaces, write_source_spaces
from ..surface import read_surface, write_surface, read_bem_surfaces, \
                      write_bem_surface
from ..utils import get_config, get_subjects_dir
from .transforms import rotation, rotation3d, scaling, translation


trans_fname = os.path.join('{raw_dir}', '{subject}-trans.fif')


def create_default_subject(mne_root=None, fs_home=None, subjects_dir=None):
    """Create a default subject in subjects_dir

    Create a copy of fsaverage from the freesurfer directory in subjects_dir
    and add auxiliary files from the mne package.

    Parameters
    ----------
    mne_root : None | str
        The mne root directory (only needed if MNE_ROOT is not specified as
        environment variable).
    fs_home : None | str
        The freesurfer home directory (only needed if FREESURFER_HOME is not
        specified as environment variable).
    subjects_dir : None | path
        Override the SUBJECTS_DIR environment variable
        (os.environ['SUBJECTS_DIR']) as destination for the new subject.
    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    if fs_home is None:
        fs_home = get_config('FREESURFER_HOME', fs_home)
        if fs_home is None:
            err = ("FREESURFER_HOME environment variable not found. Please "
                   "specify the fs_home parameter in your call to "
                   "create_default_subject().")
            raise ValueError(err)
    if mne_root is None:
        mne_root = get_config('MNE_ROOT', mne_root)
        if mne_root is None:
            err = ("MNE_ROOT environment variable not found. Please "
                   "specify the mne_root parameter in your call to "
                   "create_default_subject().")
            raise ValueError(err)

    # make sure freesurfer files exist
    fs_src = os.path.join(fs_home, 'subjects', 'fsaverage')
    if not os.path.exists(fs_src):
        err = ('fsaverage not found at %r. Is fs_home specified '
               'correctly?' % fs_src)
        raise IOError(err)
    for name in ('label', 'mri', 'surf'):
        dirname = os.path.join(fs_src, name)
        if not os.path.isdir(dirname):
            err = ("Freesurfer fsaverage seems to be incomplete: No directory "
                   "named %s found in %s" % (name, fs_src))
            raise IOError(err)

    # make sure destination does not already exist
    dest = os.path.join(subjects_dir, 'fsaverage')
    if dest == fs_src:
        err = ("Your subjects_dir points to the freesurfer subjects_dir (%r). "
               "The default subject can not be created in the freesurfer "
               "installation directory; please specify a different "
               "subjects_dir." % subjects_dir)
        raise IOError(err)
    elif os.path.exists(dest):
        err = ("Can not create fsaverage because %r already exists in "
               "subjects_dir %r. Delete or rename the existing fsaverage "
               "subject folder." % ('fsaverage', subjects_dir))
        raise IOError(err)

    # make sure mne files exist
    mne_fname = os.path.join(mne_root, 'share', 'mne', 'mne_analyze',
                             'fsaverage', 'fsaverage-%s.fif')
    mne_files = ('fiducials', 'head', 'inner_skull-bem', 'trans')
    for name in mne_files:
        fname = mne_fname % name
        if not os.path.isfile(fname):
            err = ("MNE fsaverage incomplete: %s file not found at "
                   "%s" % (name, fname))
            raise IOError(err)

    # copy fsaverage from freesurfer
    logger.info("Copying fsaverage subject from freesurfer directory...")
    shutil.copytree(fs_src, dest)

    # add files from mne
    dest_bem = os.path.join(dest, 'bem')
    if not os.path.exists(dest_bem):
        os.mkdir(dest_bem)
    logger.info("Copying auxiliary fsaverage files from mne directory...")
    for name in mne_files:
        shutil.copy(mne_fname % name, dest_bem)


def decimate_points(pts, res=10):
    """Decimate the number of points using a voxel grid

    Create a voxel grid with a specified resolution and retain at most one
    point per voxel. For each voxel, the point closest to its center is
    retained.

    Parameters
    ----------
    pts : array, shape = (n_points, 3)
        The points making up the head shape.
    res : scalar
        The resolution of the voxel space (side length of each voxel).

    Returns
    -------
    pts : array, shape = (n_points, 3)
        The decimated points.
    """
    pts = np.asarray(pts)

    # find the bin edges for the voxel space
    xmin, ymin, zmin = pts.min(0) - res / 2.
    xmax, ymax, zmax = pts.max(0) + res
    xax = np.arange(xmin, xmax, res)
    yax = np.arange(ymin, ymax, res)
    zax = np.arange(zmin, zmax, res)

    # find voxels containing one or more point
    H, _ = np.histogramdd(pts, bins=(xax, yax, zax), normed=False)

    # for each voxel, select one point
    X, Y, Z = pts.T
    out = np.empty((np.sum(H > 0), 3))
    for i, (xbin, ybin, zbin) in enumerate(zip(*np.nonzero(H))):
        x = xax[xbin]
        y = yax[ybin]
        z = zax[zbin]
        xi = np.logical_and(X >= x, X < x + res)
        yi = np.logical_and(Y >= y, Y < y + res)
        zi = np.logical_and(Z >= z, Z < z + res)
        idx = np.logical_and(zi, np.logical_and(yi, xi))
        ipts = pts[idx]

        mid = np.array([x, y, z]) + res / 2.
        dist = cdist(ipts, [mid])
        i_min = np.argmin(dist)
        ipt = ipts[i_min]
        out[i] = ipt

    return out


def _trans_from_params(param_info, params):
    """Convert transformation parameters into a transformation matrix

    Parameters
    ----------
    param_info : tuple,  len = 3
        Tuple describing the parameters in x (do_translate, do_rotate,
        do_scale).
    params : tuple
        The transformation parameters.

    Returns
    -------
    trans : array, shape = (4, 4)
        Transformation matrix.
    """
    do_rotate, do_translate, do_scale = param_info
    i = 0
    trans = []

    if do_rotate:
        x, y, z = params[:3]
        trans.append(rotation(x, y, z))
        i += 3

    if do_translate:
        x, y, z = params[i:i + 3]
        trans.insert(0, translation(x, y, z))
        i += 3

    if do_scale == 1:
        s = params[i]
        trans.append(scaling(s, s, s))
    elif do_scale == 3:
        x, y, z = params[i:i + 3]
        trans.append(scaling(x, y, z))

    trans = reduce(dot, trans)
    return trans


def fit_matched_pts(src_pts, tgt_pts, rotate=True, translate=True, scale=0,
                    tol=None, x0=None, out='trans'):
    """Find a transform that minimizes the squared distance between two
    matching sets of points.

    Uses :func:`scipy.optimize.leastsq` to find a transformation involving
    a combination of rotation, translation, and scaling (in that order).

    Parameters
    ----------
    src_pts : array, shape = (n, 3)
        Points to which the transform should be applied.
    tgt_pts : array, shape = (n, 3)
        Points to which src_pts should be fitted. Each point in tgt_pts should
        correspond to the point in src_pts with the same index.
    rotate : bool
        Allow rotation of the ``src_pts``.
    translate : bool
        Allow translation of the ``src_pts``.
    scale : 0 | 1
        Number of scaling parameters.
    tol : scalar | None
        The error tolerance. If the distance between any of the matched points
        exceeds this value in the solution, a RuntimeError is raised. With
        None, no error check is performed.
    x0 : None | tuple
        Initial values for the fit parameters.
    out : 'params' | 'trans'
        In what format to return the estimate: 'params' returns a tuple with
        the fit parameters; 'trans' returns a transformation matrix of shape
        (4, 4).


    Returns
    -------
    One of the following, depending on the ``out`` parameter:

    trans : array, shape = (4, 4)
        Transformation that, if applied to src_pts, minimizes the squared
        distance to tgt_pts.
    params : array, shape = (n_params, )
        A single tuple containing the translation, rotation and scaling
        parameters in that order.
    """
    src_pts = np.atleast_2d(src_pts)
    tgt_pts = np.atleast_2d(tgt_pts)
    if src_pts.shape != tgt_pts.shape:
        err = ("src_pts and tgt_pts must have same shape "
               "(got {0}, {1})".format(src_pts.shape, tgt_pts.shape))
        raise ValueError(err)

    rotate = bool(rotate)
    translate = bool(translate)
    scale = int(scale)
    if translate:
        src_pts = np.hstack((src_pts, np.ones((len(src_pts), 1))))

    param_info = (rotate, translate, scale)
    if param_info == (True, False, 0):
        def error(x):
            rx, ry, rz = x
            trans = rotation3d(rx, ry, rz)
            est = dot(src_pts, trans.T)
            return (tgt_pts - est).ravel()
        if x0 is None:
            x0 = (0, 0, 0)
    elif param_info == (True, False, 1):
        def error(x):
            rx, ry, rz, s = x
            trans = rotation3d(rx, ry, rz) * s
            est = dot(src_pts, trans.T)
            return (tgt_pts - est).ravel()
        if x0 is None:
            x0 = (0, 0, 0, 1)
    elif param_info == (True, True, 0):
        def error(x):
            rx, ry, rz, tx, ty, tz = x
            trans = dot(translation(tx, ty, tz), rotation(rx, ry, rz))
            est = dot(src_pts, trans.T)
            return (tgt_pts - est[:, :3]).ravel()
        if x0 is None:
            x0 = (0, 0, 0, 0, 0, 0)
    elif param_info == (True, True, 1):
        def error(x):
            rx, ry, rz, tx, ty, tz, s = x
            trans = reduce(dot, (translation(tx, ty, tz), rotation(rx, ry, rz),
                                 scaling(s, s, s)))
            est = dot(src_pts, trans.T)
            return (tgt_pts - est[:, :3]).ravel()
        if x0 is None:
            x0 = (0, 0, 0, 0, 0, 0, 1)
    else:
        err = ("The specified parameter combination is not implemented: "
               "rotate=%r, translate=%r, scale=%r" % param_info)
        raise NotImplementedError(err)

    x, _, _, _, _ = leastsq(error, x0, full_output=True)

    # re-create the final transformation matrix
    if (tol is not None) or (out == 'trans'):
        trans = _trans_from_params(param_info, x)

    # assess the error of the solution
    if tol is not None:
        if not translate:
            src_pts = np.hstack((src_pts, np.ones((len(src_pts), 1))))
        est_pts = dot(src_pts, trans.T)[:, :3]
        err = np.sqrt(np.sum((est_pts - tgt_pts) ** 2, axis=1))
        if np.any(err > tol):
            raise RuntimeError("Error exceeds tolerance. Error = %r" % err)

    if out == 'params':
        return x
    elif out == 'trans':
        return trans
    else:
        err = ("Invalid out parameter: %r. Needs to be 'params' or "
              "'trans'." % out)
        raise ValueError(err)


def _point_cloud_error(src_pts, tgt_pts):
    """Find the distance from each source point to its closest target point

    Parameters
    ----------
    src_pts : array, shape = (n, 3)
        Source points.
    tgt_pts : array, shape = (m, 3)
        Target points.

    Returns
    -------
    dist : array, shape = (n, )
        For each point in ``src_pts``, te distance to the closest point in
        ``tgt_pts``.
    """
    Y = cdist(src_pts, tgt_pts, 'euclidean')
    dist = Y.min(axis=1)
    return dist


def fit_point_cloud(src_pts, tgt_pts, rotate=True, translate=True,
                    scale=0, x0=None, lsq_args={}, out='params'):
    """Find a transform that minimizes the squared distance from each source
    point to its closest target point

    Uses :func:`scipy.optimize.leastsq` to find a transformation involving
    a combination of rotation, translation, and scaling (in that order).

    Parameters
    ----------
    src_pts : array, shape = (n, 3)
        Points to which the transform should be applied.
    tgt_pts : array, shape = (n, 3)
        Points to which src_pts should be fitted. Each point in tgt_pts should
        correspond to the point in src_pts with the same index.
    rotate : bool
        Allow rotation of the ``src_pts``.
    translate : bool
        Allow translation of the ``src_pts``.
    scale : 0 | 1 | 3
        Number of scaling parameters.
    x0 : None | tuple
        Initial values for the fit parameters.
    lsq_args : dict
        Additional parameters to submit to :func:`scipy.optimize.leastsq`.
    out : 'params' | 'trans'
        In what format to return the estimate: 'params' returns a tuple with
        the fit parameters; 'trans' returns a transformation matrix of shape
        (4, 4).

    Returns
    -------
    x : array, shape = (n_params, )
        Estimated parameters for the transformation.

    Notes
    -----
    Assumes that the target points form a dense enough point cloud so that
    the distance of each src_pt to the closest tgt_pt can be used as an
    estimate of the distance of src_pt to tgt_pts.
    """
    kwargs = {'epsfcn': 0.01}
    kwargs.update(lsq_args)

    # assert correct argument types
    src_pts = np.atleast_2d(src_pts)
    tgt_pts = np.atleast_2d(tgt_pts)
    translate = bool(translate)
    rotate = bool(rotate)
    scale = int(scale)

    if translate:
        src_pts = np.hstack((src_pts, np.ones((len(src_pts), 1))))

    # for efficiency, define parameter specific error function
    param_info = (rotate, translate, scale)
    if param_info == (True, False, 0):
        x0 = x0 or (0, 0, 0)
        def error(x):
            rx, ry, rz = x
            trans = rotation3d(rx, ry, rz)
            est = dot(src_pts, trans.T)
            err = _point_cloud_error(est, tgt_pts)
            return err
    elif param_info == (True, False, 1):
        x0 = x0 or (0, 0, 0, 1)
        def error(x):
            rx, ry, rz, s = x
            trans = rotation3d(rx, ry, rz) * s
            est = dot(src_pts, trans.T)
            err = _point_cloud_error(est, tgt_pts)
            return err
    elif param_info == (True, False, 3):
        x0 = x0 or (0, 0, 0, 1, 1, 1)
        def error(x):
            rx, ry, rz, sx, sy, sz = x
            trans = rotation3d(rx, ry, rz) * [sx, sy, sz]
            est = dot(src_pts, trans.T)
            err = _point_cloud_error(est, tgt_pts)
            return err
    elif param_info == (True, True, 0):
        x0 = x0 or (0, 0, 0, 0, 0, 0)
        def error(x):
            rx, ry, rz, tx, ty, tz = x
            trans = dot(translation(tx, ty, tz), rotation(rx, ry, rz))
            est = dot(src_pts, trans.T)
            err = _point_cloud_error(est[:, :3], tgt_pts)
            return err
    else:
        err = ("The specified parameter combination is not implemented: "
               "rotate=%r, translate=%r, scale=%r" % param_info)
        raise NotImplementedError(err)

    est, _, info, msg, _ = leastsq(error, x0, full_output=True, **kwargs)
    logging.debug("fit_point_cloud leastsq (%i calls) info: %s", info['nfev'],
                  msg)

    if out == 'params':
        return est
    elif out == 'trans':
        return _trans_from_params(param_info, est)
    else:
        err = ("Invalid out parameter: %r. Needs to be 'params' or "
              "'trans'." % out)
        raise ValueError(err)


def _find_label_paths(subject='fsaverage', pattern=None, subjects_dir=None):
    """Find paths to label files in a subject's label directory

    Parameters
    ----------
    subject : str
        Name of the mri subject.
    pattern : str | None
        Pattern for finding the labels relative to the label directory in the
        MRI subject directory (e.g., "aparc/*.label" will find all labels
        in the "subject/label/aparc" directory). With None, find all labels.
    subjects_dir : None | path
        Override the SUBJECTS_DIR environment variable
        (sys.environ['SUBJECTS_DIR'])

    Returns
    ------
    paths : list
        List of paths relative to the subject's label directory
    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    subject_dir = os.path.join(subjects_dir, subject)
    lbl_dir = os.path.join(subject_dir, 'label')

    if pattern is None:
        paths = []
        for dirpath, _, filenames in os.walk(lbl_dir):
            rel_dir = os.path.relpath(dirpath, lbl_dir)
            for filename in fnmatch.filter(filenames, '*.label'):
                path = os.path.join(rel_dir, filename)
                paths.append(path)
    else:
        paths = [os.path.relpath(path, lbl_dir) for path in iglob(pattern)]

    return paths


def _find_mri_paths(subject='fsaverage', src=False, subjects_dir=None):
    """Find all files of an mri relevant for source transformation

    Parameters
    ----------
    subject : str
        Name of the mri subject.
    src : bool
        Include source spaces.
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

    paths['dirs'] = [bem_dir, surf_dir]

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
            path = os.path.realpath(fname.format(sub=subject))
            if not os.path.exists(path):
                raise IOError("Required file not found: %r" % path)

    # source spaces
    if src:
        paths['src'] = src = []
        basename = '{sub}-{kind}-src.fif'
        path = os.path.join(bem_dir, basename)
        p = re.compile('\A' + basename.format(sub=subject, kind='(.+)'))
        for name in os.listdir(bem_dir.format(sub=subject)):
            match = p.match(name)
            if match:
                kind = match.group(1)
                fname = path.format(sub='{sub}', kind=kind)
                src.append(fname)

    # duplicate curvature files
    paths['duplicate'] = dup = []
    path = os.path.join(surf_dir, '{name}')
    for name in ['lh.curv', 'rh.curv']:
        fname = path.format(sub='{sub}', name=name)
        dup.append(fname)

    return paths


def is_mri_subject(subject, subjects_dir=None):
    """Check whether a directory in subjects_dir is an mri subject directory

    Parameters
    ----------
    subject : str
        Name of the potential subject/directory.
    subjects_dir : None | path
        Override the SUBJECTS_DIR environment variable.

    Returns
    -------
    is_mri_subject : bool
        Whether ``subject`` is an mri subject.
    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    sdir = os.path.join(subjects_dir, subject)
    for name in ('head',):
        fname = os.path.join(sdir, 'bem', '%s-%s.fif' % (subject, name))
        if not os.path.exists(fname):
            return False

    return True


def read_mri_cfg(subject, subjects_dir=None):
    """Read the scale factor for a scaled MRI brain

    Parameters
    ----------
    subject : str
        Name of the scaled MRI subject.
    subjects_dir : None | str
        Override the SUBJECTS_DIR environment variable.

    Returns
    -------
    scaling : array, shape = () | (3, )
        The scaling factor (shape depends on whether one or three scaling
        parameters were used).
    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    fname = os.path.join(subjects_dir, subject, 'MRI scaling parameters.cfg')

    if not os.path.exists(fname):
        err = ("%r does not seem to be a scaled mri subject: %r does not "
               "exist." % (subject, fname))
        raise IOError(err)

    config = RawConfigParser()
    config.read(fname)
    n_params = config.getint("MRI Scaling", 'n_params')
    if n_params == 1:
        scale = config.getfloat("MRI Scaling", 'scale')
    elif n_params == 3:
        scale_str = config.get("MRI Scaling", 'scale')
        scale = np.array(map(float, scale_str.split()))
    else:
        raise ValueError("Invalid n_params value in MRI cfg: %i" % n_params)

    out = {'subject_from': config.get("MRI Scaling", 'subject_from'),
           'n_params': n_params, 'scale': scale}
    return out


def scale_labels(subject_to, pattern=None, overwrite=False, subject_from=None,
                 scale=None, subjects_dir=None):
    """Scale labels to match a brain that was previously created by scaling

    Parameters
    ----------
    subject_to : str
        Name of the scaled MRI subject (the destination brain).
    pattern : str | None
        Pattern for finding the labels relative to the label directory in the
        MRI subject directory (e.g., "lh.BA3a.label" will scale
        "fsaverage/label/lh.BA3a.label"; "aparc/*.label" will find all labels
        in the "fsaverage/label/aparc" directory). With None, scale all labels.
    overwrite : bool
        Overwrite any label file that already exists for subject_to (otherwise
        existsing labels are skipped).
    subject_from : None | str
        Name of the original MRI subject (the brain that was scaled to create
        subject_to). If None, the value is read from subject_to's cfg file.
    scale : None | float | array_like, shape = (3,)
        Name of the original MRI subject (the brain that was scaled to create
        subject_to). If None, the value is read from subject_to's cfg file.
    subjects_dir : None | str
        Override the SUBJECTS_DIR environment variable.
    """
    # read parameters from cfg
    if scale is None or subject_from is None:
        cfg = read_mri_cfg(subject_to, subjects_dir)
        if subject_from is None:
            subject_from = cfg['subject_from']
        if scale is None:
            scale = cfg['scale']

    # find labels
    paths = _find_label_paths(subject_from, pattern, subjects_dir)
    if not paths:
        return

    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    src_root = os.path.join(subjects_dir, subject_from, 'label')
    dst_root = os.path.join(subjects_dir, subject_to, 'label')

    # scale labels
    for fname in paths:
        dst = os.path.join(dst_root, fname)
        if not overwrite and os.path.exists(dst):
            continue

        dirname = os.path.dirname(dst)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        src = os.path.join(src_root, fname)
        l_old = read_label(src)
        pos = l_old.pos * scale
        l_new = Label(l_old.vertices, pos, l_old.values, l_old.hemi,
                      l_old.comment)
        l_new.save(dst)


def scale_mri(subject_from, subject_to, scale, src=False, overwrite=False,
              subjects_dir=None):
    """Create a scaled copy of an MRI subject

    Parameters
    ----------
    subject_from : str
        Name of the subject providing the MRI.
    subject_to : str
        New subject name for which to save the scaled MRI.
    scale : array, shape = () | (3,)
        The scaling factor (one or 3 parameters).
    src : bool
        Also scale source spaces.
    overwrite : bool
        If an MRI already exists for subject_to, overwrite it.
    subjects_dir : None | str
        Override the SUBJECTS_DIR environment variable.
    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    paths = _find_mri_paths(subject_from, src, subjects_dir=subjects_dir)
    scale = np.asarray(scale)

    # make sure we have an empty target directory
    dest = paths['s_dir'].format(sub=subject_to)
    if os.path.exists(dest):
        if overwrite:
            shutil.rmtree(dest)
        else:
            err = ("Subject directory for %s already exists: "
                   "%r" % (subject_to, dest))
            raise IOError(err)

    for dirname in paths['dirs']:
        os.makedirs(dirname.format(sub=subject_to))

    # MRI Scaling
    if np.isscalar(scale) or scale.shape == ():
        n_params = 1
        norm_scale = None
    else:
        n_params = 3
        norm_scale = 1 / scale

    # save MRI scaling parameters
    config = RawConfigParser()
    config.add_section("MRI Scaling")
    config.set("MRI Scaling", 'subject_from', subject_from)
    config.set("MRI Scaling", 'subject_to', subject_to)
    config.set("MRI Scaling", 'n_params', str(n_params))
    if n_params == 1:
        config.set("MRI Scaling", 'scale', str(scale))
    else:
        config.set("MRI Scaling", 'scale', ' '.join(map(str, scale)))
    config.set("MRI Scaling", 'version', '1')
    fname = os.path.join(dest, 'MRI scaling parameters.cfg')
    with open(fname, 'wb') as fid:
        config.write(fid)

    # surf files [in mm]
    for fname in paths['surf']:
        src = os.path.realpath(fname.format(sub=subject_from))
        dest = fname.format(sub=subject_to)
        pts, tri = read_surface(src)
        write_surface(dest, pts * scale, tri)

    # bem files [in m]
    for fname in paths['bem']:
        src = os.path.realpath(fname.format(sub=subject_from))
        surfs = read_bem_surfaces(src)
        if len(surfs) != 1:
            err = ("Bem file with more than one surface: %r" % src)
            raise NotImplementedError(err)
        surf0 = surfs[0]
        surf0['rr'] = surf0['rr'] * scale
        dest = fname.format(sub=subject_to)
        write_bem_surface(dest, surf0)

    # fiducials [in m]
    for fname in paths['fid']:
        src = os.path.realpath(fname.format(sub=subject_from))
        pts, cframe = read_fiducials(src)
        for pt in pts:
            pt['r'] = pt['r'] * scale
        dest = fname.format(sub=subject_to)
        write_fiducials(dest, pts, cframe)

    # src [in m]
    for fname in paths.get('src', ()):
        src = fname.format(sub=subject_from)
        sss = read_source_spaces(src)
        for ss in sss:
            ss['rr'] = ss['rr'] * scale
            if norm_scale is not None:
                nn = ss['nn'] * norm_scale
                norm = np.sqrt(np.sum(nn ** 2, 1))
                nn /= norm.reshape((-1, 1))
                ss['nn'] = nn
        dest = fname.format(sub=subject_to)
        write_source_spaces(dest, sss)

    # labels [in m]
    scale_labels(subject_to, subject_from=subject_from, scale=scale,
                 subjects_dir=subjects_dir)

    # duplicate files
    for fname in paths['duplicate']:
        src = fname.format(sub=subject_from)
        dest = fname.format(sub=subject_to)
        shutil.copyfile(src, dest)
