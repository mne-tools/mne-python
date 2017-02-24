"""Coregistration between different coordinate frames."""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

from .externals.six.moves import configparser
from .externals.six import string_types
import fnmatch
from glob import glob, iglob
import os
import stat
import sys
import re
import shutil
from warnings import warn
from functools import reduce

import numpy as np
from numpy import dot

from .io import read_fiducials, write_fiducials, read_info
from .io.constants import FIFF
from .label import read_label, Label
from .source_space import (add_source_space_distances, read_source_spaces,
                           write_source_spaces)
from .surface import read_surface, write_surface, _normalize_vectors
from .bem import read_bem_surfaces, write_bem_surfaces
from .transforms import rotation, rotation3d, scaling, translation, Transform
from .utils import get_config, get_subjects_dir, logger, pformat
from .viz._3d import _fiducial_coords
from .externals.six.moves import zip


# some path templates
trans_fname = os.path.join('{raw_dir}', '{subject}-trans.fif')
subject_dirname = os.path.join('{subjects_dir}', '{subject}')
bem_dirname = os.path.join(subject_dirname, 'bem')
surf_dirname = os.path.join(subject_dirname, 'surf')
bem_fname = os.path.join(bem_dirname, "{subject}-{name}.fif")
head_bem_fname = pformat(bem_fname, name='head')
fid_fname = pformat(bem_fname, name='fiducials')
fid_fname_general = os.path.join(bem_dirname, "{head}-fiducials.fif")
src_fname = os.path.join(bem_dirname, '{subject}-{spacing}-src.fif')
_head_fnames = (head_bem_fname, pformat(bem_fname, name='head-medium'))
_high_res_head_fnames = (os.path.join(bem_dirname, '{subject}-head-dense.fif'),
                         os.path.join(surf_dirname, 'lh.seghead'),
                         os.path.join(surf_dirname, 'lh.smseghead'))


def _make_writable(fname):
    """Make a file writable."""
    os.chmod(fname, stat.S_IMODE(os.lstat(fname)[stat.ST_MODE]) | 128)  # write


def _make_writable_recursive(path):
    """Recursively set writable."""
    if sys.platform.startswith('win'):
        return  # can't safely set perms
    for root, dirs, files in os.walk(path, topdown=False):
        for f in dirs + files:
            _make_writable(os.path.join(root, f))


def _find_head_bem(subject, subjects_dir, high_res=False):
    """Find a high resolution head."""
    # XXX this should be refactored with mne.surface.get_head_surf ...
    fnames = _high_res_head_fnames if high_res else _head_fnames
    for fname in fnames:
        path = fname.format(subjects_dir=subjects_dir, subject=subject)
        if os.path.exists(path):
            return path


def coregister_fiducials(info, fiducials, tol=0.01):
    """Create a head-MRI transform by aligning 3 fiducial points.

    Parameters
    ----------
    info : Info
        Measurement info object with fiducials in head coordinate space.
    fiducials : str | list of dict
        Fiducials in MRI coordinate space (either path to a ``*-fiducials.fif``
        file or list of fiducials as returned by :func:`read_fiducials`.

    Returns
    -------
    trans : Transform
        The device-MRI transform.
    """
    if isinstance(info, string_types):
        info = read_info(info)
    if isinstance(fiducials, string_types):
        fiducials, coord_frame_to = read_fiducials(fiducials)
    else:
        coord_frame_to = FIFF.FIFFV_COORD_MRI
    frames_from = {d['coord_frame'] for d in info['dig']}
    if len(frames_from) > 1:
        raise ValueError("info contains fiducials from different coordinate "
                         "frames")
    else:
        coord_frame_from = frames_from.pop()
    coords_from = _fiducial_coords(info['dig'])
    coords_to = _fiducial_coords(fiducials, coord_frame_to)
    trans = fit_matched_points(coords_from, coords_to, tol=tol)
    return Transform(coord_frame_from, coord_frame_to, trans)


def create_default_subject(mne_root=None, fs_home=None, update=False,
                           subjects_dir=None):
    """Create an average brain subject for subjects without structural MRI.

    Create a copy of fsaverage from the Freesurfer directory in subjects_dir
    and add auxiliary files from the mne package.

    Parameters
    ----------
    mne_root : None
        This argument is not used anymore and will be removed in 0.15. Use
        keyword arguments to make your application forward compatible.
    fs_home : None | str
        The freesurfer home directory (only needed if FREESURFER_HOME is not
        specified as environment variable).
    update : bool
        In cases where a copy of the fsaverage brain already exists in the
        subjects_dir, this option allows to only copy files that don't already
        exist in the fsaverage directory.
    subjects_dir : None | str
        Override the SUBJECTS_DIR environment variable
        (os.environ['SUBJECTS_DIR']) as destination for the new subject.

    Notes
    -----
    When no structural MRI is available for a subject, an average brain can be
    substituted. Freesurfer comes with such an average brain model, and MNE
    comes with some auxiliary files which make coregistration easier (see
    :ref:`CACGEAFI`). :py:func:`create_default_subject` copies the relevant
    files from Freesurfer into the current subjects_dir, and also adds the
    auxiliary files provided by MNE.
    """
    if mne_root is not None:
        warn("Because files from MNE-C are not needed anymore for "
             "creat_default_subject(), the mne_root argument is deprecated "
             "and will be removed in 0.15. Please call this function with "
             "keyword arguments to make your application forward compatible.",
             DeprecationWarning)

    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    if fs_home is None:
        fs_home = get_config('FREESURFER_HOME', fs_home)
        if fs_home is None:
            raise ValueError(
                "FREESURFER_HOME environment variable not found. Please "
                "specify the fs_home parameter in your call to "
                "create_default_subject().")

    # make sure freesurfer files exist
    fs_src = os.path.join(fs_home, 'subjects', 'fsaverage')
    if not os.path.exists(fs_src):
        raise IOError('fsaverage not found at %r. Is fs_home specified '
                      'correctly?' % fs_src)
    for name in ('label', 'mri', 'surf'):
        dirname = os.path.join(fs_src, name)
        if not os.path.isdir(dirname):
            raise IOError("Freesurfer fsaverage seems to be incomplete: No "
                          "directory named %s found in %s" % (name, fs_src))

    # make sure destination does not already exist
    dest = os.path.join(subjects_dir, 'fsaverage')
    if dest == fs_src:
        raise IOError(
            "Your subjects_dir points to the freesurfer subjects_dir (%r). "
            "The default subject can not be created in the freesurfer "
            "installation directory; please specify a different "
            "subjects_dir." % subjects_dir)
    elif (not update) and os.path.exists(dest):
        raise IOError(
            "Can not create fsaverage because %r already exists in "
            "subjects_dir %r. Delete or rename the existing fsaverage "
            "subject folder." % ('fsaverage', subjects_dir))

    # copy fsaverage from freesurfer
    logger.info("Copying fsaverage subject from freesurfer directory...")
    if (not update) or not os.path.exists(dest):
        shutil.copytree(fs_src, dest)
        _make_writable_recursive(dest)

    # copy files from mne
    source_fname = os.path.join(os.path.dirname(__file__), 'data', 'fsaverage',
                                'fsaverage-%s.fif')
    dest_bem = os.path.join(dest, 'bem')
    if not os.path.exists(dest_bem):
        os.mkdir(dest_bem)
    logger.info("Copying auxiliary fsaverage files from mne...")
    dest_fname = os.path.join(dest_bem, 'fsaverage-%s.fif')
    _make_writable_recursive(dest_bem)
    for name in ('fiducials', 'head', 'inner_skull-bem', 'trans'):
        if not os.path.exists(dest_fname % name):
            shutil.copy(source_fname % name, dest_bem)


def _decimate_points(pts, res=10):
    """Decimate the number of points using a voxel grid.

    Create a voxel grid with a specified resolution and retain at most one
    point per voxel. For each voxel, the point closest to its center is
    retained.

    Parameters
    ----------
    pts : array, shape (n_points, 3)
        The points making up the head shape.
    res : scalar
        The resolution of the voxel space (side length of each voxel).

    Returns
    -------
    pts : array, shape = (n_points, 3)
        The decimated points.
    """
    from scipy.spatial.distance import cdist
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
    """Convert transformation parameters into a transformation matrix.

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


def fit_matched_points(src_pts, tgt_pts, rotate=True, translate=True,
                       scale=False, tol=None, x0=None, out='trans'):
    """Find a transform between matched sets of points.

    This minimizes the squared distance between two matching sets of points.

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
    scale : bool
        Number of scaling parameters. With False, points are not scaled. With
        True, points are scaled by the same factor along all axes.
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
    from scipy.optimize import leastsq
    src_pts = np.atleast_2d(src_pts)
    tgt_pts = np.atleast_2d(tgt_pts)
    if src_pts.shape != tgt_pts.shape:
        raise ValueError("src_pts and tgt_pts must have same shape (got "
                         "{0}, {1})".format(src_pts.shape, tgt_pts.shape))

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
        raise NotImplementedError(
            "The specified parameter combination is not implemented: "
            "rotate=%r, translate=%r, scale=%r" % param_info)

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
        raise ValueError("Invalid out parameter: %r. Needs to be 'params' or "
                         "'trans'." % out)


def _point_cloud_error(src_pts, tgt_pts):
    """Find the distance from each source point to its closest target point.

    Parameters
    ----------
    src_pts : array, shape = (n, 3)
        Source points.
    tgt_pts : array, shape = (m, 3)
        Target points.

    Returns
    -------
    dist : array, shape = (n, )
        For each point in ``src_pts``, the distance to the closest point in
        ``tgt_pts``.
    """
    from scipy.spatial.distance import cdist
    Y = cdist(src_pts, tgt_pts, 'euclidean')
    dist = Y.min(axis=1)
    return dist


def _point_cloud_error_balltree(src_pts, tgt_tree):
    """Find the distance from each source point to its closest target point.

    Uses sklearn.neighbors.BallTree for greater efficiency

    Parameters
    ----------
    src_pts : array, shape = (n, 3)
        Source points.
    tgt_tree : sklearn.neighbors.BallTree
        BallTree of the target points.

    Returns
    -------
    dist : array, shape = (n, )
        For each point in ``src_pts``, the distance to the closest point in
        ``tgt_pts``.
    """
    dist, _ = tgt_tree.query(src_pts)
    return dist.ravel()


def fit_point_cloud(src_pts, tgt_pts, rotate=True, translate=True,
                    scale=0, x0=None, leastsq_args={}, out='params'):
    """Find a transform between unmatched sets of points.

    This minimizes the squared distance from each source point to its closest
    target point, using :func:`scipy.optimize.leastsq` to find a
    transformation using rotation, translation, and scaling (in that order).

    Parameters
    ----------
    src_pts : array, shape = (n, 3)
        Points to which the transform should be applied.
    tgt_pts : array, shape = (m, 3)
        Points to which src_pts should be fitted. Each point in tgt_pts should
        correspond to the point in src_pts with the same index.
    rotate : bool
        Allow rotation of the ``src_pts``.
    translate : bool
        Allow translation of the ``src_pts``.
    scale : 0 | 1 | 3
        Number of scaling parameters. With 0, points are not scaled. With 1,
        points are scaled by the same factor along all axes. With 3, points are
        scaled by a separate factor along each axis.
    x0 : None | tuple
        Initial values for the fit parameters.
    leastsq_args : dict
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
    from scipy.optimize import leastsq
    kwargs = {'epsfcn': 0.01}
    kwargs.update(leastsq_args)

    # assert correct argument types
    src_pts = np.atleast_2d(src_pts)
    tgt_pts = np.atleast_2d(tgt_pts)
    translate = bool(translate)
    rotate = bool(rotate)
    scale = int(scale)

    if translate:
        src_pts = np.hstack((src_pts, np.ones((len(src_pts), 1))))

    try:
        from sklearn.neighbors import BallTree
        tgt_pts = BallTree(tgt_pts)
        errfunc = _point_cloud_error_balltree
    except ImportError:
        warn("Sklearn could not be imported. Fitting points will be slower. "
             "To improve performance, install the sklearn module.")
        errfunc = _point_cloud_error

    # for efficiency, define parameter specific error function
    param_info = (rotate, translate, scale)
    if param_info == (True, False, 0):
        x0 = x0 or (0, 0, 0)

        def error(x):
            rx, ry, rz = x
            trans = rotation3d(rx, ry, rz)
            est = dot(src_pts, trans.T)
            err = errfunc(est, tgt_pts)
            return err
    elif param_info == (True, False, 1):
        x0 = x0 or (0, 0, 0, 1)

        def error(x):
            rx, ry, rz, s = x
            trans = rotation3d(rx, ry, rz) * s
            est = dot(src_pts, trans.T)
            err = errfunc(est, tgt_pts)
            return err
    elif param_info == (True, False, 3):
        x0 = x0 or (0, 0, 0, 1, 1, 1)

        def error(x):
            rx, ry, rz, sx, sy, sz = x
            trans = rotation3d(rx, ry, rz) * [sx, sy, sz]
            est = dot(src_pts, trans.T)
            err = errfunc(est, tgt_pts)
            return err
    elif param_info == (True, True, 0):
        x0 = x0 or (0, 0, 0, 0, 0, 0)

        def error(x):
            rx, ry, rz, tx, ty, tz = x
            trans = dot(translation(tx, ty, tz), rotation(rx, ry, rz))
            est = dot(src_pts, trans.T)
            err = errfunc(est[:, :3], tgt_pts)
            return err
    else:
        raise NotImplementedError(
            "The specified parameter combination is not implemented: "
            "rotate=%r, translate=%r, scale=%r" % param_info)

    est, _, info, msg, _ = leastsq(error, x0, full_output=True, **kwargs)
    logger.debug("fit_point_cloud leastsq (%i calls) info: %s", info['nfev'],
                 msg)

    if out == 'params':
        return est
    elif out == 'trans':
        return _trans_from_params(param_info, est)
    else:
        raise ValueError("Invalid out parameter: %r. Needs to be 'params' or "
                         "'trans'." % out)


def _find_label_paths(subject='fsaverage', pattern=None, subjects_dir=None):
    """Find paths to label files in a subject's label directory.

    Parameters
    ----------
    subject : str
        Name of the mri subject.
    pattern : str | None
        Pattern for finding the labels relative to the label directory in the
        MRI subject directory (e.g., "aparc/*.label" will find all labels
        in the "subject/label/aparc" directory). With None, find all labels.
    subjects_dir : None | str
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


def _find_mri_paths(subject, skip_fiducials, subjects_dir):
    """Find all files of an mri relevant for source transformation.

    Parameters
    ----------
    subject : str
        Name of the mri subject.
    skip_fiducials : bool
        Do not scale the MRI fiducials. If False, an IOError will be raised
        if no fiducials file can be found.
    subjects_dir : None | str
        Override the SUBJECTS_DIR environment variable
        (sys.environ['SUBJECTS_DIR'])

    Returns
    -------
    paths : dict
        Dictionary whose keys are relevant file type names (str), and whose
        values are lists of paths.
    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    paths = {}

    # directories to create
    paths['dirs'] = [bem_dirname, surf_dirname]

    # surf/ files
    paths['surf'] = surf = []
    surf_fname = os.path.join(surf_dirname, '{name}')
    surf_names = ('inflated', 'sphere', 'sphere.reg', 'white', 'orig',
                  'orig_avg', 'inflated_avg', 'inflated_pre', 'pial',
                  'pial_avg', 'smoothwm', 'white_avg', 'sphere.reg.avg')
    if os.getenv('_MNE_FEW_SURFACES', '') == 'true':  # for testing
        surf_names = surf_names[:4]
    for surf_name in surf_names:
        for hemi in ('lh.', 'rh.'):
            name = hemi + surf_name
            path = surf_fname.format(subjects_dir=subjects_dir,
                                     subject=subject, name=name)
            if os.path.exists(path):
                surf.append(pformat(surf_fname, name=name))

    # BEM files
    paths['bem'] = bem = []
    path = head_bem_fname.format(subjects_dir=subjects_dir, subject=subject)
    if os.path.exists(path):
        bem.append('head')
    bem_pattern = pformat(bem_fname, subjects_dir=subjects_dir,
                          subject=subject, name='*-bem')
    re_pattern = pformat(bem_fname, subjects_dir=subjects_dir, subject=subject,
                         name='(.+)').replace('\\', '\\\\')
    for path in iglob(bem_pattern):
        match = re.match(re_pattern, path)
        name = match.group(1)
        bem.append(name)

    # fiducials
    if skip_fiducials:
        paths['fid'] = []
    else:
        paths['fid'] = _find_fiducials_files(subject, subjects_dir)
        # check that we found at least one
        if len(paths['fid']) == 0:
            raise IOError("No fiducials file found for %s. The fiducials "
                          "file should be named "
                          "{subject}/bem/{subject}-fiducials.fif. In "
                          "order to scale an MRI without fiducials set "
                          "skip_fiducials=True." % subject)

    # duplicate curvature files
    paths['duplicate'] = dup = []
    path = os.path.join(surf_dirname, '{name}')
    for name in ['lh.curv', 'rh.curv']:
        fname = pformat(path, name=name)
        dup.append(fname)

    # check presence of required files
    for ftype in ['surf', 'duplicate']:
        for fname in paths[ftype]:
            path = fname.format(subjects_dir=subjects_dir, subject=subject)
            path = os.path.realpath(path)
            if not os.path.exists(path):
                raise IOError("Required file not found: %r" % path)

    # find source space files
    paths['src'] = src = []
    bem_dir = bem_dirname.format(subjects_dir=subjects_dir, subject=subject)
    fnames = fnmatch.filter(os.listdir(bem_dir), '*-src.fif')
    prefix = subject + '-'
    for fname in fnames:
        if fname.startswith(prefix):
            fname = "{subject}-%s" % fname[len(prefix):]
        path = os.path.join(bem_dirname, fname)
        src.append(path)

    return paths


def _find_fiducials_files(subject, subjects_dir):
    """Find fiducial files."""
    fid = []
    # standard fiducials
    if os.path.exists(fid_fname.format(subjects_dir=subjects_dir,
                                       subject=subject)):
        fid.append(fid_fname)
    # fiducials with subject name
    pattern = pformat(fid_fname_general, subjects_dir=subjects_dir,
                      subject=subject, head='*')
    regex = pformat(fid_fname_general, subjects_dir=subjects_dir,
                    subject=subject, head='(.+)').replace('\\', '\\\\')
    for path in iglob(pattern):
        match = re.match(regex, path)
        head = match.group(1).replace(subject, '{subject}')
        fid.append(pformat(fid_fname_general, head=head))
    return fid


def _is_mri_subject(subject, subjects_dir=None):
    """Check whether a directory in subjects_dir is an mri subject directory.

    Parameters
    ----------
    subject : str
        Name of the potential subject/directory.
    subjects_dir : None | str
        Override the SUBJECTS_DIR environment variable.

    Returns
    -------
    is_mri_subject : bool
        Whether ``subject`` is an mri subject.
    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    return bool(_find_head_bem(subject, subjects_dir) or
                _find_head_bem(subject, subjects_dir, high_res=True))


def _is_scaled_mri_subject(subject, subjects_dir=None):
    """Check whether a directory in subjects_dir is a scaled mri subject.

    Parameters
    ----------
    subject : str
        Name of the potential subject/directory.
    subjects_dir : None | str
        Override the SUBJECTS_DIR environment variable.

    Returns
    -------
    is_scaled_mri_subject : bool
        Whether ``subject`` is a scaled mri subject.
    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    if not _is_mri_subject(subject, subjects_dir):
        return False
    fname = os.path.join(subjects_dir, subject, 'MRI scaling parameters.cfg')
    return os.path.exists(fname)


def _mri_subject_has_bem(subject, subjects_dir=None):
    """Check whether an mri subject has a file matching the bem pattern.

    Parameters
    ----------
    subject : str
        Name of the subject.
    subjects_dir : None | str
        Override the SUBJECTS_DIR environment variable.

    Returns
    -------
    has_bem_file : bool
        Whether ``subject`` has a bem file.
    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    pattern = bem_fname.format(subjects_dir=subjects_dir, subject=subject,
                               name='*-bem')
    fnames = glob(pattern)
    return bool(len(fnames))


def read_mri_cfg(subject, subjects_dir=None):
    """Read information from the cfg file of a scaled MRI brain.

    Parameters
    ----------
    subject : str
        Name of the scaled MRI subject.
    subjects_dir : None | str
        Override the SUBJECTS_DIR environment variable.

    Returns
    -------
    cfg : dict
        Dictionary with entries from the MRI's cfg file.
    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    fname = os.path.join(subjects_dir, subject, 'MRI scaling parameters.cfg')

    if not os.path.exists(fname):
        raise IOError("%r does not seem to be a scaled mri subject: %r does "
                      "not exist." % (subject, fname))

    logger.info("Reading MRI cfg file %s" % fname)
    config = configparser.RawConfigParser()
    config.read(fname)
    n_params = config.getint("MRI Scaling", 'n_params')
    if n_params == 1:
        scale = config.getfloat("MRI Scaling", 'scale')
    elif n_params == 3:
        scale_str = config.get("MRI Scaling", 'scale')
        scale = np.array([float(s) for s in scale_str.split()])
    else:
        raise ValueError("Invalid n_params value in MRI cfg: %i" % n_params)

    out = {'subject_from': config.get("MRI Scaling", 'subject_from'),
           'n_params': n_params, 'scale': scale}
    return out


def _write_mri_config(fname, subject_from, subject_to, scale):
    """Write the cfg file describing a scaled MRI subject.

    Parameters
    ----------
    fname : str
        Target file.
    subject_from : str
        Name of the source MRI subject.
    subject_to : str
        Name of the scaled MRI subject.
    scale : float | array_like, shape = (3,)
        The scaling parameter.
    """
    scale = np.asarray(scale)
    if np.isscalar(scale) or scale.shape == ():
        n_params = 1
    else:
        n_params = 3

    config = configparser.RawConfigParser()
    config.add_section("MRI Scaling")
    config.set("MRI Scaling", 'subject_from', subject_from)
    config.set("MRI Scaling", 'subject_to', subject_to)
    config.set("MRI Scaling", 'n_params', str(n_params))
    if n_params == 1:
        config.set("MRI Scaling", 'scale', str(scale))
    else:
        config.set("MRI Scaling", 'scale', ' '.join([str(s) for s in scale]))
    config.set("MRI Scaling", 'version', '1')
    with open(fname, 'w') as fid:
        config.write(fid)


def _scale_params(subject_to, subject_from, scale, subjects_dir):
    """Assemble parameters for scaling.

    Returns
    -------
    subjects_dir : str
        Subjects directory.
    subject_from : str
        Name of the source subject.
    scale : array
        Scaling factor, either shape=() for uniform scaling or shape=(3,) for
        non-uniform scaling.
    nn_scale : None | array
        Scaling factor for surface normal. If scaling is uniform, normals are
        unchanged and nn_scale is None. If scaling is non-uniform nn_scale is
        an array of shape (3,).
    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    if (subject_from is None) != (scale is None):
        raise TypeError("Need to provide either both subject_from and scale "
                        "parameters, or neither.")

    if subject_from is None:
        cfg = read_mri_cfg(subject_to, subjects_dir)
        subject_from = cfg['subject_from']
        n_params = cfg['n_params']
        scale = cfg['scale']
    else:
        scale = np.asarray(scale)
        if scale.ndim == 0:
            n_params = 1
        elif scale.shape == (3,):
            n_params = 3
        else:
            raise ValueError("Invalid shape for scale parameer. Need scalar "
                             "or array of length 3. Got %s." % str(scale))

    # prepare scaling parameter for normals
    if n_params == 1:
        nn_scale = None
    elif n_params == 3:
        nn_scale = 1. / scale
    else:
        raise RuntimeError("Invalid n_params value: %s" % repr(n_params))

    return subjects_dir, subject_from, scale, nn_scale


def scale_bem(subject_to, bem_name, subject_from=None, scale=None,
              subjects_dir=None):
    """Scale a bem file.

    Parameters
    ----------
    subject_to : str
        Name of the scaled MRI subject (the destination mri subject).
    bem_name : str
        Name of the bem file. For example, to scale
        ``fsaverage-inner_skull-bem.fif``, the bem_name would be
        "inner_skull-bem".
    subject_from : None | str
        The subject from which to read the source space. If None, subject_from
        is read from subject_to's config file.
    scale : None | float | array, shape = (3,)
        Scaling factor. Has to be specified if subjects_from is specified,
        otherwise it is read from subject_to's config file.
    subjects_dir : None | str
        Override the SUBJECTS_DIR environment variable.
    """
    subjects_dir, subject_from, scale, nn_scale = \
        _scale_params(subject_to, subject_from, scale, subjects_dir)

    src = bem_fname.format(subjects_dir=subjects_dir, subject=subject_from,
                           name=bem_name)
    dst = bem_fname.format(subjects_dir=subjects_dir, subject=subject_to,
                           name=bem_name)

    if os.path.exists(dst):
        raise IOError("File alredy exists: %s" % dst)

    surfs = read_bem_surfaces(src)
    for surf in surfs:
        surf['rr'] *= scale
        if nn_scale is not None:
            surf['nn'] *= nn_scale
            _normalize_vectors(surf['nn'])
    write_bem_surfaces(dst, surfs)


def scale_labels(subject_to, pattern=None, overwrite=False, subject_from=None,
                 scale=None, subjects_dir=None):
    r"""Scale labels to match a brain that was previously created by scaling.

    Parameters
    ----------
    subject_to : str
        Name of the scaled MRI subject (the destination brain).
    pattern : str | None
        Pattern for finding the labels relative to the label directory in the
        MRI subject directory (e.g., "lh.BA3a.label" will scale
        "fsaverage/label/lh.BA3a.label"; "aparc/\*.label" will find all labels
        in the "fsaverage/label/aparc" directory). With None, scale all labels.
    overwrite : bool
        Overwrite any label file that already exists for subject_to (otherwise
        existsing labels are skipped).
    subject_from : None | str
        Name of the original MRI subject (the brain that was scaled to create
        subject_to). If None, the value is read from subject_to's cfg file.
    scale : None | float | array_like, shape = (3,)
        Scaling parameter. If None, the value is read from subject_to's cfg
        file.
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
                      l_old.comment, subject=subject_to)
        l_new.save(dst)


def scale_mri(subject_from, subject_to, scale, overwrite=False,
              subjects_dir=None, skip_fiducials=False, labels=True,
              annot=False):
    """Create a scaled copy of an MRI subject.

    Parameters
    ----------
    subject_from : str
        Name of the subject providing the MRI.
    subject_to : str
        New subject name for which to save the scaled MRI.
    scale : float | array_like, shape = (3,)
        The scaling factor (one or 3 parameters).
    overwrite : bool
        If an MRI already exists for subject_to, overwrite it.
    subjects_dir : None | str
        Override the SUBJECTS_DIR environment variable.
    skip_fiducials : bool
        Do not scale the MRI fiducials. If False (default), an IOError will be
        raised if no fiducials file can be found.
    labels : bool
        Also scale all labels (default True).
    annot : bool
        Copy ``*.annot`` files to the new location (default False).

    See Also
    --------
    scale_labels : add labels to a scaled MRI
    scale_source_space : add a source space to a scaled MRI
    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    paths = _find_mri_paths(subject_from, skip_fiducials, subjects_dir)
    scale = np.asarray(scale)

    # make sure we have an empty target directory
    dest = subject_dirname.format(subject=subject_to,
                                  subjects_dir=subjects_dir)
    if os.path.exists(dest):
        if overwrite:
            shutil.rmtree(dest)
        else:
            raise IOError("Subject directory for %s already exists: %r"
                          % (subject_to, dest))

    # create empty directory structure
    for dirname in paths['dirs']:
        dir_ = dirname.format(subject=subject_to, subjects_dir=subjects_dir)
        os.makedirs(dir_)

    # save MRI scaling parameters
    fname = os.path.join(dest, 'MRI scaling parameters.cfg')
    _write_mri_config(fname, subject_from, subject_to, scale)

    # surf files [in mm]
    for fname in paths['surf']:
        src = fname.format(subject=subject_from, subjects_dir=subjects_dir)
        src = os.path.realpath(src)
        dest = fname.format(subject=subject_to, subjects_dir=subjects_dir)
        pts, tri = read_surface(src)
        write_surface(dest, pts * scale, tri)

    # BEM files [in m]
    for bem_name in paths['bem']:
        scale_bem(subject_to, bem_name, subject_from, scale, subjects_dir)

    # fiducials [in m]
    for fname in paths['fid']:
        src = fname.format(subject=subject_from, subjects_dir=subjects_dir)
        src = os.path.realpath(src)
        pts, cframe = read_fiducials(src)
        for pt in pts:
            pt['r'] = pt['r'] * scale
        dest = fname.format(subject=subject_to, subjects_dir=subjects_dir)
        write_fiducials(dest, pts, cframe)

    # duplicate files
    for fname in paths['duplicate']:
        src = fname.format(subject=subject_from, subjects_dir=subjects_dir)
        dest = fname.format(subject=subject_to, subjects_dir=subjects_dir)
        shutil.copyfile(src, dest)

    # source spaces
    for fname in paths['src']:
        src_name = os.path.basename(fname)
        scale_source_space(subject_to, src_name, subject_from, scale,
                           subjects_dir)

    # labels [in m]
    os.mkdir(os.path.join(subjects_dir, subject_to, 'label'))
    if labels:
        scale_labels(subject_to, subject_from=subject_from, scale=scale,
                     subjects_dir=subjects_dir)

    # copy *.annot files (they don't contain scale-dependent information)
    if annot:
        src_pattern = os.path.join(subjects_dir, subject_from, 'label',
                                   '*.annot')
        dst_dir = os.path.join(subjects_dir, subject_to, 'label')
        for src_file in iglob(src_pattern):
            shutil.copy(src_file, dst_dir)


def scale_source_space(subject_to, src_name, subject_from=None, scale=None,
                       subjects_dir=None, n_jobs=1):
    """Scale a source space for an mri created with scale_mri().

    Parameters
    ----------
    subject_to : str
        Name of the scaled MRI subject (the destination mri subject).
    src_name : str
        Source space name. Can be a spacing parameter (e.g., ``'7'``,
        ``'ico4'``, ``'oct6'``) or a file name of a source space file relative
        to the bem directory; if the file name contains the subject name, it
        should be indicated as "{subject}" in ``src_name`` (e.g.,
        ``"{subject}-my_source_space-src.fif"``).
    subject_from : None | str
        The subject from which to read the source space. If None, subject_from
        is read from subject_to's config file.
    scale : None | float | array, shape = (3,)
        Scaling factor. Has to be specified if subjects_from is specified,
        otherwise it is read from subject_to's config file.
    subjects_dir : None | str
        Override the SUBJECTS_DIR environment variable.
    n_jobs : int
        Number of jobs to run in parallel if recomputing distances (only
        applies if scale is an array of length 3, and will not use more cores
        than there are source spaces).
    """
    subjects_dir, subject_from, scale, nn_scale = \
        _scale_params(subject_to, subject_from, scale, subjects_dir)

    # find the source space file names
    if src_name.isdigit():
        spacing = src_name  # spacing in mm
        src_pattern = src_fname
    else:
        match = re.match("(oct|ico)-?(\d+)$", src_name)
        if match:
            spacing = '-'.join(match.groups())
            src_pattern = src_fname
        else:
            spacing = None
            src_pattern = os.path.join(bem_dirname, src_name)

    src = src_pattern.format(subjects_dir=subjects_dir, subject=subject_from,
                             spacing=spacing)
    dst = src_pattern.format(subjects_dir=subjects_dir, subject=subject_to,
                             spacing=spacing)

    # read and scale the source space [in m]
    sss = read_source_spaces(src)
    logger.info("scaling source space %s:  %s -> %s", spacing, subject_from,
                subject_to)
    logger.info("Scale factor: %s", scale)
    add_dist = False
    for ss in sss:
        ss['subject_his_id'] = subject_to
        ss['rr'] *= scale

        # distances and patch info
        if nn_scale is None:  # i.e. uniform scaling
            if ss['dist'] is not None:
                ss['dist'] *= scale
                ss['nearest_dist'] *= scale
                ss['dist_limit'] *= scale
        else:  # non-uniform scaling
            ss['nn'] *= nn_scale
            _normalize_vectors(ss['nn'])
            if ss['dist'] is not None:
                add_dist = True

    if add_dist:
        logger.info("Recomputing distances, this might take a while")
        dist_limit = np.asscalar(sss[0]['dist_limit'])
        add_source_space_distances(sss, dist_limit, n_jobs)

    write_source_spaces(dst, sss)
