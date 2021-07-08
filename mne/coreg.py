# -*- coding: utf-8 -*-
"""Coregistration between different coordinate frames."""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import configparser
import fnmatch
from glob import glob, iglob
import os
import os.path as op
import stat
import sys
import re
import shutil
from functools import reduce

import numpy as np

from .io import read_fiducials, write_fiducials, read_info
from .io.constants import FIFF
from .label import read_label, Label
from .source_space import (add_source_space_distances, read_source_spaces,  # noqa: E501,F401
                           write_source_spaces, _read_mri_info,
                           get_mni_fiducials)
from .surface import read_surface, write_surface, _normalize_vectors
from .bem import read_bem_surfaces, write_bem_surfaces
from .transforms import (rotation, rotation3d, scaling, translation, Transform,
                         _read_fs_xfm, _write_fs_xfm, invert_transform,
                         combine_transforms, _quat_to_euler, write_trans,
                         _fit_matched_points, apply_trans,
                         rot_to_quat, _angle_between_quats)
from .utils import (get_config, get_subjects_dir, logger, pformat, verbose,
                    warn, has_nibabel)
from .viz._3d import _fiducial_coords

# some path templates
trans_fname = os.path.join('{raw_dir}', '{subject}-trans.fif')
subject_dirname = os.path.join('{subjects_dir}', '{subject}')
bem_dirname = os.path.join(subject_dirname, 'bem')
mri_dirname = os.path.join(subject_dirname, 'mri')
mri_transforms_dirname = os.path.join(subject_dirname, 'mri', 'transforms')
surf_dirname = os.path.join(subject_dirname, 'surf')
bem_fname = os.path.join(bem_dirname, "{subject}-{name}.fif")
head_bem_fname = pformat(bem_fname, name='head')
fid_fname = pformat(bem_fname, name='fiducials')
fid_fname_general = os.path.join(bem_dirname, "{head}-fiducials.fif")
src_fname = os.path.join(bem_dirname, '{subject}-{spacing}-src.fif')
_head_fnames = (os.path.join(bem_dirname, 'outer_skin.surf'),
                head_bem_fname)
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
    if isinstance(info, str):
        info = read_info(info)
    if isinstance(fiducials, str):
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


@verbose
def create_default_subject(fs_home=None, update=False, subjects_dir=None,
                           verbose=None):
    """Create an average brain subject for subjects without structural MRI.

    Create a copy of fsaverage from the Freesurfer directory in subjects_dir
    and add auxiliary files from the mne package.

    Parameters
    ----------
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
    %(verbose)s

    Notes
    -----
    When no structural MRI is available for a subject, an average brain can be
    substituted. Freesurfer comes with such an average brain model, and MNE
    comes with some auxiliary files which make coregistration easier.
    :py:func:`create_default_subject` copies the relevant
    files from Freesurfer into the current subjects_dir, and also adds the
    auxiliary files provided by MNE.
    """
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

    trans = reduce(np.dot, trans)
    return trans


_ALLOW_ANALITICAL = True


# XXX this function should be moved out of coreg as used elsewhere
def fit_matched_points(src_pts, tgt_pts, rotate=True, translate=True,
                       scale=False, tol=None, x0=None, out='trans',
                       weights=None):
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
    trans : array, shape (4, 4)
        Transformation that, if applied to src_pts, minimizes the squared
        distance to tgt_pts. Only returned if out=='trans'.
    params : array, shape (n_params, )
        A single tuple containing the rotation, translation, and scaling
        parameters in that order (as applicable).
    """
    src_pts = np.atleast_2d(src_pts)
    tgt_pts = np.atleast_2d(tgt_pts)
    if src_pts.shape != tgt_pts.shape:
        raise ValueError("src_pts and tgt_pts must have same shape (got "
                         "{}, {})".format(src_pts.shape, tgt_pts.shape))
    if weights is not None:
        weights = np.asarray(weights, src_pts.dtype)
        if weights.ndim != 1 or weights.size not in (src_pts.shape[0], 1):
            raise ValueError("weights (shape=%s) must be None or have shape "
                             "(%s,)" % (weights.shape, src_pts.shape[0],))
        weights = weights[:, np.newaxis]

    param_info = (bool(rotate), bool(translate), int(scale))
    del rotate, translate, scale

    # very common use case, rigid transformation (maybe with one scale factor,
    # with or without weighted errors)
    if param_info in ((True, True, 0), (True, True, 1)) and _ALLOW_ANALITICAL:
        src_pts = np.asarray(src_pts, float)
        tgt_pts = np.asarray(tgt_pts, float)
        if weights is not None:
            weights = np.asarray(weights, float)
        x, s = _fit_matched_points(
            src_pts, tgt_pts, weights, bool(param_info[2]))
        x[:3] = _quat_to_euler(x[:3])
        x = np.concatenate((x, [s])) if param_info[2] else x
    else:
        x = _generic_fit(src_pts, tgt_pts, param_info, weights, x0)

    # re-create the final transformation matrix
    if (tol is not None) or (out == 'trans'):
        trans = _trans_from_params(param_info, x)

    # assess the error of the solution
    if tol is not None:
        src_pts = np.hstack((src_pts, np.ones((len(src_pts), 1))))
        est_pts = np.dot(src_pts, trans.T)[:, :3]
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


def _generic_fit(src_pts, tgt_pts, param_info, weights, x0):
    from scipy.optimize import leastsq
    if param_info[1]:  # translate
        src_pts = np.hstack((src_pts, np.ones((len(src_pts), 1))))

    if param_info == (True, False, 0):
        def error(x):
            rx, ry, rz = x
            trans = rotation3d(rx, ry, rz)
            est = np.dot(src_pts, trans.T)
            d = tgt_pts - est
            if weights is not None:
                d *= weights
            return d.ravel()
        if x0 is None:
            x0 = (0, 0, 0)
    elif param_info == (True, True, 0):
        def error(x):
            rx, ry, rz, tx, ty, tz = x
            trans = np.dot(translation(tx, ty, tz), rotation(rx, ry, rz))
            est = np.dot(src_pts, trans.T)[:, :3]
            d = tgt_pts - est
            if weights is not None:
                d *= weights
            return d.ravel()
        if x0 is None:
            x0 = (0, 0, 0, 0, 0, 0)
    elif param_info == (True, True, 1):
        def error(x):
            rx, ry, rz, tx, ty, tz, s = x
            trans = reduce(np.dot, (translation(tx, ty, tz),
                                    rotation(rx, ry, rz),
                                    scaling(s, s, s)))
            est = np.dot(src_pts, trans.T)[:, :3]
            d = tgt_pts - est
            if weights is not None:
                d *= weights
            return d.ravel()
        if x0 is None:
            x0 = (0, 0, 0, 0, 0, 0, 1)
    elif param_info == (True, True, 3):
        def error(x):
            rx, ry, rz, tx, ty, tz, sx, sy, sz = x
            trans = reduce(np.dot, (translation(tx, ty, tz),
                                    rotation(rx, ry, rz),
                                    scaling(sx, sy, sz)))
            est = np.dot(src_pts, trans.T)[:, :3]
            d = tgt_pts - est
            if weights is not None:
                d *= weights
            return d.ravel()
        if x0 is None:
            x0 = (0, 0, 0, 0, 0, 0, 1, 1, 1)
    else:
        raise NotImplementedError(
            "The specified parameter combination is not implemented: "
            "rotate=%r, translate=%r, scale=%r" % param_info)

    x, _, _, _, _ = leastsq(error, x0, full_output=True)
    return x


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
    -------
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
    paths['surf'] = []
    surf_fname = os.path.join(surf_dirname, '{name}')
    surf_names = ('inflated', 'white', 'orig', 'orig_avg', 'inflated_avg',
                  'inflated_pre', 'pial', 'pial_avg', 'smoothwm', 'white_avg',
                  'seghead', 'smseghead')
    if os.getenv('_MNE_FEW_SURFACES', '') == 'true':  # for testing
        surf_names = surf_names[:4]
    for surf_name in surf_names:
        for hemi in ('lh.', 'rh.'):
            name = hemi + surf_name
            path = surf_fname.format(subjects_dir=subjects_dir,
                                     subject=subject, name=name)
            if os.path.exists(path):
                paths['surf'].append(pformat(surf_fname, name=name))
    surf_fname = os.path.join(bem_dirname, '{name}')
    surf_names = ('inner_skull.surf', 'outer_skull.surf', 'outer_skin.surf')
    for surf_name in surf_names:
        path = surf_fname.format(subjects_dir=subjects_dir,
                                 subject=subject, name=surf_name)
        if os.path.exists(path):
            paths['surf'].append(pformat(surf_fname, name=surf_name))
    del surf_names, surf_name, path, hemi

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
    del bem, path, bem_pattern, re_pattern

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

    # duplicate files (curvature and some surfaces)
    paths['duplicate'] = []
    path = os.path.join(surf_dirname, '{name}')
    surf_fname = os.path.join(surf_dirname, '{name}')
    surf_dup_names = ('curv', 'sphere', 'sphere.reg', 'sphere.reg.avg')
    for surf_dup_name in surf_dup_names:
        for hemi in ('lh.', 'rh.'):
            name = hemi + surf_dup_name
            path = surf_fname.format(subjects_dir=subjects_dir,
                                     subject=subject, name=name)
            if os.path.exists(path):
                paths['duplicate'].append(pformat(surf_fname, name=name))
    del surf_dup_name, name, path, hemi

    # transform files (talairach)
    paths['transforms'] = []
    transform_fname = os.path.join(mri_transforms_dirname, 'talairach.xfm')
    path = transform_fname.format(subjects_dir=subjects_dir, subject=subject)
    if os.path.exists(path):
        paths['transforms'].append(transform_fname)
    del transform_fname, path

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

    # find MRIs
    mri_dir = mri_dirname.format(subjects_dir=subjects_dir, subject=subject)
    fnames = fnmatch.filter(os.listdir(mri_dir), '*.mgz')
    paths['mri'] = [os.path.join(mri_dir, f) for f in fnames]

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
    uniform : bool
        Whether scaling is uniform.
    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    if (subject_from is None) != (scale is None):
        raise TypeError("Need to provide either both subject_from and scale "
                        "parameters, or neither.")

    if subject_from is None:
        cfg = read_mri_cfg(subject_to, subjects_dir)
        subject_from = cfg['subject_from']
        n_params = cfg['n_params']
        assert n_params in (1, 3)
        scale = cfg['scale']
    scale = np.atleast_1d(scale)
    if scale.ndim != 1 or scale.shape[0] not in (1, 3):
        raise ValueError("Invalid shape for scale parameer. Need scalar "
                         "or array of length 3. Got shape %s."
                         % (scale.shape,))
    n_params = len(scale)
    return subjects_dir, subject_from, scale, n_params == 1


@verbose
def scale_bem(subject_to, bem_name, subject_from=None, scale=None,
              subjects_dir=None, verbose=None):
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
    %(verbose)s
    """
    subjects_dir, subject_from, scale, uniform = \
        _scale_params(subject_to, subject_from, scale, subjects_dir)

    src = bem_fname.format(subjects_dir=subjects_dir, subject=subject_from,
                           name=bem_name)
    dst = bem_fname.format(subjects_dir=subjects_dir, subject=subject_to,
                           name=bem_name)

    if os.path.exists(dst):
        raise IOError("File already exists: %s" % dst)

    surfs = read_bem_surfaces(src)
    for surf in surfs:
        surf['rr'] *= scale
        if not uniform:
            assert len(surf['nn']) > 0
            surf['nn'] /= scale
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
        existing labels are skipped).
    subject_from : None | str
        Name of the original MRI subject (the brain that was scaled to create
        subject_to). If None, the value is read from subject_to's cfg file.
    scale : None | float | array_like, shape = (3,)
        Scaling parameter. If None, the value is read from subject_to's cfg
        file.
    subjects_dir : None | str
        Override the SUBJECTS_DIR environment variable.
    """
    subjects_dir, subject_from, scale, _ = _scale_params(
        subject_to, subject_from, scale, subjects_dir)

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


@verbose
def scale_mri(subject_from, subject_to, scale, overwrite=False,
              subjects_dir=None, skip_fiducials=False, labels=True,
              annot=False, verbose=None):
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
    %(verbose)s

    See Also
    --------
    scale_labels : Add labels to a scaled MRI.
    scale_source_space : Add a source space to a scaled MRI.
    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    paths = _find_mri_paths(subject_from, skip_fiducials, subjects_dir)
    scale = np.atleast_1d(scale)
    if scale.shape == (3,):
        if np.isclose(scale[1], scale[0]) and np.isclose(scale[2], scale[0]):
            scale = scale[0]  # speed up scaling conditionals using a singleton
    elif scale.shape != (1,):
        raise ValueError('scale must have shape (3,) or (1,), got %s'
                         % (scale.shape,))

    # make sure we have an empty target directory
    dest = subject_dirname.format(subject=subject_to,
                                  subjects_dir=subjects_dir)
    if os.path.exists(dest):
        if not overwrite:
            raise IOError("Subject directory for %s already exists: %r"
                          % (subject_to, dest))
        shutil.rmtree(dest)

    logger.debug('create empty directory structure')
    for dirname in paths['dirs']:
        dir_ = dirname.format(subject=subject_to, subjects_dir=subjects_dir)
        os.makedirs(dir_)

    logger.debug('save MRI scaling parameters')
    fname = os.path.join(dest, 'MRI scaling parameters.cfg')
    _write_mri_config(fname, subject_from, subject_to, scale)

    logger.debug('surf files [in mm]')
    for fname in paths['surf']:
        src = fname.format(subject=subject_from, subjects_dir=subjects_dir)
        src = os.path.realpath(src)
        dest = fname.format(subject=subject_to, subjects_dir=subjects_dir)
        pts, tri = read_surface(src)
        write_surface(dest, pts * scale, tri)

    logger.debug('BEM files [in m]')
    for bem_name in paths['bem']:
        scale_bem(subject_to, bem_name, subject_from, scale, subjects_dir,
                  verbose=False)

    logger.debug('fiducials [in m]')
    for fname in paths['fid']:
        src = fname.format(subject=subject_from, subjects_dir=subjects_dir)
        src = os.path.realpath(src)
        pts, cframe = read_fiducials(src, verbose=False)
        for pt in pts:
            pt['r'] = pt['r'] * scale
        dest = fname.format(subject=subject_to, subjects_dir=subjects_dir)
        write_fiducials(dest, pts, cframe, verbose=False)

    logger.debug('MRIs [nibabel]')
    os.mkdir(mri_dirname.format(subjects_dir=subjects_dir,
                                subject=subject_to))
    for fname in paths['mri']:
        mri_name = os.path.basename(fname)
        _scale_mri(subject_to, mri_name, subject_from, scale, subjects_dir)

    logger.debug('Transforms')
    for mri_name in paths['mri']:
        if mri_name.endswith('T1.mgz'):
            os.mkdir(mri_transforms_dirname.format(subjects_dir=subjects_dir,
                                                   subject=subject_to))
            for fname in paths['transforms']:
                xfm_name = os.path.basename(fname)
                _scale_xfm(subject_to, xfm_name, mri_name,
                           subject_from, scale, subjects_dir)
            break

    logger.debug('duplicate files')
    for fname in paths['duplicate']:
        src = fname.format(subject=subject_from, subjects_dir=subjects_dir)
        dest = fname.format(subject=subject_to, subjects_dir=subjects_dir)
        shutil.copyfile(src, dest)

    logger.debug('source spaces')
    for fname in paths['src']:
        src_name = os.path.basename(fname)
        scale_source_space(subject_to, src_name, subject_from, scale,
                           subjects_dir, verbose=False)

    logger.debug('labels [in m]')
    os.mkdir(os.path.join(subjects_dir, subject_to, 'label'))
    if labels:
        scale_labels(subject_to, subject_from=subject_from, scale=scale,
                     subjects_dir=subjects_dir)

    logger.debug('copy *.annot files')
    # they don't contain scale-dependent information
    if annot:
        src_pattern = os.path.join(subjects_dir, subject_from, 'label',
                                   '*.annot')
        dst_dir = os.path.join(subjects_dir, subject_to, 'label')
        for src_file in iglob(src_pattern):
            shutil.copy(src_file, dst_dir)


@verbose
def scale_source_space(subject_to, src_name, subject_from=None, scale=None,
                       subjects_dir=None, n_jobs=1, verbose=None):
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
    %(verbose)s

    Notes
    -----
    When scaling volume source spaces, the source (vertex) locations are
    scaled, but the reference to the MRI volume is left unchanged. Transforms
    are updated so that source estimates can be plotted on the original MRI
    volume.
    """
    subjects_dir, subject_from, scale, uniform = \
        _scale_params(subject_to, subject_from, scale, subjects_dir)
    # if n_params==1 scale is a scalar; if n_params==3 scale is a (3,) array

    # find the source space file names
    if src_name.isdigit():
        spacing = src_name  # spacing in mm
        src_pattern = src_fname
    else:
        match = re.match(r"(oct|ico|vol)-?(\d+)$", src_name)
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
        # additional tags for volume source spaces
        for key in ('vox_mri_t', 'src_mri_t'):
            # maintain transform to original MRI volume ss['mri_volume_name']
            if key in ss:
                ss[key]['trans'][:3] *= scale[:, np.newaxis]
        # distances and patch info
        if uniform:
            if ss['dist'] is not None:
                ss['dist'] *= scale[0]
                # Sometimes this is read-only due to how it's read
                ss['nearest_dist'] = ss['nearest_dist'] * scale
                ss['dist_limit'] = ss['dist_limit'] * scale
        else:  # non-uniform scaling
            ss['nn'] /= scale
            _normalize_vectors(ss['nn'])
            if ss['dist'] is not None:
                add_dist = True
                dist_limit = float(np.abs(sss[0]['dist_limit']))
            elif ss['nearest'] is not None:
                add_dist = True
                dist_limit = 0

    if add_dist:
        logger.info("Recomputing distances, this might take a while")
        add_source_space_distances(sss, dist_limit, n_jobs)

    write_source_spaces(dst, sss)


def _scale_mri(subject_to, mri_fname, subject_from, scale, subjects_dir):
    """Scale an MRI by setting its affine."""
    subjects_dir, subject_from, scale, _ = _scale_params(
        subject_to, subject_from, scale, subjects_dir)

    if not has_nibabel():
        warn('Skipping MRI scaling for %s, please install nibabel')
        return

    import nibabel
    fname_from = op.join(mri_dirname.format(
        subjects_dir=subjects_dir, subject=subject_from), mri_fname)
    fname_to = op.join(mri_dirname.format(
        subjects_dir=subjects_dir, subject=subject_to), mri_fname)
    img = nibabel.load(fname_from)
    zooms = np.array(img.header.get_zooms())
    zooms[[0, 2, 1]] *= scale
    img.header.set_zooms(zooms)
    # Hack to fix nibabel problems, see
    # https://github.com/nipy/nibabel/issues/619
    img._affine = img.header.get_affine()  # or could use None
    nibabel.save(img, fname_to)


def _scale_xfm(subject_to, xfm_fname, mri_name, subject_from, scale,
               subjects_dir):
    """Scale a transform."""
    subjects_dir, subject_from, scale, _ = _scale_params(
        subject_to, subject_from, scale, subjects_dir)

    # The nibabel warning should already be there in MRI step, if applicable,
    # as we only get here if T1.mgz is present (and thus a scaling was
    # attempted) so we can silently return here.
    if not has_nibabel():
        return

    fname_from = os.path.join(
        mri_transforms_dirname.format(
            subjects_dir=subjects_dir, subject=subject_from), xfm_fname)
    fname_to = op.join(
        mri_transforms_dirname.format(
            subjects_dir=subjects_dir, subject=subject_to), xfm_fname)
    assert op.isfile(fname_from), fname_from
    assert op.isdir(op.dirname(fname_to)), op.dirname(fname_to)
    # The "talairach.xfm" file stores the ras_mni transform.
    #
    # For "from" subj F, "to" subj T, F->T scaling S, some equivalent vertex
    # positions F_x and T_x in MRI (Freesurfer RAS) coords, knowing that
    # we have T_x = S @ F_x, we want to have the same MNI coords computed
    # for these vertices:
    #
    #              T_mri_mni @ T_x = F_mri_mni @ F_x
    #
    # We need to find the correct T_ras_mni (talaraich.xfm file) that yields
    # this. So we derive (where † indicates inversion):
    #
    #          T_mri_mni @ S @ F_x = F_mri_mni @ F_x
    #                T_mri_mni @ S = F_mri_mni
    #    T_ras_mni @ T_mri_ras @ S = F_ras_mni @ F_mri_ras
    #        T_ras_mni @ T_mri_ras = F_ras_mni @ F_mri_ras @ S⁻¹
    #                    T_ras_mni = F_ras_mni @ F_mri_ras @ S⁻¹ @ T_ras_mri
    #

    # prepare the scale (S) transform
    scale = np.atleast_1d(scale)
    scale = np.tile(scale, 3) if len(scale) == 1 else scale
    S = Transform('mri', 'mri', scaling(*scale))  # F_mri->T_mri

    #
    # Get the necessary transforms of the "from" subject
    #
    xfm, kind = _read_fs_xfm(fname_from)
    assert kind == 'MNI Transform File', kind
    _, _, F_mri_ras, _, _ = _read_mri_info(mri_name, units='mm')
    F_ras_mni = Transform('ras', 'mni_tal', xfm)
    del xfm

    #
    # Get the necessary transforms of the "to" subject
    #
    mri_name = op.join(mri_dirname.format(
        subjects_dir=subjects_dir, subject=subject_to), op.basename(mri_name))
    _, _, T_mri_ras, _, _ = _read_mri_info(mri_name, units='mm')
    T_ras_mri = invert_transform(T_mri_ras)
    del mri_name, T_mri_ras

    # Finally we construct as above:
    #
    #    T_ras_mni = F_ras_mni @ F_mri_ras @ S⁻¹ @ T_ras_mri
    #
    # By moving right to left through the equation.
    T_ras_mni = \
        combine_transforms(
            combine_transforms(
                combine_transforms(
                    T_ras_mri, invert_transform(S), 'ras', 'mri'),
                F_mri_ras, 'ras', 'ras'),
            F_ras_mni, 'ras', 'mni_tal')
    _write_fs_xfm(fname_to, T_ras_mni['trans'], kind)


class Coregistration(object):
    def __init__(self, info, subject, subjects_dir):
        from .gui._fiducials_gui import MRIHeadWithFiducialsModel
        self.parameters = [0., 0., 0., 0., 0., 0., 1., 1., 1.]
        self.last_parameters = list(self.parameters)

        self.n_scale_params = (0, 1, 3)
        self.n_scale_param = self.n_scale_params[0]
        self.coord_frames = ('mri', 'head')
        self.coord_frame = self.coord_frames[0]
        self.icp_iterations = 20
        self.icp_angle = 0.2
        self.icp_distance = 0.2
        self.icp_scale = 0.2
        self.icp_fid_matches = ('nearest', 'matched')
        self.icp_fid_match = self.icp_fid_matches[0]
        self.grow_hair = 0.
        self.lpa_weight = 1.
        self.nasion_weight = 10.
        self.rpa_weight = 1.
        self.hsp_weight = 1.
        self.eeg_weight = 1.
        self.hpi_weight = 1.

        self.hsp = _DigSource(info)
        self.mri = MRIHeadWithFiducialsModel(subjects_dir=subjects_dir,
                                             subject=subject)
        self.nearest_calc = self._nearest_calc_default()

    def _parameters_items_changed(self):
        # Only update our nearest-neighbor if necessary
        if self.parameters[6:9] != self.last_parameters[6:9]:
            self._update_nearest_calc()
        self.last_parameters[:] = self.parameters[:]

    @property
    def changes(self):
        new = np.array(self.parameters, float)
        old = np.array(self.last_parameters, float)
        move = np.linalg.norm(old[3:6] - new[3:6]) * 1e3
        angle = np.rad2deg(_angle_between_quats(
            rot_to_quat(rotation(*new[:3])[:3, :3]),
            rot_to_quat(rotation(*old[:3])[:3, :3])))
        percs = 100 * (new[6:] - old[6:]) / old[6:]
        return move, angle, percs

    @property
    def transformed_hsp_hpi(self):
        return apply_trans(self.hsp_trans, self.hsp.hpi_points)

    @property
    def transformed_hsp_eeg_points(self):
        return apply_trans(self.hsp_trans, self.hsp.eeg_points)

    @property
    def transformed_hsp_points(self):
        return apply_trans(self.hsp_trans, self.hsp.points)

    @property
    def hsp_trans(self):
        if self.coord_frame == 'head':
            t = np.eye(4)
        else:
            t = self.head_mri_t
        return t

    @property
    def transformed_orig_hsp_points(self):
        return apply_trans(self.hsp_trans, self.hsp._hsp_points)

    @property
    def nearest_transformed_high_res_mri_idx_orig_hsp(self):
        return self.nearest_calc.query(
            apply_trans(self.head_mri_t, self.hsp._hsp_points))[1]

    @property
    def nearest_transformed_high_res_mri_idx_hpi(self):
        return self.nearest_calc.query(
            apply_trans(self.head_mri_t, self.hsp.hpi_points))[1]

    @property
    def has_hpi_data(self):
        return (self.has_mri_data and
                len(self.nearest_transformed_high_res_mri_idx_hpi) > 0)

    @property
    def nearest_transformed_high_res_mri_idx_eeg(self):
        return self.nearest_calc.query(
            apply_trans(self.head_mri_t, self.hsp.eeg_points))[1]

    @property
    def has_eeg_data(self):
        return (self.has_mri_data and
                len(self.nearest_transformed_high_res_mri_idx_eeg) > 0)

    @property
    def nearest_transformed_high_res_mri_idx_rpa(self):
        return self.nearest_calc.query(
            apply_trans(self.head_mri_t, self.hsp.rpa))[1]

    @property
    def nearest_transformed_high_res_mri_idx_nasion(self):
        return self.nearest_calc.query(
            apply_trans(self.head_mri_t, self.hsp.nasion))[1]

    @property
    def nearest_transformed_high_res_mri_idx_lpa(self):
        return self.nearest_calc.query(
            apply_trans(self.head_mri_t, self.hsp.lpa))[1]

    @property
    def has_lpa_data(self):
        return (np.any(self.mri.lpa) and np.any(self.hsp.lpa))

    @property
    def has_nasion_data(self):
        return (np.any(self.mri.nasion) and np.any(self.hsp.nasion))

    @property
    def has_rpa_data(self):
        return (np.any(self.mri.rpa) and np.any(self.hsp.rpa))

    @property
    def head_mri_t(self):
        trans = rotation(*self.parameters[:3]).T
        trans[:3, 3] = -np.dot(trans[:3, :3], self.parameters[3:6])
        # should be the same as np.linalg.inv(self.mri_head_t)
        return trans

    def _nearest_calc_default(self):
        from .surface import _DistanceQuery
        return _DistanceQuery(
            self.processed_high_res_mri_points * self.parameters[6:9])

    @property
    def nearest_transformed_high_res_mri_idx_hsp(self):
        return self.nearest_calc.query(
            apply_trans(self.head_mri_t, self.hsp.points))[1]

    @property
    def processed_high_res_mri_points(self):
        return self._get_processed_mri_points('high')

    @property
    def processed_low_res_mri_points(self):
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

    @property
    def mri_trans_noscale(self):
        if self.coord_frame == 'head':
            t = self.mri_head_t
        else:
            t = np.eye(4)
        return t

    @property
    def mri_trans(self):
        t = self.mri_trans_noscale.copy()
        t[:, :3] *= self.parameters[6:9]
        return t

    @property
    def transformed_high_res_mri_points(self):
        points = apply_trans(self.mri_trans,
                             self.processed_high_res_mri_points)
        return points

    @property
    def has_mri_data(self):
        return len(self.transformed_high_res_mri_points) > 0

    @property
    def has_hsp_data(self):
        return (self.has_mri_data and
                len(self.nearest_transformed_high_res_mri_idx_hsp) > 0)

    @property
    def orig_hsp_point_distance(self):
        mri_points = self.transformed_high_res_mri_points[
            self.nearest_transformed_high_res_mri_idx_orig_hsp]
        hsp_points = self.transformed_orig_hsp_points
        return np.linalg.norm(mri_points - hsp_points, axis=-1)

    def fit_fiducials(self, lpa_weight=1., nasion_weight=10., rpa_weight=1.):
        self.lpa_weight = lpa_weight
        self.nasion_weight = nasion_weight
        self.rpa_weight = rpa_weight

        head_pts = np.vstack((self.hsp.lpa, self.hsp.nasion, self.hsp.rpa))
        mri_pts = np.vstack((self.mri.lpa, self.mri.nasion, self.mri.rpa))
        weights = [lpa_weight, nasion_weight, rpa_weight]
        if self.n_scale_param == 0:
            mri_pts *= self.parameters[6:9]  # not done in fit_matched_points
        x0 = np.array(self.parameters[:6 + self.n_scale_param])
        est = fit_matched_points(mri_pts, head_pts, x0=x0, out='params',
                                 scale=self.n_scale_param, weights=weights)
        if self.n_scale_param == 0:
            self.parameters[:6] = est
        else:
            self.parameters[:] = np.concatenate([est, [est[-1]] * 2])
        self._parameters_items_changed()

    def _setup_icp(self):
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
        if self.n_scale_param == 0:
            mri_pts *= self.parameters[6:9]  # not done in fit_matched_points
        return head_pts, mri_pts, weights

    def fit_icp(self, iterations=20, lpa_weight=1., nasion_weight=10.,
                rpa_weight=1.):
        self.lpa_weight = lpa_weight
        self.nasion_weight = nasion_weight
        self.rpa_weight = rpa_weight

        # Initial guess (current state)
        est = self.parameters[:[6, 7, None, 9][self.n_scale_param]]

        # Do the fits, assigning and evaluating at each step
        for _ in range(iterations):
            head_pts, mri_pts, weights = self._setup_icp()
            est = fit_matched_points(mri_pts, head_pts,
                                     scale=self.n_scale_param,
                                     x0=est, out='params', weights=weights)
            if self.n_scale_param == 0:
                self.parameters[:6] = est
            elif self.n_scale_param == 1:
                self.parameters[:] = list(est) + [est[-1]] * 2
            else:
                self.parameters[:] = est
            angle, move, scale = self.changes
            self._parameters_items_changed()
            if angle <= self.icp_angle and move <= self.icp_distance and \
                    all(scale <= self.icp_scale):
                break

    def omit_hsp_points(self, distance):
        import warnings
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

    def point_distance(self):
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

    def save_trans(self, fname):
        write_trans(fname, Transform('head', 'mri', self.head_mri_t))


class _DigSource(object):
    def __init__(self, info):
        self._info = info
        self.points_filter = None

    @property
    def n_omitted(self):
        if self.points_filter is None:
            return 0
        else:
            return np.sum(self.points_filter == False)  # noqa: E712

    @property
    def points(self):
        if self.points_filter is None:
            return self._hsp_points
        else:
            return self._hsp_points[self.points_filter]

    @property
    def _hsp_points(self):
        if not self._info or not self._info['dig']:
            return np.empty((0, 3))

        points = np.array([d['r'] for d in self._info['dig']
                           if d['kind'] == FIFF.FIFFV_POINT_EXTRA])
        points = np.empty((0, 3)) if len(points) == 0 else points
        return points

    @property
    def hpi_points(self):
        if not self._info or not self._info['dig']:
            return np.zeros((0, 3))

        out = [d['r'] for d in self._info['dig'] if
               d['kind'] == FIFF.FIFFV_POINT_HPI and
               d['coord_frame'] == FIFF.FIFFV_COORD_HEAD]
        out = np.empty((0, 3)) if len(out) == 0 else np.array(out)
        return out

    @property
    def eeg_points(self):
        if not self._info or not self._info['dig']:
            return np.empty((0, 3))

        out = [d['r'] for d in self._info['dig'] if
               d['kind'] == FIFF.FIFFV_POINT_EEG and
               d['coord_frame'] == FIFF.FIFFV_COORD_HEAD]
        out = np.empty((0, 3)) if len(out) == 0 else np.array(out)
        return out

    def _cardinal_point(self, ident):
        """Coordinates for a cardinal point."""
        if not self._info or not self._info['dig']:
            return np.zeros((1, 3))

        for d in self._info['dig']:
            if d['kind'] == FIFF.FIFFV_POINT_CARDINAL and d['ident'] == ident:
                return d['r'][None, :]
        return np.zeros((1, 3))

    @property
    def nasion(self):
        return self._cardinal_point(FIFF.FIFFV_POINT_NASION)

    @property
    def lpa(self):
        return self._cardinal_point(FIFF.FIFFV_POINT_LPA)

    @property
    def rpa(self):
        return self._cardinal_point(FIFF.FIFFV_POINT_RPA)
