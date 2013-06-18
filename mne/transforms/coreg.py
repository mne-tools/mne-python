"""Coregistration between different coordinate frames"""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import fnmatch
from glob import glob
import os
import cPickle as pickle
import re
import shutil

import logging
from mne.utils import get_config
logger = logging.getLogger('mne')

import numpy as np
from numpy import dot
from scipy.optimize import leastsq
from scipy.spatial.distance import cdist

from ..fiff.meas_info import read_fiducials, write_fiducials
from ..label import read_label, Label
from ..source_space import read_source_spaces, write_source_spaces, \
                           prepare_bem_model, setup_source_space
from ..surface import read_surface, write_surface, read_bem_surfaces, \
                      write_bem_surface
from ..utils import get_subjects_dir
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
                   "specify the fs_home parameter in you call to "
                   "create_default_subject().")
            raise ValueError(err)
    if mne_root is None:
        mne_root = get_config('MNE_ROOT', mne_root)
        if mne_root is None:
            err = ("MNE_ROOT environment variable not found. Please "
                   "specify the mne_root parameter in you call to "
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


def fit_matched_pts(src_pts, tgt_pts, rotate=True, translate=True, scale=0,
                    tol=None, x0=None, out='params'):
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

    params = (translate, rotate, scale)
    if params == (False, True, 0):
        def error(x):
            rx, ry, rz = x
            trans = rotation3d(rx, ry, rz)
            est = dot(src_pts, trans.T)
            return (tgt_pts - est).ravel()
        if x0 is None:
            x0 = (0, 0, 0)
    elif params == (False, True, 1):
        def error(x):
            rx, ry, rz , s = x
            trans = rotation3d(rx, ry, rz) * s
            est = dot(src_pts, trans.T)
            return (tgt_pts - est).ravel()
        if x0 is None:
            x0 = (0, 0, 0, 1)
    elif params == (True, True, 0):
        def error(x):
            rx, ry, rz, tx, ty, tz = x
            trans = dot(translation(tx, ty, tz), rotation(rx , ry, rz))
            est = dot(src_pts, trans.T)
            return (tgt_pts - est[:, :3]).ravel()
        if x0 is None:
            x0 = (0, 0, 0, 0, 0, 0)
    elif params == (True, True, 1):
        def error(x):
            rx, ry, rz, tx, ty, tz, s = x
            trans = dot(translation(tx, ty, tz), rotation(rx , ry, rz))
            trans = dot(scaling(s, s, s), trans)
            est = dot(src_pts, trans.T)
            return (tgt_pts - est[:, :3]).ravel()
        if x0 is None:
            x0 = (0, 0, 0, 0, 0, 0, 1)
    else:
        err = ("The specified parameter combination is not implemented: "
               "%s" % str(params))
        raise NotImplementedError(err)

    x, _, _, _, _ = leastsq(error, x0, full_output=True)

    # re-create the final transformation matrix
    if (tol is not None) or (out == 'trans'):
        if params[:2] == (False, True):
            trans = rotation(*x[:3])
            if scale == 1:
                s = x[3]
                trans = dot(trans, scaling(s, s, s))
        elif params[:2] == (True, True):
            rot = x[:3]
            transl = x[3:6]
            trans = dot(translation(*transl), rotation(*rot))
            if scale == 1:
                s = x[6]
                trans = dot(trans, scaling(s, s, s))

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


def _point_cloud_error(src_pts, tgt_pts):
    """Find the distance from each source point to its closest target point

    Parameters
    ----------
    src_pts : array, shape = (n, 3)
        Source points.
    tgt_pts : array, shape = (m, 3)
        Target points.
    """
    Y = cdist(src_pts, tgt_pts, 'euclidean')
    dist = Y.min(axis=1)
    return dist


def fit_point_cloud(src_pts, tgt_pts, rotate=True, translate=True,
                    scale=0, x0=None, lsq_args={}):
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

    Notes
    -----
    Assumes that the target points form a dense enough point cloud so that
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
    params = (translate, rotate, scale)
    if params == (False, True, 0):
        def error(x):
            rx, ry, rz = x
            trans = rotation3d(rx, ry, rz)
            est = dot(src_pts, trans.T)
            err = _point_cloud_error(est, tgt_pts)
            return err
    elif params == (False, True, 1):
        def error(x):
            rx, ry, rz, s = x
            trans = rotation3d(rx, ry, rz) * s
            est = dot(src_pts, trans.T)
            err = _point_cloud_error(est, tgt_pts)
            return err
    elif params == (False, True, 3):
        def error(x):
            rx, ry, rz, sx, sy, sz = x
            trans = rotation3d(rx, ry, rz) * [sx, sy, sz]
            est = dot(src_pts, trans.T)
            err = _point_cloud_error(est, tgt_pts)
            return err
    elif params == (True, True, 0):
        def error(x):
            rx, ry, rz, tx, ty, tz = x
            trans = dot(translation(tx, ty, tz), rotation(rx, ry, rz))
            est = dot(src_pts, trans.T)
            err = _point_cloud_error(est[:, :3], tgt_pts)
            return err
    else:
        err = ("The specified parameter combination is not implemented: "
               "%s" % str(params))
        raise NotImplementedError(err)

    est, _, info, msg, _ = leastsq(error, x0, full_output=True, **kwargs)
    logging.debug("fit_point_cloud leastsq (%r) info: %r", (msg, info))

    return est



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
            src = os.path.realpath(fname.format(sub=subject))
            if not os.path.exists(src):
                raise IOError("Required file not found: %r" % src)

    # src
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


def is_mri_subject(subject, subjects_dir):
    """Check whether a directory is an mri subject's directory
    """
    sdir = os.path.join(subjects_dir, subject)
    for name in ('head',):
        fname = os.path.join(sdir, 'bem', '%s-%s.fif' % (subject, name))
        if not os.path.exists(fname):
            return False

    return True


def read_mri_scale(subject, subjects_dir=None):
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
    fname = os.path.join(subjects_dir, subject, 'MRI scaling parameters.pickled')

    with open(fname) as fid:
        info = pickle.load(fid)

    scaling = info['scale']
    return scaling


def scale_labels(s_to, s_from='fsaverage', fname='aparc/*.label',
                 subjects_dir=None):
    """Scale labels to match a brain that was created by scaling fsaverage

    Parameters
    ----------
    s_to : str
        Name of the scaled MRI subject (the destination brain).
    s_from : str
        Name of the original MRI subject (the brain that was scaled to create
        s_to, usually "fsaverage").
    fname : str
        Name of the label relative to the label directory in the MRI subject
        directory (is expanded using glob, so it can contain "*"). For example,
        "lh.BA3a.label" will scale "fsaverage/label/lh.BA3a.label".
    subjects_dir : None | str
        Override the SUBJECTS_DIR environment variable.
    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    scale = read_mri_scale(s_to, subjects_dir)

    src_dir = os.path.join(subjects_dir, s_from, 'label')
    dst_dir = os.path.join(subjects_dir, s_to, 'label')

    os.chdir(src_dir)
    fnames = glob(fname)

    for fname in fnames:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)

        dirname = os.path.dirname(dst)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        l_old = read_label(src)
        pos = l_old.pos * scale
        l_new = Label(l_old.vertices, pos, l_old.values, l_old.hemi,
                      l_old.comment)
        l_new.save(dst)


def scale_mri(s_from, s_to, scale, overwrite=False, subjects_dir=None):
    """Create a scaled copy of an MRI subject

    Parameters
    ----------
    s_from : str
        Name of the subject providing the MRI.
    s_to : str
        New subject name for which to save the scaled MRI.
    overwrite : bool
        If an MRI already exists for s_to, overwrite it.
    subjects_dir : None | str
        Override the SUBJECTS_DIR environment variable.
    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    paths = find_mri_paths(s_from, subjects_dir)

    # make sure we have an empty target directory
    dest = paths['s_dir'].format(sub=s_to)
    if os.path.exists(dest):
        if overwrite:
            shutil.rmtree(dest)
        else:
            err = ("Subject directory for %s already exists: "
                   "%r" % (s_to, dest))
            raise IOError(err)

    for dirname in paths['dirs']:
        os.makedirs(dirname.format(sub=s_to))

    # MRI Scaling
    if np.isscalar(scale):
        norm_scale = scale
    else:
        norm_scale = 1 / scale

    # save MRI scaling parameters
    params = (('s_from', s_from), ('s_to', s_to), ('scale', scale),
              ('version', 0.1))
    fname = os.path.join(dest, 'MRI scaling parameters')
    with open(fname + '.pickled', 'w') as fid:
        pickle.dump(dict(params), fid, protocol=pickle.HIGHEST_PROTOCOL)
    info = os.linesep.join(["Scaled MRI"] + [': '.join(map(str, item))
                                             for item in params])
    with open(fname + '.txt', 'w') as fid:
        fid.write(info)


    # surf files [in mm]
    for fname in paths['surf']:
        src = os.path.realpath(fname.format(sub=s_from))
        dest = fname.format(sub=s_to)
        pts, tri = read_surface(src)
        write_surface(dest, pts * scale, tri)

    # bem files [in m]
    for fname in paths['bem']:
        src = os.path.realpath(fname.format(sub=s_from))
        surfs = read_bem_surfaces(src)
        if len(surfs) != 1:
            err = ("Bem file with more than one surface: %r" % src)
            raise NotImplementedError(err)
        surf0 = surfs[0]
        surf0['rr'] = surf0['rr'] * scale
        dest = fname.format(sub=s_to)
        write_bem_surface(dest, surf0)

    # fiducials [in m]
    for fname in paths['fid']:
        src = os.path.realpath(fname.format(sub=s_from))
        pts, cframe = read_fiducials(src)
        for pt in pts:
            pt['r'] = pt['r'] * scale
        dest = fname.format(sub=s_to)
        write_fiducials(dest, pts, cframe)

    # src [in m]
    for fname in paths['src']:
        src = fname.format(sub=s_from)
        sss = read_source_spaces(src)
        for ss in sss:
            ss['rr'] = ss['rr'] * scale
            if norm_scale is not None:
                nn = ss['nn'] * norm_scale
                norm = np.sqrt(np.sum(nn ** 2, 1))
                nn /= norm.reshape((-1, 1))
                ss['nn'] = nn
        dest = fname.format(sub=s_to)
        write_source_spaces(dest, sss)

    # labels [in m]
    for fname in paths['lbl']:
        src = fname.format(sub=s_from)
        l_old = read_label(src)
        pos = l_old.pos * scale
        l_new = Label(l_old.vertices, pos, l_old.values, l_old.hemi,
                      l_old.comment)
        dest = fname.format(sub=s_to)
        l_new.save(dest)

    # duplicate files
    for fname in paths['duplicate']:
        src = fname.format(sub=s_from)
        dest = fname.format(sub=s_to)
        shutil.copyfile(src, dest)
