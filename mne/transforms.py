# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os
from os import path as op
import glob
import numpy as np
from numpy import sin, cos
from scipy import linalg

from .io.constants import FIFF
from .io.open import fiff_open
from .io.tag import read_tag
from .io.write import start_file, end_file, write_coord_trans
from .utils import check_fname, logger
from .externals.six import string_types


# transformation from anterior/left/superior coordinate system to
# right/anterior/superior:
als_ras_trans = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0],
                          [0, 0, 0, 1]])
# simultaneously convert [m] to [mm]:
als_ras_trans_mm = als_ras_trans * [0.001, 0.001, 0.001, 1]


_str_to_frame = dict(meg=FIFF.FIFFV_COORD_DEVICE,
                     mri=FIFF.FIFFV_COORD_MRI,
                     mri_voxel=FIFF.FIFFV_MNE_COORD_MRI_VOXEL,
                     head=FIFF.FIFFV_COORD_HEAD,
                     mni_tal=FIFF.FIFFV_MNE_COORD_MNI_TAL,
                     ras=FIFF.FIFFV_MNE_COORD_RAS,
                     fs_tal=FIFF.FIFFV_MNE_COORD_FS_TAL,
                     ctf_head=FIFF.FIFFV_MNE_COORD_CTF_HEAD,
                     ctf_meg=FIFF.FIFFV_MNE_COORD_CTF_DEVICE)
_frame_to_str = dict((val, key) for key, val in _str_to_frame.items())

_verbose_frames = {FIFF.FIFFV_COORD_UNKNOWN: 'unknown',
                   FIFF.FIFFV_COORD_DEVICE: 'MEG device',
                   FIFF.FIFFV_COORD_ISOTRAK: 'isotrak',
                   FIFF.FIFFV_COORD_HPI: 'hpi',
                   FIFF.FIFFV_COORD_HEAD: 'head',
                   FIFF.FIFFV_COORD_MRI: 'MRI (surface RAS)',
                   FIFF.FIFFV_MNE_COORD_MRI_VOXEL: 'MRI voxel',
                   FIFF.FIFFV_COORD_MRI_SLICE: 'MRI slice',
                   FIFF.FIFFV_COORD_MRI_DISPLAY: 'MRI display',
                   FIFF.FIFFV_MNE_COORD_CTF_DEVICE: 'CTF MEG device',
                   FIFF.FIFFV_MNE_COORD_CTF_HEAD: 'CTF/4D/KIT head',
                   FIFF.FIFFV_MNE_COORD_RAS: 'RAS (non-zero origin)',
                   FIFF.FIFFV_MNE_COORD_MNI_TAL: 'MNI Talairach',
                   FIFF.FIFFV_MNE_COORD_FS_TAL_GTZ: 'Talairach (MNI z > 0)',
                   FIFF.FIFFV_MNE_COORD_FS_TAL_LTZ: 'Talairach (MNI z < 0)',
                   -1: 'unknown'}


def _to_const(cf):
    """Helper to convert string or int coord frame into int"""
    if isinstance(cf, string_types):
        if cf not in _str_to_frame:
            raise ValueError('Unknown cf %s' % cf)
        cf = _str_to_frame[cf]
    elif not isinstance(cf, int):
        raise TypeError('cf must be str or int, not %s' % type(cf))
    return cf


class Transform(dict):
    """A transform

    Parameters
    ----------
    fro : str | int
        The starting coordinate frame.
    to : str | int
        The ending coordinate frame.
    trans : array-like, shape (4, 4)
        The transformation matrix.
    """
    def __init__(self, fro, to, trans):
        super(Transform, self).__init__()
        # we could add some better sanity checks here
        fro = _to_const(fro)
        to = _to_const(to)
        trans = np.asarray(trans, dtype=np.float64)
        if trans.shape != (4, 4):
            raise ValueError('Transformation must be shape (4, 4) not %s'
                             % (trans.shape,))
        self['from'] = fro
        self['to'] = to
        self['trans'] = trans

    def __repr__(self):
        return ('<Transform  |  %s->%s>\n%s'
                % (_coord_frame_name(self['from']),
                   _coord_frame_name(self['to']), self['trans']))

    @property
    def from_str(self):
        return _coord_frame_name(self['from'])

    @property
    def to_str(self):
        return _coord_frame_name(self['to'])


def _coord_frame_name(cframe):
    """Map integers to human-readable (verbose) names"""
    return _verbose_frames.get(int(cframe), 'unknown')


def _print_coord_trans(t, prefix='Coordinate transformation: '):
    logger.info(prefix + '%s -> %s'
                % (_coord_frame_name(t['from']), _coord_frame_name(t['to'])))
    for ti, tt in enumerate(t['trans']):
        scale = 1000. if ti != 3 else 1.
        text = ' mm' if ti != 3 else ''
        logger.info('    % 8.6f % 8.6f % 8.6f    %7.2f%s' %
                    (tt[0], tt[1], tt[2], scale * tt[3], text))


def _find_trans(subject, subjects_dir=None):
    if subject is None:
        if 'SUBJECT' in os.environ:
            subject = os.environ['SUBJECT']
        else:
            raise ValueError('SUBJECT environment variable not set')

    trans_fnames = glob.glob(os.path.join(subjects_dir, subject,
                                          '*-trans.fif'))
    if len(trans_fnames) < 1:
        raise RuntimeError('Could not find the transformation for '
                           '{subject}'.format(subject=subject))
    elif len(trans_fnames) > 1:
        raise RuntimeError('Found multiple transformations for '
                           '{subject}'.format(subject=subject))
    return trans_fnames[0]


def apply_trans(trans, pts, move=True):
    """Apply a transform matrix to an array of points

    Parameters
    ----------
    trans : array, shape = (4, 4) | instance of Transform
        Transform matrix.
    pts : array, shape = (3,) | (n, 3)
        Array with coordinates for one or n points.
    move : bool
        If True (default), apply translation.

    Returns
    -------
    transformed_pts : shape = (3,) | (n, 3)
        Transformed point(s).
    """
    if isinstance(trans, dict):
        trans = trans['trans']
    trans = np.asarray(trans)
    pts = np.asarray(pts)
    if pts.size == 0:
        return pts.copy()

    # apply rotation & scale
    out_pts = np.dot(pts, trans[:3, :3].T)
    # apply translation
    if move is True:
        transl = trans[:3, 3]
        if np.any(transl != 0):
            out_pts += transl

    return out_pts


def rotation(x=0, y=0, z=0):
    """Create an array with a 4 dimensional rotation matrix

    Parameters
    ----------
    x, y, z : scalar
        Rotation around the origin (in rad).

    Returns
    -------
    r : array, shape = (4, 4)
        The rotation matrix.
    """
    cos_x = cos(x)
    cos_y = cos(y)
    cos_z = cos(z)
    sin_x = sin(x)
    sin_y = sin(y)
    sin_z = sin(z)
    r = np.array([[cos_y * cos_z, -cos_x * sin_z + sin_x * sin_y * cos_z,
                   sin_x * sin_z + cos_x * sin_y * cos_z, 0],
                  [cos_y * sin_z, cos_x * cos_z + sin_x * sin_y * sin_z,
                   - sin_x * cos_z + cos_x * sin_y * sin_z, 0],
                  [-sin_y, sin_x * cos_y, cos_x * cos_y, 0],
                  [0, 0, 0, 1]], dtype=float)
    return r


def rotation3d(x=0, y=0, z=0):
    """Create an array with a 3 dimensional rotation matrix

    Parameters
    ----------
    x, y, z : scalar
        Rotation around the origin (in rad).

    Returns
    -------
    r : array, shape = (3, 3)
        The rotation matrix.
    """
    cos_x = cos(x)
    cos_y = cos(y)
    cos_z = cos(z)
    sin_x = sin(x)
    sin_y = sin(y)
    sin_z = sin(z)
    r = np.array([[cos_y * cos_z, -cos_x * sin_z + sin_x * sin_y * cos_z,
                   sin_x * sin_z + cos_x * sin_y * cos_z],
                  [cos_y * sin_z, cos_x * cos_z + sin_x * sin_y * sin_z,
                   - sin_x * cos_z + cos_x * sin_y * sin_z],
                  [-sin_y, sin_x * cos_y, cos_x * cos_y]], dtype=float)
    return r


def rotation_angles(m):
    """Find rotation angles from a transformation matrix

    Parameters
    ----------
    m : array, shape >= (3, 3)
        Rotation matrix. Only the top left 3 x 3 partition is accessed.

    Returns
    -------
    x, y, z : float
        Rotation around x, y and z axes.
    """
    x = np.arctan2(m[2, 1], m[2, 2])
    c2 = np.sqrt(m[0, 0] ** 2 + m[1, 0] ** 2)
    y = np.arctan2(-m[2, 0], c2)
    s1 = np.sin(x)
    c1 = np.cos(x)
    z = np.arctan2(s1 * m[0, 2] - c1 * m[0, 1], c1 * m[1, 1] - s1 * m[1, 2])
    return x, y, z


def scaling(x=1, y=1, z=1):
    """Create an array with a scaling matrix

    Parameters
    ----------
    x, y, z : scalar
        Scaling factors.

    Returns
    -------
    s : array, shape = (4, 4)
        The scaling matrix.
    """
    s = np.array([[x, 0, 0, 0],
                  [0, y, 0, 0],
                  [0, 0, z, 0],
                  [0, 0, 0, 1]], dtype=float)
    return s


def translation(x=0, y=0, z=0):
    """Create an array with a translation matrix

    Parameters
    ----------
    x, y, z : scalar
        Translation parameters.

    Returns
    -------
    m : array, shape = (4, 4)
        The translation matrix.
    """
    m = np.array([[1, 0, 0, x],
                  [0, 1, 0, y],
                  [0, 0, 1, z],
                  [0, 0, 0, 1]], dtype=float)
    return m


def _ensure_trans(trans, fro='mri', to='head'):
    """Helper to ensure we have the proper transform"""
    if isinstance(fro, string_types):
        from_str = fro
        from_const = _str_to_frame[fro]
    else:
        from_str = _frame_to_str[fro]
        from_const = fro
    del fro
    if isinstance(to, string_types):
        to_str = to
        to_const = _str_to_frame[to]
    else:
        to_str = _frame_to_str[to]
        to_const = to
    del to
    err_str = 'trans must go %s<->%s, provided' % (from_str, to_str)
    if trans is None:
        raise ValueError('%s None' % err_str)
    if set([trans['from'], trans['to']]) != set([from_const, to_const]):
        raise ValueError('%s trans is %s->%s' % (err_str,
                                                 _frame_to_str[trans['from']],
                                                 _frame_to_str[trans['to']]))
    if trans['from'] != from_const:
        trans = invert_transform(trans)
    return trans


def _get_trans(trans, fro='mri', to='head'):
    """Get mri_head_t (from=mri, to=head) from mri filename"""
    if isinstance(trans, string_types):
        if not op.isfile(trans):
            raise IOError('trans file "%s" not found' % trans)
        if op.splitext(trans)[1] in ['.fif', '.gz']:
            fro_to_t = read_trans(trans)
        else:
            # convert "-trans.txt" to "-trans.fif" mri-type equivalent
            # these are usually actually in to_fro form
            t = np.genfromtxt(trans)
            if t.ndim != 2 or t.shape != (4, 4):
                raise RuntimeError('File "%s" did not have 4x4 entries'
                                   % trans)
            fro_to_t = Transform(to, fro, t)
    elif isinstance(trans, dict):
        fro_to_t = trans
        trans = 'dict'
    elif trans is None:
        fro_to_t = Transform(fro, to, np.eye(4))
        trans = 'identity'
    else:
        raise ValueError('transform type %s not known, must be str, dict, '
                         'or None' % type(trans))
    # it's usually a head->MRI transform, so we probably need to invert it
    fro_to_t = _ensure_trans(fro_to_t, fro, to)
    return fro_to_t, trans


def combine_transforms(t_first, t_second, fro, to):
    """Combine two transforms

    Parameters
    ----------
    t_first : dict
        First transform.
    t_second : dict
        Second transform.
    fro : int
        From coordinate frame.
    to : int
        To coordinate frame.

    Returns
    -------
    trans : dict
        Combined transformation.
    """
    fro = _to_const(fro)
    to = _to_const(to)
    if t_first['from'] != fro:
        raise RuntimeError('From mismatch: %s ("%s") != %s ("%s")'
                           % (t_first['from'],
                              _coord_frame_name(t_first['from']),
                              fro, _coord_frame_name(fro)))
    if t_first['to'] != t_second['from']:
        raise RuntimeError('Transform mismatch: t1["to"] = %s ("%s"), '
                           't2["from"] = %s ("%s")'
                           % (t_first['to'], _coord_frame_name(t_first['to']),
                              t_second['from'],
                              _coord_frame_name(t_second['from'])))
    if t_second['to'] != to:
        raise RuntimeError('To mismatch: %s ("%s") != %s ("%s")'
                           % (t_second['to'],
                              _coord_frame_name(t_second['to']),
                              to, _coord_frame_name(to)))
    return Transform(fro, to, np.dot(t_second['trans'], t_first['trans']))


def read_trans(fname):
    """Read a -trans.fif file

    Parameters
    ----------
    fname : str
        The name of the file.

    Returns
    -------
    trans : dict
        The transformation dictionary from the fif file.

    See Also
    --------
    write_trans
    Transform
    """
    fid, tree, directory = fiff_open(fname)

    with fid:
        for t in directory:
            if t.kind == FIFF.FIFF_COORD_TRANS:
                tag = read_tag(fid, t.pos)
                break
        else:
            raise IOError('This does not seem to be a -trans.fif file.')

    trans = tag.data
    return trans


def write_trans(fname, trans):
    """Write a -trans.fif file

    Parameters
    ----------
    fname : str
        The name of the file, which should end in '-trans.fif'.
    trans : dict
        Trans file data, as returned by read_trans.

    See Also
    --------
    read_trans
    """
    check_fname(fname, 'trans', ('-trans.fif', '-trans.fif.gz'))
    fid = start_file(fname)
    write_coord_trans(fid, trans)
    end_file(fid)


def invert_transform(trans):
    """Invert a transformation between coordinate systems

    Parameters
    ----------
    trans : dict
        Transform to invert.

    Returns
    -------
    inv_trans : dict
        Inverse transform.
    """
    return Transform(trans['to'], trans['from'], linalg.inv(trans['trans']))


def transform_surface_to(surf, dest, trans):
    """Transform surface to the desired coordinate system

    Parameters
    ----------
    surf : dict
        Surface.
    dest : 'meg' | 'mri' | 'head' | int
        Destination coordinate system. Can be an integer for using
        FIFF types.
    trans : dict
        Transformation.

    Returns
    -------
    res : dict
        Transformed source space. Data are modified in-place.
    """
    if isinstance(dest, string_types):
        if dest not in _str_to_frame:
            raise KeyError('dest must be one of %s, not "%s"'
                           % (list(_str_to_frame.keys()), dest))
        dest = _str_to_frame[dest]  # convert to integer
    if surf['coord_frame'] == dest:
        return surf

    trans = _ensure_trans(trans, int(surf['coord_frame']), dest)
    surf['coord_frame'] = dest
    surf['rr'] = apply_trans(trans, surf['rr'])
    surf['nn'] = apply_trans(trans, surf['nn'], move=False)
    return surf


def get_ras_to_neuromag_trans(nasion, lpa, rpa):
    """Construct a transformation matrix to the MNE head coordinate system

    Construct a transformation matrix from an arbitrary RAS coordinate system
    to the MNE head coordinate system, in which the x axis passes through the
    two preauricular points, and the y axis passes through the nasion and is
    normal to the x axis. (see mne manual, pg. 97)

    Parameters
    ----------
    nasion : array_like, shape (3,)
        Nasion point coordinate.
    lpa : array_like, shape (3,)
        Left peri-auricular point coordinate.
    rpa : array_like, shape (3,)
        Right peri-auricular point coordinate.

    Returns
    -------
    trans : numpy.array, shape = (4, 4)
        Transformation matrix to MNE head space.
    """
    # check input args
    nasion = np.asarray(nasion)
    lpa = np.asarray(lpa)
    rpa = np.asarray(rpa)
    for pt in (nasion, lpa, rpa):
        if pt.ndim != 1 or len(pt) != 3:
            raise ValueError("Points have to be provided as one dimensional "
                             "arrays of length 3.")

    right = rpa - lpa
    right_unit = right / linalg.norm(right)

    origin = lpa + np.dot(nasion - lpa, right_unit) * right_unit

    anterior = nasion - origin
    anterior_unit = anterior / linalg.norm(anterior)

    superior_unit = np.cross(right_unit, anterior_unit)

    x, y, z = -origin
    origin_trans = translation(x, y, z)

    trans_l = np.vstack((right_unit, anterior_unit, superior_unit, [0, 0, 0]))
    trans_r = np.reshape([0, 0, 0, 1], (4, 1))
    rot_trans = np.hstack((trans_l, trans_r))

    trans = np.dot(rot_trans, origin_trans)
    return trans


def _sphere_to_cartesian(theta, phi, r):
    """Transform spherical coordinates to cartesian"""
    z = r * np.sin(phi)
    rcos_phi = r * np.cos(phi)
    x = rcos_phi * np.cos(theta)
    y = rcos_phi * np.sin(theta)
    return x, y, z


def _polar_to_cartesian(theta, r):
    """Transform polar coordinates to cartesian"""
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def _cartesian_to_sphere(x, y, z):
    """Transform cartesian coordinates to spherical"""
    hypotxy = np.hypot(x, y)
    r = np.hypot(hypotxy, z)
    elev = np.arctan2(z, hypotxy)
    az = np.arctan2(y, x)
    return az, elev, r


def _topo_to_sphere(theta, radius):
    """Convert 2D topo coordinates to spherical."""
    sph_phi = (0.5 - radius) * 180
    sph_theta = -theta
    return sph_phi, sph_theta


def quat_to_rot(quat):
    """Convert a set of quaternions to rotations

    Parameters
    ----------
    quat : array, shape (..., 3)
        q1, q2, and q3 (x, y, z) parameters of a unit quaternion.

    Returns
    -------
    rot : array, shape (..., 3, 3)
        The corresponding rotation matrices.

    See Also
    --------
    rot_to_quat
    """
    # z = a + bi + cj + dk
    b, c, d = quat[..., 0], quat[..., 1], quat[..., 2]
    bb, cc, dd = b * b, c * c, d * d
    # use max() here to be safe in case roundoff errs put us over
    aa = np.maximum(1. - bb - cc - dd, 0.)
    a = np.sqrt(aa)
    ab_2 = 2 * a * b
    ac_2 = 2 * a * c
    ad_2 = 2 * a * d
    bc_2 = 2 * b * c
    bd_2 = 2 * b * d
    cd_2 = 2 * c * d
    rotation = np.array([(aa + bb - cc - dd, bc_2 - ad_2, bd_2 + ac_2),
                         (bc_2 + ad_2, aa + cc - bb - dd, cd_2 - ab_2),
                         (bd_2 - ac_2, cd_2 + ab_2, aa + dd - bb - cc),
                         ])
    if quat.ndim > 1:
        rotation = np.rollaxis(np.rollaxis(rotation, 1, quat.ndim + 1),
                               0, quat.ndim)
    return rotation


def _one_rot_to_quat(rot):
    """Convert a rotation matrix to quaternions"""
    # see e.g. http://www.euclideanspace.com/maths/geometry/rotations/
    #                 conversions/matrixToQuaternion/
    t = 1. + rot[0] + rot[4] + rot[8]
    if t > np.finfo(rot.dtype).eps:
        s = np.sqrt(t) * 2.
        qx = (rot[7] - rot[5]) / s
        qy = (rot[2] - rot[6]) / s
        qz = (rot[3] - rot[1]) / s
        # qw = 0.25 * s
    elif rot[0] > rot[4] and rot[0] > rot[8]:
        s = np.sqrt(1. + rot[0] - rot[4] - rot[8]) * 2.
        qx = 0.25 * s
        qy = (rot[1] + rot[3]) / s
        qz = (rot[2] + rot[6]) / s
        # qw = (rot[7] - rot[5]) / s
    elif rot[4] > rot[8]:
        s = np.sqrt(1. - rot[0] + rot[4] - rot[8]) * 2
        qx = (rot[1] + rot[3]) / s
        qy = 0.25 * s
        qz = (rot[5] + rot[7]) / s
        # qw = (rot[2] - rot[6]) / s
    else:
        s = np.sqrt(1. - rot[0] - rot[4] + rot[8]) * 2.
        qx = (rot[2] + rot[6]) / s
        qy = (rot[5] + rot[7]) / s
        qz = 0.25 * s
        # qw = (rot[3] - rot[1]) / s
    return qx, qy, qz


def rot_to_quat(rot):
    """Convert a set of rotations to quaternions

    Parameters
    ----------
    rot : array, shape (..., 3, 3)
        The rotation matrices to convert.

    Returns
    -------
    quat : array, shape (..., 3)
        The q1, q2, and q3 (x, y, z) parameters of the corresponding
        unit quaternions.

    See Also
    --------
    quat_to_rot
    """
    rot = rot.reshape(rot.shape[:-2] + (9,))
    return np.apply_along_axis(_one_rot_to_quat, -1, rot)


def _angle_between_quats(x, y):
    """Compute the angle between two quaternions w/3-element representations"""
    # convert to complete quaternion representation
    # use max() here to be safe in case roundoff errs put us over
    x0 = np.sqrt(np.maximum(1. - x[..., 0] ** 2 -
                            x[..., 1] ** 2 - x[..., 2] ** 2, 0.))
    y0 = np.sqrt(np.maximum(1. - y[..., 0] ** 2 -
                            y[..., 1] ** 2 - y[..., 2] ** 2, 0.))
    # the difference z = x * conj(y), and theta = np.arccos(z0)
    z0 = np.maximum(np.minimum(y0 * x0 + (x * y).sum(axis=-1), 1.), -1)
    return 2 * np.arccos(z0)


def _skew_symmetric_cross(a):
    """The skew-symmetric cross product of a vector"""
    return np.array([[0., -a[2], a[1]], [a[2], 0., -a[0]], [-a[1], a[0], 0.]])


def _find_vector_rotation(a, b):
    """Find the rotation matrix that maps unit vector a to b"""
    # Rodrigues' rotation formula:
    #   https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    #   http://math.stackexchange.com/a/476311
    R = np.eye(3)
    v = np.cross(a, b)
    if np.allclose(v, 0.):  # identical
        return R
    s = np.dot(v, v)  # sine of the angle between them
    c = np.dot(a, b)  # cosine of the angle between them
    vx = _skew_symmetric_cross(v)
    R += vx + np.dot(vx, vx) * (1 - c) / s
    return R
