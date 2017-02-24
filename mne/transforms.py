# -*- coding: utf-8 -*-
# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os
from os import path as op
import glob
import copy
from numbers import Integral
import numpy as np
from numpy import sin, cos
from scipy import linalg

from .fixes import _get_sph_harm
from .io.constants import FIFF
from .io.open import fiff_open
from .io.tag import read_tag
from .io.write import start_file, end_file, write_coord_trans
from .utils import check_fname, logger, verbose
from .externals.six import string_types


# transformation from anterior/left/superior coordinate system to
# right/anterior/superior:
als_ras_trans = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0],
                          [0, 0, 0, 1]])


_str_to_frame = dict(meg=FIFF.FIFFV_COORD_DEVICE,
                     mri=FIFF.FIFFV_COORD_MRI,
                     mri_voxel=FIFF.FIFFV_MNE_COORD_MRI_VOXEL,
                     head=FIFF.FIFFV_COORD_HEAD,
                     mni_tal=FIFF.FIFFV_MNE_COORD_MNI_TAL,
                     ras=FIFF.FIFFV_MNE_COORD_RAS,
                     fs_tal=FIFF.FIFFV_MNE_COORD_FS_TAL,
                     ctf_head=FIFF.FIFFV_MNE_COORD_CTF_HEAD,
                     ctf_meg=FIFF.FIFFV_MNE_COORD_CTF_DEVICE,
                     unknown=FIFF.FIFFV_COORD_UNKNOWN)
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
    """Convert string or int coord frame into int."""
    if isinstance(cf, string_types):
        if cf not in _str_to_frame:
            raise ValueError('Unknown cf %s' % cf)
        cf = _str_to_frame[cf]
    elif not isinstance(cf, (Integral, np.int32)):
        raise TypeError('cf must be str or int, not %s' % type(cf))
    return int(cf)


class Transform(dict):
    """A transform.

    Parameters
    ----------
    fro : str | int
        The starting coordinate frame.
    to : str | int
        The ending coordinate frame.
    trans : array-like, shape (4, 4) | None
        The transformation matrix. If None, an identity matrix will be
        used.
    """

    def __init__(self, fro, to, trans=None):  # noqa: D102
        super(Transform, self).__init__()
        # we could add some better sanity checks here
        fro = _to_const(fro)
        to = _to_const(to)
        trans = np.eye(4) if trans is None else np.asarray(trans, np.float64)
        if trans.shape != (4, 4):
            raise ValueError('Transformation must be shape (4, 4) not %s'
                             % (trans.shape,))
        self['from'] = fro
        self['to'] = to
        self['trans'] = trans

    def __repr__(self):  # noqa: D105
        return ('<Transform  |  %s->%s>\n%s'
                % (_coord_frame_name(self['from']),
                   _coord_frame_name(self['to']), self['trans']))

    @property
    def from_str(self):
        """The "from" frame as a string."""
        return _coord_frame_name(self['from'])

    @property
    def to_str(self):
        """The "to" frame as a string."""
        return _coord_frame_name(self['to'])

    def save(self, fname):
        """Save the transform as -trans.fif file.

        Parameters
        ----------
        fname : str
            The name of the file, which should end in '-trans.fif'.
        """
        write_trans(fname, self)

    def copy(self):
        """Make a copy of the transform."""
        return copy.deepcopy(self)


def _coord_frame_name(cframe):
    """Map integers to human-readable (verbose) names."""
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
    """Apply a transform matrix to an array of points.

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
    """Create an array with a 4 dimensional rotation matrix.

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
    """Create an array with a 3 dimensional rotation matrix.

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


def rotation3d_align_z_axis(target_z_axis):
    """Compute a rotation matrix to align [ 0 0 1] with supplied target z axis.

    Parameters
    ----------
    target_z_axis : array, shape (1, 3)
        z axis. computed matrix (r) will map [0 0 1] to target_z_axis

    Returns
    -------
    r : array, shape (3, 3)
        The rotation matrix.
    """
    target_z_axis = target_z_axis / np.linalg.norm(target_z_axis)
    r = np.zeros((3, 3))
    if ((1. + target_z_axis[2]) < 1E-12):
        r[0, 0] = 1.
        r[1, 1] = -1.
        r[2, 2] = -1.
    else:
        f = 1. / (1. + target_z_axis[2])
        r[0, 0] = 1. - 1. * f * target_z_axis[0] * target_z_axis[0]
        r[0, 1] = -1. * f * target_z_axis[0] * target_z_axis[1]
        r[0, 2] = target_z_axis[0]
        r[1, 0] = -1. * f * target_z_axis[0] * target_z_axis[1]
        r[1, 1] = 1. - 1. * f * target_z_axis[1] * target_z_axis[1]
        r[1, 2] = target_z_axis[1]
        r[2, 0] = -target_z_axis[0]
        r[2, 1] = -target_z_axis[1]
        r[2, 2] = 1. - f * (target_z_axis[0] * target_z_axis[0] +
                            target_z_axis[1] * target_z_axis[1])

    # assert that r is a rotation matrix r^t * r = I and det(r) = 1
    assert(np.any((r.dot(r.T) - np.identity(3)) < 1E-12))
    assert((linalg.det(r) - 1.0) < 1E-12)
    # assert that r maps [0 0 1] on the device z axis (target_z_axis)
    assert(linalg.norm(target_z_axis - r.dot([0, 0, 1])) < 1e-12)

    return r


def rotation_angles(m):
    """Find rotation angles from a transformation matrix.

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
    """Create an array with a scaling matrix.

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
    """Create an array with a translation matrix.

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
    """Ensure we have the proper transform."""
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
    """Get mri_head_t (from=mri, to=head) from mri filename."""
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
        fro_to_t = Transform(fro, to)
        trans = 'identity'
    else:
        raise ValueError('transform type %s not known, must be str, dict, '
                         'or None' % type(trans))
    # it's usually a head->MRI transform, so we probably need to invert it
    fro_to_t = _ensure_trans(fro_to_t, fro, to)
    return fro_to_t, trans


def combine_transforms(t_first, t_second, fro, to):
    """Combine two transforms.

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
    """Read a -trans.fif file.

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
    """Write a -trans.fif file.

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
    """Invert a transformation between coordinate systems.

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


def transform_surface_to(surf, dest, trans, copy=False):
    """Transform surface to the desired coordinate system.

    Parameters
    ----------
    surf : dict
        Surface.
    dest : 'meg' | 'mri' | 'head' | int
        Destination coordinate system. Can be an integer for using
        FIFF types.
    trans : dict
        Transformation.
    copy : bool
        If False (default), operate in-place.

    Returns
    -------
    res : dict
        Transformed source space.
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
    """Construct a transformation matrix to the MNE head coordinate system.

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


###############################################################################
# Spherical coordinates and harmonics

def _cart_to_sph(cart):
    """Convert Cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    cart_pts : ndarray, shape (n_points, 3)
        Array containing points in Cartesian coordinates (x, y, z)

    Returns
    -------
    sph_pts : ndarray, shape (n_points, 3)
        Array containing points in spherical coordinates (rad, azimuth, polar)
    """
    assert cart.ndim == 2 and cart.shape[1] == 3
    cart = np.atleast_2d(cart)
    out = np.empty((len(cart), 3))
    out[:, 0] = np.sqrt(np.sum(cart * cart, axis=1))
    out[:, 1] = np.arctan2(cart[:, 1], cart[:, 0])
    out[:, 2] = np.arccos(cart[:, 2] / out[:, 0])
    return out


def _sph_to_cart(sph):
    """Convert spherical coordinates to Cartesion coordinates."""
    assert sph.ndim == 2 and sph.shape[1] == 3
    sph = np.atleast_2d(sph)
    out = np.empty((len(sph), 3))
    out[:, 2] = sph[:, 0] * np.cos(sph[:, 2])
    xy = sph[:, 0] * np.sin(sph[:, 2])
    out[:, 0] = xy * np.cos(sph[:, 1])
    out[:, 1] = xy * np.sin(sph[:, 1])
    return out


def _get_n_moments(order):
    """Compute the number of multipolar moments (spherical harmonics).

    Equivalent to [1]_ Eq. 32.

    .. note:: This count excludes ``degree=0`` (for ``order=0``).

    Parameters
    ----------
    order : array-like
        Expansion orders, often ``[int_order, ext_order]``.

    Returns
    -------
    M : ndarray
        Number of moments due to each order.
    """
    order = np.asarray(order, int)
    return (order + 2) * order


def _sph_to_cart_partials(az, pol, g_rad, g_az, g_pol):
    """Convert spherical partial derivatives to cartesian coords.

    Note: Because we are dealing with partial derivatives, this calculation is
    not a static transformation. The transformation matrix itself is dependent
    on azimuth and polar coord.

    See the 'Spherical coordinate sytem' section here:
    wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates

    Parameters
    ----------
    az : ndarray, shape (n_points,)
        Array containing spherical coordinates points (azimuth).
    pol : ndarray, shape (n_points,)
        Array containing spherical coordinates points (polar).
    sph_grads : ndarray, shape (n_points, 3)
        Array containing partial derivatives at each spherical coordinate
        (radius, azimuth, polar).

    Returns
    -------
    cart_grads : ndarray, shape (n_points, 3)
        Array containing partial derivatives in Cartesian coordinates (x, y, z)
    """
    sph_grads = np.c_[g_rad, g_az, g_pol]
    cart_grads = np.zeros_like(sph_grads)
    c_as, s_as = np.cos(az), np.sin(az)
    c_ps, s_ps = np.cos(pol), np.sin(pol)
    trans = np.array([[c_as * s_ps, -s_as, c_as * c_ps],
                      [s_as * s_ps, c_as, c_ps * s_as],
                      [c_ps, np.zeros_like(c_as), -s_ps]])
    cart_grads = np.einsum('ijk,kj->ki', trans, sph_grads)
    return cart_grads


def _deg_ord_idx(deg, order):
    """Get the index into S_in or S_out given a degree and order."""
    # The -1 here is because we typically exclude the degree=0 term
    return deg * deg + deg + order - 1


def _sh_negate(sh, order):
    """Helper to get the negative spherical harmonic from a positive one."""
    assert order >= 0
    return sh.conj() * (-1. if order % 2 else 1.)  # == (-1) ** order


def _sh_complex_to_real(sh, order):
    """Helper function to convert complex to real basis functions.

    Parameters
    ----------
    sh : array-like
        Spherical harmonics. Must be from order >=0 even if negative orders
        are used.
    order : int
        Order (usually 'm') of multipolar moment.

    Returns
    -------
    real_sh : array-like
        The real version of the spherical harmonics.

    Notes
    -----
    This does not include the Condon-Shortely phase.
    """
    if order == 0:
        return np.real(sh)
    else:
        return np.sqrt(2.) * (np.real if order > 0 else np.imag)(sh)


def _sh_real_to_complex(shs, order):
    """Convert real spherical harmonic pair to complex.

    Parameters
    ----------
    shs : ndarray, shape (2, ...)
        The real spherical harmonics at ``[order, -order]``.
    order : int
        Order (usually 'm') of multipolar moment.

    Returns
    -------
    sh : array-like, shape (...)
        The complex version of the spherical harmonics.
    """
    if order == 0:
        return shs[0]
    else:
        return (shs[0] + 1j * np.sign(order) * shs[1]) / np.sqrt(2.)


def _compute_sph_harm(order, az, pol):
    """Compute complex spherical harmonics of spherical coordinates."""
    sph_harm = _get_sph_harm()
    out = np.empty((len(az), _get_n_moments(order) + 1))
    # _deg_ord_idx(0, 0) = -1 so we're actually okay to use it here
    for degree in range(order + 1):
        for order_ in range(degree + 1):
            sph = sph_harm(order_, degree, az, pol)
            out[:, _deg_ord_idx(degree, order_)] = \
                _sh_complex_to_real(sph, order_)
            if order_ > 0:
                out[:, _deg_ord_idx(degree, -order_)] = \
                    _sh_complex_to_real(_sh_negate(sph, order_), -order_)
    return out


###############################################################################
# Thin-plate spline transformations

# Adapted from code from the MATLAB file exchange:
#    https://www.mathworks.com/matlabcentral/fileexchange/
#            53867-3d-point-set-warping-by-thin-plate-rbf-function
#    https://www.mathworks.com/matlabcentral/fileexchange/
#            53828-rbf-or-thin-plate-splines-image-warping
# Associated (BSD 2-clause) license:
#
# Copyright (c) 2015, Wang Lin
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in
#       the documentation and/or other materials provided with the distribution
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

class _TPSWarp(object):
    """Transform points using thin-plate spline (TPS) warping.

    Notes
    -----
    Adapted from code by `Wang Lin <wanglin193@hotmail.com>`_.

    References
    ----------
    .. [1] Bookstein, F. L. "Principal Warps: Thin Plate Splines and the
           Decomposition of Deformations." IEEE Trans. Pattern Anal. Mach.
           Intell. 11, 567-585, 1989.
    """

    def fit(self, source, destination, reg=1e-3):
        from scipy.spatial.distance import cdist
        assert source.shape[1] == destination.shape[1] == 3
        assert source.shape[0] == destination.shape[0]
        # Forward warping, different from image warping, use |dist|**2
        dists = _tps(cdist(source, destination, 'sqeuclidean'))
        # Y = L * w
        # L: RBF matrix about source
        # Y: Points matrix about destination
        P = np.concatenate((np.ones((source.shape[0], 1)), source), axis=-1)
        L = np.vstack([np.hstack([dists, P]),
                       np.hstack([P.T, np.zeros((4, 4))])])
        Y = np.concatenate((destination, np.zeros((4, 3))), axis=0)
        # Regularize it a bit
        L += reg * np.eye(L.shape[0])
        self._destination = destination.copy()
        self._weights = linalg.lstsq(L, Y)[0]
        return self

    @verbose
    def transform(self, pts, verbose=None):
        """Apply the warp.

        Parameters
        ----------
        pts : shape (n_transform, 3)
            Source points to warp to the destination.

        Returns
        -------
        dest : shape (n_transform, 3)
            The transformed points.
        """
        logger.info('Transforming %s points' % (len(pts),))
        from scipy.spatial.distance import cdist
        assert pts.shape[1] == 3
        # for memory reasons, we should do this in ~100 MB chunks
        out = np.zeros_like(pts)
        n_splits = max(int((pts.shape[0] * self._destination.shape[0]) /
                           (100e6 / 8.)), 1)
        for this_out, this_pts in zip(np.array_split(out, n_splits),
                                      np.array_split(pts, n_splits)):
            dists = _tps(cdist(this_pts, self._destination, 'sqeuclidean'))
            L = np.hstack((dists, np.ones((dists.shape[0], 1)), this_pts))
            this_out[:] = np.dot(L, self._weights)
        assert not (out == 0).any()
        return out


def _tps(distsq):
    """Thin-plate function (r ** 2) * np.log(r)."""
    # NOTE: For our warping functions, a radial basis like
    # exp(-distsq / radius ** 2) could also be used
    out = np.zeros_like(distsq)
    mask = distsq > 0  # avoid log(0)
    valid = distsq[mask]
    out[mask] = valid * np.log(valid)
    return out


###############################################################################
# Spherical harmonic approximation + TPS warp

class _SphericalSurfaceWarp(object):
    """Warp surfaces via spherical harmonic smoothing and thin-plate splines.

    Notes
    -----
    This class can be used to warp data from a source subject to
    a destination subject, as described in [1]_. The procedure is:

        1. Perform a spherical harmonic approximation to the source and
           destination surfaces, which smooths them and allows arbitrary
           interpolation.
        2. Choose a set of matched points on the two surfaces.
        3. Use thin-plate spline warping (common in 2D image manipulation)
           to generate transformation coefficients.
        4. Warp points from the source subject (which should be inside the
           original surface) to the destination subject.

    .. versionadded:: 0.14

    References
    ----------
    .. [1] Darvas F, Ermer JJ, Mosher JC, Leahy RM (2006). "Generic head
           models for atlas-based EEG source analysis."
           Human Brain Mapping 27:129-143
    """

    def __repr__(self):
        rep = '<SphericalSurfaceWarp : '
        if not hasattr(self, '_warp'):
            rep += 'no fitting done >'
        else:
            rep += ('fit %d->%d pts using match=%s (%d pts), order=%s, reg=%s>'
                    % tuple(self._fit_params[key]
                            for key in ['n_src', 'n_dest', 'match', 'n_match',
                                        'order', 'reg']))
        return rep

    @verbose
    def fit(self, source, destination, order=4, reg=1e-5, center=True,
            match='oct5', verbose=None):
        """Fit the warp from source points to destination points.

        Parameters
        ----------
        source : array, shape (n_src, 3)
            The source points.
        destination : array, shape (n_dest, 3)
            The destination points.
        order : int
            Order of the spherical harmonic fit.
        reg : float
            Regularization of the TPS warp.
        center : bool
            If True, center the points by fitting a sphere to points
            that are in a reasonable region for head digitization.
        match : str
            The uniformly-spaced points to match on the two surfaces.
            Can be "ico#" or "oct#" where "#" is an integer.
            The default is "oct5".
        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more).

        Returns
        -------
        inst : instance of SphericalSurfaceWarp
            The warping object (for chaining).
        """
        from .bem import _fit_sphere
        from .source_space import _check_spacing
        match_rr = _check_spacing(match, verbose=False)[2]['rr']
        logger.info('Computing TPS warp')
        src_center = dest_center = np.zeros(3)
        if center:
            logger.info('    Centering data')
            hsp = np.array([p for p in source
                            if not (p[2] < -1e-6 and p[1] > 1e-6)])
            src_center = _fit_sphere(hsp, disp=False)[1]
            source = source - src_center
            hsp = np.array([p for p in destination
                            if not (p[2] < 0 and p[1] > 0)])
            dest_center = _fit_sphere(hsp, disp=False)[1]
            destination = destination - dest_center
            logger.info('    Using centers %s -> %s'
                        % (np.array_str(src_center, None, 3),
                           np.array_str(dest_center, None, 3)))
        self._fit_params = dict(
            n_src=len(source), n_dest=len(destination), match=match,
            n_match=len(match_rr), order=order, reg=reg)
        assert source.shape[1] == destination.shape[1] == 3
        self._destination = destination.copy()
        # 1. Compute spherical coordinates of source and destination points
        logger.info('    Converting to spherical coordinates')
        src_rad_az_pol = _cart_to_sph(source).T
        dest_rad_az_pol = _cart_to_sph(destination).T
        match_rad_az_pol = _cart_to_sph(match_rr).T
        del match_rr
        # 2. Compute spherical harmonic coefficients for all points
        logger.info('    Computing spherical harmonic approximation with '
                    'order %s' % order)
        src_sph = _compute_sph_harm(order, *src_rad_az_pol[1:])
        dest_sph = _compute_sph_harm(order, *dest_rad_az_pol[1:])
        match_sph = _compute_sph_harm(order, *match_rad_az_pol[1:])
        # 3. Fit spherical harmonics to both surfaces to smooth them
        src_coeffs = linalg.lstsq(src_sph, src_rad_az_pol[0])[0]
        dest_coeffs = linalg.lstsq(dest_sph, dest_rad_az_pol[0])[0]
        # 4. Smooth both surfaces using these coefficients, and evaluate at
        #     the "shape" points
        logger.info('    Matching %d points (%s) on smoothed surfaces'
                    % (len(match_sph), match))
        src_rad_az_pol = match_rad_az_pol.copy()
        src_rad_az_pol[0] = np.abs(np.dot(match_sph, src_coeffs))
        dest_rad_az_pol = match_rad_az_pol.copy()
        dest_rad_az_pol[0] = np.abs(np.dot(match_sph, dest_coeffs))
        # 5. Convert matched points to Cartesion coordinates and put back
        source = _sph_to_cart(src_rad_az_pol.T)
        source += src_center
        destination = _sph_to_cart(dest_rad_az_pol.T)
        destination += dest_center
        # 6. Compute TPS warp of matched points from smoothed surfaces
        self._warp = _TPSWarp().fit(source, destination, reg)
        self._matched = np.array([source, destination])
        logger.info('[done]')
        return self

    @verbose
    def transform(self, source, verbose=None):
        """Transform arbitrary source points to the destination.

        Parameters
        ----------
        source : ndarray, shape (n_pts, 3)
            Source points to transform. They do not need to be the same
            points that were used to generate the model, although ideally
            they will be inside the convex hull formed by the original
            source points.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more).

        Returns
        -------
        destination : ndarray, shape (n_pts, 3)
            The points transformed to the destination space.
        """
        return self._warp.transform(source)


###############################################################################
# Other transforms

def _pol_to_cart(pol):
    """Transform polar coordinates to cartesian."""
    out = np.empty_like(pol)
    out[:, 0] = pol[:, 0] * np.cos(pol[:, 1])
    out[:, 1] = pol[:, 0] * np.sin(pol[:, 1])
    return out


def _topo_to_sph(topo):
    """Convert 2D topo coordinates to spherical coordinates."""
    assert topo.ndim == 2 and topo.shape[1] == 2
    sph = np.ones((len(topo), 3))
    sph[:, 1] = -np.deg2rad(topo[:, 0])
    sph[:, 2] = np.pi * topo[:, 1]
    return sph


###############################################################################
# Quaternions

def quat_to_rot(quat):
    """Convert a set of quaternions to rotations.

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
    """Convert a rotation matrix to quaternions."""
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
    return np.array((qx, qy, qz))


def rot_to_quat(rot):
    """Convert a set of rotations to quaternions.

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
    """Compute the ang between two quaternions w/3-element representations."""
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
    """The skew-symmetric cross product of a vector."""
    return np.array([[0., -a[2], a[1]], [a[2], 0., -a[0]], [-a[1], a[0], 0.]])


def _find_vector_rotation(a, b):
    """Find the rotation matrix that maps unit vector a to b."""
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
