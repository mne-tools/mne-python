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


def _coord_frame_name(cframe):
    """Map integers to human-readable names"""
    types = {FIFF.FIFFV_COORD_UNKNOWN: 'unknown',
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
    return types.get(int(cframe), 'unknown')


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
    trans : array, shape = (4, 4)
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
    trans = np.asarray(trans)
    pts = np.asarray(pts)
    if pts.size == 0:
        return pts.copy()

    # apply rotation & scale
    if pts.ndim == 1:
        out_pts = np.dot(trans[:3, :3], pts)
    else:
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


def _get_mri_head_t(mri):
    """Get mri_head_t (from=mri, to=head) from mri filename"""
    if isinstance(mri, string_types):
        if not op.isfile(mri):
            raise IOError('mri file "%s" not found' % mri)
        if op.splitext(mri)[1] in ['.fif', '.gz']:
            mri_head_t = read_trans(mri)
        else:
            # convert "-trans.txt" to "-trans.fif" mri-type equivalent
            t = np.genfromtxt(mri)
            if t.ndim != 2 or t.shape != (4, 4):
                raise RuntimeError('File "%s" did not have 4x4 entries'
                                   % mri)
            mri_head_t = {'from': FIFF.FIFFV_COORD_HEAD,
                          'to': FIFF.FIFFV_COORD_MRI, 'trans': t}
    else:  # dict
        mri_head_t = mri
        mri = 'dict'
    # it's usually a head->MRI transform, so we probably need to invert it
    if mri_head_t['from'] == FIFF.FIFFV_COORD_HEAD:
        mri_head_t = invert_transform(mri_head_t)
    if not (mri_head_t['from'] == FIFF.FIFFV_COORD_MRI and
            mri_head_t['to'] == FIFF.FIFFV_COORD_HEAD):
        raise RuntimeError('Incorrect MRI transform provided')
    return mri_head_t, mri


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
    return {'from': fro, 'to': to, 'trans': np.dot(t_second['trans'],
                                                   t_first['trans'])}


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

    Notes
    -----
    The trans dictionary has the following structure:
    trans = {'from': int, 'to': int, 'trans': numpy.ndarray <4x4>}
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
    return {'to': trans['from'], 'from': trans['to'],
            'trans': linalg.inv(trans['trans'])}


_frame_dict = dict(meg=FIFF.FIFFV_COORD_DEVICE,
                   mri=FIFF.FIFFV_COORD_MRI,
                   head=FIFF.FIFFV_COORD_HEAD)


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
        if dest not in _frame_dict:
            raise KeyError('dest must be one of %s, not "%s"'
                           % [list(_frame_dict.keys()), dest])
        dest = _frame_dict[dest]  # convert to integer
    if surf['coord_frame'] == dest:
        return surf

    if trans['to'] == surf['coord_frame'] and trans['from'] == dest:
        trans = invert_transform(trans)
    elif trans['from'] != surf['coord_frame'] or trans['to'] != dest:
        raise ValueError('Cannot transform the source space using this '
                         'coordinate transformation')

    surf['coord_frame'] = dest
    surf['rr'] = apply_trans(trans['trans'], surf['rr'])
    surf['nn'] = apply_trans(trans['trans'], surf['nn'], move=False)
    return surf


def transform_coordinates(filename, pos, orig, dest):
    """Transform coordinates between various MRI-related coordinate frames

    Parameters
    ----------
    filename: string
        Name of a fif file containing the coordinate transformations
        This file can be conveniently created with mne_collect_transforms
        or ``collect_transforms``.
    pos: array of shape N x 3
        array of locations to transform (in meters)
    orig: 'meg' | 'mri'
        Coordinate frame of the above locations.
        'meg' is MEG head coordinates
        'mri' surface RAS coordinates
    dest: 'meg' | 'mri' | 'fs_tal' | 'mni_tal'
        Coordinate frame of the result.
        'mni_tal' is MNI Talairach
        'fs_tal' is FreeSurfer Talairach

    Returns
    -------
    trans_pos: array of shape N x 3
        The transformed locations

    Examples
    --------
    transform_coordinates('all-trans.fif', np.eye(3), 'meg', 'fs_tal')
    transform_coordinates('all-trans.fif', np.eye(3), 'mri', 'mni_tal')
    """
    #   Read the fif file containing all necessary transformations
    fid, tree, directory = fiff_open(filename)

    coord_names = dict(mri=FIFF.FIFFV_COORD_MRI,
                       meg=FIFF.FIFFV_COORD_HEAD,
                       mni_tal=FIFF.FIFFV_MNE_COORD_MNI_TAL,
                       fs_tal=FIFF.FIFFV_MNE_COORD_FS_TAL)

    orig = coord_names[orig]
    dest = coord_names[dest]

    T0 = T1 = T2 = T3plus = T3minus = None
    for d in directory:
        if d.kind == FIFF.FIFF_COORD_TRANS:
            tag = read_tag(fid, d.pos)
            trans = tag.data
            if (trans['from'] == FIFF.FIFFV_COORD_MRI and
                    trans['to'] == FIFF.FIFFV_COORD_HEAD):
                T0 = invert_transform(trans)
            elif (trans['from'] == FIFF.FIFFV_COORD_MRI and
                  trans['to'] == FIFF.FIFFV_MNE_COORD_RAS):
                T1 = trans
            elif (trans['from'] == FIFF.FIFFV_MNE_COORD_RAS and
                  trans['to'] == FIFF.FIFFV_MNE_COORD_MNI_TAL):
                T2 = trans
            elif trans['from'] == FIFF.FIFFV_MNE_COORD_MNI_TAL:
                if trans['to'] == FIFF.FIFFV_MNE_COORD_FS_TAL_GTZ:
                    T3plus = trans
                elif trans['to'] == FIFF.FIFFV_MNE_COORD_FS_TAL_LTZ:
                    T3minus = trans
    fid.close()
    #
    #   Check we have everything we need
    #
    if ((orig == FIFF.FIFFV_COORD_HEAD and T0 is None) or (T1 is None) or
            (T2 is None) or (dest == FIFF.FIFFV_MNE_COORD_FS_TAL and
                             ((T3minus is None) or (T3minus is None)))):
        raise ValueError('All required coordinate transforms not found')

    #
    #   Go ahead and transform the data
    #
    if pos.shape[1] != 3:
        raise ValueError('Coordinates must be given in a N x 3 array')

    if dest == orig:
        trans_pos = pos.copy()
    else:
        n_points = pos.shape[0]
        pos = np.c_[pos, np.ones(n_points)].T
        if orig == FIFF.FIFFV_COORD_HEAD:
            pos = np.dot(T0['trans'], pos)
        elif orig != FIFF.FIFFV_COORD_MRI:
            raise ValueError('Input data must be in MEG head or surface RAS '
                             'coordinates')

        if dest == FIFF.FIFFV_COORD_HEAD:
            pos = np.dot(linalg.inv(T0['trans']), pos)
        elif dest != FIFF.FIFFV_COORD_MRI:
            pos = np.dot(np.dot(T2['trans'], T1['trans']), pos)
            if dest != FIFF.FIFFV_MNE_COORD_MNI_TAL:
                if dest == FIFF.FIFFV_MNE_COORD_FS_TAL:
                    for k in range(n_points):
                        if pos[2, k] > 0:
                            pos[:, k] = np.dot(T3plus['trans'], pos[:, k])
                        else:
                            pos[:, k] = np.dot(T3minus['trans'], pos[:, k])
                else:
                    raise ValueError('Illegal choice for the output '
                                     'coordinates')

        trans_pos = pos[:3, :].T

    return trans_pos


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


def collect_transforms(fname, xforms):
    """Collect a set of transforms in a single FIFF file

    Parameters
    ----------
    fname : str
        Filename to save to.
    xforms : list of dict
        List of transformations.
    """
    check_fname(fname, 'trans', ('-trans.fif', '-trans.fif.gz'))
    with start_file(fname) as fid:
        for xform in xforms:
            write_coord_trans(fid, xform)
        end_file(fid)


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
