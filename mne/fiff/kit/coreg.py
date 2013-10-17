"""Coordinate Point Extractor for KIT system"""

# Author: Teon Brooks <teon@nyu.edu>
#
# License: BSD (3-clause)

from datetime import datetime
import cPickle as pickle
import os
from os import SEEK_CUR
import re
from struct import unpack

import numpy as np
from scipy.linalg import norm

from ... import __version__
from ...transforms import translation
from .constants import KIT


def read_mrk(fname):
    """Marker Point Extraction in MEG space directly from sqd

    Parameters
    ----------
    fname : str
        Absolute path to Marker file.
        File formats allowed: *.sqd, *.mrk, *.txt, *.pickled

    Returns
    -------
    mrk_points : numpy.array, shape = (n_points, 3)
        Marker points in MEG space [m].
    """
    ext = os.path.splitext(fname)[-1]
    if ext in ('.sqd', '.mrk'):
        with open(fname, 'r') as fid:
            fid.seek(KIT.MRK_INFO)
            mrk_offset = unpack('i', fid.read(KIT.INT))[0]
            fid.seek(mrk_offset)
            # skips match_done, meg_to_mri and mri_to_meg
            fid.seek(KIT.INT + (2 * KIT.DOUBLE * 4 ** 2), SEEK_CUR)
            mrk_count = unpack('i', fid.read(KIT.INT))[0]
            pts = []
            for _ in range(mrk_count):
                # skips mri/meg mrk_type and done, mri_marker
                fid.seek(KIT.INT * 4 + (KIT.DOUBLE * 3), SEEK_CUR)
                pts.append(np.fromfile(fid, dtype='d', count=3))
                mrk_points = np.array(pts)
    elif ext == '.txt':
        mrk_points = np.loadtxt(fname)
    elif ext == '.pickled':
        with open(fname) as fid:
            food = pickle.load(fid)
        try:
            mrk_points = food['mrk']
        except:
            err = ("%r does not contain marker points." % fname)
            raise ValueError(err)
    else:
        err = ('KIT marker file must be *.sqd, *.txt or *.pickled, '
               'not *%s.' % ext)
        raise ValueError(err)

    # check output
    mrk_points = np.asarray(mrk_points)
    if mrk_points.shape != (5, 3):
        err = ("%r is no marker file, shape is "
               "%s" % (fname, mrk_points.shape))
        raise ValueError(err)
    return mrk_points


def write_mrk(fname, points):
    """Save KIT marker coordinates

    Parameters
    ----------
    fname : str
        Path to the file to write. The kind of file to write is determined
        based on the extension: '.txt' for tab separated text file, '.pickled'
        for pickled file.
    points : array_like, shape = (5, 3)
        The marker point coordinates.
    """
    mrk = np.asarray(points)
    _, ext = os.path.splitext(fname)
    if mrk.shape != (5, 3):
        err = ("KIT marker points array needs to have shape (5, 3), got "
               "%s." % str(mrk.shape))
        raise ValueError(err)

    if ext == '.pickled':
        with open(fname, 'w') as fid:
            pickle.dump({'mrk': mrk}, fid, pickle.HIGHEST_PROTOCOL)
    elif ext == '.txt':
        np.savetxt(fname, mrk, fmt='%.18e', delimiter='\t', newline='\n')
    else:
        err = "Unrecognized extension: %r. Need '.txt' or '.pickled'." % ext
        raise ValueError(err)


def read_elp(fname):
    """ELP point extraction in Polhemus head space

    Parameters
    ----------
    fname : str
        Absolute path to laser point file acquired from Polhemus system.
        File formats allowed: *.txt

    Returns
    -------
    elp_points : numpy.array, shape = (n_points, 3)
        Fiducial and marker points in Polhemus head space.
    """
    pattern = re.compile(r'(\-?\d+\.\d+)\s+(\-?\d+\.\d+)\s+(\-?\d+\.\d+)')
    elp_points = pattern.findall(open(fname).read())
    elp_points = np.array(elp_points, dtype=float)
    if elp_points.shape[1] != 3:
        err = ("File %r does not contain 3 columns as required; got shape "
               "%s." % (fname, elp_points.shape))
        raise ValueError(err)
    elif len(elp_points) < 8:
        err = ("File %r contains fewer than 8 points; got shape "
               "%s." % (fname, elp_points.shape))
        raise ValueError(err)
    return elp_points


def read_hsp(fname):
    """Read a Polhemus ascii head shape file

    Parameters
    ----------
    fname : str
        Path to head shape file acquired from Polhemus system and saved in
        ascii format.

    Returns
    -------
    hsp_points : numpy.array, shape = (n_points, 3)
        Headshape points in Polhemus head space.
        File formats allowed: *.txt, *.pickled
    """
    pattern = re.compile(r'(\-?\d+\.\d+)\s+(\-?\d+\.\d+)\s+(\-?\d+\.\d+)')
    with open(fname) as fid:
        hsp_points = pattern.findall(fid.read())
    hsp_points = np.array(hsp_points, dtype=float)
    return hsp_points


def write_hsp(fname, pts):
    """Write a headshape hsp file

    Parameters
    ----------
    fname : str
        Target file.
    pts : array, shape = (n_pts, 3)
        Points comprising the headshape.
    """
    pts = np.asarray(pts)
    if (pts.ndim != 2) or (pts.shape[1] != 3):
        err = "pts must be of shape (n_pts, 3), not %r" % str(pts.shape)
        raise ValueError(err)

    with open(fname, 'w') as fid:
        version = __version__
        now = datetime.now().strftime("%I:%M%p on %B %d, %Y")
        fid.write("% Ascii 3D points file created by mne-python version "
                  "{version} at {now}\n".format(version=version, now=now))
        fid.write("% {N} 3D points, x y z per line\n".format(N=len(pts)))
        np.savetxt(fid, pts, '%8.2f', ' ')


def read_sns(fname):
    """Sensor coordinate extraction in MEG space

    Parameters
    ----------
    fname : str
        Absolute path to sensor definition file.

    Returns
    -------
    locs : numpy.array, shape = (n_points, 3)
        Sensor coil location.
    """
    p = re.compile(r'\d,[A-Za-z]*,([\.\-0-9]+),' +
                   r'([\.\-0-9]+),([\.\-0-9]+),' +
                   r'([\.\-0-9]+),([\.\-0-9]+)')
    with open(fname) as fid:
        locs = np.array(p.findall(fid.read()), dtype=float)
    return locs


def get_head_coord_trans(nasion, lpa, rpa):
    """Construct a transformation matrix to the MNE head coordinate system

    Construct a transformation matrix from an arbitrary RAS coordinate system
    to the MNE head coordinate system, in which the x axis passes through the
    two preauricular points, and the y axis passes through the nasion and is
    normal to the x axis. (see mne manual, pg. 97)

    Parameters
    ----------
    nasion : array_like, shape = (3,)
        Nasion point coordinate.
    lpa : array_like, shape = (3,)
        Left peri-auricular point coordinate.
    rpa : array_like, shape = (3,)
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
            err = ("Points have to be provided as one dimensional arrays of "
                   "length 3.")
            raise ValueError(err)

    right = rpa - lpa
    right_unit = right / norm(right)

    origin = lpa + np.dot(nasion - lpa, right_unit) * right_unit

    anterior = nasion - origin
    anterior_unit = anterior / norm(anterior)

    superior_unit = np.cross(right_unit, anterior_unit)

    x, y, z = -origin
    origin_trans = translation(x, y, z)

    trans_l = np.vstack((right_unit, anterior_unit, superior_unit, [0, 0, 0]))
    trans_r = np.reshape([0, 0, 0, 1], (4, 1))
    rot_trans = np.hstack((trans_l, trans_r))

    trans = np.dot(rot_trans, origin_trans)
    return trans
