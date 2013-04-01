"""Coordinate Point Extractor for KIT system"""

# Author: Teon Brooks <teon@nyu.edu>
#
# License: BSD (3-clause)

from datetime import datetime
import cPickle as pickle
from os import SEEK_CUR, path
import re
from struct import unpack

import numpy as np
from scipy import linalg

from ... import __version__
from ..constants import FIFF
from .constants import KIT



def get_dig_points(fid_points, elp_points, hsp_points):
    """Create a dig point list for the info dictionary

    Parameters
    ----------
    fid_points : array, shape = (3, 3)
        Polhemus fiducial (in the target coordinate system).
    elp_points : array, shape = (5, 3)
        Polhemus elp points (in the target coordinate system).
    hsp_points : array, shape = (n_points, 3)
        Array with headshape points (in the target coordinate system).

    Returns
    -------
    dig : list
        A list containing the mrk_points, elp_points, and hsp_points in
        the format used for raw.info['dig'].
    """
    dig = []

    nasion, lpa, rpa = fid_points

    point_dict = {}
    point_dict['coord_frame'] = FIFF.FIFFV_COORD_HEAD
    point_dict['ident'] = FIFF.FIFFV_POINT_NASION
    point_dict['kind'] = FIFF.FIFFV_POINT_CARDINAL
    point_dict['r'] = nasion
    dig.append(point_dict)

    point_dict = {}
    point_dict['coord_frame'] = FIFF.FIFFV_COORD_HEAD
    point_dict['ident'] = FIFF.FIFFV_POINT_LPA
    point_dict['kind'] = FIFF.FIFFV_POINT_CARDINAL
    point_dict['r'] = lpa
    dig.append(point_dict)

    point_dict = {}
    point_dict['coord_frame'] = FIFF.FIFFV_COORD_HEAD
    point_dict['ident'] = FIFF.FIFFV_POINT_RPA
    point_dict['kind'] = FIFF.FIFFV_POINT_CARDINAL
    point_dict['r'] = rpa
    dig.append(point_dict)

    for idx, point in enumerate(elp_points):
        point_dict = {}
        point_dict['coord_frame'] = FIFF.FIFFV_COORD_HEAD
        point_dict['ident'] = idx
        point_dict['kind'] = FIFF.FIFFV_POINT_HPI
        point_dict['r'] = point
        dig.append(point_dict)

    for idx, point in enumerate(hsp_points):
        point_dict = {}
        point_dict['coord_frame'] = FIFF.FIFFV_COORD_HEAD
        point_dict['ident'] = idx
        point_dict['kind'] = FIFF.FIFFV_POINT_EXTRA
        point_dict['r'] = point
        dig.append(point_dict)

    return dig


def read_mrk(fname):
    """Marker Point Extraction in MEG space directly from sqd

    Parameters
    ----------
    fname : str
        Absolute path to Marker file.
        File formats allowed: *.sqd, *.txt, *.pickled

    Returns
    -------
    mrk_points : numpy.array, shape = (n_points, 3)
        Marker points in MEG space [m].
    """
    ext = path.splitext(fname)[-1]
    if ext == '.sqd':
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


def read_elp(elp_fname):
    """ELP point extraction in Polhemus head space

    Parameters
    ----------
    elp_fname : str
        Absolute path to laser point file acquired from Polhemus system.
        File formats allowed: *.txt

    Returns
    -------
    elp_points : numpy.array, shape = (n_points, 3)
        Fiducial and marker points in Polhemus head space.
    """
    p = re.compile(r'(\-?\d+\.\d+)\s+(\-?\d+\.\d+)\s+(\-?\d+\.\d+)')
    elp_points = p.findall(open(elp_fname).read())
    elp_points = np.array(elp_points, dtype=float)
    if elp_points.shape != (8, 3):
        err = ("File %r does not contain correct number of points for a "
               "fiducials file. Expected shape: (8 points, 3 "
               "coordinates); got shape "
               "%s." % (elp_fname, elp_points.shape))
        raise ValueError(err)
    return elp_points


def read_hsp(hsp_fname):
    """HSP point extraction in Polhemus head space

    Parameters
    ----------
    hsp_fname : str
        Absolute path to headshape file acquired from Polhemus system.

    Returns
    -------
    hsp_points : numpy.array, shape = (n_points, 3)
        Headshape points in Polhemus head space.
        File formats allowed: *.txt, *.pickled
    max_n : None | int
        Maximum number of points. If max_n is not None and the number of
        points in the file exceeds max_n, only n_max points are returned.
    """
    ext = path.splitext(hsp_fname)[-1]
    if ext == '.txt':
        p = re.compile(r'(\-?\d+\.\d+)\s+(\-?\d+\.\d+)\s+(\-?\d+\.\d+)')
        hsp_points = p.findall(open(hsp_fname).read())
        hsp_points = np.array(hsp_points, dtype=float)
        # downsample the digitizer points
    elif ext == '.pickled':
        hsp = pickle.load(open(hsp_fname))
        hsp_points = hsp['hsp']
    else:
        err = ('Polhemus hsp file must be *.txt or *.pickled, not *%s.' % ext)
        raise ValueError(err)
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


def read_sns(sns_fname):
    """Sensor coordinate extraction in MEG space

    Parameters
    ----------
    sns_fname : str
        Absolute path to sensor definition file.

    Returns
    -------
    locs : numpy.array, shape = (n_points, 3)
        Sensor coil location.
    """

    p = re.compile(r'\d,[A-Za-z]*,([\.\-0-9]+),' +
                   r'([\.\-0-9]+),([\.\-0-9]+),' +
                   r'([\.\-0-9]+),([\.\-0-9]+)')
    locs = np.array(p.findall(open(sns_fname).read()), dtype=float)
    return locs


def get_neuromag_transform(nasion, lpa, rpa):
    """Creates a transformation matrix from RAS to Neuromag-like space

    Resets the origin to mid-distance of peri-auricular points with nasion
    passing through y-axis.
    (mne manual, pg. 97)

    Parameters
    ----------
    nasion : numpy.array, shape = (1, 3)
        Nasion point coordinate.
    lpa : numpy.array, shape = (1, 3)
        Left peri-auricular point coordinate.
    rpa : numpy.array, shape = (1, 3)
        Right peri-auricular point coordinate.

    Returns
    -------
    trans : numpy.array, shape = (3, 3)
        Transformation matrix to Neuromag-like space.
    """
    origin = (lpa + rpa) / 2
    nasion = nasion - origin
    lpa = lpa - origin
    rpa = rpa - origin
    axes = np.empty((3, 3))
    axes[1] = nasion / linalg.norm(nasion)
    axes[2] = np.cross(axes[1], lpa - rpa)
    axes[2] /= linalg.norm(axes[2])
    axes[0] = np.cross(axes[1], axes[2])

    trans = linalg.inv(axes)
    return trans


def transform_ALS_to_RAS(pts, unit='mm'):
    """Transform points from a ALS to RAS a coordinate system

    This is used to orient points in Neuromag coordinates. KIT and Polhemus
    points are originally in a ALS (anterior, left, superior) coordinate
    system and ahve to be transformed to the neuromag RAS (right, anterior,
    superior) coordinate system.
    KIT sensors are (x,y,z) in [mm].
    KIT markers are (x,y,z) in [m].
    Polhemus points are (x,y,z) in [mm].
    The transformation to RAS space is -y,x,z in [m].

    Parameters
    ----------
    pts : numpy.array, shape = (n_points, 3)
        Points to be transformed.
    unit : 'mm' | 'm'
        Unit of source points to be converted.

    Returns
    -------
    pts : numpy.array, shape = (n_points, 3)
        Points transformed to Neuromag-like head space (RAS).
    """
    pts = np.array(pts, ndmin=2)
    pts = pts[:, [1, 0, 2]]
    pts[:, 0] *= -1
    if unit == 'mm':
        pts /= 1e3
    elif unit != 'm':
        raise ValueError('The unit must be either "m" or "mm".')
    return pts
