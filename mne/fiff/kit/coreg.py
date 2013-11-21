"""Coordinate Point Extractor for KIT system"""

# Author: Teon Brooks <teon@nyu.edu>
#
# License: BSD (3-clause)

from struct import unpack
from os import SEEK_CUR, path
import re
import cPickle as pickle
import numpy as np
from scipy import linalg
from ..constants import FIFF
from ...transforms.transforms import apply_trans, rotation, translation
from .constants import KIT


def get_points(mrk_fname, elp_fname, hsp_fname):
    """Extracts dig points, elp, and mrk points from files needed for coreg

    Parameters
    ----------
    mrk_fname : str
        Path to marker file (saved as text from MEG160).
    elp_fname : str
        Path to elp digitizer file.
    hsp_fname : str
        Path to hsp headshape file.

    Returns
    -------
    mrk_points : numpy.array, shape = (n_points, 3)
        Array of 5 points by coordinate (x,y,z) from marker measurement.
    elp_points : numpy.array, shape = (n_points, 3)
        Array of 5 points by coordinate (x,y,z) from digitizer laser point.
    dig : dict
        A dictionary containing the mrk_points, elp_points, and hsp_points in
        a format used for raw.info['dig'].
    """

    mrk_points = read_mrk(mrk_fname=mrk_fname)
    mrk_points = transform_pts(mrk_points, unit='m')

    elp_points = read_elp(elp_fname=elp_fname)
    elp_points = transform_pts(elp_points)
    nasion = elp_points[0, :]
    lpa = elp_points[1, :]
    rpa = elp_points[2, :]

    trans = get_neuromag_transform(lpa, rpa, nasion)
    elp_points = np.dot(elp_points, trans.T)
    nasion = elp_points[0]
    lpa = elp_points[1]
    rpa = elp_points[2]
    elp_points = elp_points[3:]

    hsp_points = read_hsp(hsp_fname=hsp_fname)
    hsp_points = transform_pts(hsp_points)
    hsp_points = np.dot(hsp_points, trans.T)
    dig = []

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
    return mrk_points, elp_points, dig


def read_mrk(mrk_fname):
    """Marker Point Extraction in MEG space directly from sqd

    Parameters
    ----------
    mrk_fname : str
        Absolute path to Marker file.
        File formats allowed: *.sqd, *.txt, *.pickled

    Returns
    -------
    mrk_points : numpy.array, shape = (n_points, 3)
        Marker points in MEG space [m].
    """
    ext = path.splitext(mrk_fname)[-1]
    if ext == '.sqd':
        with open(mrk_fname, 'r') as fid:
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
    elif ext == '.hpi':
        mrk_points = np.loadtxt(mrk_fname)
    elif ext == '.pickled':
        mrk = pickle.load(open(mrk_fname))
        mrk_points = mrk['points']
    else:
        raise TypeError('File must be *.sqd, *.hpi or *.pickled.')
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
    """
    ext = path.splitext(hsp_fname)[-1]
    if ext == '.txt':
        p = re.compile(r'(\-?\d+\.\d+)\s+(\-?\d+\.\d+)\s+(\-?\d+\.\d+)')
        hsp_points = p.findall(open(hsp_fname).read())
        hsp_points = np.array(hsp_points, dtype=float)
        # downsample the digitizer points
        n_pts = len(hsp_points)
        if n_pts > KIT.DIG_POINTS:
            space = int(n_pts / KIT.DIG_POINTS)
            hsp_points = np.copy(hsp_points[::space])
    elif ext == '.pickled':
        hsp = pickle.load(open(hsp_fname))
        hsp_points = hsp['points']
    else:
        raise TypeError('File must be either *.txt or *.pickled.')
    return hsp_points


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


def get_neuromag_transform(lpa, rpa, nasion):
    """Creates a transformation matrix from RAS to Neuromag-like space

    Resets the origin to mid-distance of peri-auricular points with nasion
    passing through y-axis.
    (mne manual, pg. 97)

    Parameters
    ----------
    lpa : numpy.array, shape = (1, 3)
        Left peri-auricular point coordinate.
    rpa : numpy.array, shape = (1, 3)
        Right peri-auricular point coordinate.
    nasion : numpy.array, shape = (1, 3)
        Nasion point coordinate.

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


def transform_pts(pts, unit='mm'):
    """Transform KIT and Polhemus points to RAS coordinate system

    This is used to orient points in Neuromag coordinates.
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
    if unit == 'mm':
        pts = pts / 1e3
    elif unit != 'm':
        raise ValueError('The unit must be either "m" or "mm".')
    pts = np.array(pts, ndmin=2)
    pts = pts[:, [1, 0, 2]]
    pts[:, 0] = pts[:, 0] * -1
    return pts
