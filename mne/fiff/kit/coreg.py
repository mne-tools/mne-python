"""Coordinate Point Extractor for KIT system"""

# Author: Teon Brooks <teon@nyu.edu>
#
# License: BSD (3-clause)

import re
import numpy as np
from mne.fiff.constants import FIFF
from .constants import KIT
from struct import unpack


def coreg(mrk_fname, elp_fname, hsp_fname):
    """Extracts dig points, elp, and mrk points from files needed for coreg.

    Parameters
    ----------
    mrk_fname : str
        Path to marker file (saved as text from MEG160).
    elp_fname : str
        Path to elp digitizer file.
    hsp_fname : str
        Path to hsp headshape file.

    Output
    ----------
    mrk_points : np.array
        Array of 5 points by coordinate (x,y,z) from marker measurement.
    elp_points : np.array
        Array of 5 points by coordinate (x,y,z) from digitizer laser point.
    dig : dict
        A dictionary containing the mrk_points, elp_points, and hsp_points in
        a format used for raw.info.dig.

    """

    mrk_points = read_mrk(mrk_fname=mrk_fname)
    mrk_points = transform_pts(mrk_points)

    elp_points = read_elp(elp_fname=elp_fname)
    elp_points = transform_pts(elp_points)
    nasion = elp_points[0, :]
    lpa = elp_points[1, :]
    rpa = elp_points[2, :]
    elp_points = elp_points[3:, :]
    elp_points = reset_origin(lpa, rpa, elp_points)

    hsp_points = read_hsp(hsp_fname=hsp_fname)
    hsp_points = reset_origin(lpa, rpa, hsp_points)
    hsp_points = transform_pts(hsp_points)
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
    """Marker Point Extraction in MEG space directly from sqd"""
    fid = open(mrk_fname, 'r')
    fid.seek(KIT.MRK_INFO)
    mrk_offset = unpack('i', fid.read(KIT.INT))[0]
    fid.seek(mrk_offset)
    _ = fid.read(KIT.INT)  # match_done
    _ = fid.read(2 * KIT.DOUBLE * 4 ** 2)  # meg_to_mri and mri_to_meg
    mrk_count = unpack('i', fid.read(KIT.INT))[0]
    pts = []
    for _ in range(mrk_count):
        _ = fid.read(KIT.INT * 4)  # mri_mrk_type, meg_mrk_type,
                                    # mri_mrk_done, meg_mrk_done
        _ = fid.read(KIT.DOUBLE * 3)  # mri_marker
        pts.append(np.fromfile(fid, dtype='d', count=3))
    mrk_points = np.array(pts)
    return mrk_points


def read_elp(elp_fname):
    """ELP point extraction in Polhemus head space."""

    p = re.compile(r'(\-?\d+\.\d+)\s+(\-?\d+\.\d+)\s+(\-?\d+\.\d+)')
    elp_points = p.findall(open(elp_fname).read())
    elp_points = np.array(elp_points, dtype=float)
    return elp_points


def read_hsp(hsp_fname):
    """HSP point extraction in Polhemus head space."""

    p = re.compile(r'(\-?\d+\.\d+)\s+(\-?\d+\.\d+)\s+(\-?\d+\.\d+)')
    hsp_points = p.findall(open(hsp_fname).read())
    hsp_points = np.array(hsp_points, dtype=float)
    # downsample the digitizer points
    n_pts = len(hsp_points)
    space = int(n_pts / KIT.DIG_POINTS)
    hsp_points = hsp_points[::space]
    return hsp_points


def read_sns(sns_fname):
    """SNS coordinate extraction in MEG space."""

    p = re.compile(r'\d,[A-Za-z]*,([\.\-0-9]+),' +
                   r'([\.\-0-9]+),([\.\-0-9]+),' +
                   r'([\.\-0-9]+),([\.\-0-9]+)')
    locs = np.array(p.findall(open(sns_fname).read()), dtype=float)
    return locs


def reset_origin(lpa, rpa, pts):
    """Reset origin of head coordinate system.

    Resets the origin to mid-distance of peri-auricular points
    (mne manual, pg. 97)

    """
    origin = (lpa + rpa) / 2
    pts -= origin
    return pts


def transform_pts(pts, scale=True):
    """KIT-Neuromag transformer.

    This is used to orient points in Neuromag coordinates.
    The KIT system is x,y,z in [mm].
    The transformation to Neuromag-like space is -y,x,z in [m].

    """
    if scale:
        pts /= 1e3
    pts = np.array(pts, ndmin=2)
    pts = pts[:, [1, 0, 2]]
    pts[:, 0] *= -1
    return pts
