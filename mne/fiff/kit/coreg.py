"""Coordinate Point Extractor for KIT system"""

# Author: Teon Brooks <teon@nyu.edu>
#
# License: BSD (3-clause)

import re
import numpy as np
from mne.fiff.constants import FIFF
from .constants import KIT


class coreg:
    """Extracts dig points, elp, and mrk points from files needed for coreg.

    Attributes
    ----------
    mrk_points : np.array
        array of 5 points by coordinate (x,y,z) from marker measurement
    elp_points : np.array
        array of 5 points by coordinate (x,y,z) from digitizer laser point
    hsp_points : np.array
        array points by coordinate (x, y, z) from digitizer of head shape

    Parameters
    ----------
    mrk_fname : str
        Path to marker file (saved as text from MEG160).
    elp_fname : str
        Path to elp digitizer file.
    hsp_fname : str
        Path to hsp headshape file.

    """
    def __init__(self, mrk_fname, elp_fname, hsp_fname):

        mrk_points = read_mrk(mrk_fname=mrk_fname)
        self.mrk_points = transform_pts(mrk_points)

        elp_points = read_elp(elp_fname=elp_fname)
        elp_points = transform_pts(elp_points)
        self.nasion = elp_points[0, :]
        self.lpa = elp_points[1, :]
        self.rpa = elp_points[2, :]
        self.elp_points = elp_points[3:, :]
        self.elp_points = self.reset_origin(self.elp_points)

        hsp_points = read_hsp(hsp_fname=hsp_fname)
        hsp_points = self.reset_origin(hsp_points)
        self.hsp_points = transform_pts(hsp_points)
        self.dig = []

        point_dict = {}
        point_dict['coord_frame'] = FIFF.FIFFV_COORD_HEAD
        point_dict['ident'] = KIT.NASION_IDENT
        point_dict['kind'] = FIFF.FIFFV_POINT_NASION
        point_dict['r'] = self.nasion
        self.dig.append(point_dict)

        point_dict = {}
        point_dict['coord_frame'] = FIFF.FIFFV_COORD_HEAD
        point_dict['ident'] = KIT.LPA_IDENT
        point_dict['kind'] = FIFF.FIFFV_POINT_LPA
        point_dict['r'] = self.lpa
        self.dig.append(point_dict)

        point_dict = {}
        point_dict['coord_frame'] = FIFF.FIFFV_COORD_HEAD
        point_dict['ident'] = KIT.RPA_IDENT
        point_dict['kind'] = FIFF.FIFFV_POINT_RPA
        point_dict['r'] = self.rpa
        self.dig.append(point_dict)

        for idx, point in enumerate(self.elp_points):
            point_dict = {}
            point_dict['coord_frame'] = FIFF.FIFFV_COORD_HEAD
            point_dict['ident'] = idx
            point_dict['kind'] = FIFF.FIFFV_POINT_HPI
            point_dict['r'] = point
            self.dig.append(point_dict)

        for idx, point in enumerate(self.hsp_points):
            point_dict = {}
            point_dict['coord_frame'] = FIFF.FIFFV_COORD_HEAD
            point_dict['ident'] = idx
            point_dict['kind'] = FIFF.FIFFV_POINT_EXTRA
            point_dict['r'] = point
            self.dig.append(point_dict)

    def reset_origin(self, pts):
        """reset origin of head coordinate system

        Resets the origin to mid-distance of peri-auricular points
        (mne manual, pg. 97)

        """
        origin = (self.lpa + self.rpa) / 2
        pts -= origin
        return pts


def read_mrk(mrk_fname):
    """marker point extraction"""

    # pattern by Tal Linzen:
    p = re.compile(r'Marker \d:   MEG:x= *([\.\-0-9]+), ' +
                   r'y= *([\.\-0-9]+), z= *([\.\-0-9]+)')
    mrk_points = p.findall(open(mrk_fname).read())
    mrk_points = np.array(mrk_points, dtype=float)
    return mrk_points


def read_elp(elp_fname):
    """elp point extraction"""

    p = re.compile(r'(\-?\d+\.\d+)\s+(\-?\d+\.\d+)\s+(\-?\d+\.\d+)')
    elp_points = p.findall(open(elp_fname).read())
    elp_points = np.array(elp_points, dtype=float)
    return elp_points


def read_hsp(hsp_fname):
    """hsp point extraction"""

    p = re.compile(r'(\-?\d+\.\d+)\s+(\-?\d+\.\d+)\s+(\-?\d+\.\d+)')
    hsp_points = p.findall(open(hsp_fname).read())
    hsp_points = np.array(hsp_points, dtype=float)
    return hsp_points


def read_sns(sns_fname):
    """sns coordinate extraction"""

    p = re.compile(r'\d,[A-Za-z]*,([\.\-0-9]+),' +
                   r'([\.\-0-9]+),([\.\-0-9]+),' +
                   r'([\.\-0-9]+),([\.\-0-9]+)')
    locs = np.array(p.findall(open(sns_fname).read()), dtype=float)
    return locs


def transform_pts(pts):
    """KIT-Neuromag transformer

    This is used to orient points in Neuromag coordinates.
    The KIT system is x,y,z in [mm].
    The transformation to Neuromag-like space is -y,x,z in [m].

    """
    pts /= 1e3
    pts = np.array(pts, ndmin=2)
    pts = pts[:, [1, 0, 2]]
    pts[:, 0] *= -1
    return pts
