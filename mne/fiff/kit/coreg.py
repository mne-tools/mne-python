"""Coordinate Point Extractor for KIT system"""

# Author: Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

import re
import numpy as np
from mne.fiff.constants import FIFF


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

        mrk_points = self._read_mrk(mrk_fname=mrk_fname)
        self.mrk_points = transform_pts(mrk_points)

        elp_points = self._read_elp(elp_fname=elp_fname)
        self.elp_points = transform_pts(elp_points)

        hsp_points = self._read_hsp(hsp_fname=hsp_fname)
        self.hsp_points = []
        for idx, point in enumerate(hsp_points):
            point_dict = {}
            point_dict['coord_frame'] = FIFF.FIFFV_COORD_HEAD
            point_dict['ident'] = idx + 1
            point_dict['kind'] = FIFF.FIFFV_POINT_CARDINAL
            point = np.array(point, ndmin=2)
            point = transform_pts(point)
            point_dict['r'] = point
            self.hsp_points.append(point_dict)


    def _read_mrk(self, mrk_fname):
        """marker point extraction"""

        # pattern by Tal Linzen:
        p = re.compile(r'Marker \d:   MEG:x= *([\.\-0-9]+), ' +
                       r'y= *([\.\-0-9]+), z= *([\.\-0-9]+)')
        mrk_points = p.findall(open(mrk_fname).read())
        mrk_points = np.array(mrk_points, dtype=float)
        return mrk_points

    def _read_elp(self, elp_fname):
        """elp point extraction"""

        p = re.compile('%N\t\d-[A-Z]+\s+([\.\-0-9]+)\t' +
                       '([\.\-0-9]+)\t([\.\-0-9]+)')
        elp_points = p.findall(open(elp_fname).read())
        elp_points = np.array(elp_points, dtype=float)
        return elp_points

    def _read_hsp(self, hsp_fname):
        """hsp point extraction"""

        p = re.compile(r'//No.+\n(\d*)\t(\d)\s*')
        v = re.split(p, open(hsp_fname).read())[1:]
        hsp_points = np.fromstring(v[-1], sep='\t')
        hsp_points = hsp_points.reshape(int(v[0]), int(v[1]))
        return hsp_points

def transform_pts(pts):
    """KIT-Neuromag transformer

    This is used to orient points in Neuromag coordinates.
    The KIT system is x,y,z in [mm].
    The transformation to Neuromag-like space is -y,x,z in [m].

    """
    pts /= 1e3
    pts = pts[:, [1, 0, 2]]
    pts[:, 0] *= -1
    return pts
