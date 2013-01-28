'''
Created on Jan 28, 2013

@author: teon
pattern matching for coreg is from Tal Linzen
coreg methods are adapted from Christian Brodbeck's eelbrain.plot.coreg
'''

import re
import numpy as np
from numpy import sin, cos
from scipy.optimize import leastsq
from mne.fiff.constants import FIFF


class coreg:
    """
    Extracts digitizer points from file.
    Creates coreg transformation matrix from device to head coord.

    Attributes
    ----------
    mrk_points : np.array
        array of 5 points by coordinate (x,y,z) from marker measurement
    elp_points : np.array
        array of 5 points by coordinate (x,y,z) from digitizer laser point
    hsp_points : np.array
        array points by coordinate (x, y, z) from digitizer

    Parameters
    ----------
    mrk_fname : str
        Path to marker avg file (saved as text form MEG160).
    elp_fname : str
        Path to elp digitizer file.

    """
    def __init__(self, mrk_fname, elp_fname, hsp_fname):

        # marker point extraction
        self.mrk_src_path = mrk_fname
        # pattern by Tal:
        p = re.compile(r'Marker \d:   MEG:x= *([\.\-0-9]+), ' +
                       r'y= *([\.\-0-9]+), z= *([\.\-0-9]+)')
        str_points = p.findall(open(mrk_fname).read())
        self.mrk_points = self.transform_pts(np.array(str_points, dtype=float))
        # elp point extraction
        self.elp_src_path = elp_fname
        # pattern modified from Tal's mrk pattern:
        p = re.compile('%N\t\d-[A-Z]+\s+([\.\-0-9]+)\t' +
                       '([\.\-0-9]+)\t([\.\-0-9]+)')
        str_points = p.findall(open(elp_fname).read())
        self.elp_points = self.transform_pts(np.array(str_points, dtype=float))
        # hsp point extraction
        self.hsp_src_path = hsp_fname
        p = re.compile(r'//No.+\n(\d*)\t(\d)\s*')
        v = re.split(p, open(hsp_fname).read())[1:]
        hsp_points = np.fromstring(v[-1], sep='\t').reshape(int(v[0]),
                                                            int(v[1]))
        self.hsp_points = []
        for idx, point in enumerate(hsp_points):
            point_dict = {}
            point_dict['coord_frame'] = FIFF.FIFFV_COORD_HEAD
            point_dict['ident'] = idx + 1
            # equivalent in value but may not be the proper constant
            point_dict['kind'] = FIFF.FIFFV_POINT_CARDINAL
            point_dict['r'] = point
            self.hsp_points.append(point_dict)

    def transform_pts(self, pts):
        pts /= 1e3
        pts = pts[:, [1, 0, 2]]
        pts[:, 0] *= -1

    def fit(self, include=range(5)):
        """
        Fit the marker points to the digitizer points.

        Parameters
        ----------
        include : index (numpy compatible)
            Which points to include in the fit. Index should select among
            points [0, 1, 2, 3, 4].

        """
        def err(params):
            """calculates distance from target and estimate"""

            T = self.trans(*params[:3]) * self.rot(*params[3:])
            pts = T * np.vstack((self.elp_points[include].T,
                       np.ones(len(self.elp_points[include]))))
            est = np.array(pts[:3].T)
            tgt = np.array(self.mrk_points[include])
            return (tgt - est).ravel()

        # initial guess
        params = (0, 0, 0, 0, 0, 0)
        params, _ = leastsq(err, params)
        self.est_params = params
        # head-to-device
        T = self.trans(*params[:3]) * self.rot(*params[3:])
        # returns dev2head by applying the inverse
        return np.array(T.I)

    def trans(self, x=0, y=0, z=0):
        "MNE manual p. 95, a method for translating a matrix"

        m = np.matrix([[1, 0, 0, x],
                       [0, 1, 0, y],
                       [0, 0, 1, z],
                       [0, 0, 0, 1]], dtype=float)
        return m

    def rot(self, x=0, y=0, z=0):
        "From eelbrain.plot.coreg, a method for rotating a matrix"
        c_x = cos(x); c_y = cos(y); c_z = cos(z); s_x = sin(x); s_y = sin(y); s_z = sin(z);
        r = np.matrix([[c_y * c_z, -c_x * s_z + s_x * s_y * c_z,
                        s_x * s_z + c_x * s_y * c_z, 0],
                       [c_y * s_z, c_x * c_z + s_x * s_y * s_z,
                        - s_x * c_z + c_x * s_y * s_z, 0],
                       [-s_y, s_x * c_y, c_x * c_y, 0],
                       [0, 0, 0, 1]], dtype=float)
        return r
