"""Coordinate Point Extractor for KIT system."""

# Author: Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

from os import SEEK_CUR, path as op
import pickle
import re
from struct import unpack

import numpy as np

from .constants import KIT
from .._digitization import _read_dig_points


def read_mrk(fname):
    r"""Marker Point Extraction in MEG space directly from sqd.

    Parameters
    ----------
    fname : str
        Absolute path to Marker file.
        File formats allowed: \*.sqd, \*.mrk, \*.txt, \*.pickled.

    Returns
    -------
    mrk_points : ndarray, shape (n_points, 3)
        Marker points in MEG space [m].
    """
    ext = op.splitext(fname)[-1]
    if ext in ('.sqd', '.mrk'):
        with open(fname, 'rb', buffering=0) as fid:
            fid.seek(192)
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
        mrk_points = _read_dig_points(fname, unit='m')
    elif ext == '.pickled':
        with open(fname, 'rb') as fid:
            food = pickle.load(fid)
        try:
            mrk_points = food['mrk']
        except Exception:
            err = ("%r does not contain marker points." % fname)
            raise ValueError(err)
    else:
        raise ValueError('KIT marker file must be *.sqd, *.mrk, *.txt or '
                         '*.pickled, *%s is not supported.' % ext)

    # check output
    mrk_points = np.asarray(mrk_points)
    if mrk_points.shape != (5, 3):
        err = ("%r is no marker file, shape is "
               "%s" % (fname, mrk_points.shape))
        raise ValueError(err)
    return mrk_points


def read_sns(fname):
    """Sensor coordinate extraction in MEG space.

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
