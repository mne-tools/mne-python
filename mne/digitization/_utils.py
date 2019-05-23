# -*- coding: utf-8 -*-
# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Teon Brooks <teon.brooks@gmail.com>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#          Joan Massich <mailsik@gmail.com>
#
# License: BSD (3-clause)

import datetime
import os.path as op
import re

import numpy as np

from mne.io.constants import FIFF
from mne.io.tree import dir_tree_find
from mne.io.tag import read_tag
from mne.io.write import start_file
from mne.io.write import end_file
from mne.io.write import write_dig_points

from mne.transforms import _to_const
from mne.utils import logger
from mne.utils import warn
from mne.utils.check import _check_option
from mne import __version__
from .base import _format_dig_points, Digitization

b = bytes  # alias


def _read_dig_fif(fid, meas_info):
    """Read digitizer data from a FIFF file."""
    isotrak = dir_tree_find(meas_info, FIFF.FIFFB_ISOTRAK)
    dig = None
    if len(isotrak) == 0:
        logger.info('Isotrak not found')
    elif len(isotrak) > 1:
        warn('Multiple Isotrak found')
    else:
        isotrak = isotrak[0]
        dig = []
        for k in range(isotrak['nent']):
            kind = isotrak['directory'][k].kind
            pos = isotrak['directory'][k].pos
            if kind == FIFF.FIFF_DIG_POINT:
                tag = read_tag(fid, pos)
                dig.append(tag.data)
                dig[-1]['coord_frame'] = FIFF.FIFFV_COORD_HEAD
    return _format_dig_points(dig)


def write_dig(fname, pts, coord_frame=None):
    """Write digitization data to a FIF file.

    Parameters
    ----------
    fname : str
        Destination file name.
    pts : iterator of dict
        Iterator through digitizer points. Each point is a dictionary with
        the keys 'kind', 'ident' and 'r'.
    coord_frame : int | str | None
        If all the points have the same coordinate frame, specify the type
        here. Can be None (default) if the points could have varying
        coordinate frames.
    """
    if coord_frame is not None:
        coord_frame = _to_const(coord_frame)
        pts_frames = {pt.get('coord_frame', coord_frame) for pt in pts}
        bad_frames = pts_frames - {coord_frame}
        if len(bad_frames) > 0:
            raise ValueError(
                'Points have coord_frame entries that are incompatible with '
                'coord_frame=%i: %s.' % (coord_frame, str(tuple(bad_frames))))

    with start_file(fname) as fid:
        write_dig_points(fid, pts, block=True, coord_frame=coord_frame)
        end_file(fid)


def _read_dig_points(fname, comments='%', unit='auto'):
    """Read digitizer data from a text file.

    If fname ends in .hsp or .esp, the function assumes digitizer files in [m],
    otherwise it assumes space-delimited text files in [mm].

    Parameters
    ----------
    fname : str
        The filepath of space delimited file with points, or a .mat file
        (Polhemus FastTrak format).
    comments : str
        The character used to indicate the start of a comment;
        Default: '%'.
    unit : 'auto' | 'm' | 'cm' | 'mm'
        Unit of the digitizer files (hsp and elp). If not 'm', coordinates will
        be rescaled to 'm'. Default is 'auto', which assumes 'm' for *.hsp and
        *.elp files and 'mm' for *.txt files, corresponding to the known
        Polhemus export formats.

    Returns
    -------
    dig_points : np.ndarray, shape (n_points, 3)
        Array of dig points in [m].
    """
    _check_option('unit', unit, ['auto', 'm', 'mm', 'cm'])

    _, ext = op.splitext(fname)
    if ext == '.elp' or ext == '.hsp':
        with open(fname) as fid:
            file_str = fid.read()
        value_pattern = r"\-?\d+\.?\d*e?\-?\d*"
        coord_pattern = r"({0})\s+({0})\s+({0})\s*$".format(value_pattern)
        if ext == '.hsp':
            coord_pattern = '^' + coord_pattern
        points_str = [m.groups() for m in re.finditer(coord_pattern, file_str,
                                                      re.MULTILINE)]
        dig_points = np.array(points_str, dtype=float)
    elif ext == '.mat':  # like FastScan II
        from scipy.io import loadmat
        dig_points = loadmat(fname)['Points'].T
    else:
        dig_points = np.loadtxt(fname, comments=comments, ndmin=2)
        if unit == 'auto':
            unit = 'mm'
        if dig_points.shape[1] > 3:
            warn('Found %d columns instead of 3, using first 3 for XYZ '
                 'coordinates' % (dig_points.shape[1],))
            dig_points = dig_points[:, :3]

    if dig_points.shape[-1] != 3:
        err = 'Data must be (n, 3) instead of %s' % (dig_points.shape,)
        raise ValueError(err)

    if unit == 'mm':
        dig_points /= 1000.
    elif unit == 'cm':
        dig_points /= 100.

    return dig_points


def _write_dig_points(fname, dig_points):
    """Write points to text file.

    Parameters
    ----------
    fname : str
        Path to the file to write. The kind of file to write is determined
        based on the extension: '.txt' for tab separated text file.
    dig_points : numpy.ndarray, shape (n_points, 3)
        Points.
    """
    _, ext = op.splitext(fname)
    dig_points = np.asarray(dig_points)
    if (dig_points.ndim != 2) or (dig_points.shape[1] != 3):
        err = ("Points must be of shape (n_points, 3), "
               "not %s" % (dig_points.shape,))
        raise ValueError(err)

    if ext == '.txt':
        with open(fname, 'wb') as fid:
            version = __version__
            now = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
            fid.write(b'%% Ascii 3D points file created by mne-python version'
                      b' %s at %s\n' % (version.encode(), now.encode()))
            fid.write(b'%% %d 3D points, x y z per line\n' % len(dig_points))
            np.savetxt(fid, dig_points, delimiter='\t', newline='\n')
    else:
        msg = "Unrecognized extension: %r. Need '.txt'." % ext
        raise ValueError(msg)


# XXX: all points are supposed to be in FIFFV_COORD_HEAD
def _make_dig_points(nasion=None, lpa=None, rpa=None, hpi=None,
                     extra_points=None, dig_ch_pos=None):
    """Construct digitizer info for the info.

    Parameters
    ----------
    nasion : array-like | numpy.ndarray, shape (3,) | None
        Point designated as the nasion point.
    lpa : array-like |  numpy.ndarray, shape (3,) | None
        Point designated as the left auricular point.
    rpa : array-like |  numpy.ndarray, shape (3,) | None
        Point designated as the right auricular point.
    hpi : array-like | numpy.ndarray, shape (n_points, 3) | None
        Points designated as head position indicator points.
    extra_points : array-like | numpy.ndarray, shape (n_points, 3)
        Points designed as the headshape points.
    dig_ch_pos : dict
        Dict of EEG channel positions.

    Returns
    -------
    dig : Digitization
        A container of DigPoints to be added to the info['dig'].
    """
    dig = []
    if lpa is not None:
        lpa = np.asarray(lpa)
        if lpa.shape != (3,):
            raise ValueError('LPA should have the shape (3,) instead of %s'
                             % (lpa.shape,))
        dig.append({'r': lpa, 'ident': FIFF.FIFFV_POINT_LPA,
                    'kind': FIFF.FIFFV_POINT_CARDINAL,
                    'coord_frame': FIFF.FIFFV_COORD_HEAD})
    if nasion is not None:
        nasion = np.asarray(nasion)
        if nasion.shape != (3,):
            raise ValueError('Nasion should have the shape (3,) instead of %s'
                             % (nasion.shape,))
        dig.append({'r': nasion, 'ident': FIFF.FIFFV_POINT_NASION,
                    'kind': FIFF.FIFFV_POINT_CARDINAL,
                    'coord_frame': FIFF.FIFFV_COORD_HEAD})
    if rpa is not None:
        rpa = np.asarray(rpa)
        if rpa.shape != (3,):
            raise ValueError('RPA should have the shape (3,) instead of %s'
                             % (rpa.shape,))
        dig.append({'r': rpa, 'ident': FIFF.FIFFV_POINT_RPA,
                    'kind': FIFF.FIFFV_POINT_CARDINAL,
                    'coord_frame': FIFF.FIFFV_COORD_HEAD})
    if hpi is not None:
        hpi = np.asarray(hpi)
        if hpi.ndim != 2 or hpi.shape[1] != 3:
            raise ValueError('HPI should have the shape (n_points, 3) instead '
                             'of %s' % (hpi.shape,))
        for idx, point in enumerate(hpi):
            dig.append({'r': point, 'ident': idx + 1,
                        'kind': FIFF.FIFFV_POINT_HPI,
                        'coord_frame': FIFF.FIFFV_COORD_HEAD})
    if extra_points is not None:
        extra_points = np.asarray(extra_points)
        if extra_points.shape[1] != 3:
            raise ValueError('Points should have the shape (n_points, 3) '
                             'instead of %s' % (extra_points.shape,))
        for idx, point in enumerate(extra_points):
            dig.append({'r': point, 'ident': idx + 1,
                        'kind': FIFF.FIFFV_POINT_EXTRA,
                        'coord_frame': FIFF.FIFFV_COORD_HEAD})
    if dig_ch_pos is not None:
        keys = sorted(dig_ch_pos.keys())
        try:  # use the last 3 as int if possible (e.g., EEG001->1)
            idents = [int(key[-3:]) for key in keys]
        except ValueError:  # and if any conversion fails, simply use arange
            idents = np.arange(1, len(keys) + 1)
        for key, ident in zip(keys, idents):
            dig.append({'r': dig_ch_pos[key], 'ident': ident,
                        'kind': FIFF.FIFFV_POINT_EEG,
                        'coord_frame': FIFF.FIFFV_COORD_HEAD})

    return Digitization(_format_dig_points(dig))
