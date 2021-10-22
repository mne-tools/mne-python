"""Coordinate Point Extractor for KIT system."""

# Author: Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD-3-Clause

from collections import OrderedDict
from os import SEEK_CUR, path as op
import pickle
import re

import numpy as np

from .constants import KIT, FIFF
from .._digitization import _make_dig_points
from ...transforms import (Transform, apply_trans, get_ras_to_neuromag_trans,
                           als_ras_trans)
from ...utils import warn, _check_option


INT32 = '<i4'
FLOAT64 = '<f8'


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
    from .kit import _read_dirs
    ext = op.splitext(fname)[-1]
    if ext in ('.sqd', '.mrk'):
        with open(fname, 'rb', buffering=0) as fid:
            dirs = _read_dirs(fid)
            fid.seek(dirs[KIT.DIR_INDEX_COREG]['offset'])
            # skips match_done, meg_to_mri and mri_to_meg
            fid.seek(KIT.INT + (2 * KIT.DOUBLE * 16), SEEK_CUR)
            mrk_count = np.fromfile(fid, INT32, 1)[0]
            pts = []
            for _ in range(mrk_count):
                # mri_type, meg_type, mri_done, meg_done
                _, _, _, meg_done = np.fromfile(fid, INT32, 4)
                _, meg_pts = np.fromfile(fid, FLOAT64, 6).reshape(2, 3)
                if meg_done:
                    pts.append(meg_pts)
            mrk_points = np.array(pts)
    elif ext == '.txt':
        mrk_points = _read_dig_kit(fname, unit='m')
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


def _set_dig_kit(mrk, elp, hsp, eeg):
    """Add landmark points and head shape data to the KIT instance.

    Digitizer data (elp and hsp) are represented in [mm] in the Polhemus
    ALS coordinate system. This is converted to [m].

    Parameters
    ----------
    mrk : None | str | array_like, shape (5, 3)
        Marker points representing the location of the marker coils with
        respect to the MEG Sensors, or path to a marker file.
    elp : None | str | array_like, shape (8, 3)
        Digitizer points representing the location of the fiducials and the
        marker coils with respect to the digitized head shape, or path to a
        file containing these points.
    hsp : None | str | array, shape (n_points, 3)
        Digitizer head shape points, or path to head shape file. If more
        than 10`000 points are in the head shape, they are automatically
        decimated.
    eeg : dict
        Ordered dict of EEG dig points.

    Returns
    -------
    dig_points : list
        List of digitizer points for info['dig'].
    dev_head_t : dict
        A dictionary describe the device-head transformation.
    hpi_results : list
        The hpi results.
    """
    from ...coreg import fit_matched_points, _decimate_points

    if isinstance(hsp, str):
        hsp = _read_dig_kit(hsp)
    n_pts = len(hsp)
    if n_pts > KIT.DIG_POINTS:
        hsp = _decimate_points(hsp, res=0.005)
        n_new = len(hsp)
        warn("The selected head shape contained {n_in} points, which is "
             "more than recommended ({n_rec}), and was automatically "
             "downsampled to {n_new} points. The preferred way to "
             "downsample is using FastScan.".format(
                 n_in=n_pts, n_rec=KIT.DIG_POINTS, n_new=n_new))

    if isinstance(elp, str):
        elp_points = _read_dig_kit(elp)
        if len(elp_points) != 8:
            raise ValueError("File %r should contain 8 points; got shape "
                             "%s." % (elp, elp_points.shape))
        elp = elp_points
    elif len(elp) not in (6, 7, 8):
        raise ValueError("ELP should contain 6 ~ 8 points; got shape "
                         "%s." % (elp.shape,))
    if isinstance(mrk, str):
        mrk = read_mrk(mrk)

    mrk = apply_trans(als_ras_trans, mrk)

    nasion, lpa, rpa = elp[:3]
    nmtrans = get_ras_to_neuromag_trans(nasion, lpa, rpa)
    elp = apply_trans(nmtrans, elp)
    hsp = apply_trans(nmtrans, hsp)
    eeg = OrderedDict((k, apply_trans(nmtrans, p)) for k, p in eeg.items())

    # device head transform
    trans = fit_matched_points(tgt_pts=elp[3:], src_pts=mrk, out='trans')

    nasion, lpa, rpa = elp[:3]
    elp = elp[3:]

    dig_points = _make_dig_points(nasion, lpa, rpa, elp, hsp, dig_ch_pos=eeg)
    dev_head_t = Transform('meg', 'head', trans)

    hpi_results = [dict(dig_points=[
        dict(ident=ci, r=r, kind=FIFF.FIFFV_POINT_HPI,
             coord_frame=FIFF.FIFFV_COORD_UNKNOWN)
        for ci, r in enumerate(mrk)], coord_trans=dev_head_t)]

    return dig_points, dev_head_t, hpi_results


def _read_dig_kit(fname, unit='auto'):
    # Read dig points from a file and return ndarray, using FastSCAN for .txt
    from ...channels.montage import (
        read_polhemus_fastscan, read_dig_polhemus_isotrak, read_custom_montage,
        _check_dig_shape)
    assert unit in ('auto', 'm', 'mm')
    _, ext = op.splitext(fname)
    _check_option('file extension', ext[1:], ('hsp', 'elp', 'mat', 'txt'))
    if ext == '.txt':
        unit = 'mm' if unit == 'auto' else unit
        out = read_polhemus_fastscan(fname, unit=unit,
                                     on_header_missing='ignore')
    elif ext in ('.hsp', '.elp'):
        unit = 'm' if unit == 'auto' else unit
        mon = read_dig_polhemus_isotrak(fname, unit=unit)
        if fname.endswith('.hsp'):
            dig = [d['r'] for d in mon.dig
                   if d['kind'] != FIFF.FIFFV_POINT_CARDINAL]
        else:
            dig = [d['r'] for d in mon.dig]
            if dig and \
                    mon.dig[0]['kind'] == FIFF.FIFFV_POINT_CARDINAL and \
                    mon.dig[0]['ident'] == FIFF.FIFFV_POINT_LPA:
                # LPA, Nasion, RPA -> NLR
                dig[:3] = [dig[1], dig[0], dig[2]]
        out = np.array(dig, float)
    else:
        assert ext == '.mat'
        out = np.array([d['r'] for d in read_custom_montage(fname).dig])
    _check_dig_shape(out)
    return out
