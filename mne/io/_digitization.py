# -*- coding: utf-8 -*-
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#          Teon Brooks <teon.brooks@gmail.com>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#          Joan Massich <mailsik@gmail.com>
#
# License: BSD (3-clause)

from collections import OrderedDict, Counter

import datetime
import os.path as op
import re

import numpy as np

from ..utils import logger, warn, _check_option, Bunch

from .constants import FIFF
from .tree import dir_tree_find
from .tag import read_tag
from .write import (start_file, end_file, write_dig_points)

from ..transforms import (apply_trans, als_ras_trans, Transform,
                          get_ras_to_neuromag_trans, combine_transforms,
                          invert_transform, _to_const, _str_to_frame,
                          _coord_frame_name)

from .. import __version__

b = bytes  # alias

_dig_kind_dict = {
    'cardinal': FIFF.FIFFV_POINT_CARDINAL,
    'hpi': FIFF.FIFFV_POINT_HPI,
    'eeg': FIFF.FIFFV_POINT_EEG,
    'extra': FIFF.FIFFV_POINT_EXTRA,
}
_dig_kind_ints = tuple(sorted(_dig_kind_dict.values()))
_dig_kind_proper = {'cardinal': 'Cardinal',
                    'hpi': 'HPI',
                    'eeg': 'EEG',
                    'extra': 'Extra',
                    'unknown': 'Unknown'}
_dig_kind_rev = {val: key for key, val in _dig_kind_dict.items()}
_cardinal_kind_rev = {1: 'LPA', 2: 'Nasion', 3: 'RPA', 4: 'Inion'}


def _format_dig_points(dig):
    """Format the dig points nicely."""
    return [DigPoint(d) for d in dig] if dig is not None else dig


def _get_dig_eeg(dig):
    return [d for d in dig if d['kind'] == FIFF.FIFFV_POINT_EEG]


def _count_points_by_type(dig):
    """Get the number of points of each type."""
    occurrences = Counter([d['kind'] for d in dig])
    return dict(
        fid=occurrences[FIFF.FIFFV_POINT_CARDINAL],
        hpi=occurrences[FIFF.FIFFV_POINT_HPI],
        eeg=occurrences[FIFF.FIFFV_POINT_EEG],
        extra=occurrences[FIFF.FIFFV_POINT_EXTRA],
    )


class DigPoint(dict):
    """Container for a digitization point.

    This is a simple subclass of the standard dict type designed to provide
    a readable string representation.

    Parameters
    ----------
    kind : int
        The kind of channel,
        e.g. ``FIFFV_POINT_EEG``, ``FIFFV_POINT_CARDINAL``.
    r : array, shape (3,)
        3D position in m. and coord_frame.
    ident : int
        Number specifying the identity of the point.
        e.g.  ``FIFFV_POINT_NASION`` if kind is ``FIFFV_POINT_CARDINAL``,
        or 42 if kind is ``FIFFV_POINT_EEG``.
    coord_frame : int
        The coordinate frame used, e.g. ``FIFFV_COORD_HEAD``.
    """

    def __repr__(self):  # noqa: D105
        if self['kind'] == FIFF.FIFFV_POINT_CARDINAL:
            id_ = _cardinal_kind_rev.get(
                self.get('ident', -1), 'Unknown cardinal')
        else:
            id_ = _dig_kind_proper[
                _dig_kind_rev.get(self.get('kind', -1), 'unknown')]
            id_ = ('%s #%s' % (id_, self.get('ident', -1)))
        id_ = id_.rjust(10)
        cf = _coord_frame_name(self['coord_frame'])
        pos = ('(%0.1f, %0.1f, %0.1f) mm' % tuple(1000 * self['r'])).ljust(25)
        return ('<DigPoint | %s : %s : %s frame>' % (id_, pos, cf))

    # speed up info copy by only deep copying the mutable item
    def __deepcopy__(self, memodict):
        """Make a deepcopy."""
        return DigPoint(
            kind=self['kind'], r=self['r'].copy(),
            ident=self['ident'], coord_frame=self['coord_frame'])

    def __eq__(self, other):  # noqa: D105
        """Compare two DigPoints.

        Two digpoints are equal if they are the same kind, share the same
        coordinate frame and position.
        """
        my_keys = ['kind', 'ident', 'coord_frame']
        if sorted(self.keys()) != sorted(other.keys()):
            return False
        elif any([self[_] != other[_] for _ in my_keys]):
            return False
        else:
            return np.allclose(self['r'], other['r'])


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


_cardinal_ident_mapping = {
    FIFF.FIFFV_POINT_NASION: 'nasion',
    FIFF.FIFFV_POINT_LPA: 'lpa',
    FIFF.FIFFV_POINT_RPA: 'rpa',
}


def _foo_get_data_from_dig(dig):
    # XXXX:
    # This does something really similar to _read_dig_montage_fif but:
    #   - does not check coord_frame
    #   - does not do any operation that implies assumptions with the names

    # Split up the dig points by category
    hsp, hpi, elp = list(), list(), list()
    fids, dig_ch_pos_location = dict(), list()

    for d in dig:
        if d['kind'] == FIFF.FIFFV_POINT_CARDINAL:
            fids[_cardinal_ident_mapping[d['ident']]] = d['r']
        elif d['kind'] == FIFF.FIFFV_POINT_HPI:
            hpi.append(d['r'])
            elp.append(d['r'])
            # XXX: point_names.append('HPI%03d' % d['ident'])
        elif d['kind'] == FIFF.FIFFV_POINT_EXTRA:
            hsp.append(d['r'])
        elif d['kind'] == FIFF.FIFFV_POINT_EEG:
            # XXX: dig_ch_pos['EEG%03d' % d['ident']] = d['r']
            dig_ch_pos_location.append(d['r'])

    dig_coord_frames = set([d['coord_frame'] for d in dig])
    assert len(dig_coord_frames) == 1, \
        'Only single coordinate frame in dig is supported'  # XXX

    return Bunch(
        nasion=fids.get('nasion', None),
        lpa=fids.get('lpa', None),
        rpa=fids.get('rpa', None),
        hsp=np.array(hsp) if len(hsp) else None,
        hpi=np.array(hpi) if len(hpi) else None,
        elp=np.array(elp) if len(elp) else None,
        dig_ch_pos_location=dig_ch_pos_location,
        coord_frame=dig_coord_frames.pop(),
    )


def _get_fid_coords(dig):
    fid_coords = Bunch(nasion=None, lpa=None, rpa=None)
    fid_coord_frames = dict()

    for d in dig:
        if d['kind'] == FIFF.FIFFV_POINT_CARDINAL:
            key = _cardinal_ident_mapping[d['ident']]
            fid_coords[key] = d['r']
            fid_coord_frames[key] = d['coord_frame']

    if len(fid_coord_frames) > 0:
        if set(fid_coord_frames.keys()) != set(['nasion', 'lpa', 'rpa']):
            raise ValueError("Some fiducial points are missing (got %s)." %
                             fid_coords.keys())

        if len(set(fid_coord_frames.values())) > 1:
            raise ValueError(
                'All fiducial points must be in the same coordinate system '
                '(got %s)' % len(fid_coord_frames)
            )

    coord_frame = fid_coord_frames.popitem()[1] if fid_coord_frames else None

    return fid_coords, coord_frame


def _read_dig_points(fname, comments='%', unit='auto'):
    """Read digitizer data from a file.

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
        # XXX: This should be dead code, but is deeply buried in
        #      read_dig_montage. To be deprecated
        # raise RuntimeError('if you are reading isotrak files please use'
        #                    ' read_dig_polhemus_isotrak')
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
        raise ValueError(
            'Data must be of shape (n, 3) instead of %s' % (dig_points.shape,))

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


def _coord_frame_const(coord_frame):
    if not isinstance(coord_frame, str) or coord_frame not in _str_to_frame:
        raise ValueError('coord_frame must be one of %s, got %s'
                         % (sorted(_str_to_frame.keys()), coord_frame))
    return _str_to_frame[coord_frame]


def _make_dig_points(nasion=None, lpa=None, rpa=None, hpi=None,
                     extra_points=None, dig_ch_pos=None,
                     coord_frame='head'):
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
    coord_frame : str
        The coordinate frame of the points. Usually this is "unknown"
        for native digitizer space. Defaults to "head".

    Returns
    -------
    dig : list of dicts
        A container of DigPoints to be added to the info['dig'].
    """
    coord_frame = _coord_frame_const(coord_frame)

    dig = []
    if lpa is not None:
        lpa = np.asarray(lpa)
        if lpa.shape != (3,):
            raise ValueError('LPA should have the shape (3,) instead of %s'
                             % (lpa.shape,))
        dig.append({'r': lpa, 'ident': FIFF.FIFFV_POINT_LPA,
                    'kind': FIFF.FIFFV_POINT_CARDINAL,
                    'coord_frame': coord_frame})
    if nasion is not None:
        nasion = np.asarray(nasion)
        if nasion.shape != (3,):
            raise ValueError('Nasion should have the shape (3,) instead of %s'
                             % (nasion.shape,))
        dig.append({'r': nasion, 'ident': FIFF.FIFFV_POINT_NASION,
                    'kind': FIFF.FIFFV_POINT_CARDINAL,
                    'coord_frame': coord_frame})
    if rpa is not None:
        rpa = np.asarray(rpa)
        if rpa.shape != (3,):
            raise ValueError('RPA should have the shape (3,) instead of %s'
                             % (rpa.shape,))
        dig.append({'r': rpa, 'ident': FIFF.FIFFV_POINT_RPA,
                    'kind': FIFF.FIFFV_POINT_CARDINAL,
                    'coord_frame': coord_frame})
    if hpi is not None:
        hpi = np.asarray(hpi)
        if hpi.ndim != 2 or hpi.shape[1] != 3:
            raise ValueError('HPI should have the shape (n_points, 3) instead '
                             'of %s' % (hpi.shape,))
        for idx, point in enumerate(hpi):
            dig.append({'r': point, 'ident': idx + 1,
                        'kind': FIFF.FIFFV_POINT_HPI,
                        'coord_frame': coord_frame})
    if extra_points is not None:
        extra_points = np.asarray(extra_points)
        if extra_points.shape[1] != 3:
            raise ValueError('Points should have the shape (n_points, 3) '
                             'instead of %s' % (extra_points.shape,))
        for idx, point in enumerate(extra_points):
            dig.append({'r': point, 'ident': idx + 1,
                        'kind': FIFF.FIFFV_POINT_EXTRA,
                        'coord_frame': coord_frame})
    if dig_ch_pos is not None:
        keys = dig_ch_pos.keys()
        if not isinstance(dig_ch_pos, OrderedDict):
            keys = sorted(keys)
        try:  # use the last 3 as int if possible (e.g., EEG001->1)
            idents = []
            for key in keys:
                if not isinstance(key, str):
                    raise ValueError()
                idents.append(int(key[-3:]))
        except ValueError:  # and if any conversion fails, simply use arange
            idents = np.arange(1, len(keys) + 1)
        for key, ident in zip(keys, idents):
            dig.append({'r': dig_ch_pos[key], 'ident': ident,
                        'kind': FIFF.FIFFV_POINT_EEG,
                        'coord_frame': coord_frame})

    return _format_dig_points(dig)


def _call_make_dig_points(nasion, lpa, rpa, hpi, extra, convert=True):
    if convert:
        neuromag_trans = get_ras_to_neuromag_trans(nasion, lpa, rpa)
        nasion = apply_trans(neuromag_trans, nasion)
        lpa = apply_trans(neuromag_trans, lpa)
        rpa = apply_trans(neuromag_trans, rpa)

        if hpi is not None:
            hpi = apply_trans(neuromag_trans, hpi)

        extra = apply_trans(neuromag_trans, extra).astype(np.float32)
    else:
        neuromag_trans = None

    ctf_head_t = Transform(fro='ctf_head', to='head', trans=neuromag_trans)

    info_dig = _make_dig_points(nasion=nasion,
                                lpa=lpa,
                                rpa=rpa,
                                hpi=hpi,
                                extra_points=extra)

    return info_dig, ctf_head_t


##############################################################################
# From mne.io.kit
def _set_dig_kit(mrk, elp, hsp):
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

    Returns
    -------
    dig_points : list
        List of digitizer points for info['dig'].
    dev_head_t : dict
        A dictionary describe the device-head transformation.
    """
    from ..coreg import fit_matched_points, _decimate_points
    from ..io.kit.constants import KIT
    from ..io.kit.coreg import read_mrk

    if isinstance(hsp, str):
        hsp = _read_dig_points(hsp)
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
        elp_points = _read_dig_points(elp)
        if len(elp_points) != 8:
            raise ValueError("File %r should contain 8 points; got shape "
                             "%s." % (elp, elp_points.shape))
        elp = elp_points
    elif len(elp) != 8:
        raise ValueError("ELP should contain 8 points; got shape "
                         "%s." % (elp.shape,))
    if isinstance(mrk, str):
        mrk = read_mrk(mrk)

    mrk = apply_trans(als_ras_trans, mrk)

    nasion, lpa, rpa = elp[:3]
    nmtrans = get_ras_to_neuromag_trans(nasion, lpa, rpa)
    elp = apply_trans(nmtrans, elp)
    hsp = apply_trans(nmtrans, hsp)

    # device head transform
    trans = fit_matched_points(tgt_pts=elp[3:], src_pts=mrk, out='trans')

    nasion, lpa, rpa = elp[:3]
    elp = elp[3:]

    dig_points = _make_dig_points(nasion, lpa, rpa, elp, hsp)
    dev_head_t = Transform('meg', 'head', trans)

    return dig_points, dev_head_t


##############################################################################
# From artemis123 (we have modified the function a bit)
def _artemis123_read_pos(nas, lpa, rpa, hpi, extra):
    # move into MNE head coords
    dig_points, _ = _call_make_dig_points(nas, lpa, rpa, hpi, extra)
    return dig_points


##############################################################################
# From bti
def _make_bti_dig_points(nasion, lpa, rpa, hpi, extra,
                         convert=False, use_hpi=False,
                         bti_dev_t=False, dev_ctf_t=False):

    _hpi = hpi if use_hpi else None
    info_dig, ctf_head_t = _call_make_dig_points(nasion, lpa, rpa, _hpi, extra,
                                                 convert)

    if convert:
        t = combine_transforms(invert_transform(bti_dev_t), dev_ctf_t,
                               'meg', 'ctf_head')
        dev_head_t = combine_transforms(t, ctf_head_t, 'meg', 'head')
    else:
        dev_head_t = Transform('meg', 'head', trans=None)

    return info_dig, dev_head_t, ctf_head_t  # ctf_head_t should not be needed
