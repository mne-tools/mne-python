# -*- coding: utf-8 -*-
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#          Teon Brooks <teon.brooks@gmail.com>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#          Joan Massich <mailsik@gmail.com>
#
# License: BSD-3-Clause

import heapq
from collections import Counter

import datetime
import os.path as op

import numpy as np

from ..utils import logger, warn, Bunch, _validate_type

from .constants import FIFF, _coord_frame_named
from .tree import dir_tree_find
from .tag import read_tag
from .write import (start_file, end_file, write_dig_points)

from ..transforms import (apply_trans, Transform,
                          get_ras_to_neuromag_trans, combine_transforms,
                          invert_transform, _to_const, _str_to_frame,
                          _coord_frame_name)
from .. import __version__

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


def _format_dig_points(dig, enforce_order=False):
    """Format the dig points nicely."""
    if enforce_order and dig is not None:
        # reorder points based on type:
        # Fiducials/HPI, EEG, extra (headshape)
        fids_digpoints = []
        hpi_digpoints = []
        eeg_digpoints = []
        extra_digpoints = []
        head_digpoints = []

        # use a heap to enforce order on FIDS, EEG, Extra
        for idx, digpoint in enumerate(dig):
            ident = digpoint['ident']
            kind = digpoint['kind']

            # push onto heap based on 'ident' (for the order) for
            # each of the possible DigPoint 'kind's
            # keep track of 'idx' in case of any clashes in
            # the 'ident' variable, which can occur when
            # user passes in DigMontage + DigMontage
            if kind == FIFF.FIFFV_POINT_CARDINAL:
                heapq.heappush(fids_digpoints, (ident, idx, digpoint))
            elif kind == FIFF.FIFFV_POINT_HPI:
                heapq.heappush(hpi_digpoints, (ident, idx, digpoint))
            elif kind == FIFF.FIFFV_POINT_EEG:
                heapq.heappush(eeg_digpoints, (ident, idx, digpoint))
            elif kind == FIFF.FIFFV_POINT_EXTRA:
                heapq.heappush(extra_digpoints, (ident, idx, digpoint))
            elif kind == FIFF.FIFFV_POINT_HEAD:
                heapq.heappush(head_digpoints, (ident, idx, digpoint))

        # now recreate dig based on sorted order
        fids_digpoints.sort(), hpi_digpoints.sort()
        eeg_digpoints.sort()
        extra_digpoints.sort(), head_digpoints.sort()
        new_dig = []
        for idx, d in enumerate(fids_digpoints + hpi_digpoints +
                                extra_digpoints + eeg_digpoints +
                                head_digpoints):
            new_dig.append(d[-1])
        dig = new_dig

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


_dig_keys = {'kind', 'ident', 'r', 'coord_frame'}


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
            id_ = _cardinal_kind_rev.get(self['ident'], 'Unknown cardinal')
        else:
            id_ = _dig_kind_proper[
                _dig_kind_rev.get(self['kind'], 'unknown')]
            id_ = ('%s #%s' % (id_, self['ident']))
        id_ = id_.rjust(10)
        cf = _coord_frame_name(self['coord_frame'])
        if 'voxel' in cf:
            pos = ('(%0.1f, %0.1f, %0.1f)' % tuple(self['r'])).ljust(25)
        else:
            pos = ('(%0.1f, %0.1f, %0.1f) mm' %
                   tuple(1000 * self['r'])).ljust(25)
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
        if set(self.keys()) != set(other.keys()):
            return False
        elif any(self[_] != other[_] for _ in my_keys):
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
        coord_frame = FIFF.FIFFV_COORD_HEAD
        dig = []
        for k in range(isotrak['nent']):
            kind = isotrak['directory'][k].kind
            pos = isotrak['directory'][k].pos
            if kind == FIFF.FIFF_DIG_POINT:
                tag = read_tag(fid, pos)
                dig.append(tag.data)
            elif kind == FIFF.FIFF_MNE_COORD_FRAME:
                tag = read_tag(fid, pos)
                coord_frame = _coord_frame_named.get(int(tag.data))
        for d in dig:
            d['coord_frame'] = coord_frame
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


# XXXX:
# This does something really similar to _read_dig_montage_fif but:
#   - does not check coord_frame
#   - does not do any operation that implies assumptions with the names
def _get_data_as_dict_from_dig(dig, exclude_ref_channel=True):
    """Obtain coordinate data from a Dig.

    Parameters
    ----------
    dig : list of dicts
        A container of DigPoints to be added to the info['dig'].

    Returns
    -------
    ch_pos : dict
        The container of all relevant channel positions inside dig.
    """
    # Split up the dig points by category
    hsp, hpi, elp = list(), list(), list()
    fids, dig_ch_pos_location = dict(), list()

    for d in dig:
        if d['kind'] == FIFF.FIFFV_POINT_CARDINAL:
            fids[_cardinal_ident_mapping[d['ident']]] = d['r']
        elif d['kind'] == FIFF.FIFFV_POINT_HPI:
            hpi.append(d['r'])
            elp.append(d['r'])
        elif d['kind'] == FIFF.FIFFV_POINT_EXTRA:
            hsp.append(d['r'])
        elif d['kind'] == FIFF.FIFFV_POINT_EEG:
            if d['ident'] != 0 or not exclude_ref_channel:
                dig_ch_pos_location.append(d['r'])

    dig_coord_frames = set([d['coord_frame'] for d in dig])
    if len(dig_coord_frames) != 1:
        raise RuntimeError('Only single coordinate frame in dig is supported, '
                           f'got {dig_coord_frames}')

    return Bunch(
        nasion=fids.get('nasion', None),
        lpa=fids.get('lpa', None),
        rpa=fids.get('rpa', None),
        hsp=np.array(hsp) if len(hsp) else None,
        hpi=np.array(hpi) if len(hpi) else None,
        elp=np.array(elp) if len(elp) else None,
        dig_ch_pos_location=np.array(dig_ch_pos_location),
        coord_frame=dig_coord_frames.pop(),
    )


def _get_fid_coords(dig, raise_error=True):
    fid_coords = Bunch(nasion=None, lpa=None, rpa=None)
    fid_coord_frames = dict()

    for d in dig:
        if d['kind'] == FIFF.FIFFV_POINT_CARDINAL:
            key = _cardinal_ident_mapping[d['ident']]
            fid_coords[key] = d['r']
            fid_coord_frames[key] = d['coord_frame']

    if len(fid_coord_frames) > 0 and raise_error:
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
        if len(extra_points) and extra_points.shape[1] != 3:
            raise ValueError('Points should have the shape (n_points, 3) '
                             'instead of %s' % (extra_points.shape,))
        for idx, point in enumerate(extra_points):
            dig.append({'r': point, 'ident': idx + 1,
                        'kind': FIFF.FIFFV_POINT_EXTRA,
                        'coord_frame': coord_frame})
    if dig_ch_pos is not None:
        try:  # use the last 3 as int if possible (e.g., EEG001->1)
            idents = []
            for key in dig_ch_pos:
                _validate_type(key, str, 'dig_ch_pos')
                idents.append(int(key[-3:]))
        except ValueError:  # and if any conversion fails, simply use arange
            idents = np.arange(1, len(dig_ch_pos) + 1)
        for key, ident in zip(dig_ch_pos, idents):
            dig.append({'r': dig_ch_pos[key], 'ident': int(ident),
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
