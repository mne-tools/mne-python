# -*- coding: utf-8 -*-
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#
# License: BSD (3-clause)
import numpy as np
from copy import deepcopy
from collections import Counter

from ..transforms import _coord_frame_name
from ..io.constants import FIFF

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
    dig_points = [DigPoint(d) for d in dig] if dig is not None else dig
    return Digitization(dig_points)


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


class Digitization(list):
    """Represent a list of DigPoint objects.

    Parameters
    ----------
    elements : list | None
        A list of DigPoint objects.
    """

    def __init__(self, elements=None):

        elements = list() if elements is None else elements

        if not all([isinstance(_, DigPoint) for _ in elements]):
            _msg = 'Digitization expected a iterable of DigPoint objects.'
            raise ValueError(_msg)
        else:
            super(Digitization, self).__init__(deepcopy(elements))

    def __eq__(self, other):  # noqa: D105
        if not isinstance(other, (Digitization, list)) or \
           len(self) != len(other):
            return False
        else:
            return all([ss == oo for ss, oo in zip(self, other)])
