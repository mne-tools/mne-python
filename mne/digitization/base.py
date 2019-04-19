# -*- coding: utf-8 -*-
# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#
# License: BSD (3-clause)
from collections.abc import MutableMapping, MutableSequence
from mne.transforms import _coord_frame_name
from mne.io.constants import FIFF
from copy import deepcopy

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


class DigPoint(dict):
    """Container for a digitization point.

    This is a simple subclass of the standard dict type designed to provide
    a readable string representation.

    Parameters
    ----------
    kind : int
        Digitization kind, e.g. ``FIFFV_POINT_EXTRA``.
    ident : int
        Identifier.
    r : ndarray, shape (3,)
        Position.
    coord_frame : int
        Coordinate frame, e.g. ``FIFFV_COORD_HEAD``.
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


class Digitization(MutableSequence):
    """Represent a list of DigPoint objects.

    Parameters
    ----------
    elements : list
        A list of DigPoint objects.

    """
    def __init__(self, elements=None):
        if elements is None:
            self._items = list()
        elif all([isinstance(_, DigPoint) for _ in elements]):
            if elements is None:
                self._items = list()
            else:
                self._items = deepcopy(list(elements))
        else:
            _msg = 'Digitization expected a iterable of DigPoint objects.'
            raise ValueError(_msg)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]

    def __setitem__(self, index, value):
        self._items[index] = value

    def __delitem__(self, index, value):
        del self._items[index]

    def insert(self, index, value):
        self._items.insert(index, value)

    def __repr__(self):
        return self._items.__repr__()
