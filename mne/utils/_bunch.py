# -*- coding: utf-8 -*-
"""Bunch-related classes."""
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#
# License: BSD (3-clause)

from copy import deepcopy
from collections.abc import MutableSequence


###############################################################################
# Create a Bunch class that acts like a struct (mybunch.key = val)

class Bunch(dict):
    """Dictionary-like object that exposes its keys as attributes."""

    def __init__(self, **kwargs):  # noqa: D102
        dict.__init__(self, kwargs)
        self.__dict__ = self


###############################################################################
# A protected version that prevents overwriting

class BunchConst(Bunch):
    """Class to prevent us from re-defining constants (DRY)."""

    def __setattr__(self, attr, val):  # noqa: D105
        if attr != '__dict__' and hasattr(self, attr):
            raise AttributeError('Attribute "%s" already set' % attr)
        super().__setattr__(attr, val)


###############################################################################
# A version that tweaks the __repr__ of its values based on keys

class BunchConstNamed(BunchConst):
    """Class to provide nice __repr__ for our integer constants.

    Only supports string keys and int or float values.
    """

    def __setattr__(self, attr, val):  # noqa: D105
        assert isinstance(attr, str)
        if isinstance(val, int):
            val = NamedInt(attr, val)
        elif isinstance(val, float):
            val = NamedFloat(attr, val)
        else:
            assert isinstance(val, BunchConstNamed), type(val)
        super().__setattr__(attr, val)


class _Named(object):
    """Provide shared methods for giving named-representation subclasses."""

    def __new__(cls, name, val):  # noqa: D102,D105
        out = _named_subclass(cls).__new__(cls, val)
        out._name = name
        return out

    def __str__(self):  # noqa: D105
        return '%s (%s)' % (super().__str__(), self._name)

    __repr__ = __str__

    # see https://stackoverflow.com/a/15774013/2175965
    def __copy__(self):  # noqa: D105
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):  # noqa: D105
        cls = self.__class__
        result = cls.__new__(cls, self._name, self)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def __getnewargs__(self):  # noqa: D105
        return self._name, _named_subclass(self)(self)


def _named_subclass(klass):
    if not isinstance(klass, type):
        klass = klass.__class__
    subklass = klass.mro()[-2]
    assert subklass in (int, float)
    return subklass


class NamedInt(_Named, int):
    """Int with a name in __repr__."""


class NamedFloat(_Named, float):
    """Float with a name in __repr__."""


class MNEObjectsList(MutableSequence):
    """All the bolierplate for a list of specific MNE objects.

    Parameters
    ----------
    elements : list
        A list of Objects objects.

    Attributes
    ----------
    _items : list
        The container
    """

    def __init__(self, elements=None, kls=None):
        if kls is None:
            raise ValueError('kls is necessary')
        if elements is None:
            self._items = list()
        elif all([isinstance(_, kls) for _ in elements]):
            if elements is None:
                self._items = list()
            else:
                self._items = deepcopy(list(elements))
        else:
            # XXX: _msg should not be Digitization related
            _msg = 'Digitization expected a iterable of DigPoint objects.'
            raise ValueError(_msg)

    def __len__(self):  # noqa: D105
        return len(self._items)

    def __getitem__(self, index):  # noqa: D105
        return self._items[index]

    def __setitem__(self, index, value):  # noqa: D105
        self._items[index] = value

    def __delitem__(self, index):  # noqa: D105
        del self._items[index]

    def insert(self, index, value):  # noqa: D102
        self._items.insert(index, value)

    def __eq__(self, other):  # noqa: D105
        # if not isinstance(other, Digitization) or len(self) != len(other):
        if len(self) != len(other):
            return False
        else:
            return all([ss == oo for ss, oo in zip(self, other)])
