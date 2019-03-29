# -*- coding: utf-8 -*-
"""Bunch-related classes."""
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

from copy import deepcopy


###############################################################################
# Create a Bunch class that acts like a struct (mybunch.key = val )

class Bunch(dict):
    """Dictionnary-like object thatexposes its keys as attributes."""

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


class Named(object):
    """Provide shared methods for giving named-representation subclasses."""

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


class NamedInt(Named, int):
    """Int with a name in __repr__."""

    def __new__(cls, name, val):  # noqa: D102,D105
        out = int.__new__(cls, val)
        out._name = name
        return out


class NamedFloat(Named, float):
    """Float with a name in __repr__."""

    def __new__(cls, name, val):  # noqa: D102,D105
        out = float.__new__(cls, val)
        out._name = name
        return out
