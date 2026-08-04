"""Bunch-related classes."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from copy import deepcopy
from typing import Any

###############################################################################
# Create a Bunch class that acts like a struct (mybunch.key = val)


class Bunch(dict):
    """Dictionary-like object that exposes its keys as attributes."""

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

    def __getattr__(self, attr: str) -> Any:
        # Keys are stored in the instance dict (``__dict__ is self``), so normal
        # attribute access resolves them and this only runs for genuinely missing
        # names. It exists to tell the type checker that arbitrary keys are
        # accessible as attributes, with ``Any`` as the honest value type.
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {attr!r}"
        )

    def __setattr__(self, attr: str, val: Any) -> None:
        # Keys are set dynamically as attributes (e.g. ``bunch.foo = ...``); this
        # pass-through just declares that to the type checker (it is exactly what
        # ``object.__setattr__`` already does via ``Bunch.__dict__ is self``).
        super().__setattr__(attr, val)


###############################################################################
# A protected version that prevents overwriting


class BunchConst(Bunch):
    """Class to prevent us from re-defining constants (DRY)."""

    def __setitem__(self, key, val):  # noqa: D105
        if key != "__dict__" and key in self:
            raise AttributeError(f"Attribute {repr(key)} already set")
        super().__setitem__(key, val)


###############################################################################
# Named int/float subclasses used as the values of our constant bunches


class _Named:
    """Provide shared methods for giving named-representation subclasses."""

    def __new__(cls, name, val):  # noqa: D102,D105
        out = _named_subclass(cls).__new__(cls, val)
        out._name = name
        return out

    def __str__(self):  # noqa: D105
        return f"{self.__class__.mro()[-2](self)} ({self._name})"

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

    pass  # noqa


class NamedFloat(_Named, float):
    """Float with a name in __repr__."""

    pass  # noqa


###############################################################################
# A version that tweaks the __repr__ of its values based on keys


class BunchConstNamed(BunchConst):
    """Class to provide nice __repr__ for our integer constants.

    Only supports string keys and int or float values.
    """

    def __getattr__(self, attr: str) -> NamedInt | NamedFloat:
        # Constants are populated dynamically and stored in the instance dict, so
        # normal attribute access resolves them and this only runs for genuinely
        # missing names. It exists mostly to tell the type checker the type of
        # the dynamic constants (e.g. FIFF.FIFFV_ASPECT_AVERAGE).
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {attr!r}"
        )

    def __setattr__(self, attr, val):  # noqa: D105
        assert isinstance(attr, str)
        if isinstance(val, int):
            val = NamedInt(attr, val)
        elif isinstance(val, float):
            val = NamedFloat(attr, val)
        else:
            assert isinstance(val, BunchConstNamed), type(val)
        super().__setattr__(attr, val)
