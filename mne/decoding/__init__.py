# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

"""Decoding and encoding, including machine learning and receptive fields."""
import lazy_loader as lazy

(__getattr__, __dir__, __all__) = lazy.attach_stub(__name__, __file__)

_lazy_getattr = __getattr__


def __getattr__(name):
    """Load decoding symbols and explain the optional sklearn dependency."""
    try:
        return _lazy_getattr(name)
    except ModuleNotFoundError as exc:
        if exc.name == "sklearn":
            raise ModuleNotFoundError(
                "The mne.decoding module requires scikit-learn; "
                "install it with `pip install scikit-learn`."
            ) from exc
        raise
