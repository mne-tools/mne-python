# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause


from .._fiff import _io_dep_getattr


def __getattr__(name):
    """Get attribute by name.

    This function dynamically handles attribute access for an object.

    Parameters
    ----------
    name : str
        The name of the attribute.

    Returns
    -------
    attribute
        The attribute value.
    """
    return _io_dep_getattr(name, "write")
