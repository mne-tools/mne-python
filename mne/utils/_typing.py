"""Shared objects used for type annotations."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import sys
from typing import IO

if sys.version_info >= (3, 11):
    from typing import Self
else:
    # TODO VERSION: Remove this when Python 3.11+ is required (use typing.Self)
    from typing_extensions import Self

# A Matplotlib color: a named/hex string, or an RGB(A) tuple of floats. This is
# the runtime meaning of the ``color`` numpydoc pseudo-type.
Color = str | tuple
# An open file-like object (a readable/writable stream) rather than a path; the
# runtime meaning of the ``file-like`` numpydoc pseudo-type.
FileLike = IO

__all__ = ["Color", "FileLike", "Self"]
