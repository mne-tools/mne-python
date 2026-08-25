"""Shared objects used for type annotations."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from typing import IO, Self

# A Matplotlib color: a named/hex string, or an RGB(A) tuple of floats. This is
# the runtime meaning of the ``color`` numpydoc pseudo-type.
Color = str | tuple
# An open file-like object (a readable/writable stream) rather than a path; the
# runtime meaning of the ``file-like`` numpydoc pseudo-type.
FileLike = IO

__all__ = ["Color", "FileLike", "Self"]
