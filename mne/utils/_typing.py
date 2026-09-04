"""Shared objects used for type annotations."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from typing import IO, Literal, Self

# A Matplotlib color: a named/hex string, or an RGB(A) tuple of floats. This is
# the runtime meaning of the ``color`` numpydoc pseudo-type.
Color = str | tuple

# coordinate frame names
CoordFrameStr = Literal[
    "meg",
    "mri",
    "mri_voxel",
    "head",
    "mri_tal",
    "ras",
    "fs_tal",
    "ctf_head",
    "ctf_meg",
    "unknown",
]

# An open file-like object (a readable/writable stream) rather than a path; the
# runtime meaning of the ``file-like`` numpydoc pseudo-type.
FileLike = IO

# valid arguments for `verbose`
LogLevel = (
    Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", 10, 20, 30, 40, 50]
    | bool
    | None
)

# our standard on_missing args
RaiseWarnIgnore = Literal["raise", "warn", "ignore"]

__all__ = [
    "Color",
    "CoordFrameStr",
    "FileLike",
    "LogLevel",
    "RaiseWarnIgnore",
    "Self",
]
