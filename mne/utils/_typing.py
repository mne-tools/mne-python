"""Shared objects used for type annotations."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from collections.abc import Sequence
from typing import IO, TYPE_CHECKING, Annotated, Literal, Self, TypeAlias, TypeVar

import numpy as np

if TYPE_CHECKING:
    from ..bem import ConductorModel
    from ..epochs import BaseEpochs
    from ..evoked import Evoked
    from ..io import BaseRaw

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

EEGSensor = Literal["eeg", "ecog", "seeg", "dbs"]
MEGSensor = Literal["grad", "mag", "planar1", "planar2"]

# our standard on_missing args
RaiseWarnIgnore = Literal["raise", "warn", "ignore"]

# for type-hinting funcs that take in an instance and return (a copy of) the same type
RawEpoEvkT = TypeVar("RawEpoEvkT", "BaseRaw", "BaseEpochs", "Evoked")

# all the possible ways of specifying the head sphere
SphereT: TypeAlias = (
    float  # radius
    | Annotated[Sequence[float], 4]  # x, y, z, radius
    | np.ndarray[tuple[Literal[4]], np.dtype[np.floating]]  # x, y, z, radius
    | "ConductorModel"
    | Literal["auto", "cardinal", "eeg", "extra", "hpi", "eeglab"]
    | list[Literal["cardinal", "eeg", "extra", "hpi"]]
    | None
)

__all__ = [
    "Color",
    "CoordFrameStr",
    "EEGSensor",
    "FileLike",
    "LogLevel",
    "MEGSensor",
    "RaiseWarnIgnore",
    "RawEpoEvkT",
    "Self",
    "SphereT",
]
