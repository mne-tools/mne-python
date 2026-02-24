# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

from ..._fiff.open import _NoCloseRead
from ...utils import _file_like, _validate_type, logger


def _gdf_edf_get_fid(fname, **kwargs):
    """Open a EDF/BDF/GDF file with no additional parsing."""
    if _file_like(fname):
        logger.debug("Using file-like I/O")
        fid = _NoCloseRead(fname)
        fid.seek(0)
    else:
        _validate_type(fname, [Path, str], "fname", extra="or file-like")
        logger.debug("Using normal I/O")
        fid = open(fname, "rb", **kwargs)  # Open in binary mode
    return fid
