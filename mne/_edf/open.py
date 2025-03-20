# Maybe we can move this one to utils or something like that.
from pathlib import Path

from mne._fiff.open import _NoCloseRead

from ..utils import _file_like, _validate_type, logger


def __gdf_edf_get_fid(fname, **kwargs):
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
