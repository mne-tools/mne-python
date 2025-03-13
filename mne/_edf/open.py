# Maybe we can move this one to utils or ssomething like that.
from io import BytesIO
from pathlib import Path

from mne._fiff.open import _NoCloseRead

from ..utils import _file_like, _validate_type, logger, warn


def __gdf_edf_get_fid(fname):
    """Open a EDF/BDF/GDF file with no additional parsing."""
    if _file_like(fname):
        logger.debug("Using file-like I/O")
        fid = _NoCloseRead(fname)
        fid.seek(0)
    else:
        _validate_type(fname, [Path, str], "fname", extra="or file-like")
        logger.debug("Using normal I/O")
        fid = open(fname, "rb")  # Open in binary mode
    return fid


def _edf_open(fid, preload):
    # do preloading of entire file
    if preload:
        # note that StringIO objects instantiated this way are read-only,
        # but that's okay here since we are using mode "rb" anyway
        with fid as fid_old:
            fid = BytesIO(fid_old.read())

    fid.seek(0)
    return fid


def edf_open(fname, preload=False, verbose=None):
    """Open an EDF/BDF file.

    Parameters
    ----------
    fname : path-like | fid
        Name of the edf file, or an opened file (will seek back to 0).
    preload : bool
        If True, all data from the file is read into a memory buffer. This
        requires more memory, but can be faster for I/O operations that require
        frequent seeks.
    %(verbose)s

    Returns
    -------
    fid : file
        The file descriptor of the open file.
    """
    fid = __gdf_edf_get_fid(fname)
    try:
        return _edf_open(fid, preload)
    except Exception:
        fid.close()
        raise


def _gdf_open(fid, preload):
    if preload:
        # Ignore preloading, since we need to parse the file sequentially
        # in _read_gdf_header
        warn("Ignoring preload for GFS file.")

    fid.seek(0)
    return fid


def gdf_open(fname, preload=False, verbose=None):
    """Open an GDF file.

    Parameters
    ----------
    fname : path-like | fid
        Name of the gdf file, or an opened file (will seek back to 0).
    preload : bool
        Ignored.
    %(verbose)s

    Returns
    -------
    fid : file
        The file descriptor of the open file.
    """
    fid = __gdf_edf_get_fid(fname)
    try:
        return _gdf_open(fid, preload)
    except Exception:
        fid.close()
        raise
