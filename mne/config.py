""" Module-wide configuration """

# Author: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: Simplified BSD

# dictionary with module-wide options

from os import path


# dict with all module-wide configuration options
_CONFIG = {'cache_dir': None}


def set_cache_dir(cache_dir):
    """ Set the directory to be used for temporary file storage.
    This directory is used by joblib to store memmapped arrays,
    which reduces memory requirements and speeds up parallel
    computation.

    Parameters
    ----------
    cache_dir: str or None
        Directory to use for temporary file storage. None disables
        temporary file storage.
    """
    if cache_dir is not None and not path.exists(cache_dir):
        raise ValueError('Directory %s does not exist' % cache_dir)

    _CONFIG['cache_dir'] = cache_dir


def get_cache_dir():
    """ Get the directory used for temporary file storage.

    Returns
    -------
    cache_dir: str or None
        Directory used for temporary file storage
    """

    return _CONFIG['cache_dir']
