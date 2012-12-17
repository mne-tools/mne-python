# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import os.path as op
import gzip
import logging
import cStringIO
logger = logging.getLogger('mne')

from .tag import read_tag_info, read_tag
from .tree import make_dir_tree
from .constants import FIFF
from .. import verbose


@verbose
def fiff_open(fname, preload=False, verbose=None):
    """Open a FIF file.

    Parameters
    ----------
    fname : string
        name of the fif file
    preload : bool
        If True, all data from the file is read into a memory buffer. This
        requires more memory, but can be faster for I/O operations that require
        frequent seeks.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    fid : file
        The file descriptor of the open file
    tree : fif tree
        The tree is a complex structure filled with dictionaries,
        lists and tags.
    directory : list
        list of nodes.
    """
    if op.splitext(fname)[1].lower() == '.gz':
        logger.debug('Using gzip')
        fid = gzip.open(fname, "rb")  # Open in binary mode
    else:
        logger.debug('Using normal I/O')
        fid = open(fname, "rb")  # Open in binary mode

    # do preloading of entire file
    if preload:
        # note that cStringIO objects instantiated this way are read-only,
        # but that's okay here since we are using mode "rb" anyway
        fid = cStringIO.StringIO(fid.read())

    tag = read_tag_info(fid)

    #   Check that this looks like a fif file
    if tag.kind != FIFF.FIFF_FILE_ID:
        raise ValueError('file does not start with a file id tag')

    if tag.type != FIFF.FIFFT_ID_STRUCT:
        raise ValueError('file does not start with a file id tag')

    if tag.size != 20:
        raise ValueError('file does not start with a file id tag')

    tag = read_tag(fid)

    if tag.kind != FIFF.FIFF_DIR_POINTER:
        raise ValueError('file does have a directory pointer')

    #   Read or create the directory tree
    logger.debug('    Creating tag directory for %s...' % fname)

    dirpos = int(tag.data)
    if dirpos > 0:
        tag = read_tag(fid, dirpos)
        directory = tag.data
    else:
        fid.seek(0, 0)
        directory = list()
        while tag.next >= 0:
            pos = fid.tell()
            tag = read_tag_info(fid)
            tag.pos = pos
            directory.append(tag)

    tree, _ = make_dir_tree(fid, directory)

    logger.debug('[done]')

    #   Back to the beginning
    fid.seek(0)

    return fid, tree, directory
