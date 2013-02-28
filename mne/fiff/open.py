# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np
import os.path as op
import gzip
import cStringIO
import logging
logger = logging.getLogger('mne')

from .tag import read_tag_info, read_tag, read_big, Tag
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
        fid_old = fid
        fid = cStringIO.StringIO(read_big(fid_old))
        fid_old.close()

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
        raise ValueError('file does not have a directory pointer')

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
            if tag is None:
                break  # HACK : to fix file ending with empty tag...
            else:
                tag.pos = pos
                directory.append(tag)

    tree, _ = make_dir_tree(fid, directory)

    logger.debug('[done]')

    #   Back to the beginning
    fid.seek(0)

    return fid, tree, directory


def show_fiff(fname, indent='    ', read_limit=1024, max_str=30, verbose=None):
    """Show FIFF information similar to mne_show_fiff

    Parameters
    ----------
    fname : str
        Filename.
    indent : str
        How to indent the lines.
    read_limit : int | None
        Max number of bytes of data to read from a tag. If None, no limit
        is used.
    max_str : int
        Max number of characters to print as an example.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    """
    f, tree, directory = fiff_open(fname)
    with f as fid:
        out = _show_tree(fid, tree['children'][0], indent=indent, level=0,
                         read_limit=read_limit, max_str=max_str)
    return out


def _find_type(value, fmt='FIFF_'):
    vals = [k for k, v in FIFF.iteritems() if v == value and fmt in k]
    return vals[:3]


def _show_tree(fid, tree, indent, level, read_limit, max_str):
    """Helper for showing FIFF"""
    this_idt = indent * level
    next_idt = indent * (level + 1)
    out = (this_idt + str(tree['block'][0]) + ' = '
           + '/'.join(_find_type(tree['block'], fmt='FIFFB_')) + '\n')
    if tree['directory'] is not None:
        kinds = [ent.kind for ent in tree['directory']] + [-1]
        sizes = [ent.size for ent in tree['directory']]
        poss = [ent.pos for ent in tree['directory']]
        counter = 0
        good = True
        for k, kn, size, pos in zip(kinds[:-1], kinds[1:], sizes, poss):
            tag = Tag(k, size, 0, pos)
            if read_limit is None or size <= read_limit:
                try:
                    tag = read_tag(fid, pos)
                except Exception:
                    good = False

            if kn == k:
                counter += 1
            else:
                this_type = _find_type(k, fmt='FIFF_')
                prepend = 'x' + str(counter) + ': ' if counter > 0 else ''
                postpend = ''
                if tag.data is not None:
                    if not isinstance(tag.data, dict):
                        if isinstance(tag.data, basestring):
                            postpend = ' = ' + str(tag.data)[:max_str]
                        if isinstance(tag.data, np.ndarray):
                            postpend = ' = ' + str(tag.data.ravel())[:max_str]
                            if tag.data.size > 1:
                                postpend += ' ... size=' + str(tag.data.size)
                postpend = '>' * 20 + 'BAD' if good is False else postpend
                out += (next_idt + prepend + str(k) + ' = '
                        + '/'.join(this_type) + ' (' + str(size) + ')'
                        + postpend + '\n')
                counter = 0
                good = True

    # deal with children
    for branch in tree['children']:
        out += _show_tree(fid, branch, indent, level + 1, read_limit, max_str)
    return out
