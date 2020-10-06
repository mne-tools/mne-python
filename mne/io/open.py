# -*- coding: utf-8 -*-
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import os.path as op
from io import BytesIO, SEEK_SET
from gzip import GzipFile

import numpy as np
from scipy import sparse

from .tag import read_tag_info, read_tag, Tag, _call_dict_names
from .tree import make_dir_tree, dir_tree_find
from .constants import FIFF
from ..utils import logger, verbose, _file_like


class _NoCloseRead(object):
    """Create a wrapper that will not close when used as a context manager."""

    def __init__(self, fid):
        self.fid = fid

    def __enter__(self):
        return self.fid

    def __exit__(self, type_, value, traceback):
        return

    def close(self):
        return

    def seek(self, offset, whence=SEEK_SET):
        return self.fid.seek(offset, whence)

    def read(self, size=-1):
        return self.fid.read(size)


def _fiff_get_fid(fname):
    """Open a FIF file with no additional parsing."""
    if _file_like(fname):
        fid = _NoCloseRead(fname)
        fid.seek(0)
    else:
        fname = str(fname)
        if op.splitext(fname)[1].lower() == '.gz':
            logger.debug('Using gzip')
            fid = GzipFile(fname, "rb")  # Open in binary mode
        else:
            logger.debug('Using normal I/O')
            fid = open(fname, "rb")  # Open in binary mode
    return fid


def _get_next_fname(fid, fname, tree):
    """Get the next filename in split files."""
    nodes_list = dir_tree_find(tree, FIFF.FIFFB_REF)
    next_fname = None
    for nodes in nodes_list:
        next_fname = None
        for ent in nodes['directory']:
            if ent.kind == FIFF.FIFF_REF_ROLE:
                tag = read_tag(fid, ent.pos)
                role = int(tag.data)
                if role != FIFF.FIFFV_ROLE_NEXT_FILE:
                    next_fname = None
                    break
            if ent.kind == FIFF.FIFF_REF_FILE_NAME:
                tag = read_tag(fid, ent.pos)
                next_fname = op.join(op.dirname(fname), tag.data)
            if ent.kind == FIFF.FIFF_REF_FILE_NUM:
                # Some files don't have the name, just the number. So
                # we construct the name from the current name.
                if next_fname is not None:
                    continue
                next_num = read_tag(fid, ent.pos).data
                path, base = op.split(fname)
                idx = base.find('.')
                idx2 = base.rfind('-')
                num_str = base[idx2 + 1:idx]
                if not num_str.isdigit():
                    idx2 = -1

                if idx2 < 0 and next_num == 1:
                    # this is the first file, which may not be numbered
                    next_fname = op.join(
                        path, '%s-%d.%s' % (base[:idx], next_num,
                                            base[idx + 1:]))
                    continue

                next_fname = op.join(path, '%s-%d.%s'
                                     % (base[:idx2], next_num, base[idx + 1:]))
        if next_fname is not None:
            break
    return next_fname


@verbose
def fiff_open(fname, preload=False, verbose=None):
    """Open a FIF file.

    Parameters
    ----------
    fname : str | fid
        Name of the fif file, or an opened file (will seek back to 0).
    preload : bool
        If True, all data from the file is read into a memory buffer. This
        requires more memory, but can be faster for I/O operations that require
        frequent seeks.
    %(verbose)s

    Returns
    -------
    fid : file
        The file descriptor of the open file.
    tree : fif tree
        The tree is a complex structure filled with dictionaries,
        lists and tags.
    directory : list
        A list of tags.
    """
    fid = _fiff_get_fid(fname)
    try:
        return _fiff_open(fname, fid, preload)
    except Exception:
        fid.close()
        raise


def _fiff_open(fname, fid, preload):
    # do preloading of entire file
    if preload:
        # note that StringIO objects instantiated this way are read-only,
        # but that's okay here since we are using mode "rb" anyway
        with fid as fid_old:
            fid = BytesIO(fid_old.read())

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


@verbose
def show_fiff(fname, indent='    ', read_limit=np.inf, max_str=30,
              output=str, tag=None, verbose=None):
    """Show FIFF information.

    This function is similar to mne_show_fiff.

    Parameters
    ----------
    fname : str
        Filename to evaluate.
    indent : str
        How to indent the lines.
    read_limit : int
        Max number of bytes of data to read from a tag. Can be np.inf
        to always read all data (helps test read completion).
    max_str : int
        Max number of characters of string representation to print for
        each tag's data.
    output : type
        Either str or list. str is a convenience output for printing.
    tag : int | None
        Provide information about this tag. If None (default), all information
        is shown.
    %(verbose)s

    Returns
    -------
    contents : str
        The contents of the file.
    """
    if output not in [list, str]:
        raise ValueError('output must be list or str')
    if isinstance(tag, str):  # command mne show_fiff passes string
        tag = int(tag)
    f, tree, directory = fiff_open(fname)
    # This gets set to 0 (unknown) by fiff_open, but FIFFB_ROOT probably
    # makes more sense for display
    tree['block'] = FIFF.FIFFB_ROOT
    with f as fid:
        out = _show_tree(fid, tree, indent=indent, level=0,
                         read_limit=read_limit, max_str=max_str, tag_id=tag)
    if output == str:
        out = '\n'.join(out)
    return out


def _find_type(value, fmts=['FIFF_'], exclude=['FIFF_UNIT']):
    """Find matching values."""
    value = int(value)
    vals = [k for k, v in FIFF.items()
            if v == value and any(fmt in k for fmt in fmts) and
            not any(exc in k for exc in exclude)]
    if len(vals) == 0:
        vals = ['???']
    return vals


def _show_tree(fid, tree, indent, level, read_limit, max_str, tag_id):
    """Show FIFF tree."""
    this_idt = indent * level
    next_idt = indent * (level + 1)
    # print block-level information
    out = [this_idt + str(int(tree['block'])) + ' = ' +
           '/'.join(_find_type(tree['block'], fmts=['FIFFB_']))]
    tag_found = False
    if tag_id is None or out[0].strip().startswith(str(tag_id)):
        tag_found = True

    if tree['directory'] is not None:
        kinds = [ent.kind for ent in tree['directory']] + [-1]
        types = [ent.type for ent in tree['directory']]
        sizes = [ent.size for ent in tree['directory']]
        poss = [ent.pos for ent in tree['directory']]
        counter = 0
        good = True
        for k, kn, size, pos, type_ in zip(kinds[:-1], kinds[1:], sizes, poss,
                                           types):
            if not tag_found and k != tag_id:
                continue
            tag = Tag(k, size, 0, pos)
            if read_limit is None or size <= read_limit:
                try:
                    tag = read_tag(fid, pos)
                except Exception:
                    good = False

            if kn == k:
                # don't print if the next item is the same type (count 'em)
                counter += 1
            else:
                # find the tag type
                this_type = _find_type(k, fmts=['FIFF_'])
                # prepend a count if necessary
                prepend = 'x' + str(counter + 1) + ': ' if counter > 0 else ''
                postpend = ''
                # print tag data nicely
                if tag.data is not None:
                    postpend = ' = ' + str(tag.data)[:max_str]
                    if isinstance(tag.data, np.ndarray):
                        if tag.data.size > 1:
                            postpend += ' ... array size=' + str(tag.data.size)
                    elif isinstance(tag.data, dict):
                        postpend += ' ... dict len=' + str(len(tag.data))
                    elif isinstance(tag.data, str):
                        postpend += ' ... str len=' + str(len(tag.data))
                    elif isinstance(tag.data, (list, tuple)):
                        postpend += ' ... list len=' + str(len(tag.data))
                    elif sparse.issparse(tag.data):
                        postpend += (' ... sparse (%s) shape=%s'
                                     % (tag.data.getformat(), tag.data.shape))
                    else:
                        postpend += ' ... type=' + str(type(tag.data))
                postpend = '>' * 20 + 'BAD' if not good else postpend
                type_ = _call_dict_names.get(type_, '?%s?' % (type_,))
                out += [next_idt + prepend + str(k) + ' = ' +
                        '/'.join(this_type) +
                        ' (' + str(size) + 'b %s)' % type_ +
                        postpend]
                out[-1] = out[-1].replace('\n', u'¶')
                counter = 0
                good = True
        if tag_id in kinds:
            tag_found = True
    if not tag_found:
        out = ['']
        level = -1  # removes extra indent
    # deal with children
    for branch in tree['children']:
        out += _show_tree(fid, branch, indent, level + 1, read_limit, max_str,
                          tag_id)
    return out
