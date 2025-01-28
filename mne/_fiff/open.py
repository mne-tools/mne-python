# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from gzip import GzipFile
from io import SEEK_SET, BytesIO
from pathlib import Path

import numpy as np
from scipy.sparse import issparse

from ..utils import _check_fname, _file_like, _validate_type, logger, verbose, warn
from .constants import FIFF
from .tag import Tag, _call_dict_names, _matrix_info, _read_tag_header, read_tag
from .tree import dir_tree_find, make_dir_tree


class _NoCloseRead:
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
        logger.debug("Using file-like I/O")
        fid = _NoCloseRead(fname)
        fid.seek(0)
    else:
        _validate_type(fname, Path, "fname", extra="or file-like")
        if fname.suffixes[-1] == ".gz":
            logger.debug("Using gzip I/O")
            fid = GzipFile(fname, "rb")  # Open in binary mode
        else:
            logger.debug("Using normal I/O")
            fid = open(fname, "rb")  # Open in binary mode
    return fid


def _get_next_fname(fid, fname, tree):
    """Get the next filename in split files."""
    _validate_type(fname, (Path, None), "fname")
    nodes_list = dir_tree_find(tree, FIFF.FIFFB_REF)
    next_fname = None
    for nodes in nodes_list:
        next_fname = None
        for ent in nodes["directory"]:
            if ent.kind == FIFF.FIFF_REF_ROLE:
                tag = read_tag(fid, ent.pos)
                role = int(tag.data.item())
                if role != FIFF.FIFFV_ROLE_NEXT_FILE:
                    next_fname = None
                    break
            if ent.kind not in (FIFF.FIFF_REF_FILE_NAME, FIFF.FIFF_REF_FILE_NUM):
                continue
            # If we can't resolve it, assume/hope it's in the current directory
            if fname is None:
                fname = Path().resolve()
            if ent.kind == FIFF.FIFF_REF_FILE_NAME:
                tag = read_tag(fid, ent.pos)
                next_fname = fname.parent / tag.data
            if ent.kind == FIFF.FIFF_REF_FILE_NUM:
                # Some files don't have the name, just the number. So
                # we construct the name from the current name.
                if next_fname is not None:
                    continue
                next_num = read_tag(fid, ent.pos).data.item()
                base = fname.name
                idx = base.find(".")
                idx2 = base.rfind("-")
                num_str = base[idx2 + 1 : idx]
                if not num_str.isdigit():
                    idx2 = -1

                if idx2 < 0 and next_num == 1:
                    # this is the first file, which may not be numbered
                    next_fname = (
                        fname.parent / f"{base[:idx]}-{next_num:d}.{base[idx + 1 :]}"
                    )
                    continue

                next_fname = (
                    fname.parent / f"{base[:idx2]}-{next_num:d}.{base[idx + 1 :]}"
                )
        if next_fname is not None:
            break
    return next_fname


@verbose
def fiff_open(fname, preload=False, verbose=None):
    """Open a FIF file.

    Parameters
    ----------
    fname : path-like | fid
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

    tag = _read_tag_header(fid, 0)

    #   Check that this looks like a fif file
    prefix = f"file {repr(fname)} does not"
    if tag.kind != FIFF.FIFF_FILE_ID:
        raise ValueError(f"{prefix} start with a file id tag")

    if tag.type != FIFF.FIFFT_ID_STRUCT:
        raise ValueError(f"{prefix} start with a file id tag")

    if tag.size != 20:
        raise ValueError(f"{prefix} start with a file id tag")

    tag = read_tag(fid, tag.next_pos)

    if tag.kind != FIFF.FIFF_DIR_POINTER:
        raise ValueError(f"{prefix} have a directory pointer")

    #   Read or create the directory tree
    logger.debug(f"    Creating tag directory for {fname}...")

    dirpos = int(tag.data.item())
    read_slow = True
    if dirpos > 0:
        dir_tag = read_tag(fid, dirpos)
        if dir_tag is None or dir_tag.data is None:
            fid.seek(0, 2)  # move to end of file
            size = fid.tell()
            extra = "" if size > dirpos else f" > file size {size}"
            warn(
                "FIF tag directory missing at the end of the file "
                f"(at byte {dirpos}{extra}), possibly corrupted file: {fname}"
            )
        else:
            directory = dir_tag.data
            read_slow = False
    if read_slow:
        pos = 0
        fid.seek(pos, 0)
        directory = list()
        while pos is not None:
            tag = _read_tag_header(fid, pos)
            if tag is None:
                break  # HACK : to fix file ending with empty tag...
            pos = tag.next_pos
            directory.append(tag)

    tree, _ = make_dir_tree(fid, directory, indent=1)

    logger.debug("[done]")

    #   Back to the beginning
    fid.seek(0)

    return fid, tree, directory


@verbose
def show_fiff(
    fname,
    indent="    ",
    read_limit=np.inf,
    max_str=30,
    output=str,
    tag=None,
    *,
    show_bytes=False,
    verbose=None,
):
    """Show FIFF information.

    This function is similar to mne_show_fiff.

    Parameters
    ----------
    fname : path-like
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
    show_bytes : bool
        If True (default False), print the byte offsets of each tag.
    %(verbose)s

    Returns
    -------
    contents : str
        The contents of the file.
    """
    if output not in [list, str]:
        raise ValueError("output must be list or str")
    if isinstance(tag, str):  # command mne show_fiff passes string
        tag = int(tag)
    fname = _check_fname(fname, "read", True)
    f, tree, _ = fiff_open(fname)
    # This gets set to 0 (unknown) by fiff_open, but FIFFB_ROOT probably
    # makes more sense for display
    tree["block"] = FIFF.FIFFB_ROOT
    with f as fid:
        out = _show_tree(
            fid,
            tree,
            indent=indent,
            level=0,
            read_limit=read_limit,
            max_str=max_str,
            tag_id=tag,
            show_bytes=show_bytes,
        )
    if output is str:
        out = "\n".join(out)
    return out


def _find_type(value, fmts=("FIFF_",), exclude=("FIFF_UNIT",)):
    """Find matching values."""
    value = int(value)
    vals = [
        k
        for k, v in FIFF.items()
        if v == value
        and any(fmt in k for fmt in fmts)
        and not any(exc in k for exc in exclude)
    ]
    if len(vals) == 0:
        vals = ["???"]
    return vals


def _show_tree(
    fid,
    tree,
    indent,
    level,
    read_limit,
    max_str,
    tag_id,
    *,
    show_bytes=False,
):
    """Show FIFF tree."""
    this_idt = indent * level
    next_idt = indent * (level + 1)
    # print block-level information
    found_types = "/".join(_find_type(tree["block"], fmts=["FIFFB_"]))
    out = [f"{this_idt}{str(int(tree['block'])).ljust(4)} = {found_types}"]
    tag_found = False
    if tag_id is None or out[0].strip().startswith(str(tag_id)):
        tag_found = True

    if tree["directory"] is not None:
        kinds = [ent.kind for ent in tree["directory"]] + [-1]
        types = [ent.type for ent in tree["directory"]]
        sizes = [ent.size for ent in tree["directory"]]
        poss = [ent.pos for ent in tree["directory"]]
        counter = 0
        good = True
        for k, kn, size, pos, type_ in zip(kinds[:-1], kinds[1:], sizes, poss, types):
            if not tag_found and k != tag_id:
                continue
            tag = Tag(kind=k, type=type_, size=size, next=FIFF.FIFFV_NEXT_NONE, pos=pos)
            if read_limit is None or size <= read_limit:
                try:
                    tag = read_tag(fid, pos)
                except Exception:
                    good = False

            if kn == k:
                # don't print if the next item is the same type (count 'em)
                counter += 1
            else:
                if show_bytes:
                    at = f" @{pos}"
                else:
                    at = ""
                # find the tag type
                this_type = _find_type(k, fmts=["FIFF_"])
                # prepend a count if necessary
                prepend = "x" + str(counter + 1) + ": " if counter > 0 else ""
                postpend = ""
                # print tag data nicely
                if tag.data is not None:
                    postpend = " = " + str(tag.data)[:max_str]
                    if isinstance(tag.data, np.ndarray):
                        if tag.data.size > 1:
                            postpend += " ... array size=" + str(tag.data.size)
                    elif isinstance(tag.data, dict):
                        postpend += " ... dict len=" + str(len(tag.data))
                    elif isinstance(tag.data, str):
                        postpend += " ... str len=" + str(len(tag.data))
                    elif isinstance(tag.data, list | tuple):
                        postpend += " ... list len=" + str(len(tag.data))
                    elif issparse(tag.data):
                        postpend += (
                            f" ... sparse ({tag.data.__class__.__name__}) shape="
                            f"{tag.data.shape}"
                        )
                    else:
                        postpend += " ... type=" + str(type(tag.data))
                postpend = ">" * 20 + f"BAD @{pos}" if not good else postpend
                matrix_info = _matrix_info(tag)
                if matrix_info is not None:
                    _, type_, _, _ = matrix_info
                type_ = _call_dict_names.get(type_, f"?{type_}?")
                this_type = "/".join(this_type)
                out += [
                    f"{next_idt}{prepend}{str(k).ljust(4)} = "
                    f"{this_type}{at} ({size}b {type_}) {postpend}"
                ]
                out[-1] = out[-1].replace("\n", "¶")
                counter = 0
                good = True
        if tag_id in kinds:
            tag_found = True
    if not tag_found:
        out = [""]
        level = -1  # removes extra indent
    # deal with children
    for branch in tree["children"]:
        out += _show_tree(
            fid,
            branch,
            indent,
            level + 1,
            read_limit,
            max_str,
            tag_id,
            show_bytes=show_bytes,
        )
    return out
