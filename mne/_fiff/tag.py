# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import html
import re
import struct
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
from scipy.sparse import csc_array, csr_array

from ..utils import _check_option, warn
from ..utils.numerics import _julian_to_date
from .constants import (
    FIFF,
    _ch_coil_type_named,
    _ch_kind_named,
    _ch_unit_mul_named,
    _ch_unit_named,
    _dig_cardinal_named,
    _dig_kind_named,
)

##############################################################################
# HELPERS


@dataclass
class Tag:
    """Tag in FIF tree structure."""

    kind: int
    type: int
    size: int
    next: int
    pos: int
    data: Any = None

    def __eq__(self, tag):  # noqa: D105
        return int(
            self.kind == tag.kind
            and self.type == tag.type
            and self.size == tag.size
            and self.next == tag.next
            and self.pos == tag.pos
            and self.data == tag.data
        )

    @property
    def next_pos(self):
        """The next tag position."""
        if self.next == FIFF.FIFFV_NEXT_SEQ:  # 0
            return self.pos + 16 + self.size
        elif self.next > 0:
            return self.next
        else:  # self.next should be -1 if we get here
            return None  # safest to return None so that things like fid.seek die


def _frombuffer_rows(fid, tag_size, dtype=None, shape=None, rlims=None):
    """Get a range of rows from a large tag."""
    if shape is not None:
        item_size = np.dtype(dtype).itemsize
        if not len(shape) == 2:
            raise ValueError("Only implemented for 2D matrices")
        want_shape = np.prod(shape)
        have_shape = tag_size // item_size
        if want_shape != have_shape:
            raise ValueError(
                f"Wrong shape specified, requested {want_shape} but got "
                f"{have_shape}"
            )
        if not len(rlims) == 2:
            raise ValueError("rlims must have two elements")
        n_row_out = rlims[1] - rlims[0]
        if n_row_out <= 0:
            raise ValueError("rlims must yield at least one output")
        row_size = item_size * shape[1]
        # # of bytes to skip at the beginning, # to read, where to end
        start_skip = int(rlims[0] * row_size)
        read_size = int(n_row_out * row_size)
        end_pos = int(fid.tell() + tag_size)
        # Move the pointer ahead to the read point
        fid.seek(start_skip, 1)
        # Do the reading
        out = np.frombuffer(fid.read(read_size), dtype=dtype)
        # Move the pointer ahead to the end of the tag
        fid.seek(end_pos)
    else:
        out = np.frombuffer(fid.read(tag_size), dtype=dtype)
    return out


def _loc_to_coil_trans(loc):
    """Convert loc vector to coil_trans."""
    assert loc.shape[-1] == 12
    coil_trans = np.zeros(loc.shape[:-1] + (4, 4))
    coil_trans[..., :3, 3] = loc[..., :3]
    coil_trans[..., :3, :3] = np.reshape(
        loc[..., 3:], loc.shape[:-1] + (3, 3)
    ).swapaxes(-1, -2)
    coil_trans[..., -1, -1] = 1.0
    return coil_trans


def _coil_trans_to_loc(coil_trans):
    """Convert coil_trans to loc."""
    coil_trans = coil_trans.astype(np.float64)
    return np.roll(coil_trans.T[:, :3], 1, 0).flatten()


def _loc_to_eeg_loc(loc):
    """Convert a loc to an EEG loc."""
    if not np.isfinite(loc[:3]).all():
        raise RuntimeError("Missing EEG channel location")
    if np.isfinite(loc[3:6]).all() and (loc[3:6]).any():
        return np.array([loc[0:3], loc[3:6]]).T
    else:
        return loc[0:3][:, np.newaxis].copy()


##############################################################################
# READING FUNCTIONS

# None of these functions have docstring because it's more compact that way,
# and hopefully it's clear what they do by their names and variable values.
# See ``read_tag`` for variable descriptions. Return values are implied
# by the function names.


def _read_tag_header(fid, pos):
    """Read only the header of a Tag."""
    fid.seek(pos, 0)
    s = fid.read(16)
    if len(s) != 16:
        where = fid.tell() - len(s)
        extra = f" in file {fid.name}" if hasattr(fid, "name") else ""
        warn(f"Invalid tag with only {len(s)}/16 bytes at position {where}{extra}")
        return None
    # struct.unpack faster than np.frombuffer, saves ~10% of time some places
    kind, type_, size, next_ = struct.unpack(">iIii", s)
    return Tag(kind, type_, size, next_, pos)


def _read_matrix(fid, tag, shape, rlims):
    """Read a matrix (dense or sparse) tag."""
    # This should be easy to implement (see _frombuffer_rows)
    # if we need it, but for now, it's not...
    if shape is not None or rlims is not None:
        raise ValueError("Row reading not implemented for matrices yet")

    matrix_coding, matrix_type, bit, dtype = _matrix_info(tag)

    pos = tag.pos + 16
    fid.seek(pos + tag.size - 4, 0)
    if matrix_coding == "dense":
        # Find dimensions and return to the beginning of tag data
        ndim = int(np.frombuffer(fid.read(4), dtype=">i4").item())
        fid.seek(-(ndim + 1) * 4, 1)
        dims = np.frombuffer(fid.read(4 * ndim), dtype=">i4")[::-1]
        #
        # Back to where the data start
        #
        fid.seek(pos, 0)

        if ndim > 3:
            raise Exception(
                "Only 2 or 3-dimensional matrices are supported at this time"
            )

        data = fid.read(int(bit * dims.prod()))
        data = np.frombuffer(data, dtype=dtype)
        # Note: we need the non-conjugate transpose here
        if matrix_type == FIFF.FIFFT_COMPLEX_FLOAT:
            data = data.view(">c8")
        elif matrix_type == FIFF.FIFFT_COMPLEX_DOUBLE:
            data = data.view(">c16")
        data.shape = dims
    else:
        # Find dimensions and return to the beginning of tag data
        ndim = int(np.frombuffer(fid.read(4), dtype=">i4").item())
        fid.seek(-(ndim + 2) * 4, 1)
        dims = np.frombuffer(fid.read(4 * (ndim + 1)), dtype=">i4")
        if ndim != 2:
            raise Exception("Only two-dimensional matrices are supported at this time")

        # Back to where the data start
        fid.seek(pos, 0)
        nnz = int(dims[0])
        nrow = int(dims[1])
        ncol = int(dims[2])
        # We need to make a copy so that we can own the data, otherwise we get:
        #     _sparsetools.csr_sort_indices(len(self.indptr) - 1, self.indptr,
        # E   ValueError: WRITEBACKIFCOPY base is read-only
        data = np.frombuffer(fid.read(bit * nnz), dtype=dtype).astype(np.float32)
        shape = (dims[1], dims[2])
        if matrix_coding == "sparse CCS":
            tmp_indices = fid.read(4 * nnz)
            indices = np.frombuffer(tmp_indices, dtype=">i4")
            tmp_ptr = fid.read(4 * (ncol + 1))
            indptr = np.frombuffer(tmp_ptr, dtype=">i4")
            swap = nrow
            klass = csc_array
        else:
            assert matrix_coding == "sparse RCS", matrix_coding
            tmp_indices = fid.read(4 * nnz)
            indices = np.frombuffer(tmp_indices, dtype=">i4")
            tmp_ptr = fid.read(4 * (nrow + 1))
            indptr = np.frombuffer(tmp_ptr, dtype=">i4")
            swap = ncol
            klass = csr_array
        if indptr[-1] > len(indices) or np.any(indptr < 0):
            # There was a bug in MNE-C that caused some data to be
            # stored without byte swapping
            indices = np.concatenate(
                (
                    np.frombuffer(tmp_indices[: 4 * (swap + 1)], dtype=">i4"),
                    np.frombuffer(tmp_indices[4 * (swap + 1) :], dtype="<i4"),
                )
            )
            indptr = np.frombuffer(tmp_ptr, dtype="<i4")
        data = klass((data, indices, indptr), shape=shape)
    return data


def _read_simple(fid, tag, shape, rlims, dtype):
    """Read simple datatypes from tag (typically used with partial)."""
    return _frombuffer_rows(fid, tag.size, dtype=dtype, shape=shape, rlims=rlims)


def _read_string(fid, tag, shape, rlims):
    """Read a string tag."""
    # Always decode to ISO 8859-1 / latin1 (FIFF standard).
    d = _frombuffer_rows(fid, tag.size, dtype=">c", shape=shape, rlims=rlims)
    string = str(d.tobytes().decode("latin1", "ignore"))
    if re.search(r"&#[0-9a-fA-F]{6};", string):
        string = html.unescape(string)
    return string


def _read_complex_float(fid, tag, shape, rlims):
    """Read complex float tag."""
    # data gets stored twice as large
    if shape is not None:
        shape = (shape[0], shape[1] * 2)
    d = _frombuffer_rows(fid, tag.size, dtype=">f4", shape=shape, rlims=rlims)
    d = d.view(">c8")
    return d


def _read_complex_double(fid, tag, shape, rlims):
    """Read complex double tag."""
    # data gets stored twice as large
    if shape is not None:
        shape = (shape[0], shape[1] * 2)
    d = _frombuffer_rows(fid, tag.size, dtype=">f8", shape=shape, rlims=rlims)
    d = d.view(">c16")
    return d


def _read_id_struct(fid, tag, shape, rlims):
    """Read ID struct tag."""
    return dict(
        version=int(np.frombuffer(fid.read(4), dtype=">i4").item()),
        machid=np.frombuffer(fid.read(8), dtype=">i4"),
        secs=int(np.frombuffer(fid.read(4), dtype=">i4").item()),
        usecs=int(np.frombuffer(fid.read(4), dtype=">i4").item()),
    )


def _read_dig_point_struct(fid, tag, shape, rlims):
    """Read dig point struct tag."""
    kind = int(np.frombuffer(fid.read(4), dtype=">i4").item())
    kind = _dig_kind_named.get(kind, kind)
    ident = int(np.frombuffer(fid.read(4), dtype=">i4").item())
    if kind == FIFF.FIFFV_POINT_CARDINAL:
        ident = _dig_cardinal_named.get(ident, ident)
    return dict(
        kind=kind,
        ident=ident,
        r=np.frombuffer(fid.read(12), dtype=">f4"),
        coord_frame=FIFF.FIFFV_COORD_UNKNOWN,
    )


def _read_coord_trans_struct(fid, tag, shape, rlims):
    """Read coord trans struct tag."""
    from ..transforms import Transform

    fro = int(np.frombuffer(fid.read(4), dtype=">i4").item())
    to = int(np.frombuffer(fid.read(4), dtype=">i4").item())
    rot = np.frombuffer(fid.read(36), dtype=">f4").reshape(3, 3)
    move = np.frombuffer(fid.read(12), dtype=">f4")
    trans = np.r_[np.c_[rot, move], np.array([[0], [0], [0], [1]]).T]
    data = Transform(fro, to, trans)
    fid.seek(48, 1)  # Skip over the inverse transformation
    return data


_ch_coord_dict = {
    FIFF.FIFFV_MEG_CH: FIFF.FIFFV_COORD_DEVICE,
    FIFF.FIFFV_REF_MEG_CH: FIFF.FIFFV_COORD_DEVICE,
    FIFF.FIFFV_EEG_CH: FIFF.FIFFV_COORD_HEAD,
    FIFF.FIFFV_ECOG_CH: FIFF.FIFFV_COORD_HEAD,
    FIFF.FIFFV_SEEG_CH: FIFF.FIFFV_COORD_HEAD,
    FIFF.FIFFV_DBS_CH: FIFF.FIFFV_COORD_HEAD,
    FIFF.FIFFV_FNIRS_CH: FIFF.FIFFV_COORD_HEAD,
}


def _read_ch_info_struct(fid, tag, shape, rlims):
    """Read channel info struct tag."""
    d = dict(
        scanno=int(np.frombuffer(fid.read(4), dtype=">i4").item()),
        logno=int(np.frombuffer(fid.read(4), dtype=">i4").item()),
        kind=int(np.frombuffer(fid.read(4), dtype=">i4").item()),
        range=float(np.frombuffer(fid.read(4), dtype=">f4").item()),
        cal=float(np.frombuffer(fid.read(4), dtype=">f4").item()),
        coil_type=int(np.frombuffer(fid.read(4), dtype=">i4").item()),
        # deal with really old OSX Anaconda bug by casting to float64
        loc=np.frombuffer(fid.read(48), dtype=">f4").astype(np.float64),
        # unit and exponent
        unit=int(np.frombuffer(fid.read(4), dtype=">i4").item()),
        unit_mul=int(np.frombuffer(fid.read(4), dtype=">i4").item()),
    )
    # channel name
    ch_name = np.frombuffer(fid.read(16), dtype=">c")
    ch_name = ch_name[: np.argmax(ch_name == b"")].tobytes()
    d["ch_name"] = ch_name.decode()
    # coil coordinate system definition
    _update_ch_info_named(d)
    return d


def _update_ch_info_named(d):
    d["coord_frame"] = _ch_coord_dict.get(d["kind"], FIFF.FIFFV_COORD_UNKNOWN)
    d["kind"] = _ch_kind_named.get(d["kind"], d["kind"])
    d["coil_type"] = _ch_coil_type_named.get(d["coil_type"], d["coil_type"])
    d["unit"] = _ch_unit_named.get(d["unit"], d["unit"])
    d["unit_mul"] = _ch_unit_mul_named.get(d["unit_mul"], d["unit_mul"])


def _read_old_pack(fid, tag, shape, rlims):
    """Read old pack tag."""
    offset = float(np.frombuffer(fid.read(4), dtype=">f4").item())
    scale = float(np.frombuffer(fid.read(4), dtype=">f4").item())
    data = np.frombuffer(fid.read(tag.size - 8), dtype=">i2")
    data = data * scale  # to float64
    data += offset
    return data


def _read_dir_entry_struct(fid, tag, shape, rlims):
    """Read dir entry struct tag."""
    pos = tag.pos + 16
    entries = list()
    for offset in range(1, tag.size // 16):
        ent = _read_tag_header(fid, pos + offset * 16)
        # The position of the real tag on disk is stored in the "next" entry within the
        # directory, so we need to overwrite ent.pos. For safety let's also overwrite
        # ent.next to point nowhere
        ent.pos, ent.next = ent.next, FIFF.FIFFV_NEXT_NONE
        entries.append(ent)
    return entries


def _read_julian(fid, tag, shape, rlims):
    """Read julian tag."""
    return _julian_to_date(int(np.frombuffer(fid.read(4), dtype=">i4").item()))


# Read types call dict
_call_dict = {
    FIFF.FIFFT_STRING: _read_string,
    FIFF.FIFFT_COMPLEX_FLOAT: _read_complex_float,
    FIFF.FIFFT_COMPLEX_DOUBLE: _read_complex_double,
    FIFF.FIFFT_ID_STRUCT: _read_id_struct,
    FIFF.FIFFT_DIG_POINT_STRUCT: _read_dig_point_struct,
    FIFF.FIFFT_COORD_TRANS_STRUCT: _read_coord_trans_struct,
    FIFF.FIFFT_CH_INFO_STRUCT: _read_ch_info_struct,
    FIFF.FIFFT_OLD_PACK: _read_old_pack,
    FIFF.FIFFT_DIR_ENTRY_STRUCT: _read_dir_entry_struct,
    FIFF.FIFFT_JULIAN: _read_julian,
}
_call_dict_names = {
    FIFF.FIFFT_STRING: "str",
    FIFF.FIFFT_COMPLEX_FLOAT: "c8",
    FIFF.FIFFT_COMPLEX_DOUBLE: "c16",
    FIFF.FIFFT_ID_STRUCT: "ids",
    FIFF.FIFFT_DIG_POINT_STRUCT: "dps",
    FIFF.FIFFT_COORD_TRANS_STRUCT: "cts",
    FIFF.FIFFT_CH_INFO_STRUCT: "cis",
    FIFF.FIFFT_OLD_PACK: "op_",
    FIFF.FIFFT_DIR_ENTRY_STRUCT: "dir",
    FIFF.FIFFT_JULIAN: "jul",
    FIFF.FIFFT_VOID: "nul",  # 0
}

#  Append the simple types
_simple_dict = {
    FIFF.FIFFT_BYTE: ">B",
    FIFF.FIFFT_SHORT: ">i2",
    FIFF.FIFFT_INT: ">i4",
    FIFF.FIFFT_USHORT: ">u2",
    FIFF.FIFFT_UINT: ">u4",
    FIFF.FIFFT_FLOAT: ">f4",
    FIFF.FIFFT_DOUBLE: ">f8",
    FIFF.FIFFT_DAU_PACK16: ">i2",
}
for key, dtype in _simple_dict.items():
    _call_dict[key] = partial(_read_simple, dtype=dtype)
    _call_dict_names[key] = dtype


def read_tag(fid, pos, shape=None, rlims=None):
    """Read a Tag from a file at a given position.

    Parameters
    ----------
    fid : file
        The open FIF file descriptor.
    pos : int
        The position of the Tag in the file.
    shape : tuple | None
        If tuple, the shape of the stored matrix. Only to be used with
        data stored as a vector (not implemented for matrices yet).
    rlims : tuple | None
        If tuple, the first (inclusive) and last (exclusive) rows to retrieve.
        Note that data are assumed to be stored row-major in the file. Only to
        be used with data stored as a vector (not implemented for matrices
        yet).

    Returns
    -------
    tag : Tag
        The Tag read.
    """
    tag = _read_tag_header(fid, pos)
    if tag is None:
        return tag
    if tag.size > 0:
        if _matrix_info(tag) is not None:
            tag.data = _read_matrix(fid, tag, shape, rlims)
        else:
            #   All other data types
            try:
                fun = _call_dict[tag.type]
            except KeyError:
                raise Exception(f"Unimplemented tag data type {tag.type}") from None
            tag.data = fun(fid, tag, shape, rlims)
    return tag


def find_tag(fid, node, findkind):
    """Find Tag in an open FIF file descriptor.

    Parameters
    ----------
    fid : file-like
        Open file.
    node : dict
        Node to search.
    findkind : int
        Tag kind to find.

    Returns
    -------
    tag : instance of Tag
        The first tag found.
    """
    if node["directory"] is not None:
        for subnode in node["directory"]:
            if subnode.kind == findkind:
                return read_tag(fid, subnode.pos)
    return None


def has_tag(node, kind):
    """Check if the node contains a Tag of a given kind."""
    for d in node["directory"]:
        if d.kind == kind:
            return True
    return False


def _rename_list(bads, ch_names_mapping):
    return [ch_names_mapping.get(bad, bad) for bad in bads]


def _int_item(x):
    return int(x.item())


def _float_item(x):
    return float(x.item())


def _matrix_info(tag):
    matrix_coding = tag.type & 0xFFFF0000
    if matrix_coding == 0 or tag.size == 0:
        return None
    matrix_type = tag.type & 0x0000FFFF
    matrix_coding_dict = {
        FIFF.FIFFT_MATRIX: "dense",
        FIFF.FIFFT_MATRIX | FIFF.FIFFT_SPARSE_CCS_MATRIX: "sparse CCS",
        FIFF.FIFFT_MATRIX | FIFF.FIFFT_SPARSE_RCS_MATRIX: "sparse RCS",
    }
    _check_option("matrix_coding", matrix_coding, list(matrix_coding_dict))
    matrix_coding = matrix_coding_dict[matrix_coding]
    matrix_bit_dtype = {
        FIFF.FIFFT_INT: (4, ">i4"),
        FIFF.FIFFT_JULIAN: (4, ">i4"),
        FIFF.FIFFT_FLOAT: (4, ">f4"),
        FIFF.FIFFT_DOUBLE: (8, ">f8"),
        FIFF.FIFFT_COMPLEX_FLOAT: (8, ">f4"),
        FIFF.FIFFT_COMPLEX_DOUBLE: (16, ">f8"),
    }
    _check_option("matrix_type", matrix_type, list(matrix_bit_dtype))
    bit, dtype = matrix_bit_dtype[matrix_type]
    return matrix_coding, matrix_type, bit, dtype
