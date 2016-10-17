# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import gzip
from functools import partial
import os
import struct

import numpy as np

from .constants import FIFF
from ..externals.six import text_type
from ..externals.jdcal import jd2jcal


##############################################################################
# HELPERS

class Tag(object):
    """Tag in FIF tree structure.

    Parameters
    ----------
    kind : int
        Kind of Tag.
    type_ : int
        Type of Tag.
    size : int
        Size in bytes.
    int : next
        Position of next Tag.
    pos : int
        Position of Tag is the original file.
    """

    def __init__(self, kind, type_, size, next, pos=None):  # noqa: D102
        self.kind = int(kind)
        self.type = int(type_)
        self.size = int(size)
        self.next = int(next)
        self.pos = pos if pos is not None else next
        self.pos = int(self.pos)
        self.data = None

    def __repr__(self):  # noqa: D105
        out = ("kind: %s - type: %s - size: %s - next: %s - pos: %s"
               % (self.kind, self.type, self.size, self.next, self.pos))
        if hasattr(self, 'data'):
            out += " - data: %s" % self.data
        out += "\n"
        return out

    def __cmp__(self, tag):  # noqa: D105
        return int(self.kind == tag.kind and
                   self.type == tag.type and
                   self.size == tag.size and
                   self.next == tag.next and
                   self.pos == tag.pos and
                   self.data == tag.data)


def read_big(fid, size=None):
    """Function to read large chunks of data (>16MB) Windows-friendly.

    Parameters
    ----------
    fid : file
        Open file to read from.
    size : int or None
        Number of bytes to read. If None, the whole file is read.

    Returns
    -------
    buf : bytes
        The data.

    Notes
    -----
    Windows (argh) can't handle reading large chunks of data, so we
    have to do it piece-wise, possibly related to:
       http://stackoverflow.com/questions/4226941

    Examples
    --------
    This code should work for normal files and .gz files:

        >>> import numpy as np
        >>> import gzip, os, tempfile, shutil
        >>> fname = tempfile.mkdtemp()
        >>> fname_gz = os.path.join(fname, 'temp.gz')
        >>> fname = os.path.join(fname, 'temp.bin')
        >>> randgen = np.random.RandomState(9)
        >>> x = randgen.randn(3000000)  # > 16MB data
        >>> with open(fname, 'wb') as fid: x.tofile(fid)
        >>> with open(fname, 'rb') as fid: y = np.fromstring(read_big(fid))
        >>> assert np.all(x == y)
        >>> fid_gz = gzip.open(fname_gz, 'wb')
        >>> _ = fid_gz.write(x.tostring())
        >>> fid_gz.close()
        >>> fid_gz = gzip.open(fname_gz, 'rb')
        >>> y = np.fromstring(read_big(fid_gz))
        >>> assert np.all(x == y)
        >>> fid_gz.close()
        >>> shutil.rmtree(os.path.dirname(fname))

    """
    # buf_size is chosen as a largest working power of 2 (16 MB):
    buf_size = 16777216
    if size is None:
        # it's not possible to get .gz uncompressed file size
        if not isinstance(fid, gzip.GzipFile):
            size = os.fstat(fid.fileno()).st_size - fid.tell()

    if size is not None:
        # Use pre-buffering method
        segments = np.r_[np.arange(0, size, buf_size), size]
        buf = bytearray(b' ' * size)
        for start, end in zip(segments[:-1], segments[1:]):
            data = fid.read(int(end - start))
            if len(data) != end - start:
                raise ValueError('Read error')
            buf[start:end] = data
        buf = bytes(buf)
    else:
        # Use presumably less efficient concatenating method
        buf = [b'']
        new = fid.read(buf_size)
        while len(new) > 0:
            buf.append(new)
            new = fid.read(buf_size)
        buf = b''.join(buf)

    return buf


def read_tag_info(fid):
    """Read Tag info (or header)."""
    tag = _read_tag_header(fid)
    if tag is None:
        return None
    if tag.next == 0:
        fid.seek(tag.size, 1)
    elif tag.next > 0:
        fid.seek(tag.next, 0)
    return tag


def _fromstring_rows(fid, tag_size, dtype=None, shape=None, rlims=None):
    """Helper for getting a range of rows from a large tag."""
    if shape is not None:
        item_size = np.dtype(dtype).itemsize
        if not len(shape) == 2:
            raise ValueError('Only implemented for 2D matrices')
        want_shape = np.prod(shape)
        have_shape = tag_size // item_size
        if want_shape != have_shape:
            raise ValueError('Wrong shape specified, requested %s have %s'
                             % (want_shape, have_shape))
        if not len(rlims) == 2:
            raise ValueError('rlims must have two elements')
        n_row_out = rlims[1] - rlims[0]
        if n_row_out <= 0:
            raise ValueError('rlims must yield at least one output')
        row_size = item_size * shape[1]
        # # of bytes to skip at the beginning, # to read, where to end
        start_skip = int(rlims[0] * row_size)
        read_size = int(n_row_out * row_size)
        end_pos = int(fid.tell() + tag_size)
        # Move the pointer ahead to the read point
        fid.seek(start_skip, 1)
        # Do the reading
        out = np.fromstring(fid.read(read_size), dtype=dtype)
        # Move the pointer ahead to the end of the tag
        fid.seek(end_pos)
    else:
        out = np.fromstring(fid.read(tag_size), dtype=dtype)
    return out


def _loc_to_coil_trans(loc):
    """Convert loc vector to coil_trans."""
    # deal with nasty OSX Anaconda bug by casting to float64
    loc = loc.astype(np.float64)
    coil_trans = np.concatenate([loc.reshape(4, 3).T[:, [1, 2, 3, 0]],
                                 np.array([0, 0, 0, 1]).reshape(1, 4)])
    return coil_trans


def _coil_trans_to_loc(coil_trans):
    """Convert coil_trans to loc."""
    coil_trans = coil_trans.astype(np.float64)
    return np.roll(coil_trans.T[:, :3], 1, 0).flatten()


def _loc_to_eeg_loc(loc):
    """Convert a loc to an EEG loc."""
    if loc[3:6].any():
        return np.array([loc[0:3], loc[3:6]]).T
    else:
        return loc[0:3][:, np.newaxis].copy()


##############################################################################
# READING FUNCTIONS

# None of these functions have docstring because it's more compact that way,
# and hopefully it's clear what they do by their names and variable values.
# See ``read_tag`` for variable descriptions. Return values are implied
# by the function names.

_is_matrix = 4294901760  # ffff0000
_matrix_coding_dense = 16384      # 4000
_matrix_coding_CCS = 16400      # 4010
_matrix_coding_RCS = 16416      # 4020
_data_type = 65535      # ffff


def _read_tag_header(fid):
    """Read only the header of a Tag."""
    s = fid.read(4 * 4)
    if len(s) == 0:
        return None
    # struct.unpack faster than np.fromstring, saves ~10% of time some places
    return Tag(*struct.unpack('>iIii', s))


def _read_matrix(fid, tag, shape, rlims, matrix_coding):
    """Read a matrix (dense or sparse) tag."""
    matrix_coding = matrix_coding >> 16

    # This should be easy to implement (see _fromstring_rows)
    # if we need it, but for now, it's not...
    if shape is not None:
        raise ValueError('Row reading not implemented for matrices '
                         'yet')

    #   Matrices
    if matrix_coding == _matrix_coding_dense:
        # Find dimensions and return to the beginning of tag data
        pos = fid.tell()
        fid.seek(tag.size - 4, 1)
        ndim = int(np.fromstring(fid.read(4), dtype='>i4'))
        fid.seek(-(ndim + 1) * 4, 1)
        dims = np.fromstring(fid.read(4 * ndim), dtype='>i4')[::-1]
        #
        # Back to where the data start
        #
        fid.seek(pos, 0)

        if ndim > 3:
            raise Exception('Only 2 or 3-dimensional matrices are '
                            'supported at this time')

        matrix_type = _data_type & tag.type

        if matrix_type == FIFF.FIFFT_INT:
            data = np.fromstring(read_big(fid, 4 * dims.prod()), dtype='>i4')
        elif matrix_type == FIFF.FIFFT_JULIAN:
            data = np.fromstring(read_big(fid, 4 * dims.prod()), dtype='>i4')
        elif matrix_type == FIFF.FIFFT_FLOAT:
            data = np.fromstring(read_big(fid, 4 * dims.prod()), dtype='>f4')
        elif matrix_type == FIFF.FIFFT_DOUBLE:
            data = np.fromstring(read_big(fid, 8 * dims.prod()), dtype='>f8')
        elif matrix_type == FIFF.FIFFT_COMPLEX_FLOAT:
            data = np.fromstring(read_big(fid, 4 * 2 * dims.prod()),
                                 dtype='>f4')
            # Note: we need the non-conjugate transpose here
            data = (data[::2] + 1j * data[1::2])
        elif matrix_type == FIFF.FIFFT_COMPLEX_DOUBLE:
            data = np.fromstring(read_big(fid, 8 * 2 * dims.prod()),
                                 dtype='>f8')
            # Note: we need the non-conjugate transpose here
            data = (data[::2] + 1j * data[1::2])
        else:
            raise Exception('Cannot handle matrix of type %d yet'
                            % matrix_type)
        data.shape = dims
    elif matrix_coding in (_matrix_coding_CCS, _matrix_coding_RCS):
        from scipy import sparse
        # Find dimensions and return to the beginning of tag data
        pos = fid.tell()
        fid.seek(tag.size - 4, 1)
        ndim = int(np.fromstring(fid.read(4), dtype='>i4'))
        fid.seek(-(ndim + 2) * 4, 1)
        dims = np.fromstring(fid.read(4 * (ndim + 1)), dtype='>i4')
        if ndim != 2:
            raise Exception('Only two-dimensional matrices are '
                            'supported at this time')

        # Back to where the data start
        fid.seek(pos, 0)
        nnz = int(dims[0])
        nrow = int(dims[1])
        ncol = int(dims[2])
        sparse_data = np.fromstring(fid.read(4 * nnz), dtype='>f4')
        shape = (dims[1], dims[2])
        if matrix_coding == _matrix_coding_CCS:
            #    CCS
            tmp_indices = fid.read(4 * nnz)
            sparse_indices = np.fromstring(tmp_indices, dtype='>i4')
            tmp_ptrs = fid.read(4 * (ncol + 1))
            sparse_ptrs = np.fromstring(tmp_ptrs, dtype='>i4')
            if (sparse_ptrs[-1] > len(sparse_indices) or
                    np.any(sparse_ptrs < 0)):
                # There was a bug in MNE-C that caused some data to be
                # stored without byte swapping
                sparse_indices = np.concatenate(
                    (np.fromstring(tmp_indices[:4 * (nrow + 1)], dtype='>i4'),
                     np.fromstring(tmp_indices[4 * (nrow + 1):], dtype='<i4')))
                sparse_ptrs = np.fromstring(tmp_ptrs, dtype='<i4')
            data = sparse.csc_matrix((sparse_data, sparse_indices,
                                     sparse_ptrs), shape=shape)
        else:
            #    RCS
            sparse_indices = np.fromstring(fid.read(4 * nnz), dtype='>i4')
            sparse_ptrs = np.fromstring(fid.read(4 * (nrow + 1)), dtype='>i4')
            data = sparse.csr_matrix((sparse_data, sparse_indices,
                                     sparse_ptrs), shape=shape)
    else:
        raise Exception('Cannot handle other than dense or sparse '
                        'matrices yet')
    return data


def _read_simple(fid, tag, shape, rlims, dtype):
    """Read simple datatypes from tag (typically used with partial)."""
    return _fromstring_rows(fid, tag.size, dtype=dtype, shape=shape,
                            rlims=rlims)


def _read_string(fid, tag, shape, rlims):
    """Read a string tag."""
    # Always decode to unicode.
    d = _fromstring_rows(fid, tag.size, dtype='>c', shape=shape, rlims=rlims)
    return text_type(d.tostring().decode('utf-8', 'ignore'))


def _read_complex_float(fid, tag, shape, rlims):
    """Read complex float tag."""
    # data gets stored twice as large
    if shape is not None:
        shape = (shape[0], shape[1] * 2)
    d = _fromstring_rows(fid, tag.size, dtype=">f4", shape=shape, rlims=rlims)
    d = d[::2] + 1j * d[1::2]
    return d


def _read_complex_double(fid, tag, shape, rlims):
    """Read complex double tag."""
    # data gets stored twice as large
    if shape is not None:
        shape = (shape[0], shape[1] * 2)
    d = _fromstring_rows(fid, tag.size, dtype=">f8", shape=shape, rlims=rlims)
    d = d[::2] + 1j * d[1::2]
    return d


def _read_id_struct(fid, tag, shape, rlims):
    """Read ID struct tag."""
    return dict(
        version=int(np.fromstring(fid.read(4), dtype=">i4")),
        machid=np.fromstring(fid.read(8), dtype=">i4"),
        secs=int(np.fromstring(fid.read(4), dtype=">i4")),
        usecs=int(np.fromstring(fid.read(4), dtype=">i4")))


def _read_dig_point_struct(fid, tag, shape, rlims):
    """Read dig point struct tag."""
    return dict(
        kind=int(np.fromstring(fid.read(4), dtype=">i4")),
        ident=int(np.fromstring(fid.read(4), dtype=">i4")),
        r=np.fromstring(fid.read(12), dtype=">f4"),
        coord_frame=FIFF.FIFFV_COORD_UNKNOWN)


def _read_coord_trans_struct(fid, tag, shape, rlims):
    """Read coord trans struct tag."""
    from ..transforms import Transform
    fro = int(np.fromstring(fid.read(4), dtype=">i4"))
    to = int(np.fromstring(fid.read(4), dtype=">i4"))
    rot = np.fromstring(fid.read(36), dtype=">f4").reshape(3, 3)
    move = np.fromstring(fid.read(12), dtype=">f4")
    trans = np.r_[np.c_[rot, move],
                  np.array([[0], [0], [0], [1]]).T]
    data = Transform(fro, to, trans)
    fid.seek(48, 1)  # Skip over the inverse transformation
    return data


_coord_dict = {
    FIFF.FIFFV_MEG_CH: FIFF.FIFFV_COORD_DEVICE,
    FIFF.FIFFV_REF_MEG_CH: FIFF.FIFFV_COORD_DEVICE,
    FIFF.FIFFV_EEG_CH: FIFF.FIFFV_COORD_HEAD,
}


def _read_ch_info_struct(fid, tag, shape, rlims):
    """Read channel info struct tag."""
    d = dict(
        scanno=int(np.fromstring(fid.read(4), dtype=">i4")),
        logno=int(np.fromstring(fid.read(4), dtype=">i4")),
        kind=int(np.fromstring(fid.read(4), dtype=">i4")),
        range=float(np.fromstring(fid.read(4), dtype=">f4")),
        cal=float(np.fromstring(fid.read(4), dtype=">f4")),
        coil_type=int(np.fromstring(fid.read(4), dtype=">i4")),
        # deal with really old OSX Anaconda bug by casting to float64
        loc=np.fromstring(fid.read(48), dtype=">f4").astype(np.float64),
        # unit and exponent
        unit=int(np.fromstring(fid.read(4), dtype=">i4")),
        unit_mul=int(np.fromstring(fid.read(4), dtype=">i4")),
    )
    # channel name
    ch_name = np.fromstring(fid.read(16), dtype=">c")
    ch_name = ch_name[:np.argmax(ch_name == b'')].tostring()
    d['ch_name'] = ch_name.decode()
    # coil coordinate system definition
    d['coord_frame'] = _coord_dict.get(d['kind'], FIFF.FIFFV_COORD_UNKNOWN)
    return d


def _read_old_pack(fid, tag, shape, rlims):
    """Read old pack tag."""
    offset = float(np.fromstring(fid.read(4), dtype=">f4"))
    scale = float(np.fromstring(fid.read(4), dtype=">f4"))
    data = np.fromstring(fid.read(tag.size - 8), dtype=">i2")
    data = data * scale  # to float64
    data += offset
    return data


def _read_dir_entry_struct(fid, tag, shape, rlims):
    """Read dir entry struct tag."""
    return [_read_tag_header(fid) for _ in range(tag.size // 16 - 1)]


def _read_julian(fid, tag, shape, rlims):
    """Read julian tag."""
    return jd2jcal(int(np.fromstring(fid.read(4), dtype=">i4")))

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

#  Append the simple types
_simple_dict = {
    FIFF.FIFFT_BYTE: '>B1',
    FIFF.FIFFT_SHORT: '>i2',
    FIFF.FIFFT_INT: '>i4',
    FIFF.FIFFT_USHORT: '>u2',
    FIFF.FIFFT_UINT: '>u4',
    FIFF.FIFFT_FLOAT: '>f4',
    FIFF.FIFFT_DOUBLE: '>f8',
    FIFF.FIFFT_DAU_PACK16: '>i2',
}
for key, dtype in _simple_dict.items():
    _call_dict[key] = partial(_read_simple, dtype=dtype)


def read_tag(fid, pos=None, shape=None, rlims=None):
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
    if pos is not None:
        fid.seek(pos, 0)
    tag = _read_tag_header(fid)
    if tag.size > 0:
        matrix_coding = _is_matrix & tag.type
        if matrix_coding != 0:
            tag.data = _read_matrix(fid, tag, shape, rlims, matrix_coding)
        else:
            #   All other data types
            fun = _call_dict.get(tag.type)
            if fun is not None:
                tag.data = fun(fid, tag, shape, rlims)
            else:
                raise Exception('Unimplemented tag data type %s' % tag.type)
    if tag.next != FIFF.FIFFV_NEXT_SEQ:
        # f.seek(tag.next,0)
        fid.seek(tag.next, 1)  # XXX : fix? pb when tag.next < 0

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
    if node['directory'] is not None:
        for subnode in node['directory']:
            if subnode.kind == findkind:
                return read_tag(fid, subnode.pos)
    return None


def has_tag(node, kind):
    """Check if the node contains a Tag of a given kind."""
    for d in node['directory']:
        if d.kind == kind:
            return True
    return False
