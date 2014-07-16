# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import struct
import os
import gzip
import numpy as np
from scipy import linalg

from .constants import FIFF

from ..externals.six import text_type
from ..externals.jdcal import jd2jcal


class Tag(object):
    """Tag in FIF tree structure

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

    def __init__(self, kind, type_, size, next, pos=None):
        self.kind = int(kind)
        self.type = int(type_)
        self.size = int(size)
        self.next = int(next)
        self.pos = pos if pos is not None else next
        self.pos = int(self.pos)
        self.data = None

    def __repr__(self):
        out = ("kind: %s - type: %s - size: %s - next: %s - pos: %s"
               % (self.kind, self.type, self.size, self.next, self.pos))
        if hasattr(self, 'data'):
            out += " - data: %s" % self.data
        out += "\n"
        return out

    def __cmp__(self, tag):
        is_equal = (self.kind == tag.kind and
                    self.type == tag.type and
                    self.size == tag.size and
                    self.next == tag.next and
                    self.pos == tag.pos and
                    self.data == tag.data)
        if is_equal:
            return 0
        else:
            return 1


def read_big(fid, size=None):
    """Function to read large chunks of data (>16MB) Windows-friendly

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
        >>> shutil.rmtree(os.path.dirname(fname))
        >>> fid_gz.close()

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
    """Read Tag info (or header)
    """
    s = fid.read(4 * 4)
    if len(s) == 0:
        return None
    tag = Tag(*struct.unpack(">iiii", s))
    if tag.next == 0:
        fid.seek(tag.size, 1)
    elif tag.next > 0:
        fid.seek(tag.next, 0)
    return tag


def _fromstring_rows(fid, tag_size, dtype=None, shape=None, rlims=None):
    """Helper for getting a range of rows from a large tag"""
    if shape is not None:
        item_size = np.dtype(dtype).itemsize
        if not len(shape) == 2:
            raise ValueError('Only implemented for 2D matrices')
        if not np.prod(shape) == tag_size / item_size:
            raise ValueError('Wrong shape specified')
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


def _loc_to_trans(loc):
    """Helper to convert loc vector to coil_trans"""
    # deal with nasty OSX Anaconda bug by casting to float64
    loc = loc.astype(np.float64)
    coil_trans = np.concatenate([loc.reshape(4, 3).T[:, [1, 2, 3, 0]],
                                 np.array([0, 0, 0, 1]).reshape(1, 4)])
    return coil_trans


def read_tag(fid, pos=None, shape=None, rlims=None):
    """Read a Tag from a file at a given position

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
        If tuple, the first and last rows to retrieve. Note that data are
        assumed to be stored row-major in the file. Only to be used with
        data stored as a vector (not implemented for matrices yet).

    Returns
    -------
    tag : Tag
        The Tag read.
    """
    if pos is not None:
        fid.seek(pos, 0)

    s = fid.read(4 * 4)
    tag = Tag(*struct.unpack(">iIii", s))

    #
    #   The magic hexadecimal values
    #
    is_matrix = 4294901760  # ffff0000
    matrix_coding_dense = 16384      # 4000
    matrix_coding_CCS = 16400      # 4010
    matrix_coding_RCS = 16416      # 4020
    data_type = 65535      # ffff
    #
    if tag.size > 0:
        matrix_coding = is_matrix & tag.type
        if matrix_coding != 0:
            matrix_coding = matrix_coding >> 16

            # This should be easy to implement (see _fromstring_rows)
            # if we need it, but for now, it's not...
            if shape is not None:
                raise ValueError('Row reading not implemented for matrices '
                                 'yet')

            #   Matrices
            if matrix_coding == matrix_coding_dense:
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

                matrix_type = data_type & tag.type

                if matrix_type == FIFF.FIFFT_INT:
                    tag.data = np.fromstring(read_big(fid, 4 * dims.prod()),
                                             dtype='>i4').reshape(dims)
                elif matrix_type == FIFF.FIFFT_JULIAN:
                    tag.data = np.fromstring(read_big(fid, 4 * dims.prod()),
                                             dtype='>i4').reshape(dims)
                elif matrix_type == FIFF.FIFFT_FLOAT:
                    tag.data = np.fromstring(read_big(fid, 4 * dims.prod()),
                                             dtype='>f4').reshape(dims)
                elif matrix_type == FIFF.FIFFT_DOUBLE:
                    tag.data = np.fromstring(read_big(fid, 8 * dims.prod()),
                                             dtype='>f8').reshape(dims)
                elif matrix_type == FIFF.FIFFT_COMPLEX_FLOAT:
                    data = np.fromstring(read_big(fid, 4 * 2 * dims.prod()),
                                         dtype='>f4')
                    # Note: we need the non-conjugate transpose here
                    tag.data = (data[::2] + 1j * data[1::2]).reshape(dims)
                elif matrix_type == FIFF.FIFFT_COMPLEX_DOUBLE:
                    data = np.fromstring(read_big(fid, 8 * 2 * dims.prod()),
                                         dtype='>f8')
                    # Note: we need the non-conjugate transpose here
                    tag.data = (data[::2] + 1j * data[1::2]).reshape(dims)
                else:
                    raise Exception('Cannot handle matrix of type %d yet'
                                    % matrix_type)

            elif matrix_coding in (matrix_coding_CCS, matrix_coding_RCS):
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
                if matrix_coding == matrix_coding_CCS:
                    #    CCS
                    sparse.csc_matrix()
                    sparse_indices = np.fromstring(fid.read(4 * nnz),
                                                   dtype='>i4')
                    sparse_ptrs = np.fromstring(fid.read(4 * (ncol + 1)),
                                                dtype='>i4')
                    tag.data = sparse.csc_matrix((sparse_data, sparse_indices,
                                                 sparse_ptrs), shape=shape)
                else:
                    #    RCS
                    sparse_indices = np.fromstring(fid.read(4 * nnz),
                                                   dtype='>i4')
                    sparse_ptrs = np.fromstring(fid.read(4 * (nrow + 1)),
                                                dtype='>i4')
                    tag.data = sparse.csr_matrix((sparse_data, sparse_indices,
                                                 sparse_ptrs), shape=shape)
            else:
                raise Exception('Cannot handle other than dense or sparse '
                                'matrices yet')
        else:
            #   All other data types

            #   Simple types
            if tag.type == FIFF.FIFFT_BYTE:
                tag.data = _fromstring_rows(fid, tag.size, dtype=">B1",
                                            shape=shape, rlims=rlims)
            elif tag.type == FIFF.FIFFT_SHORT:
                tag.data = _fromstring_rows(fid, tag.size, dtype=">i2",
                                            shape=shape, rlims=rlims)
            elif tag.type == FIFF.FIFFT_INT:
                tag.data = _fromstring_rows(fid, tag.size, dtype=">i4",
                                            shape=shape, rlims=rlims)
            elif tag.type == FIFF.FIFFT_USHORT:
                tag.data = _fromstring_rows(fid, tag.size, dtype=">u2",
                                            shape=shape, rlims=rlims)
            elif tag.type == FIFF.FIFFT_UINT:
                tag.data = _fromstring_rows(fid, tag.size, dtype=">u4",
                                            shape=shape, rlims=rlims)
            elif tag.type == FIFF.FIFFT_FLOAT:
                tag.data = _fromstring_rows(fid, tag.size, dtype=">f4",
                                            shape=shape, rlims=rlims)
            elif tag.type == FIFF.FIFFT_DOUBLE:
                tag.data = _fromstring_rows(fid, tag.size, dtype=">f8",
                                            shape=shape, rlims=rlims)
            elif tag.type == FIFF.FIFFT_STRING:
                tag.data = _fromstring_rows(fid, tag.size, dtype=">c",
                                            shape=shape, rlims=rlims)

                # Always decode to unicode.
                td = tag.data.tostring().decode('utf-8', 'ignore')
                tag.data = text_type(td)

            elif tag.type == FIFF.FIFFT_DAU_PACK16:
                tag.data = _fromstring_rows(fid, tag.size, dtype=">i2",
                                            shape=shape, rlims=rlims)
            elif tag.type == FIFF.FIFFT_COMPLEX_FLOAT:
                # data gets stored twice as large
                if shape is not None:
                    shape = (shape[0], shape[1] * 2)
                tag.data = _fromstring_rows(fid, tag.size, dtype=">f4",
                                            shape=shape, rlims=rlims)
                tag.data = tag.data[::2] + 1j * tag.data[1::2]
            elif tag.type == FIFF.FIFFT_COMPLEX_DOUBLE:
                # data gets stored twice as large
                if shape is not None:
                    shape = (shape[0], shape[1] * 2)
                tag.data = _fromstring_rows(fid, tag.size, dtype=">f8",
                                            shape=shape, rlims=rlims)
                tag.data = tag.data[::2] + 1j * tag.data[1::2]
            #
            #   Structures
            #
            elif tag.type == FIFF.FIFFT_ID_STRUCT:
                tag.data = dict()
                tag.data['version'] = int(np.fromstring(fid.read(4),
                                                        dtype=">i4"))
                tag.data['version'] = int(np.fromstring(fid.read(4),
                                                        dtype=">i4"))
                tag.data['machid'] = np.fromstring(fid.read(8), dtype=">i4")
                tag.data['secs'] = int(np.fromstring(fid.read(4), dtype=">i4"))
                tag.data['usecs'] = int(np.fromstring(fid.read(4),
                                                      dtype=">i4"))
            elif tag.type == FIFF.FIFFT_DIG_POINT_STRUCT:
                tag.data = dict()
                tag.data['kind'] = int(np.fromstring(fid.read(4), dtype=">i4"))
                tag.data['ident'] = int(np.fromstring(fid.read(4),
                                                      dtype=">i4"))
                tag.data['r'] = np.fromstring(fid.read(12), dtype=">f4")
                tag.data['coord_frame'] = FIFF.FIFFV_COORD_UNKNOWN
            elif tag.type == FIFF.FIFFT_COORD_TRANS_STRUCT:
                tag.data = dict()
                tag.data['from'] = int(np.fromstring(fid.read(4), dtype=">i4"))
                tag.data['to'] = int(np.fromstring(fid.read(4), dtype=">i4"))
                rot = np.fromstring(fid.read(36), dtype=">f4").reshape(3, 3)
                move = np.fromstring(fid.read(12), dtype=">f4")
                tag.data['trans'] = np.r_[np.c_[rot, move],
                                          np.array([[0], [0], [0], [1]]).T]
                #
                # Skip over the inverse transformation
                # It is easier to just use inverse of trans in Matlab
                #
                fid.seek(12 * 4, 1)
            elif tag.type == FIFF.FIFFT_CH_INFO_STRUCT:
                d = dict()
                d['scanno'] = int(np.fromstring(fid.read(4), dtype=">i4"))
                d['logno'] = int(np.fromstring(fid.read(4), dtype=">i4"))
                d['kind'] = int(np.fromstring(fid.read(4), dtype=">i4"))
                d['range'] = float(np.fromstring(fid.read(4), dtype=">f4"))
                d['cal'] = float(np.fromstring(fid.read(4), dtype=">f4"))
                d['coil_type'] = int(np.fromstring(fid.read(4), dtype=">i4"))
                #
                #   Read the coil coordinate system definition
                #
                d['loc'] = np.fromstring(fid.read(48), dtype=">f4")
                d['coil_trans'] = None
                d['eeg_loc'] = None
                d['coord_frame'] = FIFF.FIFFV_COORD_UNKNOWN
                tag.data = d
                #
                #   Convert loc into a more useful format
                #
                loc = tag.data['loc']
                kind = tag.data['kind']
                if kind in [FIFF.FIFFV_MEG_CH, FIFF.FIFFV_REF_MEG_CH]:
                    tag.data['coil_trans'] = _loc_to_trans(loc)
                    tag.data['coord_frame'] = FIFF.FIFFV_COORD_DEVICE
                elif tag.data['kind'] == FIFF.FIFFV_EEG_CH:
                    # deal with nasty OSX Anaconda bug by casting to float64
                    loc = loc.astype(np.float64)
                    if linalg.norm(loc[3:6]) > 0.:
                        tag.data['eeg_loc'] = np.c_[loc[0:3], loc[3:6]]
                    else:
                        tag.data['eeg_loc'] = loc[0:3][:, np.newaxis].copy()
                    tag.data['coord_frame'] = FIFF.FIFFV_COORD_HEAD
                #
                #   Unit and exponent
                #
                tag.data['unit'] = int(np.fromstring(fid.read(4), dtype=">i4"))
                tag.data['unit_mul'] = int(np.fromstring(fid.read(4),
                                                         dtype=">i4"))
                #
                #   Handle the channel name
                #
                ch_name = np.fromstring(fid.read(16), dtype=">c")
                ch_name = ch_name[:np.argmax(ch_name == b'')].tostring()
                # Use unicode or bytes depending on Py2/3
                tag.data['ch_name'] = str(ch_name.decode())

            elif tag.type == FIFF.FIFFT_OLD_PACK:
                offset = float(np.fromstring(fid.read(4), dtype=">f4"))
                scale = float(np.fromstring(fid.read(4), dtype=">f4"))
                tag.data = np.fromstring(fid.read(tag.size - 8), dtype=">h2")
                tag.data = scale * tag.data + offset
            elif tag.type == FIFF.FIFFT_DIR_ENTRY_STRUCT:
                tag.data = list()
                for _ in range(tag.size // 16 - 1):
                    s = fid.read(4 * 4)
                    tag.data.append(Tag(*struct.unpack(">iIii", s)))
            elif tag.type == FIFF.FIFFT_JULIAN:
                tag.data = int(np.fromstring(fid.read(4), dtype=">i4"))
                tag.data = jd2jcal(tag.data)
            else:
                raise Exception('Unimplemented tag data type %s' % tag.type)

    if tag.next != FIFF.FIFFV_NEXT_SEQ:
        # f.seek(tag.next,0)
        fid.seek(tag.next, 1)  # XXX : fix? pb when tag.next < 0

    return tag


def find_tag(fid, node, findkind):
    """Find Tag in an open FIF file descriptor
    """
    for p in range(node['nent']):
        if node['directory'][p].kind == findkind:
            return read_tag(fid, node['directory'][p].pos)
    tag = None
    return tag


def has_tag(node, kind):
    """Does the node contains a Tag of a given kind?
    """
    for d in node['directory']:
        if d.kind == kind:
            return True
    return False
