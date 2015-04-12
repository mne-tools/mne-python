# Authors: Denis A. Engemann  <denis.engemann@gmail.com>
#          simplified BSD-3 license

import struct
import numpy as np
from ...externals.six import b


def _unpack_matrix(fid, fmt, rows, cols, dtype):
    """ Aux Function """
    out = np.zeros((rows, cols), dtype=dtype)
    bsize = struct.calcsize(fmt)
    string = fid.read(bsize)
    data = struct.unpack(fmt, string)
    iter_mat = [(r, c) for r in range(rows) for c in range(cols)]
    for idx, (row, col) in enumerate(iter_mat):
        out[row, col] = data[idx]

    return out


def _unpack_simple(fid, fmt, count):
    """ Aux Function """
    bsize = struct.calcsize(fmt)
    string = fid.read(bsize)
    data = list(struct.unpack(fmt, string))

    out = data if count < 2 else list(data)
    if len(out) > 0:
        out = out[0]

    return out


def read_str(fid, count=1):
    """ Read string """
    fmt = '>' + ('c' * count)
    data = list(struct.unpack(fmt, fid.read(struct.calcsize(fmt))))

    bytestr = b('').join(data[0:data.index(b('\x00')) if b('\x00') in data else
                         count])

    return str(bytestr.decode('ascii'))  # Return native str type for Py2/3


def read_char(fid, count=1):
    " Read character from bti file """
    return _unpack_simple(fid, '>' + ('c' * count), count)


def read_bool(fid, count=1):
    """ Read bool value from bti file """
    return _unpack_simple(fid, '>' + ('?' * count), count)


def read_uint8(fid, count=1):
    """ Read unsigned 8bit integer from bti file """
    return _unpack_simple(fid, '>' + ('B' * count), count)


def read_int8(fid, count=1):
    """ Read 8bit integer from bti file """
    return _unpack_simple(fid, '>' + ('b' * count),  count)


def read_uint16(fid, count=1):
    """ Read unsigned 16bit integer from bti file """
    return _unpack_simple(fid, '>' + ('H' * count), count)


def read_int16(fid, count=1):
    """ Read 16bit integer from bti file """
    return _unpack_simple(fid, '>' + ('H' * count),  count)


def read_uint32(fid, count=1):
    """ Read unsigned 32bit integer from bti file """
    return _unpack_simple(fid, '>' + ('I' * count), count)


def read_int32(fid, count=1):
    """ Read 32bit integer from bti file """
    return _unpack_simple(fid, '>' + ('i' * count), count)


def read_uint64(fid, count=1):
    """ Read unsigned 64bit integer from bti file """
    return _unpack_simple(fid, '>' + ('Q' * count), count)


def read_int64(fid, count=1):
    """ Read 64bit integer from bti file """
    return _unpack_simple(fid, '>' + ('q' * count), count)


def read_float(fid, count=1):
    """ Read 32bit float from bti file """
    return _unpack_simple(fid, '>' + ('f' * count), count)


def read_double(fid, count=1):
    """ Read 64bit float from bti file """
    return _unpack_simple(fid, '>' + ('d' * count), count)


def read_int16_matrix(fid, rows, cols):
    """ Read 16bit integer matrix from bti file """
    fmt = '>' + ('h' * rows * cols)
    return _unpack_matrix(fid, fmt, rows, cols, np.int16)


def read_float_matrix(fid, rows, cols):
    """ Read 32bit float matrix from bti file """
    fmt = '>' + ('f' * rows * cols)
    return _unpack_matrix(fid, fmt, rows, cols, 'f4')


def read_double_matrix(fid, rows, cols):
    """ Read 64bit float matrix from bti file """
    fmt = '>' + ('d' * rows * cols)
    return _unpack_matrix(fid, fmt, rows, cols, 'f8')


def read_transform(fid):
    """ Read 64bit float matrix transform from bti file """
    fmt = '>' + ('d' * 4 * 4)
    return _unpack_matrix(fid, fmt, 4, 4, 'f8')
