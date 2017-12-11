# Authors: Denis A. Engemann  <denis.engemann@gmail.com>
#          simplified BSD-3 license

import numpy as np

from ..utils import read_str


def _unpack_matrix(fid, rows, cols, dtype, out_dtype):
    """Unpack matrix."""
    dtype = np.dtype(dtype)

    string = fid.read(int(dtype.itemsize * rows * cols))
    out = np.frombuffer(string, dtype=dtype).reshape(
        rows, cols).astype(out_dtype)
    return out


def _unpack_simple(fid, dtype, out_dtype):
    """Unpack a NumPy type."""
    dtype = np.dtype(dtype)
    string = fid.read(dtype.itemsize)
    out = np.frombuffer(string, dtype=dtype).astype(out_dtype)

    if len(out) > 0:
        out = out[0]
    return out


def read_char(fid, count=1):
    """Read character from bti file."""
    return _unpack_simple(fid, '>S%s' % count, 'S')


def read_bool(fid):
    """Read bool value from bti file."""
    return _unpack_simple(fid, '>?', np.bool)


def read_uint8(fid):
    """Read unsigned 8bit integer from bti file."""
    return _unpack_simple(fid, '>u1', np.uint8)


def read_int8(fid):
    """Read 8bit integer from bti file."""
    return _unpack_simple(fid, '>i1', np.int8)


def read_uint16(fid):
    """Read unsigned 16bit integer from bti file."""
    return _unpack_simple(fid, '>u2', np.uint16)


def read_int16(fid):
    """Read 16bit integer from bti file."""
    return _unpack_simple(fid, '>i2', np.int16)


def read_uint32(fid):
    """Read unsigned 32bit integer from bti file."""
    return _unpack_simple(fid, '>u4', np.uint32)


def read_int32(fid):
    """Read 32bit integer from bti file."""
    return _unpack_simple(fid, '>i4', np.int32)


def read_uint64(fid):
    """Read unsigned 64bit integer from bti file."""
    return _unpack_simple(fid, '>u8', np.uint64)


def read_int64(fid):
    """Read 64bit integer from bti file."""
    return _unpack_simple(fid, '>u8', np.int64)


def read_float(fid):
    """Read 32bit float from bti file."""
    return _unpack_simple(fid, '>f4', np.float32)


def read_double(fid):
    """Read 64bit float from bti file."""
    return _unpack_simple(fid, '>f8', np.float64)


def read_int16_matrix(fid, rows, cols):
    """Read 16bit integer matrix from bti file."""
    return _unpack_matrix(fid, rows, cols, dtype='>i2',
                          out_dtype=np.int16)


def read_float_matrix(fid, rows, cols):
    """Read 32bit float matrix from bti file."""
    return _unpack_matrix(fid, rows, cols, dtype='>f4',
                          out_dtype=np.float32)


def read_double_matrix(fid, rows, cols):
    """Read 64bit float matrix from bti file."""
    return _unpack_matrix(fid, rows, cols, dtype='>f8',
                          out_dtype=np.float64)


def read_transform(fid):
    """Read 64bit float matrix transform from bti file."""
    return read_double_matrix(fid, rows=4, cols=4)


def read_dev_header(x):
    """Create a dev header."""
    return dict(size=read_int32(x), checksum=read_int32(x),
                reserved=read_str(x, 32))
