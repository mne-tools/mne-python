# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#
# License: BSD-3-Clause

from gzip import GzipFile
import os.path as op
import re
import time
import uuid

import numpy as np

from .constants import FIFF
from ..utils import logger, _file_like
from ..utils.numerics import _cal_to_julian

# We choose a "magic" date to store (because meas_date is obligatory)
# to treat as meas_date=None. This one should be impossible for systems
# to write -- the second field is microseconds, so anything >= 1e6
# should be moved into the first field (seconds).
DATE_NONE = (0, 2 ** 31 - 1)


def _write(fid, data, kind, data_size, FIFFT_TYPE, dtype):
    """Write data."""
    if isinstance(data, np.ndarray):
        data_size *= data.size

    # XXX for string types the data size is used as
    # computed in ``write_string``.

    fid.write(np.array(kind, dtype='>i4').tobytes())
    fid.write(np.array(FIFFT_TYPE, dtype='>i4').tobytes())
    fid.write(np.array(data_size, dtype='>i4').tobytes())
    fid.write(np.array(FIFF.FIFFV_NEXT_SEQ, dtype='>i4').tobytes())
    fid.write(np.array(data, dtype=dtype).tobytes())


def _get_split_size(split_size):
    """Convert human-readable bytes to machine-readable bytes."""
    if isinstance(split_size, str):
        exp = dict(MB=20, GB=30).get(split_size[-2:], None)
        if exp is None:
            raise ValueError('split_size has to end with either'
                             '"MB" or "GB"')
        split_size = int(float(split_size[:-2]) * 2 ** exp)

    if split_size > 2147483648:
        raise ValueError('split_size cannot be larger than 2GB')
    return split_size


_NEXT_FILE_BUFFER = 1048576  # 2 ** 20 extra cushion for last post-data tags


def write_nop(fid, last=False):
    """Write a FIFF_NOP."""
    fid.write(np.array(FIFF.FIFF_NOP, dtype='>i4').tobytes())
    fid.write(np.array(FIFF.FIFFT_VOID, dtype='>i4').tobytes())
    fid.write(np.array(0, dtype='>i4').tobytes())
    next_ = FIFF.FIFFV_NEXT_NONE if last else FIFF.FIFFV_NEXT_SEQ
    fid.write(np.array(next_, dtype='>i4').tobytes())


INT32_MAX = 2147483647


def write_int(fid, kind, data):
    """Write a 32-bit integer tag to a fif file."""
    data_size = 4
    data = np.asarray(data)
    if data.dtype.kind not in 'uib' and data.size > 0:
        raise TypeError(
            f'Cannot safely write data with dtype {data.dtype} as int')
    max_val = data.max() if data.size > 0 else 0
    if max_val > INT32_MAX:
        raise TypeError(
            f'Value {max_val} exceeds maximum allowed ({INT32_MAX}) for '
            f'tag {kind}')
    data = data.astype('>i4').T
    _write(fid, data, kind, data_size, FIFF.FIFFT_INT, '>i4')


def write_double(fid, kind, data):
    """Write a double-precision floating point tag to a fif file."""
    data_size = 8
    data = np.array(data, dtype='>f8').T
    _write(fid, data, kind, data_size, FIFF.FIFFT_DOUBLE, '>f8')


def write_float(fid, kind, data):
    """Write a single-precision floating point tag to a fif file."""
    data_size = 4
    data = np.array(data, dtype='>f4').T
    _write(fid, data, kind, data_size, FIFF.FIFFT_FLOAT, '>f4')


def write_dau_pack16(fid, kind, data):
    """Write a dau_pack16 tag to a fif file."""
    data_size = 2
    data = np.array(data, dtype='>i2').T
    _write(fid, data, kind, data_size, FIFF.FIFFT_DAU_PACK16, '>i2')


def write_complex64(fid, kind, data):
    """Write a 64 bit complex floating point tag to a fif file."""
    data_size = 8
    data = np.array(data, dtype='>c8').T
    _write(fid, data, kind, data_size, FIFF.FIFFT_COMPLEX_FLOAT, '>c8')


def write_complex128(fid, kind, data):
    """Write a 128 bit complex floating point tag to a fif file."""
    data_size = 16
    data = np.array(data, dtype='>c16').T
    _write(fid, data, kind, data_size, FIFF.FIFFT_COMPLEX_FLOAT, '>c16')


def write_julian(fid, kind, data):
    """Write a Julian-formatted date to a FIF file."""
    assert len(data) == 3
    data_size = 4
    jd = np.sum(_cal_to_julian(*data))
    data = np.array(jd, dtype='>i4')
    _write(fid, data, kind, data_size, FIFF.FIFFT_JULIAN, '>i4')


def write_string(fid, kind, data):
    """Write a string tag."""
    str_data = data.encode('latin1')
    data_size = len(str_data)  # therefore compute size here
    my_dtype = '>a'  # py2/3 compatible on writing -- don't ask me why
    if data_size > 0:
        _write(fid, str_data, kind, data_size, FIFF.FIFFT_STRING, my_dtype)


def write_name_list(fid, kind, data):
    """Write a colon-separated list of names.

    Parameters
    ----------
    data : list of strings
    """
    write_string(fid, kind, ':'.join(data))


def write_float_matrix(fid, kind, mat):
    """Write a single-precision floating-point matrix tag."""
    FIFFT_MATRIX = 1 << 30
    FIFFT_MATRIX_FLOAT = FIFF.FIFFT_FLOAT | FIFFT_MATRIX

    data_size = 4 * mat.size + 4 * (mat.ndim + 1)

    fid.write(np.array(kind, dtype='>i4').tobytes())
    fid.write(np.array(FIFFT_MATRIX_FLOAT, dtype='>i4').tobytes())
    fid.write(np.array(data_size, dtype='>i4').tobytes())
    fid.write(np.array(FIFF.FIFFV_NEXT_SEQ, dtype='>i4').tobytes())
    fid.write(np.array(mat, dtype='>f4').tobytes())

    dims = np.empty(mat.ndim + 1, dtype=np.int32)
    dims[:mat.ndim] = mat.shape[::-1]
    dims[-1] = mat.ndim
    fid.write(np.array(dims, dtype='>i4').tobytes())
    check_fiff_length(fid)


def write_double_matrix(fid, kind, mat):
    """Write a double-precision floating-point matrix tag."""
    FIFFT_MATRIX = 1 << 30
    FIFFT_MATRIX_DOUBLE = FIFF.FIFFT_DOUBLE | FIFFT_MATRIX

    data_size = 8 * mat.size + 4 * (mat.ndim + 1)

    fid.write(np.array(kind, dtype='>i4').tobytes())
    fid.write(np.array(FIFFT_MATRIX_DOUBLE, dtype='>i4').tobytes())
    fid.write(np.array(data_size, dtype='>i4').tobytes())
    fid.write(np.array(FIFF.FIFFV_NEXT_SEQ, dtype='>i4').tobytes())
    fid.write(np.array(mat, dtype='>f8').tobytes())

    dims = np.empty(mat.ndim + 1, dtype=np.int32)
    dims[:mat.ndim] = mat.shape[::-1]
    dims[-1] = mat.ndim
    fid.write(np.array(dims, dtype='>i4').tobytes())
    check_fiff_length(fid)


def write_int_matrix(fid, kind, mat):
    """Write integer 32 matrix tag."""
    FIFFT_MATRIX = 1 << 30
    FIFFT_MATRIX_INT = FIFF.FIFFT_INT | FIFFT_MATRIX

    data_size = 4 * mat.size + 4 * 3

    fid.write(np.array(kind, dtype='>i4').tobytes())
    fid.write(np.array(FIFFT_MATRIX_INT, dtype='>i4').tobytes())
    fid.write(np.array(data_size, dtype='>i4').tobytes())
    fid.write(np.array(FIFF.FIFFV_NEXT_SEQ, dtype='>i4').tobytes())
    fid.write(np.array(mat, dtype='>i4').tobytes())

    dims = np.empty(3, dtype=np.int32)
    dims[0] = mat.shape[1]
    dims[1] = mat.shape[0]
    dims[2] = 2
    fid.write(np.array(dims, dtype='>i4').tobytes())
    check_fiff_length(fid)


def write_complex_float_matrix(fid, kind, mat):
    """Write complex 64 matrix tag."""
    FIFFT_MATRIX = 1 << 30
    FIFFT_MATRIX_COMPLEX_FLOAT = FIFF.FIFFT_COMPLEX_FLOAT | FIFFT_MATRIX

    data_size = 4 * 2 * mat.size + 4 * (mat.ndim + 1)

    fid.write(np.array(kind, dtype='>i4').tobytes())
    fid.write(np.array(FIFFT_MATRIX_COMPLEX_FLOAT, dtype='>i4').tobytes())
    fid.write(np.array(data_size, dtype='>i4').tobytes())
    fid.write(np.array(FIFF.FIFFV_NEXT_SEQ, dtype='>i4').tobytes())
    fid.write(np.array(mat, dtype='>c8').tobytes())

    dims = np.empty(mat.ndim + 1, dtype=np.int32)
    dims[:mat.ndim] = mat.shape[::-1]
    dims[-1] = mat.ndim
    fid.write(np.array(dims, dtype='>i4').tobytes())
    check_fiff_length(fid)


def write_complex_double_matrix(fid, kind, mat):
    """Write complex 128 matrix tag."""
    FIFFT_MATRIX = 1 << 30
    FIFFT_MATRIX_COMPLEX_DOUBLE = FIFF.FIFFT_COMPLEX_DOUBLE | FIFFT_MATRIX

    data_size = 8 * 2 * mat.size + 4 * (mat.ndim + 1)

    fid.write(np.array(kind, dtype='>i4').tobytes())
    fid.write(np.array(FIFFT_MATRIX_COMPLEX_DOUBLE, dtype='>i4').tobytes())
    fid.write(np.array(data_size, dtype='>i4').tobytes())
    fid.write(np.array(FIFF.FIFFV_NEXT_SEQ, dtype='>i4').tobytes())
    fid.write(np.array(mat, dtype='>c16').tobytes())

    dims = np.empty(mat.ndim + 1, dtype=np.int32)
    dims[:mat.ndim] = mat.shape[::-1]
    dims[-1] = mat.ndim
    fid.write(np.array(dims, dtype='>i4').tobytes())
    check_fiff_length(fid)


def get_machid():
    """Get (mostly) unique machine ID.

    Returns
    -------
    ids : array (length 2, int32)
        The machine identifier used in MNE.
    """
    mac = b'%012x' % uuid.getnode()  # byte conversion for Py3
    mac = re.findall(b'..', mac)  # split string
    mac += [b'00', b'00']  # add two more fields

    # Convert to integer in reverse-order (for some reason)
    from codecs import encode
    mac = b''.join([encode(h, 'hex_codec') for h in mac[::-1]])
    ids = np.flipud(np.frombuffer(mac, np.int32, count=2))
    return ids


def get_new_file_id():
    """Create a new file ID tag."""
    secs, usecs = divmod(time.time(), 1.)
    secs, usecs = int(secs), int(usecs * 1e6)
    return {'machid': get_machid(), 'version': FIFF.FIFFC_VERSION,
            'secs': secs, 'usecs': usecs}


def write_id(fid, kind, id_=None):
    """Write fiff id."""
    id_ = _generate_meas_id() if id_ is None else id_

    data_size = 5 * 4                       # The id comprises five integers
    fid.write(np.array(kind, dtype='>i4').tobytes())
    fid.write(np.array(FIFF.FIFFT_ID_STRUCT, dtype='>i4').tobytes())
    fid.write(np.array(data_size, dtype='>i4').tobytes())
    fid.write(np.array(FIFF.FIFFV_NEXT_SEQ, dtype='>i4').tobytes())

    # Collect the bits together for one write
    arr = np.array([id_['version'],
                    id_['machid'][0], id_['machid'][1],
                    id_['secs'], id_['usecs']], dtype='>i4')
    fid.write(arr.tobytes())


def start_block(fid, kind):
    """Write a FIFF_BLOCK_START tag."""
    write_int(fid, FIFF.FIFF_BLOCK_START, kind)


def end_block(fid, kind):
    """Write a FIFF_BLOCK_END tag."""
    write_int(fid, FIFF.FIFF_BLOCK_END, kind)


def start_file(fname, id_=None):
    """Open a fif file for writing and writes the compulsory header tags.

    Parameters
    ----------
    fname : string | fid
        The name of the file to open. It is recommended
        that the name ends with .fif or .fif.gz. Can also be an
        already opened file.
    id_ : dict | None
        ID to use for the FIFF_FILE_ID.
    """
    if _file_like(fname):
        logger.debug('Writing using %s I/O' % type(fname))
        fid = fname
        fid.seek(0)
    else:
        fname = str(fname)
        if op.splitext(fname)[1].lower() == '.gz':
            logger.debug('Writing using gzip')
            # defaults to compression level 9, which is barely smaller but much
            # slower. 2 offers a good compromise.
            fid = GzipFile(fname, "wb", compresslevel=2)
        else:
            logger.debug('Writing using normal I/O')
            fid = open(fname, "wb")
    #   Write the compulsory items
    write_id(fid, FIFF.FIFF_FILE_ID, id_)
    write_int(fid, FIFF.FIFF_DIR_POINTER, -1)
    write_int(fid, FIFF.FIFF_FREE_LIST, -1)
    return fid


def check_fiff_length(fid, close=True):
    """Ensure our file hasn't grown too large to work properly."""
    if fid.tell() > 2147483648:  # 2 ** 31, FIFF uses signed 32-bit locations
        if close:
            fid.close()
        raise IOError('FIFF file exceeded 2GB limit, please split file, reduce'
                      ' split_size (if possible), or save to a different '
                      'format')


def end_file(fid):
    """Write the closing tags to a fif file and closes the file."""
    write_nop(fid, last=True)
    check_fiff_length(fid)
    fid.close()


def write_coord_trans(fid, trans):
    """Write a coordinate transformation structure."""
    data_size = 4 * 2 * 12 + 4 * 2
    fid.write(np.array(FIFF.FIFF_COORD_TRANS, dtype='>i4').tobytes())
    fid.write(np.array(FIFF.FIFFT_COORD_TRANS_STRUCT, dtype='>i4').tobytes())
    fid.write(np.array(data_size, dtype='>i4').tobytes())
    fid.write(np.array(FIFF.FIFFV_NEXT_SEQ, dtype='>i4').tobytes())
    fid.write(np.array(trans['from'], dtype='>i4').tobytes())
    fid.write(np.array(trans['to'], dtype='>i4').tobytes())

    #   The transform...
    rot = trans['trans'][:3, :3]
    move = trans['trans'][:3, 3]
    fid.write(np.array(rot, dtype='>f4').tobytes())
    fid.write(np.array(move, dtype='>f4').tobytes())

    #   ...and its inverse
    trans_inv = np.linalg.inv(trans['trans'])
    rot = trans_inv[:3, :3]
    move = trans_inv[:3, 3]
    fid.write(np.array(rot, dtype='>f4').tobytes())
    fid.write(np.array(move, dtype='>f4').tobytes())


def write_ch_info(fid, ch):
    """Write a channel information record to a fif file."""
    data_size = 4 * 13 + 4 * 7 + 16

    fid.write(np.array(FIFF.FIFF_CH_INFO, dtype='>i4').tobytes())
    fid.write(np.array(FIFF.FIFFT_CH_INFO_STRUCT, dtype='>i4').tobytes())
    fid.write(np.array(data_size, dtype='>i4').tobytes())
    fid.write(np.array(FIFF.FIFFV_NEXT_SEQ, dtype='>i4').tobytes())

    #   Start writing fiffChInfoRec
    fid.write(np.array(ch['scanno'], dtype='>i4').tobytes())
    fid.write(np.array(ch['logno'], dtype='>i4').tobytes())
    fid.write(np.array(ch['kind'], dtype='>i4').tobytes())
    fid.write(np.array(ch['range'], dtype='>f4').tobytes())
    fid.write(np.array(ch['cal'], dtype='>f4').tobytes())
    fid.write(np.array(ch['coil_type'], dtype='>i4').tobytes())
    fid.write(np.array(ch['loc'], dtype='>f4').tobytes())  # writing 12 values

    #   unit and unit multiplier
    fid.write(np.array(ch['unit'], dtype='>i4').tobytes())
    fid.write(np.array(ch['unit_mul'], dtype='>i4').tobytes())

    #   Finally channel name
    ch_name = ch['ch_name'][:15]
    fid.write(np.array(ch_name, dtype='>c').tobytes())
    fid.write(b'\0' * (16 - len(ch_name)))


def write_dig_points(fid, dig, block=False, coord_frame=None):
    """Write a set of digitizer data points into a fif file."""
    if dig is not None:
        data_size = 5 * 4
        if block:
            start_block(fid, FIFF.FIFFB_ISOTRAK)
        if coord_frame is not None:
            write_int(fid, FIFF.FIFF_MNE_COORD_FRAME, coord_frame)
        for d in dig:
            fid.write(np.array(FIFF.FIFF_DIG_POINT, '>i4').tobytes())
            fid.write(np.array(FIFF.FIFFT_DIG_POINT_STRUCT, '>i4').tobytes())
            fid.write(np.array(data_size, dtype='>i4').tobytes())
            fid.write(np.array(FIFF.FIFFV_NEXT_SEQ, '>i4').tobytes())
            #   Start writing fiffDigPointRec
            fid.write(np.array(d['kind'], '>i4').tobytes())
            fid.write(np.array(d['ident'], '>i4').tobytes())
            fid.write(np.array(d['r'][:3], '>f4').tobytes())
        if block:
            end_block(fid, FIFF.FIFFB_ISOTRAK)


def write_float_sparse_rcs(fid, kind, mat):
    """Write a single-precision sparse compressed row matrix tag."""
    return write_float_sparse(fid, kind, mat, fmt='csr')


def write_float_sparse_ccs(fid, kind, mat):
    """Write a single-precision sparse compressed column matrix tag."""
    return write_float_sparse(fid, kind, mat, fmt='csc')


def write_float_sparse(fid, kind, mat, fmt='auto'):
    """Write a single-precision floating-point sparse matrix tag."""
    from scipy import sparse
    from .tag import _matrix_coding_CCS, _matrix_coding_RCS
    if fmt == 'auto':
        fmt = 'csr' if isinstance(mat, sparse.csr_matrix) else 'csc'
    if fmt == 'csr':
        need = sparse.csr_matrix
        bits = _matrix_coding_RCS
    else:
        need = sparse.csc_matrix
        bits = _matrix_coding_CCS
    if not isinstance(mat, need):
        raise TypeError('Must write %s, got %s' % (fmt.upper(), type(mat),))
    FIFFT_MATRIX = bits << 16
    FIFFT_MATRIX_FLOAT_RCS = FIFF.FIFFT_FLOAT | FIFFT_MATRIX

    nnzm = mat.nnz
    nrow = mat.shape[0]
    data_size = 4 * nnzm + 4 * nnzm + 4 * (nrow + 1) + 4 * 4

    fid.write(np.array(kind, dtype='>i4').tobytes())
    fid.write(np.array(FIFFT_MATRIX_FLOAT_RCS, dtype='>i4').tobytes())
    fid.write(np.array(data_size, dtype='>i4').tobytes())
    fid.write(np.array(FIFF.FIFFV_NEXT_SEQ, dtype='>i4').tobytes())

    fid.write(np.array(mat.data, dtype='>f4').tobytes())
    fid.write(np.array(mat.indices, dtype='>i4').tobytes())
    fid.write(np.array(mat.indptr, dtype='>i4').tobytes())

    dims = [nnzm, mat.shape[0], mat.shape[1], 2]
    fid.write(np.array(dims, dtype='>i4').tobytes())
    check_fiff_length(fid)


def _generate_meas_id():
    """Generate a new meas_id dict."""
    id_ = dict()
    id_['version'] = FIFF.FIFFC_VERSION
    id_['machid'] = get_machid()
    id_['secs'], id_['usecs'] = DATE_NONE
    return id_
