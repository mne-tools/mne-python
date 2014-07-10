# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from ..externals.six import string_types, b
import time
import numpy as np
from scipy import linalg
import os.path as op
import re
import uuid

from .constants import FIFF
from ..utils import logger
from ..externals.jdcal import jcal2jd
from ..fixes import gzip_open


def _write(fid, data, kind, data_size, FIFFT_TYPE, dtype):
    if isinstance(data, np.ndarray):
        data_size *= data.size

    # XXX for string types the data size is used as
    # computed in ``write_string``.

    fid.write(np.array(kind, dtype='>i4').tostring())
    fid.write(np.array(FIFFT_TYPE, dtype='>i4').tostring())
    fid.write(np.array(data_size, dtype='>i4').tostring())
    fid.write(np.array(FIFF.FIFFV_NEXT_SEQ, dtype='>i4').tostring())
    fid.write(np.array(data, dtype=dtype).tostring())


def write_int(fid, kind, data):
    """Writes a 32-bit integer tag to a fif file"""
    data_size = 4
    data = np.array(data, dtype='>i4').T
    _write(fid, data, kind, data_size, FIFF.FIFFT_INT, '>i4')


def write_double(fid, kind, data):
    """Writes a double-precision floating point tag to a fif file"""
    data_size = 8
    data = np.array(data, dtype='>f8').T
    _write(fid, data, kind, data_size, FIFF.FIFFT_DOUBLE, '>f8')


def write_float(fid, kind, data):
    """Writes a single-precision floating point tag to a fif file"""
    data_size = 4
    data = np.array(data, dtype='>f4').T
    _write(fid, data, kind, data_size, FIFF.FIFFT_FLOAT, '>f4')


def write_dau_pack16(fid, kind, data):
    """Writes a dau_pack16 tag to a fif file"""
    data_size = 2
    data = np.array(data, dtype='>i2').T
    _write(fid, data, kind, data_size, FIFF.FIFFT_DAU_PACK16, '>i2')


def write_complex64(fid, kind, data):
    """Writes a 64 bit complex floating point tag to a fif file"""
    data_size = 8
    data = np.array(data, dtype='>c8').T
    _write(fid, data, kind, data_size, FIFF.FIFFT_COMPLEX_FLOAT, '>c8')


def write_complex128(fid, kind, data):
    """Writes a 128 bit complex floating point tag to a fif file"""
    data_size = 16
    data = np.array(data, dtype='>c16').T
    _write(fid, data, kind, data_size, FIFF.FIFFT_COMPLEX_FLOAT, '>c16')


def write_julian(fid, kind, data):
    """Writes a Julian-formatted date to a FIF file"""
    assert len(data) == 3
    data_size = 4
    jd = np.sum(jcal2jd(*data))
    data = np.array(jd, dtype='>i4')
    _write(fid, data, kind, data_size, FIFF.FIFFT_JULIAN, '>i4')


def write_string(fid, kind, data):
    """Writes a string tag"""

    str_data = data.encode('utf-8')  # Use unicode or bytes depending on Py2/3
    data_size = len(str_data)  # therefore compute size here
    my_dtype = '>a'  # py2/3 compatible on writing -- don't ask me why
    _write(fid, str_data, kind, data_size, FIFF.FIFFT_STRING, my_dtype)


def write_name_list(fid, kind, data):
    """Writes a colon-separated list of names

    Parameters
    ----------
    data : list of strings
    """
    write_string(fid, kind, ':'.join(data))


def write_float_matrix(fid, kind, mat):
    """Writes a single-precision floating-point matrix tag"""
    FIFFT_MATRIX = 1 << 30
    FIFFT_MATRIX_FLOAT = FIFF.FIFFT_FLOAT | FIFFT_MATRIX

    data_size = 4 * mat.size + 4 * (mat.ndim + 1)

    fid.write(np.array(kind, dtype='>i4').tostring())
    fid.write(np.array(FIFFT_MATRIX_FLOAT, dtype='>i4').tostring())
    fid.write(np.array(data_size, dtype='>i4').tostring())
    fid.write(np.array(FIFF.FIFFV_NEXT_SEQ, dtype='>i4').tostring())
    fid.write(np.array(mat, dtype='>f4').tostring())

    dims = np.empty(mat.ndim + 1, dtype=np.int32)
    dims[:mat.ndim] = mat.shape[::-1]
    dims[-1] = mat.ndim
    fid.write(np.array(dims, dtype='>i4').tostring())


def write_double_matrix(fid, kind, mat):
    """Writes a double-precision floating-point matrix tag"""
    FIFFT_MATRIX = 1 << 30
    FIFFT_MATRIX_DOUBLE = FIFF.FIFFT_DOUBLE | FIFFT_MATRIX

    data_size = 8 * mat.size + 4 * (mat.ndim + 1)

    fid.write(np.array(kind, dtype='>i4').tostring())
    fid.write(np.array(FIFFT_MATRIX_DOUBLE, dtype='>i4').tostring())
    fid.write(np.array(data_size, dtype='>i4').tostring())
    fid.write(np.array(FIFF.FIFFV_NEXT_SEQ, dtype='>i4').tostring())
    fid.write(np.array(mat, dtype='>f8').tostring())

    dims = np.empty(mat.ndim + 1, dtype=np.int32)
    dims[:mat.ndim] = mat.shape[::-1]
    dims[-1] = mat.ndim
    fid.write(np.array(dims, dtype='>i4').tostring())


def write_int_matrix(fid, kind, mat):
    """Writes integer 32 matrix tag"""
    FIFFT_MATRIX = 1 << 30
    FIFFT_MATRIX_INT = FIFF.FIFFT_INT | FIFFT_MATRIX

    data_size = 4 * mat.size + 4 * 3

    fid.write(np.array(kind, dtype='>i4').tostring())
    fid.write(np.array(FIFFT_MATRIX_INT, dtype='>i4').tostring())
    fid.write(np.array(data_size, dtype='>i4').tostring())
    fid.write(np.array(FIFF.FIFFV_NEXT_SEQ, dtype='>i4').tostring())
    fid.write(np.array(mat, dtype='>i4').tostring())

    dims = np.empty(3, dtype=np.int32)
    dims[0] = mat.shape[1]
    dims[1] = mat.shape[0]
    dims[2] = 2
    fid.write(np.array(dims, dtype='>i4').tostring())


def get_machid():
    """Get (mostly) unique machine ID

    Returns
    -------
    ids : array (length 2, int32)
        The machine identifier used in MNE.
    """
    mac = b('%012x' %uuid.getnode()) # byte conversion for Py3
    mac = re.findall(b'..', mac) # split string
    mac += [b'00', b'00']  # add two more fields

    # Convert to integer in reverse-order (for some reason)
    from codecs import encode
    mac = b''.join([encode(h, 'hex_codec') for h in mac[::-1]])
    ids = np.flipud(np.fromstring(mac, np.int32, count=2))
    return ids


def write_id(fid, kind, id_=None):
    """Writes fiff id"""
    id_ = _generate_meas_id() if id_ is None else id_

    FIFFT_ID_STRUCT = 31
    FIFFV_NEXT_SEQ = 0

    data_size = 5 * 4                       # The id comprises five integers
    fid.write(np.array(kind, dtype='>i4').tostring())
    fid.write(np.array(FIFFT_ID_STRUCT, dtype='>i4').tostring())
    fid.write(np.array(data_size, dtype='>i4').tostring())
    fid.write(np.array(FIFFV_NEXT_SEQ, dtype='>i4').tostring())

    # Collect the bits together for one write
    data = np.empty(5, dtype=np.int32)
    data[0] = id_['version']
    data[1] = id_['machid'][0]
    data[2] = id_['machid'][1]
    data[3] = id_['secs']
    data[4] = id_['usecs']
    fid.write(np.array(data, dtype='>i4').tostring())


def start_block(fid, kind):
    """Writes a FIFF_BLOCK_START tag"""
    write_int(fid, FIFF.FIFF_BLOCK_START, kind)


def end_block(fid, kind):
    """Writes a FIFF_BLOCK_END tag"""
    write_int(fid, FIFF.FIFF_BLOCK_END, kind)


def start_file(fname, id_=None):
    """Opens a fif file for writing and writes the compulsory header tags

    Parameters
    ----------
    fname : string | fid
        The name of the file to open. It is recommended
        that the name ends with .fif or .fif.gz. Can also be an
        already opened file.
    id_ : dict | None
        ID to use for the FIFF_FILE_ID.
    """
    if isinstance(fname, string_types):
        if op.splitext(fname)[1].lower() == '.gz':
            logger.debug('Writing using gzip')
            # defaults to compression level 9, which is barely smaller but much
            # slower. 2 offers a good compromise.
            fid = gzip_open(fname, "wb", compresslevel=2)
        else:
            logger.debug('Writing using normal I/O')
            fid = open(fname, "wb")
    else:
        logger.debug('Writing using %s I/O' % type(fname))
        fid = fname
        fid.seek(0)
    #   Write the compulsory items
    write_id(fid, FIFF.FIFF_FILE_ID, id_)
    write_int(fid, FIFF.FIFF_DIR_POINTER, -1)
    write_int(fid, FIFF.FIFF_FREE_LIST, -1)
    return fid


def end_file(fid):
    """Writes the closing tags to a fif file and closes the file"""
    data_size = 0
    fid.write(np.array(FIFF.FIFF_NOP, dtype='>i4').tostring())
    fid.write(np.array(FIFF.FIFFT_VOID, dtype='>i4').tostring())
    fid.write(np.array(data_size, dtype='>i4').tostring())
    fid.write(np.array(FIFF.FIFFV_NEXT_NONE, dtype='>i4').tostring())
    fid.close()


def write_coord_trans(fid, trans):
    """Writes a coordinate transformation structure"""

    #?typedef struct _fiffCoordTransRec {
    #  fiff_int_t   from;                          /*!< Source coordinate system. */
    #  fiff_int_t   to;                        /*!< Destination coordinate system. */
    #  fiff_float_t rot[3][3];             /*!< The forward transform (rotation part) */
    #  fiff_float_t move[3];                   /*!< The forward transform (translation part) */
    #  fiff_float_t invrot[3][3];              /*!< The inverse transform (rotation part) */
    #  fiff_float_t invmove[3];            /*!< The inverse transform (translation part) */
    #} *fiffCoordTrans, fiffCoordTransRec; /*!< Coordinate transformation descriptor */

    data_size = 4 * 2 * 12 + 4 * 2
    fid.write(np.array(FIFF.FIFF_COORD_TRANS, dtype='>i4').tostring())
    fid.write(np.array(FIFF.FIFFT_COORD_TRANS_STRUCT, dtype='>i4').tostring())
    fid.write(np.array(data_size, dtype='>i4').tostring())
    fid.write(np.array(FIFF.FIFFV_NEXT_SEQ, dtype='>i4').tostring())
    fid.write(np.array(trans['from'], dtype='>i4').tostring())
    fid.write(np.array(trans['to'], dtype='>i4').tostring())

    #   The transform...
    rot = trans['trans'][:3, :3]
    move = trans['trans'][:3, 3]
    fid.write(np.array(rot, dtype='>f4').tostring())
    fid.write(np.array(move, dtype='>f4').tostring())

    #   ...and its inverse
    trans_inv = linalg.inv(trans['trans'])
    rot = trans_inv[:3, :3]
    move = trans_inv[:3, 3]
    fid.write(np.array(rot, dtype='>f4').tostring())
    fid.write(np.array(move, dtype='>f4').tostring())


def write_ch_info(fid, ch):
    """Writes a channel information record to a fif file"""

    #typedef struct _fiffChPosRec {
    #  fiff_int_t   coil_type;      /*!< What kind of coil. */
    #  fiff_float_t r0[3];          /*!< Coil coordinate system origin */
    #  fiff_float_t ex[3];          /*!< Coil coordinate system x-axis unit vector */
    #  fiff_float_t ey[3];          /*!< Coil coordinate system y-axis unit vector */
    #  fiff_float_t ez[3];                   /*!< Coil coordinate system z-axis unit vector */
    #} fiffChPosRec,*fiffChPos;                /*!< Measurement channel position and coil type */

    #typedef struct _fiffChInfoRec {
    #  fiff_int_t    scanNo;    /*!< Scanning order # */
    #  fiff_int_t    logNo;     /*!< Logical channel # */
    #  fiff_int_t    kind;      /*!< Kind of channel */
    #  fiff_float_t  range;     /*!< Voltmeter range (only applies to raw data ) */
    #  fiff_float_t  cal;       /*!< Calibration from volts to... */
    #  fiff_ch_pos_t chpos;     /*!< Channel location */
    #  fiff_int_t    unit;      /*!< Unit of measurement */
    #  fiff_int_t    unit_mul;  /*!< Unit multiplier exponent */
    #  fiff_char_t   ch_name[16];   /*!< Descriptive name for the channel */
    #} fiffChInfoRec,*fiffChInfo;   /*!< Description of one channel */

    data_size = 4 * 13 + 4 * 7 + 16

    fid.write(np.array(FIFF.FIFF_CH_INFO, dtype='>i4').tostring())
    fid.write(np.array(FIFF.FIFFT_CH_INFO_STRUCT, dtype='>i4').tostring())
    fid.write(np.array(data_size, dtype='>i4').tostring())
    fid.write(np.array(FIFF.FIFFV_NEXT_SEQ, dtype='>i4').tostring())

    #   Start writing fiffChInfoRec
    fid.write(np.array(ch['scanno'], dtype='>i4').tostring())
    fid.write(np.array(ch['logno'], dtype='>i4').tostring())
    fid.write(np.array(ch['kind'], dtype='>i4').tostring())
    fid.write(np.array(ch['range'], dtype='>f4').tostring())
    fid.write(np.array(ch['cal'], dtype='>f4').tostring())
    fid.write(np.array(ch['coil_type'], dtype='>i4').tostring())
    fid.write(np.array(ch['loc'], dtype='>f4').tostring())  # writing 12 values

    #   unit and unit multiplier
    fid.write(np.array(ch['unit'], dtype='>i4').tostring())
    fid.write(np.array(ch['unit_mul'], dtype='>i4').tostring())

    #   Finally channel name
    if len(ch['ch_name']):
        ch_name = ch['ch_name'][:15]
    else:
        ch_name = ch['ch_name']

    fid.write(np.array(ch_name, dtype='>c').tostring())
    if len(ch_name) < 16:
        fid.write(b('\0') * (16 - len(ch_name)))


def write_dig_point(fid, dig):
    """Writes a digitizer data point into a fif file"""
    #?typedef struct _fiffDigPointRec {
    #  fiff_int_t kind;               /*!< FIFF_POINT_CARDINAL,
    #                                  *   FIFF_POINT_HPI, or
    #                                  *   FIFF_POINT_EEG */
    #  fiff_int_t ident;              /*!< Number identifying this point */
    #  fiff_float_t r[3];             /*!< Point location */
    #} *fiffDigPoint,fiffDigPointRec; /*!< Digitization point description */

    data_size = 5 * 4

    fid.write(np.array(FIFF.FIFF_DIG_POINT, dtype='>i4').tostring())
    fid.write(np.array(FIFF.FIFFT_DIG_POINT_STRUCT, dtype='>i4').tostring())
    fid.write(np.array(data_size, dtype='>i4').tostring())
    fid.write(np.array(FIFF.FIFFV_NEXT_SEQ, dtype='>i4').tostring())

    #   Start writing fiffDigPointRec
    fid.write(np.array(dig['kind'], dtype='>i4').tostring())
    fid.write(np.array(dig['ident'], dtype='>i4').tostring())
    fid.write(np.array(dig['r'][:3], dtype='>f4').tostring())


def write_float_sparse_rcs(fid, kind, mat):
    """Writes a single-precision floating-point matrix tag"""
    FIFFT_MATRIX = 16416 << 16
    FIFFT_MATRIX_FLOAT_RCS = FIFF.FIFFT_FLOAT | FIFFT_MATRIX

    nnzm = mat.nnz
    nrow = mat.shape[0]
    data_size = 4 * nnzm + 4 * nnzm + 4 * (nrow + 1) + 4 * 4

    fid.write(np.array(kind, dtype='>i4').tostring())
    fid.write(np.array(FIFFT_MATRIX_FLOAT_RCS, dtype='>i4').tostring())
    fid.write(np.array(data_size, dtype='>i4').tostring())
    fid.write(np.array(FIFF.FIFFV_NEXT_SEQ, dtype='>i4').tostring())

    fid.write(np.array(mat.data, dtype='>f4').tostring())
    fid.write(np.array(mat.indices, dtype='>i4').tostring())
    fid.write(np.array(mat.indptr, dtype='>i4').tostring())

    dims = [nnzm, mat.shape[0], mat.shape[1], 2]
    fid.write(np.array(dims, dtype='>i4').tostring())


def _generate_meas_id():
    """Helper to generate a new meas_id dict"""
    id_ = dict()
    id_['version'] = (1 << 16) | 2
    id_['machid'] = get_machid()
    id_['secs'] = time.time()
    id_['usecs'] = 0            # Do not know how we could get this XXX
    return id_
