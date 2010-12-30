import time
import numpy as np

from .constants import FIFF


def _write(fid, data, kind, data_size, FIFFT_TYPE, dtype):
    FIFFV_NEXT_SEQ = 0
    if isinstance(data, np.ndarray):
        data_size *= data.size
    fid.write(np.array(kind, dtype='>i4').tostring())
    fid.write(np.array(FIFFT_TYPE, dtype='>i4').tostring())
    fid.write(np.array(data_size, dtype='>i4').tostring())
    fid.write(np.array(FIFFV_NEXT_SEQ, dtype='>i4').tostring())
    fid.write(np.array(data, dtype=dtype).tostring())


def write_int(fid, kind, data):
    """
    %
    % fiff_write_int(fid,kind,data)
    %
    % Writes a 32-bit integer tag to a fif file
    %
    %     fid           An open fif file descriptor
    %     kind          Tag kind
    %     data          The integers to use as data
    %
    """
    FIFFT_INT = 3
    data_size = 4
    _write(fid, data, kind, data_size, FIFFT_INT, '>i4')


def write_double(fid, kind, data):
    """
    %
    % fiff_write_int(fid,kind,data)
    %
    % Writes a double-precision floating point tag to a fif file
    %
    %     fid           An open fif file descriptor
    %     kind          Tag kind
    %     data          The data
    %
    """
    FIFFT_DOUBLE = 5
    data_size = 8
    _write(fid, data, kind, data_size, FIFFT_DOUBLE, '>f8')


def write_float(fid, kind, data):
    """
    %
    % fiff_write_float(fid,kind,data)
    %
    % Writes a single-precision floating point tag to a fif file
    %
    %     fid           An open fif file descriptor
    %     kind          Tag kind
    %     data          The data
    %
    """
    FIFFT_FLOAT = 4
    data_size = 4
    _write(fid, data, kind, data_size, FIFFT_FLOAT, '>f4')


def write_string(fid, kind, data):
    """
    %
    % fiff_write_string(fid,kind,data)
    %
    % Writes a string tag
    %
    %     fid           An open fif file descriptor
    %     kind          The tag kind
    %     data          The string data to write
    %
    """
    FIFFT_STRING = 10
    data_size = len(data)
    _write(fid, data, kind, data_size, FIFFT_STRING, '>c')


def write_name_list(fid, kind, data):
    """
    %
    % fiff_write_name_list(fid,kind,mat)
    %
    % Writes a colon-separated list of names
    %
    %     fid           An open fif file descriptor
    %     kind          The tag kind
    %     data          An array of names to create the list from
    %
    """
    write_string(fid, kind, ':'.join(data))


def write_float_matrix(fid, kind, data):
    """
    %
    % fiff_write_float_matrix(fid,kind,mat)
    %
    % Writes a single-precision floating-point matrix tag
    %
    %     fid           An open fif file descriptor
    %     kind          The tag kind
    %     data          The data matrix
    %
    """

    FIFFT_FLOAT = 4
    FIFFT_MATRIX = 1 << 30
    FIFFT_MATRIX_FLOAT = FIFFT_FLOAT | FIFFT_MATRIX
    data_size = 4*data.size + 4*3

    _write(fid, data, kind, data_size, FIFFT_MATRIX_FLOAT, '>f4')

    dims = np.empty(3, dtype=np.int)
    dims[0] = data.shape[1]
    dims[1] = data.shape[0]
    dims[3] = 2
    fid.write(np.array(dims, dtype='>i4').tostring())


def write_id(fid, kind, id_=None):
    """
    %
    % fiff_write_id(fid, kind, id)
    %
    % Writes fiff id
    %
    %     fid           An open fif file descriptor
    %     kind          The tag kind
    %     id            The id to write
    %
    % If the id argument is missing it will be generated here
    %
    """

    if id_ is None:
        id_ = dict()
        id_['version'] = (1 << 16) | 2            # Version (1 << 16) | 2
        id_['machid'] = 65536 * np.random.rand(2) # Machine id (andom for now)
        id_['secs'] = time.time()
        id_['usecs'] = 0            #   Do not know how we could get this XXX

    FIFFT_ID_STRUCT = 31
    FIFFV_NEXT_SEQ = 0

    data_size = 5*4                       #   The id comprises five integers
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
    """
    %
    % fiff_start_block(fid,kind)
    %
    % Writes a FIFF_BLOCK_START tag
    %
    %     fid           An open fif file descriptor
    %     kind          The block kind to start
    %
    """

    FIFF_BLOCK_START = 104
    write_int(fid, FIFF_BLOCK_START, kind)


def end_block(fid, kind):
    """
    %
    % fiff_end_block(fid,kind)
    %
    % Writes a FIFF_BLOCK_END tag
    %
    %     fid           An open fif file descriptor
    %     kind          The block kind to end
    %
    """

    FIFF_BLOCK_END = 105
    write_int(fid, FIFF_BLOCK_END, kind)


def start_file(name):
    """
    %
    % [fid] = fiff_start_file(name)
    %
    % Opens a fif file for writing and writes the compulsory header tags
    %
    %     name           The name of the file to open. It is recommended
    %                    that the name ends with .fif
    %
    """
    fid = open(name, 'wb')

    #   Write the compulsory items
    FIFF_FILE_ID = 100
    FIFF_DIR_POINTER = 101
    FIFF_FREE_LIST = 106

    write_id(fid, FIFF_FILE_ID)
    write_int(fid, FIFF_DIR_POINTER, -1)
    write_int(fid, FIFF_FREE_LIST, -1)

    return fid


def end_file(fid):
    """
    %
    % fiff_end_file(fid)
    %
    % Writes the closing tags to a fif file and closes the file
    %
    %     fid           An open fif file descriptor
    %
    """

    data_size = 0

    fid.write(np.array(FIFF.FIFF_NOP, dtype='>i4').tostring())
    fid.write(np.array(FIFF.FIFFT_VOID, dtype='>i4').tostring())
    fid.write(np.array(data_size, dtype='>i4').tostring())
    fid.write(np.array(FIFF.FIFFV_NEXT_NONE, dtype='>i4').tostring())
    fid.close()
