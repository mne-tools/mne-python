import numpy as np


def read_stc(filename):
    """Read an STC file

    STC files contain activations or source reconstructions

    Parameters
    ----------
    filename: string
        The name of the STC file

    Returns
    -------
    data: dict
        The STC structure. It has the following keys:
           tmin           The first time point of the data in seconds
           tstep          Time between frames in seconds
           vertices       vertex indices (0 based)
           data           The data matrix (nvert * ntime)
    """
    fid = open(filename, 'rb')

    stc = dict()

    fid.seek(0, 2) # go to end of file
    file_length = fid.tell()
    fid.seek(0, 0) # go to beginning of file

    # read tmin in ms
    stc['tmin'] = float(np.fromfile(fid, dtype=">f4", count=1))
    stc['tmin'] /= 1000.0

    # read sampling rate in ms
    stc['tstep'] = float(np.fromfile(fid, dtype=">f4", count=1))
    stc['tstep'] /= 1000.0

    # read number of vertices/sources
    vertices_n = np.fromfile(fid, dtype=">I4", count=1)

    # read the source vector
    stc['vertices'] = np.fromfile(fid, dtype=">I4", count=vertices_n)

    # read the number of timepts
    data_n = np.fromfile(fid, dtype=">I4", count=1)

    if ((file_length/4 -4 - vertices_n) % (data_n*vertices_n)) != 0:
        raise ValueError, 'incorrect stc file size'

    # read the data matrix
    stc['data'] = np.fromfile(fid, dtype=">f4", count=vertices_n*data_n)
    stc['data'] = stc['data'].reshape([data_n, vertices_n]).T

    # close the file
    fid.close()
    return stc


def write_stc(filename, stc):
    """Write an STC file

    Parameters
    ----------
    filename: string
        The name of the STC file

    stc: dict
        The STC structure. It has the following keys:
           tmin           The first time point of the data in seconds
           tstep          Time between frames in seconds
           vertices       vertex indices (0 based)
           data           The data matrix (nvert * ntime)
    """
    fid = open(filename, 'wb')

    # write start time in ms
    fid.write(np.array(1000*stc['tmin'], dtype='>f4').tostring())
    # write sampling rate in ms
    fid.write(np.array(1000*stc['tstep'], dtype='>f4').tostring())
    # write number of vertices
    fid.write(np.array(stc['vertices'].shape[0], dtype='>I4').tostring())
    # write the vertex indices
    fid.write(np.array(stc['vertices'], dtype='>I4').tostring())

    # write the number of timepts
    fid.write(np.array(stc['data'].shape[1], dtype='>I4').tostring())
    #
    # write the data
    #
    fid.write(np.array(stc['data'].T, dtype='>f4').tostring())

    # close the file
    fid.close()
