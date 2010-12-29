import numpy as np


def read_stc(filename):
    """
    %
    % [stc] = mne_read_stc_file(filename)
    %
    % Reads an stc file. The returned structure has the following fields
    %
    %     tmin           The first time point of the data in seconds
    %     tstep          Time between frames in seconds
    %     vertices       vertex indices (0 based)
    %     data           The data matrix (nvert * ntime)
    %
    %
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
