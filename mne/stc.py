# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

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

    fid.seek(0, 2)  # go to end of file
    file_length = fid.tell()
    fid.seek(0, 0)  # go to beginning of file

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

    if ((file_length / 4 - 4 - vertices_n) % (data_n * vertices_n)) != 0:
        raise ValueError('incorrect stc file size')

    # read the data matrix
    stc['data'] = np.fromfile(fid, dtype=">f4", count=vertices_n * data_n)
    stc['data'] = stc['data'].reshape([data_n, vertices_n]).T

    # close the file
    fid.close()
    return stc


def write_stc(filename, tmin, tstep, vertices, data):
    """Write an STC file

    Parameters
    ----------
    filename: string
        The name of the STC file
    tmin: float
        The first time point of the data in seconds
    tstep: float
        Time between frames in seconds
    vertices: array of integers
        Vertex indices (0 based)
    data: 2D array
        The data matrix (nvert * ntime)
    """
    fid = open(filename, 'wb')

    # write start time in ms
    fid.write(np.array(1000 * tmin, dtype='>f4').tostring())
    # write sampling rate in ms
    fid.write(np.array(1000 * tstep, dtype='>f4').tostring())
    # write number of vertices
    fid.write(np.array(vertices.shape[0], dtype='>I4').tostring())
    # write the vertex indices
    fid.write(np.array(vertices, dtype='>I4').tostring())

    # write the number of timepts
    fid.write(np.array(data.shape[1], dtype='>I4').tostring())
    #
    # write the data
    #
    fid.write(np.array(data.T, dtype='>f4').tostring())

    # close the file
    fid.close()


class SourceEstimate(object):
    """SourceEstimate container

    Can be saved and loaded from .stc files

    Attributes
    ----------
    data : array of shape [n_dipoles x n_times]
        The data in source space
    times : array of shape [n_times]
        The time vector
    lh_vertno : array of shape [n_dipoles in left hemisphere]
        The indices of the dipoles in the left hemisphere
    rh_vertno : array of shape [n_dipoles in right hemisphere]
        The indices of the dipoles in the right hemisphere
    """
    def __init__(self, fname):
        if fname is not None:
            lh = read_stc(fname + '-lh.stc')
            rh = read_stc(fname + '-rh.stc')
            self.data = np.r_[lh['data'], rh['data']]
            assert lh['tmin'] == rh['tmin']
            assert lh['tstep'] == rh['tstep']
            self.tmin = lh['tmin']
            self.tstep = lh['tstep']
            self.times = self.tmin + self.tstep * np.arange(self.data.shape[1])
            self.lh_vertno = lh['vertices']
            self.rh_vertno = rh['vertices']

    def _init_times(self):
        """create self.times"""
        self.times = self.tmin + self.tstep * np.arange(self.data.shape[1])

    def save(self, fname):
        """save to source estimates to file"""
        lh_data = self.data[:len(self.lh_vertno)]
        rh_data = self.data[-len(self.rh_vertno):]

        print 'Writing STC to disk...',
        write_stc(fname + '-lh.stc', tmin=self.tmin, tstep=self.tstep,
                       vertices=self.lh_vertno, data=lh_data)
        write_stc(fname + '-rh.stc', tmin=self.tmin, tstep=self.tstep,
                       vertices=self.rh_vertno, data=rh_data)
        print '[done]'
