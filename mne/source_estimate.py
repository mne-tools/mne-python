# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import os
import copy
from math import ceil
import numpy as np
from scipy import linalg, sparse
from scipy.sparse import csr_matrix, coo_matrix
import warnings

import logging
logger = logging.getLogger('mne')

from .filter import resample
from .parallel import parallel_func
from .surface import read_surface
from .utils import get_subjects_dir
from .viz import plot_source_estimates
from . import verbose
from . fixes import in1d


def read_stc(filename):
    """Read an STC file and return as dict

    STC files contain activations or source reconstructions

    Parameters
    ----------
    filename : string
        The name of the STC file

    Returns
    -------
    data: dict
        The STC structure. It has the following keys:
           tmin           The first time point of the data in seconds
           tstep          Time between frames in seconds
           vertices       vertex indices (0 based)
           data           The data matrix (nvert * ntime)

    See Also
    --------
    read_source_estimate

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
    vertices_n = int(np.fromfile(fid, dtype=">u4", count=1))

    # read the source vector
    stc['vertices'] = np.fromfile(fid, dtype=">u4", count=vertices_n)

    # read the number of timepts
    data_n = int(np.fromfile(fid, dtype=">u4", count=1))

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
    filename : string
        The name of the STC file
    tmin : float
        The first time point of the data in seconds
    tstep : float
        Time between frames in seconds
    vertices : array of integers
        Vertex indices (0 based)
    data : 2D array
        The data matrix (nvert * ntime)
    """
    fid = open(filename, 'wb')

    # write start time in ms
    fid.write(np.array(1000 * tmin, dtype='>f4').tostring())
    # write sampling rate in ms
    fid.write(np.array(1000 * tstep, dtype='>f4').tostring())
    # write number of vertices
    fid.write(np.array(vertices.shape[0], dtype='>u4').tostring())
    # write the vertex indices
    fid.write(np.array(vertices, dtype='>u4').tostring())

    # write the number of timepts
    fid.write(np.array(data.shape[1], dtype='>u4').tostring())
    #
    # write the data
    #
    fid.write(np.array(data.T, dtype='>f4').tostring())

    # close the file
    fid.close()


def _read_3(fid):
    """ Read 3 byte integer from file
    """
    data = np.fromfile(fid, dtype=np.uint8, count=3).astype(np.int32)

    out = np.left_shift(data[0], 16) + np.left_shift(data[1], 8) + data[2]

    return out


def read_w(filename):
    """Read a w file and return as dict

    w files contain activations or source reconstructions for a single time
    point

    Parameters
    ----------
    filename : string
        The name of the w file

    Returns
    -------
    data: dict
        The w structure. It has the following keys:
           vertices       vertex indices (0 based)
           data           The data matrix (nvert long)
    """

    fid = open(filename, 'rb')

    # skip first 2 bytes
    fid.read(2)

    # read number of vertices/sources (3 byte integer)
    vertices_n = int(_read_3(fid))

    vertices = np.zeros((vertices_n), dtype=np.int32)
    data = np.zeros((vertices_n), dtype=np.float32)

    # read the vertices and data
    for i in range(vertices_n):
        vertices[i] = _read_3(fid)
        data[i] = np.fromfile(fid, dtype='>f4', count=1)

    w = dict()
    w['vertices'] = vertices
    w['data'] = data

    # close the file
    fid.close()
    return w


def _write_3(fid, val):
    """ Write 3 byte integer to file
    """

    f_bytes = np.zeros((3), dtype=np.uint8)

    f_bytes[0] = (val >> 16) & 255
    f_bytes[1] = (val >> 8) & 255
    f_bytes[2] = val & 255

    fid.write(f_bytes.tostring())


def write_w(filename, vertices, data):
    """Read a w file

    w files contain activations or source reconstructions for a single time
    point

    Parameters
    ----------
    filename: string
        The name of the w file
    vertices: array of integers
        Vertex indices (0 based)
    data: 1D array
        The data array (nvert)


    See Also
    --------
    read_source_estimate

    """

    assert(len(vertices) == len(data))

    fid = open(filename, 'wb')

    # write 2 zero bytes
    fid.write(np.zeros((2), dtype=np.uint8).tostring())

    # write number of vertices/sources (3 byte integer)
    vertices_n = len(vertices)
    _write_3(fid, vertices_n)

    # write the vertices and data
    for i in range(vertices_n):
        _write_3(fid, vertices[i])
        #XXX: without float() endianness is wrong, not sure why
        fid.write(np.array(float(data[i]), dtype='>f4').tostring())

    # close the file
    fid.close()


def read_source_estimate(fname):
    """Returns a SourceEstimate object.

    Parameters
    ----------
    The single argument ``fname`` should provide the path to (a)
    source-estimate file(s) as string.

     - for volume source estimates, ``fname`` should provide the path to a
       single file named '*-vl.stc`
     - for surface source estimates, ``fname`` should either provide the
       path to the file corresponding to a single hemisphere ('*-lh.stc',
       '*-rh.stc') or only specify the asterisk part in these patterns. In any
       case, the function expects files for both hemisphere with names
       following this pattern.
     - for single time point .w files, ``fname`` should follow the same
       pattern as for surface estimates, except that files are named
       '*-lh.w' and '*-rh.w'.

    See Also
    --------
    read_stc, read_w

    """
    fname_arg = fname

    # make sure corresponding file(s) can be found
    ftype = None
    if os.path.exists(fname):
        if fname.endswith('-vl.stc'):
            ftype = 'volume'
        elif fname.endswith('.stc'):
            ftype = 'surface'
            if fname.endswith(('-lh.stc', '-rh.stc')):
                fname = fname[:-7]
            else:
                err = ("Invalid .stc filename: %r; needs to end with "
                       "hemisphere tag ('...-lh.stc' or '...-rh.stc')"
                       % fname)
                raise IOError(err)
        elif fname.endswith('.w'):
            ftype = 'w'
            if fname.endswith(('-lh.w', '-rh.w')):
                fname = fname[:-5]
            else:
                err = ("Invalid .w filename: %r; needs to end with "
                       "hemisphere tag ('...-lh.w' or '...-rh.w')"
                       % fname)
                raise IOError(err)

    if ftype is not 'volume':
        stc_exist = map(os.path.exists, (fname + '-rh.stc', fname + '-lh.stc'))
        w_exist = map(os.path.exists, (fname + '-rh.w', fname + '-lh.w'))
        if all(stc_exist) and (ftype is not 'w'):
            ftype = 'surface'
        elif all(w_exist):
            ftype = 'w'
        elif any(stc_exist) or any(w_exist):
            raise IOError("Hemisphere missing for %r" % fname_arg)
        else:
            raise IOError("SourceEstimate File(s) not found for: %r"
                          % fname_arg)

    # read the files
    if ftype == 'volume':  # volume source space
        kwargs = read_stc(fname)
    elif ftype == 'surface':  # stc file with surface source spaces
        lh = read_stc(fname + '-lh.stc')
        rh = read_stc(fname + '-rh.stc')
        assert lh['tmin'] == rh['tmin']
        assert lh['tstep'] == rh['tstep']
        kwargs = lh.copy()
        kwargs['data'] = np.r_[lh['data'], rh['data']]
        kwargs['vertices'] = [lh['vertices'], rh['vertices']]
    elif ftype == 'w':  # w file with surface source spaces
        lh = read_w(fname + '-lh.w')
        rh = read_w(fname + '-rh.w')
        kwargs = lh.copy()
        kwargs['data'] = np.atleast_2d(np.r_[lh['data'], rh['data']]).T
        kwargs['vertices'] = [lh['vertices'], rh['vertices']]
        # w files only have a single time point
        kwargs['tmin'] = 0.0
        kwargs['tstep'] = 1.0

    return SourceEstimate(**kwargs)


class SourceEstimate(object):
    """SourceEstimate container

    Can be saved and loaded from .stc or .w files.

    Parameters
    ----------
    data : array of shape [n_dipoles x n_times]
        The data in source space.

    vertices : array | list of two arrays
        Vertex numbers corresponding to the data.

    tmin : scalar
        Time point of the first sample in data.

    tstep : scalar
        Time step between successive samples in data.

    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    .. note::
        For backwards compatibility, the SourceEstimate can also be
        initialized with a single argument, which can be ``None`` (an
        attribute-less SourceEstimate object will be returned) or a path as
        string, in which case the corresponding file(s) will be loaded. This
        usage is deprecated and will be removed in v0.6.

    Attributes
    ----------
    data : array of shape [n_dipoles x n_times]
        The data in source space.

    times : array of shape [n_times]
        The time vector.

    vertno : list of array of shape [n_dipoles in each source space]
        The indices of the dipoles in the different source spaces.

    """
    @verbose
    def __init__(self, data, vertices=None, tmin=None, tstep=None,
                 verbose=None):
        if data is None:
            warnings.warn('Constructing a SourceEstimate object with no '
                          'attributes is deprecated and will stop working in '
                          'v0.6. Use the proper constructor.')
            return
        elif isinstance(data, basestring):
            warnings.warn('Constructing a SourceEstimate object with a '
                          'filename is deprecated and will stop working in '
                          'v0.6. Use read_source_estimate().')
            se = read_source_estimate(data)
            data = se.data
            tmin = se.tmin
            tstep = se.tstep
            vertices = se.vertno

        if isinstance(vertices, list):
            if not (len(vertices) == 2 or len(vertices) == 1) or \
                    not all([isinstance(v, np.ndarray) for v in vertices]):
                raise ValueError('Vertices, if a list, must contain one or '
                                 'two numpy arrays')

        self.data = data
        self.tmin = tmin
        self.tstep = tstep
        self.times = None
        self._update_times()
        self.vertno = vertices
        self.verbose = verbose

    @verbose
    def save(self, fname, ftype='stc', verbose=None):
        """Save the source estimates to a file

        Parameters
        ----------
        fname : string
            The stem of the file name. The file names used for surface source
            spaces are obtained by adding "-lh.stc" and "-rh.stc" (or "-lh.w"
            and "-rh.w") to the stem provided, for the left and the right
            hemisphere, respectively. For volume source spaces, the stem is
            extended with "-vl.stc".
        ftype : string
            File format to use. Allowed values are "stc" (default) and "w".
            The "stc" format can be for surface and volume source spaces,
            while the "w" format only supports surface source spaces with a
            single time point.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
            Defaults to self.verbose.
        """
        if self.is_surface():
            lh_data = self.data[:len(self.lh_vertno)]
            rh_data = self.data[-len(self.rh_vertno):]

            if ftype == 'stc':
                logger.info('Writing STC to disk...')
                write_stc(fname + '-lh.stc', tmin=self.tmin, tstep=self.tstep,
                          vertices=self.lh_vertno, data=lh_data)
                write_stc(fname + '-rh.stc', tmin=self.tmin, tstep=self.tstep,
                          vertices=self.rh_vertno, data=rh_data)
            elif ftype == 'w':
                if self.data.shape[1] != 1:
                    raise ValueError('w files can only contain a single time '
                                     'point')
                logger.info('Writing STC to disk (w format)...')
                write_w(fname + '-lh.w', vertices=self.lh_vertno,
                        data=lh_data[:, 0])
                write_w(fname + '-rh.w', vertices=self.rh_vertno,
                        data=rh_data[:, 0])
            else:
                raise ValueError('invalid file type')
        else:
            if ftype != 'stc':
                raise ValueError('ftype has to be \"stc\" volume source '
                                 'spaces')
            logger.info('Writing STC to disk...')
            if not fname.endswith('-vl.stc'):
                fname += '-vl.stc'
            write_stc(fname, tmin=self.tmin, tstep=self.tstep,
                      vertices=self.vertno[0], data=self.data)
        logger.info('[done]')

    def __repr__(self):
        s = "%d vertices" % sum([len(v) for v in self.vertno])
        s += ", tmin : %s (ms)" % (1e3 * self.tmin)
        s += ", tmax : %s (ms)" % (1e3 * self.times[-1])
        s += ", tstep : %s (ms)" % (1e3 * self.tstep)
        s += ", data size : %s x %s" % self.data.shape
        return "<SourceEstimate  |  %s>" % s

    def crop(self, tmin=None, tmax=None):
        """Restrict SourceEstimate to a time interval

        Parameters
        ----------
        tmin : float or None
            The first time point in seconds. If None the first present is used.
        tmax : float or None
            The last time point in seconds. If None the last present is used.
        """
        mask = np.ones(len(self.times), dtype=np.bool)
        if tmax is not None:
            mask = mask & (self.times <= tmax)
        if tmin is not None:
            mask = mask & (self.times >= tmin)
            self.tmin = tmin
        self.data = self.data[:, mask]
        self._update_times()

    @verbose
    def resample(self, sfreq, npad=100, window='boxcar', n_jobs=1,
                 verbose=None):
        """Resample data

        Parameters
        ----------
        sfreq : float
            New sample rate to use.
        npad : int
            Amount to pad the start and end of the data.
        window : string or tuple
            Window to use in resampling. See scipy.signal.resample.
        n_jobs : int
            Number of jobs to run in parallel.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
            Defaults to self.verbose.

        Notes
        -----
        For some data, it may be more accurate to use npad=0 to reduce
        artifacts. This is dataset dependent -- check your data!

        Note that the sample rate of the original data is inferred from tstep.
        """
        o_sfreq = 1.0 / self.tstep
        self.data = resample(self.data, sfreq, o_sfreq, npad, n_jobs=n_jobs)
        # adjust indirectly affected variables
        self.tstep = 1.0 / sfreq
        self._update_times()

    @property
    def lh_data(self):
        return self.data[:len(self.lh_vertno)]

    @property
    def rh_data(self):
        return self.data[len(self.lh_vertno):]

    @property
    def lh_vertno(self):
        return self.vertno[0]

    @property
    def rh_vertno(self):
        return self.vertno[1]

    def is_surface(self):
        """Returns True if source estimate is defined over surfaces
        """
        if len(self.vertno) == 1:
            return False
        else:
            return True

    def _update_times(self):
        """Update the times attribute after changing tmin, tmax, or tstep"""
        self.times = self.tmin + (self.tstep * np.arange(self.data.shape[1]))

    def __add__(self, a):
        stc = copy.deepcopy(self)
        stc += a
        return stc

    def __iadd__(self, a):
        if isinstance(a, SourceEstimate):
            self.data += a.data
        else:
            self.data += a
        return self

    def __sub__(self, a):
        stc = copy.deepcopy(self)
        stc -= a
        return stc

    def __isub__(self, a):
        if isinstance(a, SourceEstimate):
            self.data -= a.data
        else:
            self.data -= a
        return self

    def __div__(self, a):
        stc = copy.deepcopy(self)
        stc /= a
        return stc

    def __idiv__(self, a):
        if isinstance(a, SourceEstimate):
            self.data /= a.data
        else:
            self.data /= a
        return self

    def __mul__(self, a):
        stc = copy.deepcopy(self)
        stc *= a
        return stc

    def __imul__(self, a):
        if isinstance(a, SourceEstimate):
            self.data *= a.data
        else:
            self.data *= a
        return self

    def __pow__(self, a):
        stc = copy.deepcopy(self)
        stc **= a
        return stc

    def __ipow__(self, a):
        self.data **= a
        return self

    def __radd__(self, a):
        return self + a

    def __rsub__(self, a):
        return self - a

    def __rmul__(self, a):
        return self * a

    def __rdiv__(self, a):
        return self / a

    def __neg__(self):
        stc = copy.deepcopy(self)
        stc.data *= -1
        return stc

    def __pos__(self):
        return self

    def sqrt(self):
        return self ** (0.5)

    def bin(self, width, tstart=None, tstop=None, func=np.mean):
        """Returns a SourceEstimate object with data summarized over time bins

        Time bins of ``width`` seconds. This method is intended for
        visualization only. No filter is applied to the data before binning,
        making the method inappropriate as a tool for downsampling data.

        Parameters
        ----------
        width : scalar
            Width of the individual bins in seconds.

        func : callable
            Function that is applied to summarize the data. Needs to accept a
            numpy.array as first input and an ``axis`` keyword argument.

        tstart : scalar | None
            Time point where the first bin starts. The default is the first
            time point of the stc.

        tstop : scalar | None
            Last possible time point contained in a bin (if the last bin would
            be shorter than width it is dropped). The default is the last time
            point of the stc.
        """
        if tstart is None:
            tstart = self.tmin
        if tstop is None:
            tstop = self.times[-1]

        times = np.arange(tstart, tstop + self.tstep, width)
        nv, _ = self.data.shape
        nt = len(times) - 1
        data = np.empty((nv, nt), dtype=self.data.dtype)
        for i in xrange(nt):
            idx = (self.times >= times[i]) & (self.times < times[i + 1])
            data[:, i] = func(self.data[:, idx], axis=1)

        tmin = times[0] + width / 2.
        stc = SourceEstimate(data, vertices=self.vertno,
                             tmin=tmin, tstep=width)
        return stc

    def _hemilabel_stc(self, label):
        is_surface = self.is_surface()

        # find applicable SourceEstimate vertices
        if is_surface:
            if label.hemi == 'lh':
                stc_vertices = self.vertno[0]
            else:
                stc_vertices = self.vertno[1]
        else:
            stc_vertices = self.vertno[0]

        # find index of the Label's vertices
        idx = np.nonzero(map(label.vertices.__contains__, stc_vertices))[0]

        # find output vertices
        vertices = stc_vertices[idx]

        # find data
        if is_surface and (label.hemi == 'rh'):
            values = self.data[idx + len(self.vertno[0])]
        else:
            values = self.data[idx]

        return vertices, values

    def in_label(self, label):
        """Returns a SourceEstimate object restricted to a label

        SourceEstimate contains the time course of
        activation of all sources inside the label.

        Parameters
        ----------
        label : Label | BiHemiLabel
            The label (as created for example by mne.read_label). If the label
            does not match any sources in the SourceEstimate, a ValueError is
            raised.
        """
        if not self.is_surface():
            raise NotImplementedError

        if label.hemi == 'both':
            lh_vert, lh_val = self._hemilabel_stc(label.lh)
            rh_vert, rh_val = self._hemilabel_stc(label.rh)
            vertices = [lh_vert, rh_vert]
            values = np.vstack((lh_val, rh_val))
        elif label.hemi == 'lh':
            lh_vert, values = self._hemilabel_stc(label)
            vertices = [lh_vert, np.array([])]
        elif label.hemi == 'rh':
            rh_vert, values = self._hemilabel_stc(label)
            vertices = [np.array([]), rh_vert]
        else:
            raise TypeError("Expected  Label or BiHemiLabel; got %r" % label)

        if sum(map(len, vertices)) == 0:
            raise ValueError('No vertices match the label in the stc file')

        label_stc = SourceEstimate(values, vertices=vertices,
                                   tmin=self.tmin, tstep=self.tstep)
        return label_stc

    @verbose
    def extract_label_time_course(self, labels, src, mode='mean_flip',
                                  allow_empty=False, verbose=None):
        """Extract label time courses for lists of labels

        This function will extract one time course for each label. The way the
        time courses are extracted depends on the mode parameter.

        Valid values for mode are:
        'mean': Average within each label.
        'mean_flip': Average within each label with sign flip depending on
        source orientation.
        'pca_flip': Apply an SVD to the time courses within each label and use
        the scaled and sign-flipped first right-singular vector as the label
        time course. The scaling is performed such that the power of the label
        time course is the same as the average per-vertex time course power
        within the label. The sign of the resulting time course is adjusted by
        multiplying it with "sign(dot(u, flip))" where u is the first
        left-singular vector, and flip is a sing-flip vector based on the
        vertex normals. This procedure assures that the phase does not
        randomly change by 180 degrees from one stc to the next.

        See also mne.extract_label_time_course to extract time courses for a
        list of SourceEstimate more efficiently.

        Parameters
        ----------
        labels : Label | list of Label
            The labels for which to extract the time courses.
        src : list
            Source spaces for left and right hemisphere.
        mode : str
            Extraction mode, see explanation above.
        allow_empty : bool
            Instead of emitting an error, return all-zero time course for
            labels that do not have any vertices in the source estimate.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).

        Returns
        -------
        label_tc : array, shape=(len(labels), n_times)
            Extracted time course for each label.
        """
        label_tc = extract_label_time_course(self, labels, src, mode=mode,
                                             return_generator=False,
                                             allow_empty=allow_empty,
                                             verbose=verbose)

        return label_tc

    def center_of_mass(self, subject, hemi=None, restrict_vertices=False,
                       subjects_dir=None):
        """Return the vertex on a given surface that is at the center of mass
        of  the activity in stc. Note that all activity must occur in a single
        hemisphere, otherwise an error is returned. The "mass" of each point in
        space for computing the spatial center of mass is computed by summing
        across time, and vice-versa for each point in time in computing the
        temporal center of mass. This is useful for quantifying spatio-temporal
        cluster locations, especially when combined with the function
        mne.source_space.vertex_to_mni().

        Parameters
        ----------
        subject : string
            The subject the stc is defined for.
        hemi : int, or None
            Calculate the center of mass for the left (0) or right (1)
            hemisphere. If None, one of the hemispheres must be all zeroes,
            and the center of mass will be calculated for the other
            hemisphere (useful for getting COM for clusters).
        restrict_vertices : bool, or array of int
            If True, returned vertex will be one from stc. Otherwise, it could
            be any vertex from surf. If an array of int, the returned vertex
            will come from that array. For most accuruate estimates, do not
            restrict vertices.
        subjects_dir : str, or None
            Path to the SUBJECTS_DIR. If None, the path is obtained by using
            the environment variable SUBJECTS_DIR.

        Returns
        -------
        vertex : int
            Vertex of the spatial center of mass for the inferred hemisphere,
            with each vertex weighted by the sum of the stc across time. For a
            boolean stc, then, this would be weighted purely by the duration
            each vertex was active.
        hemi : int
            Hemisphere the vertex was taken from.
        t : float
            Time of the temporal center of mass (weighted by the sum across
            source vertices).

        References:
            Used in Larson and Lee, "The cortical dynamics underlying effective
            switching of auditory spatial attention", NeuroImage 2012.
        """

        if not self.is_surface():
            raise ValueError('Finding COM must be done on surface')

        values = np.sum(self.data, axis=1)  # sum across time
        vert_inds = [np.arange(len(self.vertno[0])),
                     np.arange(len(self.vertno[1])) + len(self.vertno[0])]
        if hemi is None:
            hemi = np.where(np.array([np.sum(values[vi])
                            for vi in vert_inds]))[0]
            if not len(hemi) == 1:
                raise ValueError('Could not infer hemisphere')
            hemi = hemi[0]
        if not hemi in [0, 1]:
            raise ValueError('hemi must be 0 or 1')

        subjects_dir = get_subjects_dir(subjects_dir)

        values = values[vert_inds[hemi]]

        hemis = ['lh', 'rh']
        surf = os.path.join(subjects_dir, subject, 'surf',
                            hemis[hemi] + '.sphere')

        if isinstance(surf, basestring):  # read in surface
            surf = read_surface(surf)

        if restrict_vertices is False:
            restrict_vertices = np.arange(surf[0].shape[0])
        elif restrict_vertices is True:
            restrict_vertices = self.vertno[hemi]

        if np.any(self.data < 0):
            raise ValueError('Cannot compute COM with negative values')

        pos = surf[0][self.vertno[hemi], :].T
        c_o_m = np.sum(pos * values, axis=1) / np.sum(values)

        # Find the vertex closest to the COM
        vertex = np.argmin(np.sqrt(np.mean((surf[0][restrict_vertices, :] -
                                            c_o_m) ** 2, axis=1)))
        vertex = restrict_vertices[vertex]

        # do time center of mass by using the values across space
        masses = np.sum(self.data, axis=0).astype(float)
        t_ind = np.sum(masses * np.arange(self.data.shape[1])) / np.sum(masses)
        t = self.tmin + self.tstep * t_ind
        return vertex, hemi, t

    def plot(self, subject=None, surface='inflated', hemi='lh',
             colormap='hot', time_label='time=%0.2f ms',
             smoothing_steps=10, fmin=5., fmid=10., fmax=15.,
             transparent=True, time_viewer=False, subjects_dir=None):
        """Plot SourceEstimates with PySurfer

        Note: PySurfer currently needs the SUBJECTS_DIR environment variable,
        which will automatically be set by this function. Plotting multiple
        SourceEstimates with different values for subjects_dir will cause
        PySurfer to use the wrong FreeSurfer surfaces when using methods of
        the returned Brain object. It is therefore recommended to set the
        SUBJECTS_DIR environment variable or always use the same value for
        subjects_dir (within the same Python session).

        Parameters
        ----------
        stc : SourceEstimates
            The source estimates to plot.
        subject : str
            The subject name corresponding to FreeSurfer environment
            variable SUBJECT. If None the environment will be used.
        surface : str
            The type of surface (inflated, white etc.).
        hemi : str, 'lh' | 'rh'
            The hemisphere to display.
        colormap : str
            The type of colormap to use.
        time_label : str
            How to print info about the time instant visualized.
        smoothing_steps : int
            The amount of smoothing.
        fmin : float
            The minimum value to display.
        fmid : float
            The middle value on the colormap.
        fmax : float
            The maximum value for the colormap.
        transparent : bool
            If True, use a linear transparency between fmin and fmid.
        time_viewer : bool
            Display time viewer GUI.
        subjects_dir : str
            The path to the FreeSurfer subjects reconstructions.
            It corresponds to FreeSurfer environment variable SUBJECTS_DIR.

        Returns
        -------
        brain : Brain
            A instance of surfer.viz.Brain from PySurfer.
        """
        brain = plot_source_estimates(self, subject, surface=surface,
                        hemi=hemi, colormap=colormap, time_label=time_label,
                        smoothing_steps=smoothing_steps, fmin=fmin, fmid=fmid,
                        fmax=fmax, transparent=transparent,
                        time_viewer=time_viewer, subjects_dir=subjects_dir)
        return brain

###############################################################################
# Morphing

from .fiff.constants import FIFF
from .fiff.tag import find_tag
from .fiff.open import fiff_open
from .fiff.tree import dir_tree_find
from .surface import read_bem_surfaces


@verbose
def read_morph_map(subject_from, subject_to, subjects_dir=None,
                   verbose=None):
    """Read morph map generated with mne_make_morph_maps

    Parameters
    ----------
    subject_from : string
        Name of the original subject as named in the SUBJECTS_DIR
    subject_to : string
        Name of the subject on which to morph as named in the SUBJECTS_DIR
    subjects_dir : string
        Path to SUBJECTS_DIR is not set in the environment
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    left_map, right_map : sparse matrix
        The morph maps for the 2 hemispheres.
    """

    subjects_dir = get_subjects_dir(subjects_dir)

    # Does the file exist
    name = '%s/morph-maps/%s-%s-morph.fif' % (subjects_dir, subject_from,
                                              subject_to)
    if not os.path.exists(name):
        name = '%s/morph-maps/%s-%s-morph.fif' % (subjects_dir, subject_to,
                                                  subject_from)
        if not os.path.exists(name):
            raise ValueError('The requested morph map does not exist\n' +
                             'Perhaps you need to run the MNE tool:\n' +
                             '  mne_make_morph_maps --from %s --to %s'
                             % (subject_from, subject_to))

    fid, tree, _ = fiff_open(name)

    # Locate all maps
    maps = dir_tree_find(tree, FIFF.FIFFB_MNE_MORPH_MAP)
    if len(maps) == 0:
        fid.close()
        raise ValueError('Morphing map data not found')

    # Find the correct ones
    left_map = None
    right_map = None
    for m in maps:
        tag = find_tag(fid, m, FIFF.FIFF_MNE_MORPH_MAP_FROM)
        if tag.data == subject_from:
            tag = find_tag(fid, m, FIFF.FIFF_MNE_MORPH_MAP_TO)
            if tag.data == subject_to:
                #  Names match: which hemishere is this?
                tag = find_tag(fid, m, FIFF.FIFF_MNE_HEMI)
                if tag.data == FIFF.FIFFV_MNE_SURF_LEFT_HEMI:
                    tag = find_tag(fid, m, FIFF.FIFF_MNE_MORPH_MAP)
                    left_map = tag.data
                    logger.info('    Left-hemisphere map read.')
                elif tag.data == FIFF.FIFFV_MNE_SURF_RIGHT_HEMI:
                    tag = find_tag(fid, m, FIFF.FIFF_MNE_MORPH_MAP)
                    right_map = tag.data
                    logger.info('    Right-hemisphere map read.')

    fid.close()
    if left_map is None:
        raise ValueError('Left hemisphere map not found in %s' % name)

    if right_map is None:
        raise ValueError('Left hemisphere map not found in %s' % name)

    return left_map, right_map


def mesh_edges(tris):
    """Returns sparse matrix with edges as an adjacency matrix

    Parameters
    ----------
    tris : array of shape [n_triangles x 3]
        The triangles

    Returns
    -------
    edges : sparse matrix
        The adjacency matrix
    """
    npoints = np.max(tris) + 1
    ones_ntris = np.ones(len(tris))
    a, b, c = tris.T
    edges = coo_matrix((ones_ntris, (a, b)), shape=(npoints, npoints))
    edges = edges + coo_matrix((ones_ntris, (b, c)), shape=(npoints, npoints))
    edges = edges + coo_matrix((ones_ntris, (c, a)), shape=(npoints, npoints))
    edges = edges.tocsr()
    edges = edges + edges.T
    return edges


def mesh_dist(tris, vert):
    """Compute adjacency matrix weighted by distances

    It generates an adjacency matrix where the entries are the distances
    between neighboring vertices.

    Parameters
    ----------
    tris : array (n_tris x 3)
        Mesh triangulation
    vert : array (n_vert x 3)
        Vertex locations

    Returns
    -------
    dist_matrix : scipy.sparse.csr_matrix
        Sparse matrix with distances between adjacent vertices
    """
    edges = mesh_edges(tris).tocoo()

    # Euclidean distances between neighboring vertices
    dist = np.sqrt(np.sum((vert[edges.row, :] - vert[edges.col, :]) ** 2,
                          axis=1))

    dist_matrix = csr_matrix((dist, (edges.row, edges.col)), shape=edges.shape)

    return dist_matrix


@verbose
def _morph_buffer(data, idx_use, e, smooth, n_vertices, nearest, maps,
                  verbose=None):
    """Morph data from one subject's source space to another

    Parameters
    ----------
    data : array, or csr sparse matrix
        A n_vertices x n_times (or other dimension) dataset to morph
    idx_use : array of int
        Vertices from the original subject's data
    e : sparse matrix
        The mesh edges of the "from" subject
    smooth : int
        Number of smoothing iterations to perform. A hard limit of 100 is
        also imposed.
    n_vertices : int
        Number of vertices
    nearest : array of int
        Vertices on the destination surface to use.
    maps : sparse matrix
        Morph map from one subject to the other.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    data_morphed : array, or csr sparse matrix
        The morphed data (same type as input)
    """

    n_iter = 100  # max nb of smoothing iterations
    if sparse.issparse(data):
        use_sparse = True
        if not isinstance(data, sparse.csr_matrix):
            data = data.tocsr()
    else:
        use_sparse = False
    for k in range(n_iter):
        e_use = e[:, idx_use]
        data1 = e_use * np.ones(len(idx_use))
        data = e_use * data
        idx_use = np.where(data1)[0]
        if smooth is None:
            if ((k == (n_iter - 1)) or (len(idx_use) >= n_vertices)):
                # stop when source space in filled with non-zeros
                break
        elif k == (smooth - 1):
            break
        if use_sparse:
            data = data[idx_use, :]
            data.data /= data1[idx_use].repeat(np.diff(data.indptr))
        else:
            data = data[idx_use, :] / data1[idx_use][:, None]

    if use_sparse:
        data1[data1 == 0] = 1
        data.data /= data1.repeat(np.diff(data.indptr))
    else:
        data[idx_use, :] /= data1[idx_use][:, None]

    logger.info('    %d smooth iterations done.' % (k + 1))
    data_morphed = maps[nearest, :] * data
    return data_morphed


def _compute_nearest(xhs, rr):
    nearest = np.zeros(len(rr), dtype=np.int)
    dr = 32
    for k in range(0, len(rr), dr):
        dots = np.dot(rr[k:k + dr], xhs.T)
        nearest[k:k + dr] = np.argmax(dots, axis=1)
    return nearest


def _get_subject_sphere_tris(subject, subjects_dir):
    spheres = [os.path.join(subjects_dir, subject, 'surf',
                            xh + '.sphere.reg') for xh in ['lh', 'rh']]
    tris = [read_surface(s)[1] for s in spheres]
    return tris


@verbose
def morph_data(subject_from, subject_to, stc_from, grade=5, smooth=None,
               subjects_dir=None, buffer_size=64, n_jobs=1, verbose=None):
    """Morph a source estimate from one subject to another

    Parameters
    ----------
    subject_from : string
        Name of the original subject as named in the SUBJECTS_DIR
    subject_to : string
        Name of the subject on which to morph as named in the SUBJECTS_DIR
    stc_from : SourceEstimate
        Source estimates for subject "from" to morph
    grade : int, list (of two arrays), or None
        Resolution of the icosahedral mesh (typically 5). If None, all
        vertices will be used (potentially filling the surface). If a list,
        then values will be morphed to the set of vertices specified in
        in grade[0] and grade[1]. Note that specifying the vertices (e.g.,
        grade=[np.arange(10242), np.arange(10242)] for fsaverage on a
        standard grade 5 source space) can be substantially faster than
        computing vertex locations.
    smooth : int or None
        Number of iterations for the smoothing of the surface data.
        If None, smooth is automatically defined to fill the surface
        with non-zero values.
    subjects_dir : string, or None
        Path to SUBJECTS_DIR if it is not set in the environment.
    buffer_size : int
        Morph data in chunks of `buffer_size` time instants.
        Saves memory when morphing long time intervals.
    n_jobs : int
        Number of jobs to run in parallel
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    stc_to : SourceEstimate
        Source estimate for the destination subject.
    """

    if not stc_from.is_surface():
        raise ValueError('Morphing is only possible with surface source '
                         'estimates')

    logger.info('Morphing data...')
    subjects_dir = get_subjects_dir(subjects_dir)
    nearest = grade_to_vertices(subject_to, grade, subjects_dir, n_jobs)
    tris = _get_subject_sphere_tris(subject_from, subjects_dir)
    maps = read_morph_map(subject_from, subject_to, subjects_dir)

    # morph the data
    data = [stc_from.lh_data, stc_from.rh_data]
    data_morphed = [None, None]

    n_chunks = ceil(stc_from.data.shape[1] / float(buffer_size))

    parallel, my_morph_buffer, _ = parallel_func(_morph_buffer, n_jobs)

    for hemi in [0, 1]:
        e = mesh_edges(tris[hemi])
        e.data[e.data == 2] = 1
        n_vertices = e.shape[0]
        e = e + sparse.eye(n_vertices, n_vertices)
        idx_use = stc_from.vertno[hemi]
        if len(idx_use) == 0:
            continue
        data_morphed[hemi] = np.concatenate(
            parallel(my_morph_buffer(data_buffer, idx_use, e, smooth,
                                     n_vertices, nearest[hemi], maps[hemi])
                     for data_buffer
                     in np.array_split(data[hemi], n_chunks, axis=1)), axis=1)

    vertices = [nearest[0], nearest[1]]
    if data_morphed[0] is None:
        if data_morphed[1] is None:
            data = np.r_[[], []]
        else:
            data = data_morphed[1]
    elif data_morphed[1] is None:
        data = data_morphed[0]
    else:
        data = np.r_[data_morphed[0], data_morphed[1]]

    stc_to = SourceEstimate(data, vertices, stc_from.tmin, stc_from.tstep,
                            verbose=stc_from.verbose)
    logger.info('[done]')

    return stc_to


@verbose
def compute_morph_matrix(subject_from, subject_to, vertices_from, vertices_to,
                         smooth=None, subjects_dir=None, verbose=None):
    """Get a matrix that morphs data from one subject to another

    Parameters
    ----------
    subject_from : string
        Name of the original subject as named in the SUBJECTS_DIR
    subject_to : string
        Name of the subject on which to morph as named in the SUBJECTS_DIR
    vertices_from : list of arrays of int
        Vertices for each hemisphere (LH, RH) for subject_from
    vertices_to : list of arrays of int
        Vertices for each hemisphere (LH, RH) for subject_to
    smooth : int or None
        Number of iterations for the smoothing of the surface data.
        If None, smooth is automatically defined to fill the surface
        with non-zero values.
    subjects_dir : string
        Path to SUBJECTS_DIR is not set in the environment
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    morph_matrix : sparse matrix
        matrix that morphs data from subject_from to subject_to
    """
    logger.info('Computing morph matrix...')
    subjects_dir = get_subjects_dir(subjects_dir)
    tris = _get_subject_sphere_tris(subject_from, subjects_dir)
    maps = read_morph_map(subject_from, subject_to, subjects_dir)

    morpher = [None] * 2
    for hemi in [0, 1]:
        e = mesh_edges(tris[hemi])
        e.data[e.data == 2] = 1
        n_vertices = e.shape[0]
        e = e + sparse.eye(n_vertices, n_vertices)
        idx_use = vertices_from[hemi]
        if len(idx_use) == 0:
            morpher[hemi] = []
            continue
        m = sparse.eye(len(idx_use), len(idx_use), format='csr')
        morpher[hemi] = _morph_buffer(m, idx_use, e, smooth, n_vertices,
                                      vertices_to[hemi], maps[hemi])
    # be careful about zero-length arrays
    if isinstance(morpher[0], list):
        morpher = morpher[1]
    elif isinstance(morpher[1], list):
        morpher = morpher[0]
    else:
        morpher = sparse_block_diag(morpher, format='csr')
    logger.info('[done]')
    return morpher


@verbose
def grade_to_vertices(subject, grade, subjects_dir=None, n_jobs=1,
                      verbose=None):
    """Convert a grade to source space vertices for a given subject

    Parameters
    ----------
    subject : str
        Name of the subject
    grade : int
        Resolution of the icosahedral mesh (typically 5). If None, all
        vertices will be used (potentially filling the surface). If a list,
        then values will be morphed to the set of vertices specified in
        in grade[0] and grade[1]. Note that specifying the vertices (e.g.,
        grade=[np.arange(10242), np.arange(10242)] for fsaverage on a
        standard grade 5 source space) can be substantially faster than
        computing vertex locations.
    subjects_dir : string, or None
        Path to SUBJECTS_DIR if it is not set in the environment
    n_jobs : int
        Number of jobs to run in parallel
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    vertices : list of arrays of int
        Vertex numbers for LH and RH
    """
    subjects_dir = get_subjects_dir(subjects_dir)

    spheres_to = [os.path.join(subjects_dir, subject, 'surf',
                               xh + '.sphere.reg') for xh in ['lh', 'rh']]
    lhs, rhs = [read_surface(s)[0] for s in spheres_to]

    if grade is not None:  # fill a subset of vertices
        if isinstance(grade, list):
            if not len(grade) == 2:
                raise ValueError('grade as a list must have two elements '
                                 '(arrays of output vertices)')
            vertices = grade
        else:
            # find which vertices to use in "to mesh"
            ico = _get_ico_tris(grade, return_surf=True)
            lhs /= np.sqrt(np.sum(lhs ** 2, axis=1))[:, None]
            rhs /= np.sqrt(np.sum(rhs ** 2, axis=1))[:, None]

            # Compute nearest vertices in high dim mesh
            parallel, my_compute_nearest, _ = \
                parallel_func(_compute_nearest, n_jobs)
            lhs, rhs, rr = [a.astype(np.float32)
                            for a in [lhs, rhs, ico['rr']]]
            vertices = parallel(my_compute_nearest(xhs, rr)
                                for xhs in [lhs, rhs])
    else:  # potentially fill the surface
        vertices = [np.arange(lhs.shape[0]), np.arange(rhs.shape[0])]

    return vertices


def morph_data_precomputed(subject_from, subject_to, stc_from, vertices_to,
                           morph_mat):
    """Morph a source estimate from one subject to another using a
    morph matrix precomputed with compute_morph_matrix

    Parameters
    ----------
    subject_from : string
        Name of the original subject as named in the SUBJECTS_DIR
    subject_to : string
        Name of the subject on which to morph as named in the SUBJECTS_DIR
    stc_from : SourceEstimate
        Source estimates for subject "from" to morph
    vertices_to : list of array of int
        The vertices on the destination subject's brain
    morph_mat : sparse matrix
        The morphing matrix

    Returns
    -------
    stc_to : SourceEstimate
        Source estimate for the destination subject.
    """

    if not sparse.issparse(morph_mat):
        raise ValueError('morph_mat must be a sparse matrix')

    if not isinstance(vertices_to, list) or not len(vertices_to) == 2:
        raise ValueError('vertices_to must be a list of length 2')

    if not sum(len(v) for v in vertices_to) == morph_mat.shape[0]:
        raise ValueError('number of vertices in vertices_to must match '
                         'morph_mat.shape[0]')
    if not stc_from.data.shape[0] == morph_mat.shape[1]:
        raise ValueError('stc_from.data.shape[0] must be the same as '
                         'morph_mat.shape[0]')

    data = morph_mat * stc_from.data
    stc_to = SourceEstimate(data, vertices_to, stc_from.tmin, stc_from.tstep,
                            verbose=stc_from.verbose)
    return stc_to


@verbose
def spatio_temporal_src_connectivity(src, n_times, dist=None, verbose=None):
    """Compute connectivity for a source space activation over time

    Parameters
    ----------
    src : source space
        The source space.
    n_times : int
        Number of time instants.
    dist : float, or None
        Maximal geodesic distance (in m) between vertices in the
        source space to consider neighbors. If None, immediate neighbors
        are extracted from an ico surface.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    connectivity : sparse COO matrix
        The connectivity matrix describing the spatio-temporal
        graph structure. If N is the number of vertices in the
        source space, the N first nodes in the graph are the
        vertices are time 1, the nodes from 2 to 2N are the vertices
        during time 2, etc.
    """
    if dist is None:
        if src[0]['use_tris'] is None:
            raise Exception("The source space does not appear to be an ico "
                            "surface. Connectivity cannot be extracted from "
                            "non-ico source spaces.")
        used_verts = [np.unique(s['use_tris']) for s in src]
        lh_tris = np.searchsorted(used_verts[0], src[0]['use_tris'])
        rh_tris = np.searchsorted(used_verts[1], src[1]['use_tris'])
        tris = np.concatenate((lh_tris, rh_tris + np.max(lh_tris) + 1))
        connectivity = spatio_temporal_tris_connectivity(tris, n_times)

        # deal with source space only using a subset of vertices
        masks = [in1d(u, s['vertno']) for s, u in zip(src, used_verts)]
        if sum(u.size for u in used_verts) != connectivity.shape[0] / n_times:
            raise ValueError('Used vertices do not match connectivity shape')
        if [np.sum(m) for m in masks] != [len(s['vertno']) for s in src]:
            raise ValueError('Vertex mask does not match number of vertices')
        masks = np.concatenate(masks)
        missing = 100 * float(len(masks) - np.sum(masks)) / len(masks)
        if missing:
            warnings.warn('%0.1f%% of original source space vertices have been'
                          ' omitted, tri-based connectivity will have holes.\n'
                          'Consider using distance-based connectivity or '
                          'morphing data to all source space vertices.'
                          % missing)
            masks = np.tile(masks, n_times)
            masks = np.where(masks)[0]
            connectivity = connectivity.tocsr()
            connectivity = connectivity[masks]
            connectivity = connectivity[:, masks]
            # return to original format
            connectivity = connectivity.tocoo()

        return connectivity
    else:  # use distances computed and saved in the source space file
        return spatio_temporal_dist_connectivity(src, n_times, dist)


@verbose
def grade_to_tris(grade, verbose=None):
    """Get tris defined for a certain grade

    Parameters
    ----------
    grade : int
        Grade of an icosahedral mesh.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    tris : list
        2-element list containing Nx3 arrays of tris, suitable for use in
        spatio_temporal_tris_connectivity.
    """
    a = _get_ico_tris(grade, None, False)
    tris = np.concatenate((a, a + (np.max(a) + 1)))
    return tris


@verbose
def spatio_temporal_tris_connectivity(tris, n_times, verbose=None):
    """Compute connectivity from triangles and time instants

    Parameters
    ----------
    tris : array
        N x 3 array defining triangles.
    n_times : int
        Number of time points
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    connectivity : sparse COO matrix
        The connectivity matrix describing the spatio-temporal
        graph structure. If N is the number of vertices in the
        source space, the N first nodes in the graph are the
        vertices are time 1, the nodes from 2 to 2N are the vertices
        during time 2, etc.
    """
    edges = mesh_edges(tris).tocoo()
    return _get_connectivity_from_edges(edges, n_times)


@verbose
def spatio_temporal_dist_connectivity(src, n_times, dist, verbose=None):
    """Compute connectivity from distances in a source space and time instants

    Parameters
    ----------
    src : source space
        The source space must have distances between vertices computed, such
        that src['dist'] exists and is useful. This can be obtained using MNE
        with a call to mne_add_patch_info with the --dist option.
    n_times : int
        Number of time points
    dist : float
        Maximal geodesic distance (in m) between vertices in the
        source space to consider neighbors.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    connectivity : sparse COO matrix
        The connectivity matrix describing the spatio-temporal
        graph structure. If N is the number of vertices in the
        source space, the N first nodes in the graph are the
        vertices are time 1, the nodes from 2 to 2N are the vertices
        during time 2, etc.
    """
    if src[0]['dist'] is None:
        raise RuntimeError('src must have distances included, consider using\n'
                           'mne_add_patch_info with --dist argument')
    edges = sparse_block_diag([s['dist'][s['vertno'], :][:, s['vertno']]
                              for s in src])
    edges.data[:] = np.less_equal(edges.data, dist)
    # clean it up and put it in coo format
    edges = edges.tocsr()
    edges.eliminate_zeros()
    edges = edges.tocoo()
    return _get_connectivity_from_edges(edges, n_times)


@verbose
def spatial_src_connectivity(src, dist=None, verbose=None):
    """Compute connectivity for a source space activation

    Parameters
    ----------
    src : source space
        The source space.
    dist : float, or None
        Maximal geodesic distance (in m) between vertices in the
        source space to consider neighbors. If None, immediate neighbors
        are extracted from an ico surface.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    connectivity : sparse COO matrix
        The connectivity matrix describing the spatial graph structure.
    """
    return spatio_temporal_src_connectivity(src, 1, dist)


@verbose
def spatial_tris_connectivity(tris, verbose=None):
    """Compute connectivity from triangles

    Parameters
    ----------
    tris : array
        N x 3 array defining triangles.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    connectivity : sparse COO matrix
        The connectivity matrix describing the spatial graph structure.
    """
    return spatio_temporal_tris_connectivity(tris, 1)


def spatial_dist_connectivity(src, dist, verbose=None):
    """Compute connectivity from distances in a source space

    Parameters
    ----------
    src : source space
        The source space must have distances between vertices computed, such
        that src['dist'] exists and is useful. This can be obtained using MNE
        with a call to mne_add_patch_info with the --dist option.
    dist : float
        Maximal geodesic distance (in m) between vertices in the
        source space to consider neighbors.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    connectivity : sparse COO matrix
        The connectivity matrix describing the spatial graph structure.
    """
    return spatio_temporal_dist_connectivity(src, 1, dist)


def sparse_block_diag(mats, format=None, dtype=None):
    """An implementation of scipy.sparse.block_diag since old versions of
    scipy don't have it. Forms a sparse matrix by stacking matrices in block
    diagonal form.

    Parameters
    ----------
    mats : list of matrices
        Input matrices.
    format : str, optional
        The sparse format of the result (e.g. "csr"). If not given, the
        matrix is returned in "coo" format.
    dtype : dtype specifier, optional
        The data-type of the output matrix. If not given, the dtype is
        determined from that of blocks.

    Returns
    -------
    res : sparse matrix
    """
    try:
        return sparse.block_diag(mats, format=format, dtype=dtype)
    except AttributeError:
        nmat = len(mats)
        rows = []
        for ia, a in enumerate(mats):
            row = [None] * nmat
            row[ia] = a
            rows.append(row)
        return sparse.bmat(rows, format=format, dtype=dtype)


@verbose
def _get_connectivity_from_edges(edges, n_times, verbose=None):
    """Given edges sparse matrix, create connectivity matrix"""
    n_vertices = edges.shape[0]
    logger.info("-- number of connected vertices : %d" % n_vertices)
    nnz = edges.col.size
    aux = n_vertices * np.arange(n_times)[:, None] * np.ones((1, nnz), np.int)
    col = (edges.col[None, :] + aux).ravel()
    row = (edges.row[None, :] + aux).ravel()
    if n_times > 1:  # add temporal edges
        o = (n_vertices * np.arange(n_times - 1)[:, None]
             + np.arange(n_vertices)[None, :]).ravel()
        d = (n_vertices * np.arange(1, n_times)[:, None]
             + np.arange(n_vertices)[None, :]).ravel()
        row = np.concatenate((row, o, d))
        col = np.concatenate((col, d, o))
    data = np.ones(edges.data.size * n_times + 2 * n_vertices * (n_times - 1),
                   dtype=np.int)
    connectivity = coo_matrix((data, (row, col)),
                              shape=(n_times * n_vertices, ) * 2)
    return connectivity


@verbose
def _get_ico_tris(grade, verbose=None, return_surf=False):
    """Get triangles for ico surface."""
    ico_file_name = os.path.join(os.path.dirname(__file__), 'data',
                                 'icos.fif.gz')
    surfaces = read_bem_surfaces(ico_file_name)
    for s in surfaces:
        if s['id'] == (9000 + grade):
            ico = s
            break

    if not return_surf:
        return ico['tris']
    else:
        return ico


def save_stc_as_volume(fname, stc, src, dest='mri', mri_resolution=False):
    """Save a volume source estimate in a nifti file

    Parameters
    ----------
    fname : string
        The name of the generated nifti file.
    stc : instance of SourceEstimate
        The source estimate
    src : list
        The list of source spaces (should actually be of length 1)
    dest : 'mri' | 'surf'
        If 'mri' the volume is defined in the coordinate system of
        the original T1 image. If 'surf' the coordinate system
        of the FreeSurfer surface is used (Surface RAS).
    mri_resolution: bool
        It True the image is saved in MRI resolution.
        WARNING: if you have many time points the file produced can be
        huge.

    Returns
    -------
    img : instance Nifti1Image
        The image object.
    """
    if stc.is_surface():
        raise Exception('Only volume source estimates can be saved as '
                        'volumes')

    n_times = stc.data.shape[1]
    shape = src[0]['shape']
    shape3d = (shape[2], shape[1], shape[0])
    shape = (n_times, shape[2], shape[1], shape[0])
    vol = np.zeros(shape)
    mask3d = src[0]['inuse'].reshape(shape3d).astype(np.bool)

    if mri_resolution:
        mri_shape3d = (src[0]['mri_height'], src[0]['mri_depth'],
                       src[0]['mri_width'])
        mri_shape = (n_times, src[0]['mri_height'], src[0]['mri_depth'],
                     src[0]['mri_width'])
        mri_vol = np.zeros(mri_shape)
        interpolator = src[0]['interpolator']

    for k, v in enumerate(vol):
        v[mask3d] = stc.data[:, k]
        if mri_resolution:
            mri_vol[k] = (interpolator * v.ravel()).reshape(mri_shape3d)

    if mri_resolution:
        vol = mri_vol

    vol = vol.T

    if mri_resolution:
        affine = src[0]['vox_mri_t']['trans'].copy()
    else:
        affine = src[0]['src_mri_t']['trans'].copy()
    if dest == 'mri':
        affine = np.dot(src[0]['mri_ras_t']['trans'], affine)
    affine[:3] *= 1e3

    try:
        import nibabel as nib  # lazy import to avoid dependency
    except ImportError:
        raise ImportError("nibabel is required to save volume images.")

    header = nib.nifti1.Nifti1Header()
    header.set_xyzt_units('mm', 'msec')
    header['pixdim'][4] = 1e3 * stc.tstep
    img = nib.Nifti1Image(vol, affine, header=header)
    nib.save(img, fname)
    return img


def _get_label_flip(labels, label_vertidx, src):
    """Helper function to get sign-flip for labels"""
    # do the import here to avoid circular dependency
    from .label import label_sign_flip
    # get the sign-flip vector for every label
    label_flip = list()
    for label, vertidx in zip(labels, label_vertidx):
        if label.hemi == 'both':
            raise ValueError('BiHemiLabel not supported when using sign-flip')
        if vertidx is not None:
            flip = label_sign_flip(label, src)[:, None]
        else:
            flip = None
        label_flip.append(flip)

    return label_flip


@verbose
def _gen_extract_label_time_course(stcs, labels, src, mode='mean',
                                   allow_empty=False, verbose=None):
    """Generator for extract_label_time_course"""

    n_labels = len(labels)

    # get vertno from source space, they have to be the same as in the stcs
    vertno = [s['vertno'] for s in src]
    nvert = [len(vn) for vn in vertno]

    # do the initialization
    label_vertidx = list()
    for label in labels:
        if label.hemi == 'both':
            # handle BiHemiLabel
            sub_labels = [label.lh, label.rh]
        else:
            sub_labels = [label]
        this_vertidx = list()
        for slabel in sub_labels:
            if slabel.hemi == 'lh':
                this_vertno = np.intersect1d(vertno[0], slabel.vertices)
                vertidx = np.searchsorted(vertno[0], this_vertno)
            elif slabel.hemi == 'rh':
                this_vertno = np.intersect1d(vertno[1], slabel.vertices)
                vertidx = nvert[0] + np.searchsorted(vertno[1], this_vertno)
            else:
                raise ValueError('label %s has invalid hemi' % label.name)
            this_vertidx.append(vertidx)

        # convert it to an array
        this_vertidx = np.concatenate(this_vertidx)
        if len(this_vertidx) == 0:
            msg = ('source space does not contain any vertices for label %s'
                   % label.name)
            if not allow_empty:
                raise ValueError(msg)
            else:
                logger.warn(msg + '. Assigning all-zero time series to label.')
            this_vertidx = None  # to later check if label is empty

        label_vertidx.append(this_vertidx)

    # mode-dependent initalization
    if mode == 'mean':
        pass  # we have this here to catch invalid values for mode
    elif mode == 'mean_flip':
       # get the sign-flip vector for every label
        label_flip = _get_label_flip(labels, label_vertidx, src)
    elif mode == 'pca_flip':
       # get the sign-flip vector for every label
        label_flip = _get_label_flip(labels, label_vertidx, src)
    else:
        raise ValueError('%s is an invalid mode' % mode)

    # loop through source estimates and extract time series
    for stc in stcs:

        # make sure the stc is compatible with the source space
        if len(stc.vertno[0]) != nvert[0] or len(stc.vertno[1]) != nvert[1]:
            raise ValueError('stc not compatible with source space')
        if any([np.any(svn != vn) for svn, vn in zip(stc.vertno, vertno)]):
            raise ValueError('stc not compatible with source space')

        logger.info('Extracting time courses for %d labels (mode: %s)'
                    % (n_labels, mode))

        # do the extraction
        label_tc = np.zeros((n_labels, stc.data.shape[1]),
                            dtype=stc.data.dtype)
        if mode == 'mean':
            for i, vertidx in enumerate(label_vertidx):
                if vertidx is not None:
                    label_tc[i] = np.mean(stc.data[vertidx, :], axis=0)
        elif mode == 'mean_flip':
            for i, (vertidx, flip) in enumerate(zip(label_vertidx,
                                                    label_flip)):
                if vertidx is not None:
                    label_tc[i] = np.mean(flip * stc.data[vertidx, :], axis=0)
        elif mode == 'pca_flip':
            for i, (vertidx, flip) in enumerate(zip(label_vertidx,
                                                    label_flip)):
                if vertidx is not None:
                    U, s, V = linalg.svd(stc.data[vertidx, :],
                                         full_matrices=False)
                    # determine sign-flip
                    sign = np.sign(np.dot(U[:, 0], flip))

                    # use average power in label for scaling
                    scale = linalg.norm(s) / np.sqrt(len(vertidx))

                    label_tc[i] = sign * scale * V[0]
        else:
            raise ValueError('%s is an invalid mode' % mode)

        # this is a generator!
        yield label_tc


@verbose
def extract_label_time_course(stcs, labels, src, mode='mean_flip',
                              allow_empty=False, return_generator=False,
                              verbose=None):
    """Extract label time course for lists of labels and source estimates

    This function will extract one time course for each label and source
    estimate. The way the time courses are extracted depends on the mode
    parameter.

    Valid values for mode are:
    'mean': Average within each label.
    'mean_flip': Average within each label with sign flip depending on source
    orientation.
    'pca_flip': Apply an SVD to the time courses within each label and use the
    scaled and sign-flipped first right-singular vector as the label time
    course. The scaling is performed such that the power of the label time
    course is the same as the average per-vertex time course power within
    the label. The sign of the resulting time course is adjusted by multiplying
    it with "sign(dot(u, flip))" where u is the first left-singular vector,
    and flip is a sing-flip vector based on the vertex normals. This procedure
    assures that the phase does not randomly change by 180 degrees from one
    stc to the next.

    Parameters
    ----------
    stcs : SourceEstimate | list (or generator) of SourceEstimate
        The source estimates from which to extract the time course.
    labels : Label | list of Label
        The labels for which to extract the time course.
    src : list
        Source spaces for left and right hemisphere.
    mode : str
        Extraction mode, see explanation above.
    allow_empty : bool
        Instead of emitting an error, return all-zero time courses for labels
        that do not have any vertices in the source estimate.
    return_generator : bool
        If True, a generator instead of a list is returned.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    label_tc : array | list (or generator) of array,
               shape=(len(labels), n_times)
        Extracted time course for each label and source estimate.
    """
    # convert inputs to lists
    if isinstance(stcs, SourceEstimate):
        stcs = [stcs]
        return_several = False
        return_generator = False
    else:
        return_several = True

    if not isinstance(labels, list):
        labels = [labels]

    label_tc = _gen_extract_label_time_course(stcs, labels, src, mode=mode,
                                              allow_empty=allow_empty)

    if not return_generator:
        # do the extraction and return a list
        label_tc = list(label_tc)

    if not return_several:
        # input was a single SoureEstimate, return single array
        label_tc = label_tc[0]

    return label_tc
