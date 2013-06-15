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
from .utils import get_subjects_dir, _check_subject, \
                   _check_pandas_index_arguments, _check_pandas_installed, \
                   deprecated
from .viz import plot_source_estimates
from . import verbose
from . fixes import in1d


@deprecated('read_stc is deprecated and will be removed with version 0.7. '
            'Please use read_source_estimate instead.')
def read_stc(filename):
    """Read an STC file and return as dict

    STC files contain activations or source reconstructions.

    Parameters
    ----------
    filename : string
        The name of the STC file.

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
    return _read_stc(filename)


def _read_stc(filename):
    """ Aux Function
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

    if (vertices_n and  # vertices_n can be 0 (empty stc)
            ((file_length / 4 - 4 - vertices_n) % (data_n * vertices_n)) != 0):
        raise ValueError('incorrect stc file size')

    # read the data matrix
    stc['data'] = np.fromfile(fid, dtype=">f4", count=vertices_n * data_n)
    stc['data'] = stc['data'].reshape([data_n, vertices_n]).T

    # close the file
    fid.close()
    return stc


@deprecated('write_stc is deprecated and will be removed with version 0.7. '
            'Please use SourceEstimate.save instead.')
def write_stc(filename, tmin, tstep, vertices, data):
    """Write an STC file

    Parameters
    ----------
    filename : string
        The name of the STC file.
    tmin : float
        The first time point of the data in seconds.
    tstep : float
        Time between frames in seconds.
    vertices : array of integers
        Vertex indices (0 based).
    data : 2D array
        The data matrix (nvert * ntime).

    See Also
    --------
    SourceEstimate.save (instance method)
    """
    return _write_stc(filename, tmin, tstep, vertices, data)


def _write_stc(filename, tmin, tstep, vertices, data):
    """Write an STC file

    Parameters
    ----------
    filename : string
        The name of the STC file.
    tmin : float
        The first time point of the data in seconds.
    tstep : float
        Time between frames in seconds.
    vertices : array of integers
        Vertex indices (0 based).
    data : 2D array
        The data matrix (nvert * ntime).
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


@deprecated('read_w is deprecated and will be removed with version 0.7. '
            'Please use read_source_estimate instead.')
def read_w(filename):
    """Read a w file and return as dict

    w files contain activations or source reconstructions for a single time
    point.

    Parameters
    ----------
    filename : string
        The name of the w file.

    Returns
    -------
    data: dict
        The w structure. It has the following keys:
           vertices       vertex indices (0 based)
           data           The data matrix (nvert long)
    """
    return _read_w(filename)


def _read_w(filename):
    """Read a w file and return as dict

    w files contain activations or source reconstructions for a single time
    point.

    Parameters
    ----------
    filename : string
        The name of the w file.

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


@deprecated('read_w is deprecated and will be removed with version 0.7. '
            'Please use SoureEstimate.save instead.')
def write_w(filename, vertices, data):
    """Read a w file

    w files contain activations or source reconstructions for a single time
    point.

    Parameters
    ----------
    filename: string
        The name of the w file.
    vertices: array of int
        Vertex indices (0 based).
    data: 1D array
        The data array (nvert).

    See Also
    --------
    SourceEstimate.save (instance method)
    """
    return _write_w(filename, vertices, data)


def _write_w(filename, vertices, data):
    """Read a w file

    w files contain activations or source reconstructions for a single time
    point.

    Parameters
    ----------
    filename: string
        The name of the w file.
    vertices: array of int
        Vertex indices (0 based).
    data: 1D array
        The data array (nvert).
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


def read_source_estimate(fname, subject=None):
    """Returns a SourceEstimate object.

    Parameters
    ----------
    fname : str
        Path to (a) source-estimate file(s).
    subject : str | None
        Name of the subject the source estimate(s) is (are) from.
        It is good practice to set this attribute to avoid combining
        incompatible labels and SourceEstimates (e.g., ones from other
        subjects). Note that due to file specification limitations, the
        subject name isn't saved to or loaded from files written to disk.

    Notes
    -----
     - for volume source estimates, ``fname`` should provide the path to a
       single file named '*-vl.stc` or '*-vol.stc'
     - for surface source estimates, ``fname`` should either provide the
       path to the file corresponding to a single hemisphere ('*-lh.stc',
       '*-rh.stc') or only specify the asterisk part in these patterns. In any
       case, the function expects files for both hemisphere with names
       following this pattern.
     - for single time point .w files, ``fname`` should follow the same
       pattern as for surface estimates, except that files are named
       '*-lh.w' and '*-rh.w'.
    """
    fname_arg = fname

    # make sure corresponding file(s) can be found
    ftype = None
    if os.path.exists(fname):
        if fname.endswith('-vl.stc') or fname.endswith('-vol.stc') or \
                fname.endswith('-vl.w') or fname.endswith('-vol.w'):
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
        if fname.endswith('.stc'):
            kwargs = _read_stc(fname)
        elif fname.endswith('.w'):
            kwargs = _read_w(fname)
            kwargs['data'] = kwargs['data'][:, np.newaxis]
            kwargs['tmin'] = 0.0
            kwargs['tstep'] = 0.0
        else:
            raise IOError('Volume source estimate must end with .stc or .w')
    elif ftype == 'surface':  # stc file with surface source spaces
        lh = _read_stc(fname + '-lh.stc')
        rh = _read_stc(fname + '-rh.stc')
        assert lh['tmin'] == rh['tmin']
        assert lh['tstep'] == rh['tstep']
        kwargs = lh.copy()
        kwargs['data'] = np.r_[lh['data'], rh['data']]
        kwargs['vertices'] = [lh['vertices'], rh['vertices']]
    elif ftype == 'w':  # w file with surface source spaces
        lh = _read_w(fname + '-lh.w')
        rh = _read_w(fname + '-rh.w')
        kwargs = lh.copy()
        kwargs['data'] = np.atleast_2d(np.r_[lh['data'], rh['data']]).T
        kwargs['vertices'] = [lh['vertices'], rh['vertices']]
        # w files only have a single time point
        kwargs['tmin'] = 0.0
        kwargs['tstep'] = 1.0

    kwargs['subject'] = subject
    return SourceEstimate(**kwargs)


class _NotifyArray(np.ndarray):
    """Array class that executes a callback when it is modified
    """
    def __new__(cls, input_array, modify_callback=None):
        obj = np.asarray(input_array).view(cls)
        obj.modify_callback = modify_callback
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            # an empty constructor was used
            return

        # try to copy the callback
        self.modify_callback = getattr(obj, 'modify_callback', None)

    def _modified_(self):
        """Execute the callback if it is set"""
        if self.modify_callback is not None:
            self.modify_callback()

    def __getattribute__(self, name):
        # catch ndarray methods that modify the array inplace
        if name in ['fill', 'itemset', 'resize', 'sort']:
            self._modified_()

        return object.__getattribute__(self, name)

    def __setitem__(self, item, value):
        self._modified_()
        np.ndarray.__setitem__(self, item, value)

    def __array_wrap__(self, out_arr, context=None):
        # this method is called whenever a numpy ufunc (+, +=..) is called
        # the last entry in context is the array that receives the result
        if (context is not None and len(context[1]) == 3
                and context[1][2] is self):
            self._modified_()

        return np.ndarray.__array_wrap__(self, out_arr, context)


def _verify_source_estimate_compat(a, b):
    """Make sure two SourceEstimates are compatible for arith. operations"""
    compat = False
    if len(a.vertno) == len(b.vertno):
        if all([np.array_equal(av, vv) for av, vv in zip(a.vertno, b.vertno)]):
            compat = True
    if not compat:
        raise ValueError('Cannot combine SourceEstimates that do not have the '
                         'same vertices. Consider using stc.expand().')
    if a.subject != b.subject:
        raise ValueError('source estimates do not have the same subject '
                         'names, "%s" and "%s"' % (a.name, b.name))


class SourceEstimate(object):
    """SourceEstimate container

    Can be saved and loaded from .stc or .w files.

    Parameters
    ----------
    data : array of shape (n_dipoles, n_times) | 2-tuple (kernel, sens_data)
        The data in source space. The data can either be a single array or
        a tuple with two arrays: "kernel" shape (n_vertices, n_sensors) and
        "sens_data" shape (n_sensors, n_times). In this case, the source
        space data corresponds to "numpy.dot(kernel, sens_data)".
    vertices : array | list of two arrays
        Vertex numbers corresponding to the data.
    tmin : scalar
        Time point of the first sample in data.
    tstep : scalar
        Time step between successive samples in data.
    subject : str | None
        The subject name. While not necessary, it is safer to set the
        subject parameter to avoid analysis errors.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Attributes
    ----------
    subject : str | None
        The subject name.
    times : array of shape (n_times,)
        The time vector.
    vertno : array or list of array of shape (n_dipoles,)
        The indices of the dipoles in the different source spaces. Can
        be an array if there is only one source space (e.g., for volumes).
    data : array of shape (n_dipoles, n_times)
        The data in source space.
    shape : tuple
        The shape of the data. A tuple of int (n_dipoles, n_times).
    """
    @verbose
    def __init__(self, data, vertices=None, tmin=None, tstep=None,
                 subject=None, verbose=None):
        kernel, sens_data = None, None
        if isinstance(data, tuple):
            if len(data) != 2:
                raise ValueError('If data is a tuple it has to be length 2')
            kernel, sens_data = data
            data = None
            if kernel.shape[1] != sens_data.shape[0]:
                raise ValueError('kernel and sens_data have invalid '
                                 'dimensions')

        if isinstance(vertices, list):
            if not (len(vertices) == 2 or len(vertices) == 1) or \
                    not all([isinstance(v, np.ndarray) for v in vertices]):
                raise ValueError('Vertices, if a list, must contain one or '
                                 'two numpy arrays')
            n_src = sum([len(v) for v in vertices])
        elif not isinstance(vertices, np.ndarray):
            raise ValueError('Vertices must be a list or numpy array')
        else:
            n_src = len(vertices)
        # safeguard the user against doing something silly
        if data is not None and data.shape[0] != n_src:
            raise ValueError('Number of vertices (%i) and stc.shape[0] (%i) '
                             'must match' % (n_src, data.shape[0]))

        self._data = data
        self.tmin = tmin
        self.tstep = tstep
        self.vertno = vertices
        self.verbose = verbose
        self._kernel = kernel
        self._sens_data = sens_data
        self.times = None
        self._update_times()
        self.subject = _check_subject(None, subject, False)

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
            extended with "-vl.stc" or "-vl.w".
        ftype : string
            File format to use. Allowed values are "stc" (default) and "w".
            The "w" format only supports a single time point.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
            Defaults to self.verbose.
        """
        if ftype not in ['stc', 'w']:
            raise ValueError('ftype must be "stc" or "w", not "%s"' % ftype)
        if self.is_surface():
            lh_data = self.data[:len(self.lh_vertno)]
            rh_data = self.data[-len(self.rh_vertno):]

            if ftype == 'stc':
                logger.info('Writing STC to disk...')
                _write_stc(fname + '-lh.stc', tmin=self.tmin, tstep=self.tstep,
                           vertices=self.lh_vertno, data=lh_data)
                _write_stc(fname + '-rh.stc', tmin=self.tmin, tstep=self.tstep,
                           vertices=self.rh_vertno, data=rh_data)
            elif ftype == 'w':
                if self.shape[1] != 1:
                    raise ValueError('w files can only contain a single time '
                                     'point')
                logger.info('Writing STC to disk (w format)...')
                _write_w(fname + '-lh.w', vertices=self.lh_vertno,
                         data=lh_data[:, 0])
                _write_w(fname + '-rh.w', vertices=self.rh_vertno,
                         data=rh_data[:, 0])
        else:
            if isinstance(self.vertno, list):
                write_vertices = self.vertno[0]
            else:
                write_vertices = self.vertno
            if ftype == 'stc':
                logger.info('Writing STC to disk...')
                if not (fname.endswith('-vl.stc')
                        or fname.endswith('-vol.stc')):
                    fname += '-vl.stc'
                _write_stc(fname, tmin=self.tmin, tstep=self.tstep,
                           vertices=write_vertices, data=self.data)
            elif ftype == 'w':
                logger.info('Writing STC to disk (w format)...')
                if not (fname.endswith('-vl.w')
                        or fname.endswith('-vol.w')):
                    fname += '-vl.w'
                _write_w(fname, vertices=write_vertices, data=self.data)

        logger.info('[done]')

    def _remove_kernel_sens_data_(self):
        """Remove kernel and sensor space data

        Note: self._data is also computed if it is None
        """
        if self._kernel is not None or self._sens_data is not None:
            # we can no longer use the kernel and sens_data
            logger.info('STC data modified: removing kernel and sensor data')
            if self._data is None:
                self._data = np.dot(self._kernel, self._sens_data)
            self._kernel = None
            self._sens_data = None

    def __repr__(self):
        if isinstance(self.vertno, list):
            nv = sum([len(v) for v in self.vertno])
        else:
            nv = self.vertno.size
        s = "%d vertices" % nv
        if self.subject is not None:
            s += ", subject : %s" % self.subject
        s += ", tmin : %s (ms)" % (1e3 * self.tmin)
        s += ", tmax : %s (ms)" % (1e3 * self.times[-1])
        s += ", tstep : %s (ms)" % (1e3 * self.tstep)
        s += ", data size : %s x %s" % self.shape
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

        if self._kernel is not None and self._sens_data is not None:
            self._sens_data = self._sens_data[:, mask]
            self._data = None  # will be recomputed when data is accessed
        else:
            self._data = self._data[:, mask]

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
        # resampling in sensor instead of source space gives a somewhat
        # different result, so we don't allow it
        self._remove_kernel_sens_data_()

        o_sfreq = 1.0 / self.tstep
        self._data = resample(self._data, sfreq, o_sfreq, npad, n_jobs=n_jobs)

        # adjust indirectly affected variables
        self.tstep = 1.0 / sfreq
        self._update_times()

    @property
    def data(self):
        if self._data is None:
            # compute the solution the first time the data is accessed
            # return a "notify array", so we can later remove the kernel
            # and sensor data if the user modifies self._data
            self._data = _NotifyArray(np.dot(self._kernel, self._sens_data),
                modify_callback=self._remove_kernel_sens_data_)
        return self._data

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

    @property
    def shape(self):
        if self._data is not None:
            return self._data.shape
        return (self._kernel.shape[0], self._sens_data.shape[1])

    def is_surface(self):
        """Returns True if source estimate is defined over surfaces
        """
        if isinstance(self.vertno, list) and len(self.vertno) == 2:
            return True
        else:
            return False

    def _update_times(self):
        """Update the times attribute after changing tmin, tmax, or tstep"""
        self.times = self.tmin + (self.tstep * np.arange(self.shape[1]))

    def __add__(self, a):
        stc = copy.deepcopy(self)
        stc += a
        return stc

    def __iadd__(self, a):
        self._remove_kernel_sens_data_()
        if isinstance(a, SourceEstimate):
            _verify_source_estimate_compat(self, a)
            self._data += a.data
        else:
            self._data += a
        return self

    def __sub__(self, a):
        stc = copy.deepcopy(self)
        stc -= a
        return stc

    def __isub__(self, a):
        self._remove_kernel_sens_data_()
        if isinstance(a, SourceEstimate):
            _verify_source_estimate_compat(self, a)
            self._data -= a.data
        else:
            self._data -= a
        return self

    def __div__(self, a):
        stc = copy.deepcopy(self)
        stc /= a
        return stc

    def __idiv__(self, a):
        self._remove_kernel_sens_data_()
        if isinstance(a, SourceEstimate):
            _verify_source_estimate_compat(self, a)
            self._data /= a.data
        else:
            self._data /= a
        return self

    def __mul__(self, a):
        stc = copy.deepcopy(self)
        stc *= a
        return stc

    def __imul__(self, a):
        self._remove_kernel_sens_data_()
        if isinstance(a, SourceEstimate):
            _verify_source_estimate_compat(self, a)
            self._data *= a.data
        else:
            self._data *= a
        return self

    def __pow__(self, a):
        stc = copy.deepcopy(self)
        stc **= a
        return stc

    def __ipow__(self, a):
        self._remove_kernel_sens_data_()
        self._data **= a
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
        stc._remove_kernel_sens_data_()
        stc._data *= -1
        return stc

    def __pos__(self):
        return self

    def sqrt(self):
        """Return copy of SourceEstimate with sqrt(data)."""
        return self ** (0.5)

    def copy(self):
        """Return copy of SourceEstimate instance"""
        return copy.deepcopy(self)

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
        nv, _ = self.shape
        nt = len(times) - 1
        data = np.empty((nv, nt), dtype=self.data.dtype)
        for i in xrange(nt):
            idx = (self.times >= times[i]) & (self.times < times[i + 1])
            data[:, i] = func(self.data[:, idx], axis=1)

        tmin = times[0] + width / 2.
        stc = SourceEstimate(data, vertices=self.vertno,
                             tmin=tmin, tstep=width, subject=self.subject)
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
        # make sure label and stc are compatible
        if label.subject is not None and self.subject is not None \
                and label.subject != self.subject:
            raise RuntimeError('label and stc must have same subject names, '
                               'currently "%s" and "%s"' % (label.subject,
                                                            self.subject))

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
                                   tmin=self.tmin, tstep=self.tstep,
                                   subject=self.subject)
        return label_stc

    def expand(self, vertno):
        """Expand SourceEstimate to include more vertices

        This will add rows to stc.data (zero-filled) and modify stc.vertno
        to include all vertices in stc.vertno and the input vertno.

        Parameters
        ----------
        vertno : list of array
            New vertices to add. Can also contain old values.
        """
        if not isinstance(vertno, list):
            raise TypeError('vertno must be a list')
        if not len(self.vertno) == len(vertno):
            raise ValueError('vertno must have the same length as stc.vertno')

        # can no longer use kernel and sensor data
        self._remove_kernel_sens_data_()

        inserters = list()
        offsets = [0]
        for vi, (v_old, v_new) in enumerate(zip(self.vertno, vertno)):
            v_new = np.setdiff1d(v_new, v_old)
            inds = np.searchsorted(v_old, v_new)
            # newer numpy might overwrite inds after np.insert, copy here
            inserters += [inds.copy()]
            offsets += [len(v_old)]
            self.vertno[vi] = np.insert(v_old, inds, v_new)
        inds = [ii + offset for ii, offset in zip(inserters, offsets[:-1])]
        inds = np.concatenate(inds)
        new_data = np.zeros((len(inds), self._data.shape[1]))
        self._data = np.insert(self._data, inds, new_data, axis=0)
        return self

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

    def transform_data(self, transform_fun, fun_args=None,
                       idx=None, tmin_idx=None, tmax_idx=None, **kwargs):
        """Get data after a linear (time) transform has been applied

        The transorm is applied to each source time course independently.


        Parameters
        ----------
        transform_fun : callable
            The transform to be applied. The first parameter of the function
            is the input data. The first return value is the transformed
            data, remaining outputs are ignored. The first dimension of the
            transformed data has to be the same as the first dimension of the
            input data.
        fun_args : tuple | None
            Additional parameters to be passed to transform_fun.
        idx : array | None
            Indicices of source time courses for which to compute transform.
            If None, all time courses are used.
        tmin_idx : int | None
            Index of first time point to include. If None, the index of the
            first time point is used.
        tmax_idx : int | None
            Index of the first time point not to include. If None, time points
            up to (and including) the last time point are included.
        **kwargs : dict
            Keyword arguments to be passed to transform_fun.

        Returns
        -------
        data_t : ndarray
            The transformed data.

        .. note::
            Applying transforms can be significantly faster if the
            SourceEstimate object was created using "(kernel, sens_data)", for
            the "data" parameter as the transform is applied in sensor space.
            Inverse methods, e.g., "apply_inverse_epochs", or "lcmv_epochs" do
            this automatically (if possible).
        """

        if idx is None:
            # use all time courses by default
            idx = slice(None, None)

        if fun_args is None:
            fun_args = tuple()

        if self._kernel is None and self._sens_data is None:
            # transform source space data directly
            data_t = transform_fun(self.data[idx, tmin_idx:tmax_idx],
                                   *fun_args, **kwargs)

            if isinstance(data_t, tuple):
                # use only first return value
                data_t = data_t[0]
        else:
            # apply transform in sensor space
            sens_data_t = transform_fun(self._sens_data[:, tmin_idx:tmax_idx],
                                        *fun_args, **kwargs)

            if isinstance(sens_data_t, tuple):
                # use only first return value
                sens_data_t = sens_data_t[0]

            # apply inverse
            data_shape = sens_data_t.shape
            if len(data_shape) > 2:
                # flatten the last dimensions
                sens_data_t = sens_data_t.reshape(data_shape[0],
                                                  np.prod(data_shape[1:]))

            data_t = np.dot(self._kernel[idx, :], sens_data_t)

            # restore original shape if necessary
            if len(data_shape) > 2:
                data_t = data_t.reshape(data_t.shape[0], *data_shape[1:])

        return data_t

    def center_of_mass(self, subject=None, hemi=None, restrict_vertices=False,
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
        subject : string | None
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
        subject = _check_subject(self.subject, subject)

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
        t_ind = np.sum(masses * np.arange(self.shape[1])) / np.sum(masses)
        t = self.tmin + self.tstep * t_ind
        return vertex, hemi, t

    def plot(self, subject=None, surface='inflated', hemi='lh',
             colormap='hot', time_label='time=%0.2f ms',
             smoothing_steps=10, fmin=5., fmid=10., fmax=15.,
             transparent=True, alpha=1.0, time_viewer=False,
             config_opts={}, subjects_dir=None, figure=None):
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
        subject : str | None
            The subject name corresponding to FreeSurfer environment
            variable SUBJECT. If None stc.subject will be used. If that
            is None, the environment will be used.
        surface : str
            The type of surface (inflated, white etc.).
        hemi : str, 'lh' | 'rh' | 'both'
            The hemisphere to display. Using 'both' opens two separate figures,
            one for each hemisphere.
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
        alpha : float
            Alpha value to apply globally to the overlay.
        time_viewer : bool
            Display time viewer GUI.
        config_opts : dict
            Keyword arguments for Brain initialization.
            See pysurfer.viz.Brain.
        subjects_dir : str
            The path to the FreeSurfer subjects reconstructions.
            It corresponds to FreeSurfer environment variable SUBJECTS_DIR.
        figure : instance of mayavi.core.scene.Scene | None
            If None, the last figure will be cleaned and a new figure will
            be created.

        Returns
        -------
        brain : Brain | list of Brain
            A instance of surfer.viz.Brain from PySurfer For hemi='both',
            a list with Brain instances for the left and right hemisphere is
            returned.
        """
        brain = plot_source_estimates(self, subject, surface=surface,
                        hemi=hemi, colormap=colormap, time_label=time_label,
                        smoothing_steps=smoothing_steps, fmin=fmin, fmid=fmid,
                        fmax=fmax, transparent=transparent, alpha=alpha,
                        time_viewer=time_viewer, config_opts=config_opts,
                        subjects_dir=subjects_dir, figure=figure)
        return brain

    @verbose
    def morph(self, subject_to, grade=5, smooth=None,
              subjects_dir=None, buffer_size=64, n_jobs=1, subject_from=None,
              verbose=None):
        """Morph a source estimate from one subject to another

        Parameters
        ----------
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
            computing vertex locations. Note that if subject='fsaverage'
            and 'grade=5', this set of vertices will automatically be used
            (instead of computed) for speed, since this is a common morph.
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
            Number of jobs to run in parallel.
        subject_from : string
            Name of the original subject as named in the SUBJECTS_DIR.
            If None, self.subject will be used.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).

        Returns
        -------
        stc_to : SourceEstimate
            Source estimate for the destination subject.
        """
        subject_from = _check_subject(self.subject, subject_from)
        return morph_data(subject_from, subject_to, self, grade, smooth,
                          subjects_dir, buffer_size, n_jobs, verbose)

    def morph_precomputed(self, subject_to, vertices_to, morph_mat,
                          subject_from=None):
        """Morph source estimate between subjects using a precomputed matrix

        Parameters
        ----------
        subject_to : string
            Name of the subject on which to morph as named in the SUBJECTS_DIR.
        vertices_to : list of array of int
            The vertices on the destination subject's brain.
        morph_mat : sparse matrix
            The morphing matrix, usually from compute_morph_matrix.
        subject_from : string | None
            Name of the original subject as named in the SUBJECTS_DIR.
            If None, self.subject will be used.

        Returns
        -------
        stc_to : SourceEstimate
            Source estimate for the destination subject.
        """
        subject_from = _check_subject(self.subject, subject_from)
        return morph_data_precomputed(subject_from, subject_to, self,
                                      vertices_to, morph_mat)

    def as_data_frame(self, index=None, scale_time=1e3, copy=True):
        """Represent source estimates as Pandas DataFrame

        Export source estimates in tabular structure with vertices as columns
        and two additional info columns 'subject' and 'time'.
        This function is useful to visualize and analyse source time courses
        with external statistical software such as statsmodels or R.

        Parameters
        ----------
        index : tuple of str | None
            Column to be used as index for the data. Valid string options
            are 'subject' and 'time'. If None, all three info
            columns will be included in the table as categorial data.
        scale_time : float
            Scaling to be applied to time units.
        copy : bool
            If true, data will be copied. Else data may be modified in place.

        Returns
        -------
        df : instance of DataFrame
            Source estimates exported into tabular data structure.
        """
        pd = _check_pandas_installed()

        default_index = ['subject', 'time']
        if index is not None:
            _check_pandas_index_arguments(index, default_index)
        else:
            index = default_index

        data = self.data.T
        shape = data.shape
        mindex = list()
        mindex.append(('time', self.times * scale_time))
        mindex.append(('subject', np.repeat(self.subject, shape[0])))

        if copy:
            data = data.copy()
        assert all(len(mdx) == len(mindex[0]) for mdx in mindex)

        vert_names = [i for e in [['%s %i' % ('LH' if ii < 1 else 'RH', vert)
                      for vert in vertno]
                      for ii, vertno in enumerate(self.vertno)] for i in e]
        df = pd.DataFrame(data, columns=vert_names)
        [df.insert(i, k, v) for i, (k, v) in enumerate(mindex)]

        if index is not None:
            with warnings.catch_warnings(True):
                df.set_index(index, inplace=True)
            if 'time' in df.index.names and hasattr(df.index, 'levels'):
                df.index.levels[1] = df.index.levels[1].astype(int)

        return df

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
        Name of the original subject as named in the SUBJECTS_DIR.
    subject_to : string
        Name of the subject on which to morph as named in the SUBJECTS_DIR.
    subjects_dir : string
        Path to SUBJECTS_DIR is not set in the environment.
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
        The triangles.

    Returns
    -------
    edges : sparse matrix
        The adjacency matrix.
    """
    npoints = np.max(tris) + 1
    ones_ntris = np.ones(3 * len(tris))
    a, b, c = tris.T
    x = np.concatenate((a, b, c))
    y = np.concatenate((b, c, a))
    edges = coo_matrix((ones_ntris, (x, y)), shape=(npoints, npoints))
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
        A n_vertices x n_times (or other dimension) dataset to morph.
    idx_use : array of int
        Vertices from the original subject's data.
    e : sparse matrix
        The mesh edges of the "from" subject.
    smooth : int
        Number of smoothing iterations to perform. A hard limit of 100 is
        also imposed.
    n_vertices : int
        Number of vertices.
    nearest : array of int
        Vertices on the destination surface to use.
    maps : sparse matrix
        Morph map from one subject to the other.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    data_morphed : array, or csr sparse matrix
        The morphed data (same type as input).
    """

    n_iter = 99  # max nb of smoothing iterations (minus one)
    if smooth is not None:
        smooth -= 1
    # make sure we're in CSR format
    e = e.tocsr()
    if sparse.issparse(data):
        use_sparse = True
        if not isinstance(data, sparse.csr_matrix):
            data = data.tocsr()
    else:
        use_sparse = False
    done = False
    # do the smoothing
    for k in range(n_iter + 1):
        # get the row sum
        mult = np.zeros(e.shape[1])
        mult[idx_use] = 1
        idx_use_data = idx_use
        data_sum = e * mult

        # new indices are non-zero sums
        idx_use = np.where(data_sum)[0]

        # typically want to make the next iteration have these indices
        idx_out = idx_use

        # figure out if this is the last iteration
        if smooth is None:
            if k == n_iter or len(idx_use) >= n_vertices:
                # stop when vertices filled
                idx_out = None
                done = True
        elif k == smooth:
            idx_out = None
            done = True

        # do standard smoothing multiplication
        data = _morph_mult(data, e, use_sparse, idx_use_data, idx_out)

        if done is True:
            break

        # do standard normalization
        if use_sparse:
            data.data /= data_sum[idx_use].repeat(np.diff(data.indptr))
        else:
            data /= data_sum[idx_use][:, None]

    # do special normalization for last iteration
    if use_sparse:
        data_sum[data_sum == 0] = 1
        data.data /= data_sum.repeat(np.diff(data.indptr))
    else:
        data[idx_use, :] /= data_sum[idx_use][:, None]

    logger.info('    %d smooth iterations done.' % (k + 1))
    data_morphed = maps[nearest, :] * data
    return data_morphed


def _morph_mult(data, e, use_sparse, idx_use_data, idx_use_out=None):
    """Helper for morphing

    Equivalent to "data = (e[:, idx_use_data] * data)[idx_use_out]"
    but faster.
    """
    if len(idx_use_data) < e.shape[1]:
        if use_sparse:
            data = e[:, idx_use_data] * data
        else:
            # constructing a new sparse matrix is faster than sub-indexing
            # e[:, idx_use_data]!
            col, row = np.meshgrid(np.arange(data.shape[1]), idx_use_data)
            d_sparse = sparse.csr_matrix((data.ravel(),
                                          (row.ravel(), col.ravel())),
                                         shape=(e.shape[1], data.shape[1]))
            data = e * d_sparse
            data = np.asarray(data.todense())
    else:
        data = e * data

    # trim data
    if idx_use_out is not None:
        data = data[idx_use_out]
    return data


def _compute_nearest(xhs, rr, use_balltree=True):
    """Find nearest neighbors

    Note: The rows in xhs and rr must all be unit-length vectors, otherwise
    the result will be incorrect.

    Parameters
    ----------
    xhs : array, shape=(n_samples, n_dim)
        Points of data set.
    rr : array, shape=(n_query, n_dim)
        Points to find nearest neighbors for.
    use_balltree : bool
        Use fast BallTree based search from scikit-learn. If scikit-learn
        is not installed it will fall back to the slow brute force search.

    Returns
    -------
    nearest : array, shape=(n_query,)
        Index of nearest neighbor in xhs for every point in rr.
    """
    if use_balltree:
        try:
            from sklearn.neighbors import BallTree
        except ImportError:
            logger.info('Nearest-neighbor searches will be significantly '
                        'faster if scikit-learn is installed.')
            use_balltree = False

    if use_balltree:
        ball_tree = BallTree(xhs)
        nearest = ball_tree.query(rr, k=1, return_distance=False)[:, 0]
    else:
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
        computing vertex locations. Note that if subject='fsaverage'
        and 'grade=5', this set of vertices will automatically be used
        (instead of computed) for speed, since this is a common morph.
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
            vertices = [np.array([], dtype=int), np.array([], dtype=int)]
        else:
            data = data_morphed[1]
            vertices = [np.array([], dtype=int), vertices[1]]
    elif data_morphed[1] is None:
        data = data_morphed[0]
        vertices = [vertices[0], np.array([], dtype=int)]
    else:
        data = np.r_[data_morphed[0], data_morphed[1]]

    stc_to = SourceEstimate(data, vertices, stc_from.tmin, stc_from.tstep,
                            subject=subject_to, verbose=stc_from.verbose)
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
        computing vertex locations. Note that if subject='fsaverage'
        and 'grade=5', this set of vertices will automatically be used
        (instead of computed) for speed, since this is a common morph.
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
    # add special case for fsaverage for speed
    if subject == 'fsaverage' and grade == 5:
        return [np.arange(10242), np.arange(10242)]
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
    """Morph source estimate between subjects using a precomputed matrix

    Parameters
    ----------
    subject_from : string
        Name of the original subject as named in the SUBJECTS_DIR.
    subject_to : string
        Name of the subject on which to morph as named in the SUBJECTS_DIR.
    stc_from : SourceEstimate
        Source estimates for subject "from" to morph.
    vertices_to : list of array of int
        The vertices on the destination subject's brain.
    morph_mat : sparse matrix
        The morphing matrix, typically from compute_morph_matrix.

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

    if stc_from.subject is not None and stc_from.subject != subject_from:
        raise ValueError('stc_from.subject and subject_from must match')
    data = morph_mat * stc_from.data
    stc_to = SourceEstimate(data, vertices_to, stc_from.tmin, stc_from.tstep,
                            verbose=stc_from.verbose, subject=subject_to)
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
    ico = read_bem_surfaces(ico_file_name, s_id=9000 + grade)

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
