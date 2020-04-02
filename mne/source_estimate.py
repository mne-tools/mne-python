# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Mads Jensen <mje.mads@gmail.com>
#
# License: BSD (3-clause)

import contextlib
import copy
import os.path as op
import numpy as np
from scipy import linalg, sparse
from scipy.sparse import coo_matrix, block_diag as sparse_block_diag

from .cov import Covariance
from .evoked import _get_peak
from .filter import resample
from .fixes import einsum
from .surface import read_surface, _get_ico_surface, mesh_edges
from .source_space import (_ensure_src, _get_morph_src_reordering,
                           _ensure_src_subject, SourceSpaces, _get_src_nn)
from .utils import (get_subjects_dir, _check_subject, logger, verbose,
                    _time_mask, warn as warn_, copy_function_doc_to_method_doc,
                    fill_doc, _check_option, _validate_type, _check_src_normal,
                    _check_stc_units, _check_pandas_installed,
                    _check_pandas_index_arguments, _convert_times,
                    _build_data_frame, _check_time_format, _check_scaling_time)
from .viz import (plot_source_estimates, plot_vector_source_estimates,
                  plot_volume_source_estimates)
from .io.base import TimeMixin
from .io.meas_info import Info
from .externals.h5io import read_hdf5, write_hdf5


def _read_stc(filename):
    """Aux Function."""
    with open(filename, 'rb') as fid:
        buf = fid.read()

    stc = dict()
    offset = 0
    num_bytes = 4

    # read tmin in ms
    stc['tmin'] = float(np.frombuffer(buf, dtype=">f4", count=1,
                                      offset=offset))
    stc['tmin'] /= 1000.0
    offset += num_bytes

    # read sampling rate in ms
    stc['tstep'] = float(np.frombuffer(buf, dtype=">f4", count=1,
                                       offset=offset))
    stc['tstep'] /= 1000.0
    offset += num_bytes

    # read number of vertices/sources
    vertices_n = int(np.frombuffer(buf, dtype=">u4", count=1, offset=offset))
    offset += num_bytes

    # read the source vector
    stc['vertices'] = np.frombuffer(buf, dtype=">u4", count=vertices_n,
                                    offset=offset)
    offset += num_bytes * vertices_n

    # read the number of timepts
    data_n = int(np.frombuffer(buf, dtype=">u4", count=1, offset=offset))
    offset += num_bytes

    if (vertices_n and  # vertices_n can be 0 (empty stc)
            ((len(buf) // 4 - 4 - vertices_n) % (data_n * vertices_n)) != 0):
        raise ValueError('incorrect stc file size')

    # read the data matrix
    stc['data'] = np.frombuffer(buf, dtype=">f4", count=vertices_n * data_n,
                                offset=offset)
    stc['data'] = stc['data'].reshape([data_n, vertices_n]).T

    return stc


def _write_stc(filename, tmin, tstep, vertices, data):
    """Write an STC file.

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
    fid.write(np.array(1000 * tmin, dtype='>f4').tobytes())
    # write sampling rate in ms
    fid.write(np.array(1000 * tstep, dtype='>f4').tobytes())
    # write number of vertices
    fid.write(np.array(vertices.shape[0], dtype='>u4').tobytes())
    # write the vertex indices
    fid.write(np.array(vertices, dtype='>u4').tobytes())

    # write the number of timepts
    fid.write(np.array(data.shape[1], dtype='>u4').tobytes())
    #
    # write the data
    #
    fid.write(np.array(data.T, dtype='>f4').tobytes())

    # close the file
    fid.close()


def _read_3(fid):
    """Read 3 byte integer from file."""
    data = np.fromfile(fid, dtype=np.uint8, count=3).astype(np.int32)

    out = np.left_shift(data[0], 16) + np.left_shift(data[1], 8) + data[2]

    return out


def _read_w(filename):
    """Read a w file.

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
    with open(filename, 'rb', buffering=0) as fid:  # buffering=0 for np bug
        # skip first 2 bytes
        fid.read(2)

        # read number of vertices/sources (3 byte integer)
        vertices_n = int(_read_3(fid))

        vertices = np.zeros((vertices_n), dtype=np.int32)
        data = np.zeros((vertices_n), dtype=np.float32)

        # read the vertices and data
        for i in range(vertices_n):
            vertices[i] = _read_3(fid)
            data[i] = np.fromfile(fid, dtype='>f4', count=1)[0]

        w = dict()
        w['vertices'] = vertices
        w['data'] = data

    return w


def _write_3(fid, val):
    """Write 3 byte integer to file."""
    f_bytes = np.zeros((3), dtype=np.uint8)
    f_bytes[0] = (val >> 16) & 255
    f_bytes[1] = (val >> 8) & 255
    f_bytes[2] = val & 255
    fid.write(f_bytes.tobytes())


def _write_w(filename, vertices, data):
    """Write a w file.

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
    assert (len(vertices) == len(data))

    fid = open(filename, 'wb')

    # write 2 zero bytes
    fid.write(np.zeros((2), dtype=np.uint8).tobytes())

    # write number of vertices/sources (3 byte integer)
    vertices_n = len(vertices)
    _write_3(fid, vertices_n)

    # write the vertices and data
    for i in range(vertices_n):
        _write_3(fid, vertices[i])
        # XXX: without float() endianness is wrong, not sure why
        fid.write(np.array(float(data[i]), dtype='>f4').tobytes())

    # close the file
    fid.close()


def read_source_estimate(fname, subject=None):
    """Read a source estimate object.

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

    Returns
    -------
    stc : SourceEstimate | VectorSourceEstimate | VolSourceEstimate | MixedSourceEstimate
        The source estimate object loaded from file.

    Notes
    -----
     - for volume source estimates, ``fname`` should provide the path to a
       single file named '*-vl.stc` or '*-vol.stc'
     - for surface source estimates, ``fname`` should either provide the
       path to the file corresponding to a single hemisphere ('*-lh.stc',
       '*-rh.stc') or only specify the asterisk part in these patterns. In any
       case, the function expects files for both hemisphere with names
       following this pattern.
     - for vector surface source estimates, only HDF5 files are supported.
     - for mixed source estimates, only HDF5 files are supported.
     - for single time point .w files, ``fname`` should follow the same
       pattern as for surface estimates, except that files are named
       '*-lh.w' and '*-rh.w'.
    """  # noqa: E501
    fname_arg = fname
    _validate_type(fname, 'path-like', 'fname')
    fname = str(fname)

    # make sure corresponding file(s) can be found
    ftype = None
    if op.exists(fname):
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
        elif fname.endswith('.h5'):
            ftype = 'h5'
            fname = fname[:-3]
        else:
            raise RuntimeError('Unknown extension for file %s' % fname_arg)

    if ftype != 'volume':
        stc_exist = [op.exists(f)
                     for f in [fname + '-rh.stc', fname + '-lh.stc']]
        w_exist = [op.exists(f)
                   for f in [fname + '-rh.w', fname + '-lh.w']]
        if all(stc_exist) and ftype != 'w':
            ftype = 'surface'
        elif all(w_exist):
            ftype = 'w'
        elif op.exists(fname + '.h5'):
            ftype = 'h5'
        elif op.exists(fname + '-stc.h5'):
            ftype = 'h5'
            fname += '-stc'
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
        ftype = 'surface'
    elif ftype == 'h5':
        kwargs = read_hdf5(fname + '.h5', title='mnepython')
        ftype = kwargs.pop('src_type', 'surface')

    if ftype != 'volume':
        # Make sure the vertices are ordered
        vertices = kwargs['vertices']
        if any(np.any(np.diff(v.astype(int)) <= 0) for v in vertices):
            sidx = [np.argsort(verts) for verts in vertices]
            vertices = [verts[idx] for verts, idx in zip(vertices, sidx)]
            data = kwargs['data'][np.r_[sidx[0], len(sidx[0]) + sidx[1]]]
            kwargs['vertices'] = vertices
            kwargs['data'] = data

    if 'subject' not in kwargs:
        kwargs['subject'] = subject
    if subject is not None and subject != kwargs['subject']:
        raise RuntimeError('provided subject name "%s" does not match '
                           'subject name from the file "%s'
                           % (subject, kwargs['subject']))

    vector = kwargs['data'].ndim == 3
    if ftype in ('volume', 'discrete'):
        klass = VolVectorSourceEstimate if vector else VolSourceEstimate
    elif ftype == 'mixed':
        if vector:
            # XXX we should really support this at some point
            raise NotImplementedError('Vector mixed source estimates not yet '
                                      'supported')
        klass = MixedSourceEstimate
    else:
        assert ftype == 'surface'
        klass = VectorSourceEstimate if vector else SourceEstimate
    return klass(**kwargs)


def _get_src_type(src, vertices, warn_text=None):
    src_type = None
    if src is None:
        if warn_text is None:
            warn_("src should not be None for a robust guess of stc type.")
        else:
            warn_(warn_text)
        if isinstance(vertices, list) and len(vertices) == 2:
            src_type = 'surface'
        elif isinstance(vertices, np.ndarray) or isinstance(vertices, list) \
                and len(vertices) == 1:
            src_type = 'volume'
        elif isinstance(vertices, list) and len(vertices) > 2:
            src_type = 'mixed'
    else:
        src_type = src.kind
    assert src_type in ('surface', 'volume', 'mixed', 'discrete')
    return src_type


def _make_stc(data, vertices, src_type=None, tmin=None, tstep=None,
              subject=None, vector=False, source_nn=None, warn_text=None):
    """Generate a surface, vector-surface, volume or mixed source estimate."""
    def guess_src_type():
        return _get_src_type(src=None, vertices=vertices, warn_text=warn_text)

    src_type = guess_src_type() if src_type is None else src_type

    if vector and src_type == 'mixed':  # XXX this should be supported someday
        raise NotImplementedError(
            'Vector source estimates for mixed source spaces are not supported'
        )

    if vector and src_type == 'surface' and source_nn is None:
        raise RuntimeError('No source vectors supplied.')

    # infer Klass from src_type
    if src_type == 'surface':
        Klass = VectorSourceEstimate if vector else SourceEstimate
    elif src_type in ('volume', 'discrete'):
        Klass = VolVectorSourceEstimate if vector else VolSourceEstimate
    elif src_type == 'mixed':
        Klass = MixedSourceEstimate
    else:
        raise ValueError('vertices has to be either a list with one or more '
                         'arrays or an array')

    # massage the data
    if src_type == 'surface' and vector:
        n_vertices = len(vertices[0]) + len(vertices[1])
        data = np.matmul(
            np.transpose(source_nn.reshape(n_vertices, 3, 3), axes=[0, 2, 1]),
            data.reshape(n_vertices, 3, -1)
        )
    elif src_type in ('volume', 'discrete') and vector:
        data = data.reshape((-1, 3, data.shape[-1]))
    else:
        pass  # noqa

    return Klass(
        data=data, vertices=vertices, tmin=tmin, tstep=tstep, subject=subject
    )


def _verify_source_estimate_compat(a, b):
    """Make sure two SourceEstimates are compatible for arith. operations."""
    compat = False
    if type(a) != type(b):
        raise ValueError('Cannot combine %s and %s.' % (type(a), type(b)))
    if len(a.vertices) == len(b.vertices):
        if all(np.array_equal(av, vv)
               for av, vv in zip(a.vertices, b.vertices)):
            compat = True
    if not compat:
        raise ValueError('Cannot combine source estimates that do not have '
                         'the same vertices. Consider using stc.expand().')
    if a.subject != b.subject:
        raise ValueError('source estimates do not have the same subject '
                         'names, %r and %r' % (a.subject, b.subject))


class _BaseSourceEstimate(TimeMixin):
    """Base class for all source estimates.

    Parameters
    ----------
    data : array, shape (n_dipoles, n_times) | tuple, shape (2,)
        The data in source space. The data can either be a single array or
        a tuple with two arrays: "kernel" shape (n_vertices, n_sensors) and
        "sens_data" shape (n_sensors, n_times). In this case, the source
        space data corresponds to ``np.dot(kernel, sens_data)``.
    vertices : array | list of array
        Vertex numbers corresponding to the data.
    tmin : float
        Time point of the first sample in data.
    tstep : float
        Time step between successive samples in data.
    subject : str | None
        The subject name. While not necessary, it is safer to set the
        subject parameter to avoid analysis errors.
    %(verbose)s

    Attributes
    ----------
    subject : str | None
        The subject name.
    times : array, shape (n_times,)
        The time vector.
    vertices : array | list of array of shape (n_dipoles,)
        The indices of the dipoles in the different source spaces. Can
        be an array if there is only one source space (e.g., for volumes).
    data : array of shape (n_dipoles, n_times)
        The data in source space.
    shape : tuple
        The shape of the data. A tuple of int (n_dipoles, n_times).
    """

    @verbose
    def __init__(self, data, vertices=None, tmin=None, tstep=None,
                 subject=None, verbose=None):  # noqa: D102
        assert hasattr(self, '_data_ndim'), self.__class__.__name__
        assert hasattr(self, '_src_type'), self.__class__.__name__
        kernel, sens_data = None, None
        if isinstance(data, tuple):
            if len(data) != 2:
                raise ValueError('If data is a tuple it has to be length 2')
            kernel, sens_data = data
            data = None
            if kernel.shape[1] != sens_data.shape[0]:
                raise ValueError('kernel (%s) and sens_data (%s) have invalid '
                                 'dimensions'
                                 % (kernel.shape, sens_data.shape))
            if sens_data.ndim != 2:
                raise ValueError('The sensor data must have 2 dimensions, got '
                                 '%s' % (sens_data.ndim,))

        if isinstance(vertices, list):
            vertices = [np.asarray(v, int) for v in vertices]
            if any(np.any(np.diff(v.astype(int)) <= 0) for v in vertices):
                raise ValueError('Vertices must be ordered in increasing '
                                 'order.')

            n_src = sum([len(v) for v in vertices])

            if len(vertices) == 1:
                vertices = vertices[0]
        elif isinstance(vertices, np.ndarray):
            n_src = len(vertices)
        else:
            raise ValueError('Vertices must be a list or numpy array')

        # safeguard the user against doing something silly
        if data is not None:
            if data.shape[0] != n_src:
                raise ValueError('Number of vertices (%i) and stc.shape[0] '
                                 '(%i) must match' % (n_src, data.shape[0]))
            if data.ndim == self._data_ndim - 1:  # allow upbroadcasting
                data = data[..., np.newaxis]
            if data.ndim != self._data_ndim:
                raise ValueError('Data (shape %s) must have %s dimensions for '
                                 '%s' % (data.shape, self._data_ndim,
                                         self.__class__.__name__))

        self._data = data
        self._tmin = tmin
        self._tstep = tstep
        self.vertices = vertices
        self.verbose = verbose
        self._kernel = kernel
        self._sens_data = sens_data
        self._kernel_removed = False
        self._times = None
        self._update_times()
        self.subject = _check_subject(None, subject, False)

    def __repr__(self):  # noqa: D105
        s = "%d vertices" % (sum(len(v) for v in self._vertices_list),)
        if self.subject is not None:
            s += ", subject : %s" % self.subject
        s += ", tmin : %s (ms)" % (1e3 * self.tmin)
        s += ", tmax : %s (ms)" % (1e3 * self.times[-1])
        s += ", tstep : %s (ms)" % (1e3 * self.tstep)
        s += ", data shape : %s" % (self.shape,)
        return "<%s  |  %s>" % (type(self).__name__, s)

    @property
    def _vertices_list(self):
        return self.vertices

    @verbose
    def save(self, fname, ftype='h5', verbose=None):
        """Save the full source estimate to an HDF5 file.

        Parameters
        ----------
        fname : str
            The file name to write the source estimate to, should end in
            '-stc.h5'.
        ftype : str
            File format to use. Currently, the only allowed values is "h5".
        %(verbose_meth)s
        """
        _validate_type(fname, 'path-like', 'fname')
        fname = str(fname)
        if ftype != 'h5':
            raise ValueError('%s objects can only be written as HDF5 files.'
                             % (self.__class__.__name__,))
        if not fname.endswith('.h5'):
            fname += '-stc.h5'
        write_hdf5(fname,
                   dict(vertices=self.vertices, data=self.data, tmin=self.tmin,
                        tstep=self.tstep, subject=self.subject,
                        src_type=self._src_type),
                   title='mnepython', overwrite=True)

    @property
    def sfreq(self):
        """Sample rate of the data."""
        return 1. / self.tstep

    def _remove_kernel_sens_data_(self):
        """Remove kernel and sensor space data and compute self._data."""
        if self._kernel is not None or self._sens_data is not None:
            self._kernel_removed = True
            self._data = np.dot(self._kernel, self._sens_data)
            self._kernel = None
            self._sens_data = None

    @fill_doc
    def crop(self, tmin=None, tmax=None, include_tmax=True):
        """Restrict SourceEstimate to a time interval.

        Parameters
        ----------
        tmin : float | None
            The first time point in seconds. If None the first present is used.
        tmax : float | None
            The last time point in seconds. If None the last present is used.
        %(include_tmax)s

        Returns
        -------
        stc : instance of SourceEstimate
            The cropped source estimate.
        """
        mask = _time_mask(self.times, tmin, tmax, sfreq=self.sfreq,
                          include_tmax=include_tmax)
        self.tmin = self.times[np.where(mask)[0][0]]
        if self._kernel is not None and self._sens_data is not None:
            self._sens_data = self._sens_data[..., mask]
        else:
            self.data = self.data[..., mask]

        return self  # return self for chaining methods

    @verbose
    def resample(self, sfreq, npad='auto', window='boxcar', n_jobs=1,
                 verbose=None):
        """Resample data.

        Parameters
        ----------
        sfreq : float
            New sample rate to use.
        npad : int | str
            Amount to pad the start and end of the data.
            Can also be "auto" to use a padding that will result in
            a power-of-two size (can be much faster).
        window : str | tuple
            Window to use in resampling. See :func:`scipy.signal.resample`.
        %(n_jobs)s
        %(verbose_meth)s

        Returns
        -------
        stc : instance of SourceEstimate
            The resampled source estimate.

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
        data = self.data
        if data.dtype == np.float32:
            data = data.astype(np.float64)
        self.data = resample(data, sfreq, o_sfreq, npad, n_jobs=n_jobs)

        # adjust indirectly affected variables
        self.tstep = 1.0 / sfreq
        return self

    @property
    def data(self):
        """Numpy array of source estimate data."""
        if self._data is None:
            # compute the solution the first time the data is accessed and
            # remove the kernel and sensor data
            self._remove_kernel_sens_data_()
        return self._data

    @data.setter
    def data(self, value):
        value = np.asarray(value)
        if self._data is not None and value.ndim != self._data.ndim:
            raise ValueError('Data array should have %d dimensions.' %
                             self._data.ndim)

        # vertices can be a single number, so cast to ndarray
        if isinstance(self.vertices, list):
            n_verts = sum([len(v) for v in self.vertices])
        elif isinstance(self.vertices, np.ndarray):
            n_verts = len(self.vertices)
        else:
            raise ValueError('Vertices must be a list or numpy array')

        if value.shape[0] != n_verts:
            raise ValueError('The first dimension of the data array must '
                             'match the number of vertices (%d != %d)' %
                             (value.shape[0], n_verts))

        self._data = value
        self._update_times()

    @property
    def shape(self):
        """Shape of the data."""
        if self._data is not None:
            return self._data.shape
        return (self._kernel.shape[0], self._sens_data.shape[1])

    @property
    def tmin(self):
        """The first timestamp."""
        return self._tmin

    @tmin.setter
    def tmin(self, value):
        self._tmin = float(value)
        self._update_times()

    @property
    def tstep(self):
        """The change in time between two consecutive samples (1 / sfreq)."""
        return self._tstep

    @tstep.setter
    def tstep(self, value):
        if value <= 0:
            raise ValueError('.tstep must be greater than 0.')
        self._tstep = float(value)
        self._update_times()

    @property
    def times(self):
        """A timestamp for each sample."""
        return self._times

    @times.setter
    def times(self, value):
        raise ValueError('You cannot write to the .times attribute directly. '
                         'This property automatically updates whenever '
                         '.tmin, .tstep or .data changes.')

    def _update_times(self):
        """Update the times attribute after changing tmin, tmax, or tstep."""
        self._times = self.tmin + (self.tstep * np.arange(self.shape[-1]))
        self._times.flags.writeable = False

    def __add__(self, a):
        """Add source estimates."""
        stc = self.copy()
        stc += a
        return stc

    def __iadd__(self, a):  # noqa: D105
        self._remove_kernel_sens_data_()
        if isinstance(a, _BaseSourceEstimate):
            _verify_source_estimate_compat(self, a)
            self.data += a.data
        else:
            self.data += a
        return self

    def mean(self):
        """Make a summary stc file with mean over time points.

        Returns
        -------
        stc : SourceEstimate | VectorSourceEstimate
            The modified stc.
        """
        out = self.sum()
        out /= len(self.times)
        return out

    def sum(self):
        """Make a summary stc file with sum over time points.

        Returns
        -------
        stc : SourceEstimate | VectorSourceEstimate
            The modified stc.
        """
        data = self.data
        tmax = self.tmin + self.tstep * data.shape[-1]
        tmin = (self.tmin + tmax) / 2.
        tstep = tmax - self.tmin
        sum_stc = self.__class__(self.data.sum(axis=-1, keepdims=True),
                                 vertices=self.vertices, tmin=tmin,
                                 tstep=tstep, subject=self.subject)
        return sum_stc

    def __sub__(self, a):
        """Subtract source estimates."""
        stc = self.copy()
        stc -= a
        return stc

    def __isub__(self, a):  # noqa: D105
        self._remove_kernel_sens_data_()
        if isinstance(a, _BaseSourceEstimate):
            _verify_source_estimate_compat(self, a)
            self.data -= a.data
        else:
            self.data -= a
        return self

    def __truediv__(self, a):  # noqa: D105
        return self.__div__(a)

    def __div__(self, a):  # noqa: D105
        """Divide source estimates."""
        stc = self.copy()
        stc /= a
        return stc

    def __itruediv__(self, a):  # noqa: D105
        return self.__idiv__(a)

    def __idiv__(self, a):  # noqa: D105
        self._remove_kernel_sens_data_()
        if isinstance(a, _BaseSourceEstimate):
            _verify_source_estimate_compat(self, a)
            self.data /= a.data
        else:
            self.data /= a
        return self

    def __mul__(self, a):
        """Multiply source estimates."""
        stc = self.copy()
        stc *= a
        return stc

    def __imul__(self, a):  # noqa: D105
        self._remove_kernel_sens_data_()
        if isinstance(a, _BaseSourceEstimate):
            _verify_source_estimate_compat(self, a)
            self.data *= a.data
        else:
            self.data *= a
        return self

    def __pow__(self, a):  # noqa: D105
        stc = self.copy()
        stc **= a
        return stc

    def __ipow__(self, a):  # noqa: D105
        self._remove_kernel_sens_data_()
        self.data **= a
        return self

    def __radd__(self, a):  # noqa: D105
        return self + a

    def __rsub__(self, a):  # noqa: D105
        return self - a

    def __rmul__(self, a):  # noqa: D105
        return self * a

    def __rdiv__(self, a):  # noqa: D105
        return self / a

    def __neg__(self):  # noqa: D105
        """Negate the source estimate."""
        stc = self.copy()
        stc._remove_kernel_sens_data_()
        stc.data *= -1
        return stc

    def __pos__(self):  # noqa: D105
        return self

    def __abs__(self):
        """Compute the absolute value of the data.

        Returns
        -------
        stc : instance of _BaseSourceEstimate
            A version of the source estimate, where the data attribute is set
            to abs(self.data).
        """
        stc = self.copy()
        stc._remove_kernel_sens_data_()
        stc._data = abs(stc._data)
        return stc

    def sqrt(self):
        """Take the square root.

        Returns
        -------
        stc : instance of SourceEstimate
            A copy of the SourceEstimate with sqrt(data).
        """
        return self ** (0.5)

    def copy(self):
        """Return copy of source estimate instance.

        Returns
        -------
        stc : instance of SourceEstimate
            A copy of the source estimate.
        """
        return copy.deepcopy(self)

    def bin(self, width, tstart=None, tstop=None, func=np.mean):
        """Return a source estimate object with data summarized over time bins.

        Time bins of ``width`` seconds. This method is intended for
        visualization only. No filter is applied to the data before binning,
        making the method inappropriate as a tool for downsampling data.

        Parameters
        ----------
        width : scalar
            Width of the individual bins in seconds.
        tstart : scalar | None
            Time point where the first bin starts. The default is the first
            time point of the stc.
        tstop : scalar | None
            Last possible time point contained in a bin (if the last bin would
            be shorter than width it is dropped). The default is the last time
            point of the stc.
        func : callable
            Function that is applied to summarize the data. Needs to accept a
            numpy.array as first input and an ``axis`` keyword argument.

        Returns
        -------
        stc : SourceEstimate | VectorSourceEstimate
            The binned source estimate.
        """
        if tstart is None:
            tstart = self.tmin
        if tstop is None:
            tstop = self.times[-1]

        times = np.arange(tstart, tstop + self.tstep, width)
        nt = len(times) - 1
        data = np.empty(self.shape[:-1] + (nt,), dtype=self.data.dtype)
        for i in range(nt):
            idx = (self.times >= times[i]) & (self.times < times[i + 1])
            data[..., i] = func(self.data[..., idx], axis=-1)

        tmin = times[0] + width / 2.
        stc = self.copy()
        stc._data = data
        stc.tmin = tmin
        stc.tstep = width
        return stc

    def transform_data(self, func, idx=None, tmin_idx=None, tmax_idx=None):
        """Get data after a linear (time) transform has been applied.

        The transform is applied to each source time course independently.

        Parameters
        ----------
        func : callable
            The transform to be applied, including parameters (see, e.g.,
            :func:`functools.partial`). The first parameter of the function is
            the input data. The first return value is the transformed data,
            remaining outputs are ignored. The first dimension of the
            transformed data has to be the same as the first dimension of the
            input data.
        idx : array | None
            Indicices of source time courses for which to compute transform.
            If None, all time courses are used.
        tmin_idx : int | None
            Index of first time point to include. If None, the index of the
            first time point is used.
        tmax_idx : int | None
            Index of the first time point not to include. If None, time points
            up to (and including) the last time point are included.

        Returns
        -------
        data_t : ndarray
            The transformed data.

        Notes
        -----
        Applying transforms can be significantly faster if the
        SourceEstimate object was created using "(kernel, sens_data)", for
        the "data" parameter as the transform is applied in sensor space.
        Inverse methods, e.g., "apply_inverse_epochs", or "apply_lcmv_epochs"
        do this automatically (if possible).
        """
        if idx is None:
            # use all time courses by default
            idx = slice(None, None)

        if self._kernel is None and self._sens_data is None:
            if self._kernel_removed:
                warn_('Performance can be improved by not accessing the data '
                      'attribute before calling this method.')

            # transform source space data directly
            data_t = func(self.data[idx, ..., tmin_idx:tmax_idx])

            if isinstance(data_t, tuple):
                # use only first return value
                data_t = data_t[0]
        else:
            # apply transform in sensor space
            sens_data_t = func(self._sens_data[:, tmin_idx:tmax_idx])

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

    def transform(self, func, idx=None, tmin=None, tmax=None, copy=False):
        """Apply linear transform.

        The transform is applied to each source time course independently.

        Parameters
        ----------
        func : callable
            The transform to be applied, including parameters (see, e.g.,
            :func:`functools.partial`). The first parameter of the function is
            the input data. The first two dimensions of the transformed data
            should be (i) vertices and (ii) time.  See Notes for details.
        idx : array | None
            Indices of source time courses for which to compute transform.
            If None, all time courses are used.
        tmin : float | int | None
            First time point to include (ms). If None, self.tmin is used.
        tmax : float | int | None
            Last time point to include (ms). If None, self.tmax is used.
        copy : bool
            If True, return a new instance of SourceEstimate instead of
            modifying the input inplace.

        Returns
        -------
        stcs : SourceEstimate | VectorSourceEstimate | list
            The transformed stc or, in the case of transforms which yield
            N-dimensional output (where N > 2), a list of stcs. For a list,
            copy must be True.

        Notes
        -----
        Transforms which yield 3D
        output (e.g. time-frequency transforms) are valid, so long as the
        first two dimensions are vertices and time.  In this case, the
        copy parameter must be True and a list of
        SourceEstimates, rather than a single instance of SourceEstimate,
        will be returned, one for each index of the 3rd dimension of the
        transformed data.  In the case of transforms yielding 2D output
        (e.g. filtering), the user has the option of modifying the input
        inplace (copy = False) or returning a new instance of
        SourceEstimate (copy = True) with the transformed data.

        Applying transforms can be significantly faster if the
        SourceEstimate object was created using "(kernel, sens_data)", for
        the "data" parameter as the transform is applied in sensor space.
        Inverse methods, e.g., "apply_inverse_epochs", or "apply_lcmv_epochs"
        do this automatically (if possible).
        """
        # min and max data indices to include
        times = 1000. * self.times
        t_idx = np.where(_time_mask(times, tmin, tmax, sfreq=self.sfreq))[0]
        if tmin is None:
            tmin_idx = None
        else:
            tmin_idx = t_idx[0]

        if tmax is None:
            tmax_idx = None
        else:
            # +1, because upper boundary needs to include the last sample
            tmax_idx = t_idx[-1] + 1

        data_t = self.transform_data(func, idx=idx, tmin_idx=tmin_idx,
                                     tmax_idx=tmax_idx)

        # account for change in n_vertices
        if idx is not None:
            idx_lh = idx[idx < len(self.lh_vertno)]
            idx_rh = idx[idx >= len(self.lh_vertno)] - len(self.lh_vertno)
            verts_lh = self.lh_vertno[idx_lh]
            verts_rh = self.rh_vertno[idx_rh]
        else:
            verts_lh = self.lh_vertno
            verts_rh = self.rh_vertno
        verts = [verts_lh, verts_rh]

        tmin_idx = 0 if tmin_idx is None else tmin_idx
        tmin = self.times[tmin_idx]

        if data_t.ndim > 2:
            # return list of stcs if transformed data has dimensionality > 2
            if copy:
                stcs = [SourceEstimate(data_t[:, :, a], verts, tmin,
                                       self.tstep, self.subject)
                        for a in range(data_t.shape[-1])]
            else:
                raise ValueError('copy must be True if transformed data has '
                                 'more than 2 dimensions')
        else:
            # return new or overwritten stc
            stcs = self if not copy else self.copy()
            stcs.vertices = verts
            stcs.data = data_t
            stcs.tmin = tmin

        return stcs

    @fill_doc
    def to_data_frame(self, index=None, scaling_time=None, scalings=None,
                      long_format=False, time_format='ms'):
        """Export data in tabular structure as a pandas DataFrame.

        Vertices are converted to columns in the DataFrame. By default,
        an additional column "time" is added, unless ``index='time'``
        (in which case time values form the DataFrame's index).

        Parameters
        ----------
        %(df_index_evk)s
            Defaults to ``None``.
        %(df_scaling_time_deprecated)s
        %(df_scalings)s
        %(df_longform_stc)s
        %(df_time_format)s

            .. versionadded:: 0.20

        Returns
        -------
        %(df_return)s
        """
        # check deprecation
        _check_scaling_time(scaling_time)
        # check pandas once here, instead of in each private utils function
        pd = _check_pandas_installed()  # noqa
        # arg checking
        valid_index_args = ['time', 'subject']
        valid_time_formats = ['ms', 'timedelta']
        index = _check_pandas_index_arguments(index, valid_index_args)
        time_format = _check_time_format(time_format, valid_time_formats)
        # get data
        data = self.data.T
        times = self.times
        # prepare extra columns / multiindex
        mindex = list()
        default_index = ['time']
        if self.subject is not None:
            default_index = ['subject', 'time']
            mindex.append(('subject', np.repeat(self.subject, data.shape[0])))
        times = _convert_times(self, times, time_format)
        mindex.append(('time', times))
        # triage surface vs volume source estimates
        if isinstance(self.vertices, list):
            col_names = list()
            for ii, vertno in enumerate(self.vertices):
                col_names.extend(['{}_{}'.format(('LH', 'RH')[ii], vert)
                                  for vert in vertno])
        else:
            col_names = ['VOL_{}'.format(vert) for vert in self.vertices]
        # build DataFrame
        df = _build_data_frame(self, data, None, long_format, mindex, index,
                               default_index=default_index,
                               col_names=col_names, col_kind='source')
        return df


def _center_of_mass(vertices, values, hemi, surf, subject, subjects_dir,
                    restrict_vertices):
    """Find the center of mass on a surface."""
    if (values == 0).all() or (values < 0).any():
        raise ValueError('All values must be non-negative and at least one '
                         'must be non-zero, cannot compute COM')
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    surf = read_surface(op.join(subjects_dir, subject, 'surf',
                                hemi + '.' + surf))
    if restrict_vertices is True:
        restrict_vertices = vertices
    elif restrict_vertices is False:
        restrict_vertices = np.arange(surf[0].shape[0])
    elif isinstance(restrict_vertices, SourceSpaces):
        idx = 1 if restrict_vertices.kind == 'surface' and hemi == 'rh' else 0
        restrict_vertices = restrict_vertices[idx]['vertno']
    else:
        restrict_vertices = np.array(restrict_vertices, int)
    pos = surf[0][vertices, :].T
    c_o_m = np.sum(pos * values, axis=1) / np.sum(values)
    vertex = np.argmin(np.sqrt(np.mean((surf[0][restrict_vertices, :] -
                                        c_o_m) ** 2, axis=1)))
    vertex = restrict_vertices[vertex]
    return vertex


@fill_doc
class _BaseSurfaceSourceEstimate(_BaseSourceEstimate):
    """Abstract base class for surface source estimates.

    Parameters
    ----------
    data : array
        The data in source space.
    vertices : list of array, shape (2,)
        Vertex numbers corresponding to the data. The first element of the list
        contains vertices of left hemisphere and the second element contains
        vertices of right hemisphere.
    tmin : scalar
        Time point of the first sample in data.
    tstep : scalar
        Time step between successive samples in data.
    subject : str | None
        The subject name. While not necessary, it is safer to set the
        subject parameter to avoid analysis errors.
    %(verbose)s

    Attributes
    ----------
    subject : str | None
        The subject name.
    times : array of shape (n_times,)
        The time vector.
    vertices : list of array, shape (2,)
        Vertex numbers corresponding to the data. The first element of the list
        contains vertices of left hemisphere and the second element contains
        vertices of right hemisphere.
    data : array
        The data in source space.
    shape : tuple
        The shape of the data. A tuple of int (n_dipoles, n_times).
    """

    _data_ndim = 2
    _src_type = 'surface'

    @verbose
    def __init__(self, data, vertices=None, tmin=None, tstep=None,
                 subject=None, verbose=None):  # noqa: D102

        if not (isinstance(vertices, list) and len(vertices) == 2):
            raise ValueError('Vertices must be a list containing two '
                             'numpy arrays, got type %s (%s)'
                             % (type(vertices), vertices))

        _BaseSourceEstimate.__init__(self, data, vertices=vertices, tmin=tmin,
                                     tstep=tstep, subject=subject,
                                     verbose=verbose)

    @property
    def lh_data(self):
        """Left hemisphere data."""
        return self.data[:len(self.lh_vertno)]

    @property
    def rh_data(self):
        """Right hemisphere data."""
        return self.data[len(self.lh_vertno):]

    @property
    def lh_vertno(self):
        """Left hemisphere vertno."""
        return self.vertices[0]

    @property
    def rh_vertno(self):
        """Right hemisphere vertno."""
        return self.vertices[1]

    def _hemilabel_stc(self, label):
        if label.hemi == 'lh':
            stc_vertices = self.vertices[0]
        else:
            stc_vertices = self.vertices[1]

        # find index of the Label's vertices
        idx = np.nonzero(np.in1d(stc_vertices, label.vertices))[0]

        # find output vertices
        vertices = stc_vertices[idx]

        # find data
        if label.hemi == 'rh':
            values = self.data[idx + len(self.vertices[0])]
        else:
            values = self.data[idx]

        return vertices, values

    def in_label(self, label):
        """Get a source estimate object restricted to a label.

        SourceEstimate contains the time course of
        activation of all sources inside the label.

        Parameters
        ----------
        label : Label | BiHemiLabel
            The label (as created for example by mne.read_label). If the label
            does not match any sources in the SourceEstimate, a ValueError is
            raised.

        Returns
        -------
        stc : SourceEstimate | VectorSourceEstimate
            The source estimate restricted to the given label.
        """
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
            vertices = [lh_vert, np.array([], int)]
        elif label.hemi == 'rh':
            rh_vert, values = self._hemilabel_stc(label)
            vertices = [np.array([], int), rh_vert]
        else:
            raise TypeError("Expected  Label or BiHemiLabel; got %r" % label)

        if sum([len(v) for v in vertices]) == 0:
            raise ValueError('No vertices match the label in the stc file')

        label_stc = self.__class__(values, vertices=vertices, tmin=self.tmin,
                                   tstep=self.tstep, subject=self.subject)
        return label_stc

    def expand(self, vertices):
        """Expand SourceEstimate to include more vertices.

        This will add rows to stc.data (zero-filled) and modify stc.vertices
        to include all vertices in stc.vertices and the input vertices.

        Parameters
        ----------
        vertices : list of array
            New vertices to add. Can also contain old values.

        Returns
        -------
        stc : SourceEstimate | VectorSourceEstimate
            The modified stc (note: method operates inplace).
        """
        if not isinstance(vertices, list):
            raise TypeError('vertices must be a list')
        if not len(self.vertices) == len(vertices):
            raise ValueError('vertices must have the same length as '
                             'stc.vertices')

        # can no longer use kernel and sensor data
        self._remove_kernel_sens_data_()

        inserters = list()
        offsets = [0]
        for vi, (v_old, v_new) in enumerate(zip(self.vertices, vertices)):
            v_new = np.setdiff1d(v_new, v_old)
            inds = np.searchsorted(v_old, v_new)
            # newer numpy might overwrite inds after np.insert, copy here
            inserters += [inds.copy()]
            offsets += [len(v_old)]
            self.vertices[vi] = np.insert(v_old, inds, v_new)
        inds = [ii + offset for ii, offset in zip(inserters, offsets[:-1])]
        inds = np.concatenate(inds)
        new_data = np.zeros((len(inds),) + self.data.shape[1:])
        self.data = np.insert(self.data, inds, new_data, axis=0)
        return self

    @verbose
    def to_original_src(self, src_orig, subject_orig=None,
                        subjects_dir=None, verbose=None):
        """Get a source estimate from morphed source to the original subject.

        Parameters
        ----------
        src_orig : instance of SourceSpaces
            The original source spaces that were morphed to the current
            subject.
        subject_orig : str | None
            The original subject. For most source spaces this shouldn't need
            to be provided, since it is stored in the source space itself.
        %(subjects_dir)s
        %(verbose_meth)s

        Returns
        -------
        stc : SourceEstimate | VectorSourceEstimate
            The transformed source estimate.

        See Also
        --------
        morph_source_spaces

        Notes
        -----
        .. versionadded:: 0.10.0
        """
        if self.subject is None:
            raise ValueError('stc.subject must be set')
        src_orig = _ensure_src(src_orig, kind='surface')
        subject_orig = _ensure_src_subject(src_orig, subject_orig)
        data_idx, vertices = _get_morph_src_reordering(
            self.vertices, src_orig, subject_orig, self.subject, subjects_dir)
        return self.__class__(self._data[data_idx], vertices,
                              self.tmin, self.tstep, subject_orig)


@fill_doc
class SourceEstimate(_BaseSurfaceSourceEstimate):
    """Container for surface source estimates.

    Parameters
    ----------
    data : array of shape (n_dipoles, n_times) | tuple, shape (2,)
        The data in source space. When it is a single array, the
        left hemisphere is stored in data[:len(vertices[0])] and the right
        hemisphere is stored in data[-len(vertices[1]):].
        When data is a tuple, it contains two arrays:

        - "kernel" shape (n_vertices, n_sensors) and
        - "sens_data" shape (n_sensors, n_times).

        In this case, the source space data corresponds to
        ``np.dot(kernel, sens_data)``.
    vertices : list of array, shape (2,)
        Vertex numbers corresponding to the data. The first element of the list
        contains vertices of left hemisphere and the second element contains
        vertices of right hemisphere.
    tmin : scalar
        Time point of the first sample in data.
    tstep : scalar
        Time step between successive samples in data.
    subject : str | None
        The subject name. While not necessary, it is safer to set the
        subject parameter to avoid analysis errors.
    %(verbose)s

    Attributes
    ----------
    subject : str | None
        The subject name.
    times : array of shape (n_times,)
        The time vector.
    vertices : list of array, shape (2,)
        The indices of the dipoles in the left and right source space.
    data : array of shape (n_dipoles, n_times)
        The data in source space.
    shape : tuple
        The shape of the data. A tuple of int (n_dipoles, n_times).

    See Also
    --------
    VectorSourceEstimate : A container for vector source estimates.
    VolSourceEstimate : A container for volume source estimates.
    MixedSourceEstimate : A container for mixed surface + volume source
                          estimates.
    """

    @verbose
    def save(self, fname, ftype='stc', verbose=None):
        """Save the source estimates to a file.

        Parameters
        ----------
        fname : str
            The stem of the file name. The file names used for surface source
            spaces are obtained by adding "-lh.stc" and "-rh.stc" (or "-lh.w"
            and "-rh.w") to the stem provided, for the left and the right
            hemisphere, respectively.
        ftype : str
            File format to use. Allowed values are "stc" (default), "w",
            and "h5". The "w" format only supports a single time point.
        %(verbose_meth)s
        """
        _validate_type(fname, 'path-like', 'fname')
        fname = str(fname)
        _check_option('ftype', ftype, ['stc', 'w', 'h5'])

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

        elif ftype == 'h5':
            super().save(fname)
        logger.info('[done]')

    @copy_function_doc_to_method_doc(plot_source_estimates)
    def plot(self, subject=None, surface='inflated', hemi='lh',
             colormap='auto', time_label='auto', smoothing_steps=10,
             transparent=True, alpha=1.0, time_viewer='auto',
             subjects_dir=None,
             figure=None, views='lat', colorbar=True, clim='auto',
             cortex="classic", size=800, background="black",
             foreground="white", initial_time=None, time_unit='s',
             backend='auto', spacing='oct6', title=None,
             show_traces='auto', verbose=None):
        brain = plot_source_estimates(
            self, subject, surface=surface, hemi=hemi, colormap=colormap,
            time_label=time_label, smoothing_steps=smoothing_steps,
            transparent=transparent, alpha=alpha, time_viewer=time_viewer,
            subjects_dir=subjects_dir, figure=figure, views=views,
            colorbar=colorbar, clim=clim, cortex=cortex, size=size,
            background=background, foreground=foreground,
            initial_time=initial_time, time_unit=time_unit, backend=backend,
            spacing=spacing, title=title, show_traces=show_traces,
            verbose=verbose)
        return brain

    @verbose
    def extract_label_time_course(self, labels, src, mode='mean_flip',
                                  allow_empty=False, verbose=None):
        """Extract label time courses for lists of labels.

        This function will extract one time course for each label. The way the
        time courses are extracted depends on the mode parameter.

        Parameters
        ----------
        %(eltc_labels)s
        %(eltc_src)s
        %(eltc_mode)s
        %(eltc_allow_empty)s
        %(verbose_meth)s

        Returns
        -------
        label_tc : array, shape=(n_labels, n_times)
            Extracted time course for each label.

        See Also
        --------
        extract_label_time_course : Extract time courses for multiple STCs.

        Notes
        -----
        %(eltc_mode_notes)s
        """
        label_tc = extract_label_time_course(
            self, labels, src, mode=mode, return_generator=False,
            allow_empty=allow_empty, verbose=verbose)

        return label_tc

    @verbose
    def estimate_snr(self, info, fwd, cov, verbose=None):
        r"""Compute time-varying SNR in the source space.

        This function should only be used with source estimates with units
        nanoAmperes (i.e., MNE-like solutions, *not* dSPM or sLORETA).

        .. warning:: This function currently only works properly for fixed
                     orientation.

        Parameters
        ----------
        info : instance Info
            The measurement info.
        fwd : instance of Forward
            The forward solution used to create the source estimate.
        cov : instance of Covariance
            The noise covariance used to estimate the resting cortical
            activations. Should be an evoked covariance, not empty room.
        %(verbose)s

        Returns
        -------
        snr_stc : instance of SourceEstimate
            The source estimate with the SNR computed.

        Notes
        -----
        We define the SNR in decibels for each source location at each
        time point as:

        .. math::

            {\rm SNR} = 10\log_10[\frac{a^2}{N}\sum_k\frac{b_k^2}{s_k^2}]

        where :math:`\\b_k` is the signal on sensor :math:`k` provided by the
        forward model for a source with unit amplitude, :math:`a` is the
        source amplitude, :math:`N` is the number of sensors, and
        :math:`s_k^2` is the noise variance on sensor :math:`k`.

        References
        ----------
        .. [1] Goldenholz, D. M., Ahlfors, S. P., Hämäläinen, M. S., Sharon,
               D., Ishitobi, M., Vaina, L. M., & Stufflebeam, S. M. (2009).
               Mapping the Signal-To-Noise-Ratios of Cortical Sources in
               Magnetoencephalography and Electroencephalography.
               Human Brain Mapping, 30(4), 1077–1086. doi:10.1002/hbm.20571
        """
        from .forward import convert_forward_solution, Forward
        from .minimum_norm.inverse import _prepare_forward
        _validate_type(fwd, Forward, 'fwd')
        _validate_type(info, Info, 'info')
        _validate_type(cov, Covariance, 'cov')
        _check_stc_units(self)
        if (self.data >= 0).all():
            warn_('This STC appears to be from free orientation, currently SNR'
                  ' function is valid only for fixed orientation')

        fwd = convert_forward_solution(fwd, surf_ori=True, force_fixed=False)

        # G is gain matrix [ch x src], cov is noise covariance [ch x ch]
        G, _, _, _, _, _, _, cov, _ = _prepare_forward(
            fwd, info, cov, fixed=True, loose=0, rank=None, pca=False,
            use_cps=True, exp=None, limit_depth_chs=False, combine_xyz='fro',
            allow_fixed_depth=False, limit=None)
        G = G['sol']['data']
        n_channels = cov['dim']  # number of sensors/channels
        b_k2 = (G * G).T
        s_k2 = np.diag(cov['data'])
        scaling = (1 / n_channels) * np.sum(b_k2 / s_k2, axis=1, keepdims=True)
        snr_stc = self.copy()
        snr_stc._data[:] = 10 * np.log10((self.data * self.data) * scaling)
        return snr_stc

    def get_peak(self, hemi=None, tmin=None, tmax=None, mode='abs',
                 vert_as_index=False, time_as_index=False):
        """Get location and latency of peak amplitude.

        Parameters
        ----------
        hemi : {'lh', 'rh', None}
            The hemi to be considered. If None, the entire source space is
            considered.
        tmin : float | None
            The minimum point in time to be considered for peak getting.
        tmax : float | None
            The maximum point in time to be considered for peak getting.
        mode : {'pos', 'neg', 'abs'}
            How to deal with the sign of the data. If 'pos' only positive
            values will be considered. If 'neg' only negative values will
            be considered. If 'abs' absolute values will be considered.
            Defaults to 'abs'.
        vert_as_index : bool
            Whether to return the vertex index instead of of its ID.
            Defaults to False.
        time_as_index : bool
            Whether to return the time index instead of the latency.
            Defaults to False.

        Returns
        -------
        pos : int
            The vertex exhibiting the maximum response, either ID or index.
        latency : float | int
            The time point of the maximum response, either latency in seconds
            or index.
        """
        data = {'lh': self.lh_data, 'rh': self.rh_data, None: self.data}[hemi]
        vertno = {'lh': self.lh_vertno, 'rh': self.rh_vertno,
                  None: np.concatenate(self.vertices)}[hemi]

        vert_idx, time_idx, _ = _get_peak(data, self.times, tmin, tmax, mode)

        return (vert_idx if vert_as_index else vertno[vert_idx],
                time_idx if time_as_index else self.times[time_idx])

    @fill_doc
    def center_of_mass(self, subject=None, hemi=None, restrict_vertices=False,
                       subjects_dir=None, surf='sphere'):
        """Compute the center of mass of activity.

        This function computes the spatial center of mass on the surface
        as well as the temporal center of mass as in [1]_.

        .. note:: All activity must occur in a single hemisphere, otherwise
                  an error is raised. The "mass" of each point in space for
                  computing the spatial center of mass is computed by summing
                  across time, and vice-versa for each point in time in
                  computing the temporal center of mass. This is useful for
                  quantifying spatio-temporal cluster locations, especially
                  when combined with :func:`mne.vertex_to_mni`.

        Parameters
        ----------
        subject : str | None
            The subject the stc is defined for.
        hemi : int, or None
            Calculate the center of mass for the left (0) or right (1)
            hemisphere. If None, one of the hemispheres must be all zeroes,
            and the center of mass will be calculated for the other
            hemisphere (useful for getting COM for clusters).
        restrict_vertices : bool | array of int | instance of SourceSpaces
            If True, returned vertex will be one from stc. Otherwise, it could
            be any vertex from surf. If an array of int, the returned vertex
            will come from that array. If instance of SourceSpaces (as of
            0.13), the returned vertex will be from the given source space.
            For most accuruate estimates, do not restrict vertices.
        %(subjects_dir)s
        surf : str
            The surface to use for Euclidean distance center of mass
            finding. The default here is "sphere", which finds the center
            of mass on the spherical surface to help avoid potential issues
            with cortical folding.

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

        See Also
        --------
        mne.Label.center_of_mass
        mne.vertex_to_mni

        References
        ----------
        .. [1] Larson and Lee, "The cortical dynamics underlying effective
               switching of auditory spatial attention", NeuroImage 2012.
        """
        if not isinstance(surf, str):
            raise TypeError('surf must be a string, got %s' % (type(surf),))
        subject = _check_subject(self.subject, subject)
        if np.any(self.data < 0):
            raise ValueError('Cannot compute COM with negative values')
        values = np.sum(self.data, axis=1)  # sum across time
        vert_inds = [np.arange(len(self.vertices[0])),
                     np.arange(len(self.vertices[1])) + len(self.vertices[0])]
        if hemi is None:
            hemi = np.where(np.array([np.sum(values[vi])
                                      for vi in vert_inds]))[0]
            if not len(hemi) == 1:
                raise ValueError('Could not infer hemisphere')
            hemi = hemi[0]
        _check_option('hemi', hemi, [0, 1])
        vertices = self.vertices[hemi]
        values = values[vert_inds[hemi]]  # left or right
        del vert_inds
        vertex = _center_of_mass(
            vertices, values, hemi=['lh', 'rh'][hemi], surf=surf,
            subject=subject, subjects_dir=subjects_dir,
            restrict_vertices=restrict_vertices)
        # do time center of mass by using the values across space
        masses = np.sum(self.data, axis=0).astype(float)
        t_ind = np.sum(masses * np.arange(self.shape[1])) / np.sum(masses)
        t = self.tmin + self.tstep * t_ind
        return vertex, hemi, t


class _BaseVectorSourceEstimate(_BaseSourceEstimate):
    _data_ndim = 3

    @verbose
    def __init__(self, data, vertices=None, tmin=None, tstep=None,
                 subject=None, verbose=None):  # noqa: D102
        assert hasattr(self, '_scalar_class')
        super().__init__(data, vertices, tmin, tstep, subject, verbose)
        if self._data is not None and self._data.shape[1] != 3:
            raise ValueError('Data for VectorSourceEstimate must have second '
                             'dimension of length 3, got length %s'
                             % (self._data.shape[1],))

    def magnitude(self):
        """Compute magnitude of activity without directionality.

        Returns
        -------
        stc : instance of SourceEstimate
            The source estimate without directionality information.
        """
        data_mag = np.linalg.norm(self.data, axis=1)
        return self._scalar_class(
            data_mag, self.vertices, self.tmin, self.tstep, self.subject,
            self.verbose)

    @fill_doc
    def normal(self, src, use_cps=True):
        """Compute activity orthogonal to the cortex.

        Parameters
        ----------
        src : instance of SourceSpaces
            The source space for which this source estimate is specified.
        %(use_cps)s
            Should be the same value that was used when the forward model
            was computed (typically True).

            .. versionadded:: 0.20

        Returns
        -------
        stc : instance of SourceEstimate
            The source estimate only retaining the activity orthogonal to the
            cortex.
        """
        _check_src_normal('normal', src)
        normals = np.vstack([_get_src_nn(s, use_cps, v) for s, v in
                             zip(src, self._vertices_list)])
        data_norm = einsum('ijk,ij->ik', self.data, normals)
        return self._scalar_class(
            data_norm, self.vertices, self.tmin, self.tstep, self.subject,
            self.verbose)


class _BaseVolSourceEstimate(_BaseSourceEstimate):

    _data_ndim = 2
    _src_type = 'volume'

    @verbose
    def __init__(self, data, vertices=None, tmin=None, tstep=None,
                 subject=None, verbose=None):  # noqa: D102
        _validate_type(vertices, (np.ndarray, list), 'vertices')
        _BaseSourceEstimate.__init__(self, data, vertices=vertices, tmin=tmin,
                                     tstep=tstep, subject=subject,
                                     verbose=verbose)

    @property
    def _vertices_list(self):
        return [self.vertices]

    @copy_function_doc_to_method_doc(plot_volume_source_estimates)
    def plot(self, src, subject=None, subjects_dir=None, mode='stat_map',
             bg_img=None, colorbar=True, colormap='auto', clim='auto',
             transparent='auto', show=True, initial_time=None,
             initial_pos=None, verbose=None):
        data = self.magnitude() if self._data_ndim == 3 else self
        return plot_volume_source_estimates(
            data, src=src, subject=subject, subjects_dir=subjects_dir,
            mode=mode, bg_img=bg_img, colorbar=colorbar, colormap=colormap,
            clim=clim, transparent=transparent, show=show,
            initial_time=initial_time, initial_pos=initial_pos,
            verbose=verbose)

    def save_as_volume(self, fname, src, dest='mri', mri_resolution=False,
                       format='nifti1'):
        """Save a volume source estimate in a NIfTI file.

        Parameters
        ----------
        fname : str
            The name of the generated nifti file.
        src : list
            The list of source spaces (should all be of type volume).
        dest : 'mri' | 'surf'
            If 'mri' the volume is defined in the coordinate system of
            the original T1 image. If 'surf' the coordinate system
            of the FreeSurfer surface is used (Surface RAS).
        mri_resolution : bool
            It True the image is saved in MRI resolution.

            .. warning:: If you have many time points, the file produced can be
                         huge.
        format : str
            Either 'nifti1' (default) or 'nifti2'.

            .. versionadded:: 0.17

        Returns
        -------
        img : instance Nifti1Image
            The image object.

        Notes
        -----
        .. versionadded:: 0.9.0
        """
        import nibabel as nib
        _validate_type(fname, 'path-like', 'fname')
        fname = str(fname)
        img = self.as_volume(src, dest=dest, mri_resolution=mri_resolution,
                             format=format)
        nib.save(img, fname)

    def as_volume(self, src, dest='mri', mri_resolution=False,
                  format='nifti1'):
        """Export volume source estimate as a nifti object.

        Parameters
        ----------
        src : list
            The list of source spaces (should all be of type volume).
        dest : 'mri' | 'surf'
            If 'mri' the volume is defined in the coordinate system of
            the original T1 image. If 'surf' the coordinate system
            of the FreeSurfer surface is used (Surface RAS).
        mri_resolution : bool
            It True the image is saved in MRI resolution.

            .. warning:: If you have many time points, the file produced can be
                         huge.
        format : str
            Either 'nifti1' (default) or 'nifti2'.

        Returns
        -------
        img : instance of Nifti1Image
            The image object.

        Notes
        -----
        .. versionadded:: 0.9.0
        """
        from .morph import _interpolate_data
        data = self.magnitude() if self._data_ndim == 3 else self
        return _interpolate_data(data, src, mri_resolution=mri_resolution,
                                 mri_space=True, output=format)

    def get_peak(self, tmin=None, tmax=None, mode='abs',
                 vert_as_index=False, time_as_index=False):
        """Get location and latency of peak amplitude.

        Parameters
        ----------
        tmin : float | None
            The minimum point in time to be considered for peak getting.
        tmax : float | None
            The maximum point in time to be considered for peak getting.
        mode : {'pos', 'neg', 'abs'}
            How to deal with the sign of the data. If 'pos' only positive
            values will be considered. If 'neg' only negative values will
            be considered. If 'abs' absolute values will be considered.
            Defaults to 'abs'.
        vert_as_index : bool
            Whether to return the vertex index instead of of its ID.
            Defaults to False.
        time_as_index : bool
            Whether to return the time index instead of the latency.
            Defaults to False.

        Returns
        -------
        pos : int
            The vertex exhibiting the maximum response, either ID or index.
        latency : float
            The latency in seconds.
        """
        stc = self.magnitude() if self._data_ndim == 3 else self
        vert_idx, time_idx, _ = _get_peak(stc.data, self.times, tmin, tmax,
                                          mode)

        return (vert_idx if vert_as_index else self.vertices[vert_idx],
                time_idx if time_as_index else self.times[time_idx])


@fill_doc
class VolSourceEstimate(_BaseVolSourceEstimate):
    """Container for volume source estimates.

    Parameters
    ----------
    data : array of shape (n_dipoles, n_times) | tuple, shape (2,)
        The data in source space. The data can either be a single array or
        a tuple with two arrays: "kernel" shape (n_vertices, n_sensors) and
        "sens_data" shape (n_sensors, n_times). In this case, the source
        space data corresponds to ``np.dot(kernel, sens_data)``.
    vertices : array of shape (n_dipoles,)
        The indices of the dipoles in the source space.
    tmin : scalar
        Time point of the first sample in data.
    tstep : scalar
        Time step between successive samples in data.
    subject : str | None
        The subject name. While not necessary, it is safer to set the
        subject parameter to avoid analysis errors.
    %(verbose)s

    Attributes
    ----------
    subject : str | None
        The subject name.
    times : array of shape (n_times,)
        The time vector.
    vertices : array of shape (n_dipoles,)
        The indices of the dipoles in the source space.
    data : array of shape (n_dipoles, n_times)
        The data in source space.
    shape : tuple
        The shape of the data. A tuple of int (n_dipoles, n_times).

    See Also
    --------
    SourceEstimate : A container for surface source estimates.
    VolVectorSourceEstimate : A container for volume vector source estimates.
    MixedSourceEstimate : A container for mixed surface + volume source
                          estimates.

    Notes
    -----
    .. versionadded:: 0.9.0
    """

    @verbose
    def save(self, fname, ftype='stc', verbose=None):
        """Save the source estimates to a file.

        Parameters
        ----------
        fname : str
            The stem of the file name. The stem is extended with "-vl.stc"
            or "-vl.w".
        ftype : str
            File format to use. Allowed values are "stc" (default), "w",
            and "h5". The "w" format only supports a single time point.
        %(verbose_meth)s
        """
        _validate_type(fname, 'path-like', 'fname')
        fname = str(fname)
        _check_option('ftype', ftype, ['stc', 'w', 'h5'])
        if ftype == 'stc':
            logger.info('Writing STC to disk...')
            if not (fname.endswith('-vl.stc') or fname.endswith('-vol.stc')):
                fname += '-vl.stc'
            _write_stc(fname, tmin=self.tmin, tstep=self.tstep,
                       vertices=self.vertices, data=self.data)
        elif ftype == 'w':
            logger.info('Writing STC to disk (w format)...')
            if not (fname.endswith('-vl.w') or fname.endswith('-vol.w')):
                fname += '-vl.w'
            _write_w(fname, vertices=self.vertices, data=self.data)
        elif ftype == 'h5':
            super().save(fname, 'h5')
        logger.info('[done]')


@fill_doc
class VolVectorSourceEstimate(_BaseVectorSourceEstimate,
                              _BaseVolSourceEstimate):
    """Container for volume source estimates.

    Parameters
    ----------
    data : array of shape (n_dipoles, 3, n_times)
        The data in source space. Each dipole contains three vectors that
        denote the dipole strength in X, Y and Z directions over time.
    vertices : array of shape (n_dipoles,)
        The indices of the dipoles in the source space.
    tmin : scalar
        Time point of the first sample in data.
    tstep : scalar
        Time step between successive samples in data.
    subject : str | None
        The subject name. While not necessary, it is safer to set the
        subject parameter to avoid analysis errors.
    %(verbose)s

    Attributes
    ----------
    subject : str | None
        The subject name.
    times : array of shape (n_times,)
        The time vector.
    vertices : array of shape (n_dipoles,)
        The indices of the dipoles in the source space.
    data : array of shape (n_dipoles, n_times)
        The data in source space.
    shape : tuple
        The shape of the data. A tuple of int (n_dipoles, n_times).

    See Also
    --------
    SourceEstimate : A container for surface source estimates.
    VectorSourceEstimate : A container for vector source estimates.
    MixedSourceEstimate : A container for mixed surface + volume source
                          estimates.

    Notes
    -----
    .. versionadded:: 0.9.0
    """

    _data_ndim = 3
    _scalar_class = VolSourceEstimate


@fill_doc
class VectorSourceEstimate(_BaseVectorSourceEstimate,
                           _BaseSurfaceSourceEstimate):
    """Container for vector surface source estimates.

    For each vertex, the magnitude of the current is defined in the X, Y and Z
    directions.

    Parameters
    ----------
    data : array of shape (n_dipoles, 3, n_times)
        The data in source space. Each dipole contains three vectors that
        denote the dipole strength in X, Y and Z directions over time.
    vertices : list of array, shape (2,)
        Vertex numbers corresponding to the data. The first element of the list
        contains vertices of left hemisphere and the second element contains
        vertices of right hemisphere.
    tmin : float
        Time point of the first sample in data.
    tstep : float
        Time step between successive samples in data.
    subject : str | None
        The subject name. While not necessary, it is safer to set the
        subject parameter to avoid analysis errors.
    %(verbose)s

    Attributes
    ----------
    subject : str | None
        The subject name.
    times : array of shape (n_times,)
        The time vector.
    shape : tuple
        The shape of the data. A tuple of int (n_dipoles, n_times).

    See Also
    --------
    SourceEstimate : A container for surface source estimates.
    VolSourceEstimate : A container for volume source estimates.
    MixedSourceEstimate : A container for mixed surface + volume source
                          estimates.

    Notes
    -----
    .. versionadded:: 0.15
    """

    _data_ndim = 3
    _scalar_class = SourceEstimate

    @copy_function_doc_to_method_doc(plot_vector_source_estimates)
    def plot(self, subject=None, hemi='lh', colormap='hot', time_label='auto',
             smoothing_steps=10, transparent=True, brain_alpha=0.4,
             overlay_alpha=None, vector_alpha=1.0, scale_factor=None,
             time_viewer='auto', subjects_dir=None, figure=None, views='lat',
             colorbar=True, clim='auto', cortex='classic', size=800,
             background='black', foreground='white', initial_time=None,
             time_unit='s', show_traces='auto', verbose=None):  # noqa: D102

        return plot_vector_source_estimates(
            self, subject=subject, hemi=hemi, colormap=colormap,
            time_label=time_label, smoothing_steps=smoothing_steps,
            transparent=transparent, brain_alpha=brain_alpha,
            overlay_alpha=overlay_alpha, vector_alpha=vector_alpha,
            scale_factor=scale_factor, time_viewer=time_viewer,
            subjects_dir=subjects_dir, figure=figure, views=views,
            colorbar=colorbar, clim=clim, cortex=cortex, size=size,
            background=background, foreground=foreground,
            initial_time=initial_time, time_unit=time_unit,
            show_traces=show_traces, verbose=verbose,
        )


@fill_doc
class MixedSourceEstimate(_BaseSourceEstimate):
    """Container for mixed surface and volume source estimates.

    Parameters
    ----------
    data : array of shape (n_dipoles, n_times) | tuple, shape (2,)
        The data in source space. The data can either be a single array or
        a tuple with two arrays: "kernel" shape (n_vertices, n_sensors) and
        "sens_data" shape (n_sensors, n_times). In this case, the source
        space data corresponds to ``np.dot(kernel, sens_data)``.
    vertices : list of array
        Vertex numbers corresponding to the data. The list contains arrays
        with one array per source space.
    tmin : scalar
        Time point of the first sample in data.
    tstep : scalar
        Time step between successive samples in data.
    subject : str | None
        The subject name. While not necessary, it is safer to set the
        subject parameter to avoid analysis errors.
    %(verbose)s

    Attributes
    ----------
    subject : str | None
        The subject name.
    times : array of shape (n_times,)
        The time vector.
    vertices : list of array
        Vertex numbers corresponding to the data. The list contains arrays
        with one array per source space.
    data : array of shape (n_dipoles, n_times)
        The data in source space.
    shape : tuple
        The shape of the data. A tuple of int (n_dipoles, n_times).

    See Also
    --------
    SourceEstimate : A container for surface source estimates.
    VectorSourceEstimate : A container for vector source estimates.
    VolSourceEstimate : A container for volume source estimates.
    VolVectorSourceEstimate : A container for Volume vector source estimates.

    Notes
    -----
    .. versionadded:: 0.9.0
    """

    _data_ndim = 2
    _src_type = 'mixed'

    @verbose
    def __init__(self, data, vertices=None, tmin=None, tstep=None,
                 subject=None, verbose=None):  # noqa: D102
        if not isinstance(vertices, list) or len(vertices) < 2:
            raise ValueError('Vertices must be a list of numpy arrays with '
                             'one array per source space.')

        _BaseSourceEstimate.__init__(self, data, vertices=vertices, tmin=tmin,
                                     tstep=tstep, subject=subject,
                                     verbose=verbose)

    @fill_doc
    def plot_surface(self, src, subject=None, surface='inflated', hemi='lh',
                     colormap='auto', time_label='time=%02.f ms',
                     smoothing_steps=10,
                     transparent=None, alpha=1.0, time_viewer='auto',
                     subjects_dir=None, figure=None,
                     views='lat', colorbar=True, clim='auto'):
        """Plot surface source estimates with PySurfer.

        Note: PySurfer currently needs the SUBJECTS_DIR environment variable,
        which will automatically be set by this function. Plotting multiple
        SourceEstimates with different values for subjects_dir will cause
        PySurfer to use the wrong FreeSurfer surfaces when using methods of
        the returned Brain object. It is therefore recommended to set the
        SUBJECTS_DIR environment variable or always use the same value for
        subjects_dir (within the same Python session).

        Parameters
        ----------
        src : SourceSpaces
            The source spaces to plot.
        subject : str | None
            The subject name corresponding to FreeSurfer environment
            variable SUBJECT. If None stc.subject will be used. If that
            is None, the environment will be used.
        surface : str
            The type of surface (inflated, white etc.).
        hemi : str, 'lh' | 'rh' | 'split' | 'both'
            The hemisphere to display. Using 'both' or 'split' requires
            PySurfer version 0.4 or above.
        colormap : str | np.ndarray of float, shape(n_colors, 3 | 4)
            Name of colormap to use. See `plot_source_estimates`.
        time_label : str
            How to print info about the time instant visualized.
        smoothing_steps : int
            The amount of smoothing.
        transparent : bool | None
            If True, use a linear transparency between fmin and fmid.
            None will choose automatically based on colormap type.
        alpha : float
            Alpha value to apply globally to the overlay.
        time_viewer : bool
            Display time viewer GUI.
        %(subjects_dir)s
        figure : instance of mayavi.mlab.Figure | None
            If None, the last figure will be cleaned and a new figure will
            be created.
        views : str | list
            View to use. See `surfer.Brain`.
        colorbar : bool
            If True, display colorbar on scene.
        clim : str | dict
            Colorbar properties specification. See `plot_source_estimates`.

        Returns
        -------
        brain : instance of surfer.Brain
            A instance of `surfer.Brain` from PySurfer.
        """
        # extract surface source spaces
        surf = _ensure_src(src, kind='surface')

        # extract surface source estimate
        data = self.data[:surf[0]['nuse'] + surf[1]['nuse']]
        vertices = [s['vertno'] for s in surf]

        stc = SourceEstimate(data, vertices, self.tmin, self.tstep,
                             self.subject, self.verbose)

        return plot_source_estimates(stc, subject, surface=surface, hemi=hemi,
                                     colormap=colormap, time_label=time_label,
                                     smoothing_steps=smoothing_steps,
                                     transparent=transparent, alpha=alpha,
                                     time_viewer=time_viewer,
                                     subjects_dir=subjects_dir, figure=figure,
                                     views=views, colorbar=colorbar, clim=clim)


###############################################################################
# Morphing


def _get_vol_mask(src):
    """Get the volume source space mask."""
    assert len(src) == 1  # not a mixed source space
    shape = src[0]['shape'][::-1]
    mask = np.zeros(shape, bool)
    mask.flat[src[0]['vertno']] = True
    return mask


def _spatio_temporal_src_connectivity_vol(src, n_times):
    from sklearn.feature_extraction import grid_to_graph
    mask = _get_vol_mask(src)
    edges = grid_to_graph(*mask.shape, mask=mask)
    connectivity = _get_connectivity_from_edges(edges, n_times)
    return connectivity


def _spatio_temporal_src_connectivity_surf(src, n_times):
    if src[0]['use_tris'] is None:
        # XXX It would be nice to support non oct source spaces too...
        raise RuntimeError("The source space does not appear to be an ico "
                           "surface. Connectivity cannot be extracted from"
                           " non-ico source spaces.")
    used_verts = [np.unique(s['use_tris']) for s in src]
    offs = np.cumsum([0] + [len(u_v) for u_v in used_verts])[:-1]
    tris = np.concatenate([np.searchsorted(u_v, s['use_tris']) + off
                           for u_v, s, off in zip(used_verts, src, offs)])
    connectivity = spatio_temporal_tris_connectivity(tris, n_times)

    # deal with source space only using a subset of vertices
    masks = [np.in1d(u, s['vertno']) for s, u in zip(src, used_verts)]
    if sum(u.size for u in used_verts) != connectivity.shape[0] / n_times:
        raise ValueError('Used vertices do not match connectivity shape')
    if [np.sum(m) for m in masks] != [len(s['vertno']) for s in src]:
        raise ValueError('Vertex mask does not match number of vertices')
    masks = np.concatenate(masks)
    missing = 100 * float(len(masks) - np.sum(masks)) / len(masks)
    if missing:
        warn_('%0.1f%% of original source space vertices have been'
              ' omitted, tri-based connectivity will have holes.\n'
              'Consider using distance-based connectivity or '
              'morphing data to all source space vertices.' % missing)
        masks = np.tile(masks, n_times)
        masks = np.where(masks)[0]
        connectivity = connectivity.tocsr()
        connectivity = connectivity[masks]
        connectivity = connectivity[:, masks]
        # return to original format
        connectivity = connectivity.tocoo()
    return connectivity


@verbose
def spatio_temporal_src_connectivity(src, n_times, dist=None, verbose=None):
    """Compute connectivity for a source space activation over time.

    Parameters
    ----------
    src : instance of SourceSpaces
        The source space. It can be a surface source space or a
        volume source space.
    n_times : int
        Number of time instants.
    dist : float, or None
        Maximal geodesic distance (in m) between vertices in the
        source space to consider neighbors. If None, immediate neighbors
        are extracted from an ico surface.
    %(verbose)s

    Returns
    -------
    connectivity : ~scipy.sparse.coo_matrix
        The connectivity matrix describing the spatio-temporal
        graph structure. If N is the number of vertices in the
        source space, the N first nodes in the graph are the
        vertices are time 1, the nodes from 2 to 2N are the vertices
        during time 2, etc.
    """
    # XXX we should compute connectivity for each source space and then
    # use scipy.sparse.block_diag to concatenate them
    if src[0]['type'] == 'vol':
        if dist is not None:
            raise ValueError('dist must be None for a volume '
                             'source space. Got %s.' % dist)

        connectivity = _spatio_temporal_src_connectivity_vol(src, n_times)
    elif dist is not None:
        # use distances computed and saved in the source space file
        connectivity = spatio_temporal_dist_connectivity(src, n_times, dist)
    else:
        connectivity = _spatio_temporal_src_connectivity_surf(src, n_times)
    return connectivity


@verbose
def grade_to_tris(grade, verbose=None):
    """Get tris defined for a certain grade.

    Parameters
    ----------
    grade : int
        Grade of an icosahedral mesh.
    %(verbose)s

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
def spatio_temporal_tris_connectivity(tris, n_times, remap_vertices=False,
                                      verbose=None):
    """Compute connectivity from triangles and time instants.

    Parameters
    ----------
    tris : array
        N x 3 array defining triangles.
    n_times : int
        Number of time points.
    remap_vertices : bool
        Reassign vertex indices based on unique values. Useful
        to process a subset of triangles. Defaults to False.
    %(verbose)s

    Returns
    -------
    connectivity : ~scipy.sparse.coo_matrix
        The connectivity matrix describing the spatio-temporal
        graph structure. If N is the number of vertices in the
        source space, the N first nodes in the graph are the
        vertices are time 1, the nodes from 2 to 2N are the vertices
        during time 2, etc.
    """
    if remap_vertices:
        logger.info('Reassigning vertex indices.')
        tris = np.searchsorted(np.unique(tris), tris)

    edges = mesh_edges(tris).tocoo()
    return _get_connectivity_from_edges(edges, n_times)


@verbose
def spatio_temporal_dist_connectivity(src, n_times, dist, verbose=None):
    """Compute connectivity from distances in a source space and time instants.

    Parameters
    ----------
    src : instance of SourceSpaces
        The source space must have distances between vertices computed, such
        that src['dist'] exists and is useful. This can be obtained
        with a call to :func:`mne.setup_source_space` with the
        ``add_dist=True`` option.
    n_times : int
        Number of time points.
    dist : float
        Maximal geodesic distance (in m) between vertices in the
        source space to consider neighbors.
    %(verbose)s

    Returns
    -------
    connectivity : ~scipy.sparse.coo_matrix
        The connectivity matrix describing the spatio-temporal
        graph structure. If N is the number of vertices in the
        source space, the N first nodes in the graph are the
        vertices are time 1, the nodes from 2 to 2N are the vertices
        during time 2, etc.
    """
    if src[0]['dist'] is None:
        raise RuntimeError('src must have distances included, consider using '
                           'setup_source_space with add_dist=True')
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
    """Compute connectivity for a source space activation.

    Parameters
    ----------
    src : instance of SourceSpaces
        The source space. It can be a surface source space or a
        volume source space.
    dist : float, or None
        Maximal geodesic distance (in m) between vertices in the
        source space to consider neighbors. If None, immediate neighbors
        are extracted from an ico surface.
    %(verbose)s

    Returns
    -------
    connectivity : ~scipy.sparse.coo_matrix
        The connectivity matrix describing the spatial graph structure.
    """
    return spatio_temporal_src_connectivity(src, 1, dist)


@verbose
def spatial_tris_connectivity(tris, remap_vertices=False, verbose=None):
    """Compute connectivity from triangles.

    Parameters
    ----------
    tris : array
        N x 3 array defining triangles.
    remap_vertices : bool
        Reassign vertex indices based on unique values. Useful
        to process a subset of triangles. Defaults to False.
    %(verbose)s

    Returns
    -------
    connectivity : ~scipy.sparse.coo_matrix
        The connectivity matrix describing the spatial graph structure.
    """
    return spatio_temporal_tris_connectivity(tris, 1, remap_vertices)


@verbose
def spatial_dist_connectivity(src, dist, verbose=None):
    """Compute connectivity from distances in a source space.

    Parameters
    ----------
    src : instance of SourceSpaces
        The source space must have distances between vertices computed, such
        that src['dist'] exists and is useful. This can be obtained
        with a call to :func:`mne.setup_source_space` with the
        ``add_dist=True`` option.
    dist : float
        Maximal geodesic distance (in m) between vertices in the
        source space to consider neighbors.
    %(verbose)s

    Returns
    -------
    connectivity : ~scipy.sparse.coo_matrix
        The connectivity matrix describing the spatial graph structure.
    """
    return spatio_temporal_dist_connectivity(src, 1, dist)


@verbose
def spatial_inter_hemi_connectivity(src, dist, verbose=None):
    """Get vertices on each hemisphere that are close to the other hemisphere.

    Parameters
    ----------
    src : instance of SourceSpaces
        The source space. Must be surface type.
    dist : float
        Maximal Euclidean distance (in m) between vertices in one hemisphere
        compared to the other to consider neighbors.
    %(verbose)s

    Returns
    -------
    connectivity : ~scipy.sparse.coo_matrix
        The connectivity matrix describing the spatial graph structure.
        Typically this should be combined (addititively) with another
        existing intra-hemispheric connectivity matrix, e.g. computed
        using geodesic distances.
    """
    from scipy.spatial.distance import cdist
    src = _ensure_src(src, kind='surface')
    conn = cdist(src[0]['rr'][src[0]['vertno']],
                 src[1]['rr'][src[1]['vertno']])
    conn = sparse.csr_matrix(conn <= dist, dtype=int)
    empties = [sparse.csr_matrix((nv, nv), dtype=int) for nv in conn.shape]
    conn = sparse.vstack([sparse.hstack([empties[0], conn]),
                          sparse.hstack([conn.T, empties[1]])])
    return conn


@verbose
def _get_connectivity_from_edges(edges, n_times, verbose=None):
    """Given edges sparse matrix, create connectivity matrix."""
    n_vertices = edges.shape[0]
    logger.info("-- number of connected vertices : %d" % n_vertices)
    nnz = edges.col.size
    aux = n_vertices * np.arange(n_times)[:, None] * np.ones((1, nnz), np.int)
    col = (edges.col[None, :] + aux).ravel()
    row = (edges.row[None, :] + aux).ravel()
    if n_times > 1:  # add temporal edges
        o = (n_vertices * np.arange(n_times - 1)[:, None] +
             np.arange(n_vertices)[None, :]).ravel()
        d = (n_vertices * np.arange(1, n_times)[:, None] +
             np.arange(n_vertices)[None, :]).ravel()
        row = np.concatenate((row, o, d))
        col = np.concatenate((col, d, o))
    data = np.ones(edges.data.size * n_times + 2 * n_vertices * (n_times - 1),
                   dtype=np.int)
    connectivity = coo_matrix((data, (row, col)),
                              shape=(n_times * n_vertices,) * 2)
    return connectivity


@verbose
def _get_ico_tris(grade, verbose=None, return_surf=False):
    """Get triangles for ico surface."""
    ico = _get_ico_surface(grade)
    if not return_surf:
        return ico['tris']
    else:
        return ico


def _pca_flip(flip, data):
    U, s, V = linalg.svd(data, full_matrices=False)
    # determine sign-flip
    sign = np.sign(np.dot(U[:, 0], flip))
    # use average power in label for scaling
    scale = linalg.norm(s) / np.sqrt(len(data))
    return sign * scale * V[0]


_label_funcs = {
    'mean': lambda flip, data: np.mean(data, axis=0),
    'mean_flip': lambda flip, data: np.mean(flip * data, axis=0),
    'max': lambda flip, data: np.max(np.abs(data), axis=0),
    'pca_flip': _pca_flip,
}


@contextlib.contextmanager
def _temporary_vertices(src, vertices):
    orig_vertices = [s['vertno'] for s in src]
    for s, v in zip(src, vertices):
        s['vertno'] = v
    try:
        yield
    finally:
        for s, v in zip(src, orig_vertices):
            s['vertno'] = v


def _prepare_label_extraction(stc, labels, src, mode, allow_empty):
    """Prepare indices and flips for extract_label_time_course."""
    # if src is a mixed src space, the first 2 src spaces are surf type and
    # the other ones are vol type. For mixed source space n_labels will be the
    # given by the number of ROIs of the cortical parcellation plus the number
    # of vol src space
    from .label import label_sign_flip

    # get vertices from source space, they have to be the same as in the stcs
    vertno = stc.vertices
    nvert = [len(vn) for vn in vertno]

    # do the initialization
    label_vertidx = list()
    label_flip = list()
    for s, v, hemi in zip(src, stc.vertices, ('left', 'right')):
        n_missing = (~np.in1d(v, s['vertno'])).sum()
        if n_missing:
            raise ValueError('%d/%d %s hemisphere stc vertices missing from '
                             'the source space, likely mismatch'
                             % (n_missing, len(v), hemi))
    for label in labels:
        if label.hemi == 'both':
            # handle BiHemiLabel
            sub_labels = [label.lh, label.rh]
        else:
            sub_labels = [label]
        this_vertidx = list()
        for slabel in sub_labels:
            if slabel.hemi == 'lh':
                this_vertices = np.intersect1d(vertno[0], slabel.vertices)
                vertidx = np.searchsorted(vertno[0], this_vertices)
            elif slabel.hemi == 'rh':
                this_vertices = np.intersect1d(vertno[1], slabel.vertices)
                vertidx = nvert[0] + np.searchsorted(vertno[1], this_vertices)
            else:
                raise ValueError('label %s has invalid hemi' % label.name)
            this_vertidx.append(vertidx)

        # convert it to an array
        this_vertidx = np.concatenate(this_vertidx)
        this_flip = None
        if len(this_vertidx) == 0:
            msg = ('source space does not contain any vertices for label %s'
                   % label.name)
            if not allow_empty:
                raise ValueError(msg)
            else:
                warn_(msg + '. Assigning all-zero time series to label.')
            this_vertidx = None  # to later check if label is empty
        elif mode not in ('mean', 'max'):  # mode-dependent initialization
            # label_sign_flip uses two properties:
            #
            # - src[ii]['nn']
            # - src[ii]['vertno']
            #
            # So if we override vertno with the stc vertices, it will pick
            # the correct normals.
            with _temporary_vertices(src, stc.vertices):
                this_flip = label_sign_flip(label, src[:2])[:, None]

        label_vertidx.append(this_vertidx)
        label_flip.append(this_flip)

    return label_vertidx, label_flip


def _gen_extract_label_time_course(stcs, labels, src, mode='mean',
                                   allow_empty=False, verbose=None):
    # loop through source estimates and extract time series
    _check_option('mode', mode, sorted(_label_funcs.keys()))
    func = _label_funcs[mode]
    if len(src) > 2:
        if src[0]['type'] != 'surf' or src[1]['type'] != 'surf':
            raise ValueError('The first 2 source spaces have to be surf type')
        if any(np.any(s['type'] != 'vol') for s in src[2:]):
            raise ValueError('source spaces have to be of vol type')

        n_aparc = len(labels)
        n_aseg = len(src[2:])
        n_labels = n_aparc + n_aseg
    else:
        n_labels = len(labels)
    vertno = None
    for stc in stcs:
        if vertno is None:
            vertno = copy.deepcopy(stc.vertices)
            nvert = [len(v) for v in vertno]
            label_vertidx, src_flip = _prepare_label_extraction(
                stc, labels, src, mode, allow_empty)
        # make sure the stc is compatible with the source space
        for i in range(len(vertno)):
            if len(stc.vertices[i]) != nvert[i]:
                raise ValueError('stc not compatible with source space. '
                                 'stc has %s time series but there are %s '
                                 'vertices in source space'
                                 % (len(stc.vertices[i]), nvert[i]))

        if any(np.any(svn != vn) for svn, vn in zip(stc.vertices, vertno)):
            raise ValueError('stc not compatible with source space')
        if sum(nvert) != stc.shape[0]:
            raise ValueError('stc not compatible with source space. '
                             'stc has %s vertices but the source space '
                             'has %s vertices'
                             % (stc.shape[0], sum(nvert)))

        logger.info('Extracting time courses for %d labels (mode: %s)'
                    % (n_labels, mode))

        # do the extraction
        label_tc = np.zeros((n_labels, stc.data.shape[1]),
                            dtype=stc.data.dtype)
        for i, (vertidx, flip) in enumerate(zip(label_vertidx, src_flip)):
            if vertidx is not None:
                label_tc[i] = func(flip, stc.data[vertidx, :])

        # extract label time series for the vol src space
        if len(src) > 2:
            v1 = nvert[0] + nvert[1]
            for i, nv in enumerate(nvert[2:]):

                v2 = v1 + nv
                v = range(v1, v2)
                if nv != 0:
                    label_tc[n_aparc + i] = np.mean(stc.data[v, :], axis=0)

                v1 = v2

        # this is a generator!
        yield label_tc


@verbose
def extract_label_time_course(stcs, labels, src, mode='mean_flip',
                              allow_empty=False, return_generator=False,
                              verbose=None):
    """Extract label time course for lists of labels and source estimates.

    This function will extract one time course for each label and source
    estimate. The way the time courses are extracted depends on the mode
    parameter (see Notes).

    Parameters
    ----------
    stcs : SourceEstimate | list (or generator) of SourceEstimate
        The source estimates from which to extract the time course.
    %(eltc_labels)s
    %(eltc_src)s
    %(eltc_mode)s
    %(eltc_allow_empty)s
    return_generator : bool
        If True, a generator instead of a list is returned.
    %(verbose)s

    Returns
    -------
    label_tc : array | list (or generator) of array, shape (n_labels, n_times)
        Extracted time course for each label and source estimate.

    Notes
    -----
    %(eltc_mode_notes)s

    If encountering a ``ValueError`` due to mismatch between number of
    source points in the subject source space and computed ``stc`` object set
    ``src`` argument to ``fwd['src']`` to ensure the source space is
    compatible between forward and inverse routines.
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
