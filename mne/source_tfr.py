# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Mads Jensen <mje.mads@gmail.com>
#
# License: BSD (3-clause)

import copy
import os.path as op
import numpy as np
from scipy import linalg, sparse
from scipy.sparse import coo_matrix, block_diag as sparse_block_diag

from .filter import resample
from .fixes import einsum
from .evoked import _get_peak
from .surface import read_surface, _get_ico_surface, mesh_edges
from .source_space import (_ensure_src, _get_morph_src_reordering,
                           _ensure_src_subject, SourceSpaces)
from .utils import (get_subjects_dir, _check_subject, logger, verbose,
                    _time_mask, warn as warn_, copy_function_doc_to_method_doc,
                    fill_doc, _check_option)
from .viz import (plot_source_estimates, plot_vector_source_estimates,
                  plot_volume_source_estimates)
from .io.base import ToDataFrameMixin, TimeMixin
from .externals.h5io import read_hdf5, write_hdf5


class _BaseSourceTFR(ToDataFrameMixin, TimeMixin):
    """Base class for all source estimates.

    Parameters
    ----------
    data : array, shape (n_dipoles, n_times) | tuple, shape (2,)
        The data in source space. The data can either be a single array or
        a tuple with two arrays: "kernel" shape (n_vertices, n_sensors) and
        "sens_data" shape (n_sensors, n_times). In this case, the source
        space data corresponds to "numpy.dot(kernel, sens_data)".
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
                raise ValueError('kernel and sens_data have invalid '
                                 'dimensions')
            if sens_data.ndim != self._data_ndim:
                raise ValueError('The sensor data must have %s dimensions, got '
                                 '%s' % (self._data_ndim, sens_data.ndim,))

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
        fname : string
            The file name to write the source estimate to, should end in
            '-stc.h5'.
        ftype : string
            File format to use. Currently, the only allowed values is "h5".
        %(verbose_meth)s
        """
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
            self._data = np.tensordot(self._kernel, self._sens_data, axes=([-1], [0]))
            self._kernel = None
            self._sens_data = None

    def crop(self, tmin=None, tmax=None):
        """Restrict SourceEstimate to a time interval.

        Parameters
        ----------
        tmin : float | None
            The first time point in seconds. If None the first present is used.
        tmax : float | None
            The last time point in seconds. If None the last present is used.
        """
        mask = _time_mask(self.times, tmin, tmax, sfreq=self.sfreq)
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
        window : string or tuple
            Window to use in resampling. See scipy.signal.resample.
        %(n_jobs)s
        %(verbose_meth)s

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
        self.data = resample(self.data, sfreq, o_sfreq, npad, n_jobs=n_jobs)

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
        return (self._kernel.shape[0], self._sens_data.shape[1], self._sens_data.shape[2])

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

    def copy(self):
        """Return copy of source estimate instance."""
        return copy.deepcopy(self)

        return stcs


class SourceTFR(_BaseSourceTFR):
    """
    Parameters
    ----------
    data :
    %(verbose)s

    Attributes
    ----------
    subject :



    """

    _data_ndim = 3
    _src_type = 'SourceTFR'

    @verbose
    def __init__(self, data, vertices=None, tmin=None, tstep=None,
                 subject=None, verbose=None, ):  # noqa: D102
        if not (isinstance(vertices, np.ndarray) or
                isinstance(vertices, list)):
            raise ValueError('Vertices must be a numpy array or a list of '
                             'arrays')

        _BaseSourceTFR.__init__(self, data, vertices=vertices, tmin=tmin,
                                tstep=tstep, subject=subject,
                                verbose=verbose)
