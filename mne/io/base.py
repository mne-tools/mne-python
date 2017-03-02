# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Denis Engemann <denis.engemann@gmail.com>
#          Teon Brooks <teon.brooks@gmail.com>
#          Marijn van Vliet <w.m.vanvliet@gmail.com>
#
# License: BSD (3-clause)

import copy
from copy import deepcopy
import os
import os.path as op

import numpy as np

from .constants import FIFF
from .pick import pick_types, channel_type, pick_channels, pick_info
from .pick import _pick_data_channels, _pick_data_or_ica
from .meas_info import write_meas_info
from .proj import setup_proj, activate_proj, _proj_equal, ProjMixin
from ..channels.channels import (ContainsMixin, UpdateChannelsMixin,
                                 SetChannelsMixin, InterpolationMixin)
from ..channels.montage import read_montage, _set_montage, Montage
from .compensator import set_current_comp, make_compensator
from .write import (start_file, end_file, start_block, end_block,
                    write_dau_pack16, write_float, write_double,
                    write_complex64, write_complex128, write_int,
                    write_id, write_string, write_name_list, _get_split_size)

from ..filter import (filter_data, notch_filter, resample, next_fast_len,
                      _resample_stim_channels)
from ..parallel import parallel_func
from ..utils import (_check_fname, _check_pandas_installed, sizeof_fmt,
                     _check_pandas_index_arguments,
                     check_fname, _get_stim_channel,
                     logger, verbose, _time_mask, warn, SizeMixin,
                     copy_function_doc_to_method_doc)
from ..viz import plot_raw, plot_raw_psd, plot_raw_psd_topo
from ..defaults import _handle_default
from ..externals.six import string_types
from ..event import find_events, concatenate_events
from ..annotations import Annotations, _combine_annotations, _sync_onset


class ToDataFrameMixin(object):
    """Class to add to_data_frame capabilities to certain classes."""

    def _get_check_picks(self, picks, picks_check):
        """Get and check picks."""
        if picks is None:
            picks = list(range(self.info['nchan']))
        else:
            if not np.in1d(picks, np.arange(len(picks_check))).all():
                raise ValueError('At least one picked channel is not present '
                                 'in this object instance.')
        return picks

    def to_data_frame(self, picks=None, index=None, scale_time=1e3,
                      scalings=None, copy=True, start=None, stop=None):
        """Export data in tabular structure as a pandas DataFrame.

        Columns and indices will depend on the object being converted.
        Generally this will include as much relevant information as
        possible for the data type being converted. This makes it easy
        to convert data for use in packages that utilize dataframes,
        such as statsmodels or seaborn.

        Parameters
        ----------
        picks : array-like of int | None
            If None only MEG and EEG channels are kept
            otherwise the channels indices in picks are kept.
        index : tuple of str | None
            Column to be used as index for the data. Valid string options
            are 'epoch', 'time' and 'condition'. If None, all three info
            columns will be included in the table as categorial data.
        scale_time : float
            Scaling to be applied to time units.
        scalings : dict | None
            Scaling to be applied to the channels picked. If None, defaults to
            ``scalings=dict(eeg=1e6, grad=1e13, mag=1e15, misc=1.0)``.
        copy : bool
            If true, data will be copied. Else data may be modified in place.
        start : int | None
            If it is a Raw object, this defines a starting index for creating
            the dataframe from a slice. The times will be interpolated from the
            index and the sampling rate of the signal.
        stop : int | None
            If it is a Raw object, this defines a stop index for creating
            the dataframe from a slice. The times will be interpolated from the
            index and the sampling rate of the signal.

        Returns
        -------
        df : instance of pandas.core.DataFrame
            A dataframe suitable for usage with other
            statistical/plotting/analysis packages. Column/Index values will
            depend on the object type being converted, but should be
            human-readable.
        """
        from ..epochs import BaseEpochs
        from ..evoked import Evoked
        from ..source_estimate import _BaseSourceEstimate

        pd = _check_pandas_installed()
        mindex = list()
        # Treat SourceEstimates special because they don't have the same info
        if isinstance(self, _BaseSourceEstimate):
            if self.subject is None:
                default_index = ['time']
            else:
                default_index = ['subject', 'time']
            data = self.data.T
            times = self.times
            shape = data.shape
            mindex.append(('subject', np.repeat(self.subject, shape[0])))

            if isinstance(self.vertices, list):
                # surface source estimates
                col_names = [i for e in [
                    ['{0} {1}'.format('LH' if ii < 1 else 'RH', vert)
                     for vert in vertno]
                    for ii, vertno in enumerate(self.vertices)]
                    for i in e]
            else:
                # volume source estimates
                col_names = ['VOL {0}'.format(vert) for vert in self.vertices]
        elif isinstance(self, (BaseEpochs, BaseRaw, Evoked)):
            picks = self._get_check_picks(picks, self.ch_names)
            if isinstance(self, BaseEpochs):
                default_index = ['condition', 'epoch', 'time']
                data = self.get_data()[:, picks, :]
                times = self.times
                n_epochs, n_picks, n_times = data.shape
                data = np.hstack(data).T  # (time*epochs) x signals

                # Multi-index creation
                times = np.tile(times, n_epochs)
                id_swapped = dict((v, k) for k, v in self.event_id.items())
                names = [id_swapped[k] for k in self.events[:, 2]]
                mindex.append(('condition', np.repeat(names, n_times)))
                mindex.append(('epoch',
                              np.repeat(np.arange(n_epochs), n_times)))
                col_names = [self.ch_names[k] for k in picks]

            elif isinstance(self, (BaseRaw, Evoked)):
                default_index = ['time']
                if isinstance(self, BaseRaw):
                    data, times = self[picks, start:stop]
                elif isinstance(self, Evoked):
                    data = self.data[picks, :]
                    times = self.times
                    n_picks, n_times = data.shape
                data = data.T
                col_names = [self.ch_names[k] for k in picks]

            types = [channel_type(self.info, idx) for idx in picks]
            n_channel_types = 0
            ch_types_used = []

            scalings = _handle_default('scalings', scalings)
            for t in scalings.keys():
                if t in types:
                    n_channel_types += 1
                    ch_types_used.append(t)

            for t in ch_types_used:
                scaling = scalings[t]
                idx = [picks[i] for i in range(len(picks)) if types[i] == t]
                if len(idx) > 0:
                    data[:, idx] *= scaling
        else:
            # In case some other object gets this mixin w/o an explicit check
            raise NameError('Object must be one of Raw, Epochs, Evoked,  or ' +
                            'SourceEstimate. This is {0}'.format(type(self)))

        # Make sure that the time index is scaled correctly
        times = np.round(times * scale_time)
        mindex.append(('time', times))

        if index is not None:
            _check_pandas_index_arguments(index, default_index)
        else:
            index = default_index

        if copy is True:
            data = data.copy()

        assert all(len(mdx) == len(mindex[0]) for mdx in mindex)

        df = pd.DataFrame(data, columns=col_names)
        for i, (k, v) in enumerate(mindex):
            df.insert(i, k, v)
        if index is not None:
            if 'time' in index:
                logger.info('Converting time column to int64...')
                df['time'] = df['time'].astype(np.int64)
            df.set_index(index, inplace=True)
        if all(i in default_index for i in index):
            df.columns.name = 'signal'
        return df


class TimeMixin(object):
    """Class to add sfreq and time_as_index capabilities to certain classes."""

    def time_as_index(self, times, use_rounding=False):
        """Convert time to indices.

        Parameters
        ----------
        times : list-like | float | int
            List of numbers or a number representing points in time.
        use_rounding : boolean
            If True, use rounding (instead of truncation) when converting
            times to indices. This can help avoid non-unique indices.

        Returns
        -------
        index : ndarray
            Indices corresponding to the times supplied.
        """
        from ..source_estimate import _BaseSourceEstimate
        if isinstance(self, _BaseSourceEstimate):
            sfreq = 1. / self.tstep
        else:
            sfreq = self.info['sfreq']
        index = (np.atleast_1d(times) - self.times[0]) * sfreq
        if use_rounding:
            index = np.round(index)
        return index.astype(int)


def _check_fun(fun, d, *args, **kwargs):
    """Check shapes."""
    want_shape = d.shape
    d = fun(d, *args, **kwargs)
    if not isinstance(d, np.ndarray):
        raise TypeError('Return value must be an ndarray')
    if d.shape != want_shape:
        raise ValueError('Return data must have shape %s not %s'
                         % (want_shape, d.shape))
    return d


class BaseRaw(ProjMixin, ContainsMixin, UpdateChannelsMixin,
              SetChannelsMixin, InterpolationMixin, ToDataFrameMixin,
              TimeMixin, SizeMixin):
    """Base class for Raw data.

    Parameters
    ----------
    info : dict
        A dict passed from the subclass.
    preload : bool | str | ndarray
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, preload is the
        file name of a memory-mapped file which is used to store the data
        on the hard drive (slower, requires less memory). If preload is an
        ndarray, the data are taken from that array. If False, data are not
        read until save.
    first_samps : iterable
        Iterable of the first sample number from each raw file. For unsplit raw
        files this should be a length-one list or tuple.
    last_samps : iterable | None
        Iterable of the last sample number from each raw file. For unsplit raw
        files this should be a length-one list or tuple. If None, then preload
        must be an ndarray.
    filenames : tuple
        Tuple of length one (for unsplit raw files) or length > 1 (for split
        raw files).
    raw_extras : list
        Whatever data is necessary for on-demand reads. For `RawFIF` this means
        a list of variables formerly known as ``_rawdirs``.
    orig_format : str
        The data format of the original raw file (e.g., ``'double'``).
    dtype : dtype | None
        The dtype of the raw data. If preload is an ndarray, its dtype must
        match what is passed here.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Notes
    -----
    The `BaseRaw` class is public to allow for stable type-checking in user
    code (i.e., ``isinstance(my_raw_object, BaseRaw)``) but should not be used
    as a constructor for `Raw` objects (use instead one of the subclass
    constructors, or one of the ``mne.io.read_raw_*`` functions).

    Subclasses must provide the following methods:

        * _read_segment_file(self, data, idx, fi, start, stop, cals, mult)
          (only needed for types that support on-demand disk reads)

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """

    @verbose
    def __init__(self, info, preload=False,
                 first_samps=(0,), last_samps=None,
                 filenames=(None,), raw_extras=(None,),
                 orig_format='double', dtype=np.float64,
                 verbose=None):  # noqa: D102
        # wait until the end to preload data, but triage here
        if isinstance(preload, np.ndarray):
            # some functions (e.g., filtering) only work w/64-bit data
            if preload.dtype not in (np.float64, np.complex128):
                raise RuntimeError('datatype must be float64 or complex128, '
                                   'not %s' % preload.dtype)
            if preload.dtype != dtype:
                raise ValueError('preload and dtype must match')
            self._data = preload
            self.preload = True
            assert len(first_samps) == 1
            last_samps = [first_samps[0] + self._data.shape[1] - 1]
            load_from_disk = False
        else:
            if last_samps is None:
                raise ValueError('last_samps must be given unless preload is '
                                 'an ndarray')
            if preload is False:
                self.preload = False
                load_from_disk = False
            elif preload is not True and not isinstance(preload, string_types):
                raise ValueError('bad preload: %s' % preload)
            else:
                load_from_disk = True
        self._last_samps = np.array(last_samps)
        self._first_samps = np.array(first_samps)
        info._check_consistency()  # make sure subclass did a good job
        self.info = info
        if info.get('buffer_size_sec', None) is None:
            raise RuntimeError('Reader error, notify mne-python developers')
        cals = np.empty(info['nchan'])
        for k in range(info['nchan']):
            cals[k] = info['chs'][k]['range'] * info['chs'][k]['cal']
        bad = np.where(cals == 0)[0]
        if len(bad) > 0:
            raise ValueError('Bad cals for channels %s'
                             % dict((ii, self.ch_names[ii]) for ii in bad))
        self.verbose = verbose
        self._cals = cals
        self._raw_extras = list(raw_extras)
        # deal with compensation (only relevant for CTF data, either CTF
        # reader or MNE-C converted CTF->FIF files)
        self._read_comp_grade = self.compensation_grade  # read property
        if self._read_comp_grade is not None:
            logger.info('Current compensation grade : %d'
                        % self._read_comp_grade)
        self._comp = None
        self._filenames = list(filenames)
        self.orig_format = orig_format
        self._projectors = list()
        self._projector = None
        self._dtype_ = dtype
        self.annotations = None
        # If we have True or a string, actually do the preloading
        self._update_times()
        if load_from_disk:
            self._preload_data(preload)

    @verbose
    def apply_gradient_compensation(self, grade, verbose=None):
        """Apply CTF gradient compensation.

        .. warning:: The compensation matrices are stored with single
                     precision, so repeatedly switching between different
                     of compensation (e.g., 0->1->3->2) can increase
                     numerical noise, especially if data are saved to
                     disk in between changing grades. It is thus best to
                     only use a single gradient compensation level in
                     final analyses.

        Parameters
        ----------
        grade : int
            CTF gradient compensation level.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more).

        Returns
        -------
        raw : instance of Raw
            The modified Raw instance. Works in-place.
        """
        grade = int(grade)
        current_comp = self.compensation_grade
        if current_comp != grade:
            if self.proj:
                raise RuntimeError('Cannot change compensation on data where '
                                   'projectors have been applied')
            # Figure out what operator to use (varies depending on preload)
            from_comp = current_comp if self.preload else self._read_comp_grade
            comp = make_compensator(self.info, from_comp, grade)
            logger.info('Compensator constructed to change %d -> %d'
                        % (current_comp, grade))
            set_current_comp(self.info, grade)
            # We might need to apply it to our data now
            if self.preload:
                logger.info('Applying compensator to loaded data')
                lims = np.concatenate([np.arange(0, len(self.times), 10000),
                                       [len(self.times)]])
                for start, stop in zip(lims[:-1], lims[1:]):
                    self._data[:, start:stop] = np.dot(
                        comp, self._data[:, start:stop])
            else:
                self._comp = comp  # store it for later use
        return self

    @property
    def _dtype(self):
        """Datatype for loading data (property so subclasses can override)."""
        # most classes only store real data, they won't need anything special
        return self._dtype_

    def _read_segment(self, start=0, stop=None, sel=None, data_buffer=None,
                      projector=None, verbose=None):
        """Read a chunk of raw data.

        Parameters
        ----------
        start : int, (optional)
            first sample to include (first is 0). If omitted, defaults to the
            first sample in data.
        stop : int, (optional)
            First sample to not include.
            If omitted, data is included to the end.
        sel : array, optional
            Indices of channels to select.
        data_buffer : array or str, optional
            numpy array to fill with data read, must have the correct shape.
            If str, a np.memmap with the correct data type will be used
            to store the data.
        projector : array
            SSP operator to apply to the data.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more).

        Returns
        -------
        data : array, [channels x samples]
           the data matrix (channels x samples).
        """
        #  Initial checks
        start = int(start)
        stop = self.n_times if stop is None else min([int(stop), self.n_times])

        if start >= stop:
            raise ValueError('No data in this range')

        #  Initialize the data and calibration vector
        n_sel_channels = self.info['nchan'] if sel is None else len(sel)
        # convert sel to a slice if possible for efficiency
        if sel is not None and len(sel) > 1 and np.all(np.diff(sel) == 1):
            sel = slice(sel[0], sel[-1] + 1)
        idx = slice(None, None, None) if sel is None else sel
        data_shape = (n_sel_channels, stop - start)
        dtype = self._dtype
        if isinstance(data_buffer, np.ndarray):
            if data_buffer.shape != data_shape:
                raise ValueError('data_buffer has incorrect shape: %s != %s'
                                 % (data_buffer.shape, data_shape))
            data = data_buffer
        elif isinstance(data_buffer, string_types):
            # use a memmap
            data = np.memmap(data_buffer, mode='w+',
                             dtype=dtype, shape=data_shape)
        else:
            data = np.zeros(data_shape, dtype=dtype)

        # deal with having multiple files accessed by the raw object
        cumul_lens = np.concatenate(([0], np.array(self._raw_lengths,
                                                   dtype='int')))
        cumul_lens = np.cumsum(cumul_lens)
        files_used = np.logical_and(np.less(start, cumul_lens[1:]),
                                    np.greater_equal(stop - 1,
                                                     cumul_lens[:-1]))

        # set up cals and mult (cals, compensation, and projector)
        cals = self._cals.ravel()[np.newaxis, :]
        if self._comp is not None:
            if projector is not None:
                mult = self._comp * cals
                mult = np.dot(projector[idx], mult)
            else:
                mult = self._comp[idx] * cals
        elif projector is not None:
            mult = projector[idx] * cals
        else:
            mult = None
        cals = cals.T[idx]

        # read from necessary files
        offset = 0
        for fi in np.nonzero(files_used)[0]:
            start_file = self._first_samps[fi]
            # first iteration (only) could start in the middle somewhere
            if offset == 0:
                start_file += start - cumul_lens[fi]
            stop_file = np.min([stop - cumul_lens[fi] + self._first_samps[fi],
                                self._last_samps[fi] + 1])
            if start_file < self._first_samps[fi] or stop_file < start_file:
                raise ValueError('Bad array indexing, could be a bug')
            n_read = stop_file - start_file
            this_sl = slice(offset, offset + n_read)
            self._read_segment_file(data[:, this_sl], idx, fi,
                                    int(start_file), int(stop_file),
                                    cals, mult)
            offset += n_read
        return data

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a segment of data from a file.

        Only needs to be implemented for readers that support
        ``preload=False``.

        Parameters
        ----------
        data : ndarray, shape (len(idx), stop - start + 1)
            The data array. Should be modified inplace.
        idx : ndarray | slice
            The requested channel indices.
        fi : int
            The file index that must be read from.
        start : int
            The start sample in the given file.
        stop : int
            The stop sample in the given file (inclusive).
        cals : ndarray, shape (len(idx), 1)
            Channel calibrations (already sub-indexed).
        mult : ndarray, shape (len(idx), len(info['chs']) | None
            The compensation + projection + cals matrix, if applicable.
        """
        raise NotImplementedError

    def _check_bad_segment(self, start, stop, picks,
                           reject_by_annotation=False):
        """Function for checking if data segment is bad.

        If the slice is good, returns the data in desired range.
        If rejected based on annotation, returns description of the
        bad segment as a string.

        Parameters
        ----------
        start : int
            First sample of the slice.
        stop : int
            End of the slice.
        picks : array of int
            Channel picks.
        reject_by_annotation : bool
            Whether to perform rejection based on annotations.
            False by default.

        Returns
        -------
        data : array | str
            Data in the desired range (good segment) or description of the bad
            segment.
        """
        if start < 0:
            return None
        if reject_by_annotation and self.annotations is not None:
            annot = self.annotations
            sfreq = self.info['sfreq']
            onset = _sync_onset(self, annot.onset)
            overlaps = np.where(onset < stop / sfreq)
            overlaps = np.where(onset[overlaps] + annot.duration[overlaps] >
                                start / sfreq)
            for descr in annot.description[overlaps]:
                if descr.lower().startswith('bad'):
                    return descr
        return self[picks, start:stop][0]

    @verbose
    def load_data(self, verbose=None):
        """Load raw data.

        Parameters
        ----------
        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more).

        Returns
        -------
        raw : instance of Raw
            The raw object with data.

        Notes
        -----
        This function will load raw data if it was not already preloaded.
        If data were already preloaded, it will do nothing.

        .. versionadded:: 0.10.0
        """
        if not self.preload:
            self._preload_data(True)
        return self

    @verbose
    def _preload_data(self, preload, verbose=None):
        """Actually preload the data."""
        data_buffer = preload if isinstance(preload, (string_types,
                                                      np.ndarray)) else None
        logger.info('Reading %d ... %d  =  %9.3f ... %9.3f secs...' %
                    (0, len(self.times) - 1, 0., self.times[-1]))
        self._data = self._read_segment(data_buffer=data_buffer)
        assert len(self._data) == self.info['nchan']
        self.preload = True
        self._comp = None  # no longer needed
        self.close()

    def _update_times(self):
        """Update times."""
        self._times = np.arange(self.n_times) / float(self.info['sfreq'])
        # make it immutable
        self._times.flags.writeable = False

    @property
    def _first_time(self):
        return self.first_samp / float(self.info['sfreq'])

    @property
    def first_samp(self):
        """The first data sample."""
        return self._first_samps[0]

    @property
    def last_samp(self):
        """The last data sample."""
        return self.first_samp + sum(self._raw_lengths) - 1

    @property
    def _raw_lengths(self):
        return [l - f + 1 for f, l in zip(self._first_samps, self._last_samps)]

    @property
    def annotations(self):  # noqa: D401
        """Annotations for marking segments of data."""
        return self._annotations

    @property
    def filenames(self):
        """The filenames used."""
        return tuple(self._filenames)

    @annotations.setter
    def annotations(self, annotations):
        """Setter for annotations.

        This setter checks if they are inside the data range.

        Parameters
        ----------
        annotations : Instance of mne.Annotations
            Annotations to set.
        """
        if annotations is not None:
            if not isinstance(annotations, Annotations):
                raise ValueError('Annotations must be an instance of '
                                 'mne.Annotations. Got %s.' % annotations)
            meas_date = self.info['meas_date']
            if meas_date is None:
                meas_date = 0
            elif not np.isscalar(meas_date):
                if len(meas_date) > 1:
                    meas_date = meas_date[0] + meas_date[1] / 1000000.
            if annotations.orig_time is not None:
                offset = (annotations.orig_time - meas_date -
                          self.first_samp / self.info['sfreq'])
            else:
                offset = 0
            omit_ind = list()
            for ind, onset in enumerate(annotations.onset):
                onset += offset
                if onset > self.times[-1]:
                    warn('Omitting annotation outside data range.')
                    omit_ind.append(ind)
                elif onset < self.times[0]:
                    if onset + annotations.duration[ind] < self.times[0]:
                        warn('Omitting annotation outside data range.')
                        omit_ind.append(ind)
                    else:
                        warn('Annotation starting outside the data range. '
                             'Limiting to the start of data.')
                        duration = annotations.duration[ind] + onset
                        annotations.duration[ind] = duration
                        annotations.onset[ind] = self.times[0] - offset
                elif onset + annotations.duration[ind] > self.times[-1]:
                    warn('Annotation expanding outside the data range. '
                         'Limiting to the end of data.')
                    annotations.duration[ind] = self.times[-1] - onset
            annotations.onset = np.delete(annotations.onset, omit_ind)
            annotations.duration = np.delete(annotations.duration, omit_ind)
            annotations.description = np.delete(annotations.description,
                                                omit_ind)

        self._annotations = annotations

    def __del__(self):  # noqa: D105
        # remove file for memmap
        if hasattr(self, '_data') and hasattr(self._data, 'filename'):
            # First, close the file out; happens automatically on del
            filename = self._data.filename
            del self._data
            # Now file can be removed
            try:
                os.remove(filename)
            except OSError:
                pass  # ignore file that no longer exists

    def __enter__(self):
        """Entering with block."""
        return self

    def __exit__(self, exception_type, exception_val, trace):
        """Exiting with block."""
        try:
            self.close()
        except:
            return exception_type, exception_val, trace

    def _parse_get_set_params(self, item):
        """Parse the __getitem__ / __setitem__ tuples."""
        # make sure item is a tuple
        if not isinstance(item, tuple):  # only channel selection passed
            item = (item, slice(None, None, None))

        if len(item) != 2:  # should be channels and time instants
            raise RuntimeError("Unable to access raw data (need both channels "
                               "and time)")

        if isinstance(item[0], slice):
            start = item[0].start if item[0].start is not None else 0
            nchan = self.info['nchan']
            if start < 0:
                start += nchan
                if start < 0:
                    raise ValueError('start must be >= -%s' % nchan)
            stop = item[0].stop if item[0].stop is not None else nchan
            step = item[0].step if item[0].step is not None else 1
            sel = list(range(start, stop, step))
        else:
            sel = item[0]

        if isinstance(item[1], slice):
            time_slice = item[1]
            start, stop, step = (time_slice.start, time_slice.stop,
                                 time_slice.step)
        else:
            item1 = item[1]
            # Let's do automated type conversion to integer here
            if np.array(item[1]).dtype.kind == 'i':
                item1 = int(item1)
            if isinstance(item1, (int, np.integer)):
                start, stop, step = item1, item1 + 1, 1
            else:
                raise ValueError('Must pass int or slice to __getitem__')

        if start is None:
            start = 0
        if (step is not None) and (step is not 1):
            raise ValueError('step needs to be 1 : %d given' % step)

        if isinstance(sel, (int, np.integer)):
            sel = np.array([sel])

        if sel is not None and len(sel) == 0:
            raise ValueError("Empty channel list")

        return sel, start, stop

    def __getitem__(self, item):
        """Get raw data and times.

        Parameters
        ----------
        item : tuple or array-like
            See below for use cases.

        Returns
        -------
        data : ndarray, shape (n_channels, n_times)
            The raw data.
        times : ndarray, shape (n_times,)
            The times associated with the data.

        Examples
        --------
        Generally raw data is accessed as::

            >>> data, times = raw[picks, time_slice]  # doctest: +SKIP

        To get all data, you can thus do either of::

            >>> data, times = raw[:]  # doctest: +SKIP

        Which will be equivalent to:

            >>> data, times = raw[:, :]  # doctest: +SKIP

        To get only the good MEG data from 10-20 seconds, you could do::

            >>> picks = mne.pick_types(raw.info, meg=True, exclude='bads')  # doctest: +SKIP
            >>> t_idx = raw.time_as_index([10., 20.])  # doctest: +SKIP
            >>> data, times = raw[picks, t_idx[0]:t_idx[1]]  # doctest: +SKIP

        """  # noqa: E501
        sel, start, stop = self._parse_get_set_params(item)
        if self.preload:
            data = self._data[sel, start:stop]
        else:
            data = self._read_segment(start=start, stop=stop, sel=sel,
                                      projector=self._projector,
                                      verbose=self.verbose)
        times = self.times[start:stop]
        return data, times

    def __setitem__(self, item, value):
        """Set raw data content."""
        _check_preload(self, 'Modifying data of Raw')
        sel, start, stop = self._parse_get_set_params(item)
        # set the data
        self._data[sel, start:stop] = value

    def get_data(self, picks=None, start=0, stop=None,
                 reject_by_annotation=None, return_times=False):
        """Get data in the given range.

        Parameters
        ----------
        picks : array-like of int | None
            Indices of channels to get data from. If None, data from all
            channels is returned
        start : int
            The first sample to include. Defaults to 0.
        stop : int | None
            End sample (first not to include). If None (default), the end of
            the data is  used.
        reject_by_annotation : None | 'omit' | 'NaN'
            Whether to reject by annotation. If None (default), no rejection is
            done. If 'omit', segments annotated with description starting with
            'bad' are omitted. If 'NaN', the bad samples are filled with NaNs.
        return_times : bool
            Whether to return times as well.

        Returns
        -------
        data : ndarray, shape (n_channels, n_times)
            Copy of the data in the given range.
        times : ndarray, shape (n_times,)
            Times associated with the data samples. Only returned if
            return_times=True.

        Notes
        -----
        .. versionadded:: 0.14.0
        """
        if picks is None:
            picks = np.arange(self.info['nchan'])
        start = 0 if start is None else start
        stop = self.n_times if stop is None else stop
        if self.annotations is None or reject_by_annotation is None:
            data, times = self[picks, start:stop]
            if return_times:
                return data, times
            return data
        if reject_by_annotation.lower() not in ['omit', 'nan']:
            raise ValueError("reject_by_annotation must be None, 'omit' or "
                             "'NaN'. Got %s." % reject_by_annotation)
        sfreq = self.info['sfreq']
        bads = [idx for idx, desc in enumerate(self.annotations.description)
                if desc.upper().startswith('BAD')]
        onsets = self.annotations.onset[bads]
        onsets = _sync_onset(self, onsets)
        ends = onsets + self.annotations.duration[bads]
        omit = np.concatenate([np.where(onsets > stop / sfreq)[0],
                               np.where(ends < start / sfreq)[0]])
        onsets, ends = np.delete(onsets, omit), np.delete(ends, omit)
        if len(onsets) == 0:
            data, times = self[picks, start:stop]
            if return_times:
                return data, times
            return data
        stop = min(stop, self.n_times)
        order = np.argsort(onsets)
        onsets = self.time_as_index(onsets[order])
        ends = self.time_as_index(ends[order])

        np.clip(onsets, start, stop, onsets)
        np.clip(ends, start, stop, ends)
        used = np.ones(stop - start, bool)
        for onset, end in zip(onsets, ends):
            if onset >= end:
                continue
            used[onset - start: end - start] = False
        used = np.concatenate([[False], used, [False]])
        starts = np.where(~used[:-1] & used[1:])[0] + start
        stops = np.where(used[:-1] & ~used[1:])[0] + start
        if reject_by_annotation == 'omit':

            data = np.zeros((len(picks), (stops - starts).sum()))
            times = np.zeros(data.shape[1])
            idx = 0
            for start, stop in zip(starts, stops):  # get the data
                if start == stop:
                    continue
                end = idx + stop - start
                data[:, idx:end], times[idx:end] = self[picks, start:stop]
                idx = end
        else:
            data, times = self[picks, start:stop]
            data[:, ~used[1:-1]] = np.nan

        if return_times:
            return data, times
        return data

    @verbose
    def apply_function(self, fun, picks=None, dtype=None,
                       n_jobs=1, *args, **kwargs):
        """Apply a function to a subset of channels.

        The function "fun" is applied to the channels defined in "picks". The
        data of the Raw object is modified inplace. If the function returns
        a different data type (e.g. numpy.complex) it must be specified using
        the dtype parameter, which causes the data type used for representing
        the raw data to change.

        The Raw object has to have the data loaded e.g. with ``preload=True``
        or ``self.load_data()``.

        .. note:: If n_jobs > 1, more memory is required as
                  ``len(picks) * n_times`` additional time points need to
                  be temporaily stored in memory.

        .. note:: If the data type changes (dtype != None), more memory is
                  required since the original and the converted data needs
                  to be stored in memory.

        Parameters
        ----------
        fun : function
            A function to be applied to the channels. The first argument of
            fun has to be a timeseries (numpy.ndarray). The function must
            return an numpy.ndarray with the same size as the input.
        picks : array-like of int (default: None)
            Indices of channels to apply the function to. If None, all data
            channels are used.
        dtype : numpy.dtype (default: None)
            Data type to use for raw data after applying the function. If None
            the data type is not modified.
        n_jobs: int (default: 1)
            Number of jobs to run in parallel.
        *args :
            Additional positional arguments to pass to fun (first pos. argument
            of fun is the timeseries of a channel).
        **kwargs :
            Keyword arguments to pass to fun. Note that if "verbose" is passed
            as a member of ``kwargs``, it will be consumed and will override
            the default mne-python verbose level (see :func:`mne.verbose` and
            :ref:`Logging documentation <tut_logging>` for more).

        Returns
        -------
        self : instance of Raw
            The raw object with transformed data.
        """
        _check_preload(self, 'raw.apply_function')
        if picks is None:
            picks = _pick_data_channels(self.info, exclude=[],
                                        with_ref_meg=False)

        if not callable(fun):
            raise ValueError('fun needs to be a function')

        data_in = self._data
        if dtype is not None and dtype != self._data.dtype:
            self._data = self._data.astype(dtype)

        if n_jobs == 1:
            # modify data inplace to save memory
            for idx in picks:
                self._data[idx, :] = _check_fun(fun, data_in[idx, :],
                                                *args, **kwargs)
        else:
            # use parallel function
            parallel, p_fun, _ = parallel_func(_check_fun, n_jobs)
            data_picks_new = parallel(p_fun(fun, data_in[p], *args, **kwargs)
                                      for p in picks)
            for pp, p in enumerate(picks):
                self._data[p, :] = data_picks_new[pp]
        return self

    @verbose
    def apply_hilbert(self, picks=None, envelope=False, n_jobs=1, n_fft='auto',
                      verbose=None):
        """Compute analytic signal or envelope for a subset of channels.

        If envelope=False, the analytic signal for the channels defined in
        "picks" is computed and the data of the Raw object is converted to
        a complex representation (the analytic signal is complex valued).

        If envelope=True, the absolute value of the analytic signal for the
        channels defined in "picks" is computed, resulting in the envelope
        signal.

        .. warning: Do not use ``envelope=True`` if you intend to compute
                    an inverse solution from the raw data. If you want to
                    compute the envelope in source space, use
                    ``envelope=False`` and compute the envelope after the
                    inverse solution has been obtained.

        .. note:: If envelope=False, more memory is required since the
                  original raw data as well as the analytic signal have
                  temporarily to be stored in memory.

        .. note:: If n_jobs > 1, more memory is required as
                  ``len(picks) * n_times`` additional time points need to
                  be temporaily stored in memory.

        Parameters
        ----------
        picks : array-like of int (default: None)
            Indices of channels to apply the function to. If None, all data
            channels are used.
        envelope : bool (default: False)
            Compute the envelope signal of each channel.
        n_jobs: int
            Number of jobs to run in parallel.
        n_fft : int | None | str
            Points to use in the FFT for Hilbert transformation. The signal
            will be padded with zeros before computing Hilbert, then cut back
            to original length. If None, n == self.n_times. If 'auto',
            the next highest fast FFT length will be use.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more). Defaults to self.verbose.

        Returns
        -------
        self : instance of Raw
            The raw object with transformed data.

        Notes
        -----
        The analytic signal "x_a(t)" of "x(t)" is::

            x_a = F^{-1}(F(x) 2U) = x + i y

        where "F" is the Fourier transform, "U" the unit step function,
        and "y" the Hilbert transform of "x". One usage of the analytic
        signal is the computation of the envelope signal, which is given by
        "e(t) = abs(x_a(t))". Due to the linearity of Hilbert transform and the
        MNE inverse solution, the enevlope in source space can be obtained
        by computing the analytic signal in sensor space, applying the MNE
        inverse, and computing the envelope in source space.

        Also note that the n_fft parameter will allow you to pad the signal
        with zeros before performing the Hilbert transform. This padding
        is cut off, but it may result in a slightly different result
        (particularly around the edges). Use at your own risk.
        """
        if n_fft is None:
            n_fft = len(self.times)
        elif isinstance(n_fft, string_types):
            if n_fft != 'auto':
                raise ValueError('n_fft must be an integer, string, or None, '
                                 'got %s' % (type(n_fft),))
            n_fft = next_fast_len(len(self.times))
        n_fft = int(n_fft)
        if n_fft < self.n_times:
            raise ValueError("n_fft must be greater than n_times")
        if envelope is True:
            dtype = None
        else:
            dtype = np.complex64
        return self.apply_function(_my_hilbert, picks, dtype, n_jobs, n_fft,
                                   envelope=envelope)

    @verbose
    def filter(self, l_freq, h_freq, picks=None, filter_length='auto',
               l_trans_bandwidth='auto', h_trans_bandwidth='auto', n_jobs=1,
               method='fir', iir_params=None, phase='zero',
               fir_window='hamming', verbose=None):
        """Filter a subset of channels.

        Applies a zero-phase low-pass, high-pass, band-pass, or band-stop
        filter to the channels selected by ``picks``. By default the data
        of the Raw object is modified inplace.

        The Raw object has to have the data loaded e.g. with ``preload=True``
        or ``self.load_data()``.

        ``l_freq`` and ``h_freq`` are the frequencies below which and above
        which, respectively, to filter out of the data. Thus the uses are:

            * ``l_freq < h_freq``: band-pass filter
            * ``l_freq > h_freq``: band-stop filter
            * ``l_freq is not None and h_freq is None``: high-pass filter
            * ``l_freq is None and h_freq is not None``: low-pass filter

        ``self.info['lowpass']`` and ``self.info['highpass']`` are only
        updated with picks=None.

        .. note:: If n_jobs > 1, more memory is required as
                  ``len(picks) * n_times`` additional time points need to
                  be temporaily stored in memory.

        Parameters
        ----------
        l_freq : float | None
            Low cut-off frequency in Hz. If None the data are only low-passed.
        h_freq : float | None
            High cut-off frequency in Hz. If None the data are only
            high-passed.
        picks : array-like of int | None
            Indices of channels to filter. If None only the data (MEG/EEG)
            channels will be filtered.
        filter_length : str | int
            Length of the FIR filter to use (if applicable):

                * int: specified length in samples.
                * 'auto' (default): the filter length is chosen based
                  on the size of the transition regions (6.6 times the
                  reciprocal of the shortest transition band for
                  fir_window='hamming').
                * str: a human-readable time in
                  units of "s" or "ms" (e.g., "10s" or "5500ms") will be
                  converted to that number of samples if ``phase="zero"``, or
                  the shortest power-of-two length at least that duration for
                  ``phase="zero-double"``.

        l_trans_bandwidth : float | str
            Width of the transition band at the low cut-off frequency in Hz
            (high pass or cutoff 1 in bandpass). Can be "auto"
            (default) to use a multiple of ``l_freq``::

                min(max(l_freq * 0.25, 2), l_freq)

            Only used for ``method='fir'``.
        h_trans_bandwidth : float | str
            Width of the transition band at the high cut-off frequency in Hz
            (low pass or cutoff 2 in bandpass). Can be "auto"
            (default) to use a multiple of ``h_freq``::

                min(max(h_freq * 0.25, 2.), info['sfreq'] / 2. - h_freq)

            Only used for ``method='fir'``.
        n_jobs : int | str
            Number of jobs to run in parallel. Can be 'cuda' if scikits.cuda
            is installed properly, CUDA is initialized, and method='fir'.
        method : str
            'fir' will use overlap-add FIR filtering, 'iir' will use IIR
            forward-backward filtering (via filtfilt).
        iir_params : dict | None
            Dictionary of parameters to use for IIR filtering.
            See mne.filter.construct_iir_filter for details. If iir_params
            is None and method="iir", 4th order Butterworth will be used.
        phase : str
            Phase of the filter, only used if ``method='fir'``.
            By default, a symmetric linear-phase FIR filter is constructed.
            If ``phase='zero'`` (default), the delay of this filter
            is compensated for. If ``phase=='zero-double'``, then this filter
            is applied twice, once forward, and once backward. If 'minimum',
            then a minimum-phase, causal filter will be used.

            .. versionadded:: 0.13

        fir_window : str
            The window to use in FIR design, can be "hamming" (default),
            "hann" (default in 0.13), or "blackman".

            .. versionadded:: 0.13

        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more). Defaults to self.verbose.

        Returns
        -------
        raw : instance of Raw
            The raw instance with filtered data.

        See Also
        --------
        mne.Epochs.savgol_filter
        mne.io.Raw.notch_filter
        mne.io.Raw.resample
        mne.filter.filter_data
        mne.filter.construct_iir_filter

        Notes
        -----
        For more information, see the tutorials :ref:`tut_background_filtering`
        and :ref:`tut_artifacts_filter`.
        """
        _check_preload(self, 'raw.filter')
        data_picks = _pick_data_or_ica(self.info)
        update_info = False
        if picks is None:
            picks = data_picks
            update_info = True
            # let's be safe.
            if len(picks) == 0:
                raise RuntimeError('Could not find any valid channels for '
                                   'your Raw object. Please contact the '
                                   'MNE-Python developers.')
        elif h_freq is not None or l_freq is not None:
            if np.in1d(data_picks, picks).all():
                update_info = True
            else:
                logger.info('Filtering a subset of channels. The highpass and '
                            'lowpass values in the measurement info will not '
                            'be updated.')
        filter_data(self._data, self.info['sfreq'], l_freq, h_freq, picks,
                    filter_length, l_trans_bandwidth, h_trans_bandwidth,
                    n_jobs, method, iir_params, copy=False, phase=phase,
                    fir_window=fir_window)
        # update info if filter is applied to all data channels,
        # and it's not a band-stop filter
        if update_info:
            if h_freq is not None and (l_freq is None or l_freq < h_freq) and \
                    (self.info["lowpass"] is None or
                     h_freq < self.info['lowpass']):
                self.info['lowpass'] = float(h_freq)
            if l_freq is not None and (h_freq is None or l_freq < h_freq) and \
                    (self.info["highpass"] is None or
                     l_freq > self.info['highpass']):
                self.info['highpass'] = float(l_freq)
        return self

    @verbose
    def notch_filter(self, freqs, picks=None, filter_length='auto',
                     notch_widths=None, trans_bandwidth=1.0, n_jobs=1,
                     method='fft', iir_params=None, mt_bandwidth=None,
                     p_value=0.05, phase='zero', fir_window='hamming',
                     verbose=None):
        """Notch filter a subset of channels.

        Applies a zero-phase notch filter to the channels selected by
        "picks". By default the data of the Raw object is modified inplace.

        The Raw object has to have the data loaded e.g. with ``preload=True``
        or ``self.load_data()``.

        .. note:: If n_jobs > 1, more memory is required as
                  ``len(picks) * n_times`` additional time points need to
                  be temporaily stored in memory.

        Parameters
        ----------
        freqs : float | array of float | None
            Specific frequencies to filter out from data, e.g.,
            np.arange(60, 241, 60) in the US or np.arange(50, 251, 50) in
            Europe. None can only be used with the mode 'spectrum_fit',
            where an F test is used to find sinusoidal components.
        picks : array-like of int | None
            Indices of channels to filter. If None only the data (MEG/EEG)
            channels will be filtered.
        filter_length : str | int
            Length of the FIR filter to use (if applicable):

                * int: specified length in samples.
                * 'auto' (default): the filter length is chosen based
                  on the size of the transition regions (6.6 times the
                  reciprocal of the shortest transition band for
                  fir_window='hamming').
                * str: a human-readable time in
                  units of "s" or "ms" (e.g., "10s" or "5500ms") will be
                  converted to that number of samples if ``phase="zero"``, or
                  the shortest power-of-two length at least that duration for
                  ``phase="zero-double"``.

        notch_widths : float | array of float | None
            Width of each stop band (centred at each freq in freqs) in Hz.
            If None, freqs / 200 is used.
        trans_bandwidth : float
            Width of the transition band in Hz.
            Only used for ``method='fir'``.
        n_jobs : int | str
            Number of jobs to run in parallel. Can be 'cuda' if scikits.cuda
            is installed properly, CUDA is initialized, and method='fir'.
        method : str
            'fir' will use overlap-add FIR filtering, 'iir' will use IIR
            forward-backward filtering (via filtfilt). 'spectrum_fit' will
            use multi-taper estimation of sinusoidal components.
        iir_params : dict | None
            Dictionary of parameters to use for IIR filtering.
            See mne.filter.construct_iir_filter for details. If iir_params
            is None and method="iir", 4th order Butterworth will be used.
        mt_bandwidth : float | None
            The bandwidth of the multitaper windowing function in Hz.
            Only used in 'spectrum_fit' mode.
        p_value : float
            p-value to use in F-test thresholding to determine significant
            sinusoidal components to remove when method='spectrum_fit' and
            freqs=None. Note that this will be Bonferroni corrected for the
            number of frequencies, so large p-values may be justified.
        phase : str
            Phase of the filter, only used if ``method='fir'``.
            By default, a symmetric linear-phase FIR filter is constructed.
            If ``phase='zero'`` (default), the delay of this filter
            is compensated for. If ``phase=='zero-double'``, then this filter
            is applied twice, once forward, and once backward. If 'minimum',
            then a minimum-phase, causal filter will be used.

            .. versionadded:: 0.13

        fir_window : str
            The window to use in FIR design, can be "hamming" (default),
            "hann", or "blackman".

            .. versionadded:: 0.13

        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more). Defaults to self.verbose.

        Returns
        -------
        raw : instance of Raw
            The raw instance with filtered data.

        See Also
        --------
        mne.io.Raw.filter

        Notes
        -----
        For details, see :func:`mne.filter.notch_filter`.
        """
        fs = float(self.info['sfreq'])
        if picks is None:
            picks = _pick_data_or_ica(self.info)
            # let's be safe.
            if len(picks) < 1:
                raise RuntimeError('Could not find any valid channels for '
                                   'your Raw object. Please contact the '
                                   'MNE-Python developers.')
        _check_preload(self, 'raw.notch_filter')
        self._data = notch_filter(
            self._data, fs, freqs, filter_length=filter_length,
            notch_widths=notch_widths, trans_bandwidth=trans_bandwidth,
            method=method, iir_params=iir_params, mt_bandwidth=mt_bandwidth,
            p_value=p_value, picks=picks, n_jobs=n_jobs, copy=False,
            phase=phase, fir_window=fir_window)
        return self

    @verbose
    def resample(self, sfreq, npad='auto', window='boxcar', stim_picks=None,
                 n_jobs=1, events=None, verbose=None):
        """Resample all channels.

        The Raw object has to have the data loaded e.g. with ``preload=True``
        or ``self.load_data()``.

        .. warning:: The intended purpose of this function is primarily to
                     speed up computations (e.g., projection calculation) when
                     precise timing of events is not required, as downsampling
                     raw data effectively jitters trigger timings. It is
                     generally recommended not to epoch downsampled data,
                     but instead epoch and then downsample, as epoching
                     downsampled data jitters triggers.
                     For more, see
                     `this illustrative gist <https://gist.github.com/Eric89GXL/01642cb3789992fbca59>`_.

                     If resampling the continuous data is desired, it is
                     recommended to construct events using the original data.
                     The event onsets can be jointly resampled with the raw
                     data using the 'events' parameter.

        Parameters
        ----------
        sfreq : float
            New sample rate to use.
        npad : int | str
            Amount to pad the start and end of the data.
            Can also be "auto" to use a padding that will result in
            a power-of-two size (can be much faster).
        window : string or tuple
            Frequency-domain window to use in resampling.
            See :func:`scipy.signal.resample`.
        stim_picks : array of int | None
            Stim channels. These channels are simply subsampled or
            supersampled (without applying any filtering). This reduces
            resampling artifacts in stim channels, but may lead to missing
            triggers. If None, stim channels are automatically chosen using
            :func:`mne.pick_types`.
        n_jobs : int | str
            Number of jobs to run in parallel. Can be 'cuda' if scikits.cuda
            is installed properly and CUDA is initialized.
        events : 2D array, shape (n_events, 3) | None
            An optional event matrix. When specified, the onsets of the events
            are resampled jointly with the data.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more). Defaults to self.verbose.

        Returns
        -------
        raw : instance of Raw
            The resampled version of the raw object.

        See Also
        --------
        mne.io.Raw.filter
        mne.Epochs.resample

        Notes
        -----
        For some data, it may be more accurate to use ``npad=0`` to reduce
        artifacts. This is dataset dependent -- check your data!
        """  # noqa: E501
        _check_preload(self, 'raw.resample')

        # When no event object is supplied, some basic detection of dropped
        # events is performed to generate a warning. Finding events can fail
        # for a variety of reasons, e.g. if no stim channel is present or it is
        # corrupted. This should not stop the resampling from working. The
        # warning should simply not be generated in this case.
        if events is None:
            try:
                original_events = find_events(self)
            except:
                pass

        sfreq = float(sfreq)
        o_sfreq = float(self.info['sfreq'])

        offsets = np.concatenate(([0], np.cumsum(self._raw_lengths)))
        new_data = list()

        ratio = sfreq / o_sfreq

        # set up stim channel processing
        if stim_picks is None:
            stim_picks = pick_types(self.info, meg=False, ref_meg=False,
                                    stim=True, exclude=[])
        stim_picks = np.asanyarray(stim_picks)

        for ri in range(len(self._raw_lengths)):
            data_chunk = self._data[:, offsets[ri]:offsets[ri + 1]]
            new_data.append(resample(data_chunk, sfreq, o_sfreq, npad,
                                     window=window, n_jobs=n_jobs))
            new_ntimes = new_data[ri].shape[1]

            # In empirical testing, it was faster to resample all channels
            # (above) and then replace the stim channels than it was to only
            # resample the proper subset of channels and then use np.insert()
            # to restore the stims.
            if len(stim_picks) > 0:
                stim_resampled = _resample_stim_channels(
                    data_chunk[stim_picks], new_data[ri].shape[1],
                    data_chunk.shape[1])
                new_data[ri][stim_picks] = stim_resampled

            self._first_samps[ri] = int(self._first_samps[ri] * ratio)
            self._last_samps[ri] = self._first_samps[ri] + new_ntimes - 1
            self._raw_lengths[ri] = new_ntimes

        self._data = np.concatenate(new_data, axis=1)
        self.info['sfreq'] = sfreq
        if self.info.get('lowpass') is not None:
            self.info['lowpass'] = min(self.info['lowpass'], sfreq / 2.)
        self._update_times()

        # See the comment above why we ignore all errors here.
        if events is None:
            try:
                # Did we loose events?
                resampled_events = find_events(self)
                if len(resampled_events) != len(original_events):
                    warn('Resampling of the stim channels caused event '
                         'information to become unreliable. Consider finding '
                         'events on the original data and passing the event '
                         'matrix as a parameter.')
            except:
                pass

            return self
        else:
            if copy:
                events = events.copy()

            events[:, 0] = np.minimum(
                np.round(events[:, 0] * ratio).astype(int),
                self._data.shape[1]
            )
            return self, events

    def crop(self, tmin=0.0, tmax=None):
        """Crop raw data file.

        Limit the data from the raw file to go between specific times. Note
        that the new tmin is assumed to be t=0 for all subsequently called
        functions (e.g., time_as_index, or Epochs). New first_samp and
        last_samp are set accordingly.

        Parameters
        ----------
        tmin : float
            New start time in seconds (must be >= 0).
        tmax : float | None
            New end time in seconds of the data (cannot exceed data duration).

        Returns
        -------
        raw : instance of Raw
            The cropped raw object.
        """
        max_time = (self.n_times - 1) / self.info['sfreq']
        if tmax is None:
            tmax = max_time

        if tmin > tmax:
            raise ValueError('tmin must be less than tmax')
        if tmin < 0.0:
            raise ValueError('tmin must be >= 0')
        elif tmax > max_time:
            raise ValueError('tmax must be less than or equal to the max raw '
                             'time (%0.4f sec)' % max_time)

        smin, smax = np.where(_time_mask(self.times, tmin, tmax,
                                         sfreq=self.info['sfreq']))[0][[0, -1]]
        cumul_lens = np.concatenate(([0], np.array(self._raw_lengths,
                                                   dtype='int')))
        cumul_lens = np.cumsum(cumul_lens)
        keepers = np.logical_and(np.less(smin, cumul_lens[1:]),
                                 np.greater_equal(smax, cumul_lens[:-1]))
        keepers = np.where(keepers)[0]
        self._first_samps = np.atleast_1d(self._first_samps[keepers])
        # Adjust first_samp of first used file!
        self._first_samps[0] += smin - cumul_lens[keepers[0]]
        self._last_samps = np.atleast_1d(self._last_samps[keepers])
        self._last_samps[-1] -= cumul_lens[keepers[-1] + 1] - 1 - smax
        self._raw_extras = [r for ri, r in enumerate(self._raw_extras)
                            if ri in keepers]
        self._filenames = [r for ri, r in enumerate(self._filenames)
                           if ri in keepers]
        if self.preload:
            # slice and copy to avoid the reference to large array
            self._data = self._data[:, smin:smax + 1].copy()
        self._update_times()
        if self.annotations is not None:
            annotations = self.annotations
            annotations.onset -= tmin
            self.annotations = annotations
        return self

    @verbose
    def save(self, fname, picks=None, tmin=0, tmax=None, buffer_size_sec=None,
             drop_small_buffer=False, proj=False, fmt='single',
             overwrite=False, split_size='2GB', verbose=None):
        """Save raw data to file.

        Parameters
        ----------
        fname : string
            File name of the new dataset. This has to be a new filename
            unless data have been preloaded. Filenames should end with
            raw.fif, raw.fif.gz, raw_sss.fif, raw_sss.fif.gz, raw_tsss.fif
            or raw_tsss.fif.gz.
        picks : array-like of int | None
            Indices of channels to include. If None all channels are kept.
        tmin : float | None
            Time in seconds of first sample to save. If None first sample
            is used.
        tmax : float | None
            Time in seconds of last sample to save. If None last sample
            is used.
        buffer_size_sec : float | None
            Size of data chunks in seconds. If None (default), the buffer
            size of the original file is used.
        drop_small_buffer : bool
            Drop or not the last buffer. It is required by maxfilter (SSS)
            that only accepts raw files with buffers of the same size.
        proj : bool
            If True the data is saved with the projections applied (active).

            .. note:: If ``apply_proj()`` was used to apply the projections,
                      the projectons will be active even if ``proj`` is False.

        fmt : str
            Format to use to save raw data. Valid options are 'double',
            'single', 'int', and 'short' for 64- or 32-bit float, or 32- or
            16-bit integers, respectively. It is **strongly** recommended to
            use 'single', as this is backward-compatible, and is standard for
            maintaining precision. Note that using 'short' or 'int' may result
            in loss of precision, complex data cannot be saved as 'short',
            and neither complex data types nor real data stored as 'double'
            can be loaded with the MNE command-line tools. See raw.orig_format
            to determine the format the original data were stored in.
        overwrite : bool
            If True, the destination file (if it exists) will be overwritten.
            If False (default), an error will be raised if the file exists.
        split_size : string | int
            Large raw files are automatically split into multiple pieces. This
            parameter specifies the maximum size of each piece. If the
            parameter is an integer, it specifies the size in Bytes. It is
            also possible to pass a human-readable string, e.g., 100MB.

            .. note:: Due to FIFF file limitations, the maximum split
                      size is 2GB.

        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more). Defaults to self.verbose.

        Notes
        -----
        If Raw is a concatenation of several raw files, **be warned** that
        only the measurement information from the first raw file is stored.
        This likely means that certain operations with external tools may not
        work properly on a saved concatenated file (e.g., probably some
        or all forms of SSS). It is recommended not to concatenate and
        then save raw files for this reason.
        """
        check_fname(fname, 'raw', ('raw.fif', 'raw_sss.fif', 'raw_tsss.fif',
                                   'raw.fif.gz', 'raw_sss.fif.gz',
                                   'raw_tsss.fif.gz'))

        split_size = _get_split_size(split_size)

        fname = op.realpath(fname)
        if not self.preload and fname in self._filenames:
            raise ValueError('You cannot save data to the same file.'
                             ' Please use a different filename.')

        if self.preload:
            if np.iscomplexobj(self._data):
                warn('Saving raw file with complex data. Loading with '
                     'command-line MNE tools will not work.')

        type_dict = dict(short=FIFF.FIFFT_DAU_PACK16,
                         int=FIFF.FIFFT_INT,
                         single=FIFF.FIFFT_FLOAT,
                         double=FIFF.FIFFT_DOUBLE)
        if fmt not in type_dict.keys():
            raise ValueError('fmt must be "short", "int", "single", '
                             'or "double"')
        reset_dict = dict(short=False, int=False, single=True, double=True)
        reset_range = reset_dict[fmt]
        data_type = type_dict[fmt]

        data_test = self[0, 0][0]
        if fmt == 'short' and np.iscomplexobj(data_test):
            raise ValueError('Complex data must be saved as "single" or '
                             '"double", not "short"')

        # check for file existence
        _check_fname(fname, overwrite)

        if proj:
            info = copy.deepcopy(self.info)
            projector, info = setup_proj(info)
            activate_proj(info['projs'], copy=False)
        else:
            info = self.info
            projector = None

        #
        #   Set up the reading parameters
        #

        #   Convert to samples
        start = int(np.floor(tmin * self.info['sfreq']))

        # "stop" is the first sample *not* to save, so we need +1's here
        if tmax is None:
            stop = np.inf
        else:
            stop = self.time_as_index(float(tmax), use_rounding=True)[0] + 1
        stop = min(stop, self.last_samp - self.first_samp + 1)
        buffer_size = self._get_buffer_size(buffer_size_sec)

        # write the raw file
        _write_raw(fname, self, info, picks, fmt, data_type, reset_range,
                   start, stop, buffer_size, projector, drop_small_buffer,
                   split_size, 0, None)

    @copy_function_doc_to_method_doc(plot_raw)
    def plot(self, events=None, duration=10.0, start=0.0, n_channels=20,
             bgcolor='w', color=None, bad_color=(0.8, 0.8, 0.8),
             event_color='cyan', scalings=None, remove_dc=True, order='type',
             show_options=False, title=None, show=True, block=False,
             highpass=None, lowpass=None, filtorder=4, clipping=None,
             show_first_samp=False):
        return plot_raw(self, events, duration, start, n_channels, bgcolor,
                        color, bad_color, event_color, scalings, remove_dc,
                        order, show_options, title, show, block, highpass,
                        lowpass, filtorder, clipping, show_first_samp)

    @verbose
    @copy_function_doc_to_method_doc(plot_raw_psd)
    def plot_psd(self, tmin=0.0, tmax=None, fmin=0, fmax=np.inf,
                 proj=False, n_fft=None, picks=None, ax=None,
                 color='black', area_mode='std', area_alpha=0.33,
                 n_overlap=0, dB=True, average=None, show=True,
                 n_jobs=1, line_alpha=None, spatial_colors=None,
                 xscale='linear', verbose=None):
        if tmax is None:
            tmax = 60.
            warn('tmax defaults to 60. in 0.14 but will change to np.inf in '
                 '0.15. Set it explicitly to avoid this warning',
                 DeprecationWarning)
        return plot_raw_psd(
            self, tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, proj=proj,
            n_fft=n_fft, picks=picks, ax=ax, color=color, area_mode=area_mode,
            area_alpha=area_alpha, n_overlap=n_overlap, dB=dB, average=average,
            show=show, n_jobs=n_jobs, line_alpha=line_alpha,
            spatial_colors=spatial_colors, xscale=xscale)

    @copy_function_doc_to_method_doc(plot_raw_psd_topo)
    def plot_psd_topo(self, tmin=0., tmax=None, fmin=0, fmax=100, proj=False,
                      n_fft=2048, n_overlap=0, layout=None, color='w',
                      fig_facecolor='k', axis_facecolor='k', dB=True,
                      show=True, block=False, n_jobs=1, verbose=None):
        return plot_raw_psd_topo(self, tmin=tmin, tmax=tmax, fmin=fmin,
                                 fmax=fmax, proj=proj, n_fft=n_fft,
                                 n_overlap=n_overlap, layout=layout,
                                 color=color, fig_facecolor=fig_facecolor,
                                 axis_facecolor=axis_facecolor, dB=dB,
                                 show=show, block=block, n_jobs=n_jobs,
                                 verbose=verbose)

    def estimate_rank(self, tstart=0.0, tstop=30.0, tol=1e-4,
                      return_singular=False, picks=None, scalings='norm'):
        """Estimate rank of the raw data.

        This function is meant to provide a reasonable estimate of the rank.
        The true rank of the data depends on many factors, so use at your
        own risk.

        Parameters
        ----------
        tstart : float
            Start time to use for rank estimation. Default is 0.0.
        tstop : float | None
            End time to use for rank estimation. Default is 30.0.
            If None, the end time of the raw file is used.
        tol : float
            Tolerance for singular values to consider non-zero in
            calculating the rank. The singular values are calculated
            in this method such that independent data are expected to
            have singular value around one.
        return_singular : bool
            If True, also return the singular values that were used
            to determine the rank.
        picks : array_like of int, shape (n_selected_channels,)
            The channels to be considered for rank estimation.
            If None (default) meg and eeg channels are included.
        scalings : dict | 'norm'
            To achieve reliable rank estimation on multiple sensors,
            sensors have to be rescaled. This parameter controls the
            rescaling. If dict, it will update the
            following dict of defaults:

                dict(mag=1e11, grad=1e9, eeg=1e5)

            If 'norm' data will be scaled by internally computed
            channel-wise norms.
            Defaults to 'norm'.

        Returns
        -------
        rank : int
            Estimated rank of the data.
        s : array
            If return_singular is True, the singular values that were
            thresholded to determine the rank are also returned.

        Notes
        -----
        If data are not pre-loaded, the appropriate data will be loaded
        by this function (can be memory intensive).

        Projectors are not taken into account unless they have been applied
        to the data using apply_proj(), since it is not always possible
        to tell whether or not projectors have been applied previously.

        Bad channels will be excluded from calculations.
        """
        from ..cov import _estimate_rank_meeg_signals

        start = max(0, self.time_as_index(tstart)[0])
        if tstop is None:
            stop = self.n_times - 1
        else:
            stop = min(self.n_times - 1, self.time_as_index(tstop)[0])
        tslice = slice(start, stop + 1)
        if picks is None:
            picks = _pick_data_channels(self.info, exclude='bads',
                                        with_ref_meg=False)
        # ensure we don't get a view of data
        if len(picks) == 1:
            return 1.0, 1.0
        # this should already be a copy, so we can overwrite it
        data = self[picks, tslice][0]
        out = _estimate_rank_meeg_signals(
            data, pick_info(self.info, picks),
            scalings=scalings, tol=tol, return_singular=return_singular)
        return out

    @property
    def ch_names(self):
        """Channel names."""
        return self.info['ch_names']

    @property
    def times(self):
        """Time points."""
        return self._times

    @property
    def n_times(self):
        """Number of time points."""
        return self.last_samp - self.first_samp + 1

    def __len__(self):
        """The number of time points.

        Returns
        -------
        len : int
            The number of time points.

        Examples
        --------
        This can be used as::

            >>> len(raw)  # doctest: +SKIP
            1000

        """
        return self.n_times

    def load_bad_channels(self, bad_file=None, force=False):
        """Mark channels as bad from a text file.

        This function operates mostly in the style of the C function
        ``mne_mark_bad_channels``.

        Parameters
        ----------
        bad_file : string
            File name of the text file containing bad channels
            If bad_file = None, bad channels are cleared, but this
            is more easily done directly as raw.info['bads'] = [].
        force : boolean
            Whether or not to force bad channel marking (of those
            that exist) if channels are not found, instead of
            raising an error.
        """
        if bad_file is not None:
            # Check to make sure bad channels are there
            names = frozenset(self.info['ch_names'])
            with open(bad_file) as fid:
                bad_names = [l for l in fid.read().splitlines() if l]
            names_there = [ci for ci in bad_names if ci in names]
            count_diff = len(bad_names) - len(names_there)

            if count_diff > 0:
                if not force:
                    raise ValueError('Bad channels from:\n%s\n not found '
                                     'in:\n%s' % (bad_file,
                                                  self.filenames[0]))
                else:
                    warn('%d bad channels from:\n%s\nnot found in:\n%s'
                         % (count_diff, bad_file, self.filenames[0]))
            self.info['bads'] = names_there
        else:
            self.info['bads'] = []

    def append(self, raws, preload=None):
        """Concatenate raw instances as if they were continuous.

        Parameters
        ----------
        raws : list, or Raw instance
            list of Raw instances to concatenate to the current instance
            (in order), or a single raw instance to concatenate.
        preload : bool, str, or None (default None)
            Preload data into memory for data manipulation and faster indexing.
            If True, the data will be preloaded into memory (fast, requires
            large amount of memory). If preload is a string, preload is the
            file name of a memory-mapped file which is used to store the data
            on the hard drive (slower, requires less memory). If preload is
            None, preload=True or False is inferred using the preload status
            of the raw files passed in.
        """
        if not isinstance(raws, list):
            raws = [raws]

        # make sure the raws are compatible
        all_raws = [self]
        all_raws += raws
        _check_raw_compatibility(all_raws)

        # deal with preloading data first (while files are separate)
        all_preloaded = self.preload and all(r.preload for r in raws)
        if preload is None:
            if all_preloaded:
                preload = True
            else:
                preload = False

        if preload is False:
            if self.preload:
                self._data = None
            self.preload = False
        else:
            # do the concatenation ourselves since preload might be a string
            nchan = self.info['nchan']
            c_ns = np.cumsum([rr.n_times for rr in ([self] + raws)])
            nsamp = c_ns[-1]

            if not self.preload:
                this_data = self._read_segment()
            else:
                this_data = self._data

            # allocate the buffer
            if isinstance(preload, string_types):
                _data = np.memmap(preload, mode='w+', dtype=this_data.dtype,
                                  shape=(nchan, nsamp))
            else:
                _data = np.empty((nchan, nsamp), dtype=this_data.dtype)

            _data[:, 0:c_ns[0]] = this_data

            for ri in range(len(raws)):
                if not raws[ri].preload:
                    # read the data directly into the buffer
                    data_buffer = _data[:, c_ns[ri]:c_ns[ri + 1]]
                    raws[ri]._read_segment(data_buffer=data_buffer)
                else:
                    _data[:, c_ns[ri]:c_ns[ri + 1]] = raws[ri]._data
            self._data = _data
            self.preload = True

        # now combine information from each raw file to construct new self
        annotations = self.annotations
        for r in raws:
            self._first_samps = np.r_[self._first_samps, r._first_samps]
            self._last_samps = np.r_[self._last_samps, r._last_samps]
            self._raw_extras += r._raw_extras
            self._filenames += r._filenames
            annotations = _combine_annotations((annotations, r.annotations),
                                               self._last_samps,
                                               self._first_samps,
                                               self.info['sfreq'],
                                               self.info['meas_date'])

        self._update_times()
        self.annotations = annotations

        if not (len(self._first_samps) == len(self._last_samps) ==
                len(self._raw_extras) == len(self._filenames)):
            raise RuntimeError('Append error')  # should never happen

    def close(self):
        """Clean up the object.

        Does nothing for objects that close their file descriptors.
        Things like RawFIF will override this method.
        """
        pass

    def copy(self):
        """Return copy of Raw instance."""
        return deepcopy(self)

    def __repr__(self):  # noqa: D105
        name = self.filenames[0]
        name = 'None' if name is None else op.basename(name)
        size_str = str(sizeof_fmt(self._size))  # str in case it fails -> None
        size_str += ', data%s loaded' % ('' if self.preload else ' not')
        s = ('%s, n_channels x n_times : %s x %s (%0.1f sec), ~%s'
             % (name, len(self.ch_names), self.n_times, self.times[-1],
                size_str))
        return "<%s  |  %s>" % (self.__class__.__name__, s)

    def add_events(self, events, stim_channel=None):
        """Add events to stim channel.

        Parameters
        ----------
        events : ndarray, shape (n_events, 3)
            Events to add. The first column specifies the sample number of
            each event, the second column is ignored, and the third column
            provides the event value. If events already exist in the Raw
            instance at the given sample numbers, the event values will be
            added together.
        stim_channel : str | None
            Name of the stim channel to add to. If None, the config variable
            'MNE_STIM_CHANNEL' is used. If this is not found, it will default
            to 'STI 014'.

        Notes
        -----
        Data must be preloaded in order to add events.
        """
        if not self.preload:
            raise RuntimeError('cannot add events unless data are preloaded')
        events = np.asarray(events)
        if events.ndim != 2 or events.shape[1] != 3:
            raise ValueError('events must be shape (n_events, 3)')
        stim_channel = _get_stim_channel(stim_channel, self.info)
        pick = pick_channels(self.ch_names, stim_channel)
        if len(pick) == 0:
            raise ValueError('Channel %s not found' % stim_channel)
        pick = pick[0]
        idx = events[:, 0].astype(int)
        if np.any(idx < self.first_samp) or np.any(idx > self.last_samp):
            raise ValueError('event sample numbers must be between %s and %s'
                             % (self.first_samp, self.last_samp))
        if not all(idx == events[:, 0]):
            raise ValueError('event sample numbers must be integers')
        self._data[pick, idx - self.first_samp] += events[:, 2]

    def _get_buffer_size(self, buffer_size_sec=None):
        """Get the buffer size."""
        if buffer_size_sec is None:
            buffer_size_sec = self.info.get('buffer_size_sec', 1.)
        return int(np.ceil(buffer_size_sec * self.info['sfreq']))


def _check_preload(raw, msg):
    """Ensure data are preloaded."""
    if not raw.preload:
        raise RuntimeError(msg + ' requires raw data to be loaded. Use '
                           'preload=True (or string) in the constructor or '
                           'raw.load_data().')


def _allocate_data(data, data_buffer, data_shape, dtype):
    """Allocate data in memory or in memmap for preloading."""
    if data is None:
        # if not already done, allocate array with right type
        if isinstance(data_buffer, string_types):
            # use a memmap
            data = np.memmap(data_buffer, mode='w+',
                             dtype=dtype, shape=data_shape)
        else:
            data = np.zeros(data_shape, dtype=dtype)
    return data


def _index_as_time(index, sfreq, first_samp=0, use_first_samp=False):
    """Convert indices to time.

    Parameters
    ----------
    index : list-like | int
        List of ints or int representing points in time.
    use_first_samp : boolean
        If True, the time returned is relative to the session onset, else
        relative to the recording onset.

    Returns
    -------
    times : ndarray
        Times corresponding to the index supplied.
    """
    times = np.atleast_1d(index) + (first_samp if use_first_samp else 0)
    return times / sfreq


class _RawShell():
    """Create a temporary raw object."""

    def __init__(self):  # noqa: D102
        self.first_samp = None
        self.last_samp = None
        self._cals = None
        self._rawdir = None
        self._projector = None

    @property
    def n_times(self):  # noqa: D102
        return self.last_samp - self.first_samp + 1


###############################################################################
# Writing
def _write_raw(fname, raw, info, picks, fmt, data_type, reset_range, start,
               stop, buffer_size, projector, drop_small_buffer,
               split_size, part_idx, prev_fname):
    """Write raw file with splitting."""
    # we've done something wrong if we hit this
    n_times_max = len(raw.times)
    if start >= stop or stop > n_times_max:
        raise RuntimeError('Cannot write raw file with no data: %s -> %s '
                           '(max: %s) requested' % (start, stop, n_times_max))

    if part_idx > 0:
        # insert index in filename
        base, ext = op.splitext(fname)
        use_fname = '%s-%d%s' % (base, part_idx, ext)
    else:
        use_fname = fname
    logger.info('Writing %s' % use_fname)

    fid, cals = _start_writing_raw(use_fname, info, picks, data_type,
                                   reset_range, raw.annotations)
    use_picks = slice(None) if picks is None else picks

    first_samp = raw.first_samp + start
    if first_samp != 0:
        write_int(fid, FIFF.FIFF_FIRST_SAMPLE, first_samp)

    # previous file name and id
    if part_idx > 0 and prev_fname is not None:
        start_block(fid, FIFF.FIFFB_REF)
        write_int(fid, FIFF.FIFF_REF_ROLE, FIFF.FIFFV_ROLE_PREV_FILE)
        write_string(fid, FIFF.FIFF_REF_FILE_NAME, prev_fname)
        if info['meas_id'] is not None:
            write_id(fid, FIFF.FIFF_REF_FILE_ID, info['meas_id'])
        write_int(fid, FIFF.FIFF_REF_FILE_NUM, part_idx - 1)
        end_block(fid, FIFF.FIFFB_REF)

    pos_prev = fid.tell()
    if pos_prev > split_size:
        raise ValueError('file is larger than "split_size" after writing '
                         'measurement information, you must use a larger '
                         'value for split size: %s plus enough bytes for '
                         'the chosen buffer_size' % pos_prev)
    next_file_buffer = 2 ** 20  # extra cushion for last few post-data tags
    for first in range(start, stop, buffer_size):
        # Write blocks <= buffer_size in size
        last = min(first + buffer_size, stop)
        data, times = raw[use_picks, first:last]
        assert len(times) == last - first

        if projector is not None:
            data = np.dot(projector, data)

        if ((drop_small_buffer and (first > start) and
             (len(times) < buffer_size))):
            logger.info('Skipping data chunk due to small buffer ... '
                        '[done]')
            break
        logger.debug('Writing ...')
        _write_raw_buffer(fid, data, cals, fmt)

        pos = fid.tell()
        this_buff_size_bytes = pos - pos_prev
        overage = pos - split_size + next_file_buffer
        if overage > 0:
            # This should occur on the first buffer write of the file, so
            # we should mention the space required for the meas info
            raise ValueError(
                'buffer size (%s) is too large for the given split size (%s) '
                'by %s bytes after writing info (%s) and leaving enough space '
                'for end tags (%s): decrease "buffer_size_sec" or increase '
                '"split_size".' % (this_buff_size_bytes, split_size, overage,
                                   pos_prev, next_file_buffer))

        # Split files if necessary, leave some space for next file info
        # make sure we check to make sure we actually *need* another buffer
        # with the "and" check
        if pos >= split_size - this_buff_size_bytes - next_file_buffer and \
                first + buffer_size < stop:
            next_fname, next_idx = _write_raw(
                fname, raw, info, picks, fmt,
                data_type, reset_range, first + buffer_size, stop, buffer_size,
                projector, drop_small_buffer, split_size,
                part_idx + 1, use_fname)

            start_block(fid, FIFF.FIFFB_REF)
            write_int(fid, FIFF.FIFF_REF_ROLE, FIFF.FIFFV_ROLE_NEXT_FILE)
            write_string(fid, FIFF.FIFF_REF_FILE_NAME, op.basename(next_fname))
            if info['meas_id'] is not None:
                write_id(fid, FIFF.FIFF_REF_FILE_ID, info['meas_id'])
            write_int(fid, FIFF.FIFF_REF_FILE_NUM, next_idx)
            end_block(fid, FIFF.FIFFB_REF)
            break

        pos_prev = pos

    logger.info('Closing %s [done]' % use_fname)
    if info.get('maxshield', False):
        end_block(fid, FIFF.FIFFB_SMSH_RAW_DATA)
    else:
        end_block(fid, FIFF.FIFFB_RAW_DATA)
    end_block(fid, FIFF.FIFFB_MEAS)
    end_file(fid)
    return use_fname, part_idx


def _start_writing_raw(name, info, sel=None, data_type=FIFF.FIFFT_FLOAT,
                       reset_range=True, annotations=None):
    """Start write raw data in file.

    Parameters
    ----------
    name : string
        Name of the file to create.
    info : dict
        Measurement info.
    sel : array of int, optional
        Indices of channels to include. By default all channels are included.
    data_type : int
        The data_type in case it is necessary. Should be 4 (FIFFT_FLOAT),
        5 (FIFFT_DOUBLE), 16 (FIFFT_DAU_PACK16), or 3 (FIFFT_INT) for raw data.
    reset_range : bool
        If True, the info['chs'][k]['range'] parameter will be set to unity.
    annotations : instance of Annotations or None
        The annotations to write.

    Returns
    -------
    fid : file
        The file descriptor.
    cals : list
        calibration factors.
    """
    #
    # Measurement info
    #
    info = pick_info(info, sel)

    #
    # Create the file and save the essentials
    #
    fid = start_file(name)
    start_block(fid, FIFF.FIFFB_MEAS)
    write_id(fid, FIFF.FIFF_BLOCK_ID)
    if info['meas_id'] is not None:
        write_id(fid, FIFF.FIFF_PARENT_BLOCK_ID, info['meas_id'])

    cals = []
    for k in range(info['nchan']):
        #
        #   Scan numbers may have been messed up
        #
        info['chs'][k]['scanno'] = k + 1  # scanno starts at 1 in FIF format
        if reset_range is True:
            info['chs'][k]['range'] = 1.0
        cals.append(info['chs'][k]['cal'] * info['chs'][k]['range'])

    write_meas_info(fid, info, data_type=data_type, reset_range=reset_range)

    #
    # Annotations
    #
    if annotations is not None:
        start_block(fid, FIFF.FIFFB_MNE_ANNOTATIONS)
        write_float(fid, FIFF.FIFF_MNE_BASELINE_MIN, annotations.onset)
        write_float(fid, FIFF.FIFF_MNE_BASELINE_MAX,
                    annotations.duration + annotations.onset)
        # To allow : in description, they need to be replaced for serialization
        write_name_list(fid, FIFF.FIFF_COMMENT, [d.replace(':', ';') for d in
                                                 annotations.description])
        if annotations.orig_time is not None:
            write_double(fid, FIFF.FIFF_MEAS_DATE, annotations.orig_time)
        end_block(fid, FIFF.FIFFB_MNE_ANNOTATIONS)

    #
    # Start the raw data
    #
    if info.get('maxshield', False):
        start_block(fid, FIFF.FIFFB_SMSH_RAW_DATA)
    else:
        start_block(fid, FIFF.FIFFB_RAW_DATA)

    return fid, cals


def _write_raw_buffer(fid, buf, cals, fmt):
    """Write raw buffer.

    Parameters
    ----------
    fid : file descriptor
        an open raw data file.
    buf : array
        The buffer to write.
    cals : array
        Calibration factors.
    fmt : str
        'short', 'int', 'single', or 'double' for 16/32 bit int or 32/64 bit
        float for each item. This will be doubled for complex datatypes. Note
        that short and int formats cannot be used for complex data.
    """
    if buf.shape[0] != len(cals):
        raise ValueError('buffer and calibration sizes do not match')

    if fmt not in ['short', 'int', 'single', 'double']:
        raise ValueError('fmt must be "short", "single", or "double"')

    if np.isrealobj(buf):
        if fmt == 'short':
            write_function = write_dau_pack16
        elif fmt == 'int':
            write_function = write_int
        elif fmt == 'single':
            write_function = write_float
        else:
            write_function = write_double
    else:
        if fmt == 'single':
            write_function = write_complex64
        elif fmt == 'double':
            write_function = write_complex128
        else:
            raise ValueError('only "single" and "double" supported for '
                             'writing complex data')

    buf = buf / np.ravel(cals)[:, None]
    write_function(fid, FIFF.FIFF_DATA_BUFFER, buf)


def _my_hilbert(x, n_fft=None, envelope=False):
    """Compute Hilbert transform of signals w/ zero padding.

    Parameters
    ----------
    x : array, shape (n_times)
        The signal to convert
    n_fft : int
        Size of the FFT to perform, must be at least ``len(x)``.
        The signal will be cut back to original length.
    envelope : bool
        Whether to compute amplitude of the hilbert transform in order
        to return the signal envelope.

    Returns
    -------
    out : array, shape (n_times)
        The hilbert transform of the signal, or the envelope.
    """
    from scipy.signal import hilbert
    n_x = x.shape[-1]
    out = hilbert(x, N=n_fft)[:n_x]
    if envelope is True:
        out = np.abs(out)
    return out


def _check_raw_compatibility(raw):
    """Ensure all instances of Raw have compatible parameters."""
    for ri in range(1, len(raw)):
        if not isinstance(raw[ri], type(raw[0])):
            raise ValueError('raw[%d] type must match' % ri)
        if not raw[ri].info['nchan'] == raw[0].info['nchan']:
            raise ValueError('raw[%d][\'info\'][\'nchan\'] must match' % ri)
        if not raw[ri].info['bads'] == raw[0].info['bads']:
            raise ValueError('raw[%d][\'info\'][\'bads\'] must match' % ri)
        if not raw[ri].info['sfreq'] == raw[0].info['sfreq']:
            raise ValueError('raw[%d][\'info\'][\'sfreq\'] must match' % ri)
        if not set(raw[ri].info['ch_names']) == set(raw[0].info['ch_names']):
            raise ValueError('raw[%d][\'info\'][\'ch_names\'] must match' % ri)
        if not all(raw[ri]._cals == raw[0]._cals):
            raise ValueError('raw[%d]._cals must match' % ri)
        if len(raw[0].info['projs']) != len(raw[ri].info['projs']):
            raise ValueError('SSP projectors in raw files must be the same')
        if not all(_proj_equal(p1, p2) for p1, p2 in
                   zip(raw[0].info['projs'], raw[ri].info['projs'])):
            raise ValueError('SSP projectors in raw files must be the same')
    if not all(r.orig_format == raw[0].orig_format for r in raw):
        warn('raw files do not all have the same data format, could result in '
             'precision mismatch. Setting raw.orig_format="unknown"')
        raw[0].orig_format = 'unknown'


def concatenate_raws(raws, preload=None, events_list=None):
    """Concatenate raw instances as if they were continuous.

    .. note:: ``raws[0]`` is modified in-place to achieve the concatenation.

    Parameters
    ----------
    raws : list
        list of Raw instances to concatenate (in order).
    preload : bool, or None
        If None, preload status is inferred using the preload status of the
        raw files passed in. True or False sets the resulting raw file to
        have or not have data preloaded.
    events_list : None | list
        The events to concatenate. Defaults to None.

    Returns
    -------
    raw : instance of Raw
        The result of the concatenation (first Raw instance passed in).
    events : ndarray of int, shape (n events, 3)
        The events. Only returned if `event_list` is not None.
    """
    if events_list is not None:
        if len(events_list) != len(raws):
            raise ValueError('`raws` and `event_list` are required '
                             'to be of the same length')
        first, last = zip(*[(r.first_samp, r.last_samp) for r in raws])
        events = concatenate_events(events_list, first, last)
    raws[0].append(raws[1:], preload)

    if events_list is None:
        return raws[0]
    else:
        return raws[0], events


def _check_update_montage(info, montage, path=None, update_ch_names=False):
    """Help eeg readers to add montage."""
    if montage is not None:
        if not isinstance(montage, (string_types, Montage)):
            err = ("Montage must be str, None, or instance of Montage. "
                   "%s was provided" % type(montage))
            raise TypeError(err)
        if montage is not None:
            if isinstance(montage, string_types):
                montage = read_montage(montage, path=path)
            _set_montage(info, montage, update_ch_names=update_ch_names)

            missing_positions = []
            exclude = (FIFF.FIFFV_EOG_CH, FIFF.FIFFV_MISC_CH,
                       FIFF.FIFFV_STIM_CH)
            for ch in info['chs']:
                if not ch['kind'] in exclude:
                    if np.unique(ch['loc']).size == 1:
                        missing_positions.append(ch['ch_name'])

            # raise error if positions are missing
            if missing_positions:
                raise KeyError(
                    "The following positions are missing from the montage "
                    "definitions: %s. If those channels lack positions "
                    "because they are EOG channels use the eog parameter."
                    % str(missing_positions))


def _check_maxshield(allow_maxshield):
    """Warn or error about MaxShield."""
    msg = ('This file contains raw Internal Active '
           'Shielding data. It may be distorted. Elekta '
           'recommends it be run through MaxFilter to '
           'produce reliable results. Consider closing '
           'the file and running MaxFilter on the data.')
    if allow_maxshield:
        if not (isinstance(allow_maxshield, string_types) and
                allow_maxshield == 'yes'):
            warn(msg)
        allow_maxshield = 'yes'
    else:
        msg += (' Use allow_maxshield=True if you are sure you'
                ' want to load the data despite this warning.')
        raise ValueError(msg)
