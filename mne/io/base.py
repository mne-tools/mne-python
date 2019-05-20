# -*- coding: utf-8 -*-
# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Denis Engemann <denis.engemann@gmail.com>
#          Teon Brooks <teon.brooks@gmail.com>
#          Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)

from copy import deepcopy
import os
import os.path as op

import numpy as np

from .constants import FIFF
from .utils import _construct_bids_filename, _check_orig_units
from .pick import (pick_types, channel_type, pick_channels, pick_info,
                   _picks_to_idx)
from .meas_info import write_meas_info
from .proj import setup_proj, activate_proj, _proj_equal, ProjMixin
from ..channels.channels import (ContainsMixin, UpdateChannelsMixin,
                                 SetChannelsMixin, InterpolationMixin)
from ..channels.montage import read_montage, _set_montage, Montage
from .compensator import set_current_comp, make_compensator
from .write import (start_file, end_file, start_block, end_block,
                    write_dau_pack16, write_float, write_double,
                    write_complex64, write_complex128, write_int,
                    write_id, write_string, _get_split_size)

from ..annotations import (_annotations_starts_stops, _write_annotations,
                           _handle_meas_date)
from ..filter import (filter_data, notch_filter, resample,
                      _resample_stim_channels, _filt_check_picks,
                      _filt_update_info, _check_fun, HilbertMixin)
from ..parallel import parallel_func
from ..utils import (_check_fname, _check_pandas_installed, sizeof_fmt,
                     _check_pandas_index_arguments, _pl, fill_doc,
                     check_fname, _get_stim_channel, deprecated,
                     logger, verbose, _time_mask, warn, SizeMixin,
                     copy_function_doc_to_method_doc,
                     _check_preload, _get_argvalues, _check_option)
from ..viz import plot_raw, plot_raw_psd, plot_raw_psd_topo
from ..defaults import _handle_default
from ..event import find_events, concatenate_events
from ..annotations import Annotations, _combine_annotations, _sync_onset
from ..annotations import _ensure_annotation_object


def _set_pandas_dtype(df, columns, dtype):
    """Try to set the right columns to dtype."""
    for column in columns:
        df[column] = df[column].astype(dtype)
        logger.info('Converting "%s" to "%s"...' % (column, dtype))


class ToDataFrameMixin(object):
    """Class to add to_data_frame capabilities to certain classes."""

    @fill_doc
    def to_data_frame(self, picks=None, index=None, scaling_time=1e3,
                      scalings=None, copy=True, start=None, stop=None,
                      long_format=False):
        """Export data in tabular structure as a pandas DataFrame.

        Columns and indices will depend on the object being converted.
        Generally this will include as much relevant information as
        possible for the data type being converted. This makes it easy
        to convert data for use in packages that utilize dataframes,
        such as statsmodels or seaborn.

        Parameters
        ----------
        %(picks_all)s
        index : tuple of str | None
            Column to be used as index for the data. Valid string options
            are 'epoch', 'time' and 'condition'. If None, all three info
            columns will be included in the table as categorial data.
        scaling_time : float
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
        long_format : bool
            If True, the dataframe is returned in long format where each row
            is one observation of the signal at a unique coordinate of
            channels, time points, epochs and conditions. The number of
            factors depends on the data container. For convenience,
            a `ch_type` column is added when using this option that will
            facilitate subsetting the resulting dataframe.
            Defaults to False.

        Returns
        -------
        df : instance of pandas.DataFrame
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
        ch_map = None
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
                    ['{} {}'.format('LH' if ii < 1 else 'RH', vert)
                     for vert in vertno]
                    for ii, vertno in enumerate(self.vertices)]
                    for i in e]
            else:
                # volume source estimates
                col_names = ['VOL {}'.format(vert) for vert in self.vertices]
        elif isinstance(self, (BaseEpochs, BaseRaw, Evoked)):
            picks = _picks_to_idx(self.info, picks, 'all', exclude=())
            if isinstance(self, BaseEpochs):
                default_index = ['condition', 'epoch', 'time']
                data = self.get_data()[:, picks, :]
                times = self.times
                n_epochs, n_picks, n_times = data.shape
                data = np.hstack(data).T  # (time*epochs) x signals

                # Multi-index creation
                times = np.tile(times, n_epochs)
                id_swapped = {v: k for k, v in self.event_id.items()}
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
                data = data.T
                col_names = [self.ch_names[k] for k in picks]

            ch_types = [channel_type(self.info, idx) for idx in picks]
            ch_map = dict(
                zip([self.info['ch_names'][pp] for pp in picks],
                    ch_types))

            ch_types_used = list()
            scalings = _handle_default('scalings', scalings)
            for tt in scalings.keys():
                if tt in ch_types:
                    ch_types_used.append(tt)

            for tt in ch_types_used:
                scaling = scalings[tt]
                idx = [ii for ii in range(len(picks)) if ch_types[ii] == tt]
                if len(idx) > 0:
                    data[:, idx] *= scaling
        else:
            # In case some other object gets this mixin w/o an explicit check
            raise NameError('Object must be one of Raw, Epochs, Evoked,  or ' +
                            'SourceEstimate. This is {}'.format(type(self)))

        # Make sure that the time index is scaled correctly
        times = np.round(times * scaling_time)
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
            if 'time' in index and not long_format:
                _set_pandas_dtype(df, ['time'], np.int64)
            df.set_index(index, inplace=True)
        if all(i in default_index for i in index):
            if isinstance(self, _BaseSourceEstimate):
                df.columns.name = 'source'
            else:
                df.columns.name = 'channel'

        if long_format:
            df = df.stack().reset_index()
            columns = list(df.columns)
            sig_idx = columns.index(0)
            columns[sig_idx] = 'observation'
            df.columns = columns

            if not isinstance(self, _BaseSourceEstimate):
                df['ch_type'] = df.channel.map(ch_map)

            columns = list(df.columns)
            to_factor = [
                cc for cc in columns if cc not in ['observation', 'time']]
            _set_pandas_dtype(df, to_factor, 'category')

        return df


class TimeMixin(object):
    """Class to add sfreq and time_as_index capabilities to certain classes."""

    # Overridden method signature does not match call...
    def time_as_index(self, times, use_rounding=False):  # lgtm
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


@fill_doc
class BaseRaw(ProjMixin, ContainsMixin, UpdateChannelsMixin, SetChannelsMixin,
              InterpolationMixin, ToDataFrameMixin, TimeMixin, SizeMixin,
              HilbertMixin):
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
        Whatever data is necessary for on-demand reads for the given
        reader format.
    orig_format : str
        The data format of the original raw file (e.g., ``'double'``).
    dtype : dtype | None
        The dtype of the raw data. If preload is an ndarray, its dtype must
        match what is passed here.
    buffer_size_sec : float
        The buffer size in seconds that should be written by default using
        :meth:`mne.io.Raw.save`.
    orig_units : dict | None
        Dictionary mapping channel names to their units as specified in
        the header file. Example: {'FC1': 'nV'}

        .. versionadded:: 0.17
    %(verbose)s

    Notes
    -----
    This class is public to allow for stable type-checking in user
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
                 buffer_size_sec=1., orig_units=None,
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
            elif preload is not True and not isinstance(preload, str):
                raise ValueError('bad preload: %s' % preload)
            else:
                load_from_disk = True
        self._last_samps = np.array(last_samps)
        self._first_samps = np.array(first_samps)
        info._check_consistency()  # make sure subclass did a good job
        self.info = info
        self.buffer_size_sec = float(buffer_size_sec)
        cals = np.empty(info['nchan'])
        for k in range(info['nchan']):
            cals[k] = info['chs'][k]['range'] * info['chs'][k]['cal']
        bad = np.where(cals == 0)[0]
        if len(bad) > 0:
            raise ValueError('Bad cals for channels %s'
                             % {ii: self.ch_names[ii] for ii in bad})
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
        # Sanity check and set original units, if provided by the reader:
        if orig_units:
            if not isinstance(orig_units, dict):
                raise ValueError('orig_units must be of type dict, but got '
                                 ' {}'.format(type(orig_units)))

            # original units need to be truncated to 15 chars, which is what
            # the MNE IO procedure also does with the other channels
            orig_units_trunc = [ch[:15] for ch in orig_units]

            # STI 014 channel is native only to fif ... for all other formats
            # this was artificially added by the IO procedure, so remove it
            ch_names = list(info['ch_names'])
            if ('STI 014' in ch_names) and not \
               (self.filenames[0].endswith('.fif')):
                ch_names.remove('STI 014')

            # Each channel in the data must have a corresponding channel in
            # the original units.
            ch_correspond = [ch in orig_units_trunc for ch in ch_names]
            if not all(ch_correspond):
                ch_without_orig_unit = ch_names[ch_correspond.index(False)]
                raise ValueError('Channel {} has no associated original '
                                 'unit.'.format(ch_without_orig_unit))

            # Final check of orig_units, editing a unit if it is not a valid
            # unit
            orig_units = _check_orig_units(orig_units)
        self._orig_units = orig_units
        self._projectors = list()
        self._projector = None
        self._dtype_ = dtype
        self.set_annotations(None)
        # If we have True or a string, actually do the preloading
        self._update_times()
        if load_from_disk:
            self._preload_data(preload)
        self._init_kwargs = _get_argvalues()

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
        %(verbose_meth)s

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
        %(verbose_meth)s

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
        assert n_sel_channels <= self.info['nchan']
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
        elif isinstance(data_buffer, str):
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
        """Check if data segment is bad.

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
        if reject_by_annotation and len(self.annotations) > 0:
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
        %(verbose_meth)s

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
        data_buffer = preload if isinstance(preload, (str,
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
    def _last_time(self):
        return self.last_samp / float(self.info['sfreq'])

    # "Overridden method signature does not match call..." in LGTM
    def time_as_index(self, times, use_rounding=False, origin=None):  # lgtm
        """Convert time to indices.

        Parameters
        ----------
        times : list-like | float | int
            List of numbers or a number representing points in time.
        use_rounding : boolean
            If True, use rounding (instead of truncation) when converting
            times to indices. This can help avoid non-unique indices.
        origin: time-like | float | int | None
            Time reference for times. If None, ``times`` are assumed to be
            relative to ``first_samp``.

            .. versionadded:: 0.17.0

        Returns
        -------
        index : ndarray
            Indices relative to ``first_samp`` corresponding to the times
            supplied.
        """
        first_samp_in_abs_time = (_handle_meas_date(self.info['meas_date']) +
                                  self._first_time)
        if origin is None:
            origin = first_samp_in_abs_time

        absolute_time = np.atleast_1d(times) + _handle_meas_date(origin)
        times = (absolute_time - first_samp_in_abs_time)

        return super(BaseRaw, self).time_as_index(times, use_rounding)

    @property
    def _raw_lengths(self):
        return [l - f + 1 for f, l in zip(self._first_samps, self._last_samps)]

    @property
    def annotations(self):  # noqa: D401
        """:class:`~mne.Annotations` for marking segments of data."""
        return self._annotations

    @property
    def filenames(self):
        """The filenames used."""
        return tuple(self._filenames)

    def set_annotations(self, annotations, emit_warning=True):
        """Setter for annotations.

        This setter checks if they are inside the data range.

        Parameters
        ----------
        annotations : instance of mne.Annotations | None
            Annotations to set. If None, the annotations is defined
            but empty.
        emit_warning : bool
            Whether to emit warnings when limiting or omitting annotations.

        Returns
        -------
        self : instance of Raw
            The raw object with annotations.
        """
        meas_date = _handle_meas_date(self.info['meas_date'])
        if annotations is None:
            if self.info['meas_date'] is not None:
                orig_time = meas_date
            else:
                orig_time = None
            self._annotations = Annotations([], [], [], orig_time)
        else:
            _ensure_annotation_object(annotations)

            if self.info['meas_date'] is None and \
               annotations.orig_time is not None:
                raise RuntimeError('Ambiguous operation. Setting an Annotation'
                                   ' object with known ``orig_time`` to a raw'
                                   ' object which has ``meas_date`` set to'
                                   ' None is ambiguous. Please, either set a'
                                   ' meaningful ``meas_date`` to the raw'
                                   ' object; or set ``orig_time`` to None in'
                                   ' which case the annotation onsets would be'
                                   ' taken in reference to the first sample of'
                                   ' the raw object.')

            delta = 1. / self.info['sfreq']
            time_of_first_sample = meas_date + self.first_samp * delta
            new_annotations = annotations.copy()
            if annotations.orig_time is None:
                # Assume annotations to be relative to the data
                new_annotations.orig_time = time_of_first_sample

            tmin = time_of_first_sample
            tmax = tmin + self.times[-1] + delta
            new_annotations.crop(tmin=tmin, tmax=tmax,
                                 emit_warning=emit_warning)

            if self.info['meas_date'] is None:
                new_annotations.orig_time = None
            elif annotations.orig_time != meas_date:
                # XXX, TODO: this should be a function, method or something.
                # maybe orig_time should have a setter
                # new_annotations.orig_time = xxxxx # resets onset based on x
                # new_annotations._update_orig(xxxx)
                orig_time = new_annotations.orig_time
                new_annotations.orig_time = meas_date
                new_annotations.onset -= (meas_date - orig_time)

            self._annotations = new_annotations

        return self

    def __del__(self):  # noqa: D105
        # remove file for memmap
        if hasattr(self, '_data') and \
                getattr(self._data, 'filename', None) is not None:
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
        """Exit with block."""
        try:
            self.close()
        except Exception:
            return exception_type, exception_val, trace

    def _parse_get_set_params(self, item):
        """Parse the __getitem__ / __setitem__ tuples."""
        # make sure item is a tuple
        if not isinstance(item, tuple):  # only channel selection passed
            item = (item, slice(None, None, None))

        if len(item) != 2:  # should be channels and time instants
            raise RuntimeError("Unable to access raw data (need both channels "
                               "and time)")

        sel = _picks_to_idx(self.info, item[0])

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
        if step is not None and step != 1:
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

    @verbose
    def get_data(self, picks=None, start=0, stop=None,
                 reject_by_annotation=None, return_times=False, verbose=None):
        """Get data in the given range.

        Parameters
        ----------
        %(picks_all)s
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
            Whether to return times as well. Defaults to False.
        %(verbose_meth)s

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
        picks = _picks_to_idx(self.info, picks, 'all', exclude=())
        # convert to ints
        picks = np.atleast_1d(np.arange(self.info['nchan'])[picks])
        start = 0 if start is None else start
        stop = min(self.n_times if stop is None else stop, self.n_times)
        if len(self.annotations) == 0 or reject_by_annotation is None:
            data, times = self[picks, start:stop]
            return (data, times) if return_times else data
        _check_option('reject_by_annotation', reject_by_annotation.lower(),
                      ['omit', 'nan'])
        onsets, ends = _annotations_starts_stops(self, ['BAD'])
        keep = (onsets < stop) & (ends > start)
        onsets = np.maximum(onsets[keep], start)
        ends = np.minimum(ends[keep], stop)
        if len(onsets) == 0:
            data, times = self[picks, start:stop]
            if return_times:
                return data, times
            return data
        n_samples = stop - start  # total number of samples
        used = np.ones(n_samples, bool)
        for onset, end in zip(onsets, ends):
            if onset >= end:
                continue
            used[onset - start: end - start] = False
        used = np.concatenate([[False], used, [False]])
        starts = np.where(~used[:-1] & used[1:])[0] + start
        stops = np.where(used[:-1] & ~used[1:])[0] + start
        n_kept = (stops - starts).sum()  # kept samples
        n_rejected = n_samples - n_kept  # rejected samples
        if n_rejected > 0:
            if reject_by_annotation == 'omit':
                msg = ("Omitting {} of {} ({:.2%}) samples, retaining {}"
                       " ({:.2%}) samples.")
                logger.info(msg.format(n_rejected, n_samples,
                                       n_rejected / n_samples,
                                       n_kept, n_kept / n_samples))
                data = np.zeros((len(picks), n_kept))
                times = np.zeros(data.shape[1])
                idx = 0
                for start, stop in zip(starts, stops):  # get the data
                    if start == stop:
                        continue
                    end = idx + stop - start
                    data[:, idx:end], times[idx:end] = self[picks, start:stop]
                    idx = end
            else:
                msg = ("Setting {} of {} ({:.2%}) samples to NaN, retaining {}"
                       " ({:.2%}) samples.")
                logger.info(msg.format(n_rejected, n_samples,
                                       n_rejected / n_samples,
                                       n_kept, n_kept / n_samples))
                data, times = self[picks, start:stop]
                data[:, ~used[1:-1]] = np.nan
        else:
            data, times = self[picks, start:stop]

        if return_times:
            return data, times
        return data

    @verbose
    def apply_function(self, fun, picks=None, dtype=None, n_jobs=1,
                       channel_wise=True, *args, **kwargs):
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
        fun : callable
            A function to be applied to the channels. The first argument of
            fun has to be a timeseries (numpy.ndarray). The function must
            operate on an array of shape ``(n_times,)`` if
            ``channel_wise=True`` and ``(len(picks), n_times)`` otherwise.
            The function must return an ndarray shaped like its input.
        %(picks_all_data_noref)s
        dtype : numpy.dtype (default: None)
            Data type to use for raw data after applying the function. If None
            the data type is not modified.
        n_jobs: int (default: 1)
            Number of jobs to run in parallel. Ignored if `channel_wise` is
            False.
        channel_wise: bool (default: True)
            Whether to apply the function to each channel individually. If
            False, the function will be applied to all channels at once.

            .. versionadded:: 0.18
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
        picks = _picks_to_idx(self.info, picks, exclude=(), with_ref_meg=False)

        if not callable(fun):
            raise ValueError('fun needs to be a function')

        data_in = self._data
        if dtype is not None and dtype != self._data.dtype:
            self._data = self._data.astype(dtype)

        if channel_wise:
            if n_jobs == 1:
                # modify data inplace to save memory
                for idx in picks:
                    self._data[idx, :] = _check_fun(fun, data_in[idx, :],
                                                    *args, **kwargs)
            else:
                # use parallel function
                parallel, p_fun, _ = parallel_func(_check_fun, n_jobs)
                data_picks_new = parallel(
                    p_fun(fun, data_in[p], *args, **kwargs) for p in picks)
                for pp, p in enumerate(picks):
                    self._data[p, :] = data_picks_new[pp]
        else:
            self._data[picks, :] = _check_fun(
                fun, data_in[picks, :], *args, **kwargs)

        return self

    @verbose
    def filter(self, l_freq, h_freq, picks=None, filter_length='auto',
               l_trans_bandwidth='auto', h_trans_bandwidth='auto', n_jobs=1,
               method='fir', iir_params=None, phase='zero',
               fir_window='hamming', fir_design='firwin',
               skip_by_annotation=('edge', 'bad_acq_skip'),
               pad='reflect_limited', verbose=None):
        """Filter a subset of channels.

        Applies a zero-phase low-pass, high-pass, band-pass, or band-stop
        filter to the channels selected by ``picks``. By default the data
        of the Raw object is modified inplace.

        Parameters
        ----------
        %(l_freq)s
        %(h_freq)s
        %(picks_all_data)s
        %(filter_length)s
        %(l_trans_bandwidth)s
        %(h_trans_bandwidth)s
        %(n_jobs-fir)s
        %(method-fir)s
        %(iir_params)s
        %(phase)s
        %(fir_window)s
        %(fir_design)s
        skip_by_annotation : str | list of str
            If a string (or list of str), any annotation segment that begins
            with the given string will not be included in filtering, and
            segments on either side of the given excluded annotated segment
            will be filtered separately (i.e., as independent signals).
            The default (``('edge', 'bad_acq_skip')`` will separately filter
            any segments that were concatenated by :func:`mne.concatenate_raws`
            or :meth:`mne.io.Raw.append`, or separated during acquisition.
            To disable, provide an empty list.

            .. versionadded:: 0.16.
        %(pad-fir)s
            The default is ``'reflect-limited'``.

            .. versionadded:: 0.15
        %(verbose_meth)s

        Returns
        -------
        raw : instance of Raw
            The raw instance with filtered data.

        See Also
        --------
        mne.Epochs.savgol_filter
        mne.io.Raw.notch_filter
        mne.io.Raw.resample
        mne.filter.create_filter
        mne.filter.filter_data
        mne.filter.construct_iir_filter

        Notes
        -----
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

        For more information, see the tutorials
        :ref:`disc-filtering` and :ref:`tut-filter-resample` and
        :func:`mne.filter.create_filter`.
        """
        _check_preload(self, 'raw.filter')
        update_info, picks = _filt_check_picks(self.info, picks,
                                               l_freq, h_freq)
        # Deal with annotations
        onsets, ends = _annotations_starts_stops(
            self, skip_by_annotation, 'skip_by_annotation', invert=True)
        logger.info('Filtering raw data in %d contiguous segment%s'
                    % (len(onsets), _pl(onsets)))
        max_idx = (ends - onsets).argmax()
        for si, (start, stop) in enumerate(zip(onsets, ends)):
            # Only output filter params once (for info level), and only warn
            # once about the length criterion (longest segment is too short)
            use_verbose = verbose if si == max_idx else 'error'
            filter_data(
                self._data[:, start:stop], self.info['sfreq'], l_freq, h_freq,
                picks, filter_length, l_trans_bandwidth, h_trans_bandwidth,
                n_jobs, method, iir_params, copy=False, phase=phase,
                fir_window=fir_window, fir_design=fir_design, pad=pad,
                verbose=use_verbose)
        # update info if filter is applied to all data channels,
        # and it's not a band-stop filter
        _filt_update_info(self.info, update_info, l_freq, h_freq)
        return self

    @verbose
    def notch_filter(self, freqs, picks=None, filter_length='auto',
                     notch_widths=None, trans_bandwidth=1.0, n_jobs=1,
                     method='fir', iir_params=None, mt_bandwidth=None,
                     p_value=0.05, phase='zero', fir_window='hamming',
                     fir_design='firwin', pad='reflect_limited', verbose=None):
        """Notch filter a subset of channels.

        Parameters
        ----------
        freqs : float | array of float | None
            Specific frequencies to filter out from data, e.g.,
            np.arange(60, 241, 60) in the US or np.arange(50, 251, 50) in
            Europe. None can only be used with the mode 'spectrum_fit',
            where an F test is used to find sinusoidal components.
        %(picks_all_data)s
        %(filter_length)s
        notch_widths : float | array of float | None
            Width of each stop band (centred at each freq in freqs) in Hz.
            If None, freqs / 200 is used.
        trans_bandwidth : float
            Width of the transition band in Hz.
            Only used for ``method='fir'``.
        %(n_jobs-fir)s
        %(method-fir)s
        %(iir_params)s
        mt_bandwidth : float | None
            The bandwidth of the multitaper windowing function in Hz.
            Only used in 'spectrum_fit' mode.
        p_value : float
            p-value to use in F-test thresholding to determine significant
            sinusoidal components to remove when method='spectrum_fit' and
            freqs=None. Note that this will be Bonferroni corrected for the
            number of frequencies, so large p-values may be justified.
        %(phase)s
        %(fir_window)s
        %(fir_design)s
        %(pad-fir)s
            The default is ``'reflect_limited'``.

            .. versionadded:: 0.15
        %(verbose_meth)s

        Returns
        -------
        raw : instance of Raw
            The raw instance with filtered data.

        See Also
        --------
        mne.filter.notch_filter
        mne.io.Raw.filter

        Notes
        -----
        Applies a zero-phase notch filter to the channels selected by
        "picks". By default the data of the Raw object is modified inplace.

        The Raw object has to have the data loaded e.g. with ``preload=True``
        or ``self.load_data()``.

        .. note:: If n_jobs > 1, more memory is required as
                  ``len(picks) * n_times`` additional time points need to
                  be temporaily stored in memory.

        For details, see :func:`mne.filter.notch_filter`.
        """
        fs = float(self.info['sfreq'])
        picks = _picks_to_idx(self.info, picks, exclude=(), none='data_or_ica')
        _check_preload(self, 'raw.notch_filter')
        self._data = notch_filter(
            self._data, fs, freqs, filter_length=filter_length,
            notch_widths=notch_widths, trans_bandwidth=trans_bandwidth,
            method=method, iir_params=iir_params, mt_bandwidth=mt_bandwidth,
            p_value=p_value, picks=picks, n_jobs=n_jobs, copy=False,
            phase=phase, fir_window=fir_window, fir_design=fir_design,
            pad=pad)
        return self

    @verbose
    def resample(self, sfreq, npad='auto', window='boxcar', stim_picks=None,
                 n_jobs=1, events=None, pad='reflect_limited', verbose=None):
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
                     `this illustrative gist
                     <https://gist.github.com/larsoner/01642cb3789992fbca59>`_.

                     If resampling the continuous data is desired, it is
                     recommended to construct events using the original data.
                     The event onsets can be jointly resampled with the raw
                     data using the 'events' parameter (a resampled copy is
                     returned).

        Parameters
        ----------
        sfreq : float
            New sample rate to use.
        %(npad)s
        %(window-resample)s
        stim_picks : list of int | None
            Stim channels. These channels are simply subsampled or
            supersampled (without applying any filtering). This reduces
            resampling artifacts in stim channels, but may lead to missing
            triggers. If None, stim channels are automatically chosen using
            :func:`mne.pick_types`.
        %(n_jobs-cuda)s
        events : 2D array, shape (n_events, 3) | None
            An optional event matrix. When specified, the onsets of the events
            are resampled jointly with the data. NB: The input events are not
            modified, but a new array is returned with the raw instead.
        %(pad-fir)s
            The default is ``'reflect_limited'``.

            .. versionadded:: 0.15
        %(verbose_meth)s

        Returns
        -------
        raw : instance of Raw
            The resampled version of the raw object.
        events : array, shape (n_events, 3) | None
            If events are jointly resampled, these are returned with the raw.

        See Also
        --------
        mne.io.Raw.filter
        mne.Epochs.resample

        Notes
        -----
        For some data, it may be more accurate to use ``npad=0`` to reduce
        artifacts. This is dataset dependent -- check your data!
        """
        _check_preload(self, 'raw.resample')

        # When no event object is supplied, some basic detection of dropped
        # events is performed to generate a warning. Finding events can fail
        # for a variety of reasons, e.g. if no stim channel is present or it is
        # corrupted. This should not stop the resampling from working. The
        # warning should simply not be generated in this case.
        if events is None:
            try:
                original_events = find_events(self)
            except Exception:
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
                                     window=window, n_jobs=n_jobs, pad=pad))
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
            except Exception:
                pass

            return self
        else:
            # always make a copy of events
            events = events.copy()

            events[:, 0] = np.minimum(
                np.round(events[:, 0] * ratio).astype(int),
                self._data.shape[1] + self.first_samp - 1
            )
            return self, events

    def crop(self, tmin=0.0, tmax=None):
        """Crop raw data file.

        Limit the data from the raw file to go between specific times. Note
        that the new tmin is assumed to be t=0 for all subsequently called
        functions (e.g., time_as_index, or Epochs). New first_samp and
        last_samp are set accordingly.

        Thus function operates in-place on the instance.
        Use :meth:`mne.io.Raw.copy` if operation on a copy is desired.

        Parameters
        ----------
        tmin : float
            New start time in seconds (must be >= 0).
        tmax : float | None
            New end time in seconds of the data (cannot exceed data duration).

        Returns
        -------
        raw : instance of Raw
            The cropped raw object, modified in-place.
        """
        max_time = (self.n_times - 1) / self.info['sfreq']
        if tmax is None:
            tmax = max_time

        if tmin > tmax:
            raise ValueError('tmin (%s) must be less than tmax (%s)'
                             % (tmin, tmax))
        if tmin < 0.0:
            raise ValueError('tmin (%s) must be >= 0' % (tmin,))
        elif tmax > max_time:
            raise ValueError('tmax (%s) must be less than or equal to the max '
                             'time (%0.4f sec)' % (tmax, max_time))

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

        if self.annotations.orig_time is None:
            self.annotations.onset -= tmin
        # now call setter to filter out annotations outside of interval
        self.set_annotations(self.annotations, False)

        return self

    @verbose
    def save(self, fname, picks=None, tmin=0, tmax=None, buffer_size_sec=None,
             drop_small_buffer=False, proj=False, fmt='single',
             overwrite=False, split_size='2GB', split_naming='neuromag',
             verbose=None):
        """Save raw data to file.

        Parameters
        ----------
        fname : string
            File name of the new dataset. This has to be a new filename
            unless data have been preloaded. Filenames should end with
            raw.fif, raw.fif.gz, raw_sss.fif, raw_sss.fif.gz, raw_tsss.fif
            or raw_tsss.fif.gz.
        %(picks_all)s
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

        fmt : 'single' | 'double' | 'int' | 'short'
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
            To overwrite original file (the same one that was loaded),
            data must be preloaded upon reading.
        split_size : string | int
            Large raw files are automatically split into multiple pieces. This
            parameter specifies the maximum size of each piece. If the
            parameter is an integer, it specifies the size in Bytes. It is
            also possible to pass a human-readable string, e.g., 100MB.

            .. note:: Due to FIFF file limitations, the maximum split
                      size is 2GB.

        split_naming : {'neuromag' | 'bids'}
            Add the filename partition with the appropriate naming schema.

            .. versionadded:: 0.17

        %(verbose_meth)s

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
        _check_option('fmt', fmt, type_dict.keys())
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
            info = deepcopy(self.info)
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
        if split_naming == 'neuromag':
            part_idx = 0
        elif split_naming == 'bids':
            part_idx = 1
        else:
            raise ValueError(
                "split_naming must be either 'neuromag' or 'bids' instead "
                "of '{}'.".format(split_naming))
        _write_raw(fname, self, info, picks, fmt, data_type, reset_range,
                   start, stop, buffer_size, projector, drop_small_buffer,
                   split_size, split_naming, part_idx, None, overwrite)

    @copy_function_doc_to_method_doc(plot_raw)
    def plot(self, events=None, duration=10.0, start=0.0, n_channels=20,
             bgcolor='w', color=None, bad_color=(0.8, 0.8, 0.8),
             event_color='cyan', scalings=None, remove_dc=True, order=None,
             show_options=False, title=None, show=True, block=False,
             highpass=None, lowpass=None, filtorder=4, clipping=None,
             show_first_samp=False, proj=True, group_by='type',
             butterfly=False, decim='auto', noise_cov=None, event_id=None):
        return plot_raw(self, events, duration, start, n_channels, bgcolor,
                        color, bad_color, event_color, scalings, remove_dc,
                        order, show_options, title, show, block, highpass,
                        lowpass, filtorder, clipping, show_first_samp, proj,
                        group_by, butterfly, decim, noise_cov=noise_cov,
                        event_id=event_id)

    @verbose
    @copy_function_doc_to_method_doc(plot_raw_psd)
    def plot_psd(self, tmin=0.0, tmax=np.inf, fmin=0, fmax=np.inf,
                 proj=False, n_fft=None, picks=None, ax=None,
                 color='black', area_mode='std', area_alpha=0.33,
                 n_overlap=0, dB=True, estimate='auto', average=None,
                 show=True, n_jobs=1, line_alpha=None, spatial_colors=None,
                 xscale='linear', reject_by_annotation=True, verbose=None):
        return plot_raw_psd(
            self, tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, proj=proj,
            n_fft=n_fft, picks=picks, ax=ax, color=color, area_mode=area_mode,
            area_alpha=area_alpha, n_overlap=n_overlap, dB=dB,
            estimate=estimate, average=average, show=show, n_jobs=n_jobs,
            line_alpha=line_alpha, spatial_colors=spatial_colors,
            xscale=xscale, reject_by_annotation=reject_by_annotation)

    @copy_function_doc_to_method_doc(plot_raw_psd_topo)
    def plot_psd_topo(self, tmin=0., tmax=None, fmin=0, fmax=100, proj=False,
                      n_fft=2048, n_overlap=0, layout=None, color='w',
                      fig_facecolor='k', axis_facecolor='k', dB=True,
                      show=True, block=False, n_jobs=1, axes=None,
                      verbose=None):
        return plot_raw_psd_topo(self, tmin=tmin, tmax=tmax, fmin=fmin,
                                 fmax=fmax, proj=proj, n_fft=n_fft,
                                 n_overlap=n_overlap, layout=layout,
                                 color=color, fig_facecolor=fig_facecolor,
                                 axis_facecolor=axis_facecolor, dB=dB,
                                 show=show, block=block, n_jobs=n_jobs,
                                 axes=axes, verbose=verbose)

    @deprecated('raw.estimate_rank is deprecated and will be removed in 0.19, '
                'use mne.compute_rank instead.')
    @verbose
    def estimate_rank(self, tstart=0.0, tstop=30.0, tol=1e-4,
                      return_singular=False, picks=None, scalings='norm',
                      verbose=None):
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
        %(picks_good_data)s
        scalings : dict | 'norm' | None
            To achieve reliable rank estimation on multiple sensors,
            sensors have to be rescaled. This parameter controls the
            rescaling. If dict, it will update the
            following dict of defaults:

                dict(mag=1e11, grad=1e9, eeg=1e5)

            If 'norm' data will be scaled by internally computed
            channel-wise norms. None will perform no scaling.
            Defaults to 'norm'.
        %(verbose)s

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
        from ..rank import _estimate_rank_meeg_signals

        start = max(0, self.time_as_index(tstart)[0])
        if tstop is None:
            stop = self.n_times - 1
        else:
            stop = min(self.n_times - 1, self.time_as_index(tstop)[0])
        tslice = slice(start, stop + 1)
        picks = _picks_to_idx(self.info, picks, with_ref_meg=False)
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
        """Return the number of time points.

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

        .. note:: Boundaries of the raw files are annotated bad. If you wish to
                  use the data as continuous recording, you can remove the
                  boundary annotations after concatenation (see
                  :meth:`mne.Annotations.delete`).

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
            if isinstance(preload, str):
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
        edge_samps = list()
        for ri, r in enumerate(raws):
            n_samples = self.last_samp - self.first_samp + 1
            r_annot = Annotations(onset=r.annotations.onset - r._first_time,
                                  duration=r.annotations.duration,
                                  description=r.annotations.description,
                                  orig_time=None)
            annotations = _combine_annotations(
                annotations, r_annot, n_samples,
                self.first_samp, r.first_samp,
                self.info['sfreq'], self.info['meas_date'])
            edge_samps.append(sum(self._last_samps) -
                              sum(self._first_samps) + (ri + 1))
            self._first_samps = np.r_[self._first_samps, r._first_samps]
            self._last_samps = np.r_[self._last_samps, r._last_samps]
            self._raw_extras += r._raw_extras
            self._filenames += r._filenames
        self._update_times()
        self.set_annotations(annotations)
        for edge_samp in edge_samps:
            onset = _sync_onset(self, (edge_samp) / self.info['sfreq'], True)
            self.annotations.append(onset, 0., 'BAD boundary')
            self.annotations.append(onset, 0., 'EDGE boundary')
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

    def add_events(self, events, stim_channel=None, replace=False):
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
        replace : bool
            If True the old events on the stim channel are removed before
            adding the new ones.

        Notes
        -----
        Data must be preloaded in order to add events.
        """
        _check_preload(self, 'Adding events')
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
        if replace:
            self._data[pick, :] = 0.
        self._data[pick, idx - self.first_samp] += events[:, 2]

    def _get_buffer_size(self, buffer_size_sec=None):
        """Get the buffer size."""
        if buffer_size_sec is None:
            buffer_size_sec = self.buffer_size_sec
        buffer_size_sec = float(buffer_size_sec)
        return int(np.ceil(buffer_size_sec * self.info['sfreq']))


def _allocate_data(data, data_buffer, data_shape, dtype):
    """Allocate data in memory or in memmap for preloading."""
    if data is None:
        # if not already done, allocate array with right type
        if isinstance(data_buffer, str):
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


class _RawShell(object):
    """Create a temporary raw object."""

    def __init__(self):  # noqa: D102
        self.first_samp = None
        self.last_samp = None
        self._first_time = None
        self._last_time = None
        self._cals = None
        self._rawdir = None
        self._projector = None

    @property
    def n_times(self):  # noqa: D102
        return self.last_samp - self.first_samp + 1

    @property
    def annotations(self):  # noqa: D102
        return self._annotations

    def set_annotations(self, annotations):
        if annotations is None:
            annotations = Annotations([], [], [], None)
        self._annotations = annotations


###############################################################################
# Writing
def _write_raw(fname, raw, info, picks, fmt, data_type, reset_range, start,
               stop, buffer_size, projector, drop_small_buffer,
               split_size, split_naming, part_idx, prev_fname, overwrite):
    """Write raw file with splitting."""
    # we've done something wrong if we hit this
    n_times_max = len(raw.times)
    if start >= stop or stop > n_times_max:
        raise RuntimeError('Cannot write raw file with no data: %s -> %s '
                           '(max: %s) requested' % (start, stop, n_times_max))

    if part_idx > 0:
        base, ext = op.splitext(fname)
        if split_naming == 'neuromag':
            # insert index in filename
            use_fname = '%s-%d%s' % (base, part_idx, ext)
        elif split_naming == 'bids':
            use_fname = _construct_bids_filename(base, ext, part_idx)
            # check for file existence
            _check_fname(use_fname, overwrite)

    else:
        use_fname = fname
    logger.info('Writing %s' % use_fname)

    picks = _picks_to_idx(info, picks, 'all', ())
    fid, cals = _start_writing_raw(use_fname, info, picks, data_type,
                                   reset_range, raw.annotations)

    first_samp = raw.first_samp + start
    if first_samp != 0:
        write_int(fid, FIFF.FIFF_FIRST_SAMPLE, first_samp)

    # previous file name and id
    if split_naming == 'neuromag':
        part_idx_tag = part_idx - 1
    else:
        part_idx_tag = part_idx - 2
    if part_idx > 0 and prev_fname is not None:
        start_block(fid, FIFF.FIFFB_REF)
        write_int(fid, FIFF.FIFF_REF_ROLE, FIFF.FIFFV_ROLE_PREV_FILE)
        write_string(fid, FIFF.FIFF_REF_FILE_NAME, prev_fname)
        if info['meas_id'] is not None:
            write_id(fid, FIFF.FIFF_REF_FILE_ID, info['meas_id'])
        write_int(fid, FIFF.FIFF_REF_FILE_NUM, part_idx_tag)
        end_block(fid, FIFF.FIFFB_REF)

    pos_prev = fid.tell()
    if pos_prev > split_size:
        fid.close()
        raise ValueError('file is larger than "split_size" after writing '
                         'measurement information, you must use a larger '
                         'value for split size: %s plus enough bytes for '
                         'the chosen buffer_size' % pos_prev)
    next_file_buffer = 2 ** 20  # extra cushion for last few post-data tags

    # Check to see if this has acquisition skips and, if so, if we can
    # write out empty buffers instead of zeroes
    firsts = list(range(start, stop, buffer_size))
    lasts = np.array(firsts) + buffer_size
    if lasts[-1] > stop:
        lasts[-1] = stop
    sk_onsets, sk_ends = _annotations_starts_stops(raw, 'bad_acq_skip')
    do_skips = False
    if len(sk_onsets) > 0:
        if np.in1d(sk_onsets, firsts).all() and np.in1d(sk_ends, lasts).all():
            do_skips = True
        else:
            if part_idx == 0:
                warn('Acquisition skips detected but did not fit evenly into '
                     'output buffer_size, will be written as zeroes.')

    n_current_skip = 0
    for first, last in zip(firsts, lasts):
        if do_skips:
            if ((first >= sk_onsets) & (last <= sk_ends)).any():
                # Track how many we have
                n_current_skip += 1
                continue
            elif n_current_skip > 0:
                # Write out an empty buffer instead of data
                write_int(fid, FIFF.FIFF_DATA_SKIP, n_current_skip)
                # These two NOPs appear to be optional (MaxFilter does not do
                # it, but some acquisition machines do) so let's not bother.
                # write_nop(fid)
                # write_nop(fid)
                n_current_skip = 0
        data, times = raw[picks, first:last]
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
            fid.close()
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
                projector, drop_small_buffer, split_size, split_naming,
                part_idx + 1, use_fname, overwrite)

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


def _start_writing_raw(name, info, sel, data_type,
                       reset_range, annotations):
    """Start write raw data in file.

    Parameters
    ----------
    name : string
        Name of the file to create.
    info : dict
        Measurement info.
    sel : array of int | None
        Indices of channels to include. If None, all channels
        are included.
    data_type : int
        The data_type in case it is necessary. Should be 4 (FIFFT_FLOAT),
        5 (FIFFT_DOUBLE), 16 (FIFFT_DAU_PACK16), or 3 (FIFFT_INT) for raw data.
    reset_range : bool
        If True, the info['chs'][k]['range'] parameter will be set to unity.
    annotations : instance of Annotations
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
    # XXX do we need this?
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
    if len(annotations) > 0:  # don't save empty annot
        _write_annotations(fid, annotations)

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

    _check_option('fmt', fmt, ['short', 'int', 'single', 'double'])

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


@verbose
def concatenate_raws(raws, preload=None, events_list=None, verbose=None):
    """Concatenate raw instances as if they were continuous.

    .. note:: ``raws[0]`` is modified in-place to achieve the concatenation.
              Boundaries of the raw files are annotated bad. If you wish to use
              the data as continuous recording, you can remove the boundary
              annotations after concatenation (see
              :meth:`mne.Annotations.delete`).

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
    %(verbose)s

    Returns
    -------
    raw : instance of Raw
        The result of the concatenation (first Raw instance passed in).
    events : ndarray of int, shape (n_events, 3)
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


def _check_update_montage(info, montage, path=None, update_ch_names=False,
                          raise_missing=True):
    """Help eeg readers to add montage."""
    if montage is not None:
        if not isinstance(montage, (str, Montage)):
            err = ("Montage must be str, None, or instance of Montage. "
                   "%s was provided" % type(montage))
            raise TypeError(err)
        if montage is not None:
            if isinstance(montage, str):
                montage = read_montage(montage, path=path)
            _set_montage(info, montage, update_ch_names=update_ch_names)

            missing_positions = []
            exclude = (FIFF.FIFFV_EOG_CH, FIFF.FIFFV_MISC_CH,
                       FIFF.FIFFV_STIM_CH)
            for ch in info['chs']:
                if not ch['kind'] in exclude:
                    if not np.isfinite(ch['loc'][:3]).all():
                        missing_positions.append(ch['ch_name'])

            # raise error if positions are missing
            if missing_positions and raise_missing:
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
        if not (isinstance(allow_maxshield, str) and
                allow_maxshield == 'yes'):
            warn(msg)
    else:
        msg += (' Use allow_maxshield=True if you are sure you'
                ' want to load the data despite this warning.')
        raise ValueError(msg)
