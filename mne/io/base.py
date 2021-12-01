# -*- coding: utf-8 -*-
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Denis Engemann <denis.engemann@gmail.com>
#          Teon Brooks <teon.brooks@gmail.com>
#          Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#          Clemens Brunner <clemens.brunner@gmail.com>
#
# License: BSD-3-Clause

from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
import os
import os.path as op
import shutil
from collections import defaultdict

import numpy as np

from .constants import FIFF
from .utils import _construct_bids_filename, _check_orig_units
from .pick import (pick_types, pick_channels, pick_info, _picks_to_idx,
                   channel_type)
from .meas_info import write_meas_info, _ensure_infos_match
from .proj import setup_proj, activate_proj, _proj_equal, ProjMixin
from ..channels.channels import (ContainsMixin, UpdateChannelsMixin,
                                 SetChannelsMixin, InterpolationMixin,
                                 _unit2human)
from .compensator import set_current_comp, make_compensator
from .write import (start_file, end_file, start_block, end_block,
                    write_dau_pack16, write_float, write_double,
                    write_complex64, write_complex128, write_int,
                    write_id, write_string, _get_split_size, _NEXT_FILE_BUFFER)

from ..annotations import (_annotations_starts_stops, _write_annotations,
                           _handle_meas_date)
from ..filter import (FilterMixin, notch_filter, resample, _resamp_ratio_len,
                      _resample_stim_channels, _check_fun)
from ..parallel import parallel_func
from ..utils import (_check_fname, _check_pandas_installed, sizeof_fmt,
                     _check_pandas_index_arguments, fill_doc, copy_doc,
                     check_fname, _get_stim_channel, _stamp_to_dt,
                     logger, verbose, _time_mask, warn, SizeMixin,
                     copy_function_doc_to_method_doc, _validate_type,
                     _check_preload, _get_argvalues, _check_option,
                     _build_data_frame, _convert_times, _scale_dataframe_data,
                     _check_time_format, _arange_div)
from ..defaults import _handle_default
from ..viz import plot_raw, plot_raw_psd, plot_raw_psd_topo, _RAW_CLIP_DEF
from ..event import find_events, concatenate_events
from ..annotations import Annotations, _combine_annotations, _sync_onset
from ..data.html_templates import raw_template


class TimeMixin(object):
    """Class to add sfreq and time_as_index capabilities to certain classes."""

    # Overridden method signature does not match call...
    def time_as_index(self, times, use_rounding=False):  # lgtm
        """Convert time to indices.

        Parameters
        ----------
        times : list-like | float | int
            List of numbers or a number representing points in time.
        use_rounding : bool
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

    def _handle_tmin_tmax(self, tmin, tmax):
        """Convert seconds to index into data.

        Parameters
        ----------
        tmin : int | float | None
            Start time of data to get in seconds.
        tmax : int | float | None
            End time of data to get in seconds.

        Returns
        -------
        start : int
            Integer index into data corresponding to tmin.
        stop : int
            Integer index into data corresponding to tmax.

        """
        _validate_type(tmin, types=('numeric', None), item_name='tmin',
                       type_name="int, float, None")
        _validate_type(tmax, types=('numeric', None), item_name='tmax',
                       type_name='int, float, None')

        # handle tmin/tmax as start and stop indices into data array
        n_times = self.times.size
        start = 0 if tmin is None else self.time_as_index(tmin)[0]
        stop = n_times if tmax is None else self.time_as_index(tmax)[0]

        # truncate start/stop to the open interval [0, n_times]
        start = min(max(0, start), n_times)
        stop = min(max(0, stop), n_times)

        return start, stop


@fill_doc
class BaseRaw(ProjMixin, ContainsMixin, UpdateChannelsMixin, SetChannelsMixin,
              InterpolationMixin, TimeMixin, SizeMixin, FilterMixin):
    """Base class for Raw data.

    Parameters
    ----------
    %(info_not_none)s
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
    raw_extras : list of dict
        The data necessary for on-demand reads for the given reader format.
        Should be the same length as ``filenames``. Will have the entry
        ``raw_extras['orig_nchan']`` added to it for convenience.
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
        the header file. Example: {'FC1': 'nV'}.

        .. versionadded:: 0.17
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.

    Notes
    -----
    This class is public to allow for stable type-checking in user
    code (i.e., ``isinstance(my_raw_object, BaseRaw)``) but should not be used
    as a constructor for `Raw` objects (use instead one of the subclass
    constructors, or one of the ``mne.io.read_raw_*`` functions).

    Subclasses must provide the following methods:

        * _read_segment_file(self, data, idx, fi, start, stop, cals, mult)
          (only needed for types that support on-demand disk reads)
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
            if not preload:
                self.preload = False
                load_from_disk = False
            else:
                load_from_disk = True
        self._last_samps = np.array(last_samps)
        self._first_samps = np.array(first_samps)
        orig_ch_names = info['ch_names']
        with info._unlock(check_after=True):
            # be permissive of old code
            if isinstance(info['meas_date'], tuple):
                info['meas_date'] = _stamp_to_dt(info['meas_date'])
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
        self._raw_extras = list(dict() if r is None else r for r in raw_extras)
        for r in self._raw_extras:
            r['orig_nchan'] = info['nchan']
        self._read_picks = [np.arange(info['nchan'])
                            for _ in range(len(raw_extras))]
        # deal with compensation (only relevant for CTF data, either CTF
        # reader or MNE-C converted CTF->FIF files)
        self._read_comp_grade = self.compensation_grade  # read property
        if self._read_comp_grade is not None and len(info['comps']):
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

            # original units need to be truncated to 15 chars or renamed
            # to match MNE conventions (channel name unique and less than
            # 15 characters).
            orig_units = deepcopy(orig_units)
            for old_ch, new_ch in zip(orig_ch_names, info['ch_names']):
                if old_ch in orig_units:
                    this_unit = orig_units[old_ch]
                    del orig_units[old_ch]
                    orig_units[new_ch] = this_unit

            # STI 014 channel is native only to fif ... for all other formats
            # this was artificially added by the IO procedure, so remove it
            ch_names = list(info['ch_names'])
            if ('STI 014' in ch_names) and not \
                    (self.filenames[0].endswith('.fif')):
                ch_names.remove('STI 014')

            # Each channel in the data must have a corresponding channel in
            # the original units.
            ch_correspond = [ch in orig_units for ch in ch_names]
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

    @verbose
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
        if sel is None:
            n_out = self.info['nchan']
            idx = slice(None)
        else:
            n_out = len(sel)
            idx = _convert_slice(sel)
        del sel
        assert n_out <= self.info['nchan']
        data_shape = (n_out, stop - start)
        dtype = self._dtype
        if isinstance(data_buffer, np.ndarray):
            if data_buffer.shape != data_shape:
                raise ValueError('data_buffer has incorrect shape: %s != %s'
                                 % (data_buffer.shape, data_shape))
            data = data_buffer
        else:
            data = _allocate_data(data_buffer, data_shape, dtype)

        # deal with having multiple files accessed by the raw object
        cumul_lens = np.concatenate(([0], np.array(self._raw_lengths,
                                                   dtype='int')))
        cumul_lens = np.cumsum(cumul_lens)
        files_used = np.logical_and(np.less(start, cumul_lens[1:]),
                                    np.greater_equal(stop - 1,
                                                     cumul_lens[:-1]))

        # set up cals and mult (cals, compensation, and projector)
        n_out = len(np.arange(len(self.ch_names))[idx])
        cals = self._cals.ravel()[np.newaxis, :]
        if projector is not None:
            assert projector.shape[0] == projector.shape[1] == cals.shape[1]
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
        del projector

        if mult is None:
            cals = cals.T[idx]
            assert cals.shape == (n_out, 1)
            need_idx = idx  # sufficient just to read the given channels
        else:
            cals = None  # shouldn't be used
            assert mult.shape == (n_out, len(self.ch_names))
            # read all necessary for proj
            need_idx = np.where(np.any(mult, axis=0))[0]
            mult = mult[:, need_idx]
            logger.debug(
                f'Reading {len(need_idx)}/{len(self.ch_names)} channels '
                f'due to projection')
        assert (mult is None) ^ (cals is None)  # xor

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
            # reindex back to original file
            orig_idx = _convert_slice(self._read_picks[fi][need_idx])
            _ReadSegmentFileProtector(self)._read_segment_file(
                data[:, this_sl], orig_idx, fi,
                int(start_file), int(stop_file), cals, mult)
            offset += n_read
        return data

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a segment of data from a file.

        Only needs to be implemented for readers that support
        ``preload=False``. Any implementation should only make use of:

        - self._raw_extras[fi]
        - self._filenames[fi]

        So be sure to store any information necessary for reading raw data
        in self._raw_extras[fi]. Things like ``info`` can be decoupled
        from the original data (e.g., different subsets of channels) due
        to picking before preload, for example.

        Parameters
        ----------
        data : ndarray, shape (n_out, stop - start + 1)
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
        mult : ndarray, shape (n_out, len(idx) | None
            The compensation + projection + cals matrix, if applicable.
        """
        raise NotImplementedError

    def _check_bad_segment(self, start, stop, picks,
                           reject_start, reject_stop,
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
        reject_start : int
            First sample to check for overlaps with bad annotations.
        reject_stop : int
            Last sample to check for overlaps with bad annotations.
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
            overlaps = np.where(onset < reject_stop / sfreq)
            overlaps = np.where(onset[overlaps] + annot.duration[overlaps] >
                                reject_start / sfreq)
            for descr in annot.description[overlaps]:
                if descr.lower().startswith('bad'):
                    return descr
        return self._getitem((picks, slice(start, stop)), return_times=False)

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

    def _preload_data(self, preload):
        """Actually preload the data."""
        data_buffer = preload
        if isinstance(preload, (bool, np.bool_)) and not preload:
            data_buffer = None
        logger.info('Reading %d ... %d  =  %9.3f ... %9.3f secs...' %
                    (0, len(self.times) - 1, 0., self.times[-1]))
        self._data = self._read_segment(
            data_buffer=data_buffer, projector=self._projector)
        assert len(self._data) == self.info['nchan']
        self.preload = True
        self._comp = None  # no longer needed
        self.close()

    @property
    def _first_time(self):
        return self.first_samp / float(self.info['sfreq'])

    @property
    def first_samp(self):
        """The first data sample."""
        return self._first_samps[0]

    @property
    def first_time(self):
        """The first time point (including first_samp but not meas_date)."""
        return self._first_time

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
        use_rounding : bool
            If True, use rounding (instead of truncation) when converting
            times to indices. This can help avoid non-unique indices.
        origin : datetime | float | int | None
            Time reference for times. If None, ``times`` are assumed to be
            relative to ``first_samp``.

            .. versionadded:: 0.17.0

        Returns
        -------
        index : ndarray
            Indices relative to ``first_samp`` corresponding to the times
            supplied.
        """
        origin = _handle_meas_date(origin)
        if origin is None:
            delta = 0
        elif self.info['meas_date'] is None:
            raise ValueError('origin must be None when info["meas_date"] '
                             'is None, got %s' % (origin,))
        else:
            first_samp_in_abs_time = (self.info['meas_date'] +
                                      timedelta(0, self._first_time))
            delta = (origin - first_samp_in_abs_time).total_seconds()
        times = np.atleast_1d(times) + delta

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

    @verbose
    def set_annotations(self, annotations, emit_warning=True,
                        on_missing='raise', *, verbose=None):
        """Setter for annotations.

        This setter checks if they are inside the data range.

        Parameters
        ----------
        annotations : instance of mne.Annotations | None
            Annotations to set. If None, the annotations is defined
            but empty.
        emit_warning : bool
            Whether to emit warnings when cropping or omitting annotations.
        %(on_missing_ch_names)s
        %(verbose_meth)s

        Returns
        -------
        self : instance of Raw
            The raw object with annotations.
        """
        meas_date = _handle_meas_date(self.info['meas_date'])
        if annotations is None:
            self._annotations = Annotations([], [], [], meas_date)
        else:
            _validate_type(annotations, Annotations, 'annotations')

            if meas_date is None and annotations.orig_time is not None:
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
            new_annotations = annotations.copy()
            new_annotations._prune_ch_names(self.info, on_missing)
            if annotations.orig_time is None:
                new_annotations.crop(0, self.times[-1] + delta,
                                     emit_warning=emit_warning)
                new_annotations.onset += self._first_time
            else:
                tmin = meas_date + timedelta(0, self._first_time)
                tmax = tmin + timedelta(seconds=self.times[-1] + delta)
                new_annotations.crop(tmin=tmin, tmax=tmax,
                                     emit_warning=emit_warning)
                new_annotations.onset -= (
                    meas_date - new_annotations.orig_time).total_seconds()
            new_annotations._orig_time = meas_date

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
        return self._getitem(item)

    def _getitem(self, item, return_times=True):
        sel, start, stop = self._parse_get_set_params(item)
        if self.preload:
            data = self._data[sel, start:stop]
        else:
            data = self._read_segment(start=start, stop=stop, sel=sel,
                                      projector=self._projector)

        if return_times:
            # Rather than compute the entire thing just compute the subset
            # times = self.times[start:stop]
            # stop can be None here so don't use it directly
            times = np.arange(start, start + data.shape[1], dtype=float)
            times /= self.info['sfreq']
            return data, times
        else:
            return data

    def __setitem__(self, item, value):
        """Set raw data content."""
        _check_preload(self, 'Modifying data of Raw')
        sel, start, stop = self._parse_get_set_params(item)
        # set the data
        self._data[sel, start:stop] = value

    @verbose
    def get_data(self, picks=None, start=0, stop=None,
                 reject_by_annotation=None, return_times=False, units=None,
                 *, tmin=None, tmax=None, verbose=None):
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
        %(units)s
        tmin : int | float | None
            Start time of data to get in seconds. The ``tmin`` parameter is
            ignored if the ``start`` parameter is bigger than 0.

            .. versionadded:: 0.24.0
        tmax : int | float | None
            End time of data to get in seconds. The ``tmax`` parameter is
            ignored if the ``stop`` parameter is defined.

            .. versionadded:: 0.24.0
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
        # validate types
        _validate_type(start, types=('int-like'), item_name='start',
                       type_name='int')
        _validate_type(stop, types=('int-like', None), item_name='stop',
                       type_name='int, None')

        picks = _picks_to_idx(self.info, picks, 'all', exclude=())

        # Get channel factors for conversion into specified unit
        # (vector of ones if no conversion needed)
        if units is not None:
            ch_factors = _get_ch_factors(self, units, picks)

        # convert to ints
        picks = np.atleast_1d(np.arange(self.info['nchan'])[picks])

        # handle start/tmin stop/tmax
        tmin_start, tmax_stop = self._handle_tmin_tmax(tmin, tmax)

        # tmin/tmax are ignored if start/stop are defined to
        # something other than their defaults
        start = tmin_start if start == 0 else start
        stop = tmax_stop if stop is None else stop

        # truncate start/stop to the open interval [0, n_times]
        start = min(max(0, start), self.n_times)
        stop = min(max(0, stop), self.n_times)

        if len(self.annotations) == 0 or reject_by_annotation is None:
            getitem = self._getitem(
                (picks, slice(start, stop)), return_times=return_times)
            if return_times:
                data, times = getitem
                if units is not None:
                    data *= ch_factors[:, np.newaxis]
                return data, times
            if units is not None:
                getitem *= ch_factors[:, np.newaxis]
            return getitem
        _check_option('reject_by_annotation', reject_by_annotation.lower(),
                      ['omit', 'nan'])
        onsets, ends = _annotations_starts_stops(self, ['BAD'])
        keep = (onsets < stop) & (ends > start)
        onsets = np.maximum(onsets[keep], start)
        ends = np.minimum(ends[keep], stop)
        if len(onsets) == 0:
            data, times = self[picks, start:stop]
            if units is not None:
                data *= ch_factors[:, np.newaxis]
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

        if units is not None:
            data *= ch_factors[:, np.newaxis]
        if return_times:
            return data, times
        return data

    @verbose
    def apply_function(self, fun, picks=None, dtype=None, n_jobs=1,
                       channel_wise=True, verbose=None, **kwargs):
        """Apply a function to a subset of channels.

        %(applyfun_summary_raw)s

        Parameters
        ----------
        %(applyfun_fun)s
        %(picks_all_data_noref)s
        %(applyfun_dtype)s
        %(n_jobs)s
        %(applyfun_chwise)s

            .. versionadded:: 0.18
        %(verbose_meth)s
        %(kwarg_fun)s

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
                                                    **kwargs)
            else:
                # use parallel function
                parallel, p_fun, _ = parallel_func(_check_fun, n_jobs)
                data_picks_new = parallel(
                    p_fun(fun, data_in[p], **kwargs) for p in picks)
                for pp, p in enumerate(picks):
                    self._data[p, :] = data_picks_new[pp]
        else:
            self._data[picks, :] = _check_fun(
                fun, data_in[picks, :], **kwargs)

        return self

    # Need a separate method because the default pad is different for raw
    @copy_doc(FilterMixin.filter)
    def filter(self, l_freq, h_freq, picks=None, filter_length='auto',
               l_trans_bandwidth='auto', h_trans_bandwidth='auto', n_jobs=1,
               method='fir', iir_params=None, phase='zero',
               fir_window='hamming', fir_design='firwin',
               skip_by_annotation=('edge', 'bad_acq_skip'),
               pad='reflect_limited', verbose=None):  # noqa: D102
        return super().filter(
            l_freq, h_freq, picks, filter_length, l_trans_bandwidth,
            h_trans_bandwidth, n_jobs, method, iir_params, phase,
            fir_window, fir_design, skip_by_annotation, pad, verbose)

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
        %(filter_length_notch)s
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
            P-value to use in F-test thresholding to determine significant
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
                 n_jobs=1, events=None, pad='reflect_limited',
                 verbose=None):  # lgtm
        """Resample all channels.

        If appropriate, an anti-aliasing filter is applied before resampling.
        See :ref:`resampling-and-decimating` for more information.

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

        For optimum performance and to make use of ``n_jobs > 1``, the raw
        object has to have the data loaded e.g. with ``preload=True`` or
        ``self.load_data()``, but this increases memory requirements. The
        resulting raw object will have the data loaded into memory.
        """
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

        # set up stim channel processing
        if stim_picks is None:
            stim_picks = pick_types(self.info, meg=False, ref_meg=False,
                                    stim=True, exclude=[])
        else:
            stim_picks = _picks_to_idx(self.info, stim_picks, exclude=(),
                                       with_ref_meg=False)

        kwargs = dict(up=sfreq, down=o_sfreq, npad=npad, window=window,
                      n_jobs=n_jobs, pad=pad)
        ratio, n_news = zip(*(_resamp_ratio_len(sfreq, o_sfreq, old_len)
                              for old_len in self._raw_lengths))
        ratio, n_news = ratio[0], np.array(n_news, int)
        new_offsets = np.cumsum([0] + list(n_news))
        if self.preload:
            new_data = np.empty(
                (len(self.ch_names), new_offsets[-1]), self._data.dtype)
        for ri, (n_orig, n_new) in enumerate(zip(self._raw_lengths, n_news)):
            this_sl = slice(new_offsets[ri], new_offsets[ri + 1])
            if self.preload:
                data_chunk = self._data[:, offsets[ri]:offsets[ri + 1]]
                new_data[:, this_sl] = resample(data_chunk, **kwargs)
                # In empirical testing, it was faster to resample all channels
                # (above) and then replace the stim channels than it was to
                # only resample the proper subset of channels and then use
                # np.insert() to restore the stims.
                if len(stim_picks) > 0:
                    new_data[stim_picks, this_sl] = _resample_stim_channels(
                        data_chunk[stim_picks], n_new, data_chunk.shape[1])
            else:  # this will not be I/O efficient, but will be mem efficient
                for ci in range(len(self.ch_names)):
                    data_chunk = self.get_data(
                        ci, offsets[ri], offsets[ri + 1], verbose='error')[0]
                    if ci == 0 and ri == 0:
                        new_data = np.empty(
                            (len(self.ch_names), new_offsets[-1]),
                            data_chunk.dtype)
                    if ci in stim_picks:
                        resamp = _resample_stim_channels(
                            data_chunk, n_new, data_chunk.shape[-1])[0]
                    else:
                        resamp = resample(data_chunk, **kwargs)
                    new_data[ci, this_sl] = resamp

        self._first_samps = (self._first_samps * ratio).astype(int)
        self._last_samps = (np.array(self._first_samps) + n_news - 1)
        self._raw_lengths[ri] = list(n_news)
        assert np.array_equal(n_news, self._last_samps - self._first_samps + 1)
        self._data = new_data
        self.preload = True
        lowpass = self.info.get('lowpass')
        lowpass = np.inf if lowpass is None else lowpass
        with self.info._unlock():
            self.info['lowpass'] = min(lowpass, sfreq / 2.)
            self.info['sfreq'] = sfreq

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

    @fill_doc
    def crop(self, tmin=0.0, tmax=None, include_tmax=True):
        """Crop raw data file.

        Limit the data from the raw file to go between specific times. Note
        that the new tmin is assumed to be t=0 for all subsequently called
        functions (e.g., time_as_index, or Epochs). New first_samp and
        last_samp are set accordingly.

        Thus function operates in-place on the instance.
        Use :meth:`mne.io.Raw.copy` if operation on a copy is desired.

        Parameters
        ----------
        %(raw_tmin)s
        %(raw_tmax)s
        %(include_tmax)s

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

        smin, smax = np.where(_time_mask(
            self.times, tmin, tmax, sfreq=self.info['sfreq'],
            include_tmax=include_tmax))[0][[0, -1]]
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
        self._read_picks = [self._read_picks[ri] for ri in keepers]
        assert all(len(r) == len(self._read_picks[0])
                   for r in self._read_picks)
        self._raw_extras = [self._raw_extras[ri] for ri in keepers]
        self._filenames = [self._filenames[ri] for ri in keepers]
        if self.preload:
            # slice and copy to avoid the reference to large array
            self._data = self._data[:, smin:smax + 1].copy()

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
        fname : str
            File name of the new dataset. This has to be a new filename
            unless data have been preloaded. Filenames should end with
            ``raw.fif`` (common raw data), ``raw_sss.fif``
            (Maxwell-filtered continuous data),
            ``raw_tsss.fif`` (temporally signal-space-separated data),
            ``_meg.fif`` (common MEG data), ``_eeg.fif`` (common EEG data),
            or ``_ieeg.fif`` (common intracranial EEG data). You may also
            append an additional ``.gz`` suffix to enable gzip compression.
        %(picks_all)s
        %(raw_tmin)s
        %(raw_tmax)s
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
        %(overwrite)s
            To overwrite original file (the same one that was loaded),
            data must be preloaded upon reading.
        split_size : str | int
            Large raw files are automatically split into multiple pieces. This
            parameter specifies the maximum size of each piece. If the
            parameter is an integer, it specifies the size in Bytes. It is
            also possible to pass a human-readable string, e.g., 100MB.

            .. note:: Due to FIFF file limitations, the maximum split
                      size is 2GB.
        %(split_naming)s

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
        endings = ('raw.fif', 'raw_sss.fif', 'raw_tsss.fif',
                   '_meg.fif', '_eeg.fif', '_ieeg.fif')
        endings += tuple([f'{e}.gz' for e in endings])
        endings_err = ('.fif', '.fif.gz')

        # convert to str, check for overwrite a few lines later
        fname = _check_fname(fname, overwrite=True)
        check_fname(fname, 'raw', endings, endings_err=endings_err)

        split_size = _get_split_size(split_size)
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

        # check for file existence and expand `~` if present
        fname = _check_fname(fname=fname, overwrite=overwrite)

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
        start, stop = self._tmin_tmax_to_start_stop(tmin, tmax)
        buffer_size = self._get_buffer_size(buffer_size_sec)

        # write the raw file
        _validate_type(split_naming, str, 'split_naming')
        _check_option('split_naming', split_naming, ('neuromag', 'bids'))
        _write_raw(fname, self, info, picks, fmt, data_type, reset_range,
                   start, stop, buffer_size, projector, drop_small_buffer,
                   split_size, split_naming, 0, None, overwrite)

    @verbose
    def export(self, fname, fmt='auto', physical_range='auto',
               add_ch_type=False, *, overwrite=False, verbose=None):
        """Export Raw to external formats.

        Supported formats: EEGLAB (set, uses :mod:`eeglabio`)

        %(export_warning)s

        Parameters
        ----------
        %(export_params_fname)s
        %(export_params_fmt)s
        %(export_params_physical_range)s
        %(export_params_add_ch_type)s
        %(overwrite)s

            .. versionadded:: 0.24.1
        %(verbose)s

        Notes
        -----
        .. versionadded:: 0.24

        %(export_warning_note_raw)s
        %(export_eeglab_note)s
        %(export_edf_note)s
        """
        from ..export import export_raw
        export_raw(fname, self, fmt, physical_range=physical_range,
                   add_ch_type=add_ch_type, overwrite=overwrite,
                   verbose=verbose)

    def _tmin_tmax_to_start_stop(self, tmin, tmax):
        start = int(np.floor(tmin * self.info['sfreq']))

        # "stop" is the first sample *not* to save, so we need +1's here
        if tmax is None:
            stop = np.inf
        else:
            stop = self.time_as_index(float(tmax), use_rounding=True)[0] + 1
        stop = min(stop, self.last_samp - self.first_samp + 1)
        if stop <= start or stop <= 0:
            raise ValueError('tmin (%s) and tmax (%s) yielded no samples'
                             % (tmin, tmax))
        return start, stop

    @copy_function_doc_to_method_doc(plot_raw)
    def plot(self, events=None, duration=10.0, start=0.0, n_channels=20,
             bgcolor='w', color=None, bad_color='lightgray',
             event_color='cyan', scalings=None, remove_dc=True, order=None,
             show_options=False, title=None, show=True, block=False,
             highpass=None, lowpass=None, filtorder=4, clipping=_RAW_CLIP_DEF,
             show_first_samp=False, proj=True, group_by='type',
             butterfly=False, decim='auto', noise_cov=None, event_id=None,
             show_scrollbars=True, show_scalebars=True, time_format='float',
             precompute='auto', use_opengl=None, verbose=None):
        return plot_raw(self, events, duration, start, n_channels, bgcolor,
                        color, bad_color, event_color, scalings, remove_dc,
                        order, show_options, title, show, block, highpass,
                        lowpass, filtorder, clipping, show_first_samp,
                        proj, group_by, butterfly, decim, noise_cov=noise_cov,
                        event_id=event_id, show_scrollbars=show_scrollbars,
                        show_scalebars=show_scalebars, time_format=time_format,
                        precompute=precompute, use_opengl=use_opengl,
                        verbose=verbose)

    @verbose
    @copy_function_doc_to_method_doc(plot_raw_psd)
    def plot_psd(self, fmin=0, fmax=np.inf, tmin=None, tmax=None, proj=False,
                 n_fft=None, n_overlap=0, reject_by_annotation=True,
                 picks=None, ax=None, color='black', xscale='linear',
                 area_mode='std', area_alpha=0.33, dB=True, estimate='auto',
                 show=True, n_jobs=1, average=False, line_alpha=None,
                 spatial_colors=True, sphere=None, window='hamming',
                 exclude='bads', verbose=None):
        return plot_raw_psd(self, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax,
                            proj=proj, n_fft=n_fft, n_overlap=n_overlap,
                            reject_by_annotation=reject_by_annotation,
                            picks=picks, ax=ax, color=color, xscale=xscale,
                            area_mode=area_mode, area_alpha=area_alpha,
                            dB=dB, estimate=estimate, show=show, n_jobs=n_jobs,
                            average=average, line_alpha=line_alpha,
                            spatial_colors=spatial_colors, sphere=sphere,
                            window=window, exclude=exclude, verbose=verbose)

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

    @property
    def ch_names(self):
        """Channel names."""
        return self.info['ch_names']

    @property
    def times(self):
        """Time points."""
        out = _arange_div(self.n_times, float(self.info['sfreq']))
        out.flags['WRITEABLE'] = False
        return out

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
        ``mne_mark_bad_channels``. Each line in the text file will be
        interpreted as a name of a bad channel.

        Parameters
        ----------
        bad_file : str
            File name of the text file containing bad channels
            If bad_file = None, bad channels are cleared, but this
            is more easily done directly as raw.info['bads'] = [].
        force : bool
            Whether or not to force bad channel marking (of those
            that exist) if channels are not found, instead of
            raising an error.
        """
        if bad_file is not None:
            # Check to make sure bad channels are there
            names = frozenset(self.info['ch_names'])
            with open(bad_file) as fid:
                bad_names = [line for line in fid.read().splitlines() if line]
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

    @fill_doc
    def append(self, raws, preload=None):
        """Concatenate raw instances as if they were continuous.

        .. note:: Boundaries of the raw files are annotated bad. If you wish to
                  use the data as continuous recording, you can remove the
                  boundary annotations after concatenation (see
                  :meth:`mne.Annotations.delete`).

        Parameters
        ----------
        raws : list, or Raw instance
            List of Raw instances to concatenate to the current instance
            (in order), or a single raw instance to concatenate.
        %(preload_concatenate)s
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
                this_data = self._read_segment(projector=self._projector)
            else:
                this_data = self._data

            # allocate the buffer
            _data = _allocate_data(preload, (nchan, nsamp), this_data.dtype)
            _data[:, 0:c_ns[0]] = this_data

            for ri in range(len(raws)):
                if not raws[ri].preload:
                    # read the data directly into the buffer
                    data_buffer = _data[:, c_ns[ri]:c_ns[ri + 1]]
                    raws[ri]._read_segment(data_buffer=data_buffer,
                                           projector=self._projector)
                else:
                    _data[:, c_ns[ri]:c_ns[ri + 1]] = raws[ri]._data
            self._data = _data
            self.preload = True

        # now combine information from each raw file to construct new self
        annotations = self.annotations
        assert annotations.orig_time == self.info['meas_date']
        edge_samps = list()
        for ri, r in enumerate(raws):
            n_samples = self.last_samp - self.first_samp + 1
            annotations = _combine_annotations(
                annotations, r.annotations, n_samples,
                self.first_samp, r.first_samp,
                self.info['sfreq'], self.info['meas_date'])
            edge_samps.append(sum(self._last_samps) -
                              sum(self._first_samps) + (ri + 1))
            self._first_samps = np.r_[self._first_samps, r._first_samps]
            self._last_samps = np.r_[self._last_samps, r._last_samps]
            self._read_picks += r._read_picks
            self._raw_extras += r._raw_extras
            self._filenames += r._filenames
        assert annotations.orig_time == self.info['meas_date']
        self.set_annotations(annotations)
        for edge_samp in edge_samps:
            onset = _sync_onset(self, (edge_samp) / self.info['sfreq'], True)
            self.annotations.append(onset, 0., 'BAD boundary')
            self.annotations.append(onset, 0., 'EDGE boundary')
        if not (len(self._first_samps) == len(self._last_samps) ==
                len(self._raw_extras) == len(self._filenames) ==
                len(self._read_picks)):
            raise RuntimeError('Append error')  # should never happen

    def close(self):
        """Clean up the object.

        Does nothing for objects that close their file descriptors.
        Things like RawFIF will override this method.
        """
        pass  # noqa

    def copy(self):
        """Return copy of Raw instance.

        Returns
        -------
        inst : instance of Raw
            A copy of the instance.
        """
        return deepcopy(self)

    def __repr__(self):  # noqa: D105
        name = self.filenames[0]
        name = '' if name is None else op.basename(name) + ', '
        size_str = str(sizeof_fmt(self._size))  # str in case it fails -> None
        size_str += ', data%s loaded' % ('' if self.preload else ' not')
        s = ('%s%s x %s (%0.1f s), ~%s'
             % (name, len(self.ch_names), self.n_times, self.times[-1],
                size_str))
        return "<%s | %s>" % (self.__class__.__name__, s)

    def _repr_html_(self, caption=None):
        basenames = [
            os.path.basename(f) for f in self._filenames if f is not None
        ]
        m, s = divmod(self._last_time - self.first_time, 60)
        h, m = divmod(m, 60)
        duration = f'{int(h):02d}:{int(m):02d}:{int(s):02d}'
        return raw_template.substitute(
            info_repr=self.info._repr_html_(caption=caption),
            filenames=basenames, duration=duration)

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

    @fill_doc
    def to_data_frame(self, picks=None, index=None,
                      scalings=None, copy=True, start=None, stop=None,
                      long_format=False, time_format='ms'):
        """Export data in tabular structure as a pandas DataFrame.

        Channels are converted to columns in the DataFrame. By default, an
        additional column "time" is added, unless ``index`` is not ``None``
        (in which case time values form the DataFrame's index).

        Parameters
        ----------
        %(picks_all)s
        %(df_index_raw)s
            Defaults to ``None``.
        %(df_scalings)s
        %(df_copy)s
        start : int | None
            Starting sample index for creating the DataFrame from a temporal
            span of the Raw object. ``None`` (the default) uses the first
            sample.
        stop : int | None
            Ending sample index for creating the DataFrame from a temporal span
            of the Raw object. ``None`` (the default) uses the last sample.
        %(df_longform_raw)s
        %(df_time_format_raw)s

            .. versionadded:: 0.20

        Returns
        -------
        %(df_return)s
        """
        # check pandas once here, instead of in each private utils function
        pd = _check_pandas_installed()  # noqa
        # arg checking
        valid_index_args = ['time']
        valid_time_formats = ['ms', 'timedelta', 'datetime']
        index = _check_pandas_index_arguments(index, valid_index_args)
        time_format = _check_time_format(time_format, valid_time_formats,
                                         self.info['meas_date'])
        # get data
        picks = _picks_to_idx(self.info, picks, 'all', exclude=())
        data, times = self[picks, start:stop]
        data = data.T
        if copy:
            data = data.copy()
        data = _scale_dataframe_data(self, data, picks, scalings)
        # prepare extra columns / multiindex
        mindex = list()
        times = _convert_times(self, times, time_format)
        mindex.append(('time', times))
        # build DataFrame
        df = _build_data_frame(self, data, picks, long_format, mindex, index,
                               default_index=['time'])
        return df

    def describe(self, data_frame=False):
        """Describe channels (name, type, descriptive statistics).

        Parameters
        ----------
        data_frame : bool
            If True, return results in a pandas.DataFrame. If False, only print
            results. Columns 'ch', 'type', and 'unit' indicate channel index,
            channel type, and unit of the remaining five columns. These columns
            are 'min' (minimum), 'Q1' (first quartile or 25% percentile),
            'median', 'Q3' (third quartile or 75% percentile), and 'max'
            (maximum).

        Returns
        -------
        result : None | pandas.DataFrame
            If data_frame=False, returns None. If data_frame=True, returns
            results in a pandas.DataFrame (requires pandas).
        """
        from scipy.stats import scoreatpercentile as q
        nchan = self.info["nchan"]

        # describe each channel
        cols = defaultdict(list)
        cols["name"] = self.ch_names
        for i in range(nchan):
            ch = self.info["chs"][i]
            data = self[i][0]
            cols["type"].append(channel_type(self.info, i))
            cols["unit"].append(_unit2human[ch["unit"]])
            cols["min"].append(np.min(data))
            cols["Q1"].append(q(data, 25))
            cols["median"].append(np.median(data))
            cols["Q3"].append(q(data, 75))
            cols["max"].append(np.max(data))

        if data_frame:  # return data frame
            import pandas as pd
            df = pd.DataFrame(cols)
            df.index.name = "ch"
            return df

        # convert into commonly used units
        scalings = _handle_default("scalings")
        units = _handle_default("units")
        for i in range(nchan):
            unit = units.get(cols['type'][i])
            scaling = scalings.get(cols['type'][i], 1)
            if scaling != 1:
                cols['unit'][i] = unit
                for col in ["min", "Q1", "median", "Q3", "max"]:
                    cols[col][i] *= scaling

        lens = {"ch": max(2, len(str(nchan))),
                "name": max(4, max([len(n) for n in cols["name"]])),
                "type": max(4, max([len(t) for t in cols["type"]])),
                "unit": max(4, max([len(u) for u in cols["unit"]]))}

        # print description, start with header
        print(self)
        print(f"{'ch':>{lens['ch']}}  "
              f"{'name':<{lens['name']}}  "
              f"{'type':<{lens['type']}}  "
              f"{'unit':<{lens['unit']}}  "
              f"{'min':>9}  "
              f"{'Q1':>9}  "
              f"{'median':>9}  "
              f"{'Q3':>9}  "
              f"{'max':>9}")
        # print description for each channel
        for i in range(nchan):
            msg = (f"{i:>{lens['ch']}}  "
                   f"{cols['name'][i]:<{lens['name']}}  "
                   f"{cols['type'][i].upper():<{lens['type']}}  "
                   f"{cols['unit'][i]:<{lens['unit']}}  ")
            for col in ["min", "Q1", "median", "Q3"]:
                msg += f"{cols[col][i]:>9.2f}  "
            msg += f"{cols['max'][i]:>9.2f}"
            print(msg)


def _allocate_data(preload, shape, dtype):
    """Allocate data in memory or in memmap for preloading."""
    if preload in (None, True):  # None comes from _read_segment
        data = np.zeros(shape, dtype)
    else:
        _validate_type(preload, 'path-like', 'preload')
        data = np.memmap(str(preload), mode='w+', dtype=dtype, shape=shape)
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


def _convert_slice(sel):
    if len(sel) and (np.diff(sel) == 1).all():
        return slice(sel[0], sel[-1] + 1)
    else:
        return sel


def _get_ch_factors(inst, units, picks_idxs):
    """Get scaling factors for data, given units.

    Parameters
    ----------
    inst : instance of Raw | Epochs | Evoked
        The instance.
    %(units)s
    picks_idxs : ndarray
        The picks as provided through _picks_to_idx.

    Returns
    -------
    ch_factors : ndarray of floats, shape(len(picks),)
        The sacling factors for each channel, ordered according
        to picks.

    """
    _validate_type(units, types=(None, str, dict), item_name="units")
    ch_factors = np.ones(len(picks_idxs))
    si_units = _handle_default('si_units')
    ch_types = inst.get_channel_types(picks=picks_idxs)
    # Convert to dict if str units
    if isinstance(units, str):
        # Check that there is only one channel type
        unit_ch_type = list(set(ch_types) & set(si_units.keys()))
        if len(unit_ch_type) > 1:
            raise ValueError('"units" cannot be str if there is more than '
                             'one channel type with a unit '
                             f'{unit_ch_type}.')
        units = {unit_ch_type[0]: units}  # make the str argument a dict
    # Loop over the dict to get channel factors
    if isinstance(units, dict):
        for ch_type, ch_unit in units.items():
            # Get the scaling factors
            scaling = _get_scaling(ch_type, ch_unit)
            if scaling != 1:
                indices = [i_ch for i_ch, ch in enumerate(ch_types)
                           if ch == ch_type]
                ch_factors[indices] *= scaling

    return ch_factors


def _get_scaling(ch_type, target_unit):
    """Return the scaling factor based on the channel type and a target unit.

    Parameters
    ----------
    ch_type : str
        The channel type.
    target_unit : str
        The target unit for the provided channel type.

    Returns
    -------
    scaling : float
        The scaling factor to convert from the si_unit (used by default for MNE
        objects) to the target unit.
    """
    scaling = 1.
    si_units = _handle_default('si_units')
    si_units_splitted = {key: si_units[key].split('/') for key in si_units}
    prefixes = _handle_default('prefixes')
    prefix_list = list(prefixes.keys())

    # Check that the provided unit exists for the ch_type
    unit_list = target_unit.split('/')
    if ch_type not in si_units.keys():
        raise KeyError(
            f'{ch_type} is not a channel type that can be scaled '
            'from units.')
    si_unit_list = si_units_splitted[ch_type]
    if len(unit_list) != len(si_unit_list):
        raise ValueError(
            f'{target_unit} is not a valid unit for {ch_type}, use a '
            f'sub-multiple of {si_units[ch_type]} instead.')
    for i, unit in enumerate(unit_list):
        valid = [prefix + si_unit_list[i]
                 for prefix in prefix_list]
        if unit not in valid:
            raise ValueError(
                f'{target_unit} is not a valid unit for {ch_type}, use a '
                f'sub-multiple of {si_units[ch_type]} instead.')

    # Get the scaling factors
    for i, unit in enumerate(unit_list):
        has_square = False
        # XXX power normally not used as csd cannot get_data()
        if unit[-1] == '²':
            has_square = True
        if unit == 'm' or unit == 'm²':
            factor = 1.
        elif unit[0] in prefixes.keys():
            factor = prefixes[unit[0]]
        else:
            factor = 1.
        if factor != 1:
            if has_square:
                factor *= factor
            if i == 0:
                scaling = scaling * factor
            elif i == 1:
                scaling = scaling / factor
    return scaling


class _ReadSegmentFileProtector(object):
    """Ensure only _filenames, _raw_extras, and _read_segment_file are used."""

    def __init__(self, raw):
        self.__raw = raw
        assert hasattr(raw, '_projector')
        self._filenames = raw._filenames
        self._raw_extras = raw._raw_extras

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        return self.__raw.__class__._read_segment_file(
            self, data, idx, fi, start, stop, cals, mult)


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
        self._annotations = annotations.copy()


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

    # Expand `~` if present
    fname = _check_fname(fname=fname, overwrite=overwrite)

    base, ext = op.splitext(fname)
    if part_idx > 0:
        if split_naming == 'neuromag':
            # insert index in filename
            use_fname = '%s-%d%s' % (base, part_idx, ext)
        else:
            assert split_naming == 'bids'
            use_fname = _construct_bids_filename(base, ext, part_idx + 1)
            # check for file existence
            _check_fname(use_fname, overwrite)
    else:
        use_fname = fname
    # reserve our BIDS split fname in case we need to split
    if split_naming == 'bids' and part_idx == 0:
        # reserve our possible split name
        reserved_fname = _construct_bids_filename(base, ext, part_idx + 1)
        logger.info(
            f'Reserving possible split file {op.basename(reserved_fname)}')
        _check_fname(reserved_fname, overwrite)
        ctx = _ReservedFilename(reserved_fname)
    else:
        reserved_fname = use_fname
        ctx = nullcontext()
    logger.info('Writing %s' % use_fname)

    picks = _picks_to_idx(info, picks, 'all', ())
    fid, cals = _start_writing_raw(use_fname, info, picks, data_type,
                                   reset_range, raw.annotations)
    with ctx, fid:
        final_fname = _write_raw_fid(
            raw, info, picks, fid, cals, part_idx, start, stop,
            buffer_size, prev_fname, split_size, use_fname,
            projector, drop_small_buffer, fmt, fname, reserved_fname,
            data_type, reset_range, split_naming,
            overwrite=True  # we've started writing already above
        )
    if final_fname != use_fname:
        assert split_naming == 'bids'
        logger.info(f'Renaming BIDS split file {op.basename(final_fname)}')
        ctx.remove = False
        shutil.move(use_fname, final_fname)
    if part_idx == 0:
        logger.info('[done]')
    return final_fname, part_idx


class _ReservedFilename:

    def __init__(self, fname):
        self.fname = fname
        assert op.isdir(op.dirname(fname)), fname
        with open(fname, 'w'):
            pass
        self.remove = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.remove:
            os.remove(self.fname)


def _write_raw_fid(raw, info, picks, fid, cals, part_idx, start, stop,
                   buffer_size, prev_fname, split_size, use_fname,
                   projector, drop_small_buffer, fmt, fname, reserved_fname,
                   data_type, reset_range, split_naming, overwrite):
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
    final_fname = use_fname
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
        logger.debug(f'Writing FIF {first:6d} ... {last:6d} ...')
        _write_raw_buffer(fid, data, cals, fmt)

        pos = fid.tell()
        this_buff_size_bytes = pos - pos_prev
        overage = pos - split_size + _NEXT_FILE_BUFFER
        if overage > 0:
            # This should occur on the first buffer write of the file, so
            # we should mention the space required for the meas info
            raise ValueError(
                'buffer size (%s) is too large for the given split size (%s) '
                'by %s bytes after writing info (%s) and leaving enough space '
                'for end tags (%s): decrease "buffer_size_sec" or increase '
                '"split_size".' % (this_buff_size_bytes, split_size, overage,
                                   pos_prev, _NEXT_FILE_BUFFER))

        # Split files if necessary, leave some space for next file info
        # make sure we check to make sure we actually *need* another buffer
        # with the "and" check
        if pos >= split_size - this_buff_size_bytes - _NEXT_FILE_BUFFER and \
                first + buffer_size < stop:
            final_fname = reserved_fname
            next_fname, next_idx = _write_raw(
                fname, raw, info, picks, fmt,
                data_type, reset_range, first + buffer_size, stop, buffer_size,
                projector, drop_small_buffer, split_size, split_naming,
                part_idx + 1, final_fname, overwrite)

            start_block(fid, FIFF.FIFFB_REF)
            write_int(fid, FIFF.FIFF_REF_ROLE, FIFF.FIFFV_ROLE_NEXT_FILE)
            write_string(fid, FIFF.FIFF_REF_FILE_NAME, op.basename(next_fname))
            if info['meas_id'] is not None:
                write_id(fid, FIFF.FIFF_REF_FILE_ID, info['meas_id'])
            write_int(fid, FIFF.FIFF_REF_FILE_NUM, next_idx)
            end_block(fid, FIFF.FIFFB_REF)
            break
        pos_prev = pos

    logger.info('Closing %s' % use_fname)
    if info.get('maxshield', False):
        end_block(fid, FIFF.FIFFB_IAS_RAW_DATA)
    else:
        end_block(fid, FIFF.FIFFB_RAW_DATA)
    end_block(fid, FIFF.FIFFB_MEAS)
    end_file(fid)
    return final_fname


@fill_doc
def _start_writing_raw(name, info, sel, data_type,
                       reset_range, annotations):
    """Start write raw data in file.

    Parameters
    ----------
    name : string
        Name of the file to create.
    %(info_not_none)s
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
        start_block(fid, FIFF.FIFFB_IAS_RAW_DATA)
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

    cast_int = False  # allow unsafe cast
    if np.isrealobj(buf):
        if fmt == 'short':
            write_function = write_dau_pack16
            cast_int = True
        elif fmt == 'int':
            write_function = write_int
            cast_int = True
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
    if cast_int:
        buf = buf.astype(np.int32)
    write_function(fid, FIFF.FIFF_DATA_BUFFER, buf)


def _check_raw_compatibility(raw):
    """Ensure all instances of Raw have compatible parameters."""
    for ri in range(1, len(raw)):
        if not isinstance(raw[ri], type(raw[0])):
            raise ValueError(f'raw[{ri}] type must match')
        for key in ('nchan', 'bads', 'sfreq'):
            a, b = raw[ri].info[key], raw[0].info[key]
            if a != b:
                raise ValueError(
                    f'raw[{ri}].info[{key}] must match:\n'
                    f'{repr(a)} != {repr(b)}')
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
def concatenate_raws(raws, preload=None, events_list=None, *,
                     on_mismatch='raise', verbose=None):
    """Concatenate `~mne.io.Raw` instances as if they were continuous.

    .. note:: ``raws[0]`` is modified in-place to achieve the concatenation.
              Boundaries of the raw files are annotated bad. If you wish to use
              the data as continuous recording, you can remove the boundary
              annotations after concatenation (see
              :meth:`mne.Annotations.delete`).

    Parameters
    ----------
    raws : list
        List of `~mne.io.Raw` instances to concatenate (in order).
    %(preload_concatenate)s
    events_list : None | list
        The events to concatenate. Defaults to ``None``.
    %(on_info_mismatch)s
    %(verbose)s

    Returns
    -------
    raw : instance of Raw
        The result of the concatenation (first Raw instance passed in).
    events : ndarray of int, shape (n_events, 3)
        The events. Only returned if ``event_list`` is not None.
    """
    for idx, raw in enumerate(raws[1:], start=1):
        _ensure_infos_match(info1=raws[0].info, info2=raw.info,
                            name=f'raws[{idx}]', on_mismatch=on_mismatch)

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
