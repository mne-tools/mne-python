# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

from math import floor, ceil
import copy
from copy import deepcopy
import warnings
import os
import os.path as op

import numpy as np
from scipy.signal import hilbert
from scipy import linalg

from .constants import FIFF
from .pick import pick_types, channel_type, pick_channels
from .meas_info import write_meas_info
from .proj import (setup_proj, activate_proj, proj_equal, ProjMixin,
                   _has_eeg_average_ref_proj, make_eeg_average_ref_proj)
from ..channels import ContainsMixin, PickDropChannelsMixin
from .compensator import set_current_comp
from .write import (start_file, end_file, start_block, end_block,
                    write_dau_pack16, write_float, write_double,
                    write_complex64, write_complex128, write_int,
                    write_id, write_string)

from ..filter import (low_pass_filter, high_pass_filter, band_pass_filter,
                      notch_filter, band_stop_filter, resample)
from ..parallel import parallel_func
from ..utils import (_check_fname, estimate_rank, _check_pandas_installed,
                     check_fname, _get_stim_channel, object_hash,
                     logger, verbose)
from ..viz import plot_raw, plot_raw_psds, _mutable_defaults
from ..externals.six import string_types
from ..event import concatenate_events


class _BaseRaw(ProjMixin, ContainsMixin, PickDropChannelsMixin):
    """Base class for Raw data"""
    @verbose
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def _read_segment(start, stop, sel, projector, verbose):
        raise NotImplementedError

    def __del__(self):
        # remove file for memmap
        if hasattr(self, '_data') and hasattr(self._data, 'filename'):
            # First, close the file out; happens automatically on del
            filename = self._data.filename
            del self._data
            # Now file can be removed
            os.remove(filename)

    def __enter__(self):
        """ Entering with block """
        return self

    def __exit__(self, exception_type, exception_val, trace):
        """ Exiting with block """
        try:
            self.close()
        except:
            return exception_type, exception_val, trace

    def __hash__(self):
        if not self.preload:
            raise RuntimeError('Cannot hash raw unless preloaded')
        return object_hash(dict(info=self.info, data=self._data))

    def _add_eeg_ref(self, add_eeg_ref):
        """Helper to add an average EEG reference"""
        if add_eeg_ref:
            eegs = pick_types(self.info, meg=False, eeg=True, ref_meg=False)
            projs = self.info['projs']
            if len(eegs) > 0 and not _has_eeg_average_ref_proj(projs):
                eeg_ref = make_eeg_average_ref_proj(self.info, activate=False)
                projs.append(eeg_ref)

    def _parse_get_set_params(self, item):
        # make sure item is a tuple
        if not isinstance(item, tuple):  # only channel selection passed
            item = (item, slice(None, None, None))

        if len(item) != 2:  # should be channels and time instants
            raise RuntimeError("Unable to access raw data (need both channels "
                               "and time)")

        if isinstance(item[0], slice):
            start = item[0].start if item[0].start is not None else 0
            nchan = self.info['nchan']
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
            if isinstance(item1, int):
                start, stop, step = item1, item1 + 1, 1
            else:
                raise ValueError('Must pass int or slice to __getitem__')

        if start is None:
            start = 0
        if (step is not None) and (step is not 1):
            raise ValueError('step needs to be 1 : %d given' % step)

        if isinstance(sel, int):
            sel = np.array([sel])

        if sel is not None and len(sel) == 0:
            raise ValueError("Empty channel list")

        return sel, start, stop

    def __getitem__(self, item):
        """getting raw data content with python slicing"""
        sel, start, stop = self._parse_get_set_params(item)
        if self.preload:
            data, times = self._data[sel, start:stop], self._times[start:stop]
        else:
            data, times = self._read_segment(start=start, stop=stop, sel=sel,
                                             projector=self._projector,
                                             verbose=self.verbose)
        return data, times

    def __setitem__(self, item, value):
        """setting raw data content with python slicing"""
        if not self.preload:
            raise RuntimeError('Modifying data of Raw is only supported '
                               'when preloading is used. Use preload=True '
                               '(or string) in the constructor.')
        sel, start, stop = self._parse_get_set_params(item)
        # set the data
        self._data[sel, start:stop] = value

    def anonymize(self):
        """Anonymize data

        This function will remove info['subject_info'] if it exists."""
        self.info._anonymize()

    @verbose
    def apply_function(self, fun, picks, dtype, n_jobs, verbose=None, *args,
                       **kwargs):
        """ Apply a function to a subset of channels.

        The function "fun" is applied to the channels defined in "picks". The
        data of the Raw object is modified inplace. If the function returns
        a different data type (e.g. numpy.complex) it must be specified using
        the dtype parameter, which causes the data type used for representing
        the raw data to change.

        The Raw object has to be constructed using preload=True (or string).

        Note: If n_jobs > 1, more memory is required as "len(picks) * n_times"
              additional time points need to be temporaily stored in memory.

        Note: If the data type changes (dtype != None), more memory is required
              since the original and the converted data needs to be stored in
              memory.

        Parameters
        ----------
        fun : function
            A function to be applied to the channels. The first argument of
            fun has to be a timeseries (numpy.ndarray). The function must
            return an numpy.ndarray with the same size as the input.
        picks : array-like of int
            Indices of channels to apply the function to.
        dtype : numpy.dtype
            Data type to use for raw data after applying the function. If None
            the data type is not modified.
        n_jobs: int
            Number of jobs to run in parallel.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
            Defaults to self.verbose.
        *args :
            Additional positional arguments to pass to fun (first pos. argument
            of fun is the timeseries of a channel).
        **kwargs :
            Keyword arguments to pass to fun.
        """
        if not self.preload:
            raise RuntimeError('Raw data needs to be preloaded. Use '
                               'preload=True (or string) in the constructor.')

        if not callable(fun):
            raise ValueError('fun needs to be a function')

        data_in = self._data
        if dtype is not None and dtype != self._data.dtype:
            self._data = self._data.astype(dtype)

        if n_jobs == 1:
            # modify data inplace to save memory
            for idx in picks:
                self._data[idx, :] = fun(data_in[idx, :], *args, **kwargs)
        else:
            # use parallel function
            parallel, p_fun, _ = parallel_func(fun, n_jobs)
            data_picks_new = parallel(p_fun(data_in[p], *args, **kwargs)
                                      for p in picks)
            for pp, p in enumerate(picks):
                self._data[p, :] = data_picks_new[pp]

    @verbose
    def apply_hilbert(self, picks, envelope=False, n_jobs=1, verbose=None):
        """ Compute analytic signal or envelope for a subset of channels.

        If envelope=False, the analytic signal for the channels defined in
        "picks" is computed and the data of the Raw object is converted to
        a complex representation (the analytic signal is complex valued).

        If envelope=True, the absolute value of the analytic signal for the
        channels defined in "picks" is computed, resulting in the envelope
        signal.

        Note: DO NOT use envelope=True if you intend to compute an inverse
              solution from the raw data. If you want to compute the
              envelope in source space, use envelope=False and compute the
              envelope after the inverse solution has been obtained.

        Note: If envelope=False, more memory is required since the original
              raw data as well as the analytic signal have temporarily to
              be stored in memory.

        Note: If n_jobs > 1 and envelope=True, more memory is required as
              "len(picks) * n_times" additional time points need to be
              temporaily stored in memory.

        Parameters
        ----------
        picks : array-like of int
            Indices of channels to apply the function to.
        envelope : bool (default: False)
            Compute the envelope signal of each channel.
        n_jobs: int
            Number of jobs to run in parallel.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
            Defaults to self.verbose.

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
        """
        if envelope:
            self.apply_function(_envelope, picks, None, n_jobs)
        else:
            self.apply_function(hilbert, picks, np.complex64, n_jobs)

    @verbose
    def filter(self, l_freq, h_freq, picks=None, filter_length='10s',
               l_trans_bandwidth=0.5, h_trans_bandwidth=0.5, n_jobs=1,
               method='fft', iir_params=None, verbose=None):
        """Filter a subset of channels.

        Applies a zero-phase low-pass, high-pass, band-pass, or band-stop
        filter to the channels selected by "picks". The data of the Raw
        object is modified inplace.

        The Raw object has to be constructed using preload=True (or string).

        l_freq and h_freq are the frequencies below which and above which,
        respectively, to filter out of the data. Thus the uses are:
            l_freq < h_freq: band-pass filter
            l_freq > h_freq: band-stop filter
            l_freq is not None, h_freq is None: low-pass filter
            l_freq is None, h_freq is not None: high-pass filter

        Note: If n_jobs > 1, more memory is required as "len(picks) * n_times"
              additional time points need to be temporarily stored in memory.

        Note: self.info['lowpass'] and self.info['highpass'] are only updated
              with picks=None.

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
        filter_length : str (Default: '10s') | int | None
            Length of the filter to use. If None or "len(x) < filter_length",
            the filter length used is len(x). Otherwise, if int, overlap-add
            filtering with a filter of the specified length in samples) is
            used (faster for long signals). If str, a human-readable time in
            units of "s" or "ms" (e.g., "10s" or "5500ms") will be converted
            to the shortest power-of-two length at least that duration.
        l_trans_bandwidth : float
            Width of the transition band at the low cut-off frequency in Hz.
        h_trans_bandwidth : float
            Width of the transition band at the high cut-off frequency in Hz.
        n_jobs : int | str
            Number of jobs to run in parallel. Can be 'cuda' if scikits.cuda
            is installed properly, CUDA is initialized, and method='fft'.
        method : str
            'fft' will use overlap-add FIR filtering, 'iir' will use IIR
            forward-backward filtering (via filtfilt).
        iir_params : dict | None
            Dictionary of parameters to use for IIR filtering.
            See mne.filter.construct_iir_filter for details. If iir_params
            is None and method="iir", 4th order Butterworth will be used.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
            Defaults to self.verbose.
        """
        if verbose is None:
            verbose = self.verbose
        fs = float(self.info['sfreq'])
        if l_freq == 0:
            l_freq = None
        if h_freq is not None and h_freq > (fs / 2.):
            h_freq = None
        if l_freq is not None and not isinstance(l_freq, float):
            l_freq = float(l_freq)
        if h_freq is not None and not isinstance(h_freq, float):
            h_freq = float(h_freq)

        if not self.preload:
            raise RuntimeError('Raw data needs to be preloaded to filter. Use '
                               'preload=True (or string) in the constructor.')
        if picks is None:
            if 'ICA ' in ','.join(self.ch_names):
                pick_parameters = dict(misc=True, ref_meg=False)
            else:
                pick_parameters = dict(meg=True, eeg=True, ref_meg=False)
            picks = pick_types(self.info, exclude=[], **pick_parameters)
            # let's be safe.
            if len(picks) < 1:
                raise RuntimeError('Could not find any valid channels for '
                                   'your Raw object. Please contact the '
                                   'MNE-Python developers.')

            # update info if filter is applied to all data channels,
            # and it's not a band-stop filter
            if h_freq is not None and (l_freq is None or l_freq < h_freq) and \
                    h_freq < self.info['lowpass']:
                self.info['lowpass'] = h_freq
            if l_freq is not None and (h_freq is None or l_freq < h_freq) and \
                    l_freq > self.info['highpass']:
                self.info['highpass'] = l_freq
        if l_freq is None and h_freq is not None:
            logger.info('Low-pass filtering at %0.2g Hz' % h_freq)
            low_pass_filter(self._data, fs, h_freq,
                            filter_length=filter_length,
                            trans_bandwidth=l_trans_bandwidth, method=method,
                            iir_params=iir_params, picks=picks, n_jobs=n_jobs,
                            copy=False)
        if l_freq is not None and h_freq is None:
            logger.info('High-pass filtering at %0.2g Hz' % l_freq)
            high_pass_filter(self._data, fs, l_freq,
                             filter_length=filter_length,
                             trans_bandwidth=h_trans_bandwidth, method=method,
                             iir_params=iir_params, picks=picks, n_jobs=n_jobs,
                             copy=False)
        if l_freq is not None and h_freq is not None:
            if l_freq < h_freq:
                logger.info('Band-pass filtering from %0.2g - %0.2g Hz'
                            % (l_freq, h_freq))
                self._data = band_pass_filter(self._data, fs, l_freq, h_freq,
                    filter_length=filter_length,
                    l_trans_bandwidth=l_trans_bandwidth,
                    h_trans_bandwidth=h_trans_bandwidth,
                    method=method, iir_params=iir_params, picks=picks,
                    n_jobs=n_jobs, copy=False)
            else:
                logger.info('Band-stop filtering from %0.2g - %0.2g Hz'
                            % (h_freq, l_freq))
                self._data = band_stop_filter(self._data, fs, h_freq, l_freq,
                    filter_length=filter_length,
                    l_trans_bandwidth=h_trans_bandwidth,
                    h_trans_bandwidth=l_trans_bandwidth, method=method,
                    iir_params=iir_params, picks=picks, n_jobs=n_jobs,
                    copy=False)

    @verbose
    def notch_filter(self, freqs, picks=None, filter_length='10s',
                     notch_widths=None, trans_bandwidth=1.0, n_jobs=1,
                     method='fft', iir_params=None,
                     mt_bandwidth=None, p_value=0.05, verbose=None):
        """Notch filter a subset of channels.

        Applies a zero-phase notch filter to the channels selected by
        "picks". The data of the Raw object is modified inplace.

        The Raw object has to be constructed using preload=True (or string).

        Note: If n_jobs > 1, more memory is required as "len(picks) * n_times"
              additional time points need to be temporaily stored in memory.

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
        filter_length : str (Default: '10s') | int | None
            Length of the filter to use. If None or "len(x) < filter_length",
            the filter length used is len(x). Otherwise, if int, overlap-add
            filtering with a filter of the specified length in samples) is
            used (faster for long signals). If str, a human-readable time in
            units of "s" or "ms" (e.g., "10s" or "5500ms") will be converted
            to the shortest power-of-two length at least that duration.
        notch_widths : float | array of float | None
            Width of each stop band (centred at each freq in freqs) in Hz.
            If None, freqs / 200 is used.
        trans_bandwidth : float
            Width of the transition band in Hz.
        n_jobs : int | str
            Number of jobs to run in parallel. Can be 'cuda' if scikits.cuda
            is installed properly, CUDA is initialized, and method='fft'.
        method : str
            'fft' will use overlap-add FIR filtering, 'iir' will use IIR
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
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
            Defaults to self.verbose.

        Notes
        -----
        For details, see mne.filter.notch_filter.
        """
        if verbose is None:
            verbose = self.verbose
        fs = float(self.info['sfreq'])
        if picks is None:
            if 'ICA ' in ','.join(self.ch_names):
                pick_parameters = dict(misc=True)
            else:
                pick_parameters = dict(meg=True, eeg=True)
            picks = pick_types(self.info, exclude=[], **pick_parameters)
            # let's be safe.
            if len(picks) < 1:
                raise RuntimeError('Could not find any valid channels for '
                                   'your Raw object. Please contact the '
                                   'MNE-Python developers.')
        if not self.preload:
            raise RuntimeError('Raw data needs to be preloaded to filter. Use '
                               'preload=True (or string) in the constructor.')

        self._data = notch_filter(self._data, fs, freqs,
                                  filter_length=filter_length,
                                  notch_widths=notch_widths,
                                  trans_bandwidth=trans_bandwidth,
                                  method=method, iir_params=iir_params,
                                  mt_bandwidth=mt_bandwidth, p_value=p_value,
                                  picks=picks, n_jobs=n_jobs, copy=False)

    @verbose
    def resample(self, sfreq, npad=100, window='boxcar',
                 stim_picks=None, n_jobs=1, verbose=None):
        """Resample data channels.

        Resamples all channels. The data of the Raw object is modified inplace.

        The Raw object has to be constructed using preload=True (or string).

        WARNING: The intended purpose of this function is primarily to speed
        up computations (e.g., projection calculation) when precise timing
        of events is not required, as downsampling raw data effectively
        jitters trigger timings. It is generally recommended not to epoch
        downsampled data, but instead epoch and then downsample, as epoching
        downsampled data jitters triggers.

        Parameters
        ----------
        sfreq : float
            New sample rate to use.
        npad : int
            Amount to pad the start and end of the data.
        window : string or tuple
            Window to use in resampling. See scipy.signal.resample.
        stim_picks : array of int | None
            Stim channels. These channels are simply subsampled or
            supersampled (without applying any filtering). This reduces
            resampling artifacts in stim channels, but may lead to missing
            triggers. If None, stim channels are automatically chosen using
            mne.pick_types(raw.info, meg=False, stim=True, exclude=[]).
        n_jobs : int | str
            Number of jobs to run in parallel. Can be 'cuda' if scikits.cuda
            is installed properly and CUDA is initialized.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
            Defaults to self.verbose.

        Notes
        -----
        For some data, it may be more accurate to use npad=0 to reduce
        artifacts. This is dataset dependent -- check your data!
        """
        if not self.preload:
            raise RuntimeError('Can only resample preloaded data')
        sfreq = float(sfreq)
        o_sfreq = float(self.info['sfreq'])

        offsets = np.concatenate(([0], np.cumsum(self._raw_lengths)))
        new_data = list()
        # set up stim channel processing
        if stim_picks is None:
            stim_picks = pick_types(self.info, meg=False, ref_meg=False,
                                    stim=True, exclude=[])
        stim_picks = np.asanyarray(stim_picks)
        ratio = sfreq / o_sfreq
        for ri in range(len(self._raw_lengths)):
            data_chunk = self._data[:, offsets[ri]:offsets[ri + 1]]
            new_data.append(resample(data_chunk, sfreq, o_sfreq, npad,
                                     n_jobs=n_jobs))
            new_ntimes = new_data[ri].shape[1]

            # Now deal with the stim channels. In empirical testing, it was
            # faster to resample all channels (above) and then replace the
            # stim channels than it was to only resample the proper subset
            # of channels and then use np.insert() to restore the stims

            # figure out which points in old data to subsample
            # protect against out-of-bounds, which can happen (having
            # one sample more than expected) due to padding
            stim_inds = np.minimum(np.floor(np.arange(new_ntimes)
                                            / ratio).astype(int),
                                   data_chunk.shape[1] - 1)
            for sp in stim_picks:
                new_data[ri][sp] = data_chunk[[sp]][:, stim_inds]

            self._first_samps[ri] = int(self._first_samps[ri] * ratio)
            self._last_samps[ri] = self._first_samps[ri] + new_ntimes - 1
            self._raw_lengths[ri] = new_ntimes

        # adjust affected variables
        self._data = np.concatenate(new_data, axis=1)
        self.first_samp = self._first_samps[0]
        self.last_samp = self.first_samp + self._data.shape[1] - 1
        self.info['sfreq'] = sfreq
        self._times = (np.arange(self.n_times, dtype=np.float64)
                       / self.info['sfreq'])

    def crop(self, tmin=0.0, tmax=None, copy=True):
        """Crop raw data file.

        Limit the data from the raw file to go between specific times. Note
        that the new tmin is assumed to be t=0 for all subsequently called
        functions (e.g., time_as_index, or Epochs). New first_samp and
        last_samp are set accordingly. And data are modified in-place when
        called with copy=False.

        Parameters
        ----------
        tmin : float
            New start time (must be >= 0).
        tmax : float | None
            New end time of the data (cannot exceed data duration).
        copy : bool
            If False Raw is cropped in place.

        Returns
        -------
        raw : instance of Raw
            The cropped raw object.
        """
        raw = self.copy() if copy is True else self
        max_time = (raw.n_times - 1) / raw.info['sfreq']
        if tmax is None:
            tmax = max_time

        if tmin > tmax:
            raise ValueError('tmin must be less than tmax')
        if tmin < 0.0:
            raise ValueError('tmin must be >= 0')
        elif tmax > max_time:
            raise ValueError('tmax must be less than or equal to the max raw '
                             'time (%0.4f sec)' % max_time)

        smin = raw.time_as_index(tmin)[0]
        smax = raw.time_as_index(tmax)[0]
        cumul_lens = np.concatenate(([0], np.array(raw._raw_lengths,
                                                   dtype='int')))
        cumul_lens = np.cumsum(cumul_lens)
        keepers = np.logical_and(np.less(smin, cumul_lens[1:]),
                                 np.greater_equal(smax, cumul_lens[:-1]))
        keepers = np.where(keepers)[0]
        raw._first_samps = np.atleast_1d(raw._first_samps[keepers])
        # Adjust first_samp of first used file!
        raw._first_samps[0] += smin - cumul_lens[keepers[0]]
        raw._last_samps = np.atleast_1d(raw._last_samps[keepers])
        raw._last_samps[-1] -= cumul_lens[keepers[-1] + 1] - 1 - smax
        raw._raw_lengths = raw._last_samps - raw._first_samps + 1
        raw.rawdirs = [r for ri, r in enumerate(raw.rawdirs)
                       if ri in keepers]
        raw.first_samp = raw._first_samps[0]
        raw.last_samp = raw.first_samp + (smax - smin)
        if raw.preload:
            raw._data = raw._data[:, smin:smax + 1]
            raw._times = np.arange(raw.n_times) / raw.info['sfreq']
        return raw

    @verbose
    def save(self, fname, picks=None, tmin=0, tmax=None, buffer_size_sec=10,
             drop_small_buffer=False, proj=False, format='single',
             overwrite=False, split_size='2GB', verbose=None):
        """Save raw data to file

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
            Size of data chunks in seconds. If None, the buffer size of
            the original file is used.
        drop_small_buffer : bool
            Drop or not the last buffer. It is required by maxfilter (SSS)
            that only accepts raw files with buffers of the same size.
        proj : bool
            If True the data is saved with the projections applied (active).
            Note: If apply_proj() was used to apply the projections,
            the projectons will be active even if proj is False.
        format : str
            Format to use to save raw data. Valid options are 'double',
            'single', 'int', and 'short' for 64- or 32-bit float, or 32- or
            16-bit integers, respectively. It is STRONGLY recommended to use
            'single', as this is backward-compatible, and is standard for
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
            Note: Due to FIFF file limitations, the maximum split size is 2GB.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
            Defaults to self.verbose.

        Notes
        -----
        If Raw is a concatenation of several raw files, *be warned* that only
        the measurement information from the first raw file is stored. This
        likely means that certain operations with external tools may not
        work properly on a saved concatenated file (e.g., probably some
        or all forms of SSS). It is recommended not to concatenate and
        then save raw files for this reason.
        """
        check_fname(fname, 'raw', ('raw.fif', 'raw_sss.fif', 'raw_tsss.fif',
                                   'raw.fif.gz', 'raw_sss.fif.gz',
                                   'raw_tsss.fif.gz'))

        if isinstance(split_size, string_types):
            exp = dict(MB=20, GB=30).get(split_size[-2:], None)
            if exp is None:
                raise ValueError('split_size has to end with either'
                                 '"MB" or "GB"')
            split_size = int(float(split_size[:-2]) * 2 ** exp)

        if split_size > 2147483648:
            raise ValueError('split_size cannot be larger than 2GB')

        fname = op.realpath(fname)
        if not self.preload and fname in self._filenames:
            raise ValueError('You cannot save data to the same file.'
                             ' Please use a different filename.')

        if self.preload:
            if np.iscomplexobj(self._data):
                warnings.warn('Saving raw file with complex data. Loading '
                              'with command-line MNE tools will not work.')

        type_dict = dict(short=FIFF.FIFFT_DAU_PACK16,
                         int=FIFF.FIFFT_INT,
                         single=FIFF.FIFFT_FLOAT,
                         double=FIFF.FIFFT_DOUBLE)
        if not format in type_dict.keys():
            raise ValueError('format must be "short", "int", "single", '
                             'or "double"')
        reset_dict = dict(short=False, int=False, single=True, double=True)
        reset_range = reset_dict[format]
        data_type = type_dict[format]

        data_test = self[0, 0][0]
        if format == 'short' and np.iscomplexobj(data_test):
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

        # set the correct compensation grade and make inverse compensator
        inv_comp = None
        if self.comp is not None:
            inv_comp = linalg.inv(self.comp)
            set_current_comp(info, self._orig_comp_grade)

        #
        #   Set up the reading parameters
        #

        #   Convert to samples
        start = int(floor(tmin * self.info['sfreq']))

        if tmax is None:
            stop = self.last_samp + 1 - self.first_samp
        else:
            stop = int(floor(tmax * self.info['sfreq']))

        if buffer_size_sec is None:
            if 'buffer_size_sec' in self.info:
                buffer_size_sec = self.info['buffer_size_sec']
            else:
                buffer_size_sec = 10.0
        buffer_size = int(ceil(buffer_size_sec * self.info['sfreq']))

        # write the raw file
        _write_raw(fname, self, info, picks, format, data_type, reset_range,
                   start, stop, buffer_size, projector, inv_comp,
                   drop_small_buffer, split_size, 0, None)

    def plot(raw, events=None, duration=10.0, start=0.0, n_channels=20,
             bgcolor='w', color=None, bad_color=(0.8, 0.8, 0.8),
             event_color='cyan', scalings=None, remove_dc=True, order='type',
             show_options=False, title=None, show=True, block=False):
        """Plot raw data

        Parameters
        ----------
        raw : instance of Raw
            The raw data to plot.
        events : array | None
            Events to show with vertical bars.
        duration : float
            Time window (sec) to plot in a given time.
        start : float
            Initial time to show (can be changed dynamically once plotted).
        n_channels : int
            Number of channels to plot at once.
        bgcolor : color object
            Color of the background.
        color : dict | color object | None
            Color for the data traces. If None, defaults to:
            `dict(mag='darkblue', grad='b', eeg='k', eog='k', ecg='r', emg='k',
                 ref_meg='steelblue', misc='k', stim='k', resp='k', chpi='k')`
        bad_color : color object
            Color to make bad channels.
        event_color : color object
            Color to use for events.
        scalings : dict | None
            Scale factors for the traces. If None, defaults to:
            `dict(mag=1e-12, grad=4e-11, eeg=20e-6,
                  eog=150e-6, ecg=5e-4, emg=1e-3,
                  ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4)`
        remove_dc : bool
            If True remove DC component when plotting data.
        order : 'type' | 'original' | array
            Order in which to plot data. 'type' groups by channel type,
            'original' plots in the order of ch_names, array gives the
            indices to use in plotting.
        show_options : bool
            If True, a dialog for options related to projection is shown.
        title : str | None
            The title of the window. If None, and either the filename of the
            raw object or '<unknown>' will be displayed as title.
        show : bool
            Show figures if True
        block : bool
            Whether to halt program execution until the figure is closed.
            Useful for setting bad channels on the fly (click on line).

        Returns
        -------
        fig : Instance of matplotlib.figure.Figure
            Raw traces.

        Notes
        -----
        The arrow keys (up/down/left/right) can typically be used to navigate
        between channels and time ranges, but this depends on the backend
        matplotlib is configured to use (e.g., mpl.use('TkAgg') should work).
        To mark or un-mark a channel as bad, click on the rather flat segments
        of a channel's time series. The changes will be reflected immediately
        in the raw object's ``raw.info['bads']`` entry.
        """
        return plot_raw(raw, events, duration, start, n_channels, bgcolor,
                        color, bad_color, event_color, scalings, remove_dc,
                        order, show_options, title, show, block)

    @verbose
    def plot_psds(self, tmin=0.0, tmax=60.0, fmin=0, fmax=np.inf,
                  proj=False, n_fft=2048, picks=None, ax=None, color='black',
                  area_mode='std', area_alpha=0.33, n_jobs=1, verbose=None):
        """Plot the power spectral density across channels

        Parameters
        ----------
        tmin : float
            Start time for calculations.
        tmax : float
            End time for calculations.
        fmin : float
            Start frequency to consider.
        fmax : float
            End frequency to consider.
        proj : bool
            Apply projection.
        n_fft : int
            Number of points to use in Welch FFT calculations.
        picks : array-like of int | None
            List of channels to use. Cannot be None if `ax` is supplied. If
            both `picks` and `ax` are None, separate subplots will be created
            for each standard channel type (`mag`, `grad`, and `eeg`).
        ax : instance of matplotlib Axes | None
            Axes to plot into. If None, axes will be created.
        color : str | tuple
            A matplotlib-compatible color to use.
        area_mode : str | None
            How to plot area. If 'std', the mean +/- 1 STD (across channels)
            will be plotted. If 'range', the min and max (across channels)
            will be plotted. Bad channels will be excluded from these
            calculations. If None, no area will be plotted.
        area_alpha : float
            Alpha for the area.
        n_jobs : int
            Number of jobs to run in parallel.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
        """
        return plot_raw_psds(self, tmin, tmax, fmin, fmax, proj, n_fft, picks,
                             ax, color, area_mode, area_alpha, n_jobs)

    def time_as_index(self, times, use_first_samp=False):
        """Convert time to indices

        Parameters
        ----------
        times : list-like | float | int
            List of numbers or a number representing points in time.
        use_first_samp : boolean
            If True, time is treated as relative to the session onset, else
            as relative to the recording onset.

        Returns
        -------
        index : ndarray
            Indices corresponding to the times supplied.
        """
        return _time_as_index(times, self.info['sfreq'], self.first_samp,
                              use_first_samp)

    def index_as_time(self, index, use_first_samp=False):
        """Convert indices to time

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
        return _index_as_time(index, self.info['sfreq'], self.first_samp,
                              use_first_samp)

    def estimate_rank(self, tstart=0.0, tstop=30.0, tol=1e-4,
                      return_singular=False, picks=None):
        """Estimate rank of the raw data

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
        start = max(0, self.time_as_index(tstart)[0])
        if tstop is None:
            stop = self.n_times - 1
        else:
            stop = min(self.n_times - 1, self.time_as_index(tstop)[0])
        tslice = slice(start, stop + 1)
        if picks is None:
            picks = pick_types(self.info, meg=True, eeg=True, ref_meg=False,
                               exclude='bads')

        # ensure we don't get a view of data
        if len(picks) == 1:
            return 1.0, 1.0
        # this should already be a copy, so we can overwrite it
        data = self[picks, tslice][0]
        return estimate_rank(data, tol, return_singular, copy=False)

    @property
    def ch_names(self):
        """Channel names"""
        return self.info['ch_names']

    @property
    def n_times(self):
        """Number of time points"""
        return self.last_samp - self.first_samp + 1

    def __len__(self):
        return self.n_times

    def load_bad_channels(self, bad_file=None, force=False):
        """
        Mark channels as bad from a text file, in the style
        (mostly) of the C function mne_mark_bad_channels

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
                                                  self._filenames[0]))
                else:
                    warnings.warn('%d bad channels from:\n%s\nnot found '
                                  'in:\n%s' % (count_diff, bad_file,
                                               self._filenames[0]))
            self.info['bads'] = names_there
        else:
            self.info['bads'] = []

    def append(self, raws, preload=None):
        """Concatenate raw instances as if they were continuous

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
                self._times = None
            self.preload = False
        else:
            # do the concatenation ourselves since preload might be a string
            nchan = self.info['nchan']
            c_ns = np.cumsum([rr.n_times for rr in ([self] + raws)])
            nsamp = c_ns[-1]

            if not self.preload:
                this_data = self._read_segment()[0]
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
        for r in raws:
            self._first_samps = np.r_[self._first_samps, r._first_samps]
            self._last_samps = np.r_[self._last_samps, r._last_samps]
            self._raw_lengths = np.r_[self._raw_lengths, r._raw_lengths]
            self.rawdirs += r.rawdirs
            self._filenames += r._filenames
        self.last_samp = self.first_samp + sum(self._raw_lengths) - 1

        # this has to be done after first and last sample are set appropriately
        if self.preload:
            self._times = np.arange(self.n_times) / self.info['sfreq']

    def close(self):
        """Clean up the object.

        Does nothing for now.
        """
        pass

    def copy(self):
        """ Return copy of Raw instance
        """
        return deepcopy(self)

    def as_data_frame(self, picks=None, start=None, stop=None, scale_time=1e3,
                      scalings=None, use_time_index=True, copy=True):
        """Get the epochs as Pandas DataFrame

        Export raw data in tabular structure with MEG channels.

        Caveat! To save memory, depending on selected data size consider
        setting copy to False.

        Parameters
        ----------
        picks : array-like of int | None
            If None only MEG and EEG channels are kept
            otherwise the channels indices in picks are kept.
        start : int | None
            Data-extraction start index. If None, data will be exported from
            the first sample.
        stop : int | None
            Data-extraction stop index. If None, data will be exported to the
            last index.
        scale_time : float
            Scaling to be applied to time units.
        scalings : dict | None
            Scaling to be applied to the channels picked. If None, defaults to
            ``scalings=dict(eeg=1e6, grad=1e13, mag=1e15, misc=1.0)`.
        use_time_index : bool
            If False, times will be included as in the data table, else it will
            be used as index object.
        copy : bool
            If true, data will be copied. Else data may be modified in place.

        Returns
        -------
        df : instance of pandas.core.DataFrame
            Raw data exported into tabular data structure.
        """

        pd = _check_pandas_installed()
        if picks is None:
            picks = list(range(self.info['nchan']))

        data, times = self[picks, start:stop]

        if copy:
            data = data.copy()

        types = [channel_type(self.info, idx) for idx in picks]
        n_channel_types = 0
        ch_types_used = []

        scalings = _mutable_defaults(('scalings', scalings))[0]
        for t in scalings.keys():
            if t in types:
                n_channel_types += 1
                ch_types_used.append(t)

        for t in ch_types_used:
            scaling = scalings[t]
            idx = [picks[i] for i in range(len(picks)) if types[i] == t]
            if len(idx) > 0:
                data[idx] *= scaling

        assert times.shape[0] == data.shape[1]
        col_names = [self.ch_names[k] for k in picks]

        df = pd.DataFrame(data.T, columns=col_names)
        df.insert(0, 'time', times * scale_time)

        if use_time_index is True:
            if 'time' in df:
                df['time'] = df['time'].astype(np.int64)
            with warnings.catch_warnings(record=True):
                df.set_index('time', inplace=True)

        return df

    def to_nitime(self, picks=None, start=None, stop=None,
                  use_first_samp=False, copy=True):
        """ Raw data as nitime TimeSeries

        Parameters
        ----------
        picks : array-like of int | None
            Indices of channels to apply. If None, all channels will be
            exported.
        start : int | None
            Data-extraction start index. If None, data will be exported from
            the first sample.
        stop : int | None
            Data-extraction stop index. If None, data will be exported to the
            last index.
        use_first_samp: bool
            If True, the time returned is relative to the session onset, else
            relative to the recording onset.
        copy : bool
            Whether to copy the raw data or not.

        Returns
        -------
        raw_ts : instance of nitime.TimeSeries
        """
        try:
            from nitime import TimeSeries  # to avoid strong dependency
        except ImportError:
            raise Exception('the nitime package is missing')

        data, _ = self[picks, start:stop]
        if copy:
            data = data.copy()

        start_time = self.index_as_time(start if start else 0, use_first_samp)
        raw_ts = TimeSeries(data, sampling_rate=self.info['sfreq'],
                            t0=start_time)

        raw_ts.ch_names = [self.ch_names[k] for k in picks]

        return raw_ts

    def __repr__(self):
        s = "n_channels x n_times : %s x %s" % (len(self.info['ch_names']),
                                                self.n_times)
        return "<Raw  |  %s>" % s

    def add_events(self, events, stim_channel=None):
        """Add events to stim channel

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
        stim_channel = _get_stim_channel(stim_channel)
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


def set_eeg_reference(raw, ref_channels, copy=True):
    """Rereference eeg channels to new reference channel(s).

    If multiple reference channels are specified, they will be averaged.

    Parameters
    ----------
    raw : instance of Raw
        Instance of Raw with eeg channels and reference channel(s).

    ref_channels : list of str
        The name(s) of the reference channel(s).

    copy : bool
        Specifies whether instance of Raw will be copied or modified in place.

    Returns
    -------
    raw : instance of Raw
        Instance of Raw with eeg channels rereferenced.

    ref_data : array
        Array of reference data subtracted from eeg channels.
    """
    # Check to see that raw data is preloaded
    if not raw.preload:
        raise RuntimeError('Raw data needs to be preloaded. Use '
                           'preload=True (or string) in the constructor.')
    # Make sure that reference channels are loaded as list of string
    if not isinstance(ref_channels, list):
        raise IOError('Reference channel(s) must be a list of string. '
                      'If using a single reference channel, enter as '
                      'a list with one element.')
    # Find the indices to the reference electrodes
    ref_idx = [raw.ch_names.index(c) for c in ref_channels]

    # Get the reference array
    ref_data = raw._data[ref_idx].mean(0)

    # Get the indices to the eeg channels using the pick_types function
    eeg_idx = pick_types(raw.info, exclude="bads", eeg=True, meg=False,
                         ref_meg=False)

    # Copy raw data or modify raw data in place
    if copy:  # copy data
        raw = raw.copy()

    # Rereference the eeg channels
    raw._data[eeg_idx] -= ref_data

    # Return rereferenced data and reference array
    return raw, ref_data


def _allocate_data(data, data_buffer, data_shape, dtype):
    if data is None:
        # if not already done, allocate array with right type
        if isinstance(data_buffer, string_types):
            # use a memmap
            data = np.memmap(data_buffer, mode='w+',
                             dtype=dtype, shape=data_shape)
        else:
            data = np.zeros(data_shape, dtype=dtype)
    return data


def _time_as_index(times, sfreq, first_samp=0, use_first_samp=False):
    """Convert time to indices

    Parameters
    ----------
    times : list-like | float | int
        List of numbers or a number representing points in time.
    use_first_samp : boolean
        If True, time is treated as relative to the session onset, else
        as relative to the recording onset.

    Returns
    -------
    index : ndarray
        Indices corresponding to the times supplied.
    """
    index = np.atleast_1d(times) * sfreq
    index -= (first_samp if use_first_samp else 0)
    return index.astype(int)


def _index_as_time(index, sfreq, first_samp=0, use_first_samp=False):
    """Convert indices to time

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
    """Used for creating a temporary raw object"""

    def __init__(self):
        self.first_samp = None
        self.last_samp = None
        self.cals = None
        self.rawdir = None
        self._projector = None

    @property
    def n_times(self):
        return self.last_samp - self.first_samp + 1


###############################################################################
# Writing
def _write_raw(fname, raw, info, picks, format, data_type, reset_range, start,
               stop, buffer_size, projector, inv_comp, drop_small_buffer,
               split_size, part_idx, prev_fname):
    """Write raw file with splitting
    """

    if part_idx > 0:
        # insert index in filename
        path, base = op.split(fname)
        idx = base.find('.')
        use_fname = op.join(path, '%s-%d.%s' % (base[:idx], part_idx,
                                                base[idx + 1:]))
    else:
        use_fname = fname
    logger.info('Writing %s' % use_fname)

    meas_id = info['meas_id']

    fid, cals = _start_writing_raw(use_fname, info, picks, data_type,
                                   reset_range)

    first_samp = raw.first_samp + start
    if first_samp != 0:
        write_int(fid, FIFF.FIFF_FIRST_SAMPLE, first_samp)

    # previous file name and id
    if part_idx > 0 and prev_fname is not None:
        start_block(fid, FIFF.FIFFB_REF)
        write_int(fid, FIFF.FIFF_REF_ROLE, FIFF.FIFFV_ROLE_PREV_FILE)
        write_string(fid, FIFF.FIFF_REF_FILE_NAME, prev_fname)
        if meas_id is not None:
            write_id(fid, FIFF.FIFF_REF_FILE_ID, meas_id)
        write_int(fid, FIFF.FIFF_REF_FILE_NUM, part_idx - 1)
        end_block(fid, FIFF.FIFFB_REF)

    pos_prev = None
    for first in range(start, stop, buffer_size):
        last = first + buffer_size
        if last >= stop:
            last = stop + 1

        if picks is None:
            data, times = raw[:, first:last]
        else:
            data, times = raw[picks, first:last]

        if projector is not None:
            data = np.dot(projector, data)

        if ((drop_small_buffer and (first > start)
             and (len(times) < buffer_size))):
            logger.info('Skipping data chunk due to small buffer ... '
                        '[done]')
            break
        logger.info('Writing ...')

        if pos_prev is None:
            pos_prev = fid.tell()

        _write_raw_buffer(fid, data, cals, format, inv_comp)

        pos = fid.tell()
        this_buff_size_bytes = pos - pos_prev
        if this_buff_size_bytes > split_size / 2:
            raise ValueError('buffer size is too large for the given split'
                             'size: decrease "buffer_size_sec" or increase'
                             '"split_size".')
        if pos > split_size:
            raise logger.warning('file is larger than "split_size"')

        # Split files if necessary, leave some space for next file info
        if pos >= split_size - this_buff_size_bytes - 2 ** 20:
            next_fname, next_idx = _write_raw(fname, raw, info, picks, format,
                data_type, reset_range, first + buffer_size, stop, buffer_size,
                projector, inv_comp, drop_small_buffer, split_size,
                part_idx + 1, use_fname)

            start_block(fid, FIFF.FIFFB_REF)
            write_int(fid, FIFF.FIFF_REF_ROLE, FIFF.FIFFV_ROLE_NEXT_FILE)
            write_string(fid, FIFF.FIFF_REF_FILE_NAME, op.basename(next_fname))
            if meas_id is not None:
                write_id(fid, FIFF.FIFF_REF_FILE_ID, meas_id)
            write_int(fid, FIFF.FIFF_REF_FILE_NUM, next_idx)
            end_block(fid, FIFF.FIFFB_REF)
            break

        pos_prev = pos

    logger.info('Closing %s [done]' % use_fname)
    _finish_writing_raw(fid)

    return use_fname, part_idx


def _start_writing_raw(name, info, sel=None, data_type=FIFF.FIFFT_FLOAT,
                       reset_range=True):
    """Start write raw data in file

    Data will be written in float

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

    Returns
    -------
    fid : file
        The file descriptor.
    cals : list
        calibration factors.
    """
    #
    #  Create the file and save the essentials
    #
    fid = start_file(name)
    start_block(fid, FIFF.FIFFB_MEAS)
    write_id(fid, FIFF.FIFF_BLOCK_ID)
    if info['meas_id'] is not None:
        write_id(fid, FIFF.FIFF_PARENT_BLOCK_ID, info['meas_id'])
    #
    #    Measurement info
    #
    info = copy.deepcopy(info)
    if sel is not None:
        info['chs'] = [info['chs'][k] for k in sel]
        info['nchan'] = len(sel)

        ch_names = [c['ch_name'] for c in info['chs']]  # name of good channels
        comps = copy.deepcopy(info['comps'])
        for c in comps:
            row_idx = [k for k, n in enumerate(c['data']['row_names'])
                       if n in ch_names]
            row_names = [c['data']['row_names'][i] for i in row_idx]
            rowcals = c['rowcals'][row_idx]
            c['rowcals'] = rowcals
            c['data']['nrow'] = len(row_names)
            c['data']['row_names'] = row_names
            c['data']['data'] = c['data']['data'][row_idx]
        info['comps'] = comps

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
    # Start the raw data
    #
    start_block(fid, FIFF.FIFFB_RAW_DATA)

    return fid, cals


def _write_raw_buffer(fid, buf, cals, format, inv_comp):
    """Write raw buffer

    Parameters
    ----------
    fid : file descriptor
        an open raw data file.
    buf : array
        The buffer to write.
    cals : array
        Calibration factors.
    format : str
        'short', 'int', 'single', or 'double' for 16/32 bit int or 32/64 bit
        float for each item. This will be doubled for complex datatypes. Note
        that short and int formats cannot be used for complex data.
    inv_comp : array | None
        The CTF compensation matrix used to revert compensation
        change when reading.
    """
    if buf.shape[0] != len(cals):
        raise ValueError('buffer and calibration sizes do not match')

    if not format in ['short', 'int', 'single', 'double']:
        raise ValueError('format must be "short", "single", or "double"')

    if np.isrealobj(buf):
        if format == 'short':
            write_function = write_dau_pack16
        elif format == 'int':
            write_function = write_int
        elif format == 'single':
            write_function = write_float
        else:
            write_function = write_double
    else:
        if format == 'single':
            write_function = write_complex64
        elif format == 'double':
            write_function = write_complex128
        else:
            raise ValueError('only "single" and "double" supported for '
                             'writing complex data')

    if inv_comp is not None:
        buf = np.dot(inv_comp / np.ravel(cals)[:, None], buf)
    else:
        buf = buf / np.ravel(cals)[:, None]

    write_function(fid, FIFF.FIFF_DATA_BUFFER, buf)


def _finish_writing_raw(fid):
    """Finish writing raw FIF file

    Parameters
    ----------
    fid : file descriptor
        an open raw data file.
    """
    end_block(fid, FIFF.FIFFB_RAW_DATA)
    end_block(fid, FIFF.FIFFB_MEAS)
    end_file(fid)


def _envelope(x):
    """ Compute envelope signal """
    return np.abs(hilbert(x))


def _check_raw_compatibility(raw):
    """Check to make sure all instances of Raw
    in the input list raw have compatible parameters"""
    for ri in range(1, len(raw)):
        if not raw[ri].info['nchan'] == raw[0].info['nchan']:
            raise ValueError('raw[%d][\'info\'][\'nchan\'] must match' % ri)
        if not raw[ri].info['bads'] == raw[0].info['bads']:
            raise ValueError('raw[%d][\'info\'][\'bads\'] must match' % ri)
        if not raw[ri].info['sfreq'] == raw[0].info['sfreq']:
            raise ValueError('raw[%d][\'info\'][\'sfreq\'] must match' % ri)
        if not set(raw[ri].info['ch_names']) == set(raw[0].info['ch_names']):
            raise ValueError('raw[%d][\'info\'][\'ch_names\'] must match' % ri)
        if not all(raw[ri].cals == raw[0].cals):
            raise ValueError('raw[%d].cals must match' % ri)
        if len(raw[0].info['projs']) != len(raw[ri].info['projs']):
            raise ValueError('SSP projectors in raw files must be the same')
        if not all(proj_equal(p1, p2) for p1, p2 in
                   zip(raw[0].info['projs'], raw[ri].info['projs'])):
            raise ValueError('SSP projectors in raw files must be the same')
    if not all([r.orig_format == raw[0].orig_format for r in raw]):
        warnings.warn('raw files do not all have the same data format, '
                      'could result in precision mismatch. Setting '
                      'raw.orig_format="unknown"')
        raw[0].orig_format = 'unknown'


def concatenate_raws(raws, preload=None, events_list=None):
    """Concatenate raw instances as if they were continuous. Note that raws[0]
    is modified in-place to achieve the concatenation.

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


def get_chpi_positions(raw, t_step=None):
    """Extract head positions

    Note that the raw instance must have CHPI channels recorded.

    Parameters
    ----------
    raw : instance of Raw | str
        Raw instance to extract the head positions from. Can also be a
        path to a Maxfilter log file (str).
    t_step : float | None
        Sampling interval to use when converting data. If None, it will
        be automatically determined. By default, a sampling interval of
        1 second is used if processing a raw data. If processing a
        Maxfilter log file, this must be None because the log file
        itself will determine the sampling interval.

    Returns
    -------
    translation : array
        A 2-dimensional array of head position vectors (n_time x 3).
    rotation : array
        A 3-dimensional array of rotation matrices (n_time x 3 x 3).
    t : array
        The time points associated with each position (n_time).

    Notes
    -----
    The digitized HPI head frame y is related to the frame position X as:

        Y = np.dot(rotation, X) + translation

    Note that if a Maxfilter log file is being processed, the start time
    may not use the same reference point as the rest of mne-python (i.e.,
    it could be referenced relative to raw.first_samp or something else).
    """
    if isinstance(raw, _BaseRaw):
        # for simplicity, we'll sample at 1 sec intervals like maxfilter
        if t_step is None:
            t_step = 1.0
        if not np.isscalar(t_step):
            raise TypeError('t_step must be a scalar or None')
        picks = pick_types(raw.info, meg=False, ref_meg=False,
                           chpi=True, exclude=[])
        if len(picks) == 0:
            raise RuntimeError('raw file has no CHPI channels')
        time_idx = raw.time_as_index(np.arange(0, raw.n_times
                                               / raw.info['sfreq'], t_step))
        data = [raw[picks, ti] for ti in time_idx]
        t = np.array([d[1] for d in data])
        data = np.array([d[0][:, 0] for d in data])
        data = np.c_[t, data]
    else:
        if not isinstance(raw, string_types):
            raise TypeError('raw must be an instance of Raw or string')
        if not op.isfile(raw):
            raise IOError('File "%s" does not exist' % raw)
        if t_step is not None:
            raise ValueError('t_step must be None if processing a log')
        data = np.loadtxt(raw, skiprows=1)  # first line is header, skip it
    t = data[:, 0]
    translation = data[:, 4:7].copy()
    rotation = _quart_to_rot(data[:, 1:4])
    return translation, rotation, t


def _quart_to_rot(q):
    """Helper to convert quarternions to rotations"""
    q0 = np.sqrt(1 - np.sum(q[:, 0:3] ** 2, 1))
    q1 = q[:, 0]
    q2 = q[:, 1]
    q3 = q[:, 2]
    rotation = np.array((np.c_[(q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2,
                                2 * (q1 * q2 - q0 * q3),
                                2 * (q1 * q3 + q0 * q2))],
                         np.c_[(2 * (q1 * q2 + q0 * q3),
                                q0 ** 2 + q2 ** 2 - q1 ** 2 - q3 ** 2,
                                2 * (q2 * q3 - q0 * q1))],
                         np.c_[(2 * (q1 * q3 - q0 * q2),
                                2 * (q2 * q3 + q0 * q1),
                                q0 ** 2 + q3 ** 2 - q1 ** 2 - q2 ** 2)]
                         ))
    rotation = np.swapaxes(rotation, 0, 1).copy()
    return rotation
