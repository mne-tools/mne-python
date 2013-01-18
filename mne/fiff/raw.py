# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

from math import floor, ceil
import copy
import warnings
import os
import os.path as op

import numpy as np
from scipy.signal import hilbert
from copy import deepcopy

import logging
logger = logging.getLogger('mne')

from .constants import FIFF
from .open import fiff_open
from .meas_info import read_meas_info, write_meas_info
from .tree import dir_tree_find
from .tag import read_tag
from .pick import pick_types, channel_type
from .proj import setup_proj, activate_proj, deactivate_proj, proj_equal

from ..filter import low_pass_filter, high_pass_filter, band_pass_filter, \
                     notch_filter, band_stop_filter, resample
from ..parallel import parallel_func
from ..utils import deprecated
from .. import verbose


class Raw(object):
    """Raw data

    Parameters
    ----------
    fnames : list, or string
        A list of the raw files to treat as a Raw instance, or a single
        raw file.
    allow_maxshield : bool, (default False)
        allow_maxshield if True, allow loading of data that has been
        processed with Maxshield. Maxshield-processed data should generally
        not be loaded directly, but should be processed using SSS first.
    preload : bool or str (default False)
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, preload is the
        file name of a memory-mapped file which is used to store the data
        on the hard drive (slower, requires less memory).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    proj_active : bool
        Apply the signal space projection (SSP) operators present in
        the file to the data. Note: Once the projectors have been
        applied, they can no longer be removed. It is usually not
        recommended to apply the projectors at this point as they are
        applied automatically later on (e.g. when computing inverse
        solutions).

    Attributes
    ----------
    info : dict
        Measurement info.
    ch_names : list of string
        List of channels' names.
    n_times : int
        Total number of time points in the raw file.
    verbose : bool, str, int, or None
        See above.
    """
    @verbose
    def __init__(self, fnames, allow_maxshield=False, preload=False,
                 verbose=None, proj_active=False):

        if not isinstance(fnames, list):
            fnames = [op.abspath(fnames)] if not op.isabs(fnames) else [fnames]
        else:
            fnames = [op.abspath(f) if not op.isabs(f) else f for f in fnames]

        raws = [self._read_raw_file(fname, allow_maxshield, preload)
                for fname in fnames]

        _check_raw_compatibility(raws)

        # combine information from each raw file to construct self
        self.first_samp = raws[0].first_samp  # meta first sample
        self._first_samps = np.array([r.first_samp for r in raws])
        self._last_samps = np.array([r.last_samp for r in raws])
        self._raw_lengths = np.array([r.last_samp - r.first_samp + 1
                                      for r in raws])
        self.last_samp = self.first_samp + sum(self._raw_lengths) - 1
        self.cals = raws[0].cals
        self.rawdirs = [r.rawdir for r in raws]
        self.comp = None
        self.fids = [r.fid for r in raws]
        self.info = copy.deepcopy(raws[0].info)
        self.verbose = verbose
        self.info['filenames'] = fnames

        if preload:
            self._preload_data(preload)
        else:
            self._preloaded = False

        # setup the SSP projector
        self._projector = None
        if proj_active:
            self.apply_projector()

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

    def _preload_data(self, preload):
        """This function actually preloads the data"""
        if isinstance(preload, basestring):
            # we will use a memmap: preload is a filename
            data_buffer = preload
        else:
            data_buffer = None

        self._data, self._times = read_raw_segment(self,
                                                   data_buffer=data_buffer)
        self._preloaded = True

    @verbose
    def _read_raw_file(self, fname, allow_maxshield, preload, verbose=None):
        """Read in header information from a raw file"""
        logger.info('Opening raw data file %s...' % fname)

        #   Read in the whole file if preload is on and .fif.gz (saves time)
        ext = os.path.splitext(fname)[1].lower()
        whole_file = preload if '.gz' in ext else False
        fid, tree, _ = fiff_open(fname, preload=whole_file)

        #   Read the measurement info
        info, meas = read_meas_info(fid, tree)

        #   Locate the data of interest
        raw_node = dir_tree_find(meas, FIFF.FIFFB_RAW_DATA)
        if len(raw_node) == 0:
            raw_node = dir_tree_find(meas, FIFF.FIFFB_CONTINUOUS_DATA)
            if allow_maxshield:
                raw_node = dir_tree_find(meas, FIFF.FIFFB_SMSH_RAW_DATA)
                if len(raw_node) == 0:
                    raise ValueError('No raw data in %s' % fname)
            else:
                if len(raw_node) == 0:
                    raise ValueError('No raw data in %s' % fname)

        if len(raw_node) == 1:
            raw_node = raw_node[0]

        #   Set up the output structure
        info['filename'] = fname

        #   Process the directory
        directory = raw_node['directory']
        nent = raw_node['nent']
        nchan = int(info['nchan'])
        first = 0
        first_samp = 0
        first_skip = 0

        #   Get first sample tag if it is there
        if directory[first].kind == FIFF.FIFF_FIRST_SAMPLE:
            tag = read_tag(fid, directory[first].pos)
            first_samp = int(tag.data)
            first += 1

        #   Omit initial skip
        if directory[first].kind == FIFF.FIFF_DATA_SKIP:
            # This first skip can be applied only after we know the buffer size
            tag = read_tag(fid, directory[first].pos)
            first_skip = int(tag.data)
            first += 1

        #  Get first sample tag if it is there
        if directory[first].kind == FIFF.FIFF_FIRST_SAMPLE:
            tag = read_tag(fid, directory[first].pos)
            first_samp += int(tag.data)
            first += 1

        raw = _RawShell()
        raw.first_samp = first_samp

        #   Go through the remaining tags in the directory
        rawdir = list()
        nskip = 0
        for k in range(first, nent):
            ent = directory[k]
            if ent.kind == FIFF.FIFF_DATA_SKIP:
                tag = read_tag(fid, ent.pos)
                nskip = int(tag.data)
            elif ent.kind == FIFF.FIFF_DATA_BUFFER:
                #   Figure out the number of samples in this buffer
                if ent.type == FIFF.FIFFT_DAU_PACK16:
                    nsamp = ent.size / (2 * nchan)
                elif ent.type == FIFF.FIFFT_SHORT:
                    nsamp = ent.size / (2 * nchan)
                elif ent.type == FIFF.FIFFT_FLOAT:
                    nsamp = ent.size / (4 * nchan)
                elif ent.type == FIFF.FIFFT_INT:
                    nsamp = ent.size / (4 * nchan)
                elif ent.type == FIFF.FIFFT_COMPLEX_FLOAT:
                    nsamp = ent.size / (8 * nchan)
                else:
                    fid.close()
                    raise ValueError('Cannot handle data buffers of type %d' %
                                                                      ent.type)

                #  Do we have an initial skip pending?
                if first_skip > 0:
                    first_samp += nsamp * first_skip
                    raw.first_samp = first_samp
                    first_skip = 0

                #  Do we have a skip pending?
                if nskip > 0:
                    rawdir.append(dict(ent=None, first=first_samp,
                                       last=first_samp + nskip * nsamp - 1,
                                       nsamp=nskip * nsamp))
                    first_samp += nskip * nsamp
                    nskip = 0

                #  Add a data buffer
                rawdir.append(dict(ent=ent, first=first_samp,
                                   last=first_samp + nsamp - 1,
                                   nsamp=nsamp))
                first_samp += nsamp

        raw.last_samp = first_samp - 1

        #   Add the calibration factors
        cals = np.zeros(info['nchan'])
        for k in range(info['nchan']):
            cals[k] = info['chs'][k]['range'] * \
                      info['chs'][k]['cal']

        raw.cals = cals
        raw.rawdir = rawdir
        raw.comp = None
        # XXX raw.comp never changes!
        logger.info('    Range : %d ... %d =  %9.3f ... %9.3f secs' % (
                    raw.first_samp, raw.last_samp,
                    float(raw.first_samp) / info['sfreq'],
                    float(raw.last_samp) / info['sfreq']))

        raw.fid = fid
        raw.info = info
        raw.verbose = verbose

        logger.info('Ready.')

        return raw

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
            sel = range(start, stop, step)
        else:
            sel = item[0]

        if isinstance(item[1], slice):
            time_slice = item[1]
            start, stop, step = time_slice.start, time_slice.stop, \
                                time_slice.step
        elif isinstance(item[1], int):
            start, stop, step = item[1], item[1] + 1, 1
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
        if self._preloaded:
            data, times = self._data[sel, start:stop], self._times[start:stop]
        else:
            data, times = read_raw_segment(self, start=start, stop=stop,
                                           sel=sel, proj=self._projector,
                                           verbose=self.verbose)
        return data, times

    def __setitem__(self, item, value):
        """setting raw data content with python slicing"""
        if not self._preloaded:
            raise RuntimeError('Modifying data of Raw is only supported '
                               'when preloading is used. Use preload=True '
                               '(or string) in the constructor.')
        sel, start, stop = self._parse_get_set_params(item)
        # set the data
        self._data[sel, start:stop] = value

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
              addtional time points need to be temporaily stored in memory.

        Note: If the data type changes (dtype != None), more memory is required
              since the original and the converted data needs to be stored in
              memory.

        Parameters
        ----------
        fun : function
            A function to be applied to the channels. The first argument of
            fun has to be a timeseries (numpy.ndarray). The function must
            return an numpy.ndarray with the same size as the input.
        picks : list of int
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
        if not self._preloaded:
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
              "len(picks) * n_times" addtional time points need to be
              temporaily stored in memory.

        Parameters
        ----------
        picks : list of int
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
    def filter(self, l_freq, h_freq, picks=None, filter_length=None,
               l_trans_bandwidth=0.5, h_trans_bandwidth=0.5, n_jobs=1,
               method='fft', iir_params=dict(order=4, ftype='butter'),
               verbose=None):
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
              addtional time points need to be temporaily stored in memory.

        Note: self.info['lowpass'] and self.info['highpass'] are only updated
              with picks=None.

        Parameters
        ----------
        l_freq : float | None
            Low cut-off frequency in Hz. If None the data are only low-passed.
        h_freq : float
            High cut-off frequency in Hz. If None the data are only
            high-passed.
        picks : list of int | None
            Indices of channels to filter. If None only the data (MEG/EEG)
            channels will be filtered (except bad channels).
        filter_length : int (default: None)
            Length of the filter to use (e.g. 4096).
            If None or "n_times < filter_length",
            (n_times: number of timepoints in Raw object) the filter length
            used is n_times. Otherwise, overlap-add filtering with a
            filter of the specified length is used (faster for long signals).
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
        iir_params : dict
            Dictionary of parameters to use for IIR filtering.
            See mne.filter.construct_iir_filter for details.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
            Defaults to self.verbose.
        """
        if verbose is None:
            verbose = self.verbose
        fs = float(self.info['sfreq'])
        if l_freq == 0:
            l_freq = None
        if h_freq > (fs / 2.):
            h_freq = None
        if not self._preloaded:
            raise RuntimeError('Raw data needs to be preloaded to filter. Use '
                               'preload=True (or string) in the constructor.')
        if picks is None:
            picks = pick_types(self.info, meg=True, eeg=True, exclude='bads')

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
    def notch_filter(self, freqs, picks=None, filter_length=None,
                     notch_widths=None, trans_bandwidth=1.0, n_jobs=1,
                     method='fft', iir_params=dict(order=4, ftype='butter'),
                     mt_bandwidth=None, p_value=0.05, verbose=None):
        """Notch filter a subset of channels.

        Applies a zero-phase notch filter to the channels selected by
        "picks". The data of the Raw object is modified inplace.

        The Raw object has to be constructed using preload=True (or string).

        Note: If n_jobs > 1, more memory is required as "len(picks) * n_times"
              addtional time points need to be temporaily stored in memory.

        Parameters
        ----------
        freqs : float | array of float | None
            Specific frequencies to filter out from data, e.g.,
            np.arange(60, 60, 241) in the US or np.arange(50, 50, 251) in
            Europe. None can only be used with the mode 'spectrum_fit',
            where an F test is used to find sinusoidal components.
        picks : list of int | None
            Indices of channels to filter. If None only the data (MEG/EEG)
            channels will be filtered (excep bad channels).
        filter_length : int (default: None)
            Length of the filter to use (e.g. 4096).
            If None or "n_times < filter_length",
            (n_times: number of timepoints in Raw object) the filter length
            used is n_times. Otherwise, overlap-add filtering with a
            filter of the specified length is used (faster for long signals).
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
        iir_params : dict
            Dictionary of parameters to use for IIR filtering.
            See mne.filter.construct_iir_filter for details.
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
            picks = pick_types(self.info, meg=True, eeg=True, exclude='bads')
        if not self._preloaded:
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
            mne.fiff.pick_types(raw.info, meg=False, stim=True, exclude=[]).
        n_jobs : int
            Number of jobs to run in parallel.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
            Defaults to self.verbose.

        Notes
        -----
        For some data, it may be more accurate to use npad=0 to reduce
        artifacts. This is dataset dependent -- check your data!
        """
        if not self._preloaded:
            raise RuntimeError('Can only resample preloaded data')

        o_sfreq = self.info['sfreq']
        offsets = np.concatenate(([0], np.cumsum(self._raw_lengths)))
        new_data = list()
        # set up stim channel processing
        if stim_picks is None:
            stim_picks = pick_types(self.info, meg=False, stim=True,
                                    exclude=[])
        stim_picks = np.asanyarray(stim_picks)
        ratio = sfreq / float(o_sfreq)
        for ri in range(len(self._raw_lengths)):
            data_chunk = self._data[:, offsets[ri]:offsets[ri + 1]]
            # use parallel function to resample each channel separately
            # for speed and to save memory (faster not to use array_split, too)
            parallel, my_resample, _ = parallel_func(resample, n_jobs)
            new_data.append(np.array(parallel(my_resample(d, sfreq, o_sfreq,
                                              npad, 0) for d in data_chunk)))
            new_ntimes = new_data[ri].shape[1]

            # Now deal with the stim channels. In empirical testing, it was
            # faster to resample all channels (above) and then replace the
            # stim channels than it was to only resample the proper subset
            # of channels and then use np.insert() to restore the stims

            # figure out which points in old data to subsample
            stim_inds = np.floor(np.arange(new_ntimes) / ratio).astype(int)
            for sp in stim_picks:
                new_data[ri][sp] = data_chunk[sp][:, stim_inds]

            self._first_samps[ri] = int(self._first_samps[ri] * ratio)
            self._last_samps[ri] = self._first_samps[ri] + new_ntimes - 1
            self._raw_lengths[ri] = new_ntimes

        # adjust affected variables
        self._data = np.concatenate(new_data, axis=1)
        self.first_samp = self._first_samps[0]
        self.last_samp = self.first_samp + self._data.shape[1] - 1
        self.info['sfreq'] = sfreq

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

        smin = raw.time_as_index(tmin)
        smax = raw.time_as_index(tmax)
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
        raw.fids = [f for fi, f in enumerate(raw.fids) if fi in keepers]
        raw.rawdirs = [r for ri, r in enumerate(raw.rawdirs)
                       if ri in keepers]
        if raw._preloaded:
            raw._data = raw._data[:, smin:smax + 1]
        raw.first_samp = raw._first_samps[0]
        raw.last_samp = raw.first_samp + (smax - smin)
        return raw

    def apply_projector(self):
        """Apply the signal space projection (SSP) operators to the data.

        Note: Once the projectors have been applied, they can no longer be
              removed. It is usually not recommended to apply the projectors at
              this point, as they are applied automatically later on (e.g. when
              computing inverse solutions).
       """
        self._projector, self.info = setup_proj(self.info,
                                                verbose=self.verbose)
        activate_proj(self.info['projs'], copy=False, verbose=self.verbose)

        if self._preloaded:
            self._data = np.dot(self._projector, self._data)

    def add_proj(self, projs, remove_existing=False):
        """Add SSP projection vectors

        Parameters
        ----------
        projs : list
            List with projection vectors.
        remove_existing : bool
            Remove the projection vectors currently in the file.
        """
        # mark proj as inactive, as they have not been applied
        projs = deactivate_proj(projs, copy=True, verbose=self.verbose)

        if remove_existing:
            # we cannot remove the proj if they are active
            if any(p['active'] for p in self.info['projs']):
                raise ValueError('Cannot remove projectors that have '
                                 'already been applied')
            self.info['projs'] = projs
        else:
            self.info['projs'].extend(projs)

    def del_proj(self, idx):
        """Remove SSP projection vector

        Note: The projection vector can only be removed if it is inactive
              (has not been applied to the data).

        Parameters:
        -----------
        idx : int
            Index of the projector to remove.
        """
        if self.info['projs'][idx]['active']:
            raise ValueError('Cannot remove projectors that have already '
                             'been applied')

        self.info['projs'].pop(idx)

    @verbose
    def save(self, fname, picks=None, tmin=0, tmax=None, buffer_size_sec=10,
             drop_small_buffer=False, proj_active=False, verbose=None):
        """Save raw data to file

        Parameters
        ----------
        fname : string
            File name of the new dataset. Caveat! This has to be a new
            filename.
        picks : list of int
            Indices of channels to include.
        tmin : float
            Time in seconds of first sample to save.
        tmax : float
            Time in seconds of last sample to save.
        buffer_size_sec : float
            Size of data chuncks in seconds.
        drop_small_buffer : bool
            Drop or not the last buffer. It is required by maxfilter (SSS)
            that only accepts raw files with buffers of the same size.
        proj_active : bool
            If True the data is saved with the projections applied (active).
            Note: If apply_projector() was used to apply the projectons,
            the projectons will be active even if proj_active is False.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
            Defaults to self.verbose.
        """
        fname = op.abspath(fname)
        if fname in self.info['filenames']:
            raise ValueError('You cannot save data to the same file.'
                               ' Please use a different filename.')

        if self._preloaded:
            if np.iscomplexobj(self._data):
                warnings.warn('Saving raw file with complex data. Loading '
                              'with command-line MNE tools will not work.')

        if proj_active:
            info = copy.deepcopy(self.info)
            proj, info = setup_proj(info)
            activate_proj(info['projs'], copy=False)
        else:
            info = self.info
            proj = None

        outfid, cals = start_writing_raw(fname, info, picks)
        #
        #   Set up the reading parameters
        #

        #   Convert to samples
        start = int(floor(tmin * self.info['sfreq']))
        first_samp = self.first_samp + start

        if tmax is None:
            stop = self.last_samp + 1 - self.first_samp
        else:
            stop = int(floor(tmax * self.info['sfreq']))

        buffer_size = int(ceil(buffer_size_sec * self.info['sfreq']))
        #
        #   Read and write all the data
        #
        write_int(outfid, FIFF.FIFF_FIRST_SAMPLE, first_samp)
        for first in range(start, stop, buffer_size):
            last = first + buffer_size
            if last >= stop:
                last = stop + 1

            if picks is None:
                data, times = self[:, first:last]
            else:
                data, times = self[picks, first:last]

            if proj is not None:
                data = np.dot(proj, data)

            if (drop_small_buffer and (first > start)
                                            and (len(times) < buffer_size)):
                logger.info('Skipping data chunk due to small buffer ... '
                            '[done]')
                break
            logger.info('Writing ...')
            write_raw_buffer(outfid, data, cals)
            logger.info('[done]')

        finish_writing_raw(outfid)

    @deprecated('time_to_index is deprecated please use time_as_index instead.'
                ' Will be removed in v0.7.')
    def time_to_index(self, *args):
        """Convert time to indices"""
        indices = []
        for time in args:
            ind = int(time * self.info['sfreq'])
            indices.append(ind)
        return indices

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
        """Convert time to indices

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

    @property
    def ch_names(self):
        return self.info['ch_names']

    @property
    def n_times(self):
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
            bad_names = filter(None, open(bad_file).read().splitlines())
            names_there = [ci for ci in bad_names if ci in names]
            count_diff = len(bad_names) - len(names_there)

            if count_diff > 0:
                if not force:
                    raise ValueError('Bad channels from:\n%s\n not found '
                                     'in:\n%s' % (bad_file,
                                                  self.info['filenames'][0]))
                else:
                    warnings.warn('%d bad channels from:\n%s\nnot found '
                                  'in:\n%s' % (count_diff, bad_file,
                                               self.info['filenames'][0]))
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
        all_preloaded = self._preloaded and all(r._preloaded for r in raws)
        if preload is None:
            if all_preloaded:
                preload = True
            else:
                preload = False

        if preload is False:
            if self._preloaded:
                self._data = None
                self._times = None
            self._preloaded = False
        else:
            # do the concatenation ourselves since preload might be a string
            nchan = self.info['nchan']
            c_ns = [self.last_samp - self.first_samp + 1]
            c_ns += [r.last_samp - r.first_samp + 1 for r in raws]
            c_ns = np.cumsum(np.array(c_ns, dtype='int'))
            nsamp = c_ns[-1]

            if not self._preloaded:
                this_data = read_raw_segment(self)[0]
            else:
                this_data = self._data

            # allocate the buffer
            if isinstance(preload, basestring):
                _data = np.memmap(preload, mode='w+', dtype=this_data.dtype,
                                  shape=(nchan, nsamp))
            else:
                _data = np.empty((nchan, nsamp), dtype=this_data.dtype)

            _data[:, 0:c_ns[0]] = this_data

            for ri in range(len(raws)):
                if not r._preloaded:
                    # read the data directly into the buffer
                    data_buffer = _data[:, c_ns[ri]:c_ns[ri + 1]]
                    read_raw_segment(raws[ri], data_buffer=data_buffer)
                else:
                    _data[:, c_ns[ri]:c_ns[ri + 1]] = raws[ri]._data

            self._data = _data
            stop = self.last_samp - self.first_samp + 1
            self._times = np.arange(0, stop) / self.info['sfreq']
            self._preloaded = True

        # now combine information from each raw file to construct new self
        for r in raws:
            self._first_samps = np.r_[self._first_samps, r._first_samps]
            self._last_samps = np.r_[self._last_samps, r._last_samps]
            self._raw_lengths = np.r_[self._raw_lengths, r._raw_lengths]
            self.rawdirs += r.rawdirs
            self.fids += r.fids
            self.info['filenames'] += r.info['filenames']
        self.last_samp = self.first_samp + sum(self._raw_lengths) - 1

    def close(self):
        [f.close() for f in self.fids]

    def copy(self):
        """ Return copy of Raw instance
        """
        new = deepcopy(self)
        if self._preloaded:
            new.fids = []
        else:
            new.fids = [open(fname, "rb") for fname in self.info['filenames']]
            for new_fid, this_fid in zip(new.fids, self.fids):
                new_fid.seek(this_fid.tell())

        return new

    def as_data_frame(self, picks=None, start=None, stop=None, scale_time=1e3,
                      scalings=dict(mag=1e15, grad=1e13, eeg=1e6),
                      use_time_index=True, copy=True):
        """Get the epochs as Pandas DataFrame

        Export raw data in tabular structure with MEG channels.

        Caveat! To save memory, depending on selected data size consider
        setting copy to False.

        Parameters
        ----------
        picks : None | array of int
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
            Scaling to be applied to the channels picked. If None, no scaling
            will be applied.
        use_time_index : bool
            If False, times will be included as in the data table, else it will
            be used as index object.
        copy : bool
            If true, data will be copied. Else data may be modified in place.

        Returns
        -------
        df : instance of DataFrame
            Raw data exported into tabular data structure.
        """
        try:
            import pandas as pd
        except:
            raise RuntimeError('For this method you need an installation of '
                               'the Pandas library.')

        if picks is None:
            picks = range(self.info['nchan'])

        data, times = self[picks, start:stop]

        if copy:
            data = data.copy()

        types = [channel_type(self.info, idx) for idx in picks]
        n_channel_types = 0
        ch_types_used = []
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
            df.set_index('time', inplace=True)
            df.index = df.index.astype(int)

        return df

    def to_nitime(self, picks=None, start=None, stop=None,
                  use_first_samp=False, copy=True):
        """ Raw data as nitime TimeSeries

        Parameters
        ----------
        picks : array-like | None
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
                                       self.last_samp - self.first_samp + 1)
        return "<Raw  |  %s>" % s


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
    """Convert time to indices

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


@verbose
def read_raw_segment(raw, start=0, stop=None, sel=None, data_buffer=None,
    verbose=None, proj=None):
    """Read a chunck of raw data

    Parameters
    ----------
    raw : Raw object
        An instance of Raw.
    start : int, (optional)
        first sample to include (first is 0). If omitted, defaults to the first
        sample in data.
    stop : int, (optional)
        First sample to not include.
        If omitted, data is included to the end.
    sel : array, optional
        Indices of channels to select.
    data_buffer : array or str, optional
        numpy array to fill with data read, must have the correct shape.
        If str, a np.memmap with the correct data type will be used
        to store the data.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    proj : array
        SSP operator to apply to the data.

    Returns
    -------
    data : array, [channels x samples]
       the data matrix (channels x samples).
    times : array, [samples]
        returns the time values corresponding to the samples.
    """
    if stop is None:
        stop = raw.last_samp - raw.first_samp + 1

    #  Initial checks
    start = int(start)
    stop = int(stop)
    stop = min([stop, raw.last_samp - raw.first_samp + 1])

    if start >= stop:
        raise ValueError('No data in this range')

    logger.info('Reading %d ... %d  =  %9.3f ... %9.3f secs...' % (
                           start, stop - 1, start / float(raw.info['sfreq']),
                           (stop - 1) / float(raw.info['sfreq'])))

    #  Initialize the data and calibration vector
    nchan = raw.info['nchan']

    n_sel_channels = nchan if sel is None else len(sel)
    idx = slice(None, None, None) if sel is None else sel
    data_shape = (n_sel_channels, stop - start)
    if isinstance(data_buffer, np.ndarray):
        if data_buffer.shape != data_shape:
            raise ValueError('data_buffer has incorrect shape')
        data = data_buffer
    else:
        data = None  # we will allocate it later, once we know the type

    if proj is not None:
        mult = list()
        for ri in range(len(raw._raw_lengths)):
            mult.append(np.diag(raw.cals.ravel()))
            if raw.comp is not None:
                mult[ri] = np.dot(raw.comp[idx, :], mult[ri])
            mult[ri] = np.dot(proj, mult[ri])
    else:
        mult = None

    # deal with having multiple files accessed by the raw object
    cumul_lens = np.concatenate(([0], np.array(raw._raw_lengths, dtype='int')))
    cumul_lens = np.cumsum(cumul_lens)
    files_used = np.logical_and(np.less(start, cumul_lens[1:]),
                                np.greater_equal(stop - 1, cumul_lens[:-1]))

    first_file_used = False
    s_off = 0
    dest = 0
    for fi in np.nonzero(files_used)[0]:
        start_loc = raw._first_samps[fi]
        # first iteration (only) could start in the middle somewhere
        if not first_file_used:
            first_file_used = True
            start_loc += start - cumul_lens[fi]
        stop_loc = np.min([stop - 1 - cumul_lens[fi] + raw._first_samps[fi],
                           raw._last_samps[fi]])
        if start_loc < raw._first_samps[fi]:
            raise ValueError('Bad array indexing, could be a bug')
        if stop_loc > raw._last_samps[fi]:
            raise ValueError('Bad array indexing, could be a bug')
        if stop_loc < start_loc:
            raise ValueError('Bad array indexing, could be a bug')
        len_loc = stop_loc - start_loc + 1

        for this in raw.rawdirs[fi]:

            #  Do we need this buffer
            if this['last'] >= start_loc:
                if this['ent'] is None:
                    #  Take the easy route: skip is translated to zeros
                    logger.debug('S')
                    one = np.zeros((n_sel_channels, this['nsamp']))
                else:
                    tag = read_tag(raw.fids[fi], this['ent'].pos)

                    # decide what datatype to use
                    if np.isrealobj(tag.data):
                        dtype = np.float
                    else:
                        dtype = np.complex64

                    one = tag.data.reshape(this['nsamp'],
                                           nchan).astype(dtype).T
                    if mult is not None:  # use proj + cal factors in mult
                        one = np.dot(mult[fi], one)
                        one = one[idx]
                    else:  # apply just the calibration factors
                        one = raw.cals.ravel()[idx][:, np.newaxis] * one[idx]

                #  The picking logic is a bit complicated
                if stop_loc > this['last'] and start_loc < this['first']:
                    #    We need the whole buffer
                    first_pick = 0
                    last_pick = this['nsamp']
                    logger.debug('W')

                elif start_loc >= this['first']:
                    first_pick = start_loc - this['first']
                    if stop_loc <= this['last']:
                        #   Something from the middle
                        last_pick = this['nsamp'] + stop_loc - this['last']
                        logger.debug('M')
                    else:
                        #   From the middle to the end
                        last_pick = this['nsamp']
                        logger.debug('E')
                else:
                    #    From the beginning to the middle
                    first_pick = 0
                    last_pick = stop_loc - this['first'] + 1
                    logger.debug('B')

                #   Now we are ready to pick
                picksamp = last_pick - first_pick
                if picksamp > 0:
                    if data is None:
                        # if not already done, allocate array with right type
                        if isinstance(data_buffer, basestring):
                            # use a memmap
                            data = np.memmap(data_buffer, mode='w+',
                                             dtype=dtype, shape=data_shape)
                        else:
                            data = np.empty(data_shape, dtype=dtype)
                    data[:, dest:(dest + picksamp)] = \
                        one[:, first_pick:last_pick]
                    dest += picksamp

            #   Done?
            if this['last'] >= stop_loc:
                break

        raw.fids[fi].seek(0, 0)  # Go back to beginning of the file
        s_off += len_loc
        # double-check our math
        if not s_off == dest:
            raise ValueError('Incorrect file reading')

    logger.info('[done]')
    times = np.arange(start, stop) / raw.info['sfreq']

    return data, times


@verbose
def read_raw_segment_times(raw, start, stop, sel=None, verbose=None):
    """Read a chunck of raw data

    Parameters
    ----------
    raw : Raw object
        An instance of Raw.
    start : float
        Starting time of the segment in seconds.
    stop : float
        End time of the segment in seconds.
    sel : array, optional
        Indices of channels to select.
    node : tree node
        The node of the tree where to look.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    data : array, [channels x samples]
       the data matrix (channels x samples).
    times : array, [samples]
        returns the time values corresponding to the samples.
    """
    #   Convert to samples
    start = floor(start * raw.info['sfreq'])
    stop = ceil(stop * raw.info['sfreq'])

    #   Read it
    return read_raw_segment(raw, start, stop, sel)

###############################################################################
# Writing

from .write import start_file, end_file, start_block, end_block, \
                   write_float, write_complex64, write_int, write_id


def start_writing_raw(name, info, sel=None):
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
    if sel is not None:
        info = copy.deepcopy(info)
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
        info['chs'][k]['range'] = 1.0
        cals.append(info['chs'][k]['cal'])

    write_meas_info(fid, info, data_type=4)

    #
    # Start the raw data
    #
    start_block(fid, FIFF.FIFFB_RAW_DATA)

    return fid, cals


def write_raw_buffer(fid, buf, cals):
    """Write raw buffer

    Parameters
    ----------
    fid : file descriptor
        an open raw data file.

    buf : array
        The buffer to write.

    cals : array
        Calibration factors.
    """
    if buf.shape[0] != len(cals):
        raise ValueError('buffer and calibration sizes do not match')

    if np.isrealobj(buf):
        write_float(fid, FIFF.FIFF_DATA_BUFFER, buf / np.ravel(cals)[:, None])
    else:
        write_complex64(fid, FIFF.FIFF_DATA_BUFFER,
                        buf / np.ravel(cals)[:, None])


def finish_writing_raw(fid):
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
        if not set(raw[ri].info['ch_names']) \
                   == set(raw[0].info['ch_names']):
            raise ValueError('raw[%d][\'info\'][\'ch_names\'] must match' % ri)
        if not all(raw[ri].cals == raw[0].cals):
            raise ValueError('raw[%d].cals must match' % ri)
        if len(raw[0].info['projs']) != len(raw[ri].info['projs']):
            raise ValueError('SSP projectors in raw files must be the same')
        if not all(proj_equal(p1, p2) for p1, p2 in
                   zip(raw[0].info['projs'], raw[ri].info['projs'])):
            raise ValueError('SSP projectors in raw files must be the same')


def concatenate_raws(raws, preload=None):
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

    Returns
    -------
    raw : instance of Raw
        The result of the concatenation (first Raw instance passed in).
    """
    raws[0].append(raws[1:], preload)
    return raws[0]
