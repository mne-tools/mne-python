# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Daniel Strohmeier <daniel.strohmeier@tu-ilmenau.de>
#          Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import copy as cp
import warnings

import numpy as np
from copy import deepcopy

import logging
logger = logging.getLogger('mne')

from . import fiff
from .fiff.write import start_file, start_block, end_file, end_block, \
                    write_int, write_float_matrix, write_float, \
                    write_id
from .fiff.meas_info import read_meas_info, write_meas_info
from .fiff.open import fiff_open
from .fiff.tree import dir_tree_find
from .fiff.tag import read_tag
from .fiff import Evoked, FIFF
from .fiff.pick import pick_types, channel_indices_by_type
from .fiff.proj import setup_proj
from .baseline import rescale
from .utils import check_random_state
from .filter import resample
from .event import _read_events_fif
from . import verbose


class Epochs(object):
    """List of Epochs

    Parameters
    ----------
    raw : Raw object
        A instance of Raw

    events : array, of shape [n_events, 3]
        Returned by the read_events function

    event_id : int | None
        The id of the event to consider. If None all events are used.

    tmin : float
        Start time before event

    tmax : float
        End time after event

    name : string
        Comment that describes the Evoked data created.

    keep_comp : boolean
        Apply CTF gradient compensation

    baseline: None (default) or tuple of length 2
        The time interval to apply baseline correction.
        If None do not apply it. If baseline is (a, b)
        the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used
        and if b is None then b is set to the end of the interval.
        If baseline is equal to (None, None) all the time
        interval is used.

    preload : boolean
        Load all epochs from disk when creating the object
        or wait before accessing each epoch (more memory
        efficient but can be slower).

    reject : dict
        Epoch rejection parameters based on peak to peak amplitude.
        Valid keys are 'grad' | 'mag' | 'eeg' | 'eog' | 'ecg'.
        If reject is None then no rejection is done.
        Values are float. Example:
        reject = dict(grad=4000e-13, # T / m (gradiometers)
                      mag=4e-12, # T (magnetometers)
                      eeg=40e-6, # uV (EEG channels)
                      eog=250e-6 # uV (EOG channels)
                      )

    flat : dict
        Epoch rejection parameters based on flatness of signal
        Valid keys are 'grad' | 'mag' | 'eeg' | 'eog' | 'ecg'
        If flat is None then no rejection is done.

    proj : bool, optional
        Apply SSP projection vectors

    decim : int
        Factor by which to downsample the data from the raw file upon import.
        Warning: This simply selects every nth sample, data is not filtered
        here. If data is not properly filtered, aliasing artifacts may occur.

    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
        Defaults to raw.verbose.

    Attributes
    ----------
    info: dict
        Measurement info

    ch_names: list of string
        List of channels' names

    drop_log: list of lists
        This list (same length as events) contains the channel(s),
        if any, that caused an event in the original event list
        to be dropped by drop_bad_epochs().

    verbose : bool, str, int, or None
        See above.

    Methods
    -------
    get_data() : self
        Return all epochs as a 3D array [n_epochs x n_channels x n_times].

    average() : self
        Return Evoked object containing averaged epochs as a
        2D array [n_channels x n_times].

    standard_error() : self
        Return Evoked object containing standard error over epochs as a
        2D array [n_channels x n_times].

    drop_bad_epochs() : None
        Drop all epochs marked as bad. Should be used before indexing and
        slicing operations, and is done automatically by preload=True.

    drop_epochs() : self, indices
        Drop a set of epochs (both from preloaded data and event list).

    resample() : self, int, int, int, string or list
        Resample preloaded data.

    Notes
    -----
    For indexing and slicing:

    epochs = Epochs(...)

    epochs[idx] : Epochs
        Return Epochs object with a subset of epochs (supports single
        index and python style slicing)
    """
    @verbose
    def __init__(self, raw, events, event_id, tmin, tmax, baseline=(None, 0),
                 picks=None, name='Unknown', keep_comp=False, dest_comp=0,
                 preload=False, reject=None, flat=None, proj=True,
                 decim=1, verbose=None):
        if raw is None:
            return

        self.raw = raw
        self.verbose = raw.verbose if verbose is None else verbose
        self.event_id = event_id
        self.tmin = tmin
        self.tmax = tmax
        self.name = name
        self.keep_comp = keep_comp
        self.dest_comp = dest_comp
        self.baseline = baseline
        self.preload = preload
        self.reject = reject
        self.flat = flat
        self.proj = proj
        self.decim = decim = int(decim)
        self._bad_dropped = False
        self.drop_log = None

        # Handle measurement info
        self.info = cp.deepcopy(raw.info)
        if picks is not None:
            self.info['chs'] = [self.info['chs'][k] for k in picks]
            self.info['ch_names'] = [self.info['ch_names'][k] for k in picks]
            self.info['nchan'] = len(picks)

        if picks is None:
            picks = range(len(raw.info['ch_names']))
            self.ch_names = raw.info['ch_names']
        else:
            self.ch_names = [raw.info['ch_names'][k] for k in picks]
        self.picks = picks

        if len(picks) == 0:
            raise ValueError("Picks cannot be empty.")

        self._projector, self.info = setup_proj(self.info)

        #   Set up the CTF compensator
        current_comp = fiff.get_current_comp(self.info)
        if current_comp > 0:
            logger.info('Current compensation grade : %d' % current_comp)

        if keep_comp:
            dest_comp = current_comp

        if current_comp != dest_comp:
            raw['comp'] = fiff.raw.make_compensator(raw.info, current_comp,
                                                    dest_comp)
            logger.info('Appropriate compensator added to change to '
                        'grade %d.' % (dest_comp))

        #    Select the desired events
        self.events = events
        if event_id is not None:
            selected = np.logical_and(events[:, 1] == 0,
                                      events[:, 2] == event_id)
            self.events = self.events[selected]

        n_events = len(self.events)

        if n_events > 0:
            logger.info('%d matching events found' % n_events)
        else:
            raise ValueError('No desired events found.')

        # Handle times
        assert tmin < tmax
        sfreq = float(raw.info['sfreq'])
        n_times_min = int(round(tmin * sfreq))
        n_times_max = int(round(tmax * sfreq))
        times = np.arange(n_times_min, n_times_max + 1, dtype=np.float) / sfreq
        self.times = self._raw_times = times
        self._epoch_stop = ep_len = len(self.times)
        if decim > 1:
            new_sfreq = sfreq / decim
            lowpass = self.info['lowpass']
            if new_sfreq < 2.5 * lowpass:  # nyquist says 2 but 2.5 is safer
                msg = ("The raw file indicates a low-pass frequency of %g Hz. "
                       "The decim=%i parameter will result in a sampling "
                       "frequency of %g Hz, which can cause aliasing "
                       "artifacts." % (lowpass, decim, new_sfreq))
                warnings.warn(msg)

            i_start = n_times_min % decim
            self._decim_idx = slice(i_start, ep_len, decim)
            self.times = self.times[self._decim_idx]
            self.info['sfreq'] = new_sfreq

        # setup epoch rejection
        self._reject_setup()

        if self.preload:
            self._data = self._get_data_from_disk()

    def drop_picks(self, bad_picks):
        """Drop some picks

        Allows to discard some channels.
        """
        self.picks = list(self.picks)
        idx = [k for k, p in enumerate(self.picks) if p not in bad_picks]
        self.picks = [self.picks[k] for k in idx]

        # XXX : could maybe be factorized
        self.info['chs'] = [self.info['chs'][k] for k in idx]
        self.info['ch_names'] = [self.info['ch_names'][k] for k in idx]
        self.info['nchan'] = len(idx)
        self.ch_names = self.info['ch_names']

        if self._projector is not None:
            self._projector = self._projector[idx][:, idx]

        if self.preload:
            self._data = self._data[:, idx, :]

    def drop_bad_epochs(self):
        """Drop bad epochs without retaining the epochs data.

        Should be used before slicing operations.

        .. Warning:: Operation is slow since all epochs have to be read from
            disk. To avoid reading epochs form disk multiple times, initialize
            Epochs object with preload=True.

        """
        self._get_data_from_disk(out=False)

    @verbose
    def drop_epochs(self, indices, verbose=None):
        """Drop epochs based on indices

        Parameters
        ----------
        indices : array of ints
            Set epochs to remove using indices. Events are
            correspondingly modified.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
            Defaults to raw.verbose.
        """
        indices = np.asarray(indices)
        self.events = np.delete(self.events, indices, axis=0)
        if(self.preload):
            self._data = np.delete(self._data, indices, axis=0)
        count = len(indices)
        logger.info('Dropped %d epoch%s' % (count, '' if count == 1 else 's'))

    @verbose
    def _get_epoch_from_disk(self, idx, verbose=None):
        """Load one epoch from disk"""
        sfreq = self.raw.info['sfreq']

        if self.events.ndim == 1:
            # single event
            event_samp = self.events[0]
        else:
            event_samp = self.events[idx, 0]

        # Read a data segment
        first_samp = self.raw.first_samp
        start = int(round(event_samp + self.tmin * sfreq)) - first_samp
        stop = start + self._epoch_stop
        if start < 0:
            return None
        epoch, _ = self.raw[self.picks, start:stop]

        if self.proj and self._projector is not None:
            logger.info("SSP projectors applied...")
            epoch = np.dot(self._projector, epoch)

        # Run baseline correction
        epoch = rescale(epoch, self._raw_times, self.baseline, 'mean',
                        copy=False)

        if self.decim > 1:
            epoch = epoch[:, self._decim_idx]

        return epoch

    @verbose
    def _get_data_from_disk(self, out=True, verbose=None):
        """Load all data from disk

        Parameters
        ----------
        out : bool
            Return the data. Setting this to False is used to reject bad
            epochs without caching all the data, which saves memory.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
            Defaults to self.verbose.
        """
        if self._bad_dropped:
            if not out:
                return
            epochs = map(self._get_epoch_from_disk, xrange(len(self.events)))
        else:
            good_events = []
            epochs = []
            n_events = len(self.events)
            drop_log = [[] for _ in range(n_events)]

            for idx in xrange(n_events):
                epoch = self._get_epoch_from_disk(idx)
                is_good, offenders = self._is_good_epoch(epoch)
                if is_good:
                    good_events.append(idx)
                    if out:
                        epochs.append(epoch)
                else:
                    drop_log[idx] = offenders

            self.drop_log = drop_log
            self.events = np.atleast_2d(self.events[good_events])
            self._bad_dropped = True
            logger.info("%d bad epochs dropped"
                        % (n_events - len(good_events)))
            if not out:
                return

        data = np.array(epochs)
        return data

    @verbose
    def _is_good_epoch(self, data, verbose=None):
        """Determine if epoch is good"""
        if data is None:
            return False, ['NO_DATA']
        n_times = len(self.times)
        if data.shape[1] < n_times:
            # epoch is too short ie at the end of the data
            return False, ['TOO_SHORT']
        if self.reject is None and self.flat is None:
            return True, None
        else:
            return _is_good(data, self.ch_names, self._channel_type_idx,
                            self.reject, self.flat, full_report=True)

    def get_data(self):
        """Get all epochs as a 3D array

        Returns
        -------
        data : array of shape [n_epochs, n_channels, n_times]
            The epochs data
        """
        if self.preload:
            return self._data
        else:
            data = self._get_data_from_disk()
            return data

    def _reject_setup(self):
        """Setup reject process
        """
        if self.reject is None and self.flat is None:
            return

        idx = channel_indices_by_type(self.info)

        for key in idx.keys():
            if (self.reject is not None and key in self.reject) \
                    or (self.flat is not None and key in self.flat):
                if len(idx[key]) == 0:
                    raise ValueError("No %s channel found. Cannot reject based"
                                     " on %s." % (key.upper(), key.upper()))

        self._channel_type_idx = idx

    def __iter__(self):
        """To make iteration over epochs easy.
        """
        self._current = 0
        return self

    def next(self):
        """To make iteration over epochs easy.
        """
        if self.preload:
            if self._current >= len(self._data):
                raise StopIteration
            epoch = self._data[self._current]
            self._current += 1
        else:
            if self._current >= len(self.events):
                raise StopIteration
            epoch = self._get_epoch_from_disk(self._current)
            self._current += 1
            is_good, _ = self._is_good_epoch(epoch)
            if not is_good:
                return self.next()

        return epoch

    def __repr__(self):
        if not self._bad_dropped:
            s = "n_events : %s (good & bad)" % len(self.events)
        else:
            s = "n_events : %s (all good)" % len(self.events)
        s += ", tmin : %s (s)" % self.tmin
        s += ", tmax : %s (s)" % self.tmax
        s += ", baseline : %s" % str(self.baseline)
        return "Epochs (%s)" % s

    def __getitem__(self, key):
        """Return an Epochs object with a subset of epochs
        """
        if not self._bad_dropped:
            warnings.warn("Bad epochs have not been dropped, indexing will be "
                          "inaccurate. Use drop_bad_epochs() or preload=True")

        epochs = cp.copy(self)  # XXX : should use deepcopy but breaks ...
        epochs.events = np.atleast_2d(self.events[key])

        if self.preload:
            if isinstance(key, slice):
                epochs._data = self._data[key]
            else:
                key = np.atleast_1d(key)
                epochs._data = self._data[key]
        return epochs

    def average(self, keep_only_data_channels=True):
        """Compute average of epochs

        Parameters
        ----------
        keep_only_data_channels: bool
            If False, all channels with be kept. Otherwise
            only MEG and EEG channels are kept.

        Returns
        -------
        evoked : Evoked instance
            The averaged epochs
        """
        return self._compute_mean_or_stderr(keep_only_data_channels, 'ave')

    def standard_error(self, keep_only_data_channels=True):
        """Compute standard error over epochs

        Parameters
        ----------
        keep_only_data_channels: bool
            If False, all channels with be kept. Otherwise
            only MEG and EEG channels are kept.

        Returns
        -------
        evoked : Evoked instance
            The standard error over epochs
        """
        return self._compute_mean_or_stderr(keep_only_data_channels, 'stderr')

    def _compute_mean_or_stderr(self, keep_only_data_channels, mode='ave'):
        """Compute the mean or std over epochs and return Evoked"""
        if mode == 'stderr':
            _do_std = True
        else:
            _do_std = False
        evoked = Evoked(None)
        evoked.info = cp.deepcopy(self.info)
        n_channels = len(self.ch_names)
        n_times = len(self.times)
        if self.preload:
            n_events = len(self.events)
            if not _do_std:
                data = np.mean(self._data, axis=0)
            else:
                data = np.std(self._data, axis=0)
            assert len(self.events) == len(self._data)
        else:
            data = np.zeros((n_channels, n_times))
            n_events = 0
            for e in self:
                data += e
                n_events += 1
            data /= n_events
            # convert to stderr if requested, could do in one pass but do in
            # two (slower) in case there are large numbers
            if _do_std:
                data_mean = cp.copy(data)
                data.fill(0.)
                for e in self:
                    data += (e - data_mean) ** 2
                data = np.sqrt(data / n_events)

        evoked.data = data
        evoked.times = self.times.copy()
        evoked.comment = self.name
        evoked.nave = n_events
        evoked.first = -int(np.sum(self.times < 0))
        evoked.last = int(np.sum(self.times > 0))
        if not _do_std:
            evoked.aspect_kind = np.array([FIFF.FIFFV_ASPECT_AVERAGE])
        else:
            evoked.aspect_kind = np.array([FIFF.FIFFV_ASPECT_STD_ERR])
            evoked.data /= np.sqrt(evoked.nave)

        # dropping EOG, ECG and STIM channels. Keeping only data
        if keep_only_data_channels:
            data_picks = pick_types(evoked.info, meg=True, eeg=True,
                                    stim=False, eog=False, ecg=False,
                                    emg=False)
            if len(data_picks) == 0:
                raise ValueError('No data channel found when averaging.')

            evoked.info['chs'] = [evoked.info['chs'][k] for k in data_picks]
            evoked.info['ch_names'] = [evoked.info['ch_names'][k]
                                       for k in data_picks]
            evoked.info['nchan'] = len(data_picks)
            evoked.data = evoked.data[data_picks]
        return evoked

    def crop(self, tmin=None, tmax=None, copy=False):
        """Crops a time interval from epochs object.

        Parameters
        ----------
        tmin : float
            Start time of selection in seconds
        tmax : float
            End time of selection in seconds
        copy : bool
            If False epochs is cropped in place

        Returns
        -------
        epochs : Epochs instance
            The bootstrap samples
        """
        if not self.preload:
            raise RuntimeError('Modifying data of epochs is only supported '
                               'when preloading is used. Use preload=True '
                               'in the constructor.')

        if tmin is None:
            tmin = self.tmin
        elif tmin < self.tmin:
            warnings.warn("tmin is not in epochs' time interval."
                          "tmin is set to epochs.tmin")
            tmin = self.tmin

        if tmax is None:
            tmax = self.tmax
        elif tmax > self.tmax:
            warnings.warn("tmax is not in epochs' time interval."
                          "tmax is set to epochs.tmax")
            tmax = self.tmax

        tmask = (self.times >= tmin) & (self.times <= tmax)
        tidx = np.where(tmask)[0]

        this_epochs = self if not copy else cp.deepcopy(self)
        this_epochs.tmin = this_epochs.times[tidx[0]]
        this_epochs.tmax = this_epochs.times[tidx[-1]]
        this_epochs.times = this_epochs.times[tmask]
        this_epochs._data = this_epochs._data[:, :, tmask]
        return this_epochs

    def resample(self, sfreq, npad=100, window='boxcar'):
        """Resample preloaded data

        Parameters
        ----------
        sfreq: float
            New sample rate to use
        npad : int
            Amount to pad the start and end of the data. If None,
            a (hopefully) sensible choice is used.
        window : string or tuple
            Window to use in resampling. See scipy.signal.resample.
        """
        if self.preload:
            o_sfreq = self.info['sfreq']
            self._data = resample(self._data, sfreq, o_sfreq, npad, 2, window)
            # adjust indirectly affected variables
            self.info['sfreq'] = sfreq
            self.times = (np.arange(self._data.shape[2], dtype=np.float)
                          / sfreq + self.times[0])
        else:
            raise RuntimeError('Can only resample preloaded data')

    def copy(self):
        """ Return copy of Epochs instance
        """
        raw = self.raw.copy()
        new = deepcopy(self)
        new.raw = raw
        return new

    def save(self, fname):
        """Save epochs in a fif file

        Parameters
        ----------
        fname : str
            The name of the file.
        """
        # Create the file and save the essentials
        fid = start_file(fname)

        start_block(fid, FIFF.FIFFB_MEAS)
        write_id(fid, FIFF.FIFF_BLOCK_ID)
        if self.info['meas_id'] is not None:
            write_id(fid, FIFF.FIFF_PARENT_BLOCK_ID, self.info['meas_id'])

        # Write measurement info
        write_meas_info(fid, self.info)

        # One or more evoked data sets
        start_block(fid, FIFF.FIFFB_PROCESSED_DATA)
        start_block(fid, FIFF.FIFFB_EPOCHS)

        start_block(fid, FIFF.FIFFB_MNE_EVENTS)
        write_int(fid, FIFF.FIFF_MNE_EVENT_LIST, self.events.T)
        end_block(fid, FIFF.FIFFB_MNE_EVENTS)

        # First and last sample
        first = -int(np.sum(self.times < 0))
        last = int(np.sum(self.times > 0))
        write_int(fid, FIFF.FIFF_FIRST_SAMPLE, first)
        write_int(fid, FIFF.FIFF_LAST_SAMPLE, last)

        # save baseline
        if self.baseline is not None:
            bmin, bmax = self.baseline
            bmin = self.times[0] if bmin is None else bmin
            bmax = self.times[-1] if bmax is None else bmax
            write_float(fid, FIFF.FIFF_MNE_BASELINE_MIN, bmin)
            write_float(fid, FIFF.FIFF_MNE_BASELINE_MAX, bmax)

        # The epochs itself
        decal = np.empty(self.info['nchan'])
        for k in range(self.info['nchan']):
            decal[k] = 1.0 / self.info['chs'][k]['cal']

        data = self.get_data()
        data *= decal[None, :, None]

        write_float_matrix(fid, FIFF.FIFF_EPOCH, data)
        end_block(fid, FIFF.FIFFB_EPOCHS)

        end_block(fid, FIFF.FIFFB_PROCESSED_DATA)
        end_block(fid, FIFF.FIFFB_MEAS)
        end_file(fid)


    def as_data_frame(self, frame=True):
        """Get the epochs as Pandas panel of data frames

        Parameters
        ----------
        frame : boolean
            If frame, data frame will be returned with a hierarchical
            epochs * time-slices index, else a panel object of
            channels * time-slices data frames for each epoch.

        Returns
        -------
        out : depending on arguments
            data frame object or panel object

        """
        import pandas as pd
        data = self.get_data()
        epoch_ids = ["epoch %i" % (i + 1) for i in np.arange(data.shape[0])]

        out = pd.Panel(data=data, items=epoch_ids, major_axis=self.ch_names)
        if frame:
            out = out.swapaxes(0, 1).to_frame()
            out.index.names = ["epochs", "time slices"]
        else:
            out.swapaxes(1, 2)

        return out

    def to_nitime(self, picks=None, epochs_idx=None, collapse=False,
                  copy=True, use_first_samp=False):
        """ Export epochs as nitime TimeSeries

        Parameters
        ----------
        picks : array-like | None
            Indices for exporting subsets of the epochs channels. If None
            all good channels will be used.
        epochs_idx : slice | array-like | None
            Epochs index for single or selective epochs exports. If None, all
            epochs will be used.
        collapse : boolean
            If True export epochs and time slices will be collapsed to 2D
            array. This may be required by some nitime functions.
        copy : boolean
            If True exports copy of epochs data.
        use_first_samp: boolean
            If True, the time returned is relative to the session onset, else
            relative to the recording onset.

        Returns
        -------
        epochs_ts : instance of nitime.TimeSeries
            The Epochs as nitime TimeSeries object
        """
        try:
            from nitime import TimeSeries  # to avoid strong dependency
        except ImportError:
            raise Exception('the nitime package is missing')

        if picks is None:
            picks = pick_types(self.info, include=self.ch_names,
                               exclude=self.info['bads'])
        if epochs_idx is None:
            epochs_idx = slice(len(self.events))

        data = self.get_data()[epochs_idx, picks]

        if copy is True:
            data = data.copy()

        if collapse is True:
            data = np.hstack(data).copy()

        offset = self.raw.time_as_index(abs(self.tmin), use_first_samp)
        t0 = self.raw.index_as_time(self.events[0, 0] - offset)[0]
        epochs_ts = TimeSeries(data, sampling_rate=self.info['sfreq'], t0=t0)
        epochs_ts.ch_names = np.array(self.ch_names)[picks].tolist()

        return epochs_ts


def equalize_epoch_counts(*args):
    """Equalize the number of trials in multiple Epoch instances

    It tries to make the remaining epochs occuring as close as possible in
    time. This method works based on the idea that if there happened to be some
    time-varying (like on the scale of minutes) noise characteristics during
    a recording, they could be compensated for (to some extent) in the
    equalization process. This method thus seeks to reduce any of those effects
    by minimizing the differences in the times of the events in the two sets of
    epochs. For example, if one had event times [1, 2, 3, 4, 120, 121] and the
    other one had [3.5, 4.5, 120.5, 121.5], it would remove events at times
    [1, 2] in the first epochs and not [20, 21].

    Note that this operates on the Epochs instances in-place.

    It chooses the epochs to eliminate by minimizing the differences in timing
    between Epochs instances with more trials and the Epochs instance with the
    fewest trials. This function will also call drop_bad_epochs() on any epochs
    instance that hasn't yet had bad epochs dropped.

    Example:

    equalize_epoch_counts(epochs1, epochs2)

    Parameters
    ----------
    e1, e2, ... : sequence of Epochs instances
        The Epochs instances to equalize trial counts for.
    method : str
        If 'truncate', events will be truncated from the end of each event
        list. If 'mintime', timing differences between each event list will be
        minimized.
    """
    epochs_list = args
    if not all([isinstance(e, Epochs) for e in epochs_list]):
        raise ValueError('All inputs must be Epochs instances')

    # make sure bad epochs are dropped
    [e.drop_bad_epochs() if not e._bad_dropped else None for e in epochs_list]

    small_idx = np.argmin([e.events.shape[0] for e in epochs_list])
    small_e_times = epochs_list[small_idx].events[:, 0]
    for e in epochs_list:
        mask = _minimize_time_diff(small_e_times, e.events[:, 0])
        indices = np.where(np.logical_not(mask))[0]
        e.drop_epochs(indices)


def _minimize_time_diff(t_shorter, t_longer):
    """Find a boolean mask to minimize timing differences"""
    keep = np.ones((len(t_longer)), dtype=bool)
    scores = np.ones((len(t_longer)))
    for iter in range(len(t_longer) - len(t_shorter)):
        scores.fill(np.inf)
        # Check every possible removal to see if it minimizes
        for idx in np.where(keep)[0]:
            keep[idx] = False
            scores[idx] = _area_between_times(t_shorter, t_longer[keep])
            keep[idx] = True
        keep[np.argmin(scores)] = False
    return keep


def _area_between_times(t1, t2):
    """Quantify the difference between two timing sets"""
    x1 = range(len(t1))
    x2 = range(len(t2))
    xs = np.concatenate((x1, x2))
    return np.sum(np.abs(np.interp(xs, x1, t1) - np.interp(xs, x2, t2)))


@verbose
def _is_good(e, ch_names, channel_type_idx, reject, flat, full_report=False,
             verbose=None):
    """Test if data segment e is good according to the criteria
    defined in reject and flat. If full_report=True, it will give
    True/False as well as a list of all offending channels.
    """
    bad_list = list()
    has_printed = False
    if reject is not None:
        for key, thresh in reject.iteritems():
            idx = channel_type_idx[key]
            name = key.upper()
            if len(idx) > 0:
                e_idx = e[idx]
                deltas = np.max(e_idx, axis=1) - np.min(e_idx, axis=1)
                idx_deltas = np.where(deltas > thresh)[0]

                if len(idx_deltas) > 0:
                    ch_name = [ch_names[idx[i]] for i in idx_deltas]
                    if (not has_printed):
                        logger.info('    Rejecting epoch based on %s : %s'
                                    % (name, ch_name))
                        has_printed = True
                    if not full_report:
                        return False
                    else:
                        bad_list.extend(ch_name)

    if flat is not None:
        for key, thresh in flat.iteritems():
            idx = channel_type_idx[key]
            name = key.upper()
            if len(idx) > 0:
                e_idx = e[idx]
                deltas = np.max(e_idx, axis=1) - np.min(e_idx, axis=1)
                idx_min_delta = np.argmin(deltas)
                delta = deltas[idx_min_delta]
                if delta < thresh:
                    ch_name = ch_names[idx[idx_min_delta]]
                    if (not has_printed):
                        logger.info('    Rejecting flat epoch based on %s : '
                                    '%s (%s < %s).' % (name, ch_name, delta,
                                                       thresh))
                        has_printed = True
                    if not full_report:
                        return False
                    else:
                        bad_list.append(ch_name)

    if not full_report:
        return True
    else:
        if bad_list == []:
            return True, None
        else:
            return False, bad_list


@verbose
def read_epochs(fname, proj=True, verbose=None):
    """Read epochs from a fif file

    Parameters
    ----------
    fname : str
        The name of the file.
    proj : bool, optional
        Apply SSP projection vectors.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
        Defaults to raw.verbose.

    Returns
    -------
    epochs : instance of Epochs
        The epochs
    """
    epochs = Epochs(None, None, None, None, None)

    logger.info('Reading %s ...' % fname)
    fid, tree, _ = fiff_open(fname)

    #   Read the measurement info
    info, meas = read_meas_info(fid, tree)
    info['filename'] = fname

    events = _read_events_fif(fid, tree)

    #   Locate the data of interest
    processed = dir_tree_find(meas, FIFF.FIFFB_PROCESSED_DATA)
    if len(processed) == 0:
        fid.close()
        raise ValueError('Could not find processed data')

    epochs_node = dir_tree_find(tree, FIFF.FIFFB_EPOCHS)
    if len(epochs_node) == 0:
        fid.close()
        raise ValueError('Could not find epochs data')

    my_epochs = epochs_node[0]

    # Now find the data in the block
    comment = None
    data = None
    bmin, bmax = None, None

    for k in range(my_epochs['nent']):
        kind = my_epochs['directory'][k].kind
        pos = my_epochs['directory'][k].pos
        if kind == FIFF.FIFF_FIRST_SAMPLE:
            tag = read_tag(fid, pos)
            first = int(tag.data)
        elif kind == FIFF.FIFF_LAST_SAMPLE:
            tag = read_tag(fid, pos)
            last = int(tag.data)
        elif kind == FIFF.FIFF_COMMENT:
            tag = read_tag(fid, pos)
            comment = tag.data
        elif kind == FIFF.FIFF_EPOCH:
            tag = read_tag(fid, pos)
            data = tag.data
        elif kind == FIFF.FIFF_MNE_BASELINE_MIN:
            tag = read_tag(fid, pos)
            bmin = float(tag.data)
        elif kind == FIFF.FIFF_MNE_BASELINE_MAX:
            tag = read_tag(fid, pos)
            bmax = float(tag.data)

    if bmin is not None or bmax is not None:
        baseline = (bmin, bmax)

    nsamp = last - first + 1
    logger.info('    Found the data of interest:')
    logger.info('        t = %10.2f ... %10.2f ms (%s)'
                % (1000 * first / info['sfreq'],
                   1000 * last / info['sfreq'], comment))
    if info['comps'] is not None:
        logger.info('        %d CTF compensation matrices available'
                                               % len(info['comps']))

    # Read the data
    if data is None:
        raise ValueError('Epochs data not found')

    if data.shape[2] != nsamp:
        fid.close()
        raise ValueError('Incorrect number of samples (%d instead of %d)'
                          % (data.shape[2], nsamp))

    # Calibrate
    cals = np.array([info['chs'][k]['cal'] for k in range(info['nchan'])])
    data = cals[None, :, None] * data

    times = np.arange(first, last + 1, dtype=np.float) / info['sfreq']
    tmin = times[0]
    tmax = times[-1]

    # Put it all together
    epochs.preload = True
    epochs._bad_dropped = True
    epochs.events = events
    epochs._data = data
    epochs.info = info
    epochs.tmin = tmin
    epochs.tmax = tmax
    epochs.name = comment
    epochs.times = times
    epochs.data = data
    epochs.proj = proj
    epochs._projector, epochs.info = setup_proj(info)
    epochs.ch_names = info['ch_names']
    epochs.baseline = baseline
    epochs.event_id = int(np.unique(events[:, 2]))
    fid.close()

    return epochs


def bootstrap(epochs, random_state=None):
    """Compute epochs selected by bootstrapping

    Parameters
    ----------
    epochs : Epochs instance
        epochs data to be bootstrapped
    random_state : None | int | np.random.RandomState
        To specify the random generator state

    Returns
    -------
    epochs : Epochs instance
        The bootstrap samples
    """
    if not epochs.preload:
        raise RuntimeError('Modifying data of epochs is only supported '
                           'when preloading is used. Use preload=True '
                           'in the constructor.')

    rng = check_random_state(random_state)
    epochs_bootstrap = cp.deepcopy(epochs)
    n_events = len(epochs_bootstrap.events)
    idx = rng.randint(0, n_events, n_events)
    epochs_bootstrap = epochs_bootstrap[idx]
    return epochs_bootstrap
