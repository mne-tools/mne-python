# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Daniel Strohmeier <daniel.strohmeier@tu-ilmenau.de>
#
# License: BSD (3-clause)

import copy as cp
import warnings

import numpy as np

import fiff
from .fiff import Evoked
from .fiff.pick import pick_types, channel_indices_by_type
from .fiff.proj import activate_proj, make_eeg_average_ref_proj
from .baseline import rescale
from .utils import check_random_state


class Epochs(object):
    """List of Epochs

    Parameters
    ----------
    raw : Raw object, or list
        A instance of Raw, or a list of Raw instances

    events : array, of shape [n_events, 3], or list
        Returned by the read_events function, or a list of the same
        length as the list of Raw instances

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

    verbose : None | bool
        Use verbose output. None defaults to raw.verbose.


    Attributes
    ----------
    info: dict
        Measurement info, with 'projs' stored as a list
        in 'projs_all' since they can vary across runs

    ch_names: list of string
        List of channels' names

    events: array of int (n x 3)
        Array of the events being used to parse epochs.
        NOTE: Be wary of changing this directly (as opposed
        to making a new epochs object with different events)
        because it will likely cause indexing errors.

    drop_log: list of lists
        This list (same length as events) contains the channel(s),
        if any, that caused an event in the original event list
        to be dropped by drop_bad_epochs().

    Methods
    -------
    get_data() : self
        Return all epochs as a 3D array [n_epochs x n_channels x n_times].

    average() : self
        Return Evoked object containing averaged epochs as a
        2D array [n_channels x n_times].

    drop_bad_epochs() : None
        Drop all epochs marked as bad. Should be used before indexing and
        slicing operations, and is done automatically by preload=True.

    Notes
    -----
    For indexing and slicing:

    epochs = Epochs(...)

    epochs[idx] : Epochs
        Return Epochs object with a subset of epochs (supports single
        index and python style slicing)
    """

    def __init__(self, raw, events, event_id, tmin, tmax, baseline=(None, 0),
                picks=None, name='Unknown', keep_comp=False, dest_comp=0,
                preload=False, reject=None, flat=None, proj=True, verbose=None):
        if not isinstance(raw, list):
            raw = [raw]
        self.raw = raw
        self.verbose = raw[0].verbose if verbose is None else verbose
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
        self._bad_dropped = False
        self.drop_log = None

        # Handle measurement info
        self.info = cp.deepcopy(raw[0].info)

        # setup epoch rejection
        self._reject_setup()

        # check to make sure everything necessary matches before changing info
        check_raw_compatibility(raw)

        # now modify (if necessary) based on picks
        if picks is not None:
            self.info['chs'] = [self.info['chs'][k] for k in picks]
            self.info['ch_names'] = [self.info['ch_names'][k] for k in picks]
            self.info['nchan'] = len(picks)
        else:
            picks = range(len(self.info['ch_names']))
        self.ch_names = self.info['ch_names']

        if len(picks) == 0:
            raise ValueError("Picks cannot be empty.")

        self.picks = picks

        #   Set up projection
        self.info['projs_all'] = [cp.deepcopy(raw[ri].info['projs']) for ri in range(len(raw))]
        self.proj = [None]*len(raw)
        del self.info['projs'] # Delete old, now potentially ambiguous key
        for ri in range(len(raw)):
            if self.info['projs_all'][ri] is None or not proj:
                print 'No projector specified for these data'
                self.proj[ri] = None
            else:
                #   Add EEG ref reference proj
                eeg_sel = pick_types(self.info, meg=False, eeg=True)
                if len(eeg_sel) > 0:
                    eeg_proj = make_eeg_average_ref_proj(self.info)
                    self.info['projs_all'][ri].append(eeg_proj)

                #   Create the projector, temporarily adding 'projs'
                r_proj, nproj, _ = fiff.proj.make_projector(
                                    self.info['projs_all'][ri],
                                    self.info['ch_names'], self.info['bads'])
                if nproj == 0:
                    print 'The projection vectors do not apply to these channels'
                    self.proj[ri] = None
                else:
                    print ('Created an SSP operator (subspace dimension = %d)'
                                                                        % nproj)
                    self.proj[ri] = r_proj

                #   The projection items have been activated
                self.info['projs_all'][ri] = activate_proj(self.info['projs_all'][ri], copy=False)

        #   Set up the CTF compensator
        current_comp = [fiff.get_current_comp(self.info) \
                            for ri in range(len(raw))]
        for ri in range(len(raw)):
            if current_comp[ri] > 0:
                print 'Current compensation grade : %d' % current_comp

        if keep_comp:
            dest_comp = current_comp

        for ri in range(len(raw)):
            if current_comp[ri] != dest_comp:
                raw[ri]['comp'] = fiff.raw.make_compensator(raw[ri].info,
                                                 current_comp, dest_comp)
                print 'Appropriate compensator added to change to grade %d.' % (
                                                                    dest_comp)

        # concatenate events from all epochs
        self._replace_events(events)
        events = self.events

        # select the desired events before preloading
        if event_id is not None:
            selected = np.logical_and(events[:, 1] == 0,
                                      events[:, 2] == event_id)
            self._pick_event_subset(selected)

        n_events = len(self.events)

        if n_events > 0:
            print '%d matching events found' % n_events
        else:
            raise ValueError('No desired events found.')

        # Handle times
        assert tmin < tmax
        sfreq = self.info['sfreq']
        n_times_min = int(round(tmin * float(sfreq)))
        n_times_max = int(round(tmax * float(sfreq)))
        self.times = np.arange(n_times_min, n_times_max + 1,
                               dtype=np.float) / sfreq

        # setup epoch rejection
        self._reject_setup()

        if self.preload:
            self._data = self._get_data_from_disk()

    def _replace_events(self, new_events):
        """Replace current events with new set of events.
        """
        if type(new_events) is not list:
            new_events = [new_events]
        if len(new_events) is not len(self.raw):
            raise ValueError('new_events must be a list of the same length as '
                             'the raw list (or not a list if raw is not a list)')
        idx_limits = np.zeros(len(new_events) + 1, dtype='uint64') # First limit is zero
        events = np.zeros((0,3), dtype='uint64')
        for ei in range(len(new_events)):
            events = np.vstack((events, cp.deepcopy(new_events[ei])))
            idx_limits[ei + 1] = events.shape[0]
        self.events = events
        self.idx_limits = idx_limits

    def _idx_to_raw_sidx(self, idx):
        """Function to convert input index (idx) to raw # (ri)
        and (sub)index (sidx) within that raw
        """
        ri = np.nonzero(self.idx_limits <= idx)[0][-1]
        sidx = idx - self.idx_limits[ri]
        return ri, sidx

    def _pick_event_subset(self, key):
        """Choose a subset of events to keep from the originals
        """
        # Update the idx_limits and events simultaneously
        key = np.array(range(len(self.events)), dtype='uint64')[key]
        for ri in range(len(self.raw)):
            inds = np.logical_and(np.less_equal(self.idx_limits[ri], key), \
                                  np.less(key, self.idx_limits[ri+1]))
            self.idx_limits[ri + 1] = np.sum(inds) + self.idx_limits[ri]
        self.events = np.atleast_2d(self.events[key])

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

        for ri in range(len(self.raw)):
            if self.proj[ri] is not None:
                self.proj[ri] = self.proj[ri][idx][:, idx]

        if self.preload:
            self._data = self._data[:, idx, :]

    def drop_bad_epochs(self):
        """Drop bad epochs.

        Should be used before slicing operations.

        Warning: Operation is slow since all epochs have to be read from disk
        """
        if self._bad_dropped:
            return

        good_events = []
        n_events = len(self.events)
        drop_log = [[]]*n_events
        for idx in range(n_events):
            epoch = self._get_epoch_from_disk(idx)
            is_good, offenders = self._is_good_epoch(epoch)
            if is_good:
                good_events.append(idx)
            else:
                drop_log[idx] = offenders

        self.drop_log = drop_log
        self._pick_event_subset(good_events)
        self._bad_dropped = True

        print "%d bad epochs dropped" % (n_events - len(good_events))

    def _get_epoch_from_disk(self, idx):
        """Load one epoch from disk"""
        ri, sidx = self._idx_to_raw_sidx(idx)
        sfreq = self.raw[ri].info['sfreq']

        if self.events.ndim == 1:
            #single event
            event_samp = self.events[0]
        else:
            event_samp = self.events[idx, 0]

        # Read a data segment
        first_samp = self.raw[ri].first_samp
        start = int(round(event_samp + self.tmin * sfreq)) - first_samp
        stop = start + len(self.times)
        if start < 0:
            return None
        epoch, _ = self.raw[ri][self.picks, start:stop]

        if self.proj[ri] is not None:
            print "SSP projectors applied..."
            epoch = np.dot(self.proj[ri], epoch)

        # Run baseline correction
        epoch = rescale(epoch, self.times, self.baseline, 'mean',
                        verbose=self.verbose, copy=False)
        return epoch

    def _get_data_from_disk(self):
        """Load all data from disk"""
        # drop bad epochs first so we don't have to replicate checking here
        self.drop_bad_epochs()
        # note that events is automatically updated by drop_bad_epochs, too
        data = [self._get_epoch_from_disk(k) for k in range(len(self.events))]
        return np.array(data)

    def _is_good_epoch(self, data):
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
            raise RuntimeError("Bad epochs have not been dropped, indexing "
                               "will be inaccurate. Use drop_bad_epochs() or "
                               "preload=True")

        epochs = cp.copy(self)  # XXX : should use deepcopy but breaks ...
        # Ensure these actually refer to different arrays
        epochs.events = cp.deepcopy(self.events)
        epochs.idx_limits = cp.deepcopy(self.idx_limits)
        epochs._pick_event_subset(key)

        if self.preload:
            if isinstance(key, slice):
                epochs._data = cp.deepcopy(self._data[key])
            else:
                key = np.atleast_1d(key)
                epochs._data = cp.deepcopy(self._data[key])
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
        evoked = Evoked(None)
        evoked.info = cp.deepcopy(self.info)
        n_channels = len(self.ch_names)
        n_times = len(self.times)
        if self.preload:
            n_events = len(self.events)
            data = np.mean(self._data, axis=0)
            assert len(self.events) == len(self._data)
        else:
            data = np.zeros((n_channels, n_times))
            n_events = 0
            for e in self:
                data += e
                n_events += 1
            data /= n_events
        evoked.data = data
        evoked.times = self.times.copy()
        evoked.comment = self.name
        evoked.aspect_kind = np.array([100])  # for standard average file
        evoked.nave = n_events
        evoked.first = - int(np.sum(self.times < 0))
        evoked.last = int(np.sum(self.times > 0))

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


def check_raw_compatibility(raw):
    """Check to make sure all instances of Raw
    in the input list raw have compatible parameters"""
    for ri in range(1,len(raw)):
        if not raw[ri].info['nchan'] == raw[0].info['nchan']:
            raise ValueError('raw[%d][\'info\'][\'nchan\'] must match' % ri)
        if not raw[ri].info['bads'] == raw[0].info['bads']:
            raise ValueError('raw[%d][\'info\'][\'bads\'] must match' % ri)
        if not raw[ri].info['sfreq'] == raw[0].info['sfreq']:
            raise ValueError('ra[%d][\'info\'][\'sfreq\'] must match' % ri)
        if not set(raw[ri].info['ch_names']) \
                   == set(raw[0].info['ch_names']):
            raise ValueError('raw[%d][\'info\'][\'ch_names\'] must match' % ri)


def _is_good(e, ch_names, channel_type_idx, reject, flat, full_report=False):
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
                idx_max_delta = np.argmax(deltas)
                delta = deltas[idx_max_delta]
                if delta > thresh:
                    ch_name = ch_names[idx[idx_max_delta]]
                    if not has_printed:
                        print '    Rejecting epoch based on %s : %s (%s > %s).' \
                                    % (name, ch_name, delta, thresh)
                        has_printed = True
                    if not full_report:
                        return False
                    else:
                        bad_list.append(ch_name)

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
                    if not has_printed:
                        print ('    Rejecting flat epoch based on '
                               '%s : %s (%s < %s).' % (name, ch_name, delta,
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
