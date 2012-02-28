# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import copy
import numpy as np
import fiff
import warnings
from .fiff import Evoked
from .fiff.pick import pick_types, channel_indices_by_type
from .baseline import rescale


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

    Attributes
    ----------
    info: dict
        Measurement info

    ch_names: list of string
        List of channels' names

    Methods
    -------
    get_data() : self
        Return all epochs as a 3D array [n_epochs x n_channels x n_times].

    average() : self
        Return Evoked object containing averaged epochs as a
        2D array [n_channels x n_times].

    drop_bad_epochs() : None
        Drop all epochs marked as bad. Should be used before indexing and
        slicing operations.

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
                preload=False, reject=None, flat=None, proj=True):
        self.raw = raw
        self.event_id = event_id
        self.tmin = tmin
        self.tmax = tmax
        self.picks = picks
        self.name = name
        self.keep_comp = keep_comp
        self.dest_comp = dest_comp
        self.baseline = baseline
        self.preload = preload
        self.reject = reject
        self.flat = flat
        self._bad_dropped = False

        # Handle measurement info
        self.info = copy.deepcopy(raw.info)
        if picks is not None:
            self.info['chs'] = [self.info['chs'][k] for k in picks]
            self.info['ch_names'] = [self.info['ch_names'][k] for k in picks]
            self.info['nchan'] = len(picks)

        if picks is None:
            picks = range(len(raw.info['ch_names']))
            self.ch_names = raw.info['ch_names']
        else:
            self.ch_names = [raw.info['ch_names'][k] for k in picks]

        if len(picks) == 0:
            raise ValueError("Picks cannot be empty.")

        #   Set up projection
        if self.info['projs'] is None or not proj:
            print 'No projector specified for these data'
            self.proj = None
        else:
            #   Activate the projection items
            for proj in self.info['projs']:
                proj['active'] = True

            print '%d projection items activated' % len(self.info['projs'])

            # Add EEG ref reference proj
            print "Adding average EEG reference projection."
            eeg_sel = pick_types(self.info, meg=False, eeg=True)
            eeg_names = [self.ch_names[k] for k in eeg_sel]
            n_eeg = len(eeg_sel)
            if n_eeg > 0:
                vec = np.ones((1, n_eeg)) / n_eeg
                eeg_proj_data = dict(col_names=eeg_names, row_names=None,
                                     data=vec, nrow=1, ncol=n_eeg)
                eeg_proj = dict(active=True, data=eeg_proj_data,
                                desc='Average EEG reference', kind=1)
                self.info['projs'].append(eeg_proj)

            #   Create the projector
            proj, nproj = fiff.proj.make_projector_info(self.info)
            if nproj == 0:
                print 'The projection vectors do not apply to these channels'
                self.proj = None
            else:
                print ('Created an SSP operator (subspace dimension = %d)'
                                                                    % nproj)
                self.proj = proj

        #   Set up the CTF compensator
        current_comp = fiff.get_current_comp(self.info)
        if current_comp > 0:
            print 'Current compensation grade : %d' % current_comp

        if keep_comp:
            dest_comp = current_comp

        if current_comp != dest_comp:
            raw['comp'] = fiff.raw.make_compensator(raw.info, current_comp,
                                                 dest_comp)
            print 'Appropriate compensator added to change to grade %d.' % (
                                                                    dest_comp)

        #    Select the desired events
        self.events = events
        if event_id is not None:
            selected = np.logical_and(events[:, 1] == 0, events[:, 2] == event_id)
            self.events = self.events[selected]

        n_events = len(self.events)

        if n_events > 0:
            print '%d matching events found' % n_events
        else:
            raise ValueError('No desired events found.')

        # Handle times
        assert tmin < tmax
        sfreq = raw.info['sfreq']
        n_times_min = int(round(tmin * float(sfreq)))
        n_times_max = int(round(tmax * float(sfreq)))
        self.times = np.arange(n_times_min, n_times_max + 1,
                               dtype=np.float) / sfreq

        # setup epoch rejection
        self._reject_setup()

        if self.preload:
            self._data, good_events = self._get_data_from_disk()
            self.events = np.atleast_2d(self.events[good_events, :])
            self._bad_dropped = True

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

        if self.proj is not None:
            self.proj = self.proj[idx][:, idx]

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
        for idx in range(n_events):
            epoch = self._get_epoch_from_disk(idx)
            if self._is_good_epoch(epoch):
                good_events.append(idx)

        self.events = np.atleast_2d(self.events[good_events])
        self._bad_dropped = True

        print "%d bad epochs dropped" % (n_events - len(good_events))

    def _get_epoch_from_disk(self, idx):
        """Load one epoch from disk"""
        sfreq = self.raw.info['sfreq']

        if self.events.ndim == 1:
            #single event
            event_samp = self.events[0]
        else:
            event_samp = self.events[idx, 0]

        # Read a data segment
        first_samp = self.raw.first_samp
        start = int(round(event_samp + self.tmin * sfreq)) - first_samp
        stop = start + len(self.times)
        if start < 0:
            return None
        epoch, _ = self.raw[self.picks, start:stop]

        if self.proj is not None:
            print "SSP projectors applied..."
            epoch = np.dot(self.proj, epoch)

        # Run baseline correction
        epoch = rescale(epoch, self.times, self.baseline, 'mean', verbose=True,
                        copy=False)
        return epoch

    def _get_data_from_disk(self):
        """Load all data from disk
        """
        n_events = len(self.events)
        data = list()
        n_reject = 0
        event_idx = list()
        for k in range(n_events):
            epoch = self._get_epoch_from_disk(k)
            if self._is_good_epoch(epoch):
                data.append(epoch)
                event_idx.append(k)
            else:
                n_reject += 1
        print "Rejecting %d epochs." % n_reject
        return np.array(data), event_idx

    def _is_good_epoch(self, data):
        """Determine if epoch is good
        """
        if data is None:
            return False
        n_times = len(self.times)
        if self.reject is None and self.flat is None:
            return True
        elif data.shape[1] < n_times:
            return False  # epoch is too short ie at the end of the data
        else:
            return _is_good(data, self.ch_names, self._channel_type_idx,
                            self.reject, self.flat)

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
            data, _ = self._get_data_from_disk()
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
            if not self._is_good_epoch(epoch):
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

        epochs = copy.copy(self)  # XXX : should use deepcopy but breaks ...
        epochs.events = np.atleast_2d(self.events[key])

        if self.preload:
            if isinstance(key, slice):
                epochs._data = self._data[key]
            else:
                #make sure data remains a 3D array
                #Note: np.atleast_3d() doesn't do what we want
                epochs._data = np.array([self._data[key]])

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
        evoked.info = copy.deepcopy(self.info)
        n_channels = len(self.ch_names)
        n_times = len(self.times)
        n_events = len(self.events)
        if self.preload:
            data = np.mean(self._data, axis=0)
        else:
            data = np.zeros((n_channels, n_times))
            for e in self:
                data += e
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


def _is_good(e, ch_names, channel_type_idx, reject, flat):
    """Test if data segment e is good according to the criteria
    defined in reject and flat.
    """
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
                    print '    Rejecting epoch based on %s : %s (%s > %s).' \
                                % (name, ch_name, delta, thresh)
                    return False
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
                    print ('    Rejecting flat epoch based on %s : %s (%s < %s).'
                                % (name, ch_name, delta, thresh))
                    return False

    return True
