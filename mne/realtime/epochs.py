# Authors: Christoph Dinh <chdinh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)
import warnings
import time

import numpy as np

import logging
logger = logging.getLogger('mne')

from .. import Epochs, verbose, fiff
from ..baseline import rescale
from ..event import _find_events
from ..filter import detrend
from ..fiff.proj import setup_proj


class RtEpochs(Epochs):
    """Realtime Epochs

    Can receive epochs in real time from an RtClient.

    Parameters
    ----------
    client : instance of mne.realtime.RtClient
        The realtime client.
    event_id : int
        The id of the event to consider.
    tmin : float
        Start time before event.
    tmax : float
        End time after event.
    n_epochs : int
        Number of epochs to return before iteration over epochs stops.
    stim_channel : string or list of string
        Name of the stim channel or all the stim channels affected by
        the trigger.
    name : string
        Comment that describes the Evoked data created.
    keep_comp : boolean
        Apply CTF gradient compensation.
    baseline : None (default) or tuple of length 2
        The time interval to apply baseline correction.
        If None do not apply it. If baseline is (a, b)
        the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used
        and if b is None then b is set to the end of the interval.
        If baseline is equal to (None, None) all the time
        interval is used.
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
    reject_tmin : scalar | None
        Start of the time window used to reject epochs (with the default None,
        the window will start with tmin).
    reject_tmax : scalar | None
        End of the time window used to reject epochs (with the default None,
        the window will end with tmax).
    detrend : int | None
        If 0 or 1, the data channels (MEG and EEG) will be detrended when
        loaded. 0 is a constant (DC) detrend, 1 is a linear detrend. None
        is no detrending. Note that detrending is performed before baseline
        correction. If no DC offset is preferred (zeroth order detrending),
        either turn off baseline correction, as this may introduce a DC
        shift, or set baseline correction to use the entire time interval
        (will yield equivalent results but be slower).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
        Defaults to client.verbose.
    """
    @verbose
    def __init__(self, client, event_id, tmin, tmax, n_epochs,
                 stim_channel='STI 014', baseline=(None, 0), picks=None,
                 name='Unknown', keep_comp=False, dest_comp=0, reject=None,
                 flat=None, proj=True, decim=1, reject_tmin=None,
                 reject_tmax=None, detrend=None, verbose=None):

        self.client = client

        # get the measurement info
        self.info = client.get_measurement_info()

        # Realtime epochs cannot be preloaded
        self.preload = False
        self._data = None

        self.n_epochs = n_epochs

        if not isinstance(stim_channel, list):
            stim_channel = [stim_channel]

        stim_picks = fiff.pick_channels(self.info['ch_names'],
                                        include=stim_channel, exclude=[])

        if len(stim_picks) == 0:
            raise ValueError('No stim channel found to extract event triggers.')
        self._stim_picks = stim_picks

        self.name = name
        self.event_id = event_id
        self.tmin = tmin
        self.tmax = tmax
        self.keep_comp = keep_comp
        self.dest_comp = dest_comp
        self.baseline = baseline
        self.reject = reject
        self.reject_tmin = reject_tmin
        self.reject_tmax = reject_tmax
        self.flat = flat
        self.proj = proj
        self.decim = decim = int(decim)
        self._bad_dropped = False
        self.drop_log = None
        self.detrend = detrend
        self.verbose = client.verbose if verbose is None else verbose

        # handle picks
        if picks is None:
            picks = range(len(self.info['ch_names']))
            self.ch_names = self.info['ch_names']
        else:
            self.info['chs'] = [self.info['chs'][k] for k in picks]
            self.info['ch_names'] = [self.info['ch_names'][k] for k in picks]
            self.ch_names = self.info['ch_names']
            self.info['nchan'] = len(picks)
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
            self.comp = fiff.raw.make_compensator(self.info, current_comp,
                                                  dest_comp)
            logger.info('Appropriate compensator added to change to '
                        'grade %d.' % (dest_comp))
        else:
            self.comp = current_comp

        # Handle times
        assert tmin < tmax
        sfreq = float(self.info['sfreq'])
        n_times_min = int(round(tmin * sfreq))
        n_times_max = int(round(tmax * sfreq))
        times = np.arange(n_times_min, n_times_max + 1, dtype=np.float) / sfreq
        self.times = self._raw_times = times
        self._epoch_stop = ep_len = len(self.times)
        if decim > 1:
            new_sfreq = sfreq / decim
            lowpass = self.info['lowpass']
            if new_sfreq < 2.5 * lowpass:  # nyquist says 2 but 2.5 is safer
                msg = ("The client indicates a low-pass frequency of %g Hz. "
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

        # add callibration factors
        cals = np.zeros(self.info['nchan'])
        for k in range(self.info['nchan']):
            cals[k] = (self.info['chs'][k]['range']
                       * self.info['chs'][k]['cal'])
        self.cals = cals[:, None]

        # FIFO buffer for received epochs
        self._epoch_queue = list()

        # variables needed for receiving raw buffers
        self._last_buffer = None
        self._first_samp = 0
        self._event_samp_backlog = np.empty(0, dtype=np.int)

        # Number of good and bad epochs received
        self._n_good = 0
        self._n_bad = 0

        self._started = False

    def start(self):
        """Start receiving epochs

        The measurement will be started if it has not already been started.
        """
        if not self._started:
            # register the callback
            self.client.register_receive_callback(self._process_raw_buffer)

            # start the measurement and the receive thread
            self.client.start_receive_thread(self.info['nchan'])

            self._started = True

    def stop(self, stop_receive_thread=False, stop_measurement=False):
        """Stop receiving epochs

        Parameters
        ----------
        stop_receive_thread : bool
            Stop the receive thread. Note: Other RtEpochs instances will also
            stop receiving epochs when the receive thread is stopped. The
            receive thread will always be stopped if stop_measurement is True.

        stop_measurement : bool
            Also stop the measurement. Note: Other clients attached to the
            server will also stop receiving data.
        """
        if self._started:
            self.client.unregister_receive_callback(self._process_raw_buffer)
            self._started = False

        if stop_receive_thread or stop_measurement:
            self.client.stop_receive_thread(stop_measurement=stop_measurement)

    def __iter__(self):
        """To make iteration over epochs easy.
        """
        self._current = 0
        return self

    def next(self):
        """To make iteration over epochs easy.
        """
        if self._current >= self.n_epochs:
            raise StopIteration

        first = True
        while True:
            if len(self._epoch_queue) > 0:
                epoch = self._epoch_queue.pop()
                self._current += 1
                return epoch
            if self._started:
                if first:
                    logger.info('Waiting for epochs..',)
                    first = False
                time.sleep(0.1)

    def _process_raw_buffer(self, raw_buffer):
        """Process raw buffer (callback from RtClient)

        Note: Do not print log messages during regular use. It will be printed
        asynchronously which is annyoing when working in an interactive shell.

        Parameters
        ----------
        raw_buffer : array of float, shape=(nchan, n_times)
            The raw buffer.
        """
        verbose = 'ERROR'
        sfreq = self.info['sfreq']
        n_samp = len(self.times)

        # relative start and stop positions in samples
        tmin_samp = int(round(sfreq * self.tmin))
        tmax_samp = tmin_samp + n_samp

        last_samp = self._first_samp + raw_buffer.shape[1] - 1

        # apply callibration
        raw_buffer = self.cals * raw_buffer

        # detect events
        data = np.abs(raw_buffer[self._stim_picks]).astype(np.int)
        data = np.atleast_2d(data)
        events = _find_events(data, self._first_samp, verbose=verbose)
        idx = np.where(events[:, -1] == self.event_id)[0]

        event_samples = np.r_[self._event_samp_backlog, events[idx, 0]]
        event_samp_backlog = list()
        for event_samp in event_samples:
            epoch = None
            if (event_samp + tmin_samp >= self._first_samp
                    and event_samp + tmax_samp <= last_samp):
                # easy case: whole epoch is in this buffer
                start = event_samp + tmin_samp - self._first_samp
                stop = event_samp + tmax_samp - self._first_samp
                epoch = raw_buffer[:, start:stop]
            elif (event_samp + tmin_samp < self._first_samp
                    and event_samp + tmax_samp <= last_samp):
                # have to use some samples from previous buffer
                if self._last_buffer is None:
                    continue
                n_last = self._first_samp - (event_samp + tmin_samp)
                n_this = n_samp - n_last
                epoch = np.c_[self._last_buffer[:, -n_last:],
                              raw_buffer[:, :n_this]]
            elif event_samp + tmax_samp > last_samp:
                # we need samples from next buffer
                if event_samp + tmin_samp < self._first_samp:
                    raise RuntimeError('Epoch spans more than two raw '
                                       'buffers, increase buffer size!')
                # we will process this epoch with the next buffer
                event_samp_backlog.append(event_samp)
            else:
                raise RuntimeError('Unhandled case..')

            if epoch is not None:
                self._append_epoch_to_queue(epoch)

        self._event_samp_backlog = np.array(event_samp_backlog, dtype=np.int)
        self._first_samp = last_samp + 1
        self._last_buffer = raw_buffer

    def _append_epoch_to_queue(self, epoch):
        """Append a (raw) epoch to queue

        Note: Do not print log messages during regular use. It will be printed
        asynchronously which is annyoing when working in an interactive shell.

        Parameters
        ----------
        epoch : array of float, shape=(nchan, n_times)
            The raw epoch (only calibration has been applied) over all
            channels.
        """

        # select the channels
        epoch = epoch[self.picks, :]

        # apply SSP
        if self.proj and self._projector is not None:
            epoch = np.dot(self._projector, epoch)

        # Detrend
        if self.detrend is not None:
            picks = fiff.pick_types(self.info, meg=True, eeg=True, stim=False,
                                    eog=False, ecg=False, emg=False)
            epoch[picks] = detrend(epoch[picks], self.detrend, axis=1)

        # Baseline correct
        epoch = rescale(epoch, self._raw_times, self.baseline, 'mean',
                        copy=False, verbose='ERROR')

        # Decimate
        if self.decim > 1:
            epoch = epoch[:, self._decim_idx]

        # Decide if this is a good epoch
        is_good, _ = self._is_good_epoch(epoch, verbose='ERROR')

        if is_good:
            self._epoch_queue.append(epoch)
            self._n_good += 1
        else:
            self._n_bad += 1

    def __repr__(self):
        s = 'good / bad epochs received: %d / %d, epochs in queue: %d, '\
            % (self._n_good, self._n_bad, len(self._epoch_queue))
        s += ', tmin : %s (s)' % self.tmin
        s += ', tmax : %s (s)' % self.tmax
        s += ', baseline : %s' % str(self.baseline)
        return '<RtEpochs  |  %s>' % s

    def __getattribute__(self, name):
        """Don't allow calling some methods from Epochs, XXX ugly Hack!"""
        unsupported_methods = ['drop_picks', 'drop_bad_epochs', 'drop_epochs',
                               '_get_epoch_from_disk', '__getitem__', 'copy',
                               'save', 'as_data_frame', 'to_nitime',
                               'equalize_event_counts']

        if name in unsupported_methods:
            raise AttributeError('Method %s not supported with RtEpochs' % name)
        return super(Epochs, self).__getattribute__(name)
