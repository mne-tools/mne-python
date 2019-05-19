# Authors: Mainak Jas <mainak@neuro.hut.fi>
#          Denis Engemann <denis.engemann@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import copy
import numpy as np

from ..event import find_events
from ..io.pick import _picks_to_idx
from ..utils import fill_doc, deprecated

RT_MSG = ('The realtime module is being deprecated from `mne-python` '
          'and moved to its own package, `mne-realtime`. '
          'To install, please use `$ pip install mne_realtime`.')


@fill_doc
@deprecated(RT_MSG)
class MockRtClient(object):
    """Mock Realtime Client.

    Parameters
    ----------
    raw : instance of Raw object
        The raw object which simulates the RtClient
    %(verbose)s
    """

    def __init__(self, raw, verbose=None):  # noqa: D102
        self.raw = raw
        self.info = copy.deepcopy(self.raw.info)
        self.verbose = verbose

        self._current = dict()  # pointer to current index for the event
        self._last = dict()  # Last index for the event

    def get_measurement_info(self):
        """Return the measurement info.

        Returns
        -------
        self.info : dict
            The measurement info.
        """
        return self.info

    @fill_doc
    def send_data(self, epochs, picks, tmin, tmax, buffer_size):
        """Read from raw object and send them to RtEpochs for processing.

        Parameters
        ----------
        epochs : instance of RtEpochs
            The epochs object.
        %(picks_all)s
        tmin : float
            Time instant to start receiving buffers.
        tmax : float
            Time instant to stop receiving buffers.
        buffer_size : int
            Size of each buffer in terms of number of samples.
        """
        # this is important to emulate a thread, instead of automatically
        # or constantly sending data, we will invoke this explicitly to send
        # the next buffer

        picks = _picks_to_idx(self.info, picks, 'all', exclude=())
        sfreq = self.info['sfreq']
        tmin_samp = int(round(sfreq * tmin))
        tmax_samp = int(round(sfreq * tmax))

        iter_times = list(zip(
            list(range(tmin_samp, tmax_samp, buffer_size)),
            list(range(buffer_size, tmax_samp + 1, buffer_size))))
        last_iter_sample = iter_times[-1][1] if iter_times else 0
        if last_iter_sample < tmax_samp:
            iter_times.append((last_iter_sample, tmax_samp))

        for ii, (start, stop) in enumerate(iter_times):
            # channels are picked in _append_epoch_to_queue. No need to pick
            # here
            data, times = self.raw[:, start:stop]

            # to undo the calibration done in _process_raw_buffer
            cals = np.array([[self.info['chs'][k]['range'] *
                              self.info['chs'][k]['cal'] for k in picks]]).T

            data[picks, :] = data[picks, :] / cals

            epochs._process_raw_buffer(data)

    # The following methods do not seem to be important for this use case,
    # but they need to be present for the emulation to work because
    # RtEpochs expects them to be there.

    @fill_doc
    def get_event_data(self, event_id, tmin, tmax, picks=None,
                       stim_channel=None, min_duration=0):
        """Simulate the data for a particular event-id.

        The epochs corresponding to a particular event-id are returned. The
        method remembers the epoch that was returned in the previous call and
        returns the next epoch in sequence. Once all epochs corresponding to
        an event-id have been exhausted, the method returns None.

        Parameters
        ----------
        event_id : int
            The id of the event to consider.
        tmin : float
            Start time before event.
        tmax : float
            End time after event.
        %(picks_all)s
        stim_channel : None | string | list of string
            Name of the stim channel or all the stim channels
            affected by the trigger. If None, the config variables
            'MNE_STIM_CHANNEL', 'MNE_STIM_CHANNEL_1', 'MNE_STIM_CHANNEL_2',
            etc. are read. If these are not found, it will default to
            'STI 014'.
        min_duration : float
            The minimum duration of a change in the events channel required
            to consider it as an event (in seconds).

        Returns
        -------
        data : 2D array with shape [n_channels, n_times]
            The epochs that are being simulated
        """
        # Get the list of all events
        picks = _picks_to_idx(self.info, picks, 'all', exclude=())
        events = find_events(self.raw, stim_channel=stim_channel,
                             verbose=False, output='onset',
                             consecutive='increasing',
                             min_duration=min_duration)

        # Get the list of only the specified event
        idx = np.where(events[:, -1] == event_id)[0]
        event_samp = events[idx, 0]

        # Only do this the first time for each event type
        if event_id not in self._current:

            # Initialize pointer for the event to 0
            self._current[event_id] = 0
            self._last[event_id] = len(event_samp)

        # relative start and stop positions in samples
        tmin_samp = int(round(self.info['sfreq'] * tmin))
        tmax_samp = int(round(self.info['sfreq'] * tmax)) + 1

        if self._current[event_id] < self._last[event_id]:

            # Select the current event from the events list
            ev_samp = event_samp[self._current[event_id]]

            # absolute start and stop positions in samples
            start = ev_samp + tmin_samp - self.raw.first_samp
            stop = ev_samp + tmax_samp - self.raw.first_samp

            self._current[event_id] += 1  # increment pointer

            data, _ = self.raw[picks, start:stop]

            return data

        else:
            return None

    def register_receive_callback(self, x):
        """Fake API boilerplate.

        Parameters
        ----------
        x : None
            Not used.
        """
        pass

    def start_receive_thread(self, x):
        """Fake API boilerplate.

        Parameters
        ----------
        x : None
            Not used.
        """
        pass

    def unregister_receive_callback(self, x):
        """Fake API boilerplate.

        Parameters
        ----------
        x : None
            Not used.
        """  # noqa: D401
        pass

    def _stop_receive_thread(self):
        """Fake API boilerplate."""
        pass
