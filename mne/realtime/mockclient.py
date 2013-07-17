# Authors: Mainak Jas <mainak@neuro.hut.fi>
#          Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import copy
import numpy as np


class MockRtClient(object):
    """Mock Realtime Client

    Attributes
    ----------
    raw : instance of Raw object
        The raw object which simulates the RtClient
    info : dict
        Measurement info.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    """
    def __init__(self, raw, verbose=None):
        self.raw = raw
        self.info = copy.deepcopy(self.raw.info)
        self.verbose = verbose

    def get_measurement_info(self):
        """Returns the measurement info

        Returns
        -------
        self.info : dict
            The measurement info.
        """
        return self.info

    def send_data(self, epochs, picks, tmin, tmax, buffer_size):
        """ Read from raw object and send them to RtEpochs for processing

        Parameters
        ----------
        epochs : instance of RtEpochs
            The epochs object.
        picks : array of int
            Indices of channels.
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

        sfreq = self.info['sfreq']
        tmin_samp = int(round(sfreq * tmin))
        tmax_samp = int(round(sfreq * tmax))

        iter_times = zip(range(tmin_samp, tmax_samp, buffer_size),
                         range(buffer_size, tmax_samp, buffer_size))

        for ii, (start, stop) in enumerate(iter_times):
            # channels are picked in _append_epoch_to_queue. No need to pick
            # here
            data, times = self.raw[:, start:stop]

            # to undo the calibration done in _process_raw_buffer
            cals = np.zeros(self.info['nchan'])
            for k in range(self.info['nchan']):
                cals[k] = (self.info['chs'][k]['range']
                           * self.info['chs'][k]['cal'])

            self._cals = cals[:, None]
            data[picks, :] = data[picks, :] / self._cals

            epochs._process_raw_buffer(data)

    # The following methods do not seem to be important for this use case,
    # but they need to be present for the emulation to work because
    # RtEpochs expects them to be there.

    def register_receive_callback(self, x):
        """API boilerplate"""
        pass

    def start_receive_thread(self, x):
        """API boilerplate"""
        pass

    def unregister_receive_callback(self, x):
        """API boilerplate"""
        pass

    def _stop_receive_thread(self):
        """API boilerplate"""
        pass
