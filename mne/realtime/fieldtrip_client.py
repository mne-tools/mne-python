# Author: Mainak Jas
#
# License: BSD (3-clause)

import copy
import threading
import numpy as np

from ..externals.FieldTrip import Client as FtClient


def _buffer_recv_worker(ft_client):
    """Worker thread that constantly receives buffers."""

    try:
        for raw_buffer in ft_client.raw_buffers():
            ft_client._push_raw_buffer(raw_buffer)
    except RuntimeError as err:
        # something is wrong, the server stopped (or something)
        ft_client._recv_thread = None
        print('Buffer receive thread stopped: %s' % err)


class FieldTripClient(object):
    """ Realtime FieldTrip client

    Parameters
    ----------
    raw : Raw object
        An instance of Raw.
    host : str
        Hostname (or IP address) of the host where Fieldtrip buffer is running.
    port : int
        Port to use for the connection.
    tmin : float
        Time instant to start receiving buffers.
    tmax : float
        Time instant to stop receiving buffers.
    buffer_size : int
        Size of each buffer in terms of number of samples.
    verbose : bool, str, int, or None
        Log verbosity see mne.verbose.
    """
    def __init__(self, raw, host='localhost', port=1972, tmin=0,
                 tmax=np.inf, buffer_size=1000, verbose=None):
        self.raw = raw
        self.verbose = verbose
        self.info = copy.deepcopy(self.raw.info)

        self.tmin = tmin
        self.tmax = tmax
        self.buffer_size = buffer_size

        self.host = host
        self.port = port

        self._recv_thread = None
        self._recv_callbacks = list()

    def __enter__(self):
        # instantiate Fieldtrip client and connect
        self.ft_client = FtClient()
        self.ft_client.connect(self.host, self.port)

        self.ft_header = self.ft_client.getHeader()

        if self.ft_header is None:
            raise RuntimeError('Failed to retrieve Fieldtrip header!')

        # modify info attributes according to the fieldtrip header
        self.raw.info['nchan'] = self.ft_header.nChannels
        self.raw.info['sfreq'] = self.ft_header.fSample
        self.raw.info['ch_names'] = self.ft_header.labels
        self.ch_names = self.ft_header.labels

        # find start and end samples
        sfreq = self.raw.info['sfreq']
        self.tmin_samp = int(round(sfreq * self.tmin))
        if self.tmax != np.inf:
            self.tmax_samp = int(round(sfreq * self.tmax))
        else:
            self.tmax_samp = np.iinfo(np.uint32).max

        return self

    def __exit__(self, type, value, traceback):
        self.ft_client.disconnect()

    def get_measurement_info(self):
        """Returns the measurement info.

        Returns
        -------
        self.info : dict
            The measurement info.
        """
        return self.info

    def register_receive_callback(self, callback):
        """Register a raw buffer receive callback.

        Parameters
        ----------
        callback : callable
            The callback. The raw buffer is passed as the first parameter
            to callback.
        """
        if callback not in self._recv_callbacks:
            self._recv_callbacks.append(callback)

    def unregister_receive_callback(self, callback):
        """Unregister a raw buffer receive callback."""
        if callback in self._recv_callbacks:
            self._recv_callbacks.remove(callback)

    def _push_raw_buffer(self, raw_buffer):
        """Push raw buffer to clients using callbacks."""
        for callback in self._recv_callbacks:
            callback(raw_buffer)

    def start_receive_thread(self, nchan):
        """Start the receive thread.

        If the measurement has not been started, it will also be started.

        Parameters
        ----------
        nchan : int
            The number of channels in the data.
        """

        if self._recv_thread is None:

            self._recv_thread = threading.Thread(target=_buffer_recv_worker,
                                                 args=(self, ))
            self._recv_thread.daemon = True
            self._recv_thread.start()

    def stop_receive_thread(self, nchan, stop_measurement=False):
        """Stop the receive thread

        Parameters
        ----------
        stop_measurement : bool
            Also stop the measurement.
        """
        if self._recv_thread is not None:
            self._recv_thread.stop()
            self._recv_thread = None

    def raw_buffers(self):
        """Return an iterator over raw buffers

        Returns
        -------
        raw_buffer : generator
            Generator for iteration over raw buffers.
        """

        iter_times = zip(range(self.tmin_samp, self.tmax_samp,
                               self.buffer_size),
                         range(self.buffer_size, self.tmax_samp,
                               self.buffer_size))

        for ii, (start, stop) in enumerate(iter_times):

            # wait for currect number of samples to be available
            self.ft_client.wait(stop, np.iinfo(np.uint32).max,
                                np.iinfo(np.uint32).max)

            # get the samples
            raw_buffer = self.ft_client.getData([start, stop]).transpose()

            yield raw_buffer
