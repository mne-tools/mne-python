# Authors: Teon Brooks <teon.brooks@gmail.com>
#          Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

import copy
import numpy as np
from ..event import find_events


class _BaseClient(object):
    """Base Realtime Client.

    Parameters
    ----------
    host : str
        The IP address of the server.
    port : int
        Port to use for the connection.
    tmin : float | None
        Time instant to start receiving buffers. If None, start from the latest
        samples available.
    tmax : float
        Time instant to stop receiving buffers.
    buffer_size : int
        Size of each buffer in terms of number of samples.
    wait_max : float
        Maximum time (in seconds) to wait for real-time buffer to start
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).
    """

    def __init__(self, host, port, tmin, tmax, buffer_size, verbose=None):  # noqa: D102
        self.host = host
        self.port = port
        self.tmin = tmin
        self.tmax = tmax
        self.buffer_size = buffer_size
        self.verbose = verbose

    def connect(self):
        pass

    def __enter__(self):  # noqa: D105

        # connect to FieldTrip buffer
        logger.info("Client: Waiting for server to start")
        start_time, current_time = time.time(), time.time()
        success = False
        while current_time < (start_time + self.wait_max):
            try:
                self.connect()
                logger.info("FieldTripClient: Connected")
                success = True
                break
            except:
                current_time = time.time()
                time.sleep(0.1)


    def _register_receive_callback(self, callback):
        """Register a raw buffer receive callback.

        Parameters
        ----------
        callback : callable
            The callback. The raw buffer is passed as the first parameter
            to callback.
        """
        if callback not in self._recv_callbacks:
            self._recv_callbacks.append(callback)

    def _unregister_receive_callback(self, callback):
        """Unregister a raw buffer receive callback.

        Parameters
        ----------
        callback : callable
            The callback to unregister.
        """
        if callback in self._recv_callbacks:
            self._recv_callbacks.remove(callback)

    def _push_raw_buffer(self, raw_buffer):
        """Push raw buffer to clients using callbacks."""
        for callback in self._recv_callbacks:
            callback(raw_buffer)

    def _start_receive_thread(self, nchan):
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

    def _stop_receive_thread(self, stop_measurement=False):
        """Stop the receive thread.

        Parameters
        ----------
        stop_measurement : bool
            Also stop the measurement.
        """
        if self._recv_thread is not None:
            self._recv_thread.stop()
            self._recv_thread = None
