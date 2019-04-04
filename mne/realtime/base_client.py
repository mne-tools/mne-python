# Authors: Teon Brooks <teon.brooks@gmail.com>
#          Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

import threading
import time
import numpy as np

from ..utils import logger

def _buffer_recv_worker(client):
    """Worker thread that constantly receives buffers.
    """
    try:
        for raw_buffer in client.iter_raw_buffers():
            client._push_raw_buffer(raw_buffer)
    except RuntimeError as err:
        # something is wrong, the server stopped (or something)
        client._recv_thread = None
        print('Buffer receive thread stopped: %s' % err)


class _BaseClient(object):
    """Base Realtime Client.

    Parameters
    ----------
    identifier : str
        The identifier of the server. IP address or LSL id or raw filename.
    port : int | None
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

    def __init__(self, identifier, port=None, tmin=None, tmax=np.inf,
                 wait_max=10, buffer_size=1000, verbose=None):  # noqa: D102
        self.identifier = identifier
        self.port = port
        self.tmin = tmin
        self.tmax = tmax
        self.wait_max = wait_max
        self.buffer_size = buffer_size
        self.verbose = verbose

    def __enter__(self):  # noqa: D105

        # connect to buffer
        logger.info("Client: Waiting for server to start")
        start_time, current_time = time.time(), time.time()
        success = False
        while current_time < (start_time + self.wait_max):
            try:
                self.connect()
                logger.info("Client: Connected")
                success = True
                break
            except Exception:
                current_time = time.time()
                time.sleep(0.1)

        if not success:
            raise RuntimeError('Could not connect to Buffer')

        self.create_info()
        self._enter_extra()

        return self

    def __exit__(self, type, value, traceback):
        self.disconnect()

        return self

    def connect(self):
        pass

    def create_info(self):
        pass

    def get_data_as_epoch(self, n_samples=1024, picks=None):
        """Return last n_samples from current time.

        Parameters
        ----------
        n_samples : int
            Number of samples to fetch.
        %(picks_all)s

        Returns
        -------
        epoch : instance of Epochs
            The samples fetched as an Epochs object.

        See Also
        --------
        mne.Epochs.iter_evoked
        """
        pass

    def get_measurement_info(self):
        """Return the measurement info.

        Returns
        -------
        self.info : dict
            The measurement info.
        """
        return self.info

    def iter_raw_buffers(self):
        """Return an iterator over raw buffers.
        """
        pass

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

    def stop_receive_thread(self, stop_measurement=False):
        """Stop the receive thread.

        Parameters
        ----------
        stop_measurement : bool
            Also stop the measurement.
        """
        if self._recv_thread is not None:
            self._recv_thread.stop()
            self._recv_thread = None

    def unregister_receive_callback(self, callback):
        """Unregister a raw buffer receive callback.

        Parameters
        ----------
        callback : callable
            The callback to unregister.
        """
        if callback in self._recv_callbacks:
            self._recv_callbacks.remove(callback)

    def _enter_extra(self):
        """For system-specific loading and initializing after connect but
           during the enter.
        """
        pass

    def _push_raw_buffer(self, raw_buffer):
        """Push raw buffer to clients using callbacks."""
        for callback in self._recv_callbacks:
            callback(raw_buffer)
