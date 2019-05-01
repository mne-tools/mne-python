# Authors: Teon Brooks <teon.brooks@gmail.com>
#          Mainak Jas <mainakjas@gmail.com>
#
# License: BSD (3-clause)

import threading
import time
import numpy as np

from ..utils import logger, fill_doc


def _buffer_recv_worker(client):
    """Worker thread that constantly receives buffers."""
    try:
        for raw_buffer in client.iter_raw_buffers():
            client._push_raw_buffer(raw_buffer)
    except RuntimeError as err:
        # something is wrong, the server stopped (or something)
        client._recv_thread = None
        print('Buffer receive thread stopped: %s' % err)


@fill_doc
class _BaseClient(object):
    """Base Realtime Client.

    Parameters
    ----------
    info : instance of mne.Info | None
        The measurement info read in from a file. If None, it is generated from
        the realtime stream. This method may result in less info than expected.
    host : str
        The identifier of the server. IP address, LSL id, or raw filename.
    port : int | None
        Port to use for the connection.
    wait_max : float
        Maximum time (in seconds) to wait for real-time buffer to start.
    tmin : float | None
        Time instant to start receiving buffers. If None, start from the latest
        samples available.
    tmax : float
        Time instant to stop receiving buffers.
    buffer_size : int
        Size of each buffer in terms of number of samples.
    %(verbose)s
    """

    def __init__(self, info=None, host='localhost', port=None,
                 wait_max=10., tmin=None, tmax=np.inf,
                 buffer_size=1000, verbose=None):  # noqa: D102
        self.info = info
        self.host = host
        self.port = port
        self.wait_max = wait_max
        self.tmin = tmin
        self.tmax = tmax
        self.buffer_size = buffer_size
        self.verbose = verbose
        self._recv_thread = None
        self._recv_callbacks = list()

    def __enter__(self):  # noqa: D105

        # connect to buffer
        logger.info("Client: Waiting for server to start")
        start_time = time.time()
        while time.time() < (start_time + self.wait_max):
            try:
                self._connect()
                logger.info("Client: Connected")
                break
            except Exception:
                time.sleep(0.1)
        else:
            raise RuntimeError('Could not connect to Client.')

        if not self.info:
            self.info = self._create_info()
        self._enter_extra()

        return self

    def __exit__(self, type, value, traceback):
        self._disconnect()

        return self

    @fill_doc
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
        """Return an iterator over raw buffers."""
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

    def start(self):
        """Start the client."""
        self.__enter__()

        return self

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

    def stop(self):
        """Stop the client."""
        self._disconnect()

        return self

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

    def _connect(self):
        """Connect to client device."""
        pass

    def _create_info(self):
        """Create an mne.Info class for connection to client."""
        pass

    def _disconnect(self):
        """Disconnect the client device."""
        pass

    def _enter_extra(self):
        """Run additional commands in __enter__.

        For system-specific loading and initializing after connect but
        during the enter.

        """
        pass

    def _push_raw_buffer(self, raw_buffer):
        """Push raw buffer to clients using callbacks."""
        for callback in self._recv_callbacks:
            callback(raw_buffer)
