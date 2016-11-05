# Authors: Christoph Dinh <chdinh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from __future__ import print_function

import socket
import time
from ..externals.six.moves import StringIO
import threading

import numpy as np

from ..utils import logger, verbose
from ..io.constants import FIFF
from ..io.meas_info import read_meas_info
from ..io.tag import Tag, read_tag
from ..io.tree import make_dir_tree

# Constants for fiff realtime fiff messages
MNE_RT_GET_CLIENT_ID = 1
MNE_RT_SET_CLIENT_ALIAS = 2


def _recv_tag_raw(sock):
    """Read a tag and the associated data from a socket.

    Parameters
    ----------
    sock : socket.socket
        The socket from which to read the tag.

    Returns
    -------
    tag : instance of Tag
        The tag.
    buff : str
        The raw data of the tag (including header).
    """
    s = sock.recv(4 * 4)
    if len(s) != 16:
        raise RuntimeError('Not enough bytes received, something is wrong. '
                           'Make sure the mne_rt_server is running.')
    tag = Tag(*np.fromstring(s, '>i4'))
    n_received = 0
    rec_buff = [s]
    while n_received < tag.size:
        n_buffer = min(4096, tag.size - n_received)
        this_buffer = sock.recv(n_buffer)
        rec_buff.append(this_buffer)
        n_received += len(this_buffer)

    if n_received != tag.size:
        raise RuntimeError('Not enough bytes received, something is wrong. '
                           'Make sure the mne_rt_server is running.')

    buff = ''.join(rec_buff)

    return tag, buff


def _buffer_recv_worker(rt_client, nchan):
    """Worker thread that constantly receives buffers."""
    try:
        for raw_buffer in rt_client.raw_buffers(nchan):
            rt_client._push_raw_buffer(raw_buffer)
    except RuntimeError as err:
        # something is wrong, the server stopped (or something)
        rt_client._recv_thread = None
        print('Buffer receive thread stopped: %s' % err)


class RtClient(object):
    """Realtime Client.

    Client to communicate with mne_rt_server

    Parameters
    ----------
    host : str
        Hostname (or IP address) of the host where mne_rt_server is running.
    cmd_port : int
        Port to use for the command connection.
    data_port : int
        Port to use for the data connection.
    timeout : float
        Communication timeout in seconds.
    verbose : bool, str, int, or None
        Log verbosity (see :func:`mne.verbose` and
        :ref:`Logging documentation <tut_logging>` for more).
    """

    @verbose
    def __init__(self, host, cmd_port=4217, data_port=4218, timeout=1.0,
                 verbose=None):  # noqa: D102
        self._host = host
        self._data_port = data_port
        self._cmd_port = cmd_port
        self._timeout = timeout

        try:
            self._cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._cmd_sock.settimeout(timeout)
            self._cmd_sock.connect((host, cmd_port))
            self._cmd_sock.setblocking(0)
        except Exception:
            raise RuntimeError('Setting up command connection (host: %s '
                               'port: %d) failed. Make sure mne_rt_server '
                               'is running. ' % (host, cmd_port))

        try:
            self._data_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._data_sock.settimeout(timeout)
            self._data_sock.connect((host, data_port))
            self._data_sock.setblocking(1)
        except Exception:
            raise RuntimeError('Setting up data connection (host: %s '
                               'port: %d) failed. Make sure mne_rt_server '
                               'is running.' % (host, data_port))

        self.verbose = verbose

        # get my client ID
        self._client_id = self.get_client_id()

        self._recv_thread = None
        self._recv_callbacks = list()

    def _send_command(self, command):
        """Send a command to the server.

        Parameters
        ----------
        command : str
            The command to send.

        Returns
        -------
        resp : str
            The response from the server.
        """
        logger.debug('Sending command: %s' % command)
        command += '\n'
        self._cmd_sock.sendall(command.encode('utf-8'))

        buf, chunk, begin = [], '', time.time()
        while True:
            # if we got some data, then break after wait sec
            if buf and time.time() - begin > self._timeout:
                break
            # if we got no data at all, wait a little longer
            elif time.time() - begin > self._timeout * 2:
                break
            try:
                chunk = self._cmd_sock.recv(8192)
                if chunk:
                    buf.append(chunk)
                    begin = time.time()
                else:
                    time.sleep(0.1)
            except:
                pass

        return ''.join(buf)

    def _send_fiff_command(self, command, data=None):
        """Send a command through the data connection as a fiff tag.

        Parameters
        ----------
        command : int
            The command code.

        data : str
            Additional data to send.
        """
        kind = FIFF.FIFF_MNE_RT_COMMAND
        type = FIFF.FIFFT_VOID
        size = 4
        if data is not None:
            size += len(data)  # first 4 bytes are the command code
        next = 0

        msg = np.array(kind, dtype='>i4').tostring()
        msg += np.array(type, dtype='>i4').tostring()
        msg += np.array(size, dtype='>i4').tostring()
        msg += np.array(next, dtype='>i4').tostring()

        msg += np.array(command, dtype='>i4').tostring()
        if data is not None:
            msg += np.array(data, dtype='>c').tostring()

        self._data_sock.sendall(msg)

    def get_measurement_info(self):
        """Get the measurement information.

        Returns
        -------
        info : dict
            The measurement information.
        """
        cmd = 'measinfo %d' % self._client_id
        self._send_command(cmd)

        buff = []
        directory = []
        pos = 0
        while True:
            tag, this_buff = _recv_tag_raw(self._data_sock)
            tag.pos = pos
            pos += 16 + tag.size
            directory.append(tag)
            buff.append(this_buff)
            if tag.kind == FIFF.FIFF_BLOCK_END and tag.type == FIFF.FIFFT_INT:
                val = np.fromstring(this_buff[-4:], dtype=">i4")
                if val == FIFF.FIFFB_MEAS_INFO:
                    break

        buff = ''.join(buff)

        fid = StringIO(buff)
        tree, _ = make_dir_tree(fid, directory)
        info, meas = read_meas_info(fid, tree)

        return info

    def set_client_alias(self, alias):
        """Set client alias.

        Parameters
        ----------
        alias : str
            The client alias.
        """
        self._send_fiff_command(MNE_RT_SET_CLIENT_ALIAS, alias)

    def get_client_id(self):
        """Get the client ID.

        Returns
        -------
        id : int
            The client ID.
        """
        self._send_fiff_command(MNE_RT_GET_CLIENT_ID)

        # ID is send as answer
        tag, buff = _recv_tag_raw(self._data_sock)
        if (tag.kind == FIFF.FIFF_MNE_RT_CLIENT_ID and
                tag.type == FIFF.FIFFT_INT):
            client_id = int(np.fromstring(buff[-4:], dtype=">i4"))
        else:
            raise RuntimeError('wrong tag received')

        return client_id

    def start_measurement(self):
        """Start the measurement."""
        cmd = 'start %d' % self._client_id
        self._send_command(cmd)

    def stop_measurement(self):
        """Stop the measurement."""
        self._send_command('stop-all')

    def start_receive_thread(self, nchan):
        """Start the receive thread.

        If the measurement has not been started, it will also be started.

        Parameters
        ----------
        nchan : int
            The number of channels in the data.
        """
        if self._recv_thread is None:
            self.start_measurement()

            self._recv_thread = threading.Thread(target=_buffer_recv_worker,
                                                 args=(self, nchan))
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

        if stop_measurement:
            self.stop_measurement()

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
        """Unregister a raw buffer receive callback.

        Parameters
        ----------
        callback : function
            The callback to unregister.
        """
        if callback in self._recv_callbacks:
            self._recv_callbacks.remove(callback)

    def _push_raw_buffer(self, raw_buffer):
        """Push raw buffer to clients using callbacks."""
        for callback in self._recv_callbacks:
            callback(raw_buffer)

    def read_raw_buffer(self, nchan):
        """Read a single buffer with raw data.

        Parameters
        ----------
        nchan : int
            The number of channels (info['nchan']).

        Returns
        -------
        raw_buffer : float array, shape=(nchan, n_times)
            The raw data.
        """
        tag, this_buff = _recv_tag_raw(self._data_sock)

        # skip tags until we get a data buffer
        while tag.kind != FIFF.FIFF_DATA_BUFFER:
            tag, this_buff = _recv_tag_raw(self._data_sock)

        buff = StringIO(this_buff)
        tag = read_tag(buff)
        raw_buffer = tag.data.reshape(-1, nchan).T

        return raw_buffer

    def raw_buffers(self, nchan):
        """Return an iterator over raw buffers.

        Parameters
        ----------
        nchan : int
            The number of channels (info['nchan']).

        Returns
        -------
        raw_buffer : generator
            Generator for iteration over raw buffers.
        """
        while True:
            raw_buffer = self.read_raw_buffer(nchan)
            if raw_buffer is not None:
                yield raw_buffer
            else:
                break
