# Author: Mainak Jas
#
# Original code by S. Klanke can be found here:
# https://code.google.com/p/fieldtrip/source/browse/trunk/realtime/
#
# Modified by Mainak Jas to make it PEP8 compliant and
# conform to MNE-Python conventions.
#
# This is the python client for the Fieltrip buffer.
#
# License: BSD (3-clause)

import socket
import struct
import copy
import threading

import numpy as np

VERSION = 1
GET_HDR = 0x201
GET_DAT = 0x202
GET_EVT = 0x203
GET_OK = 0x204
GET_ERR = 0x205
WAIT_DAT = 0x402
WAIT_OK = 0x404
WAIT_ERR = 0x405

DATATYPE_CHAR = 0
DATATYPE_UINT8 = 1
DATATYPE_UINT16 = 2
DATATYPE_UINT32 = 3
DATATYPE_UINT64 = 4
DATATYPE_INT8 = 5
DATATYPE_INT16 = 6
DATATYPE_INT32 = 7
DATATYPE_INT64 = 8
DATATYPE_FLOAT32 = 9
DATATYPE_FLOAT64 = 10
DATATYPE_UNKNOWN = 0xFFFFFFFF

CHUNK_UNSPECIFIED = 0
CHUNK_CHANNEL_NAMES = 1
CHUNK_CHANNEL_FLAGS = 2
CHUNK_RESOLUTIONS = 3
CHUNK_ASCII_KEYVAL = 4
CHUNK_NIFTI1 = 5
CHUNK_SIEMENS_AP = 6
CHUNK_CTF_RES4 = 7
CHUNK_NEUROMAG_FIF = 8

# List for converting FieldTrip datatypes to Numpy datatypes
numpy_type = ['int8', 'uint8', 'uint16', 'uint32', 'uint64',
              'int8', 'int16', 'int32', 'int64', 'float32', 'float64']
# Corresponding word sizes
word_size = [1, 1, 2, 4, 8, 1, 2, 4, 8, 4, 8]
# FieldTrip data type as indexed by numpy dtype.num
# this goes  0 => nothing, 1..4 => int8, uint8, int16, uint16, 7..10 =>
# int32, uint32, int64, uint64  11..12 => float32, float64
data_type = [-1, 5, 1, 6, 2, -1, -1, 7, 3, 8, 4, 9, 10]


class Chunk:
    def __init__(self):
        self.type = 0
        self.size = 0
        self.buf = ''


class Header:
    """Class for storing header information in the FieldTrip buffer format."""
    def __init__(self):
        self.n_channels = 0
        self.n_samples = 0
        self.n_events = 0
        self.f_sample = 0.0
        self.data_type = 0
        self.chunks = {}
        self.labels = []

    def __str__(self):
        return ('Channels.: %i\n_samples..: %i\n_events...: %i'
                '\n_sampFreq.: %f\n_dataType.: %s\n') % (self.n_channels,
                                                         self.n_samples,
                                                         self.n_events,
                                                         self.f_sample,
                                                         numpy_type[self.data_type]
                                                         )


class FtClient(object):
    """Class for managing a client connection to a FieldTrip buffer."""
    def __init__(self):
        self.is_connected = False

    def connect(self, host, port=1972):
        """Connect client to Fieldtrip buffer.

        Parameters
        ----------
        host : str
            Hostname (or IP address) of the host where Fieldtrip buffer is
            running.
        port : int
            Port to use for the command connection.
        """
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self.sock.setblocking(True)
        self.is_connected = True

    def disconnect(self):
        """Close connection."""

        if self.is_connected:
            self.sock.close()
            self.is_connected = False

    def send_raw(self, request):
        """Send request to Fieldtrip buffer.

        Parameters
        ----------
        request : str
            The request sent to the server.
        """

        if not(self.is_connected):
            raise IOError('Not connected to FieldTrip buffer')

        N = len(request)
        nw = self.sock.send(request)
        while nw < N:
            nw += self.sock.send(request[nw:])

    def send_request(self, command, payload=None):
        """Send request to Fieldtrip buffer.

        Parameters
        ----------
        command : int
            The command to the Fieldtrip buffer (e.g. GET_HDR).
        payload : str
            Message payload to server.
        """
        if payload is None:
            request = struct.pack('HHI', VERSION, command, 0)
        else:
            request = struct.pack('HHI', VERSION,
                                  command, len(payload)) + payload
        self.send_raw(request)

    def receive_response(self):
        """Receive response from server.

        Returns
        -------
        status : int
            The status of the response (e.g. GET_OK, GET_ERR).
        buf_size : int
            Number of bytes of additional information attached
            to response.
        payload : str
            Message payload received from server.
        """

        resp_hdr = self.sock.recv(8)
        while len(resp_hdr) < 8:
            resp_hdr += self.sock.recv(8 - len(resp_hdr))

        version, command, bufsize = struct.unpack('HHI', resp_hdr)

        if version != VERSION:
            self.disconnect()
            raise IOError('Bad response from buffer server - disconnecting')

        if bufsize > 0:
            payload = self.sock.recv(bufsize)
            while len(payload) < bufsize:
                payload += self.sock.recv(bufsize - len(payload))
        else:
            payload = None
        return command, bufsize, payload

    def get_header(self):
        """Get header information from buffer.

        Returns
        -------
        header : Header object
            An instance of Header.
        """

        self.send_request(GET_HDR)
        status, bufsize, payload = self.receive_response()

        if status == GET_ERR:
            return None

        if status != GET_OK:
            self.disconnect()
            raise IOError('Bad response from buffer server - disconnecting')

        if bufsize < 24:
            self.disconnect()
            raise IOError('Invalid HEADER packet received (too few bytes) \
                          - disconnecting')

        (nchans, nsamp, nevt, fsamp,
         dtype, bfsiz) = struct.unpack('IIIfII',
                                       payload[0:24])

        header = Header()
        header.n_channels = nchans
        header.n_samples = nsamp
        header.n_events = nevt
        header.f_sample = fsamp
        header.data_type = dtype

        if bfsiz > 0:
            offset = 24
            while offset + 8 < bufsize:
                (chunk_type,
                 chunk_len) = struct.unpack('II',
                                            payload[offset:offset + 8])
                offset += 8
                if offset + chunk_len < bufsize:
                    break
                header.chunks[chunk_type] = payload[offset:offset + chunk_len]
                offset += chunk_len

            if CHUNK_CHANNEL_NAMES in header.chunks:
                L = header.chunks[CHUNK_CHANNEL_NAMES].split('\0')
                num_lab = len(L)
                if num_lab >= header.n_channels:
                    header.labels = L[0:header.n_channels]

        return header

    def get_data(self, index=None):
        """Retrieves data samples.

        Parameters
        ----------
        index : None(default) or tuple of length 2
            The start and stop index of the samples to be retreived.

        Returns
        -------
        data : array of float, shape=(nchan, n_times)
            The buffer data.
        """

        if index is None:
            request = struct.pack('HHI', VERSION, GET_DAT, 0)
        else:
            start_idx = int(index[0])
            end_idx = int(index[1])
            request = struct.pack('HHIII', VERSION, GET_DAT, 8, start_idx,
                                  end_idx)
        self.send_raw(request)

        status, bufsize, payload = self.receive_response()
        if status == GET_ERR:
            return None

        if status != GET_OK:
            self.disconnect()
            raise IOError('Bad response from buffer server - disconnecting')

        if bufsize < 16:
            self.disconnect()
            raise IOError('Invalid DATA packet received (too few bytes)')

        nchans, nsamp, datype, bfsiz = struct.unpack('IIII', payload[0:16])

        if bfsiz < bufsize - 16 or datype >= len(numpy_type):
            raise IOError('Invalid DATA packet received')

        raw = payload[16:bfsiz + 16]
        data = np.ndarray((nsamp, nchans), dtype=numpy_type[datype],
                          buffer=raw).transpose()

        # To allow changes to data matrix
        data.flags.writeable = True

        return data

    def wait(self, n_samples, n_events, timeout):
        """Makes client wait for newly arrived samples or events.

        Parameters
        ----------
        n_samples : int
            Maximum number of samples after which server responds.
        n_events : int
            Maximum number of events after which server responds.
        timeout : int
            Maximum time (in ms) after which server responds.

        Returns
        -------
        samples : int
            Number of samples when server responds.
        events : int
            Number of events when server responds.
        """
        request = struct.pack('HHIIII', VERSION, WAIT_DAT, 12,
                              int(n_samples), int(n_events), int(timeout))
        self.send_raw(request)

        status, bufsize, resp_buf = self.receive_response()

        if status != WAIT_OK or bufsize < 8:
            raise IOError('Wait request failed.')

        return struct.unpack('II', resp_buf[0:8])


def _buffer_recv_worker(ft_client):
    """Worker thread that constantly receives buffers."""
    try:
        for raw_buffer in ft_client.raw_buffers():
            ft_client._push_raw_buffer(raw_buffer)
    except RuntimeError as err:
        # something is wrong, the server stopped (or something)
        ft_client._recv_thread = None
        print('Buffer receive thread stopped: %s' % err)


# The following additions make the Fieldtrip client MNE-Python compatible
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

        self.ft_header = self.ft_client.get_header()

        if self.ft_header is None:
            raise RuntimeError('Failed to retrieve Fieldtrip header!')

        # modify info attributes according to the fieldtrip header
        self.raw.info['nchan'] = self.ft_header.n_channels
        self.raw.info['sfreq'] = self.ft_header.f_sample
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

        iter_times = zip(list(range(self.tmin_samp, self.tmax_samp,
                              self.buffer_size)),
                         list(range(self.buffer_size, self.tmax_samp,
                              self.buffer_size)))

        for ii, (start, stop) in enumerate(iter_times):

            # wait for currect number of samples to be available
            self.ft_client.wait(stop, np.iinfo(np.uint32).max,
                                np.iinfo(np.uint32).max)

            # get the samples
            raw_buffer = self.ft_client.get_data([start, stop])

            yield raw_buffer
