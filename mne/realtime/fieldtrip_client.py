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
import numpy
import copy
import time
import threading

VERSION = 1
PUT_HDR = 0x101
PUT_DAT = 0x102
PUT_EVT = 0x103
PUT_OK = 0x104
PUT_ERR = 0x105
GET_HDR = 0x201
GET_DAT = 0x202
GET_EVT = 0x203
GET_OK = 0x204
GET_ERR = 0x205
FLUSH_HDR = 0x301
FLUSH_DAT = 0x302
FLUSH_EVT = 0x303
FLUSH_OK = 0x304
FLUSH_ERR = 0x305
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


def serialize(A):
    """Returns Fieldtrip data type and string representation of the given
    object, if possible.
    """
    if isinstance(A, str):
        return (0, A)

    if isinstance(A, numpy.ndarray):
        dt = A.dtype
        if not(dt.isnative) or dt.num < 1 or dt.num >= len(data_type):
            return (DATATYPE_UNKNOWN, None)

        ft = data_type[dt.num]
        if ft == -1:
            return (DATATYPE_UNKNOWN, None)

        if A.flags['C_CONTIGUOUS']:
            # great, just use the array's buffer interface
            return (ft, str(A.data))

        # otherwise, we need a copy to C order
        AC = A.copy('C')
        return (ft, str(AC.data))

    if isinstance(A, int):
        return (DATATYPE_INT32, struct.pack('i', A))

    if isinstance(A, float):
        return (DATATYPE_FLOAT64, struct.pack('d', A))

    return (DATATYPE_UNKNOWN, None)


class Chunk:
    def __init__(self):
        self.type = 0
        self.size = 0
        self.buf = ''


class Header:
    """Class for storing header information in the FieldTrip buffer format"""
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


class FtClient:
    """Class for managing a client connection to a FieldTrip buffer."""
    def __init__(self):
        self.is_connected = False
        self.sock = []

    def connect(self, hostname, port=1972):
        """connect(hostname [, port]) -- make a connection, default port
        is 1972."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((hostname, port))
        self.sock.setblocking(True)
        self.is_connected = True

    def disconnect(self):
        """disconnect() -- close a connection."""
        if self.is_connected:
            self.sock.close()
            self.sock = []
            self.is_connected = False

    def send_raw(self, request):
        """Send all bytes of the string 'request' out to socket."""
        if not(self.is_connected):
            raise IOError('Not connected to FieldTrip buffer')

        N = len(request)
        nw = self.sock.send(request)
        while nw < N:
            nw += self.sock.send(request[nw:])

    def send_request(self, command, payload=None):
        if payload is None:
            request = struct.pack('HHI', VERSION, command, 0)
        else:
            request = struct.pack('HHI', VERSION,
                                  command, len(payload)) + payload
        self.send_raw(request)

    def receive_response(self, min_bytes=0):
        """Receive response from server on socket 's' and return it as
        (status,bufsize,payload).
        """

        resp_hdr = self.sock.recv(8)
        while len(resp_hdr) < 8:
            resp_hdr += self.sock.recv(8 - len(resp_hdr))

        (version, command, bufsize) = struct.unpack('HHI', resp_hdr)

        if version != VERSION:
            self.disconnect()
            raise IOError('Bad response from buffer server - disconnecting')

        if bufsize > 0:
            payload = self.sock.recv(bufsize)
            while len(payload) < bufsize:
                payload += self.sock.recv(bufsize - len(payload))
        else:
            payload = None
        return (command, bufsize, payload)

    def get_header(self):
        """get_header() -- grabs header information from the buffer an returns
        it as a Header object.
        """

        self.send_request(GET_HDR)
        (status, bufsize, payload) = self.receive_response()

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

        H = Header()
        H.n_channels = nchans
        H.n_samples = nsamp
        H.n_events = nevt
        H.f_sample = fsamp
        H.data_type = dtype

        if bfsiz > 0:
            offset = 24
            while offset + 8 < bufsize:
                (chunk_type,
                 chunk_len) = struct.unpack('II',
                                            payload[offset:offset + 8])
                offset += 8
                if offset + chunk_len < bufsize:
                    break
                H.chunks[chunk_type] = payload[offset:offset + chunk_len]
                offset += chunk_len

            if H.chunks.has_key(CHUNK_CHANNEL_NAMES):
                L = H.chunks[CHUNK_CHANNEL_NAMES].split('\0')
                num_lab = len(L)
                if num_lab >= H.n_channels:
                    H.labels = L[0:H.n_channels]

        return H

    def put_header(self, n_channels, f_sample, data_type,
                   labels=None, chunks=None):
        have_labels = False
        extras = ''
        if not(labels is None):
            ser_labels = ''
            try:
                for n in range(0, n_channels):
                    ser_labels += labels[n] + '\0'
            except:
                raise ValueError('Channels names (labels), if given, must \
                                  be a list of N=num_channels strings')

            extras = struct.pack('II', CHUNK_CHANNEL_NAMES,
                                 len(ser_labels)) + ser_labels
            have_labels = True

        if not(chunks is None):
            for chunk_type, chunk_data in chunks:
                if have_labels and chunk_type == CHUNK_CHANNEL_NAMES:
                    # ignore channel names chunk in case we got labels
                    continue
                extras += struct.pack('II', chunk_type,
                                      len(chunk_data)) + chunk_data

        size_chunks = len(extras)

        hdef = struct.pack('IIIfII', n_channels,
                           0, 0, f_sample, data_type, size_chunks)
        request = struct.pack('HHI', VERSION, PUT_HDR,
                              size_chunks + len(hdef)) + hdef + extras
        self.send_raw(request)
        (status, bufsize, resp_buf) = self.receive_response()
        if status != PUT_OK:
            raise IOError('Header could not be written')

    def get_data(self, index=None):
        """Retrieves data samples

        Parameters
        ----------
        index : 

        Returns
        -------

        get_data([indices]) -- retrieve data samples and return them as a
        Numpy array, samples in rows(!). The 'indices' argument is optional,
        and if given, must be a tuple or list with inclusive, zero-based
        start/end indices.
        """

        if index is None:
            request = struct.pack('HHI', VERSION, GET_DAT, 0)
        else:
            indS = int(index[0])
            indE = int(index[1])
            request = struct.pack('HHIII', VERSION, GET_DAT, 8, indS, indE)
        self.send_raw(request)

        (status, bufsize, payload) = self.receive_response()
        if status == GET_ERR:
            return None

        if status != GET_OK:
            self.disconnect()
            raise IOError('Bad response from buffer server - disconnecting')

        if bufsize < 16:
            self.disconnect()
            raise IOError('Invalid DATA packet received (too few bytes)')

        (nchans, nsamp, datype, bfsiz) = struct.unpack('IIII', payload[0:16])

        if bfsiz < bufsize - 16 or datype >= len(numpy_type):
            raise IOError('Invalid DATA packet received')

        raw = payload[16:bfsiz + 16]
        D = numpy.ndarray((nsamp, nchans), dtype=numpy_type[datype], buffer=raw)

        return D

    def put_data(self, D):
        """put_data(D) -- writes samples that must be given as a NUMPY array,
           samples x channels. The type of the samples (D) and the number of
           channels must match the corresponding quantities in the FieldTrip
           buffer.
        """

        if not(isinstance(D, numpy.ndarray)) or len(D.shape) != 2:
            raise ValueError('Data must be given as a NUMPY array \
                             (samples x channels)')

        n_samp = D.shape[0]
        n_chan = D.shape[1]

        (data_type, data_buf) = serialize(D)

        data_bufSize = len(data_buf)

        request = struct.pack('HHI', VERSION, PUT_DAT, 16 + data_bufSize)
        data_def = struct.pack('IIII', n_chan, n_samp, data_type, data_bufSize)
        self.send_raw(request + data_def + data_buf)

        (status, bufsize, resp_buf) = self.receive_response()
        if status != PUT_OK:
            raise IOError('Samples could not be written.')

    def poll(self):

        request = struct.pack('HHIIII', VERSION, WAIT_DAT, 12, 0, 0, 0)
        self.send_raw(request)

        (status, bufsize, resp_buf) = self.receive_response()

        if status != WAIT_OK or bufsize < 8:
            raise IOError('Polling failed.')

        return struct.unpack('II', resp_buf[0:8])

    def wait(self, nsamples, nevents, timeout):
        request = struct.pack('HHIIII', VERSION, WAIT_DAT, 12,
                              int(nsamples), int(nevents), int(timeout))
        self.send_raw(request)

        (status, bufsize, resp_buf) = self.receive_response()

        if status != WAIT_OK or bufsize < 8:
            raise IOError('Wait request failed.')

        return struct.unpack('II', resp_buf[0:8])


def _buffer_recv_worker(ft_client, timeout):
    """Worker thread that constantly receives buffers"""
    try:
        for raw_buffer in ft_client.raw_buffers(timeout):
            ft_client._push_raw_buffer(raw_buffer)
    except RuntimeError as err:
        # something is wrong, the server stopped (or something)
        ft_client._recv_thread = None
        print('Buffer receive thread stopped: %s' % err)


# The following additions make the Fieldtrip client MNE-Python compatible
class MneFtClient(object):
    def __init__(self, ft_client, raw, tmin, tmax, buffer_size,
                 timeout=numpy.inf, verbose=None):

        self.raw = raw
        self.ft_header = ft_client.get_header()
        self.verbose = verbose

        if self.ft_header is None:
            raise RuntimeError('Failed to retrieve Fieldtrip header!')

        self.ft_client = copy.deepcopy(ft_client)

        self.raw.info['nchan'] = self.ft_header.n_channels
        self.raw.info['sfreq'] = self.ft_header.f_sample
        self.raw.info['ch_names'] = self.ft_header.labels
        self.ch_names = self.ft_header.labels

        self.info = copy.deepcopy(self.raw.info)

        sfreq = self.raw.info['sfreq']
        self.tmin_samp = int(round(sfreq * tmin))
        self.tmax_samp = int(round(sfreq * tmax))
        self.buffer_size = buffer_size
        self.timeout = timeout

        self._recv_thread = None
        self._recv_callbacks = list()

    def get_measurement_info(self):
        """Returns the measurement info.

        Returns
        -------
        self.info : dict
            The measurement info.
        """
        return self.info

    def register_receive_callback(self, callback):
        """Register a raw buffer receive callback

        Parameters
        ----------
        callback : callable
            The callback. The raw buffer is passed as the first parameter
            to callback.
        """
        if callback not in self._recv_callbacks:
            self._recv_callbacks.append(callback)

    def unregister_receive_callback(self, callback):
        """Unregister a raw buffer receive callback
        """
        if callback in self._recv_callbacks:
            self._recv_callbacks.remove(callback)

    def _push_raw_buffer(self, raw_buffer):
        """Push raw buffer to clients using callbacks"""
        for callback in self._recv_callbacks:
            callback(raw_buffer)

    def start_receive_thread(self, nchan):
        """Start the receive thread

        If the measurement has not been started, it will also be started.

        Parameters
        ----------
        nchan : int
            The number of channels in the data.
        """

        if self._recv_thread is None:

            self._recv_thread = threading.Thread(target=_buffer_recv_worker,
                                                 args=(self, self.timeout))
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

    def raw_buffers(self, timeout):
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
            raw_buffer = self.ft_client.get_data([start, stop])

            # wait till data buffer is available
            start_time = time.time()  # init delay counter.
            while raw_buffer is None:
                current_time = time.time()
                if (current_time > start_time + timeout):
                    raise StopIteration
                time.sleep(0.1)
                raw_buffer = self.ft_client.get_data([start, stop])

            raw_buffer = raw_buffer.transpose()
            # To allow changes to data matrix
            raw_buffer.flags.writeable = True

            yield raw_buffer
