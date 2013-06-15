# Authors: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import time
import threading
import SocketServer
import StringIO
import logging
logger = logging.getLogger('mne')

import numpy as np

from ..fiff import Raw
from ..fiff.raw import write_raw_buffer
from ..fiff.meas_info import write_meas_info
from ..fiff.constants import FIFF
from ..fiff.write import start_block
from .client import _recv_tag_raw, MNE_RT_GET_CLIENT_ID,\
                    MNE_RT_SET_CLIENT_ALIAS

COMMAND_HELP = '\
{\
    "commands": {\
        "clist": {\
            "description": "Prints and sends all available FiffStreamClients.",\
            "parameters": {}\
        },\
        "close": {\
            "description": "Closes mne_rt_server.",\
            "parameters": {}\
        },\
        "conlist": {\
            "description": "Prints and sends all available connectors.",\
            "parameters": {}\
        },\
        "help": {\
            "description": "Prints and sends this list.",\
            "parameters": {}\
        },\
        "measinfo": {\
            "description": "Sends the measurement info to the specified FiffStreamClient.",\
            "parameters":\
                "id": {\
                    "description": "ID/Alias",\
                    "type": "String"\
                }\
            }\
        },\
        "selcon": {\
            "description": "Selects a new connector, if a measurement is running it will be stopped.",\
            "parameters": {\
                "ConID": {\
                    "description": "Connector ID",\
                    "type": "int"\
                }\
            }\
        },\
        "start": {\
            "description": "Adds specified FiffStreamClient to raw data buffer receivers. If acquisition is not already started, it is triggered.",\
            "parameters": {\
                "id": {\
                    "description": "ID/Alias",\
                    "type": "String"\
                }\
            }\
        },\
        "stop": {\
            "description": "Removes specified FiffStreamClient from raw data buffer receivers.",\
            "parameters": {\
                "id": {\
                    "description": "ID/Alias",\
                    "type": "String"\
                }\
            }\
        },\
        "stop-all": {\
            "description": "Stops the whole acquisition process.",\
            "parameters": {}\
        }\
        "bufsize": {\
            "description": "Sets the buffer size of the FiffStreamClient raw data buffer.",\
            "parameters": {\
                "samples": {\
                    "description": "samples",\
                    "type": "uint"\
                }\
            }\
        },\
        "getbufsize": {\
            "description": "Returns the current buffer size of the FiffStreamClient raw data buffer.",\
            "parameters": {}\
        }\
    }\
}'

CONLIST = '{"id": 0, "active": true, "MNE-Python mock RtServer"}'


class ThreadedTCPServer(SocketServer.ThreadingMixIn, SocketServer.TCPServer):
    allow_reuse_address = True

    def __init__(self, server_address, request_handler_class,
                 rt_server, bind_and_activate=True):
        SocketServer.TCPServer.__init__(self, server_address,
                                        request_handler_class,
                                        bind_and_activate)
        self.rt_server = rt_server


class CmdHandler(SocketServer.BaseRequestHandler):
    def handle(self):
        # register the client
        client_id = self.server.rt_server.add_client(self.client_address[0],
                                                     self, 'cmd')
        while True:
            data = self.request.recv(1024).strip()
            if len(data) == 0:
                continue
            parts = data.split(' ')
            command = parts[0]
            param = parts[1] if len(parts) == 2 else None

            logger.info('RtServer: command %s from %s:%s received'
                        % (command, self.client_address[0],
                           self.client_address[1]))
            if command == 'help':
                self.request.sendall(COMMAND_HELP)
            elif command == 'conlist':
                seld.request.sendall(CON_LIST)
            elif command == 'selcon':
                pass
            elif command == 'start':
                self.server.rt_server.start(int(param))
            elif command == 'stop':
                self.server.rt_server.stop(int(param))
            elif command == 'stop-all':
                self.server.rt_server.stop_all()
            elif command == 'measinfo':
                self.server.rt_server.send_measinfo(int(param))
            elif command == 'bufsize':
                self.server.rt_server.set_buffer_size(int(param))
            elif command == 'get_bufsize':
                self.request.sendall(str(self.server.rt_server._buffer_size))
            else:
                logger.error('RtServer: invalid command received')


class DataHandler(SocketServer.BaseRequestHandler):

    def send_data(self, data):
        if not hasattr(self, '_tx_queue'):
            self._tx_queue = []

        self._tx_queue.append(data)

    def handle(self):
        if not hasattr(self, '_tx_queue'):
            self._tx_queue = []

        self.request.settimeout(0.1)

        # register the client
        client_id = self.server.rt_server.add_client(self.client_address[0],
                                                     self, 'data')
        while True:
            # try to receive some data
            tag, buf = _recv_tag_raw(self.request)

            if tag is not None and buf is not None:
                logger.debug('RtServer: data handler: received %d bytes'
                             % len(buf))
                reply = None
                if tag.kind == FIFF.FIFF_MNE_RT_COMMAND:
                    command = np.fromstring(buf[16:20], dtype='>i4')
                    if command == MNE_RT_GET_CLIENT_ID:
                        # send client id
                        rep = [np.array(FIFF.FIFF_MNE_RT_CLIENT_ID, dtype='>i4'),
                            np.array(FIFF.FIFFT_INT, dtype='>i4'),
                            np.array(4, dtype='>i4'),  # size
                            np.array(0, dtype='>i4'),  # next
                            np.array(client_id, dtype='>i4')]
                        reply = ''.join([r.tostring() for r in rep])
                    elif command == MNE_RT_SET_CLIENT_ALIAS:
                        pass
                    else:
                        logger.error('RtServer: invalid command received')
                if reply is not None:
                    self.request.sendall(reply)
            else:
                # handle the tx queue
                self.request.settimeout(None)

                while len(self._tx_queue) > 0:
                    data = self._tx_queue.pop(0)
                    self.request.sendall(data)

                self.request.settimeout(0.1)


def send_data_worker(rt_server):
    """Worker thread that sends the data to the clients"""
    logger.info('RtServer: send thread starting')
    raw = rt_server._raw
    start = 0
    while rt_server._running:
        t_start = time.time()
        end = start + rt_server._buffer_size
        if end >= raw.n_times:
            if not rt_server._loop:
                break
            data1, _ = raw[:, start:]
            start, end = 0, rt_server._buffer_size - data1.shape[1]
            data2, _ = raw[:, start:end]
            data = np.c_[data1, data2]
        else:
            data, _ = raw[:, start:end]
            start = end

        # encode the data as a fif-string
        fid = StringIO.StringIO()
        write_raw_buffer(fid, data, raw.cals, 'single')
        data = fid.getvalue()

        # send it to the clients
        for client in rt_server._clients:
            if client['running']:
                client['data'].send_data(data)

        # wait until we can send the next buffer
        send_period = (rt_server._speedup * rt_server._buffer_size
                       / raw.info['sfreq'])

        t_elapsed = time.time() - t_start
        time.sleep(send_period - t_elapsed)

    logger.info('RtServer: send thread stopping')


class RtServer(object):
    """Real-time server for simulation and testing.

    For actual real-time measurements, use mne_rt_server from the MNE-CPP
    project.
    """
    def __init__(self, raw_fname, cmd_port=4217, data_port=4218,
                 buffer_size=1000, speedup=1.0, loop=False):

        self._cmd = ThreadedTCPServer(('localhost', cmd_port),
                                      CmdHandler, self)
        self._data = ThreadedTCPServer(('localhost', data_port),
                                       DataHandler, self)

        self._cmd_thread = threading.Thread(target=self._cmd.serve_forever)
        self._cmd_thread.daemon = True
        self._cmd_thread.start()

        self._data_thread = threading.Thread(target=self._data.serve_forever)
        self._data_thread.daemon = True
        self._data_thread.start()

        # open the raw file
        self._raw = Raw(raw_fname)

        # get the measurement info as a byte string that can be sent to clients
        info = StringIO.StringIO()
        start_block(info, FIFF.FIFFB_MEAS)
        write_meas_info(info, self._raw.info)
        self._info = info.getvalue()

        self._running = False
        self._clients = []

        self._buffer_size = buffer_size
        self._speedup = speedup
        self._loop = loop

    def start(self, client_id):
        for client in self._clients:
            if client['id'] == client_id:
                client['running'] = True

        if not self._running:
            logger.info('RtServer: start')
            self._running = True

            # start the send thread
            self._send_thread = threading.Thread(target=send_data_worker,
                                                 args=(self,))
            self._send_thread.start()

    def stop(self, client_id):
        for client in self._clients:
            if client['id'] == client_id:
                client['running'] = False

    def stop_all(self):
        if self._running:
            logger.info('RtServer: stop_all')
            self._running = False
            for client in self._clients:
                client['running'] = False

        # the send thread will stop automatically

    def set_buffer_size(self, buffer_size):
        logger.info('RtServer: setting buffer size to %d samples'
                    % buffer_size)
        self._buffer_size = buffer_size

    def send_measinfo(self, client_id):
        for client in self._clients:
            if client['id'] == client_id:
                logger.info('RtServer: sending measinfo to %d' % client_id)
                client['data'].send_data(self._info)
                return
        logger.error('RtServer: no client with ID %d' % client_id)

    def add_client(self, ip, socket, cl_type):
        for client in self._clients:
            if client['ip'] == ip and client.get(cl_type, None) is None:
                client[cl_type] = socket
                return client['id']

        # add a new client
        client = dict(ip=ip, id=len(self._clients), running=False)
        client[cl_type] = socket
        self._clients.append(client)

        return client['id']

    def shutdown(self):
        """Close all connections and sockets"""
        self._cmd.shutdown()
        self._cmd.server_close()
        self._cmd.socket.close()
        self._data.shutdown()
        self._data.server_close()
        self._data.socket.close()



