# Author: Mainak Jas <mainak@neuro.hut.fi>
# License: BSD (3-clause)

from ..externals.six.moves import queue
import time
import socket
from ..externals.six.moves import socketserver
import threading

import numpy as np

from ..utils import logger, verbose


class _ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Create a threaded TCP server.

    Parameters
    ----------
    server_address : str
        Address on which server is listening
    request_handler_class : subclass of BaseRequestHandler
         _TriggerHandler which defines the handle method
    stim_server : instance of StimServer
        object of StimServer class
    """

    def __init__(self, server_address, request_handler_class,
                 stim_server):  # noqa: D102

        # Basically, this server is the same as a normal TCPServer class
        # except that it has an additional attribute stim_server

        # Create the server and bind it to the desired server address
        socketserver.TCPServer.__init__(self, server_address,
                                        request_handler_class,
                                        False)

        self.stim_server = stim_server


class _TriggerHandler(socketserver.BaseRequestHandler):
    """Request handler on the server side."""

    def handle(self):
        """Handle requests on the server side."""
        self.request.settimeout(None)

        while self.server.stim_server._running:
            data = self.request.recv(1024)  # clip input at 1Kb
            data = data.decode()  # need to turn it into a string (Py3k)

            if data == 'add client':
                # Add stim_server._client
                client_id = self.server.stim_server \
                                ._add_client(self.client_address[0],
                                             self)

                # Instantiate queue for communication between threads
                # Note: new queue for each handler
                if not hasattr(self, '_tx_queue'):
                    self._tx_queue = queue.Queue()

                self.request.sendall("Client added".encode('utf-8'))

                # Mark the client as running
                for client in self.server.stim_server._clients:
                    if client['id'] == client_id:
                        client['running'] = True

            elif data == 'get trigger':

                # Pop triggers and send them
                if (self._tx_queue.qsize() > 0 and
                        self.server.stim_server, '_clients'):

                    trigger = self._tx_queue.get()
                    self.request.sendall(str(trigger).encode('utf-8'))
                else:
                    self.request.sendall("Empty".encode('utf-8'))


class StimServer(object):
    """Stimulation Server.

    Server to communicate with StimClient(s).

    Parameters
    ----------
    port : int
        The port to which the stimulation server must bind to.
    n_clients : int
        The number of clients which will connect to the server.

    See Also
    --------
    StimClient
    """

    def __init__(self, port=4218, n_clients=1):  # noqa: D102

        # Start a threaded TCP server, binding to localhost on specified port
        self._data = _ThreadedTCPServer(('', port),
                                        _TriggerHandler, self)
        self.n_clients = n_clients

    def __enter__(self):  # noqa: D105
        # This is done to avoid "[Errno 98] Address already in use"
        self._data.allow_reuse_address = True
        self._data.server_bind()
        self._data.server_activate()

        # Start a thread for the server
        self._thread = threading.Thread(target=self._data.serve_forever)

        # Ctrl-C will cleanly kill all spawned threads
        # Once the main thread exits, other threads will exit
        self._thread.daemon = True
        self._thread.start()

        self._running = False
        self._clients = list()
        return self

    def __exit__(self, type, value, traceback):  # noqa: D105
        self.shutdown()

    @verbose
    def start(self, timeout=np.inf, verbose=None):
        """Start the server.

        Parameters
        ----------
        timeout : float
            Maximum time to wait for clients to be added.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more).
        """
        # Start server
        if not self._running:
            logger.info('RtServer: Start')
            self._running = True

            start_time = time.time()  # init delay counter.

            # wait till n_clients are added
            while (len(self._clients) < self.n_clients):
                current_time = time.time()

                if (current_time > start_time + timeout):
                    raise StopIteration

                time.sleep(0.1)

    @verbose
    def _add_client(self, ip, sock, verbose=None):
        """Add client.

        Parameters
        ----------
        ip : str
            IP address of the client.
        sock : instance of socket.socket
            The client socket.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more).
        """
        logger.info("Adding client with ip = %s" % ip)

        client = dict(ip=ip, id=len(self._clients), running=False, socket=sock)
        self._clients.append(client)

        return client['id']

    @verbose
    def shutdown(self, verbose=None):
        """Shutdown the client and server.

        Parameters
        ----------
        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more).
        """
        logger.info("Shutting down ...")

        # stop running all the clients
        if hasattr(self, '_clients'):
            for client in self._clients:
                client['running'] = False

        self._running = False

        self._data.shutdown()
        self._data.server_close()
        self._data.socket.close()

    @verbose
    def add_trigger(self, trigger, verbose=None):
        """Add a trigger.

        Parameters
        ----------
        trigger : int
            The trigger to be added to the queue for sending to StimClient.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more).

        See Also
        --------
        StimClient.get_trigger
        """
        for client in self._clients:
            client_id = client['id']
            logger.info("Sending trigger %d to client %d"
                        % (trigger, client_id))
            client['socket']._tx_queue.put(trigger)


class StimClient(object):
    """Stimulation Client.

    Client to communicate with StimServer

    Parameters
    ----------
    host : str
        Hostname (or IP address) of the host where StimServer is running.
    port : int
        Port to use for the connection.
    timeout : float
        Communication timeout in seconds.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    See Also
    --------
    StimServer
    """

    @verbose
    def __init__(self, host, port=4218, timeout=5.0,
                 verbose=None):  # noqa: D102
        try:
            logger.info("Setting up client socket")
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(timeout)
            self._sock.connect((host, port))

            logger.info("Establishing connection with server")
            data = "add client".encode('utf-8')
            n_sent = self._sock.send(data)
            if n_sent != len(data):
                raise RuntimeError('Could not communicate with server')
            resp = self._sock.recv(1024).decode()  # turn bytes into str (Py3k)

            if resp == 'Client added':
                logger.info("Connection established")
            else:
                raise RuntimeError('Client not added')

        except Exception:
            raise RuntimeError('Setting up acquisition <-> stimulation '
                               'computer connection (host: %s '
                               'port: %d) failed. Make sure StimServer '
                               'is running.' % (host, port))

    def close(self):
        """Close the socket object."""
        self._sock.close()

    @verbose
    def get_trigger(self, timeout=5.0, verbose=None):
        """Get triggers from StimServer.

        Parameters
        ----------
        timeout : float
            maximum time to wait for a valid trigger from the server
        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more).

        See Also
        --------
        StimServer.add_trigger
        """
        start_time = time.time()  # init delay counter. Will stop iterations

        while True:
            try:
                current_time = time.time()

                # Raise timeout error
                if current_time > (start_time + timeout):
                        logger.info("received nothing")
                        return None

                self._sock.send("get trigger".encode('utf-8'))
                trigger = self._sock.recv(1024)

                if trigger != 'Empty':
                    logger.info("received trigger %s" % str(trigger))
                    return int(trigger)

            except RuntimeError as err:
                logger.info('Cannot receive triggers: %s' % (err))
