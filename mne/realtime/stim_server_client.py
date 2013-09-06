# Author: Mainak Jas <mainak@neuro.hut.fi>
# License: BSD (3-clause)

import Queue
import time
import socket
import SocketServer
import threading

import logging
logger = logging.getLogger('mne')
logger.propagate = False

from .. import verbose


class ThreadedTCPServer(SocketServer.ThreadingMixIn, SocketServer.TCPServer):
    """Creates a threaded TCP server

    Parameters
    ----------
    server_address : str
        Address on which server is listening

    request_handler_class : subclass of BaseRequestHandler
        TriggerHandler which defines the handle method

    stim_server : instance of StimServer
        object of StimServer class

    """

    def __init__(self, server_address, request_handler_class,
                 stim_server):

    # Basically, this server is the same as a normal TCPServer class
    # except that it has an additional attribute stim_server

        # Create the server and bind it to the desired server address
        SocketServer.TCPServer.__init__(self, server_address,
                                        request_handler_class,
                                        False)

        self.stim_server = stim_server


class TriggerHandler(SocketServer.BaseRequestHandler):
    """Request handler on the server side."""

    def send_trigger(self, trigger):
        """Create a queue of triggers to be delivered."""

        # If no attribute trigger queue in self, create it
        if not hasattr(self, '_tx_queue'):
            self._tx_queue = []

        self._tx_queue.append(trigger)

    def handle(self):
        """Method to handle requests on the server side."""

        self.request.settimeout(None)

        while self.server.stim_server._running:

            data = self.request.recv(1024)  # clip input at 1Kb

            if data == 'add client':
                # Add stim_server._client
                self.server.stim_server.add_client(self.client_address[0],
                                                   self)
                self.request.sendall("Client added")

            if data == 'get trigger':

                # If the method self has trigger queue, create it
                if not hasattr(self, '_tx_queue'):
                    self._tx_queue = []

                # Pop triggers and send them
                if len(self._tx_queue) > 0:
                    trigger = self._tx_queue.pop(0)
                    logger.info("Trigger %s popped and sent" % (str(trigger)))
                    self.request.sendall(str(trigger))
                else:
                    self.request.sendall("Empty")


class StimServer(object):
    """Stimulation Server

    Server to communicate with StimClient(s).

    Parameters
    ----------
    ip : str
        IP address of the host where StimServer is running.

    port : int
        The port to which the stimulation server must bind to

    """

    def __init__(self, ip='localhost', port=4218):

        # Start a threaded TCP server, binding to localhost on specified port
        self._data = ThreadedTCPServer((ip, port),
                                       TriggerHandler, self)

    def __enter__(self):
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
        return self

    def __exit__(self, type, value, traceback):
        self.shutdown()

    @verbose
    def start(self, verbose=None):
        """Method to start the client.

        Parameters
        ----------
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).

        """

        # Instantiate queue for communication between threads
        if not hasattr(self, '_tx_queue'):
            self._tx_queue = Queue.Queue()

        # Start server and a separate thread to send data
        if not self._running:
            logger.info('RtServer: Start')
            self._running = True

            # wait till client is added
            while not hasattr(self, '_client'):
                pass

            # start the send thread
            self._send_thread = threading.Thread(target=send_trigger_worker,
                                                 args=(self, self._tx_queue))
            self._send_thread.start()

    @verbose
    def add_client(self, ip, sock, verbose=None):
        """Add client and flag it as running.

        Parameters
        ----------
        ip : str
            IP address of the host where StimServer is running.

        sock : instance of socket.socket
            The client socket.

        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).

        """

        self._client = dict()

        logger.info("Adding client with ip = %s" % ip)
        self._client['ip'] = ip
        self._client['running'] = True
        self._client['socket'] = sock

    @verbose
    def shutdown(self, verbose=None):
        """Method to shutdown the client and server.

        Parameters
        ----------
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).

        """

        logger.info("Shutting down ...")

        self._running = False

        if hasattr(self, '_client'):
            self._client['running'] = False

        self._data.shutdown()
        self._data.server_close()
        self._data.socket.close()

    @verbose
    def add_trigger(self, trigger, verbose=None):
        """Method to add a trigger.

        Parameters
        ----------
        trigger : int
            The trigger to be added to the queue for sending to StimClient.

        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).

        """

        logger.info("Adding trigger %d" % trigger)
        self._tx_queue.put(trigger)


def send_trigger_worker(stim_server, tx_queue):
    """Worker thread that sends the data to the client.

    Parameters
    ----------
    stim_server : Instance of StimServer
        The server which delivers the triggers

    tx_queue : instance of Queue
        The queue which contains the triggers to be sent

    """

    while stim_server._running and hasattr(stim_server, '_client'):

        trigger = tx_queue.get()
        stim_server._client['socket'].send_trigger(trigger)


class StimClient(object):
    """Stimulation Client

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
        If not None, override default verbose level (see mne.verbose).

    """

    @verbose
    def __init__(self, host, port=4218, timeout=5.0, verbose=None):
        self._host = host
        self._port = port

        try:
            logger.info("Setting up client socket")
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(timeout)
            self._sock.connect((host, port))

            logger.info("Establishing connection with server")
            self._sock.send("add client")
            resp = self._sock.recv(1024)

            if resp == 'Client added':
                logger.info("Connection established")

        except Exception:
            raise RuntimeError('Setting up acquisition <-> stimulation '
                               'computer connection (host: %s '
                               'port: %d) failed. Make sure StimServer '
                               'is running.' % (host, port))

    @verbose
    def get_trigger(self, timeout=5.0, verbose=None):
        """Method to get triggers from StimServer.

        Parameters
        ----------
        timeout : float
            maximum time to wait for a valid trigger from the server

        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).

        """
        start_time = time.time()  # init delay counter. Will stop iterations

        while True:
            try:
                current_time = time.time()

                # Raise timeout error
                if current_time > (start_time + timeout):
                        logger.info("received nothing")
                        return None

                self._sock.send("get trigger")
                trigger = self._sock.recv(1024)

                if trigger != 'Empty':
                    logger.info("received trigger %s" % str(trigger))
                    return int(trigger)

            except RuntimeError as err:
                logger.info('Cannot receive triggers: %s' % (err))
