# Author: Mainak Jas <mainak@neuro.hut.fi>
# License: BSD (3-clause)

import Queue
import time
import socket
import SocketServer
import threading
import logging
logger = logging.getLogger('mne')


class ThreadedTCPServer(SocketServer.ThreadingMixIn, SocketServer.TCPServer):
    """
    Creates a threaded TCP server

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
    """
    Request handler on the server side
    """
    def send_trigger(self, trigger):
        """
        Create a queue of triggers to be delivered
        """

        # If no attribute trigger queue in self, create it
        if not hasattr(self, '_tx_queue'):
            self._tx_queue = []

        self._tx_queue.append(trigger)

        print "Trigger Queue at the moment is " + str(self._tx_queue)

    def handle(self):
        """
        Method to handle requests on the server side
        """

        # Add stim_server._client as a dictionary
        self.server.stim_server.add_client(self.client_address[0], self)
        self.request.settimeout(None)

        while True:

            data = self.request.recv(1024)  # clip input at 1Kb

            if data == 'Give me a trigger':
                print "Request received:: " + data

                # If the method self has trigger queue, create it
                if not hasattr(self, '_tx_queue'):
                    self._tx_queue = []

                # Pop triggers and send them
                if len(self._tx_queue) > 0:
                    trigger = self._tx_queue.pop(0)
                    print "Trigger " + str(trigger) + " popped and sent"
                    self.request.sendall(str(trigger))
                else:
                    print "Queue is empty"
                    self.request.sendall("Empty")


class StimServer(object):
    """ Stimulation Server

    Server to communicate with StimClient(s)

    Parameters
    ----------
    port : int
        The port to which the stimulation server must bind to

    """

    def __init__(self, port=4218):

        # Start a threaded TCP server, binding to localhost on specified port
        self._data = ThreadedTCPServer(('localhost', port),
                                       TriggerHandler, self)

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
        self._client = dict()

    def start(self, ip, stim_client):
        """Method to start the client
        """

        # Instantiate queue for communication between threads
        if not hasattr(self, '_tx_queue'):
            self._tx_queue = Queue.Queue()

        # Start server and a separate thread to send data
        if not self._running:
            logger.info('RtServer: start')
            self._running = True

            # ugly hack but otherwise we get key error for
            # stim_server._client['running']
            time.sleep(1)

            # start the send thread
            self._send_thread = threading.Thread(target=send_trigger_worker,
                                                 args=(self, self._tx_queue))
            self._send_thread.start()

    def add_client(self, ip, sock):
        """Add client and flag it as running
        """
        print "Adding client with ip = " + ip
        self._client['ip'] = ip
        self._client['running'] = True
        self._client['socket'] = sock

    def stop(self):
        """Method to stop the client and server
        """
        print "Stopping server and client ..."
        self._client['running'] = False
        self._running = False

    def shutdown(self):
        self.stop()
        print "Shutting down ..."
        self._data.shutdown()
        self._data.server_close()
        self._data.socket.close()

    def add_trigger(self, trigger):
        """Method to add a trigger
        """
        self._tx_queue.put(trigger)


def send_trigger_worker(stim_server, tx_queue):
    """Worker thread that sends the data to the client
    stim_server : Instance of StimServer

    trig : instance of Queue
        The queue which contains the triggers to be sent
    """

    while stim_server._running and stim_server._client['running']:

        trigger = tx_queue.get()
        stim_server._client['socket'].send_trigger(trigger)
        print "stim server sending trigger %d" % trigger


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
    """

    def __init__(self, host, port=4218, timeout=5.0):
        self._host = host
        self._timeout = timeout
        self._port = port

        try:
            print "Setting up client socket"
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(timeout)
            self._sock.connect((host, port))
        except Exception:
            raise RuntimeError('Setting up acquisition <-> stimulation '
                               'computer connection (host: %s '
                               'port: %d) failed. Make sure StimServer '
                               'is running.' % (host, port))

    def get_trigger(self):
        """Method to get triggers from StimServer
        """
        while True:
            try:
                self._sock.send("Give me a trigger")
                trigger = self._sock.recv(1024)

                if trigger != 'Empty':
                    print "received trigger " + trigger
                    return int(trigger)

            except RuntimeError as err:
                print 'Cannot receive triggers: %s' % err
