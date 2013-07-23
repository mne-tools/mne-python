# Author: Mainak Jas <mainak@neuro.hut.fi>
# License: BSD (3-clause)

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

    bind_and_activate : bool (default=True)
        If true, bind the socket to the desired address and activate the server
    """

    def __init__(self, server_address, request_handler_class,
                 stim_server, bind_and_activate=True):

    # Basically, this server is the same as a normal TCPServer class
    # except that it has an additional attribute stim_server

        # Create the server and bind it to the desired server address
        SocketServer.TCPServer.__init__(self, server_address,
                                        request_handler_class,
                                        bind_and_activate)

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

        data = self.request.recv(1024)  # clip input at 1Kb

        print "Request received:: " + data

        # If the method self has trigger queue, create it
        if not hasattr(self, '_tx_queue'):
            self._tx_queue = []

        # Pop triggers and send them
        if len(self._tx_queue) > 0:
            trigger = self._tx_queue.pop(0)
            print "Trigger " + str(trigger) + " popped and sent"
            self.request.sendall(str(trigger))
        elif len(self._tx_queue) == 0:
            print "Queue is empty"
        #elif len(self._tx_queue) == 1:
            # pop but append it back because otherwise there won't
            # be anything else to send when a request comes
            #trigger = self._tx_queue.pop(0)
            #self._tx_queue.append(trigger)

            #print "Trigger " + str(trigger) + " popped and sent"
            #self.request.sendall(str(trigger))

        # self.request.settimeout(0.1)

        #while self._client['running']:
            #print "receiving triggers"
            #self.request.sendall(20)  # test if sending is possible
            # 20 must be replaced with trigger
            #self._recv_trigger_worker(self._client)
        # logger.info('RtServer: send thread stopping')


class StimServer(object):
    """ Stimulation Server

    Server to communicate with StimClient(s)

    Parameters
    ----------
    port : int
        The port to which the stimulation server must bind to

    buffer_size : int (preferably power of 2)
        Buffer size (in bytes) for sending triggers
    """

    def __init__(self, port=4218, buffer_size=1024):

        # Start a threaded TCP server, binding to localhost on specified port
        self._data = ThreadedTCPServer(('localhost', port),
                                       TriggerHandler, self,
                                       False)

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
        self._buffer_size = buffer_size
        self._client = dict()

    def start(self, ip, stim_client, q, isi):
        """Method to start the client
        """

        # Start server and a separate thread to send data
        if not self._running:
            logger.info('RtServer: start')
            self._running = True

            # ugly hack but otherwise we get key error for
            # stim_server._client['running']
            time.sleep(1)

            # start the send thread
            self._send_thread = threading.Thread(target=send_trigger_worker,
                                                 args=(self, q, isi))
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


def send_trigger_worker(stim_server, trig, isi):
    """Worker thread that sends the data to the client
    stim_server : Instance of StimServer

    trig : instance of Queue
        The queue which contains the triggers to be sent
    """

    while stim_server._running and stim_server._client['running'] and not trig.empty():

        trigger = trig.get()
        stim_server._client['socket'].send_trigger(trigger)
        print "stim server sending trigger %d" % trigger
        time.sleep(isi.get())


def recv_trigger_worker(stim_client):
    """Worker thread that constantly receives trigger IDs"""

    while True:
        try:

            print "StimClient requesting data"

            # Request for a trigger
            stim_client._sock.send("Give me a trigger")

            # Get the trigger
            trigger = stim_client._sock.recv(1024)

            print "received trigger" + trigger

            # time.sleep(0.1)
            # return trig
        except RuntimeError as err:
            stim_client._recv_thread = None
            print 'Trigger receive thread stopped: %s' % err


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

    def __init__(self, host, port=4218, timeout=1.0):
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

        self._recv_thread = None

    def start_receive_thread(self):
        """Start the receive thread
        """

        if self._recv_thread is None:
            self._recv_thread = threading.Thread(target=recv_trigger_worker,
                                                 args=(self,))

        self._recv_thread.start()

    def stop_receive_thread(self):
        """Stop the receive thread
        """

        if self._recv_thread is not None:
            self._recv_thread.join()
            self._recv_thread = None
