import threading
import time
import Queue

from mne.realtime import StimServer, StimClient
from nose.tools import assert_equal


def test_connection():
    """Test TCP/IP connection for StimServer <-> StimClient.
    """

    # have to start a thread to simulate the effect of two
    # different computers since stim_server.start() is designed to
    # be a blocking method

    trig_queue = Queue.Queue()

    thread = threading.Thread(target=connect_client, args=(trig_queue,))
    thread.daemon = True
    thread.start()

    with StimServer('localhost', port=4218) as stim_server:
        stim_server.start()

        # Check if data is ok
        stim_server.add_trigger(20)

        # the assert_equal must be in the test_connection() method
        # Hence communication between threads is necessary
        assert_equal(trig_queue.get(), 20)


def connect_client(trig_queue):
    """Helper method that instantiates the StimClient.
    """
    # just wait till the main thread reaches stim_server.start()
    time.sleep(1.)

    # instantiate StimClient
    stim_client = StimClient('localhost', port=4218)

    # wait a bit more for script to reach stim_server.add_trigger()
    time.sleep(1.)

    trig_queue.put(stim_client.get_trigger())
