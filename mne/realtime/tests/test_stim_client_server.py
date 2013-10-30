import threading
import time
import Queue

from mne.realtime import StimServer, StimClient
from nose.tools import assert_equal, assert_raises


def test_connection():
    """Test TCP/IP connection for StimServer <-> StimClient.
    """

    # have to start a thread to simulate the effect of two
    # different computers since stim_server.start() is designed to
    # be a blocking method

    # use separate queues because timing matters
    trig_queue1 = Queue.Queue()
    trig_queue2 = Queue.Queue()

    # start a thread to emulate 1st client
    thread1 = threading.Thread(target=connect_client, args=(trig_queue1,))
    thread1.daemon = True
    thread1.start()

    # start another thread to emulate 2nd client
    thread2 = threading.Thread(target=connect_client, args=(trig_queue2,))
    thread2.daemon = True
    thread2.start()

    with StimServer('localhost', port=4218, n_clients=2) as stim_server:
        stim_server.start()

        # Add the trigger to the queue for both clients
        stim_server.add_trigger(20)

        # the assert_equal must be in the test_connection() method
        # Hence communication between threads is necessary
        trig1 = trig_queue1.get()
        trig2 = trig_queue2.get()
        assert_equal(trig1, 20)

        # test if both clients receive the same trigger
        assert_equal(trig1, trig2)

    # test timeout for stim_server
    with StimServer('localhost', port=4218) as stim_server:
        assert_raises(StopIteration, stim_server.start, 1.0)


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
