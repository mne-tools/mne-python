import threading
import time
from nose.tools import assert_equal, assert_raises, assert_true

from mne.realtime import StimServer, StimClient
from mne.externals.six.moves import queue
from mne.utils import requires_good_network, run_tests_if_main


_server = None
_have_put_in_trigger = False
_max_wait = 10.


@requires_good_network
def test_connection():
    """Test TCP/IP connection for StimServer <-> StimClient.
    """
    global _server, _have_put_in_trigger

    # have to start a thread to simulate the effect of two
    # different computers since stim_server.start() is designed to
    # be a blocking method

    # use separate queues because timing matters
    trig_queue1 = queue.Queue()
    trig_queue2 = queue.Queue()

    # start a thread to emulate 1st client
    thread1 = threading.Thread(target=_connect_client, args=(trig_queue1,))
    thread1.daemon = True

    # start another thread to emulate 2nd client
    thread2 = threading.Thread(target=_connect_client, args=(trig_queue2,))
    thread2.daemon = True

    thread1.start()
    thread2.start()
    with StimServer(port=4218, n_clients=2) as stim_server:
        _server = stim_server
        stim_server.start(timeout=10.0)  # don't allow test to hang

        # Add the trigger to the queue for both clients
        stim_server.add_trigger(20)
        _have_put_in_trigger = True  # monkey patch

        # the assert_equal must be in the test_connection() method
        # Hence communication between threads is necessary
        trig1 = trig_queue1.get(timeout=_max_wait)
        trig2 = trig_queue2.get(timeout=_max_wait)
        assert_equal(trig1, 20)

        # test if both clients receive the same trigger
        assert_equal(trig1, trig2)

    # test timeout for stim_server
    with StimServer(port=4218) as stim_server:
        assert_raises(StopIteration, stim_server.start, 0.1)


def _connect_client(trig_queue):
    """Helper method that instantiates the StimClient.
    """
    # just wait till the main thread reaches stim_server.start()
    t0 = time.time()
    while (time.time() - t0 < _max_wait and
           (_server is None or not _server._running)):
        time.sleep(0.01)
    assert_true(_server is not None and _server._running)

    # instantiate StimClient
    stim_client = StimClient('localhost', port=4218)

    # wait for script to reach stim_server.add_trigger()
    t0 = time.time()
    while (time.time() - t0 < _max_wait and not _have_put_in_trigger):
        time.sleep(0.01)
    assert_true(_have_put_in_trigger)

    trig_queue.put(stim_client.get_trigger())
    stim_client.close()


run_tests_if_main()
