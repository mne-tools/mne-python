from mne.realtime import StimServer, StimClient
from nose.tools import assert_equal, assert_raises


def test_connection():
    """Test TCP/IP connection for StimServer <-> StimClient
    """
    stim_server = StimServer(port=4218)
    stim_client = StimClient('localhost', port=4218)

    # start the server
    stim_server.start('localhost')

    # Check if data is ok
    stim_server.add_trigger(20)
    assert_equal(stim_client.get_trigger(), 20)

    # Check if timeout works
    assert_raises(StopIteration, stim_client.get_trigger, 2.0)

    stim_server.shutdown()
